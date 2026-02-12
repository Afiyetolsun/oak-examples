import time
import threading
from collections import deque
from dataclasses import dataclass

import depthai as dai
import numpy as np


@dataclass(frozen=True)
class PipelineHealthConfig:
    max_fps: int = 30
    min_fps: int = 5
    poll_interval_sec: float = 0.5
    settle_after_tile_change_sec: float = 2.0
    # Node types to skip (not tiling processing bottleneck indicators)
    # Script: config distribution, depth = tile count by design
    # ImageManip: inputConfig BLOCKED is structural (burst send from Script)
    # XLinkOut/XLinkOutHost: host-side reading speed
    # XLinkIn/XLinkInHost: host-device transfer
    # Sync: frame synchronization
    # Camera: capture input
    # VideoEncoder: display output encoding
    skip_node_names: tuple = (
        "Script",
        "ImageManip",
        "XLinkOut",
        "XLinkOutHost",
        "XLinkIn",
        "XLinkInHost",
        "Sync",
        "Camera",
        "VideoEncoder",
    )
    # BLOCKED-based control (only definitive overload signal)
    blocked_window_size: int = 4  # sliding window of recent polls
    blocked_threshold_drop: int = 2  # N blocked polls in window -> drop
    blocked_threshold_severe: int = 3  # N blocked polls in window -> aggressive drop
    drop_step: int = 1
    severe_drop_step: int = 3
    # Rise caution
    rise_step: int = 1
    healthy_polls_before_rise: int = 6
    ceiling_probe_polls: int = 10
    # Estimation
    safety_margin: float = 0.75
    # Nodes that run once per tile (cost scales with tile_count).
    # All other monitored nodes are assumed to run once per frame.
    per_tile_node_names: tuple = ("NeuralNetwork", "DetectionParser")


class PipelineHealthMonitor(dai.node.ThreadedHostNode):
    """
    Monitors pipeline health via BLOCKED input states and adjusts target FPS.

    Uses pipeline state API to detect overload. Only reacts to BLOCKED state
    (queue full + upstream stalling).

    Outputs target FPS to FPSController.
    """

    def __init__(self) -> None:
        super().__init__()

        self._target_fps_out = self.createOutput()

        self._config = PipelineHealthConfig()
        self._pipeline: dai.Pipeline | None = None

        self._output_fps: int = self._config.max_fps
        self._healthy_count: int = 0

        # Ceiling: effective FPS cap from estimation.
        # Only set on tile increase. Raised only after probe is confirmed.
        # None = no ceiling (free to rise to max_fps).
        self._ceiling: int | None = None
        self._ceiling_locked: bool = False
        self._ceiling_probe_count: int = 0

        self._settle_until: float = 0.0

        # Sliding window: True if BLOCKED was detected in that poll
        self._blocked_history: deque = deque(maxlen=4)

        self._old_tile_count: int = 0
        self._lock = threading.Lock()

    def build(
        self,
        pipeline: dai.Pipeline,
        initial_tile_count: int,
        config: PipelineHealthConfig | None = None,
    ) -> "PipelineHealthMonitor":
        self._pipeline = pipeline
        self._old_tile_count = initial_tile_count

        if config is not None:
            self._config = config

        self._blocked_history = deque(maxlen=self._config.blocked_window_size)
        self._output_fps = self._config.max_fps

        return self

    def run(self) -> None:
        self._send_target(self._output_fps)
        print(f"[HEALTH] Started. initial_fps={self._output_fps}")

        while self.isRunning():
            time.sleep(self._config.poll_interval_sec)

            with self._lock:
                now = time.monotonic()

                # Skip polling during settle period after tile change
                if now < self._settle_until:
                    remaining = self._settle_until - now
                    print(
                        f"[HEALTH {now:.2f}] SETTLING: fps={self._output_fps} "
                        f"remaining={remaining:.1f}s"
                    )
                    continue

                try:
                    self._poll_and_adjust()
                except Exception as e:
                    print(f"[HEALTH {now:.2f}] ERROR polling pipeline state: {e}")

    def adjust_fps_from_tile_count(self, tile_count: int) -> None:
        """Called by TilingConfigService when tile count changes."""
        with self._lock:
            now = time.monotonic()

            if tile_count > self._old_tile_count:
                # Tile increase: estimate safe FPS, set as ceiling.
                # Skip estimation if pipeline is already overloaded (at min_fps)
                # because NN timing is inflated by congestion.
                if self._output_fps <= self._config.min_fps:
                    print(
                        f"[HEALTH {now:.2f}] TILES_INCREASE "
                        f"{self._old_tile_count}->{tile_count}, "
                        f"already at min_fps={self._config.min_fps}, "
                        f"skipping estimate (pipeline congested)"
                    )
                    self._ceiling = self._config.min_fps
                    self._ceiling_locked = False
                    self._ceiling_probe_count = 0
                else:
                    try:
                        est_fps = self._estimate_max_fps(tile_count)
                    except Exception as e:
                        print(
                            f"[HEALTH {now:.2f}] TILES_INCREASE "
                            f"{self._old_tile_count}->{tile_count}, "
                            f"estimation failed: {e}"
                        )
                        self._old_tile_count = tile_count
                        return

                    print(
                        f"[HEALTH {now:.2f}] TILES_INCREASE "
                        f"{self._old_tile_count}->{tile_count}, "
                        f"est_fps={est_fps}, setting as ceiling"
                    )
                    self._set_target(est_fps)
                    self._ceiling = est_fps
                    self._ceiling_locked = False
                    self._ceiling_probe_count = 0
            else:
                # Tile decrease: keep current FPS, remove ceiling, let queue
                # monitor naturally discover it can rise. NN timing estimate
                # is unreliable here because medianMicrosRecent reflects the
                # congested pipeline state.
                print(
                    f"[HEALTH {now:.2f}] TILES_DECREASE "
                    f"{self._old_tile_count}->{tile_count}, "
                    f"keeping fps={self._output_fps}, ceiling removed, "
                    f"queue monitor will rise naturally"
                )
                self._ceiling = None
                self._ceiling_locked = False
                self._ceiling_probe_count = 0
                self._healthy_count = 0

            self._old_tile_count = tile_count
            self._blocked_history.clear()
            self._settle_until = now + self._config.settle_after_tile_change_sec

    def _poll_and_adjust(self) -> None:
        now = time.monotonic()
        state = self._pipeline.getPipelineState().nodes().detailed()

        blocked_nodes = []
        all_queue_details = []

        # Scan node input queues — only check for BLOCKED state
        for node_id, ns in state.nodeStates.items():
            node = self._pipeline.getNode(node_id)
            node_name = node.getName() if node else "unknown"

            if node_name in self._config.skip_node_names:
                continue

            label_prefix = f"{node_name}[{node_id}]"

            for input_name, iq in ns.inputStates.items():
                label = f"{label_prefix}/{input_name}"
                cur_q = iq.numQueued

                if cur_q > 0 or iq.state == iq.State.BLOCKED:
                    detail = f"{label}(cur={cur_q}, " f"state={iq.state.name})"
                    all_queue_details.append(detail)

                if iq.state == iq.State.BLOCKED:
                    blocked_nodes.append(f"{label}(queued={cur_q})")

        # Update sliding window
        has_blocked = len(blocked_nodes) > 0
        self._blocked_history.append(has_blocked)
        blocked_count = sum(self._blocked_history)

        cfg = self._config
        old_fps = self._output_fps
        queues_summary = (
            " | ".join(all_queue_details) if all_queue_details else "all empty"
        )
        blocked_info = ", ".join(blocked_nodes) if blocked_nodes else "none"
        window_str = "".join("B" if b else "." for b in self._blocked_history)

        at_floor = self._output_fps <= cfg.min_fps

        # Decision logic: only drop when current poll has BLOCKED.
        # Stale history alone (current clean) should not trigger drops.
        if has_blocked and blocked_count >= cfg.blocked_threshold_severe:
            if at_floor:
                print(
                    f"[HEALTH {now:.2f}] AT_FLOOR: blocked {blocked_count}/{len(self._blocked_history)} "
                    f"[{window_str}] at [{blocked_info}] | "
                    f"fps={old_fps} (at min, tolerating) | queues: {queues_summary}"
                )
            else:
                new_fps = max(cfg.min_fps, self._output_fps - cfg.severe_drop_step)
                self._healthy_count = 0
                self._lock_ceiling_on_drop()
                print(
                    f"[HEALTH {now:.2f}] SEVERE_DROP: blocked {blocked_count}/{len(self._blocked_history)} "
                    f"[{window_str}] at [{blocked_info}] | "
                    f"fps {old_fps}->{new_fps}, ceiling={self._ceiling_status()} | "
                    f"queues: {queues_summary}"
                )
                self._update_target(new_fps)

        elif has_blocked and blocked_count >= cfg.blocked_threshold_drop:
            if at_floor:
                print(
                    f"[HEALTH {now:.2f}] AT_FLOOR: blocked {blocked_count}/{len(self._blocked_history)} "
                    f"[{window_str}] at [{blocked_info}] | "
                    f"fps={old_fps} (at min, tolerating) | queues: {queues_summary}"
                )
            else:
                new_fps = max(cfg.min_fps, self._output_fps - cfg.drop_step)
                self._healthy_count = 0
                self._lock_ceiling_on_drop()
                print(
                    f"[HEALTH {now:.2f}] DROP: blocked {blocked_count}/{len(self._blocked_history)} "
                    f"[{window_str}] at [{blocked_info}] | "
                    f"fps {old_fps}->{new_fps}, ceiling={self._ceiling_status()} | "
                    f"queues: {queues_summary}"
                )
                self._update_target(new_fps)

        elif has_blocked:
            if at_floor:
                print(
                    f"[HEALTH {now:.2f}] AT_FLOOR: [{window_str}] "
                    f"at [{blocked_info}] | fps={old_fps} (at min, tolerating) | queues: {queues_summary}"
                )
            else:
                # Single transient BLOCKED: hold, don't react yet
                self._healthy_count = 0
                print(
                    f"[HEALTH {now:.2f}] BLOCKED_TRANSIENT: [{window_str}] "
                    f"at [{blocked_info}] | fps={old_fps} (holding) | queues: {queues_summary}"
                )

        else:
            # No BLOCKED in current poll: healthy (let stale history age out)
            self._healthy_count += 1
            self._try_rise(now, old_fps, window_str, queues_summary)

    def _try_rise(
        self, now: float, old_fps: int, window_str: str, queues_summary: str
    ) -> None:
        cfg = self._config

        # Determine effective ceiling
        effective_ceiling = cfg.max_fps
        if self._ceiling is not None:
            effective_ceiling = min(effective_ceiling, self._ceiling)

        # Are we above the ceiling? That means we're in an active probe.
        if self._ceiling is not None and self._output_fps > self._ceiling:
            # Probing above ceiling — count stable polls to confirm.
            self._ceiling_probe_count += 1
            if self._ceiling_probe_count >= cfg.ceiling_probe_polls:
                # Probe confirmed: raise ceiling to current FPS.
                self._ceiling = self._output_fps
                self._ceiling_probe_count = 0
                print(
                    f"[HEALTH {now:.2f}] CEILING_CONFIRMED: fps={old_fps} [{window_str}] | "
                    f"probe stable, ceiling raised to {self._ceiling} | queues: {queues_summary}"
                )
            else:
                print(
                    f"[HEALTH {now:.2f}] PROBING: fps={old_fps} [{window_str}] | "
                    f"above ceiling={self._ceiling}, "
                    f"confirm={self._ceiling_probe_count}/{cfg.ceiling_probe_polls} | "
                    f"queues: {queues_summary}"
                )
            return

        at_ceiling = self._output_fps >= effective_ceiling

        if at_ceiling and self._ceiling is not None:
            if self._ceiling_locked:
                print(
                    f"[HEALTH {now:.2f}] HEALTHY: fps={old_fps} [{window_str}] | "
                    f"at LOCKED ceiling={self._ceiling}, "
                    f"healthy={self._healthy_count} | queues: {queues_summary}"
                )
                return

            # At ceiling, count stability before launching a probe
            self._ceiling_probe_count += 1
            if self._ceiling_probe_count >= cfg.ceiling_probe_polls:
                # Launch probe: raise FPS +1 but DON'T raise ceiling yet.
                new_fps = min(cfg.max_fps, self._output_fps + 1)
                self._ceiling_probe_count = 0
                self._healthy_count = 0
                print(
                    f"[HEALTH {now:.2f}] CEILING_PROBE: fps {old_fps}->{new_fps} [{window_str}] | "
                    f"probing above ceiling={self._ceiling}"
                )
                self._update_target(new_fps)
            else:
                print(
                    f"[HEALTH {now:.2f}] HEALTHY: fps={old_fps} [{window_str}] | "
                    f"at ceiling={self._ceiling}, "
                    f"probe_count={self._ceiling_probe_count}/{cfg.ceiling_probe_polls} | "
                    f"queues: {queues_summary}"
                )
            return

        if self._healthy_count >= cfg.healthy_polls_before_rise:
            new_fps = min(effective_ceiling, self._output_fps + cfg.rise_step)
            if new_fps > self._output_fps:
                self._healthy_count = 0
                print(
                    f"[HEALTH {now:.2f}] RISE: fps {old_fps}->{new_fps} [{window_str}] | "
                    f"healthy for {cfg.healthy_polls_before_rise} polls, "
                    f"ceiling={effective_ceiling} | queues: {queues_summary}"
                )
                self._update_target(new_fps)
            else:
                print(
                    f"[HEALTH {now:.2f}] HEALTHY: fps={old_fps} (at max) [{window_str}] | "
                    f"healthy={self._healthy_count} | queues: {queues_summary}"
                )
        else:
            print(
                f"[HEALTH {now:.2f}] HEALTHY: fps={old_fps} [{window_str}] | "
                f"count={self._healthy_count}/{cfg.healthy_polls_before_rise} | "
                f"queues: {queues_summary}"
            )

    def _lock_ceiling_on_drop(self) -> None:
        """Lock ceiling when a drop happens at or above it.

        The ceiling stays at its current value (not lowered to new_fps).
        This prevents locking too low from aggressive drops.
        """
        if self._ceiling is not None and self._output_fps >= self._ceiling:
            self._ceiling_locked = True
            self._ceiling_probe_count = 0

    def _ceiling_status(self) -> str:
        if self._ceiling is None:
            return "none"
        if self._ceiling_locked:
            return f"LOCKED@{self._ceiling}"
        return str(self._ceiling)

    def _set_target(self, fps: int) -> None:
        """Set target from feed-forward (tile change). Resets health counters."""
        fps = max(self._config.min_fps, min(self._config.max_fps, int(fps)))
        self._update_target(fps)
        self._healthy_count = 0

    def _estimate_max_fps(self, tile_count: int) -> int:
        """Estimate safe FPS by finding the bottleneck node across the pipeline.

        Per-tile nodes (NN, DetectionParser): frame_cost = median_us × tile_count
        Per-frame nodes (TilesPatcher, QRDecoder, etc.): frame_cost = median_us
        Bottleneck = node with highest per-frame cost → max_fps = 1M / bottleneck_cost

        Nodes whose inputs are all WAITING are skipped — they're idle
        (starved for data), so their medianMicrosRecent includes wait
        time and doesn't reflect actual processing capacity.
        """
        pipeline_state = self._pipeline.getPipelineState().nodes().detailed()
        cfg = self._config

        max_frame_us = 0.0
        bottleneck_label = ""
        node_details = []

        for node_id, ns in pipeline_state.nodeStates.items():
            node = self._pipeline.getNode(node_id)
            if not node:
                continue
            node_name = node.getName()

            if node_name in cfg.skip_node_names:
                continue

            # Skip nodes whose inputs are all WAITING — they're idle,
            # so timing includes wait time, not processing capacity.
            if ns.inputStates:
                all_waiting = all(
                    iq.state == iq.State.WAITING for iq in ns.inputStates.values()
                )
                if all_waiting:
                    label = f"{node_name}[{node_id}]"
                    node_details.append(f"{label}: SKIPPED (all inputs WAITING)")
                    continue

            node_us = ns.mainLoopTiming.durationStats.medianMicrosRecent
            if node_us <= 0:
                continue

            is_per_tile = node_name in cfg.per_tile_node_names
            frame_us = node_us * tile_count if is_per_tile else node_us
            label = f"{node_name}[{node_id}]"
            scale = f"×{tile_count}" if is_per_tile else "×1"

            node_details.append(
                f"{label}: {node_us:.0f}us{scale}={frame_us:.0f}us/frame"
            )

            if frame_us > max_frame_us:
                max_frame_us = frame_us
                bottleneck_label = label

        if max_frame_us <= 0:
            raise RuntimeError("No valid timing data from any monitored node")

        est_max_fps = 1_000_000 / max_frame_us
        est_fps_target = int(est_max_fps * cfg.safety_margin)
        est_fps_target = max(cfg.min_fps, est_fps_target)

        print(
            f"[HEALTH ESTIMATE] tiles={tile_count}, "
            f"bottleneck={bottleneck_label} ({max_frame_us:.0f}us/frame), "
            f"raw_max_fps={est_max_fps:.1f}, "
            f"safety={cfg.safety_margin}, "
            f"target={est_fps_target} | "
            f"all: {' | '.join(node_details)}"
        )
        return est_fps_target

    def _update_target(self, fps: int) -> None:
        self._output_fps = fps
        self._blocked_history.clear()
        self._ceiling_probe_count = 0
        self._send_target(fps)

    def _send_target(self, fps: int) -> None:
        buff = dai.Buffer()
        buff.setData(np.array([np.uint8(fps)]))
        self._target_fps_out.send(buff)

    @property
    def out(self) -> dai.Node.Output:
        return self._target_fps_out

    @property
    def current_fps(self) -> int:
        with self._lock:
            return self._output_fps
