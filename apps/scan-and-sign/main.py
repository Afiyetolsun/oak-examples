from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
import time
import uuid
from pathlib import Path

import depthai as dai
import numpy as np
import open3d as o3d
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

import bundle as bundle_mod
import signer
from backend import BackendClient, BackendError

ROOT = Path(__file__).parent
SCANS_DIR = ROOT / "scans"
SCANS_DIR.mkdir(exist_ok=True)

MODE_NAMES = {0: "roomPlan", 1: "objectCapture", 2: "spatial"}


def _env(name: str, default=None):
    v = os.environ.get(name)
    return v if v not in (None, "") else default


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-d", "--device", default=None)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--size", default="640x400")
    p.add_argument("--http-port", type=int, default=8080)
    p.add_argument("--viz-port", type=int, default=8082)
    p.add_argument("--viz-url", default=None)
    p.add_argument("--gotee-host", default=_env("GOTEE_HOST", "10.0.0.1"))
    p.add_argument("--gotee-port", type=int, default=int(_env("GOTEE_PORT", 4000)))
    p.add_argument("--no-sign", action="store_true")
    p.add_argument("--voxel-size", type=float, default=0.02,
                   help="TSDF voxel length in meters")
    p.add_argument("--sdf-trunc", type=float, default=0.06,
                   help="TSDF truncation distance in meters")
    p.add_argument("--depth-max", type=float, default=5.0,
                   help="Drop depths beyond this (meters)")
    p.add_argument("--backend-url", default=_env("BACKEND_URL"),
                   help="proof-of-reality backend (e.g. https://proof-of-reality-landing.vercel.app). "
                        "Empty = backend disabled, scans stay local.")
    p.add_argument("--shared-secret",
                   default=_env("CAMERA_SHARED_SECRET", _env("IOS_SHARED_SECRET")),
                   help="X-Voxelio-Key for the backend. Reads CAMERA_SHARED_SECRET first, "
                        "then falls back to IOS_SHARED_SECRET.")
    p.add_argument("--mode", type=int, default=int(_env("MODE", 1)),
                   choices=[0, 1, 2],
                   help="0=roomPlan, 1=objectCapture, 2=spatial")
    p.add_argument("--device-bundle-id",
                   default=_env("DEVICE_BUNDLE_ID", "io.luxonis.scan-and-sign"))
    p.add_argument("--no-mint", action="store_true",
                   help="Build + sign the bundle locally but skip backend upload/mint.")
    a = p.parse_args()
    w, h = (int(v) for v in a.size.lower().split("x"))
    a.size = (w, h)
    return a


def _imgframe_to_np(msg):
    """Get the underlying numpy buffer for a dai.ImgFrame (any type)."""
    return msg.getCvFrame()


class O3DRecorder:
    """Open3D RGBD odometry + TSDF integration. Each feed() advances the
    estimated camera pose and integrates the new frame into a global TSDF
    volume. At Stop, the volume is meshed into a point cloud."""

    def __init__(self, voxel_size=0.02, sdf_trunc=0.06, depth_max=5.0):
        self.lock = threading.Lock()
        self.recording = False
        self.scan_id = None
        self.t_start = None
        self.intrinsics = None  # set once on first frame
        self.intrinsics_tuple = None  # (fx, fy, cx, cy, w, h) for sensorsHash
        self.voxel_size = voxel_size
        self.sdf_trunc = sdf_trunc
        self.depth_max = depth_max
        self.tsdf = None
        self.last_rgbd = None
        self.pose = np.eye(4)
        self._update_count = 0

    def start(self):
        with self.lock:
            if self.recording:
                return None
            self.scan_id = uuid.uuid4().hex[:12]
            self.t_start = time.time()
            self.tsdf = o3d.pipelines.integration.ScalableTSDFVolume(
                voxel_length=self.voxel_size,
                sdf_trunc=self.sdf_trunc,
                color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
            )
            self.last_rgbd = None
            self.pose = np.eye(4)
            self._update_count = 0
            self.recording = True
            return self.scan_id

    def stop(self):
        with self.lock:
            if not self.recording:
                return None
            self.recording = False
            pcd = self.tsdf.extract_point_cloud() if self.tsdf is not None else None
            return {
                "scan_id": self.scan_id,
                "pcd": pcd,
                "updates": self._update_count,
                "t_start": self.t_start,
                "t_end": time.time(),
                "intrinsics": self.intrinsics_tuple,
            }

    def set_intrinsics(self, fx, fy, cx, cy, width, height):
        K = o3d.camera.PinholeCameraIntrinsic()
        K.set_intrinsics(width, height, fx, fy, cx, cy)
        with self.lock:
            self.intrinsics = K
            self.intrinsics_tuple = (fx, fy, cx, cy, float(width), float(height))

    def feed(self, color_bgr, depth_u16):
        with self.lock:
            if not self.recording or self.intrinsics is None:
                return
            try:
                # depthai cvFrame for RGB888i comes back as BGR (OpenCV order).
                rgb = color_bgr[..., ::-1].copy()  # BGR -> RGB
                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    o3d.geometry.Image(np.ascontiguousarray(rgb)),
                    o3d.geometry.Image(np.ascontiguousarray(depth_u16)),
                    depth_scale=1000.0,         # depth is mm
                    depth_trunc=self.depth_max,
                    convert_rgb_to_intensity=False,
                )
                if self.last_rgbd is not None:
                    option = o3d.pipelines.odometry.OdometryOption()
                    option.depth_max = self.depth_max
                    ok, T_rel, _info = o3d.pipelines.odometry.compute_rgbd_odometry(
                        rgbd, self.last_rgbd, self.intrinsics, np.eye(4),
                        o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(),
                        option,
                    )
                    if ok:
                        self.pose = self.pose @ T_rel
                self.last_rgbd = rgbd
                # integrate at the inverse pose (Open3D wants extrinsic)
                self.tsdf.integrate(rgbd, self.intrinsics, np.linalg.inv(self.pose))
                self._update_count += 1
            except Exception as e:
                print(f"[o3d] integrate failed: {type(e).__name__}: {e}")


def _load_intrinsics(device, socket, width, height, recorder):
    try:
        calib = device.readCalibration()
        K = np.array(calib.getCameraIntrinsics(socket, width, height), dtype=np.float64)
        recorder.set_intrinsics(K[0, 0], K[1, 1], K[0, 2], K[1, 2], width, height)
        print(f"intrinsics fx={K[0,0]:.1f} fy={K[1,1]:.1f} cx={K[0,2]:.1f} cy={K[1,2]:.1f}")
    except Exception as e:
        print(f"[intrinsics] failed: {e}")


def build_pipeline(args, recorder, stop_event):
    visualizer = dai.RemoteConnection(httpPort=args.viz_port)
    device = dai.Device(dai.DeviceInfo(args.device)) if args.device else dai.Device()
    if not device.setIrLaserDotProjectorIntensity(1):
        print("Note: device does not support IR laser intensity control.")

    with dai.Pipeline(device) as pipeline:
        platform = device.getPlatform()
        WIDTH, HEIGHT = args.size

        left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
        right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
        cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)

        left_out = left.requestOutput((WIDTH, HEIGHT), type=dai.ImgFrame.Type.NV12, fps=args.fps)
        right_out = right.requestOutput((WIDTH, HEIGHT), type=dai.ImgFrame.Type.NV12, fps=args.fps)
        cam_out = cam.requestOutput((WIDTH, HEIGHT), type=dai.ImgFrame.Type.RGB888i, fps=args.fps)

        stereo = pipeline.create(dai.node.StereoDepth).build(
            left=left_out, right=right_out,
            presetMode=dai.node.StereoDepth.PresetMode.DEFAULT,
        )

        if platform == dai.Platform.RVC4:
            align = pipeline.create(dai.node.ImageAlign)
            stereo.depth.link(align.input)
            cam_out.link(align.inputAlignTo)
            depth_out = align.outputAligned
        else:
            cam_out.link(stereo.inputAlignTo)
            depth_out = stereo.depth

        # Sync color + (color-aligned) depth by timestamp.
        sync = pipeline.create(dai.node.Sync)
        cam_out.link(sync.inputs["color"])
        depth_out.link(sync.inputs["depth"])

        sync_q = sync.out.createOutputQueue(maxSize=2, blocking=False)

        visualizer.addTopic("preview", cam_out)

        pipeline.start()
        visualizer.registerPipeline(pipeline)
        _load_intrinsics(device, dai.CameraBoardSocket.CAM_A, WIDTH, HEIGHT, recorder)
        print(f"O3D pipeline running. Visualizer at http://localhost:{args.viz_port}")

        while not stop_event.is_set() and pipeline.isRunning():
            try:
                grp = sync_q.tryGet()
                if grp is None:
                    time.sleep(0.01)
                    continue
                color_msg = grp["color"]
                depth_msg = grp["depth"]
                if color_msg is None or depth_msg is None:
                    continue
                color = _imgframe_to_np(color_msg)
                depth = _imgframe_to_np(depth_msg)
                if color is None or depth is None:
                    continue
                recorder.feed(color, depth)
            except Exception as e:
                print(f"[pipeline] loop error: {type(e).__name__}: {e}")
                break


def make_app(args, recorder, backend: BackendClient | None):
    app = FastAPI()
    app.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return (ROOT / "static" / "index.html").read_text()

    @app.get("/config")
    def config():
        return {"viz_port": args.viz_port, "viz_url": args.viz_url,
                "fps": args.fps, "size": list(args.size),
                "no_sign": bool(args.no_sign),
                "no_mint": bool(args.no_mint or backend is None),
                "backend_url": args.backend_url or None,
                "mode": args.mode,
                "mode_name": MODE_NAMES.get(args.mode, str(args.mode))}

    @app.get("/healthz")
    def healthz():
        connected, msg = signer.token_status(args.gotee_host, args.gotee_port)
        backend_ok = False
        backend_msg = "disabled"
        if backend is not None:
            try:
                backend.health()
                backend_ok = True
                backend_msg = backend.base_url
            except BackendError as e:
                backend_msg = str(e)
        return {
            "token": {"connected": connected, "msg": msg,
                      "host": args.gotee_host, "port": args.gotee_port},
            "backend": {"ok": backend_ok, "msg": backend_msg,
                        "url": backend.base_url if backend else None},
            "ready": connected and (backend_ok or backend is None),
        }

    @app.post("/record/start")
    def record_start():
        scan_id = recorder.start()
        if scan_id is None:
            raise HTTPException(409, "already recording")
        return {"scan_id": scan_id, "status": "recording"}

    @app.post("/record/stop")
    def record_stop():
        snap = recorder.stop()
        if snap is None:
            raise HTTPException(409, "not recording")
        if snap["pcd"] is None or snap["updates"] == 0:
            raise HTTPException(400, "no frames integrated yet")
        pcd = snap["pcd"]
        n = len(pcd.points)
        if n == 0:
            raise HTTPException(400, "TSDF produced empty pointcloud")

        ply_path = SCANS_DIR / f"{snap['scan_id']}.ply"
        o3d.io.write_point_cloud(str(ply_path), pcd, write_ascii=False, compressed=False)

        h = hashlib.sha256()
        with open(ply_path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        ply_sha256 = h.hexdigest()

        envelope = {
            "scan_id": snap["scan_id"],
            "ply_sha256": ply_sha256,
            "frames_integrated": snap["updates"],
            "point_count": int(n),
            "size": list(args.size),
            "fps": args.fps,
            "ts": int(snap["t_end"]),
            "duration_s": round(snap["t_end"] - snap["t_start"], 3),
            "voxel_size": args.voxel_size,
            "mode": args.mode,
            "mode_name": MODE_NAMES.get(args.mode, str(args.mode)),
            "bundle": None,
            "bundle_hash": None,
            "gotee": None,
            "backend": None,
            "mint": None,
        }

        nonce_resp = None
        if backend is not None and not args.no_mint:
            try:
                nonce_resp = backend.get_nonce()
                envelope["backend"] = {"nonce": nonce_resp}
            except BackendError as e:
                envelope["backend"] = {"nonce_error": str(e)}

        nonce_value = (nonce_resp or {}).get("nonce", "")
        sat_sig = (nonce_resp or {}).get("satSig", "")
        nonce_expires_at = int((nonce_resp or {}).get("expiresAt", snap["t_end"]))

        bundle_dict, bundle_bytes, bundle_hash = bundle_mod.build(
            scene_path=ply_path,
            nonce=nonce_value,
            sat_sig=sat_sig,
            nonce_expires_at=nonce_expires_at,
            mode=MODE_NAMES.get(args.mode, "objectCapture"),
            sensors_hash_hex=bundle_mod.sensors_hash(
                snap["scan_id"], snap["t_start"], snap["t_end"],
                snap["updates"], snap.get("intrinsics"),
            ),
            device_model=os.environ.get("OAK_DEVICE_MODEL", "oak"),
            device_bundle_id=args.device_bundle_id,
            created_at=int(snap["t_end"]),
        )
        envelope["bundle"] = bundle_dict
        envelope["bundle_hash"] = bundle_hash

        attestation_hex = ""
        if not args.no_sign:
            try:
                gotee_env = signer.sign(
                    bundle_hash[2:] if bundle_hash.startswith("0x") else bundle_hash,
                    host=args.gotee_host,
                    port=args.gotee_port,
                )
                envelope["gotee"] = gotee_env
                attestation_hex = signer.pack_attestation(gotee_env)
            except signer.GoteeError as e:
                envelope["gotee"] = {"error": str(e)}

        if backend is not None and not args.no_mint and nonce_resp is not None:
            try:
                up = backend.upload_or_local(bundle_bytes, ply_path)
                envelope["backend"]["upload"] = up
                swarm_ref = up["swarmRef"]
                bundle_ref = (up.get("bundleRef")
                              or f"local:{bundle_hash[2:]}")
                cosmo_sig = up.get("cosmoSig")

                mint_resp = backend.mint(
                    swarm_ref=swarm_ref,
                    bundle_ref=bundle_ref,
                    bundle_hash=bundle_hash,
                    sat_sig=sat_sig,
                    cosmo_sig=cosmo_sig,
                    attestation=attestation_hex or "MOCK",
                    attestation_type=1,  # device-SE (USB Armory)
                    captured_at=int(snap["t_end"]),
                    mode=args.mode,
                )
                envelope["mint"] = mint_resp
            except BackendError as e:
                envelope["backend"]["error"] = str(e)
            except Exception as e:
                envelope["backend"]["error"] = f"{type(e).__name__}: {e}"

        env_path = SCANS_DIR / f"{snap['scan_id']}.json"
        env_path.write_text(json.dumps(envelope, indent=2))
        bundle_path = SCANS_DIR / f"{snap['scan_id']}.bundle.json"
        bundle_path.write_bytes(bundle_bytes)
        return envelope

    @app.get("/record/state")
    def record_state():
        with recorder.lock:
            elapsed = (time.time() - recorder.t_start) if recorder.recording else 0
            return {
                "recording": recorder.recording,
                "scan_id": recorder.scan_id if recorder.recording else None,
                "frames_integrated": recorder._update_count if recorder.recording else 0,
                "elapsed_s": round(elapsed, 2),
            }

    @app.get("/scans")
    def list_scans():
        out = []
        for p in sorted(SCANS_DIR.glob("*.json"), reverse=True):
            try:
                out.append(json.loads(p.read_text()))
            except Exception:
                continue
        return out

    @app.get("/scans/{scan_id}.ply")
    def get_ply(scan_id: str):
        p = SCANS_DIR / f"{scan_id}.ply"
        if not p.exists():
            raise HTTPException(404)
        return FileResponse(str(p), media_type="application/octet-stream",
                            filename=f"{scan_id}.ply")

    @app.get("/scans/{scan_id}.json")
    def get_envelope(scan_id: str):
        p = SCANS_DIR / f"{scan_id}.json"
        if not p.exists():
            raise HTTPException(404)
        return FileResponse(str(p), media_type="application/json")

    @app.get("/scans/{scan_id}.bundle.json")
    def get_bundle(scan_id: str):
        p = SCANS_DIR / f"{scan_id}.bundle.json"
        if not p.exists():
            raise HTTPException(404)
        return FileResponse(str(p), media_type="application/json")

    @app.get("/debug/network")
    def debug_network():
        import shlex
        import subprocess as sp
        results = {}
        cmds = [
            ("nc_bridge", f"nc -zv -w 3 {args.gotee_host} {args.gotee_port}"),
            ("nc_oak_main", "nc -zv -w 3 192.168.88.236 4000"),
            ("ip_route", "ip route"),
            ("ip_addr", "ip -4 addr"),
            ("ping_bridge", f"ping -c 1 -W 2 {args.gotee_host}"),
            ("which_nc", "which nc"),
            ("hostname_ip", "hostname -I"),
            ("env_path", "echo $PATH"),
            ("default_gw", "ip route | awk '/default/ {print $3}'"),
        ]
        for name, cmd in cmds:
            try:
                p = sp.run(["sh", "-c", cmd], capture_output=True, timeout=5)
                results[name] = {
                    "cmd": cmd,
                    "exit": p.returncode,
                    "stdout": p.stdout.decode("utf-8", "replace").strip(),
                    "stderr": p.stderr.decode("utf-8", "replace").strip(),
                }
            except Exception as e:
                results[name] = {"cmd": cmd, "error": f"{type(e).__name__}: {e}"}
        # Also try to talk to gotee through the gateway IP
        try:
            gw = sp.run(["sh", "-c", "ip route | awk '/default/ {print $3; exit}'"],
                        capture_output=True, timeout=3).stdout.decode().strip()
            if gw:
                p = sp.run(["sh", "-c", f"nc -zv -w 3 {gw} 4000"],
                           capture_output=True, timeout=5)
                results["nc_via_gateway"] = {
                    "gateway": gw,
                    "cmd": f"nc -zv -w 3 {gw} 4000",
                    "exit": p.returncode,
                    "stderr": p.stderr.decode("utf-8", "replace").strip(),
                    "stdout": p.stdout.decode("utf-8", "replace").strip(),
                }
        except Exception as e:
            results["nc_via_gateway"] = {"error": str(e)}
        return results

    return app


def main():
    args = parse_args()
    recorder = O3DRecorder(
        voxel_size=args.voxel_size,
        sdf_trunc=args.sdf_trunc,
        depth_max=args.depth_max,
    )
    stop_event = threading.Event()

    backend: BackendClient | None = None
    if args.backend_url and args.shared_secret:
        backend = BackendClient(args.backend_url, args.shared_secret)
        print(f"[backend] {args.backend_url} (auth: X-Voxelio-Key)")
    elif args.backend_url and not args.shared_secret:
        print("[backend] BACKEND_URL set but IOS_SHARED_SECRET missing — backend disabled")
    else:
        print("[backend] disabled (BACKEND_URL not set) — scans stay local")

    def pipeline_thread():
        try:
            build_pipeline(args, recorder, stop_event)
        except Exception as e:
            print(f"[pipeline] crashed: {e}")
            stop_event.set()
            os._exit(1)

    t = threading.Thread(target=pipeline_thread, daemon=True)
    t.start()

    app = make_app(args, recorder, backend)
    print(f"[http] control panel  http://0.0.0.0:{args.http_port}/")
    print(f"[http] live preview   http://0.0.0.0:{args.viz_port}/")
    try:
        uvicorn.run(app, host="0.0.0.0", port=args.http_port, log_level="info")
    finally:
        stop_event.set()


if __name__ == "__main__":
    main()
