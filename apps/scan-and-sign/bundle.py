"""Canonical proof-bundle builder. Mirrors iOS ProofBundle.swift exactly so
backend's SHA-256(bundle bytes) == bundleHash anchored on-chain.

iOS uses JSONEncoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes].
Default (non-pretty) output is compact, so:
  json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
produces byte-identical output for the schema used here.
"""

from __future__ import annotations

import hashlib
import json
import platform
import time
from pathlib import Path


def canonical_json(obj) -> bytes:
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sensors_hash(scan_id: str, t_start: float, t_end: float, frames: int,
                 intrinsics_tuple) -> str:
    """Stand-in for iOS's IMU sensors hash. Binds the bundle to the actual
    scan run (so two scans of the same object can't share a bundleHash)."""
    parts = [scan_id, f"{t_start:.6f}", f"{t_end:.6f}", str(frames)]
    if intrinsics_tuple:
        parts.append(",".join(f"{v:.6f}" for v in intrinsics_tuple))
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def build(*, scene_path: Path, nonce: str, sat_sig: str, nonce_expires_at: int,
          mode: str, sensors_hash_hex: str, device_model: str,
          device_bundle_id: str, created_at: int | None = None,
          audio_path: Path | None = None) -> tuple[dict, bytes, str]:
    """Returns (bundle_dict, canonical_bytes, '0x'+sha256_hex)."""
    if created_at is None:
        created_at = int(time.time())
    scene_bytes_count = scene_path.stat().st_size
    scene_sha = hashlib.sha256(scene_path.read_bytes()).hexdigest()
    bundle: dict = {
        "version": 1,
        "mode": mode,
        "createdAt": created_at,
        "nonce": nonce,
        "satSig": sat_sig or "",
        "nonceExpiresAt": int(nonce_expires_at),
        "scene": {
            "name": scene_path.name,
            "sha256": scene_sha,
            "bytes": scene_bytes_count,
        },
        "audio": None,
        "sensorsHash": sensors_hash_hex,
        "device": {
            "model": device_model,
            "osVersion": platform.platform(),
            "bundleId": device_bundle_id,
        },
    }
    if audio_path is not None and audio_path.exists():
        bundle["audio"] = {
            "name": audio_path.name,
            "sha256": hashlib.sha256(audio_path.read_bytes()).hexdigest(),
            "bytes": audio_path.stat().st_size,
        }
    bundle_bytes = canonical_json(bundle)
    bundle_hash = "0x" + hashlib.sha256(bundle_bytes).hexdigest()
    return bundle, bundle_bytes, bundle_hash
