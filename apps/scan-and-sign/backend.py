"""HTTP client for the proof-of-reality backend (same one iOS uses).

Three endpoints, all gated by `X-Camera-Key` (paired with the backend's
CAMERA_SHARED_SECRET — distinct from iOS's IOS_SHARED_SECRET / X-Voxelio-Key):
  POST /api/nonce   → cTRNG/Orbitport nonce + satSig
  POST /api/upload  → multipart bundle.json + scene.bin → swarm pin + cosign
  POST /api/mint    → on-chain mint + ENS subname publication

Vercel caps function bodies at ~4.5 MB. uploadOrLocal() honours that cap and
falls back to a `local:<sha>` swarmRef so the mint still goes through —
matches iOS ProofSubmitter behaviour.
"""

from __future__ import annotations

from pathlib import Path

import requests


VERCEL_BODY_CAP = 4 * 1024 * 1024  # 4 MB to leave room for multipart framing


class BackendError(RuntimeError):
    pass


_SCENE_MIME = {
    ".usdz": "model/vnd.usdz+zip",
    ".glb": "model/gltf-binary",
    ".gltf": "model/gltf+json",
    ".ply": "application/octet-stream",
    ".obj": "text/plain",
}


def _scene_mime(path: Path) -> str:
    return _SCENE_MIME.get(path.suffix.lower(), "application/octet-stream")


class BackendClient:
    def __init__(self, base_url: str, shared_secret: str, timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.shared_secret = shared_secret
        self.timeout = timeout

    def _headers(self) -> dict:
        return {"X-Camera-Key": self.shared_secret}

    def health(self) -> dict:
        try:
            r = requests.get(f"{self.base_url}/health", timeout=5)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            raise BackendError(f"health check failed: {e}") from e

    def get_nonce(self) -> dict:
        r = requests.post(
            f"{self.base_url}/api/nonce", headers=self._headers(), timeout=self.timeout
        )
        if r.status_code >= 400:
            raise BackendError(f"nonce {r.status_code}: {r.text}")
        return r.json()

    def upload_or_local(
        self, bundle_bytes: bytes, scene_path: Path, audio_path: Path | None = None,
    ) -> dict:
        """Upload to backend if under Vercel's body cap; otherwise return a
        synthetic response with `swarmRef = local:<sha>` so the mint can still
        reference the scene by its hash."""
        scene_size = scene_path.stat().st_size
        audio_size = audio_path.stat().st_size if audio_path and audio_path.exists() else 0
        estimated = scene_size + audio_size + len(bundle_bytes) + 4096
        if estimated > VERCEL_BODY_CAP:
            scene_sha = _file_sha256(scene_path)
            return {
                "swarmRef": f"local:{scene_sha}",
                "bundleHash": None,  # backend would compute; we send our own to /mint
                "bundleRef": "local:bundle",
                "audioRef": None,
                "sceneBytes": scene_size,
                "cosmoSig": None,
                "_local": True,
                "_reason": f"scene too large for backend ({estimated} > {VERCEL_BODY_CAP})",
            }
        return self._upload(bundle_bytes, scene_path, audio_path)

    def _upload(self, bundle_bytes: bytes, scene_path: Path, audio_path: Path | None) -> dict:
        with scene_path.open("rb") as scene_f:
            files = [
                ("bundle", ("bundle.json", bundle_bytes, "application/json")),
                ("scene", (scene_path.name, scene_f, _scene_mime(scene_path))),
            ]
            audio_f = None
            if audio_path is not None and audio_path.exists():
                audio_f = audio_path.open("rb")
                files.append(("audio", (audio_path.name, audio_f, "audio/mp4")))
            try:
                r = requests.post(
                    f"{self.base_url}/api/upload",
                    headers=self._headers(),
                    files=files,
                    timeout=self.timeout,
                )
            finally:
                if audio_f is not None:
                    audio_f.close()
        if r.status_code == 413:
            raise BackendError("upload rejected: 413 (body too large for Vercel)")
        if r.status_code >= 400:
            raise BackendError(f"upload {r.status_code}: {r.text}")
        return r.json()

    def mint(
        self,
        *,
        swarm_ref: str,
        bundle_ref: str,
        bundle_hash: str,
        sat_sig: str,
        cosmo_sig: str | None,
        attestation: str,
        attestation_type: int = 1,
        captured_at: int,
        mode: int = 1,
    ) -> dict:
        body = {
            "swarmRef": swarm_ref,
            "bundleRef": bundle_ref,
            "bundleHash": bundle_hash,
            "satSig": sat_sig or "STUB",
            "cosmoSig": cosmo_sig or "",
            "attestation": attestation or "MOCK",
            "attestationType": int(attestation_type),
            "capturedAt": int(captured_at),
            "mode": int(mode),
        }
        r = requests.post(
            f"{self.base_url}/api/mint",
            headers={**self._headers(), "Content-Type": "application/json"},
            json=body,
            timeout=self.timeout,
        )
        if r.status_code >= 400:
            raise BackendError(f"mint {r.status_code}: {r.text}")
        return r.json()


def _file_sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
