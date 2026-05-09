import json
import shutil
import socket
import subprocess


class GoteeError(RuntimeError):
    pass


def _have_nc():
    return shutil.which("nc") is not None


def token_status(host="10.0.0.1", port=4000, timeout=2):
    """Quick reachability probe for the gotee bridge — used by the UI to show
    whether the USB Armory is plugged in. Returns (connected, message)."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True, f"{host}:{port} reachable"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def call_bridge(method, input_str, host="10.0.0.1", port=4000, timeout=5):
    if not _have_nc():
        raise GoteeError("nc not found in PATH")
    payload = json.dumps({"Method": method, "Input": input_str}) + "\n"
    print(f"[gotee] -> {host}:{port} method={method!r} payload_len={len(payload)}")
    try:
        proc = subprocess.run(
            ["nc", "-w", str(timeout), host, str(port)],
            input=payload.encode("utf-8"),
            capture_output=True,
            timeout=timeout + 2,
        )
    except subprocess.TimeoutExpired as e:
        raise GoteeError(f"nc timed out after {timeout}s") from e
    stderr = proc.stderr.decode("utf-8", "replace").strip()
    stdout = proc.stdout.decode("utf-8", "replace").strip()
    print(f"[gotee] <- exit={proc.returncode} stdout={stdout!r} stderr={stderr!r}")
    if proc.returncode != 0:
        msg = stderr or stdout or "(no output)"
        raise GoteeError(f"nc exited {proc.returncode}: {msg}")
    if not stdout:
        raise GoteeError(f"empty reply from gotee bridge (stderr: {stderr!r})")
    line = stdout.splitlines()[0]
    try:
        reply = json.loads(line)
    except json.JSONDecodeError as e:
        raise GoteeError(f"non-JSON reply: {line!r}") from e
    if "Error" in reply and reply["Error"]:
        raise GoteeError(reply["Error"])
    return reply.get("Output", "")


def sign(payload_hex, host="10.0.0.1", port=4000, timeout=5):
    """Call applet `Sign(payload_hex)`. Returns parsed envelope dict
    {device_id, nonce, mac}. Raises GoteeError on any failure."""
    out = call_bridge("Sign", payload_hex, host=host, port=port, timeout=timeout)
    try:
        env = json.loads(out)
    except json.JSONDecodeError as e:
        raise GoteeError(f"applet returned non-JSON: {out!r}") from e
    for k in ("device_id", "nonce", "mac"):
        if k not in env:
            raise GoteeError(f"applet reply missing field {k!r}: {env}")
    return env


def pack_attestation(env: dict) -> str:
    """Pack a gotee Sign envelope into a single hex blob suitable for the
    backend's `attestation` field. The backend stores raw bytes on-chain;
    a verifier unpacks via the same fixed layout: device_id || nonce || mac.

    Each field is treated as hex (with or without 0x prefix). If a field is
    not hex-decodable we hex-encode its UTF-8 bytes so the blob is always
    a valid 0x-prefixed hex string."""
    parts = []
    for k in ("device_id", "nonce", "mac"):
        v = str(env[k])
        if v.startswith("0x") or v.startswith("0X"):
            v = v[2:]
        try:
            bytes.fromhex(v)
            parts.append(v)
        except ValueError:
            parts.append(env[k].encode("utf-8").hex())
    return "0x" + "".join(parts)
