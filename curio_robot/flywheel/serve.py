"""The shared-GPU lock and a fresh llama-server per adapter (docs/bringup-7900xt.md flags).

A fresh server for every adapter matters: llama-server's prompt cache keeps keys and values
from the previous adapter, so switching on a running server scores the old one."""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import time
import urllib.request
from collections.abc import Iterator
from pathlib import Path

LLAMA = Path("C:/Users/Rohan/dev/llama")
BASE = LLAMA / "models" / "Qwen3VL-4B-Instruct-Q8_0.gguf"
MMPROJ = LLAMA / "models" / "mmproj-Qwen3VL-4B-Instruct-F16.gguf"
LORA_DIR = LLAMA / "models" / "lora"
LOCK = Path("C:/Users/Rohan/dev/Perspective/out/gpu.lock")
OWNER = "T5-24"
POLL_S = 120.0
READY_TIMEOUT_S = 180.0


@contextlib.contextmanager
def gpu_lock(what: str, lock: Path = LOCK, poll_s: float = POLL_S) -> Iterator[None]:
    """Take the lock file atomically (create-exclusive), polling while someone holds it;
    remove it on exit, but only if it is still ours."""
    stamp = f"{OWNER} {what} {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            holder = lock.read_text(encoding="utf-8", errors="replace").strip()
            print(f"gpu lock held ({holder}); retrying in {poll_s:.0f} s", flush=True)
            time.sleep(poll_s)
            continue
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(stamp)
        break
    try:
        yield
    finally:
        if lock.is_file() and lock.read_text(encoding="utf-8").startswith(OWNER):
            lock.unlink()


def server_args(lora: Path | None, port: int) -> list[str]:
    args = [str(LLAMA / "vulkan" / "llama-server.exe"), "-m", str(BASE), "--mmproj", str(MMPROJ),
            "-dev", "Vulkan0", "-ngl", "99", "-fa", "on", "-c", "8192", "--jinja",
            "--parallel", "1", "--alias", "qwen-vl", "--host", "127.0.0.1", "--port", str(port)]
    if lora is not None:
        args += ["--lora", str(lora)]
    return args


def _ready(url: str) -> bool:
    try:
        with urllib.request.urlopen(url + "/health", timeout=2) as r:
            return json.load(r).get("status") == "ok"
    except Exception:  # noqa: BLE001 - not up yet
        return False


@contextlib.contextmanager
def llama_server(lora: Path | None, port: int, log: Path) -> Iterator[str]:
    """Start a server for one adapter, yield its /v1 URL, stop it on exit."""
    base = f"http://127.0.0.1:{port}"
    if _ready(base):
        raise RuntimeError(f"something already serves on port {port}; not reusing it")
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as out:
        proc = subprocess.Popen(server_args(lora, port), stdout=out, stderr=subprocess.STDOUT)
        try:
            t0 = time.monotonic()
            while not _ready(base):
                if proc.poll() is not None:
                    raise RuntimeError(f"llama-server exited ({proc.returncode}); see {log}")
                if time.monotonic() - t0 > READY_TIMEOUT_S:
                    raise RuntimeError(f"llama-server not ready in {READY_TIMEOUT_S:.0f} s")
                time.sleep(1.0)
            yield base + "/v1"
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
