"""HTTP side of the demo (T5-33): the console page, a JSON API and a server-sent event stream.

Standard library only (no new dependency). Every client gets its own bounded queue; a slow
client drops frames instead of slowing the robot. Requests only ever queue one of the
session's named jobs with validated arguments: nothing from a request reaches a shell or the
filesystem beyond the recordings folder. Binds to 127.0.0.1 unless told otherwise."""

from __future__ import annotations

import json
import queue
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from curio_robot.demo import status
from curio_robot.demo.session import MAX_TEXT, Busy, DemoSession
from curio_robot.sim.scenarios import SCENARIOS

STATIC = Path(__file__).resolve().parent / "static"
CLIENT_QUEUE = 64
PING_S = 15.0
MAX_BODY = 4096
MAX_SEED = 1_000_000
MAX_TRIALS = 500


class Hub:
    """Fan-out of session events to every connected stream."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._clients: list[queue.Queue[str]] = []

    def add(self) -> queue.Queue[str]:
        q: queue.Queue[str] = queue.Queue(CLIENT_QUEUE)
        with self._lock:
            self._clients.append(q)
        return q

    def remove(self, q: queue.Queue[str]) -> None:
        with self._lock:
            if q in self._clients:
                self._clients.remove(q)

    def publish(self, event: dict) -> None:
        line = json.dumps(event)
        with self._lock:
            clients = list(self._clients)
        for q in clients:
            try:
                q.put_nowait(line)
            except queue.Full:  # drop the oldest, keep the newest
                try:
                    q.get_nowait()
                    q.put_nowait(line)
                except (queue.Empty, queue.Full):
                    pass


def _int(value, lo: int, hi: int, name: str) -> int:
    try:
        v = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be an integer") from None
    if not lo <= v <= hi:
        raise ValueError(f"{name} must be in [{lo}, {hi}]")
    return v


def parse_job(path: str, body: dict) -> tuple[str, dict]:
    """A POST route and body -> (job, validated args). Raises ValueError on bad input."""
    if path == "/api/load":
        kind = body.get("kind")
        if kind not in SCENARIOS:
            raise ValueError(f"kind must be one of {sorted(SCENARIOS)}")
        return "load", {"kind": kind, "seed": _int(body.get("seed", 0), 0, MAX_SEED, "seed")}
    if path in ("/api/run", "/api/compare"):
        text = body.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")
        args = {"text": text.strip()[:MAX_TEXT]}
        if path == "/api/run":
            args["curio"] = bool(body.get("curio", True))
            return "run", args
        return "compare", args
    if path == "/api/reset":
        return "reset", {}
    if path == "/api/ab":
        return "ab", {"trials": _int(body.get("trials", 50), 1, MAX_TRIALS, "trials")}
    if path == "/api/replay":
        rid = body.get("id")
        if not isinstance(rid, str) or not rid.replace("-", "").isalnum():
            raise ValueError("id must be a recording id")
        return "replay", {"id": rid}
    raise LookupError(path)


def make_handler(session: DemoSession, hub: Hub):
    class Handler(BaseHTTPRequestHandler):
        server_version = "CurioDemo/1"

        def log_message(self, fmt, *args) -> None:  # quiet; errors still go to the page
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj) -> None:
            self._send(code, json.dumps(obj).encode("utf-8"), "application/json")

        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
            elif path == "/api/state":
                self._json(200, full_state(session))
            elif path == "/api/events":
                self._stream()
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self) -> None:
            path = self.path.split("?", 1)[0]
            if path == "/api/cancel":
                session.cancel()
                self._json(200, {"ok": True})
                return
            n = int(self.headers.get("Content-Length") or 0)
            if n > MAX_BODY:
                self._json(413, {"error": "body too large"})
                return
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
                if not isinstance(body, dict):
                    raise TypeError("body must be a JSON object")
                job, args = parse_job(path, body)
            except LookupError:
                self._json(404, {"error": "not found"})
                return
            except (ValueError, TypeError) as e:
                self._json(400, {"error": str(e)})
                return
            try:
                session.submit(job, **args)
            except Busy as e:
                self._json(HTTPStatus.CONFLICT, {"error": str(e)})
                return
            self._json(202, {"ok": True, "job": job})

        def _stream(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            q = hub.add()
            try:
                self.wfile.write(f"data: {json.dumps(full_state(session))}\n\n".encode())
                if session.last_frame is not None:
                    self.wfile.write(f"data: {json.dumps(session.last_frame)}\n\n".encode())
                self.wfile.flush()
                while True:
                    try:
                        line = q.get(timeout=PING_S)
                        self.wfile.write(f"data: {line}\n\n".encode())
                    except queue.Empty:
                        self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                pass
            finally:
                hub.remove(q)

    return Handler


def full_state(session: DemoSession) -> dict:
    s = session.state()
    s["preflight"] = status.preflight(session.render_ms, session.error)
    s["dropins"] = status.dropins()
    s["scenes"] = sorted(SCENARIOS)
    return s


def serve(host: str = "127.0.0.1", port: int = 8765, record_dir: Path | None = None,
          scene: tuple[str, int] = ("perspective", 0)) -> None:
    hub = Hub()
    session = DemoSession(hub.publish, record_dir=record_dir)
    session.start()
    if session.error is None:
        session.submit("load", kind=scene[0], seed=scene[1])
    httpd = ThreadingHTTPServer((host, port), make_handler(session, hub))
    httpd.daemon_threads = True
    print(f"Curio demo on http://{host}:{port}  (Ctrl+C to stop)", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
        session.close()
