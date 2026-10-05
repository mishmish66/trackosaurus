"""This machine's trex: where it keeps what it holds ($TREX_DAEMON_DIR, by default ~/.local/state/trex) and its
control socket there (mode 0600), through which `trex serve` hands it directories and tells whether it runs."""

import json
import os
import socket
import socketserver
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final, cast

from .node import Node, dir_base

TIMEOUT: Final = 60.0  # seconds a client waits for the reply


def state_dir() -> Path:
    """Directory of this machine's trex's socket and saved state: $TREX_DAEMON_DIR or ~/.local/state/trex."""
    return Path(os.environ.get("TREX_DAEMON_DIR") or Path.home() / ".local/state/trex")


def socket_path() -> Path:
    return state_dir() / "daemon.sock"


def default_cache() -> Path:
    return Path(os.environ.get("TREX_CACHE") or Path.home() / ".cache/trex")


class _Control(socketserver.StreamRequestHandler):
    """One JSON request line, one JSON reply line: {"op": "status"} or {"op": "add", "path", "force"}."""

    def handle(self) -> None:
        try:
            req = json.loads(self.rfile.readline())
            reply = cast(ControlServer, self.server).answer(req if isinstance(req, dict) else {})
        except (ValueError, OSError) as e:
            reply = {"error": str(e)}
        self.wfile.write(json.dumps(reply).encode() + b"\n")


class ControlServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    """The control socket at `path` of the trex running `node`, which serves at `urls`."""

    daemon_threads = True

    def __init__(self, path: Path, node: Node, urls: Sequence[str]) -> None:
        self.node = node
        self.urls = list(urls)
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if request({"op": "status"}, path) is not None:
            raise RuntimeError(f"trex is already running (socket {path})")
        path.unlink(missing_ok=True)
        old = os.umask(0o177)
        try:
            super().__init__(str(path), _Control)
        finally:
            os.umask(old)
        self.path = path

    def answer(self, req: dict[str, Any]) -> dict[str, Any]:
        match req.get("op"):
            case "status":
                return {"urls": self.urls, "dirs": [d.wire() for d in self.node.served()]}
            case "add":
                d = self.node.track(str(req.get("path")), bool(req.get("force")))
                return {"name": self.node.names()[d] if d else None,
                        "url": self.urls[0].rstrip("/") + (dir_base(d) + "/" if d else "/")}
            case op:
                return {"error": f"unknown request {op!r}"}

    def server_close(self) -> None:
        super().server_close()
        self.path.unlink(missing_ok=True)


def request(req: dict[str, Any], path: Path | None = None) -> dict[str, Any] | None:
    """The reply of this machine's trex, or None when none listens on the socket."""
    with socket.socket(socket.AF_UNIX) as s:
        s.settimeout(TIMEOUT)
        try:
            s.connect(str(path or socket_path()))
        except (FileNotFoundError, ConnectionRefusedError):
            return None
        s.sendall(json.dumps(req).encode() + b"\n")
        reply = json.loads(s.makefile("rb").readline() or b"null")
    return cast(dict[str, Any], reply) if isinstance(reply, dict) else None
