"""The trex daemon: one server for several runs directories, controlled over a user-only Unix socket.

As a systemd user service:

    uv tool install git+https://github.com/mishmish66/trackosaurus
    mkdir -p ~/.config/systemd/user
    trex systemd-unit --source git+https://github.com/mishmish66/trackosaurus > ~/.config/systemd/user/trex.service
    systemctl --user daemon-reload && systemctl --user enable --now trex
    loginctl enable-linger "$USER"

With `--source`, the UI's update button installs the newest trex and systemd restarts the daemon on it.
"""

import json
import os
import socket
import socketserver
import sys
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Final, TypedDict, cast
from urllib.parse import quote

from .index import Explorer

TIMEOUT: Final = 60.0  # seconds a client waits for the daemon's reply
HISTORY_MAX: Final = 50  # directories remembered for re-adding


class RootInfo(TypedDict):
    name: str
    root: str
    url: str


def state_dir() -> Path:
    """Directory of the daemon's socket and saved directory list: $TREX_DAEMON_DIR or ~/.local/state/trex."""
    return Path(os.environ.get("TREX_DAEMON_DIR") or Path.home() / ".local/state/trex")


def socket_path() -> Path:
    return state_dir() / "daemon.sock"


def default_cache() -> Path:
    return Path(os.environ.get("TREX_CACHE") or Path.home() / ".cache/trex")


def root_url(name: str) -> str:
    """Path of a directory's UI under the daemon."""
    return f"/r/{quote(name, safe='')}/"


class Roots:
    """Runs directories served by the daemon, by name, each with its own polling Explorer; saved to `state`.
    Every directory added or removed is remembered, most recent first, in history.json beside it."""

    def __init__(self, cache: Path, state: Path) -> None:
        self.cache = cache
        self.state = state
        self.history_path = state.with_name("history.json")
        self.lock = threading.Lock()
        self.explorers: dict[str, Explorer] = {}
        try:
            saved = json.loads(self.history_path.read_text())
        except (FileNotFoundError, ValueError):
            saved = []
        self._history: list[str] = [str(r) for r in saved] if isinstance(saved, list) else []

    def load(self) -> None:
        """Add the directories saved in `state`, skipping ones that no longer exist."""
        try:
            saved = json.loads(self.state.read_text())
        except FileNotFoundError:
            return
        for e in saved if isinstance(saved, list) else []:
            root = Path(str(e.get("root", "")))
            if root.is_dir():
                self.add(root, str(e.get("name") or root.name))
            else:
                print(f"[trex] skipping saved directory {root}: not a directory", file=sys.stderr, flush=True)

    def add(self, root: Path, name: str | None = None) -> str:
        """Name of the served directory `root`, starting to serve it if new."""
        root = root.resolve()
        with self.lock:
            for n, ex in self.explorers.items():
                if ex.root == root:
                    return n
            name = self._free_name(name or root.name)
            ex = Explorer(root, self.cache)
            threading.Thread(target=ex.poll_forever, name=f"trex-poll-{name}", daemon=True).start()
            self.explorers[name] = ex
            self._remember(root)
            self._save()
        return name

    def _free_name(self, base: str) -> str:
        name, i = base, 2
        while name in self.explorers:
            name, i = f"{base}-{i}", i + 1
        return name

    def remove(self, name: str) -> None:
        """Stop serving `name`; its index cache stays on disk."""
        with self.lock:
            ex = self.explorers.pop(name)
            self._remember(ex.root)
            self._save()
        ex.stop()
        ex.hub.close()

    def get(self, name: str) -> Explorer:
        with self.lock:
            return self.explorers[name]

    def served(self) -> list[RootInfo]:
        with self.lock:
            return [{"name": n, "root": str(ex.root), "url": root_url(n)} for n, ex in self.explorers.items()]

    def history(self) -> list[str]:
        """Remembered directories that exist and are not served, most recent first."""
        with self.lock:
            served = {str(ex.root) for ex in self.explorers.values()}
            return [r for r in self._history if r not in served and Path(r).is_dir()]

    def clear_history(self) -> None:
        with self.lock:
            self._history = []
            _write_json(self.history_path, self._history)

    def _remember(self, root: Path) -> None:
        self._history = [str(root), *(r for r in self._history if r != str(root))][:HISTORY_MAX]
        _write_json(self.history_path, self._history)

    def _save(self) -> None:
        _write_json(self.state, [{"name": n, "root": str(ex.root)} for n, ex in self.explorers.items()])


def _write_json(path: Path, obj: object) -> None:
    """Replace `path` atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1) + "\n")
    tmp.replace(path)


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
    daemon_threads = True

    def __init__(self, path: Path, roots: Roots, urls: Sequence[str]) -> None:
        self.roots = roots
        self.urls = list(urls)
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if request({"op": "status"}, path) is not None:
            raise RuntimeError(f"a trex daemon is already running (socket {path})")
        path.unlink(missing_ok=True)
        old = os.umask(0o177)
        try:
            super().__init__(str(path), _Control)
        finally:
            os.umask(old)
        self.path = path

    def answer(self, req: dict[str, object]) -> dict[str, object]:
        from .server import resolve_root

        if req.get("op") == "status":
            return {"urls": self.urls, "roots": self.roots.served()}
        if req.get("op") == "add":
            name = self.roots.add(resolve_root(str(req.get("path")), bool(req.get("force"))))
            return {"name": name, "url": self.urls[0].rstrip("/") + root_url(name)}
        return {"error": f"unknown request {req.get('op')!r}"}

    def server_close(self) -> None:
        super().server_close()
        self.path.unlink(missing_ok=True)


def request(req: dict[str, object], path: Path | None = None) -> dict[str, object] | None:
    """The daemon's reply, or None when no daemon listens on the socket."""
    with socket.socket(socket.AF_UNIX) as s:
        s.settimeout(TIMEOUT)
        try:
            s.connect(str(path or socket_path()))
        except (FileNotFoundError, ConnectionRefusedError):
            return None
        s.sendall(json.dumps(req).encode() + b"\n")
        reply = json.loads(s.makefile("rb").readline() or b"null")
    return reply if isinstance(reply, dict) else None
