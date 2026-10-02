"""The trex daemon: one server for several runs directories, controlled over a user-only Unix socket.

As a systemd user service:

    uv tool install git+https://github.com/mishmish66/trackosaurus
    mkdir -p ~/.config/systemd/user
    trex systemd-unit --source git+https://github.com/mishmish66/trackosaurus > ~/.config/systemd/user/trex.service
    systemctl --user daemon-reload && systemctl --user enable --now trex
    loginctl enable-linger "$USER"

With `--source`, the UI's update button installs the newest trex and systemd restarts the daemon on it.

A directory added as `host:path` (in the UI or with `trex serve host:path`) is served from that machine
over ssh: the daemon runs this trex there with uvx (`trex.remote`), so the machine needs only uv and an
ssh key that works without a prompt.
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

from . import remote
from .index import Explorer, TileBudget
from .remote import Remote

TIMEOUT: Final = 60.0  # seconds a client waits for the daemon's reply
HISTORY_MAX: Final = 50  # directories remembered for re-adding


class RootInfo(TypedDict):
    name: str
    root: str  # a local path, or host:path
    url: str
    state: str  # local, or a remote directory's connection state
    error: str


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
    """Runs directories served by the daemon, by name: local ones each with a polling Explorer, remote ones
    (host:path) each with a `Remote`; saved to `state`. Every directory added or removed is remembered, most
    recent first, in history.json beside it."""

    def __init__(self, cache: Path, state: Path) -> None:
        self.cache = cache
        self.state = state
        self.history_path = state.with_name("history.json")
        self.lock = threading.Lock()
        self.entries: dict[str, Explorer | Remote] = {}
        self.budget = TileBudget()
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
            spec, name = str(e.get("root", "")), e.get("name")
            if remote.parse(spec):
                self.add_remote(spec, name, wait=False)
            elif Path(spec).is_dir():
                self.add(Path(spec), str(name or Path(spec).name))
            else:
                print(f"[trex] skipping saved directory {spec}: not a directory", file=sys.stderr, flush=True)

    def add(self, root: Path, name: str | None = None) -> str:
        """Name of the served directory `root`, starting to serve it if new."""
        root = root.resolve()
        with self.lock:
            if (n := self._named(str(root))) is not None:
                return n
            name = self._free_name(name or root.name)
            self.entries[name] = Explorer(root, self.cache, budget=self.budget).start()
            self._remember(str(root))
            self._save()
        return name

    def add_remote(self, spec: str, name: str | None = None, wait: bool = True) -> str:
        """Name of the served remote directory `spec` (host:path), starting it if new. With `wait`, wait for
        its first start, and raise ValueError (forgetting it) if that fails."""
        addr = remote.parse(spec)
        if addr is None:
            raise ValueError(f"{spec} is not a host:path address")
        spec = f"{addr.host}:{addr.path}"
        with self.lock:
            if (n := self._named(spec)) is not None:
                return n
        r = Remote(spec, addr).start()
        if wait and r.settled.wait(remote.ADD_TIMEOUT) and r.state == "unreachable":
            r.close()
            raise ValueError(f"cannot serve {spec}: {r.error}")
        with self.lock:
            if (n := self._named(spec)) is not None:
                r.close()
                return n
            name = self._free_name(name or f"{Path(addr.path).name or addr.path}@{addr.host.split('@')[-1].split('.')[0]}")
            self.entries[name] = r
            self._remember(spec)
            self._save()
        return name

    def _named(self, spec: str) -> str | None:
        return next((n for n, e in self.entries.items() if _spec(e) == spec), None)

    def _free_name(self, base: str) -> str:
        name, i = base, 2
        while name in self.entries:
            name, i = f"{base}-{i}", i + 1
        return name

    def remove(self, name: str) -> None:
        """Stop serving `name`; its index cache stays on disk."""
        with self.lock:
            entry = self.entries.pop(name)
            self._remember(_spec(entry))
            self._save()
        threading.Thread(target=entry.close, name=f"trex-close-{name}", daemon=True).start()

    def close(self) -> None:
        """Stop serving every directory (they stay saved)."""
        with self.lock:
            entries, self.entries = list(self.entries.values()), {}
        for entry in entries:
            entry.close()

    def get(self, name: str) -> Explorer | Remote:
        with self.lock:
            return self.entries[name]

    def served(self) -> list[RootInfo]:
        with self.lock:
            return [{"name": n, "root": _spec(e), "url": root_url(n), "state": e.state if isinstance(e, Remote) else "local",
                     "error": e.error if isinstance(e, Remote) else ""} for n, e in self.entries.items()]

    def history(self) -> list[str]:
        """Remembered directories that are not served, most recent first; local ones only while they exist."""
        with self.lock:
            served = {_spec(e) for e in self.entries.values()}
            return [r for r in self._history if r not in served and (remote.parse(r) is not None or Path(r).is_dir())]

    def clear_history(self) -> None:
        with self.lock:
            self._history = []
            _write_json(self.history_path, self._history)

    def _remember(self, spec: str) -> None:
        self._history = [spec, *(r for r in self._history if r != spec)][:HISTORY_MAX]
        _write_json(self.history_path, self._history)

    def _save(self) -> None:
        _write_json(self.state, [{"name": n, "root": _spec(e)} for n, e in self.entries.items()])


def _spec(entry: Explorer | Remote) -> str:
    """What names a served directory: its path, or its host:path."""
    return entry.spec if isinstance(entry, Remote) else str(entry.root)


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
            path = str(req.get("path"))
            name = (self.roots.add_remote(path) if remote.parse(path) else
                    self.roots.add(resolve_root(path, bool(req.get("force")))))
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
