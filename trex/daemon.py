"""The trex daemon: one server for several runs directories, controlled over a user-only Unix socket.

As a systemd user service:

    uv tool install git+https://github.com/mishmish66/trackosaurus
    mkdir -p ~/.config/systemd/user
    trex systemd-unit > ~/.config/systemd/user/trex.service
    systemctl --user daemon-reload && systemctl --user enable --now trex
    loginctl enable-linger "$USER"

The update button in the trex panel installs the newest trex from `--source` (default this repository), and
systemd restarts the daemon on it.

/ shows every tracked directory, each as a top-level folder; clicking trex (top left) opens the panel that
switches and manages them, and workspaces: named sets of directories whose folder trees are merged, so runs
from several machines can be compared side by side (group by `dir`).

A directory added as `host:path` (in the UI or with `trex serve host:path`) is served from that machine
over ssh: the daemon runs this trex there with uvx (`trex.remote`), so the machine needs only uv and an
ssh key that works without a prompt. On a cluster, use a host that allows long-running processes, such
as a data-transfer node; runs written by jobs on other nodes are followed through their journals.
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
from .workspace import Far, Local, Workspace

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
    """Path of a tracked directory's UI under the daemon."""
    return f"/r/{quote(name, safe='')}/"


def workspace_url(name: str) -> str:
    return f"/w/{quote(name, safe='')}/"


def unique_names(specs: Sequence[str]) -> dict[str, str]:
    """Display names of tracked directories, as Emacs's uniquify does: the basename, and where basenames collide,
    `name<context>` with the shortest context that tells them apart (parent folders, or for a remote one its
    host and then its parent folders)."""
    contexts = {s: _context(s) for s in specs}
    out: dict[str, str] = {}
    by_base: dict[str, list[str]] = {}
    for s in specs:
        by_base.setdefault(contexts[s][0], []).append(s)
    for base, group in by_base.items():
        depth = 0
        while len(group) > 1 and len({tuple(contexts[s][1:depth + 1]) for s in group}) < len(group) and \
                depth < max(len(contexts[s]) for s in group):
            depth += 1
        for s in group:
            ctx = contexts[s][1:depth + 1]
            out[s] = f"{base}<{'/'.join(reversed(ctx))}>" if len(group) > 1 and ctx else base
    return out


def _context(spec: str) -> list[str]:
    """[basename, then what tells it apart, nearest first]."""
    addr = remote.parse(spec)
    if addr is None:
        parts = Path(spec).parts
        return [parts[-1] if parts else spec, *reversed([q for q in parts[:-1] if q != "/"])]
    path = addr.path.rstrip("/").split("/")
    return [path[-1] or addr.path, addr.host.split("@")[-1].split(".")[0], *reversed([q for q in path[:-1] if q])]


class Roots:
    """Tracked directories served by the daemon: local ones each with a polling Explorer, remote ones (host:path)
    each with a `Remote`, and workspaces, named sets of them; saved to `state`. Every directory added or
    removed is remembered, most recent first, in history.json beside it."""

    def __init__(self, cache: Path, state: Path) -> None:
        self.cache = cache
        self.state = state
        self.history_path = state.with_name("history.json")
        self.lock = threading.Lock()
        self.entries: dict[str, Explorer | Remote] = {}  # by spec: path, or host:path
        self.workspaces: dict[str, list[str]] = {}  # name -> member specs
        self._views: dict[str, Workspace] = {}
        self.budget = TileBudget()
        try:
            saved = json.loads(self.history_path.read_text())
        except (FileNotFoundError, ValueError):
            saved = []
        self._history: list[str] = [str(r) for r in saved] if isinstance(saved, list) else []

    def load(self) -> None:
        """Add the directories and workspaces saved in `state`, skipping directories that no longer exist."""
        try:
            saved = json.loads(self.state.read_text())
        except FileNotFoundError:
            return
        tracked = saved.get("tracked", []) if isinstance(saved, dict) else [e.get("root", "") for e in saved]
        for spec in map(str, tracked):
            if remote.parse(spec):
                self.add_remote(spec, wait=False)
            elif Path(spec).is_dir():
                self.add(Path(spec))
            else:
                print(f"[trex] skipping saved directory {spec}: not a directory", file=sys.stderr, flush=True)
        for w in saved.get("workspaces", []) if isinstance(saved, dict) else []:
            members = [m for m in map(str, w.get("members", [])) if m in self.entries]
            with self.lock:
                self.workspaces[str(w.get("name"))] = members

    def add(self, root: Path) -> str:
        """Name of the tracked directory `root`, starting to serve it if new."""
        spec = str(root.resolve())
        with self.lock:
            if spec not in self.entries:
                self.entries[spec] = Explorer(Path(spec), self.cache, budget=self.budget).start()
                self._changed(spec)
            return self.names()[spec]

    def add_remote(self, spec: str, wait: bool = True) -> str:
        """Name of the tracked remote directory `spec` (host:path), starting it if new. With `wait`, wait for its
        first start, and raise ValueError (forgetting it) if that fails."""
        addr = remote.parse(spec)
        if addr is None:
            raise ValueError(f"{spec} is not a host:path address")
        spec = f"{addr.host}:{addr.path}"
        with self.lock:
            if spec in self.entries:
                return self.names()[spec]
        r = Remote(spec, addr).start()
        if wait and r.settled.wait(remote.ADD_TIMEOUT) and r.state == "unreachable":
            r.close()
            raise ValueError(f"cannot serve {spec}: {r.error}")
        with self.lock:
            if spec in self.entries:
                r.close()
            else:
                self.entries[spec] = r
                self._changed(spec)
            return self.names()[spec]

    def names(self) -> dict[str, str]:
        """{spec: display name} of the tracked directories."""
        return unique_names(list(self.entries))

    def spec(self, name: str) -> str:
        """The spec of the tracked directory named `name`; KeyError if none."""
        for spec, n in self.names().items():
            if n == name:
                return spec
        raise KeyError(name)

    def remove(self, name: str) -> None:
        """Stop serving `name` and drop it from every workspace; its index cache stays on disk."""
        with self.lock:
            spec = self.spec(name)
            entry = self.entries.pop(spec)
            for members in self.workspaces.values():
                if spec in members:
                    members.remove(spec)
            self._changed(spec)
        threading.Thread(target=entry.close, name=f"trex-close-{name}", daemon=True).start()

    def close(self) -> None:
        """Stop serving every directory (they stay saved)."""
        with self.lock:
            entries, self.entries = list(self.entries.values()), {}
            views, self._views = list(self._views.values()), {}
        for v in views:
            v.close()
        for entry in entries:
            entry.close()

    def get(self, name: str) -> Explorer | Remote:
        with self.lock:
            return self.entries[self.spec(name)]

    def served(self) -> list[RootInfo]:
        with self.lock:
            names = self.names()
            return [{"name": names[s], "root": s, "url": root_url(names[s]),
                     "state": e.state if isinstance(e, Remote) else "local",
                     "error": e.error if isinstance(e, Remote) else ""} for s, e in self.entries.items()]

    # ---- workspaces ----

    def set_workspace(self, name: str, members: Sequence[str], old: str | None = None) -> None:
        """Create or replace workspace `name` (renaming `old`) with the tracked directories named `members`."""
        name = name.strip()
        if not name or "/" in name or name.startswith("."):
            raise ValueError(f"{name!r} is not a workspace name")
        with self.lock:
            specs = [self.spec(m) for m in members]
            if name != old and name in self.workspaces:
                raise ValueError(f"a workspace named {name} exists")
            if old is not None and old != name:
                self.workspaces.pop(old, None)
            self.workspaces[name] = specs
            self._changed(None)

    def delete_workspace(self, name: str) -> None:
        with self.lock:
            del self.workspaces[name]
            self._changed(None)

    def workspace(self, name: str) -> Workspace:
        """The workspace `name`, as an Explorer-like view of its members."""
        with self.lock:
            view = self._views.get(name)
            if view is None:
                view = self._views[name] = Workspace(name, self._members(self.workspaces[name]))
            return view

    def everything(self) -> Workspace:
        """The root view: every tracked directory, each a top-level folder."""
        with self.lock:
            view = self._views.get("")
            if view is None:
                view = self._views[""] = Workspace("/", self._members(list(self.entries)), nested=True)
            return view

    def _members(self, specs: Sequence[str]) -> list[Local | Far]:
        names = self.names()
        return [Far(names[s], e) if isinstance(e := self.entries[s], Remote) else Local(names[s], e) for s in specs]

    def workspace_list(self) -> list[dict[str, object]]:
        with self.lock:
            names = self.names()
            return [{"name": w, "url": workspace_url(w), "members": [names[s] for s in specs]}
                    for w, specs in self.workspaces.items()]

    # ---- history and saving ----

    def history(self) -> list[str]:
        """Remembered directories that are not served, most recent first; local ones only while they exist."""
        with self.lock:
            return [r for r in self._history if r not in self.entries and (remote.parse(r) is not None or Path(r).is_dir())]

    def clear_history(self) -> None:
        with self.lock:
            self._history = []
            _write_json(self.history_path, self._history)

    def _changed(self, spec: str | None) -> None:
        """After a change (inside the lock): remember `spec`, rebuild workspace views, save."""
        if spec is not None:
            self._history = [spec, *(r for r in self._history if r != spec)][:HISTORY_MAX]
            _write_json(self.history_path, self._history)
        for v in self._views.values():
            v.close()
        self._views = {}
        _write_json(self.state, {"tracked": list(self.entries),
                                 "workspaces": [{"name": w, "members": m} for w, m in self.workspaces.items()]})


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
