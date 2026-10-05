"""The trex daemon: one server for several runs directories, controlled over a user-only Unix socket.

As a systemd user service:

    uv tool install git+https://github.com/mishmish66/trackosaurus
    mkdir -p ~/.config/systemd/user
    trex systemd-unit > ~/.config/systemd/user/trex.service
    systemctl --user daemon-reload && systemctl --user enable --now trex
    loginctl enable-linger "$USER"

As a launchd agent on macOS (running while you are logged in; log in ~/Library/Logs/trex.log):

    uv tool install git+https://github.com/mishmish66/trackosaurus
    trex launchd-plist > ~/Library/LaunchAgents/trex.plist
    launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/trex.plist

The update button in the trex panel installs the newest trex from the `--source` given to `trex systemd-unit`
or `trex launchd-plist` (default this repository), and systemd or launchd restarts the daemon on it.

/ shows every tracked directory, each as a top-level folder; clicking trex (top left) opens the panel that
switches and manages them, and workspaces: named sets of directories whose folder trees are merged, so runs
from several machines can be compared side by side (group by `dir`).

A directory added as `host:path` (in the UI or with `trex serve host:path`) is crawled on that machine
over ssh: the daemon copies this trex there and runs it with uvx (`trex.remote`), so the machine needs only uv and an
ssh key that works without a prompt. On a cluster, use a host that allows long-running processes, such
as a data-transfer node; runs written by jobs on other nodes are followed through their journals.

A daemon also pulls every directory another trex holds, given as `http://host:port` (in the UI or with
`trex serve http://host:port`). Every directory it pulls or crawls over ssh is kept in its cache (`trex.mirror`),
compiled as the trex that crawls it compiles it, so it answers at once and while that trex is unreachable; running
runs come live from it while it answers. A laptop's daemon pulling a lab workstation's, which crawls the training
machines over ssh, holds everything the workstation holds without reaching those machines itself:

    workstation$ trex daemon                        # tracks ~/runs, gpu-box:~/runs, ...
    laptop$ trex daemon                             # then, once:
    laptop$ trex serve http://workstation:13898

Daemons may pull from each other in any arrangement, cycles included. A directory's id is the name of the trex
that crawls it (`--name`, the host's by default) and its path, or the `host:path` it is crawled over ssh by, and
it is one top-level folder wherever it is held. Each trex takes a directory through the link offering it by the
fewest trex, never one it crawls itself nor one that came through it, and drops it once that link no longer has it.
"""

import json
import os
import socket
import socketserver
import sys
import threading
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Final, NamedTuple, TypedDict, cast
from urllib.parse import quote, urlsplit

from . import remote
from .crawl import Crawl
from .index import Explorer
from .mirror import Pull, Unreachable, Upstream
from .remote import Remote, UnixHTTPConnection
from .workspace import Member, Workspace

TIMEOUT: Final = 60.0  # seconds a client waits for the daemon's reply
HISTORY_MAX: Final = 50  # directories remembered for re-adding
PULL_EVERY: Final = 10.0  # seconds between asking each link what it holds


class RootInfo(TypedDict):
    name: str
    root: str  # how this trex holds it: a local path or host:path it crawls, else the directory's id
    id: str
    url: str
    state: str  # local, or a mirrored directory's connection state
    error: str
    link: str | None  # the link it is pulled from


class LinkInfo(TypedDict):
    url: str
    name: str  # the node's name, once it answered
    state: str  # connecting, connected or unreachable
    error: str


def state_dir() -> Path:
    """Directory of the daemon's socket and saved directory list: $TREX_DAEMON_DIR or ~/.local/state/trex."""
    return Path(os.environ.get("TREX_DAEMON_DIR") or Path.home() / ".local/state/trex")


def socket_path() -> Path:
    return state_dir() / "daemon.sock"


def default_cache() -> Path:
    return Path(os.environ.get("TREX_CACHE") or Path.home() / ".cache/trex")


def resolve_root(root: str | os.PathLike[str], force: bool) -> Path:
    """`root` resolved; ValueError for a non-directory, or for / and $HOME (whose crawl would sweep a whole
    machine or home directory) unless `force`."""
    r = Path(root).expanduser().resolve()
    if not r.is_dir():
        raise ValueError(f"{root} is not a directory")
    if not force and r in (Path("/"), Path.home().resolve()):
        raise ValueError(f"refusing to crawl {r}; point trex at a runs directory (or pass --force)")
    return r


def root_url(name: str) -> str:
    """Path of a tracked directory's UI under the daemon."""
    return f"/r/{quote(name, safe='')}/"


def workspace_url(name: str) -> str:
    return f"/w/{quote(name, safe='')}/"


def unique_names(specs: Sequence[str]) -> dict[str, str]:
    """Display names of tracked directories, as Emacs's uniquify does: the basename, and where basenames collide,
    `name<context>` with the shortest context that tells them apart (parent folders, or for a remote one its
    host and then its parent folders; its parent folders alone when every one it collides with is on that host)."""
    contexts = {s: _context(s) for s in specs}
    by_base: dict[str, list[str]] = {}
    for s in specs:
        by_base.setdefault(contexts[s][0], []).append(s)
    out: dict[str, str] = {}
    for base, group in by_base.items():
        out.update(_told_apart(base, group, contexts))
    return out


def _told_apart(base: str, group: Sequence[str], contexts: dict[str, list[str]]) -> dict[str, str]:
    """Names of the specs in `group`, whose basename is `base`."""
    if len(group) == 1:
        return {group[0]: base}
    ctx = {s: contexts[s][1:] for s in group}
    hosts = {a.host if (a := remote.parse(s)) else None for s in group}
    if len(hosts) == 1 and None not in hosts:
        ctx = {s: c[1:] for s, c in ctx.items()}
    depth = 0
    while len({tuple(c[:depth]) for c in ctx.values()}) < len(group) and depth < max(map(len, ctx.values())):
        depth += 1
    return {s: f"{base}<{'/'.join(reversed(c[:depth]))}>" if c[:depth] else base for s, c in ctx.items()}


def _context(spec: str) -> list[str]:
    """[basename, then what tells it apart, nearest first]."""
    addr = remote.parse(spec)
    if addr is None:
        parts = Path(spec).parts
        return [parts[-1] if parts else spec, *reversed([q for q in parts[:-1] if q != "/"])]
    path = addr.path.rstrip("/").split("/")
    return [path[-1] or addr.path, addr.host.split("@")[-1].split(".")[0], *reversed([q for q in path[:-1] if q])]


class Node(NamedTuple):
    """This trex among the ones that pull from each other: `id`, unique, is what `via` lists; `name` (the host's, unless
    given) begins the ids of the directories it crawls."""

    id: str
    name: str


class Pulled(NamedTuple):
    """Where a directory this trex does not crawl comes from: the link it is pulled through, and the nodes it came
    through to that link, from the one that crawls it."""

    link: str
    via: list[str]


def host_name() -> str:
    return socket.gethostname().split(".")[0] or "localhost"


def node_at(path: Path, name: str | None = None) -> Node:
    """The node saved at `path`, made and saved on first use; renamed `name` when given."""
    try:
        saved = json.loads(path.read_text())
        node = Node(str(saved["id"]), str(saved["name"]))
    except (FileNotFoundError, ValueError, KeyError, TypeError):
        node = Node(uuid.uuid4().hex[:16], host_name())
    node = node._replace(name=name or node.name)
    try:
        unchanged = json.loads(path.read_text()) == node._asdict()
    except (FileNotFoundError, ValueError):
        unchanged = False
    if not unchanged:
        _write_json(path, node._asdict())
    return node


def link_url(spec: str) -> str | None:
    """The trex `spec` names (http://host[:port]), or None for a directory."""
    u = urlsplit(spec.strip())
    if u.scheme != "http" or not u.hostname:
        return None
    return f"http://{u.netloc}"


def dir_base(d: str) -> str:
    """The path under which a trex serves the directory whose id is `d`."""
    return f"/d/{quote(d, safe='')}"


class Link:
    """A trex this one pulls from, at `url`: its node, and what it holds, as of the last time it answered."""

    def __init__(self, url: str) -> None:
        self.url = url
        self.state = "connecting"  # connecting, connected or unreachable
        self.error = ""
        self.node: Node | None = None
        self.offers: dict[str, list[str]] | None = None  # directory id -> the nodes it came through; None while unanswered

    def refresh(self) -> None:
        """Ask the link what it holds."""
        try:
            body = json.loads(Upstream.at(self.url).request("GET", "/api/holdings"))
            self.node = Node(str(body["node"]["id"]), str(body["node"]["name"]))
            self.offers = {str(d["id"]): [str(v) for v in d["via"]] for d in body["dirs"]}
            self.state, self.error = "connected", ""
        except KeyError:
            self.offers, self.state, self.error = None, "unreachable", f"{self.url} answers with no holdings; is it a trex of this version?"
        except (Unreachable, TypeError, ValueError) as e:
            self.offers, self.state, self.error = None, "unreachable", str(e)


class Roots:
    """The directories this trex serves, by id. It crawls some (`specs`): a local path, with a polling Explorer, or
    host:path, a `Mirror` of the trex it starts there over ssh; their ids are `node name:path` and host:path. It
    pulls the others from its links (other trex) as Mirrors, each through the link offering it by the fewest nodes
    that are not this one. Workspaces are named sets of directories. Saved to `state`; every directory or link added
    or removed is remembered, most recent first, in history.json beside it."""

    def __init__(self, cache: Path, state: Path, name: str | None = None) -> None:
        self.cache = cache
        self.state = state
        self.history_path = state.with_name("history.json")
        self.node = node_at(state.with_name("node.json"), name)
        self.lock = threading.Lock()
        self.entries: dict[str, Explorer] = {}  # directory id -> what answers for it
        self.specs: dict[str, str] = {}  # directory id -> its spec, for the directories this trex crawls
        self.links: dict[str, Link] = {}  # by url
        self.pulled: dict[str, Pulled] = {}  # directory id -> where it comes from, for the directories pulled
        self.workspaces: dict[str, list[str]] = {}  # name -> member directory ids
        self._views: dict[str, Workspace] = {}
        self._stop = threading.Event()
        self._pull = threading.Event()  # set to pull from the links at once
        self._puller: threading.Thread | None = None
        try:
            saved = json.loads(self.history_path.read_text())
        except (FileNotFoundError, ValueError):
            saved = []
        self._history: list[str] = [str(r) for r in saved] if isinstance(saved, list) else []

    def load(self) -> None:
        """Add the directories, links and workspaces saved in `state`, skipping directories that no longer exist."""
        try:
            saved = json.loads(self.state.read_text())
        except FileNotFoundError:
            return
        for spec in map(str, [*saved.get("tracked", []), *saved.get("links", [])]):
            try:
                self.track(spec, force=True, wait=False)
            except ValueError as e:
                print(f"[trex] skipping saved directory {spec}: {e}", file=sys.stderr, flush=True)
        with self.lock:
            for w in saved.get("workspaces", []):
                self.workspaces[str(w.get("name"))] = [self._id(m) for m in map(str, w.get("members", []))]

    def _id(self, spec: str) -> str:
        """The id of the directory `spec` names: a local path is this node's."""
        return spec if remote.parse(spec) or not Path(spec).is_absolute() else f"{self.node.name}:{spec}"

    def track(self, spec: str, force: bool = False, wait: bool = True) -> str:
        """Name of the tracked directory `spec` (a path or host:path), starting to serve it if new, or "" for a trex to
        pull from (http://host:port); ValueError if it cannot be served."""
        if (url := link_url(spec)) is not None:
            self.add_link(url, wait)
            return ""
        if remote.parse(spec):
            return self.add_remote(spec, wait)
        return self.add(resolve_root(spec, force))

    def add(self, root: Path) -> str:
        """Name of the tracked directory `root`, starting to serve it if new."""
        spec = str(root.resolve())
        d = self._id(spec)
        with self.lock:
            if d not in self.specs:
                self._unpull(d)
                self.entries[d], self.specs[d] = Explorer(Crawl(spec), self.cache).start(), spec
                self._changed(spec)
            return self.names()[d]

    def add_remote(self, spec: str, wait: bool = True) -> str:
        """Name of the tracked remote directory `spec` (host:path), starting it if new. With `wait`, wait for its
        first start, and raise ValueError (forgetting it) if that fails."""
        addr = remote.parse(spec)
        if addr is None:
            raise ValueError(f"{spec} is not a host:path address")
        spec = f"{addr.host}:{addr.path}"
        with self.lock:
            if spec in self.specs:
                return self.names()[spec]
        r = Remote(spec, addr).start()
        if wait and r.settled.wait(remote.ADD_TIMEOUT) and r.state == "unreachable":
            r.close()
            raise ValueError(f"cannot serve {spec}: {r.error}")
        m = Explorer(Pull(Upstream(lambda timeout: UnixHTTPConnection(r.local, timeout), spec), spec, session=r), self.cache)
        with self.lock:
            if spec in self.specs:
                m.close()
            else:
                self._unpull(spec)
                self.entries[spec], self.specs[spec] = m.start(), spec
                self._changed(spec)
            return self.names()[spec]

    def add_link(self, url: str, wait: bool = True) -> None:
        """Pull every directory the trex at `url` holds. With `wait`, ask it first, and raise ValueError if it does not
        answer or is this trex."""
        link = Link(url)
        if wait:
            link.refresh()
            if link.offers is None:
                raise ValueError(f"cannot pull from {url}: {link.error}")
            if link.node is not None and link.node.id == self.node.id:
                raise ValueError(f"{url} is this trex")
        with self.lock:
            if url in self.links:
                return
            self.links[url] = link
            self._changed(url)
        self._reconcile()
        with self.lock:
            if self._puller is None:
                self._puller = threading.Thread(target=self._pull_forever, name="trex-pull", daemon=True)
                self._puller.start()
        self._pull.set()

    def _pull_forever(self) -> None:
        """Ask every link what it holds every PULL_EVERY seconds, or at once when asked to, and pull what it offers."""
        while not self._stop.is_set():
            self._pull.clear()
            for link in list(self.links.values()):
                link.refresh()
            self._reconcile()
            self._pull.wait(PULL_EVERY)

    def _reconcile(self) -> None:
        """Pull every directory the links offer that this trex does not crawl, through the link offering it by the
        fewest nodes not counting this one, keeping a directory's link while that link offers it; stop pulling one
        that its link, answering, no longer offers (one whose link does not answer stays, to browse)."""
        closing: list[Explorer] = []
        with self.lock:
            best = self._offers()
            for d, (url, via) in best.items():
                held = self.pulled.get(d)
                link = self.links.get(held.link) if held else None
                if held is not None and link is not None and d in (link.offers or {}):
                    self.pulled[d] = Pulled(held.link, (link.offers or {})[d])
                    continue
                entry = self.entries.get(d)
                if entry is not None and isinstance(entry.origin, Pull):
                    entry.origin.retarget(Upstream.at(url, dir_base(d)))
                else:
                    self.entries[d] = Explorer(Pull(Upstream.at(url, dir_base(d)), d), self.cache).start()
                self.pulled[d] = Pulled(url, via)
                self._changed(None)
            for d, p in list(self.pulled.items()):
                link = self.links.get(p.link)
                if d not in best and (link is None or link.offers is not None):
                    closing.append(self.entries.pop(d))
                    del self.pulled[d]
                    self._changed(None)
        for e in closing:
            threading.Thread(target=e.close, name=f"trex-close-{e.origin.key}", daemon=True).start()

    def _offers(self) -> dict[str, tuple[str, list[str]]]:
        """{directory id: (link url, via)}: for each directory the links offer that this trex does not crawl, the link
        offering it by the fewest nodes, none of them this one."""
        best: dict[str, tuple[str, list[str]]] = {}
        for url, link in self.links.items():
            for d, via in (link.offers or {}).items():
                if d not in self.specs and self.node.id not in via and (d not in best or len(via) < len(best[d][1])):
                    best[d] = (url, via)
        return best

    def _unpull(self, d: str) -> None:
        """Stop pulling `d`, which this trex now crawls (inside the lock)."""
        if self.pulled.pop(d, None) is not None:
            threading.Thread(target=self.entries.pop(d).close, daemon=True).start()

    def holdings(self) -> dict[str, object]:
        """{node, dirs: [{id, via}]}: this trex, and each directory it holds with the nodes it came through, from the one
        that crawls it to this one."""
        with self.lock:
            dirs = [{"id": d, "via": [*self.pulled[d].via, self.node.id] if d in self.pulled else [self.node.id]}
                    for d in self.entries]
        return {"node": self.node._asdict(), "dirs": dirs}

    def names(self) -> dict[str, str]:
        """{directory id: display name}: uniquified over the specs of the directories this trex crawls and the ids of
        the others."""
        shown = {d: self.specs.get(d, d) for d in self.entries}
        names = unique_names(list(shown.values()))
        return {d: names[s] for d, s in shown.items()}

    def id_of(self, name: str) -> str:
        """The id of the directory named `name`; KeyError if none."""
        for d, n in self.names().items():
            if n == name:
                return d
        raise KeyError(name)

    def remove(self, name: str) -> None:
        """Stop serving the directory `name` (or pulling from the link `name`) and drop it from every workspace; its
        index cache stays on disk. ValueError for a directory pulled from a link: remove that link instead."""
        if name in self.links:
            return self.remove_link(name)
        with self.lock:
            d = self.id_of(name)
            if d in self.pulled:
                raise ValueError(f"{name} comes from {self.pulled[d].link}; remove that link to stop pulling it")
            entry, spec = self.entries.pop(d), self.specs.pop(d)
            for members in self.workspaces.values():
                if d in members:
                    members.remove(d)
            self._changed(spec)
        threading.Thread(target=entry.close, name=f"trex-close-{name}", daemon=True).start()

    def remove_link(self, url: str) -> None:
        """Stop pulling from the link `url`; what only it offered is no longer served."""
        with self.lock:
            del self.links[url]
            self._changed(url)
        self._reconcile()

    def close(self) -> None:
        """Stop serving every directory and pulling from every link (they stay saved)."""
        self._stop.set()
        self._pull.set()
        with self.lock:
            entries, self.entries, self.specs, self.pulled = list(self.entries.values()), {}, {}, {}
            self._views = {}
        for entry in entries:
            entry.close()

    def get(self, name: str) -> Explorer:
        with self.lock:
            return self.entries[self.id_of(name)]

    def by_id(self, d: str) -> Explorer:
        with self.lock:
            return self.entries[d]

    def served(self) -> list[RootInfo]:
        with self.lock:
            names = self.names()
            return [self._info(d, names[d]) for d in self.entries]

    def _info(self, d: str, name: str) -> RootInfo:
        origin, pulled = self.entries[d].origin, self.pulled.get(d)
        state, error = "local", ""
        if isinstance(origin, Pull) and isinstance(origin.session, Remote):
            state, error = origin.session.state, origin.session.error or origin.error
        elif isinstance(origin, Pull) and pulled is not None:
            link = self.links.get(pulled.link)
            state = "connected" if origin.connected else link.state if link else "unreachable"
            error = (link.error if link else "") or origin.error
        return {"name": name, "root": self.specs.get(d, d), "id": d, "url": root_url(name), "state": state,
                "error": error, "link": pulled.link if pulled else None}

    def links_info(self) -> list[LinkInfo]:
        with self.lock:
            return [{"url": u, "name": link.node.name if link.node else "", "state": link.state, "error": link.error}
                    for u, link in self.links.items()]

    # ---- workspaces ----

    def set_workspace(self, name: str, members: Sequence[str], old: str | None = None) -> None:
        """Create or replace workspace `name` (renaming `old`) with the tracked directories named `members`."""
        name = name.strip()
        if not name or "/" in name or name.startswith("."):
            raise ValueError(f"{name!r} is not a workspace name")
        with self.lock:
            ids = [self.id_of(m) for m in members]
            if name != old and name in self.workspaces:
                raise ValueError(f"a workspace named {name} exists")
            if old is not None and old != name:
                self.workspaces.pop(old, None)
            self.workspaces[name] = ids
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
        """The root view: every directory, each a top-level folder."""
        with self.lock:
            view = self._views.get("")
            if view is None:
                view = self._views[""] = Workspace("/", self._members(list(self.entries)), nested=True)
            return view

    def _members(self, ids: Sequence[str]) -> list[Member]:
        names = self.names()
        return [Member(names[d], self.entries[d]) for d in ids if d in self.entries]

    def workspace_list(self) -> list[dict[str, object]]:
        with self.lock:
            names = self.names()
            return [{"name": w, "url": workspace_url(w), "members": [names[d] for d in ids if d in names]}
                    for w, ids in self.workspaces.items()]

    # ---- history and saving ----

    def history(self) -> list[str]:
        """Remembered directories and links that are not served, most recent first; local directories only while they
        exist."""
        with self.lock:
            held = {*self.specs.values(), *self.links}
            return [r for r in self._history if r not in held and (remote.parse(r) is not None or link_url(r) is not None
                                                                   or Path(r).is_dir())]

    def clear_history(self) -> None:
        with self.lock:
            self._history = []
            _write_json(self.history_path, self._history)

    def _changed(self, spec: str | None) -> None:
        """After a change (inside the lock): remember `spec`, rebuild workspace views, save."""
        if spec is not None:
            self._history = [spec, *(r for r in self._history if r != spec)][:HISTORY_MAX]
            _write_json(self.history_path, self._history)
        self._views = {}
        _write_json(self.state, {"tracked": list(self.specs.values()), "links": list(self.links),
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
        if req.get("op") == "status":
            return {"urls": self.urls, "roots": self.roots.served()}
        if req.get("op") == "add":
            name = self.roots.track(str(req.get("path")), bool(req.get("force")))
            return {"name": name, "url": self.urls[0].rstrip("/") + (root_url(name) if name else "/")}
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
