"""Nodes: every trex process is one. A node holds runs directories and serves them, with the web UI, to browsers and
to the other trex that pull from it.

    trex serve ~/runs                     # this machine's trex: started when none runs, else handed ~/runs
    trex serve gpu-box:~/runs             # crawled on gpu-box over ssh
    trex serve http://workstation:13898   # every directory the trex there holds, pulled

This machine's trex keeps what it holds (in $TREX_DAEMON_DIR, by default ~/.local/state/trex) and is reached over its
control socket there; `trex serve --temporary` runs one of its own that keeps nothing. `/` shows every directory a node
holds, each a top-level folder (a temporary node given one directory shows it alone); clicking trex (top left) opens
the panel that switches and manages them, and workspaces: named sets of directories whose folder trees are merged, so
runs from several machines can be compared side by side (group by `dir`).

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

As a runit service (Void Linux), started by the system's runsvdir and run as you (log in syslog, tagged trex; with
socklog: `svlogtail daemon`):

    uv tool install git+https://github.com/mishmish66/trackosaurus
    sudo mkdir -p /etc/sv/trex/log
    trex runit-service | sudo tee /etc/sv/trex/run >/dev/null && sudo chmod +x /etc/sv/trex/run
    sudo ln -s /usr/bin/vlogger /etc/sv/trex/log/run
    sudo ln -s /etc/sv/trex /var/service/

The update button in the panel installs the newest trex from the `--source` given to `trex systemd-unit`,
`trex launchd-plist` or `trex runit-service` (default the git source this trex was installed from, such as a branch of
this repository, else this repository), and systemd, launchd or runit restarts it on it.

A directory given as `host:path` is crawled on that machine over ssh: the node copies this trex there and runs it
with uvx (`trex.remote`), one for all the directories it tracks on that host, so the machine needs only uv and an ssh
key that works without a prompt. On a cluster, use a host that allows long-running processes, such as a data-transfer
node; runs written by jobs on other nodes are followed through their journals.

A node keeps every directory it pulls, or has crawled over ssh, in its cache (`trex.mirror`) as the trex crawling it
compiled it, so it answers at once and while that trex is unreachable; running runs' rows come live from it while it
answers. A laptop's trex pulling a lab workstation's, which crawls the training machines over ssh, holds everything the
workstation holds without reaching those machines itself:

    workstation$ trex serve ~/runs gpu-box:~/runs
    laptop$ trex serve http://workstation:13898

Nodes may pull from each other in any arrangement, cycles included. A directory has one id wherever it is held:
`<node name>:<path>` for one a node crawls (`--name`, the host's by default), its `host:path` for one crawled over ssh;
it is one top-level folder wherever it is held. A node offers each directory it holds (`holdings`) with `via`, the
nodes it came through, from the one crawling it, and takes each directory it does not crawl itself through the link
offering it by the shortest `via` without itself. It keeps a directory's link while that link offers it, and lets a
directory go once its link answers without it (one whose link does not answer stays, to browse).

The panel shows each directory under the trex crawling it, and each trex under the one it is reached through. A trex's
"+ add" box adds a directory there, however many links away: that trex crawls the path (one reached over ssh is told
to by the trex before it, which keeps it as host:path), and every trex on the way pulls it. A directory's × removes it
the same way, wherever it is crawled; its run files are kept. Each trex shows the trex it runs, and one running as a service has
an update button there; one reached over ssh runs the trex of the one before it and updates with it.
"""

import contextlib
import dataclasses
import json
import os
import socket
import sys
import threading
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Self
from urllib.parse import quote, urlsplit

from . import remote, update
from .format import JSONValue, as_dict, as_str_list
from .crawl import Crawl
from .index import Explorer
from .mirror import TIMEOUT, Connect, Pull, Unreachable, Upstream
from .remote import Remote, UnixHTTPConnection
from .workspace import Member, Workspace

HISTORY_MAX: Final = 50  # directories and links remembered for re-adding
PULL_EVERY = 10.0  # seconds between asking each link what it holds
ADD_SLACK: Final = 30.0  # seconds a node passing an add on waits, per node after it, beyond an ssh start's wait


@dataclass(frozen=True, slots=True)
class Identity:
    """Which trex a node is: `id`, unique, is what `via` lists; `name` (the host's, unless given) begins the ids of the
    directories it crawls."""

    id: str
    name: str

    def wire(self) -> dict[str, str]:
        return {"id": self.id, "name": self.name}


@dataclass(frozen=True, slots=True)
class Offer:
    """A directory a node holds, and the nodes it came through, from the one crawling it to that node."""

    id: str
    via: list[str]


@dataclass(frozen=True, slots=True, kw_only=True)
class Peer:
    """A node another reaches, and the nodes between: `via`, from it to the node telling of it; the trex it runs (None
    from a trex that tells of none), whether it can update itself (`trex.update`) and why not, and whether the node
    after it on `via` reaches it over ssh (it then runs that node's trex)."""

    id: str
    name: str
    via: list[str]
    install: update.Install | None = None
    updates: bool = False
    why: str = ""
    ssh: bool = False

    def wire(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "via": self.via, "install": self.install.wire() if self.install else None,
                "updates": self.updates, "why": self.why, "ssh": self.ssh}

    @classmethod
    def read(cls, d: Mapping[str, Any]) -> Self:
        install = d.get("install")
        return cls(id=str(d["id"]), name=str(d["name"]), via=[str(v) for v in d["via"]],
                   install=update.Install.read(install) if install is not None else None,
                   updates=d.get("updates") is True, why=str(d.get("why") or ""), ssh=d.get("ssh") is True)


@dataclass(frozen=True, slots=True)
class Holdings:
    """A node, every directory it holds, and every node it reaches (itself first; None from a trex that tells of
    none, and so passes no add on, `Node.add_at`)."""

    node: Identity
    dirs: list[Offer]
    nodes: list[Peer] | None

    def wire(self) -> dict[str, Any]:
        return {"node": self.node.wire(), "dirs": [{"id": o.id, "via": o.via} for o in self.dirs],
                "nodes": None if self.nodes is None else [p.wire() for p in self.nodes]}

    @classmethod
    def read(cls, d: Mapping[str, Any]) -> Self:
        node, nodes = d["node"], d.get("nodes")
        peers = None if nodes is None else [Peer.read(p) for p in nodes]
        return cls(Identity(str(node["id"]), str(node["name"])), [Offer(str(o["id"]), [str(v) for v in o["via"]]) for o in d["dirs"]],
                   peers)


@dataclass(frozen=True, slots=True)
class Pulled:
    """Where a directory a node does not crawl comes from: the link it is pulled through, and the nodes it came
    through to that link, from the one crawling it."""

    link: str
    via: list[str]


@dataclass(frozen=True, slots=True, kw_only=True)
class Saved:
    """What a node keeps across restarts (`roots.json`): the directories it tracks, its links (ssh sessions aside),
    the directories it pulls, and its workspaces' members."""

    tracked: list[str]
    links: list[str]
    pulled: dict[str, Pulled]
    workspaces: dict[str, list[str]]

    def wire(self) -> dict[str, Any]:
        return {"tracked": self.tracked, "links": self.links,
                "pulled": {d: [p.link, p.via] for d, p in self.pulled.items()},
                "workspaces": [{"name": w, "members": m} for w, m in self.workspaces.items()]}

    @classmethod
    def read(cls, v: JSONValue) -> Self:
        d = as_dict(v)
        pulled = {k: p for k, p in as_dict(d.get("pulled")).items() if isinstance(p, list) and len(p) == 2}
        workspaces = [as_dict(w) for w in ws] if isinstance(ws := d.get("workspaces"), list) else []
        return cls(tracked=as_str_list(d.get("tracked")), links=as_str_list(d.get("links")),
                   pulled={k: Pulled(str(p[0]), as_str_list(p[1])) for k, p in pulled.items()},
                   workspaces={str(w.get("name")): as_str_list(w.get("members")) for w in workspaces})


@dataclass(frozen=True, slots=True, kw_only=True)
class DirInfo:
    """A directory as the panel shows it: `root` is how the node holds it (a path or host:path it tracks, else its id),
    `link` the link it is pulled from unless the node tracks it, `via` the nodes it comes through, from the one
    crawling it to a link of this node (none for one crawled here)."""

    name: str
    root: str
    id: str
    url: str
    state: str  # local, or a pulled directory's connection state
    error: str
    link: str | None
    via: list[str]

    def wire(self) -> dict[str, Any]:
        return {"name": self.name, "root": self.root, "id": self.id, "url": self.url, "state": self.state,
                "error": self.error, "link": self.link, "via": self.via}


@dataclass(frozen=True, slots=True, kw_only=True)
class LinkInfo:
    """A link as the panel shows it: the node's id and name once it answered."""

    url: str
    id: str
    name: str
    state: str  # connecting, starting, connected or unreachable
    error: str

    def wire(self) -> dict[str, str]:
        return {"url": self.url, "id": self.id, "name": self.name, "state": self.state, "error": self.error}


def host_name() -> str:
    return socket.gethostname().split(".")[0] or "localhost"


def identity_at(path: Path | None, name: str | None = None) -> Identity:
    """The identity saved at `path`, made and saved there on first use (a new one each time without a path); renamed
    `name` when given."""
    saved = None
    with contextlib.suppress(FileNotFoundError, ValueError, KeyError, TypeError):
        d = json.loads(path.read_text()) if path else None
        saved = Identity(str(d["id"]), str(d["name"])) if d else None
    out = Identity(saved.id if saved else uuid.uuid4().hex[:16], name or (saved.name if saved else host_name()))
    if path is not None and out != saved:
        write_json(path, out.wire())
    return out


def link_url(spec: str) -> str | None:
    """The trex `spec` names (http://host[:port]), or None for a directory."""
    u = urlsplit(spec.strip())
    return f"http://{u.netloc}" if u.scheme == "http" and u.hostname else None


def dir_base(d: str) -> str:
    """The path under which a trex serves the directory whose id is `d`."""
    return f"/d/{quote(d, safe='')}"


def workspace_url(name: str) -> str:
    return f"/w/{quote(name, safe='')}/"


def resolve_root(root: str | os.PathLike[str], force: bool) -> Path:
    """`root` resolved; ValueError for a non-directory, or for / and $HOME (whose crawl would sweep a whole machine or
    home directory) unless `force`."""
    r = Path(root).expanduser().resolve()
    if not r.is_dir():
        raise ValueError(f"{root} is not a directory")
    if not force and r in (Path("/"), Path.home().resolve()):
        raise ValueError(f"refusing to crawl {r}; point trex at a runs directory (or pass --force)")
    return r


def unique_names(specs: Sequence[str]) -> dict[str, str]:
    """Display names of directories, as Emacs's uniquify does: the basename, and where basenames collide,
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


def _via(pulled: Pulled | None, link: "Link | None") -> list[str]:
    """The nodes a directory comes through, from the one crawling it to a link of this node: none for one crawled here,
    the ssh link's node for one crawled over ssh and not pulled yet."""
    if pulled is not None:
        return pulled.via
    return [link.identity.id] if link is not None and link.identity is not None else []


def write_json(path: Path, obj: object) -> None:
    """Replace `path` atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1) + "\n")
    tmp.replace(path)


class Link:
    """A trex a node pulls every directory of, reached through `connect`, by `key`: its URL, or the host of one the
    node starts over ssh (`session`), which it asks to crawl `crawls` (path there -> the directory's id)."""

    def __init__(self, key: str, connect: Connect, session: Remote | None = None) -> None:
        self.key, self.connect, self.session = key, connect, session
        self.crawls: dict[str, str] = {}
        self.state = "connecting"  # connecting, starting, connected or unreachable
        self.error = ""
        self.identity: Identity | None = None
        self.offers: dict[str, list[str]] | None = None  # directory id -> the nodes it came through; None while unanswered
        self.peers: list[Peer] | None = None  # the nodes it reaches, itself first; None from a trex that tells of none
        self._api = Upstream(connect, key)

    @classmethod
    def http(cls, url: str) -> Self:
        return cls(url, Upstream.at(url).connect)

    @classmethod
    def ssh(cls, host: str) -> Self:
        """The trex this starts on `host` over ssh."""
        session = Remote(host).start()
        return cls(host, lambda timeout: UnixHTTPConnection(session.local, timeout), session)

    def upstream(self, d: str) -> Upstream:
        """Directory `d` as this link serves it."""
        return Upstream(self.connect, f"{self.key} {d}", dir_base(d))

    def refresh(self) -> None:
        """Ask the link what it holds, having it crawl first the directories of `crawls` it lacks."""
        if self.session is not None and self.session.state != "connected":
            self.offers, self.state, self.error = None, self.session.state, self.session.error
            return
        try:
            offers = self._holdings()
            missing = [(path, d) for path, d in self.crawls.items() if d not in offers]
            errors = [e for path, d in missing if (e := self._crawl(path, d))]
            self.offers = self._holdings() if missing else offers
            self.state, self.error = "connected", "; ".join(errors)
        except KeyError:
            self.offers, self.state, self.error = None, "unreachable", f"{self.key} answers with no holdings; is it a trex of this version?"
        except (Unreachable, TypeError, ValueError) as e:
            self.offers, self.state, self.error = None, "unreachable", str(e)

    def _holdings(self) -> dict[str, list[str]]:
        held = Holdings.read(json.loads(self._api.request("GET", "/api/holdings")))
        self.identity, self.peers = held.node, held.nodes
        return {o.id: o.via for o in held.dirs}

    def add(self, path: str, at: Sequence[str]) -> str | None:
        """Have the link's node track `path`, or the node at the end of `at` (node ids, from one the link's node
        links to); the directory's id (None for a trex to pull from). ValueError with why, when it will not."""
        reply = self._post("/api/node/add", {"path": path, **({"at": list(at)} if at else {})},
                           remote.ADD_TIMEOUT + ADD_SLACK * (len(at) + 1))
        return d if isinstance(d := reply.get("id"), str) else None

    def remove(self, d: str, at: Sequence[str]) -> None:
        """Have the link's node stop tracking directory `d`, or the node at the end of `at` (as `add`). ValueError with
        why, when it will not."""
        self._post("/api/node/remove", {"id": d, **({"at": list(at)} if at else {})}, TIMEOUT + ADD_SLACK * len(at))

    def update(self, at: Sequence[str]) -> dict[str, JSONValue]:
        """Have the link's node update its trex, or the node at the end of `at` (as `add`): its answer ({updated, from,
        to, output}). ValueError with why, when it will not."""
        return self._post("/api/node/update", {"at": list(at)} if at else {}, update.INSTALL_TIMEOUT + ADD_SLACK * (len(at) + 1))

    def _post(self, target: str, body: dict[str, JSONValue], timeout: float) -> dict[str, JSONValue]:
        """The JSON object answering a POST of `body` to the link's node; ValueError with its error, or when it does
        not answer."""
        try:
            status, data = self._api.answer("POST", target, json.dumps(body).encode(), timeout)
            reply = as_dict(json.loads(data))
        except (Unreachable, ValueError) as e:
            raise ValueError(f"{self.key} did not answer: {e}") from e
        if status != 200:
            raise ValueError(str(reply.get("error") or f"{self.key} answered {status}"))
        return reply

    def _crawl(self, path: str, d: str) -> str:
        """Have the link crawl `path` as directory `d`; why it would not, if it would not."""
        try:
            self._api.request("POST", "/api/node/add", json.dumps({"path": path, "id": d}).encode())
        except Unreachable as e:
            return str(e)
        return ""

    def forget(self, d: str) -> None:
        """Stop having the link crawl directory `d`."""
        self.crawls = {path: x for path, x in self.crawls.items() if x != d}
        with contextlib.suppress(Unreachable, KeyError):
            self._api.request("POST", "/api/node/remove", json.dumps({"id": d}).encode())

    def close(self) -> None:
        self._api.retire()
        if self.session is not None:
            self.session.close()


class Node:
    """The directories a trex holds, under `cache`, and the trex it pulls from; saved to `state` (a daemon's) or kept
    only while it runs (None); named `name` (its host's by default). `/` shows `home` when it is set (`trex serve DIR`),
    else every directory, each a top-level folder."""

    def __init__(self, cache: Path, state: Path | None = None, name: str | None = None) -> None:
        self.cache, self.state = cache, state
        self.identity = identity_at(state.with_name("node.json") if state else None, name)
        self.lock = threading.Lock()
        self.entries: dict[str, Explorer] = {}  # directory id -> what answers for it
        self.crawled: dict[str, str] = {}  # directory id -> its path, for the directories this node crawls
        self.links: dict[str, Link] = {}  # by key
        self.pulled: dict[str, Pulled] = {}  # directory id -> where it comes from, for the others
        self.workspaces: dict[str, list[str]] = {}  # name -> member directory ids
        self.home: str | None = None
        self._views: dict[str, Workspace] = {}
        self._stop = threading.Event()
        self._pull = threading.Event()  # set to pull from the links at once
        self._puller: threading.Thread | None = None
        self._history: list[str] = []
        if state is not None:
            with contextlib.suppress(FileNotFoundError, ValueError):
                self._history = as_str_list(json.loads(state.with_name("history.json").read_text()))

    @property
    def saves(self) -> bool:
        return self.state is not None

    def load(self) -> None:
        """Track the directories and links saved in `state`, skipping directories that no longer exist; open the
        directories it pulled, from the cache, so they answer before their links do; keep its workspaces."""
        try:
            saved = Saved.read(json.loads(self.state.read_text()) if self.state else None)
        except FileNotFoundError:
            return
        for spec in [*saved.tracked, *saved.links]:
            try:
                self.track(spec, force=True, wait=False)
            except ValueError as e:
                print(f"[trex] skipping saved directory {spec}: {e}", file=sys.stderr, flush=True)
        with self.lock:
            for d, p in saved.pulled.items():
                if p.link in self.links and d not in self.entries:
                    self.entries[d] = Explorer(Pull(self.links[p.link].upstream(d), d), self.cache).start()
                    self.pulled[d] = p
            for w, members in saved.workspaces.items():
                self.workspaces[w] = [self._id(m) for m in members]
            self._update_views()

    def _id(self, spec: str) -> str:
        """The id of the directory `spec` names: a local path is this node's."""
        return spec if remote.parse(spec) or not Path(spec).is_absolute() else f"{self.identity.name}:{spec}"

    # ---- adding and removing ----

    def track(self, spec: str, force: bool = False, wait: bool = True) -> str | None:
        """Hold the directory `spec` names (a path, or host:path crawled over ssh), or pull from the trex at an
        http://host:port; the directory's id, or None for a trex. ValueError if it cannot be held."""
        if (url := link_url(spec)) is not None:
            self.add_link(url, wait)
            return None
        if remote.parse(spec):
            return self.add_remote(spec, wait)
        return self.add(resolve_root(spec, force))

    def add(self, root: Path, d: str | None = None) -> str:
        """Crawl `root` as directory `d` (by default this node's name and the path); its id."""
        path = str(root.resolve())
        d = d or self._id(path)
        with self.lock:
            if d in self.crawled:
                return d
        self.hold(d, Explorer(Crawl(path), self.cache).start(), path)
        return d

    def hold(self, d: str, ex: Explorer, path: str) -> None:
        """Serve `ex`, the Explorer of the runs directory `path` here, as directory `d`."""
        with self.lock:
            self._unpull(d)
            self.entries[d], self.crawled[d] = ex, path
            self._changed(path)

    def add_remote(self, spec: str, wait: bool = True) -> str:
        """Have the trex this node starts on the host of `spec` (host:path) crawl its path, and pull it; its id. With
        `wait`, wait for that and raise ValueError (forgetting it) if it fails."""
        addr = remote.parse(spec)
        if addr is None:
            raise ValueError(f"{spec} is not a host:path address")
        d = f"{addr.host}:{addr.path}"
        with self.lock:
            link = self.links.get(addr.host)
            if link is None:
                link = self.links[addr.host] = Link.ssh(addr.host)
            if d in link.crawls.values():
                return d
            link.crawls[addr.path] = d
        if wait:
            self._settle(link, d)
        with self.lock:
            self._changed(d)
        self._start_pulling()
        return d

    def _settle(self, link: Link, d: str) -> None:
        """Wait for the ssh link's first start and its crawl of `d`, then pull it; ValueError, forgetting `d`, if
        either fails."""
        session = link.session
        if session is not None:
            session.settled.wait(remote.ADD_TIMEOUT)
        link.refresh()
        if link.offers is not None and d in link.offers:
            return self.reconcile()
        why = link.error or (session.error if session else "") or f"{link.key} did not answer"
        self._forget_remote(link, d)
        raise ValueError(f"cannot serve {d}: {why}")

    def add_link(self, url: str, wait: bool = True) -> None:
        """Pull every directory the trex at `url` holds. With `wait`, ask it first, and raise ValueError if it does not
        answer or is this trex."""
        link = Link.http(url)
        if wait:
            link.refresh()
            if link.offers is None:
                raise ValueError(f"cannot pull from {url}: {link.error}")
            if link.identity is not None and link.identity.id == self.identity.id:
                raise ValueError(f"{url} is this trex")
        with self.lock:
            if url in self.links:
                return
            self.links[url] = link
            self._changed(url)
        self.reconcile()
        self._start_pulling()

    def add_at(self, path: str, at: Sequence[str]) -> str | None:
        """Have the node at the end of `at` (node ids, from one this node links to) track `path`, a path there (or
        another spec it takes, `track`), and pull it here; the directory's id (None for a trex to pull from). The node
        whose ssh link reaches the last one tracks `host:path` itself, so it keeps it. ValueError if a node on the way
        refuses it or cannot pass it on (a trex telling of no nodes)."""
        if not at:
            return self.track(path)
        link = self._link_to(at[0])
        if link.session is not None and len(at) == 1:
            if remote.parse(path) or link_url(path):
                raise ValueError(f"{path} is not a path on {link.key}")
            return self.add_remote(f"{link.key}:{path}")
        if len(at) > 1 and link.peers is None:
            raise ValueError(f"{link.key} runs a trex that passes no add on; update it")
        got = link.add(path, at[1:])
        link.refresh()
        self.reconcile()
        return got

    def remove_at(self, d: str, at: Sequence[str]) -> None:
        """Have the node at the end of `at` (as `add_at`) stop tracking directory `d`, and let it go here once the link
        no longer offers it; the node whose ssh link reaches the last one removes its own `host:path`. KeyError if this
        node holds no such directory, ValueError if a node on the way refuses or cannot pass it on."""
        if not at:
            return self.remove_dir(d)
        link = self._link_to(at[0])
        if link.session is not None and len(at) == 1:
            return self.remove_dir(d)
        if len(at) > 1 and link.peers is None:
            raise ValueError(f"{link.key} runs a trex that passes no removal on; update it")
        link.remove(d, at[1:])
        link.refresh()
        self.reconcile()

    def update_at(self, at: Sequence[str]) -> dict[str, JSONValue]:
        """Have the node at the end of `at` (node ids, from one this node links to) update its trex; its answer.
        ValueError if a node on the way refuses or cannot pass it on, or the last is reached over ssh (it runs the
        trex of the node before it, which starts it anew when that one restarts)."""
        link = self._link_to(at[0])
        if link.session is not None and len(at) == 1:
            raise ValueError(f"{link.key} runs the trex of {self.identity.name}, which starts it anew when it restarts: update {self.identity.name}")
        if len(at) > 1 and link.peers is None:
            raise ValueError(f"{link.key} runs a trex that passes no update on; update it")
        return link.update(at[1:])

    def _link_to(self, node: str) -> Link:
        """This node's link to node `node` (an id); ValueError if it has none."""
        with self.lock:
            link = next((k for k in self.links.values() if k.identity is not None and k.identity.id == node), None)
        if link is None:
            raise ValueError(f"{self.identity.name} has no link to node {node}")
        return link

    def remove(self, name: str) -> None:
        """Stop serving the directory `name` (or pulling from the link `name`) and drop it from every workspace; its
        index stays in the cache. ValueError for a directory pulled from an http link: remove that link instead."""
        if name in self.links and self.links[name].session is None:
            return self.remove_link(name)
        self.remove_dir(self.id_of(name))

    def remove_dir(self, d: str) -> None:
        """Stop serving directory `d`, which this node tracks (crawls, or has crawled over ssh), and drop it from every
        workspace; KeyError if it holds no such directory, ValueError if it is pulled from an http link."""
        with self.lock:
            link = self._ssh_link_of(d)
            if d not in self.crawled and link is None:
                if d not in self.pulled:
                    raise KeyError(d)
                raise ValueError(f"{d} comes from {self.pulled[d].link}; remove that link to stop pulling it")
            for members in self.workspaces.values():
                if d in members:
                    members.remove(d)
            path = self.crawled.pop(d, None)
            entry = self.entries.pop(d) if path is not None else None
            self._changed(path or d)
        if entry is not None:
            threading.Thread(target=entry.close, name=f"trex-close-{d}", daemon=True).start()
        if link is not None:
            self._forget_remote(link, d)

    def _ssh_link_of(self, d: str) -> Link | None:
        """The link to the trex crawling `d` over ssh for this node, if one does (inside the lock)."""
        return next((link for link in self.links.values() if link.session and d in link.crawls.values()), None)

    def _forget_remote(self, link: Link, d: str) -> None:
        """Stop having the ssh link crawl `d`, and pulling it, at once (the link asked anew); the link goes once it crawls
        nothing."""
        link.forget(d)
        with self.lock:
            kept = bool(link.crawls) or self.links.get(link.key) is not link
            if not kept:
                del self.links[link.key]
                threading.Thread(target=link.close, name=f"trex-close-{link.key}", daemon=True).start()
            self._changed(None)
        if kept:
            link.refresh()
        self.reconcile()

    def remove_link(self, url: str) -> None:
        """Stop pulling from the link `url`; what only it offered is no longer served."""
        with self.lock:
            link = self.links.pop(url)
            self._changed(url)
        link.close()
        self.reconcile()

    # ---- pulling ----

    def _start_pulling(self) -> None:
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
            self.reconcile()
            self._pull.wait(PULL_EVERY)

    def reconcile(self) -> None:
        """Pull every directory the links offer that this node does not crawl, through the link offering it by the
        fewest nodes not counting this one, keeping a directory's link while that link offers it; stop pulling one
        that its link, answering, no longer offers."""
        closing: list[Explorer] = []
        with self.lock:
            best = self._offers()
            for d, (key, via) in best.items():
                held = self.pulled.get(d)
                link = self.links.get(held.link) if held else None
                if held is not None and link is not None and d in (link.offers or {}):
                    self.pulled[d] = Pulled(held.link, (link.offers or {})[d])
                    continue
                entry = self.entries.get(d)
                upstream = self.links[key].upstream(d)
                if entry is not None and isinstance(entry.origin, Pull):
                    entry.origin.retarget(upstream)
                else:
                    self.entries[d] = Explorer(Pull(upstream, d), self.cache).start()
                self.pulled[d] = Pulled(key, via)
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
        """{directory id: (link key, via)}: for each directory the links offer that this node does not crawl, the link
        offering it by the fewest nodes, none of them this one."""
        best: dict[str, tuple[str, list[str]]] = {}
        for key, link in self.links.items():
            for d, via in (link.offers or {}).items():
                if d not in self.crawled and self.identity.id not in via and (d not in best or len(via) < len(best[d][1])):
                    best[d] = (key, via)
        return best

    def _unpull(self, d: str) -> None:
        """Stop pulling `d`, which this node now crawls (inside the lock)."""
        if self.pulled.pop(d, None) is not None:
            threading.Thread(target=self.entries.pop(d).close, daemon=True).start()

    def holdings(self) -> Holdings:
        """This node, each directory it holds with the nodes it came through, from the one crawling it to this one, and
        each node it reaches with the nodes between, itself first."""
        me = self.identity.id
        can = update.updates()
        peers = [Peer(id=me, name=self.identity.name, via=[me], install=update.RUNNING, updates=can.available, why=can.reason),
                 *(dataclasses.replace(p, via=[*p.via, me]) for p in self.peers())]
        with self.lock:
            return Holdings(self.identity, [Offer(d, [*self.pulled[d].via, me] if d in self.pulled else [me]) for d in self.entries], peers)

    def peers(self) -> list[Peer]:
        """Every node this one reaches through its links but itself, each by the fewest nodes from it to a link of
        this one (`via`); a link telling of no nodes, as itself alone."""
        best: dict[str, Peer] = {}
        with self.lock:
            for link in self.links.values():
                if (who := link.identity) is None:
                    continue
                for p in link.peers if link.peers is not None else [Peer(id=who.id, name=who.name, via=[who.id])]:
                    if self.identity.id not in p.via and (p.id not in best or len(p.via) < len(best[p.id].via)):
                        best[p.id] = dataclasses.replace(p, ssh=link.session is not None) if p.id == who.id else p
        return list(best.values())

    # ---- what it holds ----

    def names(self) -> dict[str, str]:
        """{directory id: display name} of every directory it tracks or holds: uniquified over the paths of the
        directories this node crawls and the ids of the others."""
        shown = {d: self.crawled.get(d, d) for d in self._dirs()}
        names = unique_names(list(shown.values()))
        return {d: names[s] for d, s in shown.items()}

    def _dirs(self) -> list[str]:
        """The directories it holds, then those it has crawled over ssh that it does not hold yet."""
        waiting = [d for link in self.links.values() if link.session for d in link.crawls.values() if d not in self.entries]
        return [*self.entries, *waiting]

    def id_of(self, name: str) -> str:
        """The id of the directory named `name`; KeyError if none."""
        for d, n in self.names().items():
            if n == name:
                return d
        raise KeyError(name)

    def get(self, name: str) -> Explorer:
        with self.lock:
            return self.entries[self.id_of(name)]

    def by_id(self, d: str) -> Explorer:
        with self.lock:
            return self.entries[d]

    def home_view(self) -> Explorer | Workspace:
        """What `/` shows."""
        with self.lock:
            entry = self.entries.get(self.home) if self.home else None
        return entry if entry is not None else self.everything()

    def served(self) -> list[DirInfo]:
        """Every directory it holds, and those it has crawled over ssh that it does not hold yet."""
        with self.lock:
            names = self.names()
            return [self._info(d, names[d]) for d in self._dirs()]

    def _info(self, d: str, name: str) -> DirInfo:
        pulled = self.pulled.get(d)
        link = self.links.get(pulled.link) if pulled else self._ssh_link_of(d)
        entry = self.entries.get(d)
        origin = entry.origin if entry else None
        state, error = "local", ""
        if entry is None:
            state, error = (link.state if link and link.state != "connected" else "unreachable"), link.error if link else ""
        elif isinstance(origin, Pull):
            state = "connected" if origin.connected else link.state if link else "unreachable"
            error = (link.error if link else "") or origin.error
        root = self.crawled.get(d, d)
        return DirInfo(name=name, root=root, id=d, url=dir_base(d) + "/", state=state, error=error,
                       link=link.key if link and link.session is None else None, via=_via(pulled, link))

    def links_info(self) -> list[LinkInfo]:
        with self.lock:
            return [LinkInfo(url=k, id=link.identity.id if link.identity else "", name=link.identity.name if link.identity else "",
                             state=link.state, error=link.error)
                    for k, link in self.links.items() if link.session is None]

    # ---- workspaces ----

    def set_workspace(self, name: str, members: Sequence[str], old: str | None = None) -> None:
        """Create or replace workspace `name` (renaming `old`) with the directories named `members`."""
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
        """The workspace `name`, as a view of its members."""
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

    def _update_views(self) -> None:
        """Each view's members as they now are (inside the lock), in place, so that open streams follow them; views of
        workspaces gone are dropped."""
        for name, view in list(self._views.items()):
            if not name:
                view.members = self._members(list(self.entries))
            elif name in self.workspaces:
                view.members = self._members(self.workspaces[name])
            else:
                del self._views[name]

    def _members(self, ids: Sequence[str]) -> list[Member]:
        names = self.names()
        return [Member(names[d], self.entries[d]) for d in ids if d in self.entries]

    def workspace_list(self) -> list[dict[str, Any]]:
        with self.lock:
            names = self.names()
            return [{"name": w, "url": workspace_url(w), "members": [names[d] for d in ids if d in names]}
                    for w, ids in self.workspaces.items()]

    # ---- history and saving ----

    def tracked(self) -> list[str]:
        """The directories this node tracks, as given: paths, and host:path for those crawled over ssh."""
        return [*self.crawled.values(), *(f"{link.key}:{path}" for link in self.links.values() if link.session for path in link.crawls)]

    def history(self) -> list[str]:
        """Remembered directories and links that are not served, most recent first; local directories only while they
        exist."""
        with self.lock:
            held = {*self.tracked(), *self.links}
            return [r for r in self._history if r not in held and (remote.parse(r) is not None or link_url(r) is not None
                                                                   or Path(r).is_dir())]

    def clear_history(self) -> None:
        with self.lock:
            self._history = []
            if self.state:
                write_json(self.state.with_name("history.json"), self._history)

    def _changed(self, spec: str | None) -> None:
        """After a change (inside the lock): remember `spec`, update the views' members, save."""
        self._update_views()
        if self.state is None:
            return
        if spec is not None:
            self._history = [spec, *(r for r in self._history if r != spec)][:HISTORY_MAX]
            write_json(self.state.with_name("history.json"), self._history)
        write_json(self.state, Saved(tracked=self.tracked(), links=[k for k, link in self.links.items() if link.session is None],
                                     pulled=dict(self.pulled), workspaces=dict(self.workspaces)).wire())

    def close(self) -> None:
        """Stop serving every directory and pulling from every link (a daemon's stay saved)."""
        self._stop.set()
        self._pull.set()
        with self.lock:
            entries, self.entries, self.crawled, self.pulled = list(self.entries.values()), {}, {}, {}
            links, self.links, self._views = list(self.links.values()), {}, {}
        for entry in entries:
            entry.close()
        for link in links:
            link.close()
