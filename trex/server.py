"""HTTP + SSE for a node (`trex.node`): its UI and its API, for browsers and for the other trex pulling from it.

`/` is the node's home view (`Node.home_view`), `/w/<name>/` a workspace, `/d/<id>/` one directory; under each, the
same API: runs, blocks, rows, the stream, media, dumps. `/api/node` and the routes under it manage the node;
`/api/holdings` lists what it holds for other trex.
"""

import contextlib
import errno
import gzip
import ipaddress
import json
import os
import re
import socket
import socketserver
import sys
import threading
from collections.abc import Callable, Mapping, Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Final, cast
from urllib.parse import parse_qs, unquote, urlsplit

from . import buckets as bk, remote, update
from .format import JSONValue, as_str_list
from .index import Ask, Explorer, Have, Which, dumps
from .node import Node, dir_base, link_url, resolve_root, workspace_url
from .workspace import Workspace

type Query = dict[str, str]
type RouteFn = Callable[..., None]

STATIC: Final = Path(__file__).parent / "static"
DEFAULT_PORT: Final = 13898
PORT_TRIES: Final = 20  # ports tried from DEFAULT_PORT when none is given
ROOT_PREFIX: Final = re.compile(r"/([wd])/([^/]+)(/.*)?")  # a directory's URLs (d, by id), a workspace's (w, by name)
PROTOCOL: Final = 8  # what the UI and this server say to each other; the UI states a mismatch (data.js PROTOCOL)
WHICH: Final[dict[str, Which]] = {"all": "all", "finished": "finished", "running": "running"}
MAX_ASKS: Final = 256  # blocks one POST /api/buckets may ask for
MAX_DUMPS: Final = 256  # runs one POST /api/dumps may ask for
RESTART_DELAY: Final = 0.5  # seconds between answering an update and restarting, so the answer is sent
CTYPES: Final = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".svg": "image/svg+xml",
    ".mp4": "video/mp4",
    ".webm": "video/webm",
}

ROUTES: list[tuple[str, re.Pattern[str], RouteFn]] = []


def route[F: RouteFn](method: str, pattern: str) -> Callable[[F], F]:
    """Route requests whose path matches `pattern`; its groups follow the query argument."""

    def deco(fn: F) -> F:
        ROUTES.append((method, re.compile(pattern), fn))
        return fn

    return deco


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "trex"

    _ex: "Explorer | Workspace | None" = None
    _body: bytes = b""

    def setup(self) -> None:
        super().setup()
        if self.connection.family != socket.AF_UNIX:
            self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)  # each write leaves without waiting for an ACK

    @property
    def srv(self) -> "Server":
        return cast(Server, self.server)

    @property
    def ex(self) -> "Explorer | Workspace":
        """What this request is for: the node's home view, a directory (/d/<id>/) or a workspace (/w/<name>/)."""
        if self._ex is None:
            raise KeyError("runs directory")
        return self._ex

    def log_message(self, format: str, *args: object) -> None:
        pass

    def do_GET(self) -> None:
        self._dispatch("GET")

    def do_POST(self) -> None:
        self._dispatch("POST")

    def _dispatch(self, method: str) -> None:
        self._body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        u = urlsplit(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if (refusal := self._refusal(method)) is not None:
                return self._json({"error": refusal}, 403)
            path = self._scope(u.path, u.query)
            if path is None:
                return
            for m, rx, fn in ROUTES:
                mt = rx.fullmatch(path)
                if m == method and mt:
                    return fn(self, q, *map(unquote, mt.groups()))
            self._json({"error": "not found"}, 404)
        except KeyError as e:
            self._json({"error": f"not found: {e}"}, 404)
        except (ValueError, TypeError) as e:
            self._json({"error": f"bad request: {e!r}"}, 400)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            print(f"[trex] {method} {self.path}: {e!r}", file=sys.stderr, flush=True)
            self._json({"error": f"internal error: {e!r}"}, 500)

    def _refusal(self, method: str) -> str | None:
        """Why a request that may come from another site is refused: a Host that is not an address or a known
        name of this server (DNS rebinding), or a POST from a page of another origin."""
        host = self.headers.get("Host", "")
        name = (urlsplit(f"//{host}").hostname or "").lower()
        if host and not (_is_ip(name) or name in self.srv.names or name.endswith(".localhost")):
            return f"Host {name!r} is not one of this server's names; allow it with --allow-host {name}"
        origin = self.headers.get("Origin")
        if method != "GET" and origin is not None and urlsplit(origin).netloc.lower() != host.lower():
            return f"a request from {origin} is not allowed to change this server"
        return None

    def _scope(self, path: str, query: str) -> str | None:
        """Choose what the request is for and return the path within it, or None after redirecting a page that names no
        directory or workspace, or lacks its trailing slash."""
        node = self.srv.node
        pm = ROOT_PREFIX.fullmatch(path)
        if pm is None:
            self._ex = node.home_view()
            return path
        kind, name, rest = pm[1], unquote(pm[2]), pm[3]
        try:
            self._ex = node.by_id(name) if kind == "d" else node.workspace(name)
        except KeyError:
            if rest in (None, "/"):
                return self.redirect("/", permanent=False)
            raise
        if rest is None:
            return self.redirect(path + "/" + (f"?{query}" if query else ""))
        return rest

    def redirect(self, location: str, permanent: bool = True) -> None:
        """A 301, or a 302 that browsers must not cache."""
        self.send_response(301 if permanent else 302)
        self.send_header("Location", location)
        if not permanent:
            self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def body(self) -> bytes:
        """The request body, read whole before routing so that a kept-alive connection stays in step."""
        return self._body

    def send(self, body: bytes, ctype: str, status: int = 200, headers: Mapping[str, str] | None = None,
             compress: bool = False) -> None:
        headers = dict(headers or {})
        if compress and len(body) > 1024 and "gzip" in self.headers.get("Accept-Encoding", ""):
            body = gzip.compress(body, compresslevel=1)
            headers["Content-Encoding"] = "gzip"
            headers["Vary"] = "Accept-Encoding"
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in headers.items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj: object, status: int = 200) -> None:
        self.send(dumps(obj).encode(), "application/json", status, {"Cache-Control": "no-store"}, compress=True)

    def send_file(self, path: Path, etag: str, cache: str, extra: Mapping[str, str] | None = None,
                  compress: bool = False) -> None:
        if not path.is_file():
            raise KeyError(path.name)
        if self.headers.get("If-None-Match") == etag:
            self.send_response(304)
            self.send_header("ETag", etag)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        ctype = CTYPES.get(path.suffix, "application/octet-stream")
        headers = {"ETag": etag, "Cache-Control": cache, "X-Content-Type-Options": "nosniff", **(extra or {})}
        size = path.stat().st_size
        rng = re.fullmatch(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
        if compress or not rng:
            headers["Accept-Ranges"] = "none" if compress else "bytes"
            return self.send(path.read_bytes(), ctype, 200, headers, compress=compress)
        a, b = rng.groups()
        start, end = (int(a), int(b) if b else size - 1) if a else (max(0, size - int(b)), size - 1)
        end = min(end, size - 1)
        if start > end:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self.send_response(206)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        for k, v in headers.items():
            self.send_header(k, v)
        self.end_headers()
        with open(path, "rb") as f:
            f.seek(start)
            left = end - start + 1
            while left:
                chunk = f.read(min(left, 1 << 20))
                if not chunk:
                    break
                self.wfile.write(chunk)
                left -= len(chunk)

    # ---- routes ----

    @route("GET", r"/")
    def index(self, q: Query) -> None:
        self.static(q, "index.html")

    @route("GET", r"/static/([\w.-]+)")
    def static(self, q: Query, name: str) -> None:
        p = STATIC / name
        st = p.stat() if p.is_file() else None
        etag = f'"{st.st_mtime_ns:x}-{st.st_size:x}"' if st else '""'
        self.send_file(p, etag, "no-cache", compress=True)

    @route("GET", r"/m/([^/]+)/(media/[\w.-]+)")
    def media_file(self, q: Query, run: str, file: str) -> None:
        """Media are content-addressed (media/<sha256>.<ext>), so they are cached as immutable."""
        html = file.endswith(".html")
        extra = {"Content-Security-Policy": "sandbox allow-scripts"} if html else None
        ex = self.ex
        if isinstance(ex, Workspace):
            m, run = ex.resolve(run)
            ex = m.src
        self.send_file(ex.media_path(run, file), f'"{file}"', "public, max-age=31536000, immutable", extra, compress=html)

    @route("GET", r"/api/info")
    def info(self, q: Query) -> None:
        self._json({**self.ex.info(), "protocol": PROTOCOL})

    @route("GET", r"/api/node")
    def node(self, q: Query) -> None:
        """{node: {id, name}, saves, home, dirs: [{name, root, id, url, state, error, link, via}], links: [{url, id,
        name, state, error}], nodes: [{id, name, via}], workspaces: [{name, url, members}], history: [root], install,
        updates}: which node this is, whether it saves what it holds, the directory `/` shows (none: every one), its
        directories, the trex it pulls from, every node it reaches through them (`via` from that node to a link of this
        one), its workspaces, the remembered directories and links it does not serve, the trex it runs, and whether it
        can update."""
        node = self.srv.node
        self._json({"node": node.identity.wire(), "saves": node.saves, "home": node.home,
                    "dirs": [d.wire() for d in node.served()], "links": [link.wire() for link in node.links_info()],
                    "nodes": [p.wire() for p in node.peers()],
                    "workspaces": node.workspace_list(), "history": node.history(), "install": update.RUNNING.wire(),
                    "updates": update.updates().wire()})

    @route("GET", r"/api/holdings")
    def holdings(self, q: Query) -> None:
        """{node: {id, name}, dirs: [{id, via}], nodes: [{id, name, via}]}: this node, each directory it holds with the
        nodes it came through, from the one crawling it to this one (another trex pulls each from /d/<id>/), and each
        node it reaches with the nodes between, itself first."""
        self._json(self.srv.node.holdings().wire())

    def _parsed(self) -> JSONValue:
        """The JSON of the body."""
        return json.loads(self.body())

    def body_json(self) -> dict[str, JSONValue]:
        """The JSON object of the body; ValueError for anything else."""
        req = self._parsed()
        if not isinstance(req, dict):
            raise ValueError("expected a JSON object")
        return req

    def body_field(self, field: str) -> str:
        """String `field` of the JSON body."""
        value = self.body_json().get(field)
        if not isinstance(value, str):
            raise ValueError(f"expected {{{field}}}")
        return value

    @route("POST", r"/api/node/add")
    def node_add(self, q: Query) -> None:
        """Body: {path, id?, at?}: absolute, ~/…, host:path for a remote directory (which this waits to start), or
        http://host:port for a trex to pull every directory of; `id` names a local directory (a node asking this one to
        crawl it); `at` (node ids, from one this node links to) has the node at its end track it, and this one pull it
        (`Node.add_at`). Response: {name, id, url} (none and the root view's for a trex), or 400 with {error} for a path
        a node refuses, a remote that fails to start or a trex that does not answer."""
        node, req = self.srv.node, self.body_json()
        path, d = self.body_field("path").strip(), req.get("id")
        try:
            at = self.body_at()
            if not remote.parse(path) and not link_url(path) and not Path(path).expanduser().is_absolute():
                raise ValueError(f"{path} is not an absolute path, host:path or http://host:port")
            if at:
                got = node.add_at(path, at)
            else:
                got = node.add(resolve_root(path, force=False), str(d)) if isinstance(d, str) else node.track(path)
        except ValueError as e:
            return self._json({"error": str(e)}, 400)
        if got is None:
            return self._json({"name": None, "id": None, "url": "/"})
        self._json({"name": node.names().get(got, got), "id": got, "url": dir_base(got) + "/"})

    @route("POST", r"/api/node/remove")
    def node_remove(self, q: Query) -> None:
        """Body: {name} (a directory's name, or a link's url), or {id, at?}. Stops serving that directory, or pulling from
        that trex; with `at` (node ids, from one this node links to), has the node at its end stop tracking directory
        `id` (`Node.remove_at`). 400 for a directory pulled from a link, or one a node on the way will not remove."""
        node, req = self.srv.node, self.body_json()
        try:
            at = self.body_at()
            if "id" in req:
                node.remove_at(str(req["id"]), at)
            else:
                node.remove(self.body_field("name"))
        except ValueError as e:
            return self._json({"error": str(e)}, 400)
        self._json({"ok": True})

    @route("POST", r"/api/node/update")
    def node_update(self, q: Query) -> None:
        """Body: {at?}. Install the newest trex from $TREX_SOURCE and, if that changed it, restart the daemon; with `at`
        (node ids, from one this node links to), have the node at its end do so (`Node.update_at`).
        Response: {updated, from, to, output}; 404 for a node that is not a daemon, 400 when updates are unavailable
        (or a node on the way refuses), 409 during another update, 502 when the install fails."""
        try:
            at = self.body_at()
            if at:
                return self._json(self.srv.node.update_at(at))
        except ValueError as e:
            return self._json({"error": str(e)}, 400)
        self.update_here()

    def body_at(self) -> list[str]:
        """The node ids of the body's `at` (none without it, or without a body); ValueError for anything else."""
        at = self.body_json().get("at") or [] if self.body() else []
        if not isinstance(at, list) or not all(isinstance(n, str) for n in at):
            raise ValueError("expected at: [node ids]")
        return [str(n) for n in at]

    def update_here(self) -> None:
        """This node's update (`node_update`)."""
        restart = self.srv.restart
        if restart is None:
            raise KeyError("a daemon")
        can = update.updates()
        if not can.available or can.source is None:
            return self._json({"error": f"updates are unavailable: {can.reason}"}, 400)
        if not self.srv.updating.acquire(blocking=False):
            return self._json({"error": "an update is already running"}, 409)
        try:
            before = update.installed()
            output = update.install(can.source)
            after = update.installed()
        except update.UpdateError as e:
            self.srv.updating.release()
            return self._json({"error": str(e)[-4000:]}, 502)
        changed = after != before
        self._json({"updated": changed, "from": before.wire(), "to": after.wire(), "output": output[-4000:]})
        if changed:
            threading.Timer(RESTART_DELAY, restart).start()
        else:
            self.srv.updating.release()

    @route("POST", r"/api/node/workspace")
    def node_workspace(self, q: Query) -> None:
        """Body: {name, members: [directory names], old?: name being renamed}. Response {url}, or 400."""
        req = self.body_json()
        if not isinstance(members := req.get("members"), list):
            raise ValueError("expected {name, members, old?}")
        try:
            self.srv.node.set_workspace(str(req.get("name", "")), [str(m) for m in members],
                                        str(req["old"]) if req.get("old") else None)
        except (ValueError, KeyError) as e:
            return self._json({"error": str(e)}, 400)
        self._json({"url": workspace_url(str(req["name"]).strip())})

    @route("POST", r"/api/node/workspace/delete")
    def node_workspace_delete(self, q: Query) -> None:
        self.srv.node.delete_workspace(self.body_field("name"))
        self._json({"ok": True})

    @route("POST", r"/api/node/history/clear")
    def node_clear_history(self, q: Query) -> None:
        self.srv.node.clear_history()
        self._json({"ok": True})

    @route("GET", r"/api/tree")
    def tree(self, q: Query) -> None:
        """[[run path, state], ...] for every run under the root."""
        self._json(self.ex.tree())

    @route("GET", r"/api/runs")
    def runs(self, q: Query) -> None:
        ex, prefix = self.ex, q.get("path", "")
        body = ex.runs_body(prefix) if isinstance(ex, Explorer) else dumps(ex.runs(prefix).wire()).encode()
        self.send(body, "application/json", 200, {"Cache-Control": "no-store"}, compress=not self._loopback())

    @route("GET", r"/api/run")
    def run(self, q: Query) -> None:
        self._json(self.ex.run(q["path"]).wire())

    @route("GET", r"/api/rows")
    def rows(self, q: Query) -> None:
        self.send(self.ex.rows_json(q["path"], int(q.get("from", 0))).encode(), "application/json", compress=True)

    @route("POST", r"/api/dumps")
    def dumps(self, q: Query) -> None:
        """Body: {runs: [{path, and unless the mirror asking holds nothing of it: uid, mseq, compiled, rebuilt (the rows
        its levels hold and their rebuild)}]}, at most MAX_DUMPS. Response: each run's index rows the mirror lacks
        (`Explorer.dump`) in one body (`buckets.frame`), empty for a run this directory does not have."""
        ex, req = self.ex, self._parsed()
        runs = req.get("runs") if isinstance(req, dict) else None
        if not isinstance(ex, Explorer):
            raise KeyError("a directory")
        if not isinstance(runs, list) or not 0 < len(runs) <= MAX_DUMPS:
            raise ValueError(f"expected {{runs: [{{path, uid?, mseq?, compiled?, rebuilt?}}]}} of 1 to {MAX_DUMPS}")
        self.send(bk.frame(ex.dump_many([_held(r) for r in runs])), "application/octet-stream",
                  headers={"Cache-Control": "no-store"})

    @route("POST", r"/api/buckets")
    def post_buckets(self, q: Query) -> None:
        """Body: {blocks: [{key, level, index, and runs (a list of run ids) or scope and which (all, finished or
        running)}]}, at most MAX_ASKS. Response: those blocks of those runs as bucket arrays (`Explorer.buckets_body`) in
        one body (`buckets.frame`)."""
        req = self._parsed()
        blocks = req.get("blocks") if isinstance(req, dict) else None
        if not isinstance(blocks, list) or not 0 < len(blocks) <= MAX_ASKS:
            raise ValueError(f"expected {{blocks: [{{key, level, index, scope | runs, which}}]}} of 1 to {MAX_ASKS}")
        body = bk.frame(self.ex.buckets_bodies([_ask(b) for b in blocks]))
        self.send(body, "application/octet-stream", headers={"Cache-Control": "no-store"}, compress=not self._loopback())

    def _loopback(self) -> bool:
        """Whether the client is on this machine; one on the Unix socket is a trex reaching this one over ssh."""
        host = str(self.client_address[0]) if self.client_address else ""
        return host == "::1" or host.startswith("127.") or host.startswith("::ffff:127.")

    @route("GET", r"/api/stream")
    def stream(self, q: Query) -> None:
        """SSE for runs under `path`: running runs' rows beyond their levels, then live events and heartbeats. Its headers
        go out once it is subscribed, so a page that waits for it to open before listing runs misses no change."""
        stop = threading.Event()
        try:
            with contextlib.closing(self.ex.messages(q.get("path", ""), stop)) as msgs:
                first = next(msgs, b"")
                self._event_stream_headers()
                self.wfile.write(b"retry: 2000\n\n" + first)
                for msg in msgs:
                    self.wfile.write(msg)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            stop.set()

    def _event_stream_headers(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True


class Server(ThreadingHTTPServer):
    """A node's server, whose open connections end when it closes."""

    daemon_threads = True
    request_queue_size = 128
    node: Node
    restart: Callable[[], None] | None = None  # ends the daemon so that its service manager starts the updated one
    updating: threading.Lock  # held during an update, and from a successful one until the restart
    names: frozenset[str] = frozenset()  # host names besides IP addresses that requests may address it by

    def __init__(self, address: Any, handler: type[BaseHTTPRequestHandler]) -> None:
        self._open: set[socket.socket] = set()  # the connections being served
        super().__init__(address, handler)

    def process_request(self, request: Any, client_address: Any) -> None:
        self._open.add(request)
        super().process_request(request, client_address)

    def shutdown_request(self, request: Any) -> None:
        self._open.discard(request)
        super().shutdown_request(request)

    def server_close(self) -> None:
        super().server_close()
        for request in list(self._open):
            with contextlib.suppress(OSError):
                request.shutdown(socket.SHUT_RDWR)


class Server6(Server):
    address_family = socket.AF_INET6


class UnixServer(Server):
    """Server on a Unix socket (mode 0600), removed when closed."""

    address_family = socket.AF_UNIX
    allow_reuse_port = False

    def server_bind(self) -> None:
        old = os.umask(0o177)
        try:
            socketserver.TCPServer.server_bind(self)
        finally:
            os.umask(old)
        self.server_name, self.server_port = "localhost", 0

    def server_close(self) -> None:
        super().server_close()
        Path(str(self.server_address)).unlink(missing_ok=True)


def serve_unix(node: Node, path: Path) -> Server:
    """Unstarted server for `node` on the Unix socket `path`."""
    path.unlink(missing_ok=True)
    srv = UnixServer(cast(Any, str(path)), Handler)  # an AF_UNIX address is a path
    srv.node, srv.names, srv.updating = node, host_names(), threading.Lock()
    return srv


def _is_ip(name: str) -> bool:
    try:
        ipaddress.ip_address(name)
    except ValueError:
        return False
    return True


def host_names(hosts: Sequence[str] = (), allow: Sequence[str] = ()) -> frozenset[str]:
    """Names a server answers to besides IP addresses: localhost, this machine's names, named `hosts`, `allow`."""
    names = {"localhost", socket.gethostname(), socket.getfqdn(), *(h for h in hosts if not _is_ip(h)), *allow}
    return frozenset(n.lower() for n in names)


def serve(node: Node, host: str, port: int, allow: Sequence[str] = ()) -> Server:
    """Unstarted server for `node` on host:port (IPv6 if host has a colon); `allow` adds host names it answers to."""
    srv = (Server6 if ":" in host else Server)((host, port), Handler)
    srv.node, srv.names = node, host_names([host], allow)
    srv.updating = threading.Lock()
    return srv


def bind(node: Node, hosts: Sequence[str], port: int | None, allow: Sequence[str] = ()) -> list[Server]:
    """Unstarted servers for `node` on every host at `port`, or without one at the first free port from DEFAULT_PORT."""
    ports = [port] if port is not None else range(DEFAULT_PORT, DEFAULT_PORT + PORT_TRIES)
    for p in ports:
        servers: list[Server] = []
        try:
            for h in hosts:
                servers.append(serve(node, h, p, allow))
            return servers
        except OSError as e:
            for s in servers:
                s.server_close()
            if e.errno != errno.EADDRINUSE or p == ports[-1]:
                raise
    raise AssertionError("unreachable")


def urls(servers: Sequence[Server]) -> list[str]:
    """http://host:port/ of each server."""
    out: list[str] = []
    for s in servers:
        host, port = str(s.server_address[0]), s.server_address[1]
        out.append(f"http://{f'[{host}]' if ':' in host else host}:{port}/")
    return out


def _held(r: JSONValue) -> tuple[str, Have | None]:
    """A run of POST /api/dumps and what the mirror asking holds of it; ValueError unless it is one."""
    if not isinstance(r, dict) or not isinstance(path := r.get("path"), str):
        raise ValueError(f"not a run: {r!r}")
    return path, Have.read(r) if "uid" in r else None


def _ask(d: JSONValue) -> Ask:
    """A block request of POST /api/buckets; ValueError unless it is one."""
    if not isinstance(d, dict) or not isinstance(key := d.get("key"), str):
        raise ValueError("a block: {key, level, index, scope | runs, which}")
    runs, which = d.get("runs"), WHICH.get(str(d.get("which", "all")))
    if runs is not None and (not isinstance(runs, list) or not all(isinstance(r, str) for r in runs)):
        raise ValueError("runs: a list of run ids")
    if which is None:
        raise ValueError("which: all, finished or running")
    level = _whole(d.get("level"))
    if not bk.MIN_LEVEL <= level <= bk.MAX_LEVEL:
        raise ValueError(f"level {level} out of range")
    return Ask(key, level, _whole(d.get("index")), str(d.get("scope", "")), None if runs is None else as_str_list(runs), which)


def _whole(v: object) -> int:
    """`v` as an integer; ValueError unless it is a whole number."""
    if isinstance(v, bool) or not isinstance(v, (int, float)) or v != int(v):
        raise ValueError(f"expected an integer, got {v!r}")
    return int(v)
