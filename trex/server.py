"""HTTP server for the explorer UI (started by `trex serve`; see cli.py)."""

import contextlib
import errno
import gzip
import html
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
from typing import TYPE_CHECKING, Any, Final, cast
from urllib.parse import parse_qs, quote, unquote, urlsplit

from . import buckets as bk, remote, update
from .daemon import root_url, workspace_url
from .remote import Remote
from .workspace import Far, Workspace
from .index import Ask, Explorer, Which, dumps

if TYPE_CHECKING:
    from .daemon import Roots

type Query = dict[str, str]
type RouteFn = Callable[..., None]

STATIC: Final = Path(__file__).parent / "static"
DEFAULT_PORT: Final = 13898
PORT_TRIES: Final = 20  # ports tried from DEFAULT_PORT when none is given
HOP_HEADERS: Final = frozenset({"connection", "keep-alive", "transfer-encoding", "te", "trailer", "upgrade",
                                "proxy-authorization", "proxy-authenticate"})
ROOT_PREFIX: Final = re.compile(r"/([rw])/([^/]+)(/.*)?")  # a tracked directory's (r) or workspace's (w) URLs
PROTOCOL: Final = 6  # what the UI and this server say to each other; the UI states a mismatch (data.js PROTOCOL)
WHICH: Final[dict[str, Which]] = {"all": "all", "finished": "finished", "running": "running"}
MAX_ASKS: Final = 256  # blocks one POST /api/buckets may ask for
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
        """What this request is for: the served directory, or the daemon's tracked directory (/r/<name>/) or
        workspace (/w/<name>/)."""
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
        """Choose what the request is for and return the path within it, or None after redirecting a daemon page
        that names no tracked directory or workspace, or lacks its trailing slash. The daemon's own root is the
        view of every tracked directory."""
        self._ex = self.srv.explorer
        if self.srv.roots is None:
            return path
        pm = ROOT_PREFIX.fullmatch(path)
        if pm is None:
            self._ex = self.srv.roots.everything()
            return path
        kind, name, rest = pm[1], unquote(pm[2]), pm[3]
        try:
            entry = self.srv.roots.get(name) if kind == "r" else self.srv.roots.workspace(name)
        except KeyError:
            if rest in (None, "/"):
                return self.redirect("/", permanent=False)
            raise
        if rest is None:
            return self.redirect(path + "/" + (f"?{query}" if query else ""))
        if isinstance(entry, Remote):
            return self._proxy(entry, rest + (f"?{query}" if query else ""))
        self._ex = entry
        return rest

    def _proxy(self, remote: Remote, target: str) -> None:
        """Pass the request through to the remote directory's server, streaming its answer back."""
        if remote.state != "connected":
            return self._unavailable(remote, target)
        conn = remote.connection()
        try:
            headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP_HEADERS | {"host", "origin", "content-length"}}
            try:
                conn.request(self.command, target, body=self._body or None, headers={**headers, "Host": "localhost"})
                resp = conn.getresponse()
            except OSError:
                return self._unavailable(remote, target)
            self.send_response(resp.status)
            for k, v in resp.getheaders():
                if k.lower() not in HOP_HEADERS | {"server", "date"}:
                    self.send_header(k, v)
            if resp.getheader("Content-Length") is None:
                self.send_header("Connection", "close")
                self.close_connection = True
            self.end_headers()
            while chunk := resp.read1(1 << 16):
                self.wfile.write(chunk)
                self.wfile.flush()
        finally:
            conn.close()

    def _unavailable(self, remote: Remote, target: str) -> None:
        """503: a page that reloads itself for the directory's page, else JSON."""
        what = f"trex on {remote.addr.host} is {remote.state}" + (f": {remote.error}" if remote.error else "")
        if target.split("?")[0] != "/":
            return self._json({"error": what}, 503)
        body = (f'<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="2"><title>trex</title>'
                f'<p>{html.escape(remote.spec)}: {html.escape(what)}</p><pre>{html.escape(remote.error)}</pre>'
                f'<p><a href="/">back</a></p>').encode()
        self.send(body, "text/html; charset=utf-8", 503, {"Cache-Control": "no-store"})

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
            m, path = ex.resolve(run)
            if isinstance(m.src, Far):
                return self._proxy(m.src.remote, f"/m/{quote(path, safe='')}/{file}")
            ex, run = m.src, path
        self.send_file(ex.media_path(run, file), f'"{file}"', "public, max-age=31536000, immutable", extra, compress=html)

    @route("GET", r"/api/info")
    def info(self, q: Query) -> None:
        self._json({**self.ex.info(), "protocol": PROTOCOL})

    @route("GET", r"/api/daemon")
    def daemon(self, q: Query) -> None:
        """{daemon, roots: [{name, root, url, state, error}], workspaces: [{name, url, members}], history: [root],
        install, updates}: whether this is the daemon, its directories and workspaces, the remembered directories it
        does not serve, the trex it runs, and whether it can update."""
        roots = self.srv.roots
        if roots is None:
            return self._json({"daemon": False, "roots": [], "workspaces": [], "history": [], "install": None,
                               "updates": None})
        self._json({"daemon": True, "roots": roots.served(), "workspaces": roots.workspace_list(),
                    "history": roots.history(), "install": update.RUNNING, "updates": update.updates()})

    @property
    def daemon_roots(self) -> "Roots":
        if self.srv.roots is None:
            raise KeyError("daemon")
        return self.srv.roots

    def body_field(self, field: str) -> str:
        """String `field` of the JSON body."""
        req = json.loads(self.body())
        if not isinstance(req, dict) or not isinstance(req.get(field), str):
            raise ValueError(f"expected {{{field}}}")
        return req[field]

    @route("POST", r"/api/daemon/add")
    def daemon_add(self, q: Query) -> None:
        """Body: {path}: absolute, ~/…, or host:path for a remote directory (which this waits to start).
        Response: {name, url}, or 400 with {error} for a path it refuses or a remote that fails to start."""
        roots, path = self.daemon_roots, self.body_field("path").strip()
        try:
            if not remote.parse(path) and not Path(path).expanduser().is_absolute():
                raise ValueError(f"{path} is not an absolute path or host:path")
            name = roots.track(path)
        except ValueError as e:
            return self._json({"error": str(e)}, 400)
        self._json({"name": name, "url": root_url(name)})

    @route("POST", r"/api/daemon/remove")
    def daemon_remove(self, q: Query) -> None:
        """Body: {name}. Stops serving that directory."""
        self.daemon_roots.remove(self.body_field("name"))
        self._json({"ok": True})

    @route("POST", r"/api/daemon/update")
    def daemon_update(self, q: Query) -> None:
        """Install the newest trex from $TREX_SOURCE and, if that changed it, restart the daemon.
        Response: {updated, from, to, output}; 400 when updates are unavailable, 409 during another update,
        502 when the install fails."""
        restart = self.srv.restart
        if self.srv.roots is None or restart is None:
            raise KeyError("daemon")
        can = update.updates()
        if not can["available"] or can["source"] is None:
            return self._json({"error": f"updates are unavailable: {can['reason']}"}, 400)
        if not self.srv.updating.acquire(blocking=False):
            return self._json({"error": "an update is already running"}, 409)
        try:
            before = update.installed()
            output = update.install(can["source"])
            after = update.installed()
        except update.UpdateError as e:
            self.srv.updating.release()
            return self._json({"error": str(e)[-4000:]}, 502)
        changed = after != before
        self._json({"updated": changed, "from": before, "to": after, "output": output[-4000:]})
        if changed:
            threading.Timer(RESTART_DELAY, restart).start()
        else:
            self.srv.updating.release()

    @route("POST", r"/api/daemon/workspace")
    def daemon_workspace(self, q: Query) -> None:
        """Body: {name, members: [tracked directory names], old?: name being renamed}. Response {url}, or 400."""
        req = json.loads(self.body())
        if not isinstance(req, dict) or not isinstance(req.get("members"), list):
            raise ValueError("expected {name, members, old?}")
        try:
            self.daemon_roots.set_workspace(str(req.get("name", "")), [str(m) for m in req["members"]],
                                            str(req["old"]) if req.get("old") else None)
        except (ValueError, KeyError) as e:
            return self._json({"error": str(e)}, 400)
        self._json({"url": workspace_url(str(req["name"]).strip())})

    @route("POST", r"/api/daemon/workspace/delete")
    def daemon_workspace_delete(self, q: Query) -> None:
        self.daemon_roots.delete_workspace(self.body_field("name"))
        self._json({"ok": True})

    @route("POST", r"/api/daemon/history/clear")
    def daemon_clear_history(self, q: Query) -> None:
        self.daemon_roots.clear_history()
        self._json({"ok": True})

    @route("GET", r"/api/tree")
    def tree(self, q: Query) -> None:
        """[[run path, state], ...] for every run under the root."""
        self._json(self.ex.tree())

    @route("GET", r"/api/runs")
    def runs(self, q: Query) -> None:
        ex, prefix = self.ex, q.get("path", "")
        body = ex.runs_body(prefix) if isinstance(ex, Explorer) else dumps(ex.runs(prefix)).encode()
        self.send(body, "application/json", 200, {"Cache-Control": "no-store"}, compress=not self._loopback())

    @route("GET", r"/api/run")
    def run(self, q: Query) -> None:
        self._json(self.ex.run(q["path"]))

    @route("GET", r"/api/rows")
    def rows(self, q: Query) -> None:
        self.send(self.ex.rows_json(q["path"], int(q.get("from", 0))).encode(), "application/json", compress=True)

    @route("POST", r"/api/buckets")
    def post_buckets(self, q: Query) -> None:
        """Body: {blocks: [{key, level, index, and runs (a list of run ids) or scope and which (all, finished or
        running)}]}, at most MAX_ASKS. Response: those blocks of those runs as bucket arrays (`Explorer.buckets_body`) in
        one body (`buckets.frame`)."""
        req = json.loads(self.body())
        blocks = req.get("blocks") if isinstance(req, dict) else None
        if not isinstance(blocks, list) or not 0 < len(blocks) <= MAX_ASKS:
            raise ValueError(f"expected {{blocks: [{{key, level, index, scope | runs, which}}]}} of 1 to {MAX_ASKS}")
        body = bk.frame(self.ex.buckets_bodies([_ask(b) for b in blocks]))
        self.send(body, "application/octet-stream", headers={"Cache-Control": "no-store"}, compress=not self._loopback())

    def _loopback(self) -> bool:
        if not isinstance(self.client_address, tuple):
            return True
        host = str(self.client_address[0])
        return host == "::1" or host.startswith("127.") or host.startswith("::ffff:127.")

    @route("GET", r"/api/stream")
    def stream(self, q: Query) -> None:
        """SSE for runs under `path`: rows not yet in kept buckets, then live events and heartbeats."""
        stop = threading.Event()
        try:
            self._event_stream_headers()
            self.wfile.write(b"retry: 2000\n\n")
            with contextlib.closing(self.ex.messages(q.get("path", ""), stop)) as msgs:
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
    daemon_threads = True
    request_queue_size = 128
    explorer: Explorer | None = None
    roots: "Roots | None" = None
    restart: Callable[[], None] | None = None  # ends the daemon so that its service manager starts the updated one
    updating: threading.Lock  # held during an update, and from a successful one until the restart
    names: frozenset[str] = frozenset()  # host names besides IP addresses that requests may address it by


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


def serve_unix(explorer: Explorer, path: Path) -> Server:
    """Unstarted server for one directory on the Unix socket `path`."""
    path.unlink(missing_ok=True)
    srv = UnixServer(cast(Any, str(path)), Handler)  # an AF_UNIX address is a path
    srv.explorer, srv.names, srv.updating = explorer, host_names(), threading.Lock()
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


def serve(explorer: Explorer | None, host: str, port: int, roots: "Roots | None" = None, allow: Sequence[str] = ()) -> Server:
    """Unstarted server on host:port (IPv6 if host has a colon) for one directory, or the daemon's `roots`;
    `allow` adds host names it answers to."""
    srv = (Server6 if ":" in host else Server)((host, port), Handler)
    srv.explorer, srv.roots, srv.names = explorer, roots, host_names([host], allow)
    srv.updating = threading.Lock()
    return srv


def bind(explorer: Explorer | None, hosts: Sequence[str], port: int | None, roots: "Roots | None" = None,
         allow: Sequence[str] = ()) -> list[Server]:
    """Unstarted servers on every host at `port`, or without one at the first free port from DEFAULT_PORT."""
    ports = [port] if port is not None else range(DEFAULT_PORT, DEFAULT_PORT + PORT_TRIES)
    for p in ports:
        servers: list[Server] = []
        try:
            for h in hosts:
                servers.append(serve(explorer, h, p, roots, allow))
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


def _ask(b: object) -> Ask:
    """A block request of POST /api/buckets; ValueError unless it is one."""
    if not isinstance(b, dict) or not isinstance(key := cast(dict[str, object], b).get("key"), str):
        raise ValueError("a block: {key, level, index, scope | runs, which}")
    d = cast(dict[str, object], b)
    runs, which = d.get("runs"), WHICH.get(str(d.get("which", "all")))
    if runs is not None and (not isinstance(runs, list) or not all(isinstance(r, str) for r in cast(list[object], runs))):
        raise ValueError("runs: a list of run ids")
    if which is None:
        raise ValueError("which: all, finished or running")
    level = _whole(d.get("level"))
    if not bk.MIN_LEVEL <= level <= bk.MAX_LEVEL:
        raise ValueError(f"level {level} out of range")
    return Ask(key, level, _whole(d.get("index")), str(d.get("scope", "")), cast(list[str] | None, runs), which)


def _whole(v: object) -> int:
    """`v` as an integer; ValueError unless it is a whole number."""
    if isinstance(v, bool) or not isinstance(v, (int, float)) or v != int(v):
        raise ValueError(f"expected an integer, got {v!r}")
    return int(v)
