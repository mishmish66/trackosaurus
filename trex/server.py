"""HTTP server for the explorer UI (started by `trex serve`; see cli.py)."""

import errno
import gzip
import json
import os
import queue
import re
import socket
import sys
import threading
from collections.abc import Callable, Mapping, Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast
from urllib.parse import parse_qs, unquote, urlsplit

from . import update
from .daemon import root_url
from .index import Explorer, dumps, sse

if TYPE_CHECKING:
    from .daemon import Roots

type Query = dict[str, str]
type RouteFn = Callable[..., None]

STATIC: Final = Path(__file__).parent / "static"
DEFAULT_PORT: Final = 13898
PORT_TRIES: Final = 20  # ports tried from DEFAULT_PORT when none is given
ROOT_PREFIX: Final = re.compile(r"/r/([^/]+)(/.*)?")  # a daemon directory's URLs
HB_INTERVAL: Final = 10.0  # seconds of stream silence after which a heartbeat is sent
MAX_TILE_REQUESTS: Final = 4096
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

    _ex: Explorer | None = None
    _body: bytes = b""

    @property
    def srv(self) -> "Server":
        return cast(Server, self.server)

    @property
    def ex(self) -> Explorer:
        """The directory this request is for: the served one, or the daemon's named in the /r/<name>/ prefix."""
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

    def _scope(self, path: str, query: str) -> str | None:
        """Choose the directory the request is for and return the path within it, or None after redirecting
        a daemon page that names no served directory, or lacks its trailing slash."""
        self._ex = self.srv.explorer
        pm = ROOT_PREFIX.fullmatch(path) if self.srv.roots is not None else None
        if self.srv.roots is None or pm is None:
            return path
        try:
            self._ex = self.srv.roots.get(unquote(pm[1]))
        except KeyError:
            if pm[2] in (None, "/"):
                return self.redirect("/")
            raise
        if pm[2] is None:
            return self.redirect(path + "/" + (f"?{query}" if query else ""))
        return pm[2]

    def redirect(self, location: str) -> None:
        self.send_response(301)
        self.send_header("Location", location)
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
        self.send_file(self.ex.media_path(run, file), f'"{file}"', "public, max-age=31536000, immutable", extra,
                       compress=html)

    @route("GET", r"/api/info")
    def info(self, q: Query) -> None:
        self._json({"root": str(self.ex.root), "name": self.ex.root.name, "cache": self.ex.cache_dir.name})

    @route("GET", r"/api/daemon")
    def daemon(self, q: Query) -> None:
        """{daemon, roots: [{name, root, url}], history: [root], install, updates}: whether this is the daemon,
        its directories, the remembered ones it does not serve, the trex it runs, and whether it can update."""
        roots = self.srv.roots
        self._json({"daemon": roots is not None, "roots": roots.served() if roots else [],
                    "history": roots.history() if roots else [], "install": update.RUNNING if roots else None,
                    "updates": update.updates() if roots else None})

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
        """Body: {path}, absolute or ~/…. Response: {name, url}, or 400 with {error} for a path it refuses."""
        roots, path = self.daemon_roots, self.body_field("path")
        try:
            if not Path(path).expanduser().is_absolute():
                raise ValueError(f"{path} is not an absolute path")
            name = roots.add(resolve_root(path, force=False))
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
        self._json(self.ex.runs(q.get("path", "")))

    @route("GET", r"/api/run")
    def run(self, q: Query) -> None:
        self._json(self.ex.run(q["path"]))

    @route("GET", r"/api/rows")
    def rows(self, q: Query) -> None:
        self.send(self.ex.rows_json(q["path"], int(q.get("from", 0)))[0].encode(), "application/json", compress=True)

    @route("POST", r"/api/tiles")
    def post_tiles(self, q: Query) -> None:
        """Body: requests as for `Explorer.tiles`. Response per request: u32 count, (u32 length, tile)*."""
        want = json.loads(self.body())
        if not isinstance(want, list) or len(want) > MAX_TILE_REQUESTS:
            raise ValueError(f"expected a list of at most {MAX_TILE_REQUESTS} tile requests")
        out = []
        for blobs in self.ex.tiles(want):
            out.append(len(blobs).to_bytes(4, "little"))
            for b in blobs:
                out += [len(b).to_bytes(4, "little"), b]
        self.send(b"".join(out), "application/octet-stream", headers={"Cache-Control": "no-store"},
                  compress=not self._loopback())

    @route("POST", r"/api/tiles/bundle")
    def post_tile_bundle(self, q: Query) -> None:
        """Body: {key, kind, scope}. Response: u32 runs, then per run: u32 length, UTF-8 path padded to 4,
        u32 count, (u32 length, tile)*."""
        req = json.loads(self.body())
        if not isinstance(req, dict) or req.get("kind") not in ("top", "overview"):
            raise ValueError("expected {key, kind: top|overview, scope}")
        entries = self.ex.tile_bundle(str(req["key"]), req["kind"], str(req.get("scope", "")))
        out = [len(entries).to_bytes(4, "little")]
        for path, blobs in entries:
            p = path.encode()
            out += [len(p).to_bytes(4, "little"), p, b"\0" * (-len(p) % 4), len(blobs).to_bytes(4, "little")]
            for b in blobs:
                out += [len(b).to_bytes(4, "little"), b]
        self.send(b"".join(out), "application/octet-stream", headers={"Cache-Control": "no-store"},
                  compress=not self._loopback())

    def _loopback(self) -> bool:
        host = self.client_address[0]
        return host == "::1" or host.startswith("127.") or host.startswith("::ffff:127.")

    @route("GET", r"/api/stream")
    def stream(self, q: Query) -> None:
        """SSE for runs under `path`: rows not yet in top tiles, then live events and heartbeats."""
        prefix = q.get("path", "")
        sub = self.ex.hub.subscribe(prefix)
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Accel-Buffering", "no")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            self.wfile.write(b"retry: 2000\n\n" + b"".join(self.ex.backfill(prefix)))
            while not sub.dead:
                msgs = []
                try:
                    msgs.append(sub.q.get(timeout=HB_INTERVAL))
                    while len(msgs) < 2000:
                        msgs.append(sub.q.get_nowait())
                except queue.Empty:
                    if not msgs:
                        msgs.append(sse("hb", self.ex.live_seqs(prefix)))
                self.wfile.write(b"".join(msgs))
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            self.ex.hub.unsubscribe(sub)


class Server(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 128
    explorer: Explorer | None = None
    roots: "Roots | None" = None
    restart: Callable[[], None] | None = None  # ends the daemon so that systemd starts the updated one
    updating: threading.Lock  # held during an update, and from a successful one until the restart


class Server6(Server):
    address_family = socket.AF_INET6


def serve(explorer: Explorer | None, host: str, port: int, roots: "Roots | None" = None) -> Server:
    """Unstarted server on host:port (IPv6 if host has a colon) for one directory, or the daemon's `roots`."""
    srv = (Server6 if ":" in host else Server)((host, port), Handler)
    srv.explorer, srv.roots = explorer, roots
    srv.updating = threading.Lock()
    return srv


def bind(explorer: Explorer | None, hosts: Sequence[str], port: int | None, roots: "Roots | None" = None) -> list[Server]:
    """Unstarted servers on every host at `port`, or without one at the first free port from DEFAULT_PORT."""
    ports = [port] if port is not None else range(DEFAULT_PORT, DEFAULT_PORT + PORT_TRIES)
    for p in ports:
        servers: list[Server] = []
        try:
            for h in hosts:
                servers.append(serve(explorer, h, p, roots))
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


def resolve_root(root: str | os.PathLike[str], force: bool) -> Path:
    """`root` resolved; ValueError for a non-directory, or for / and $HOME (whose crawl would sweep a whole
    machine or home directory) unless `force`."""
    r = Path(root).expanduser().resolve()
    if not r.is_dir():
        raise ValueError(f"{root} is not a directory")
    if not force and r in (Path("/"), Path.home().resolve()):
        raise ValueError(f"refusing to crawl {r}; point trex at a runs directory (or pass --force)")
    return r


def check_root(root: str | os.PathLike[str], force: bool) -> Path:
    """`resolve_root`, exiting with its error."""
    try:
        return resolve_root(root, force)
    except ValueError as e:
        sys.exit(f"trex: {e}")


if __name__ == "__main__":
    from .cli import main

    main()
