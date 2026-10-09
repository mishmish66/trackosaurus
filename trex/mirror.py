"""Another trex's directory as an Explorer's origin. A `Pull` lists its upstream's runs (`/api/runs`) where a crawl
walks, and for each run whose version differs takes the index rows it lacks (`POST /api/dumps`, `Explorer.dump`) where
a crawl scans: the record, new media, and the blocks of levels that changed since the rows it holds. Media files are
copied before the rows naming them are written. It follows the upstream's stream, which tells it what to take, and
keeps each running run's rows beyond its levels as the stream brings them (its tail), passing them on to its own
stream: a running run stays live by its rows, and its levels are taken anew only once its tail has grown (TAIL_ROWS),
something its rows do not carry changed, or its tail does not continue them. So its Explorer answers everything
itself, and, as far as the stream brought it, while the upstream is unreachable. `Upstream` is the upstream's API for
the directory, over http or a Unix socket.
"""

import contextlib
import gzip
import http.client
import json
import math
import os
import queue
import socket
import threading
import time
import zlib
from collections.abc import Callable, Generator, Sequence
from concurrent.futures import ThreadPoolExecutor
from itertools import batched
from pathlib import Path
from typing import Any, Final, Protocol
from urllib.parse import quote, urlsplit

from . import buckets as bk
from .index import HEARTBEAT, Dump, Explorer, Have, Rows, RunRecord, RunsView, Update, dumps, sse_text

TIMEOUT: Final = 60.0  # seconds a request to the upstream may take
RETRY: Final = 2.0  # seconds before reaching the upstream again after it failed
RUNNING_EVERY = 10.0  # seconds between dumps of a running run, at least
TAIL_ROWS: Final = 64  # rows a running run's tail holds past its levels before those are taken anew
LIST_EVERY: Final = 600.0  # seconds between reads of the whole run list, besides the stream's events
DUMPS_AT_ONCE: Final = 16  # runs one request dumps
FETCH_THREADS: Final = 8  # requests to the upstream at once for media files and tails
IDLE: Final = 4  # connections to the upstream kept open between requests

type Connect = Callable[[float], http.client.HTTPConnection]


class Closeable(Protocol):
    def close(self) -> None: ...


class Unreachable(Exception):
    """The upstream did not answer."""


class Upstream:
    """One directory of a trex server: its API under the path `base`, through `connect` (a connection, given a
    timeout); `label` names it in messages. Connections are kept open between requests."""

    def __init__(self, connect: Connect, label: str, base: str = "") -> None:
        self.connect, self.label, self.base = connect, label, base.rstrip("/")
        self._idle: queue.LifoQueue[http.client.HTTPConnection] = queue.LifoQueue(IDLE)
        self._streaming: http.client.HTTPConnection | None = None
        self._retired = threading.Event()  # set once nothing reads from it any more

    @classmethod
    def at(cls, url: str, base: str = "") -> "Upstream":
        """`base` of the server at `url` (http://host[:port])."""
        u = urlsplit(url)
        if u.scheme != "http" or not u.hostname:
            raise ValueError(f"not an http://host[:port] address: {url}")
        host, port = u.hostname, u.port or 80
        return cls(lambda timeout: http.client.HTTPConnection(host, port, timeout=timeout), url, base)

    def request(self, method: str, target: str, body: bytes | None = None) -> bytes:
        """The body answering `target`; KeyError for a 404, Unreachable for no answer or another error. A request on a
        kept connection that fails is sent once more on a new one."""
        try:
            conn = self._idle.get_nowait()
        except queue.Empty:
            status, data = self._exchange(self.connect(TIMEOUT), method, target, body)
        else:
            try:
                status, data = self._exchange(conn, method, target, body)
            except Unreachable:
                status, data = self._exchange(self.connect(TIMEOUT), method, target, body)
        if status == 404:
            raise KeyError(target)
        if status >= 400:
            raise Unreachable(f"{self.label}: {status} {data[:200]!r}")
        return data

    def answer(self, method: str, target: str, body: bytes | None, timeout: float) -> tuple[int, bytes]:
        """(status, body) answering `target`, on a new connection given `timeout` seconds; Unreachable for no answer."""
        return self._exchange(self.connect(timeout), method, target, body)

    def _exchange(self, conn: http.client.HTTPConnection, method: str, target: str, body: bytes | None) -> tuple[int, bytes]:
        """(status, body) of one request on `conn`, which is kept for the next when the server leaves it open."""
        headers = {"Accept-Encoding": "gzip", **({"Content-Type": "application/json"} if body is not None else {})}
        try:
            conn.request(method, self.base + target, body=body, headers=headers)
            r = conn.getresponse()
            data = r.read()
            if r.getheader("Content-Encoding") == "gzip":
                data = gzip.decompress(data)
        except (OSError, EOFError, zlib.error, http.client.HTTPException) as e:
            conn.close()
            raise Unreachable(f"{self.label}: {e!r}") from e
        if r.will_close or self._retired.is_set():
            conn.close()
        else:
            try:
                self._idle.put_nowait(conn)
            except queue.Full:
                conn.close()
        return r.status, data

    def stream(self, stop: Callable[[], bool], connected: Callable[[bool], None]) -> Generator[bytes, None, None]:
        """The messages of the upstream's stream of the whole directory, one at a time, reopened when it drops, until
        `stop()` or `retire`; `connected` is told when it opens and when it drops."""
        while not stop() and not self._retired.is_set():
            conn = self.connect(3 * HEARTBEAT)
            self._streaming = conn
            try:
                conn.request("GET", self.base + "/api/stream?path=")
                r, msg = conn.getresponse(), b""
                if r.status != 200:
                    raise Unreachable(f"{self.label}: {r.status}")
                connected(True)
                while not stop() and (line := r.readline()):
                    msg += line
                    if line == b"\n":
                        yield msg
                        msg = b""
            except (OSError, ValueError, http.client.HTTPException, Unreachable):
                pass
            finally:
                self._streaming = None
                conn.close()
            connected(False)
            if not stop():
                self._retired.wait(RETRY)

    def retire(self) -> None:
        """End the stream for good and close the kept connections."""
        self._retired.set()
        self.interrupt()
        with contextlib.suppress(queue.Empty):
            while True:
                self._idle.get_nowait().close()

    def interrupt(self) -> None:
        """End the stream's open connection."""
        conn = self._streaming
        if conn is not None and conn.sock is not None:
            with contextlib.suppress(OSError):
                conn.sock.shutdown(socket.SHUT_RDWR)


class Pull:
    """The directory `id` of `upstream` as an Explorer's origin; `session`, what reaches the upstream when something
    does, closes with it."""

    def __init__(self, upstream: Upstream, id: str, session: Closeable | None = None) -> None:
        self.upstream, self.id, self.session = upstream, id, session
        self.key = f"/trex-mirror/{id}"
        self.connected = False
        self.error = ""  # why the last pass failed, if it did
        self._lock = threading.Lock()
        self._media = Path()  # where media files are copied to: beside the index
        self._dirty: set[str] = set()  # runs to dump
        self._heard: dict[str, float] = {}  # run -> monotonic time the stream last told of it
        self._dumped_at: dict[str, float] = {}  # run -> monotonic time of its last dump
        self._urgent: set[str] = set()  # changed runs whose change rows do not carry: state, keys, uid or media
        self._tails: dict[str, Rows] = {}  # running run -> its rows beyond its levels
        self._wake = threading.Event()  # set when there is something to take
        self._relist = threading.Event()  # set when the whole run list is due

    def attach(self, ex: Explorer) -> None:
        self._media = ex.cache_dir / "media"
        self._media.mkdir(exist_ok=True)

    def info(self) -> dict[str, object]:
        return {"root": self.id, "name": self.id.rstrip("/").rsplit("/", 1)[-1], "upstream": self.upstream.label}

    # ---- taking what the upstream holds ----

    def run(self, ex: Explorer) -> None:
        """Follow the upstream's stream, on a thread, and until `ex` stops: read the run list at first, after the stream
        reconnects and every LIST_EVERY seconds, and dump the runs that changed whenever the stream tells of one."""
        threading.Thread(target=self._follow, args=(ex,), name=f"trex-follow-{self.id}", daemon=True).start()
        listed = -math.inf
        while not ex.stopped:
            self._wake.clear()
            try:
                if self._relist.is_set() or time.monotonic() - listed >= LIST_EVERY:
                    self._relist.clear()
                    self.rewalk(ex)
                    listed = time.monotonic()
                self.poll(ex)
                self.error = ""
            except Exception as e:
                listed = -math.inf
                if str(e) != self.error:
                    print(f"[trex] {self.id}: {e}", flush=True)
                self.error = str(e)
            ex.ready.set()
            self._wake.wait(RETRY if self.error else RUNNING_EVERY)

    def sync(self, ex: Explorer) -> list[str]:
        self.rewalk(ex)
        return self.poll(ex)

    def stop(self) -> None:
        self._wake.set()
        self.upstream.interrupt()

    def close(self) -> None:
        self.upstream.retire()
        if self.session is not None:
            self.session.close()

    def retarget(self, upstream: Upstream) -> None:
        """Pull from `upstream` from now on."""
        old, self.upstream = self.upstream, upstream
        old.retire()
        self._relist.set()
        self._wake.set()

    def rewalk(self, ex: Explorer) -> None:
        """Read the upstream's run list and folder notes: runs it no longer has leave the index, and those whose version
        differs are due a dump."""
        asked = time.monotonic()
        view = RunsView.read(json.loads(self.upstream.request("GET", "/api/runs?path=")))
        listed = {m.id: m.ver for m in view.runs}
        with ex.lock:
            gone = [p for p in ex.records if p not in listed and self._heard.get(p, -math.inf) < asked]
            differ = {p for p, ver in listed.items() if (st := ex.records.get(p)) is None or st.ver != ver}
        with self._lock:
            self._dirty |= differ
        for p in gone:
            self._gone(ex, p)
        ex.keep_folders(view.folders)

    def poll(self, ex: Explorer) -> list[str]:
        """Dump the changed runs due one (`_due`), copy their new media files and apply them; then fetch the tails that
        running runs lack. The runs dumped."""
        with self._lock:
            left, self._dirty = self._dirty, set()
        now = time.monotonic()
        todo = sorted(p for p in left if self._due(ex, p, now))
        try:
            for paths in batched(todo, DUMPS_AT_ONCE):
                updates = self._dumped(ex, paths)
                media = [m for r in updates for m in r.media]
                with ThreadPoolExecutor(FETCH_THREADS) as pool:
                    list(pool.map(self._media_file, [m.run for m in media], [m.file for m in media]))
                ex.apply(updates)
                left -= set(paths)
                self._trim(ex, paths)
        finally:
            with self._lock:
                self._dirty |= left
        self._fill(ex)
        return todo

    def _due(self, ex: Explorer, path: str, now: float) -> bool:
        """Whether changed run `path` is due a dump: at once unless it runs. A running run's rows reach the stream live,
        so its levels are taken at most every RUNNING_EVERY seconds, and then once its tail holds TAIL_ROWS rows past
        them, once something its rows do not carry changed (`_urgent`), or while it has no tail to follow it by or its
        tail begins past them."""
        st = ex.records.get(path)
        if st is None or st.state != "running":
            return True
        if now - self._dumped_at.get(path, -math.inf) < RUNNING_EVERY:
            return False
        with self._lock:
            tail = self._tails.get(path)
            return (path in self._urgent or tail is None or tail.seq0 > st.compiled
                    or tail.end - st.compiled >= TAIL_ROWS)

    def _dumped(self, ex: Explorer, paths: Sequence[str]) -> list[Update]:
        """What the upstream's dumps of `paths` change, as updates; runs it no longer has leave the index."""
        held = [{"path": p, **(Have(st.uid, st.mseq, st.compiled, st.rebuilt).wire() if (st := ex.records.get(p)) else {})}
                for p in paths]
        bodies = bk.unframe(self.upstream.request("POST", "/api/dumps", dumps({"runs": held}).encode()))
        now = time.monotonic()
        out: list[Update] = []
        for p, body in zip(paths, bodies, strict=True):
            self._dumped_at[p] = now
            with self._lock:
                self._urgent.discard(p)
            if body:
                out.append(_update(p, Dump.decode(p, body), ex.records.get(p)))
            else:
                self._gone(ex, p)
        return out

    def _gone(self, ex: Explorer, path: str) -> None:
        with self._lock:
            self._tails.pop(path, None)
            self._dirty.discard(path)
            self._urgent.discard(path)
        if path in ex.records:
            ex.drop(path)

    # ---- the stream ----

    def _follow(self, ex: Explorer) -> None:
        """The upstream's stream, the current upstream's after a `retarget`, until `ex` stops."""
        while not ex.stopped:
            for msg in self.upstream.stream(lambda: ex.stopped, self._connected):
                head, _, rest = msg.partition(b"\n")
                if head.startswith(b"event: ") and rest.startswith(b"data: "):
                    self._event(ex, head[7:].decode(), json.loads(rest[6:]), msg)

    def _connected(self, up: bool) -> None:
        if up and not self.connected:
            self._relist.set()
            self._wake.set()
        self.connected = up

    def _event(self, ex: Explorer, kind: str, ev: Any, msg: bytes) -> None:
        """Take in an upstream event: a run with a new version or new media is due a dump, a deleted one leaves the
        index, folder notes are kept, and rows that follow their run's tail join it and go on to `ex`'s stream."""
        match kind:
            case "rows":
                self._append(ex, Rows.read(ev), msg)
            case "run":
                st = ex.records.get(ev["id"])
                self._told(ex, ev["id"], ev["ver"], st is None or (ev["state"], ev["keys"], ev["uid"]) != (st.state, st.keys, st.uid))
            case "media":
                self._told(ex, ev[0], None, True)
            case "delete":
                self._gone(ex, ev["run"])
            case "folder":
                notes = {p: info for p, info in ex.folders.items() if p != ev["path"]}
                ex.keep_folders(notes if ev["info"] is None else {**notes, ev["path"]: ev["info"]})
            case _:
                pass

    def _told(self, ex: Explorer, path: str, ver: int | None, urgent: bool) -> None:
        """The stream told of run `path` at version `ver` (None: of something a version does not count): unless the
        index holds that version, the run changed, `urgent` when its rows do not carry the change (`_due`)."""
        st = ex.records.get(path)
        with self._lock:
            self._heard[path] = time.monotonic()
            if ver is None or st is None or st.ver != ver:
                self._dirty.add(path)
                if urgent:
                    self._urgent.add(path)
        self._wake.set()

    def _append(self, ex: Explorer, rows: Rows, msg: bytes) -> None:
        """Add a `rows` event's rows to their run's tail and pass the event (`msg`) on to `ex`'s stream, when they follow
        the tail; rows that leave a gap end the tail, to be fetched anew. Both under the lock, as everything that grows
        a tail, so that no run event, which counts the rows of the tail (`live`), counts rows not yet passed on."""
        with self._lock:
            tail = self._tails.get(rows.run)
            if tail is None:
                return
            if rows.seq0 > tail.end:
                del self._tails[rows.run]
                self._wake.set()
                return
            self._tails[rows.run] = Rows(tail.run, tail.seq0, [*tail.rows, *rows.since(tail.end).rows])
            ex.hub.publish_msg(rows.run, msg)

    def _trim(self, ex: Explorer, paths: Sequence[str]) -> None:
        """After `paths` were applied: a run's tail holds only rows beyond its levels, and only while it runs."""
        with self._lock:
            for p in paths:
                st, tail = ex.records.get(p), self._tails.get(p)
                if st is None or st.state != "running":
                    self._tails.pop(p, None)
                elif tail is not None:
                    self._tails[p] = tail.since(st.compiled)

    def _fill(self, ex: Explorer) -> None:
        """While the stream is up, fetch the rows beyond its levels of each running run that has no tail, and tell
        `ex`'s stream of them; a tail starting beyond the run's levels makes the run due a dump."""
        with ex.lock, self._lock:
            lacking = [(p, st.compiled) for p, st in ex.records.items() if st.state == "running" and p not in self._tails]
        if not self.connected or not lacking:
            return

        def fetch(run: tuple[str, int]) -> None:
            path, start = run
            try:
                tail = Rows.read(json.loads(self.upstream.request("GET", f"/api/rows?path={quote(path)}&from={start}")))
            except (KeyError, Unreachable):
                return
            with self._lock:
                self._tails[path] = tail
                if tail.seq0 > start:
                    self._dirty.add(path)
                    self._dumped_at.pop(path, None)
                    self._wake.set()
                if tail.rows:
                    ex.hub.publish_msg(path, sse_text("rows", tail.text()))

        with ThreadPoolExecutor(FETCH_THREADS) as pool:
            list(pool.map(fetch, lacking))

    # ---- what the index does not hold ----

    def rows_json(self, path: str, start: int) -> str:
        """The `rows` event JSON of the run's tail from `start`."""
        with self._lock:
            return (self._tails.get(path) or Rows(path, start, [])).since(start).text()

    def live(self, path: str, rec: RunRecord) -> tuple[int, int]:
        """(rows, media) held of a running run: to the end of its tail, or of its levels without one."""
        with self._lock:
            tail = self._tails.get(path)
        return (tail.end if tail else rec.compiled), rec.mseq

    def media_file(self, path: str, file: str) -> Path:
        """The copy of run `path`'s media file `file`, copied first when missing; KeyError for one it lacks."""
        name = file.removeprefix("media/")
        if "/" in name or name.startswith(".") or not name:
            raise KeyError(file)
        f = self._media_file(path, file)
        if f is None:
            raise KeyError(file)
        return f

    def _media_file(self, path: str, file: str) -> Path | None:
        """The copy of media `file` (named for its contents, so one copy serves every run), copied from the upstream
        when missing; None when it cannot be."""
        f = self._media / Path(file).name
        if f.exists():
            return f
        try:
            data = self.upstream.request("GET", f"/m/{quote(path, safe='')}/{file}")
        except (KeyError, Unreachable):
            return None
        tmp = f.with_name(f".{f.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_bytes(data)
        tmp.replace(f)
        return f


def _update(path: str, dump: Dump, st: RunRecord | None) -> Update:
    """A dump of `path`, given the record held of it, as the update it stands for."""
    rec = dump.record
    reset = st is not None and st.uid != rec.uid
    changed = dump.replace or bool(dump.blocks) or st is None or st.compiled != rec.compiled
    return Update(path=path, sig=rec.sig, uid=rec.uid, reset=reset, fresh=st is None or reset, seq=rec.seq, mseq=rec.mseq,
                  media=dump.media, state=rec.state, heartbeat=rec.heartbeat, public=rec.public, keys=rec.keys,
                  summary=rec.summary, compiled=rec.compiled if changed else None, rebuilt=rec.rebuilt,
                  metrics=dump.metrics, blocks=dump.blocks, replace=dump.replace, ver=rec.ver)
