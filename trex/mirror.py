"""Mirrors: a runs directory another trex holds, kept in an index of one's own and answered from it.

A `Mirror` is one directed link: it pulls a directory from its upstream (a trex holding it) into an Explorer's tables.
The upstream's run list stands in for a crawl, and for each run that changed, the index rows the mirror lacks
(`Explorer.dump`) stand in for a scan; the run's media files are copied after. It answers as an Explorer does, from
those tables, so it can be browsed while the upstream is unreachable. While the upstream is reachable, the live stream,
running runs' rows and blocks, and the blocks of runs whose compiled levels lag their rows come from the upstream.
"""

import contextlib
import gzip
import http.client
import json
import math
import os
import socket
import threading
import time
import zlib
from collections.abc import Callable, Generator, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Final, Protocol, Self
from urllib.parse import quote, urlsplit

from . import buckets as bk
from .format import JSONValue
from .index import (HEARTBEAT, Ask, Compiled, Explorer, KeptRecord, MediaRecord, RunMeta, ScanResult, dumps, in_scope,
                    sse_text)

TIMEOUT: Final = 60.0  # seconds a request to the upstream may take
RETRY: Final = 2.0  # seconds before reaching the upstream again after it failed
RUNNING_EVERY: Final = 60.0  # seconds between dumps of a running run
LIST_EVERY: Final = 600.0  # seconds between reads of the whole run list, besides the stream's events
DUMPS_AT_ONCE: Final = 16  # runs one request dumps
FETCH_THREADS: Final = 8  # requests to the upstream at once for media files and rows
SYNCED: Final = ("uid", "seq", "mseq", "kept_seq", "pyramid_seq", "state", "name", "tags", "config", "info", "created")

type Connect = Callable[[float], http.client.HTTPConnection]


class Closeable(Protocol):
    def close(self) -> None: ...


class Unreachable(Exception):
    """The upstream did not answer."""


class Upstream:
    """One directory of a trex server: its API under the path `base`, through `connect` (a connection, given a
    timeout); `label` names it in messages."""

    def __init__(self, connect: Connect, label: str, base: str = "") -> None:
        self.connect, self.label, self.base = connect, label, base.rstrip("/")
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
        """The body answering `target`; KeyError for a 404, Unreachable for no answer or another error."""
        conn = self.connect(TIMEOUT)
        headers = {"Accept-Encoding": "gzip", **({"Content-Type": "application/json"} if body is not None else {})}
        try:
            conn.request(method, self.base + target, body=body, headers=headers)
            r = conn.getresponse()
            data = r.read()
            if r.getheader("Content-Encoding") == "gzip":
                data = gzip.decompress(data)
        except (OSError, EOFError, zlib.error, http.client.HTTPException) as e:
            raise Unreachable(f"{self.label}: {e!r}") from e
        finally:
            conn.close()
        if r.status == 404:
            raise KeyError(target)
        if r.status >= 400:
            raise Unreachable(f"{self.label}: {r.status} {data[:200]!r}")
        return data

    def stream(self, stop: threading.Event, connected: Callable[[bool], None]) -> Generator[bytes, None, None]:
        """The messages of the upstream's stream of the whole directory, one at a time, reopened when it drops, until
        `stop` or `retire`; `connected` is told when it opens and when it drops."""
        while not stop.is_set() and not self._retired.is_set():
            conn = self.connect(3 * HEARTBEAT)
            self._streaming = conn
            try:
                conn.request("GET", self.base + "/api/stream?path=")
                r, msg = conn.getresponse(), b""
                if r.status != 200:
                    raise Unreachable(f"{self.label}: {r.status}")
                connected(True)
                while not stop.is_set() and (line := r.readline()):
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
            if not self._retired.is_set():
                stop.wait(RETRY)

    def retire(self) -> None:
        """End the stream for good."""
        self._retired.set()
        self.interrupt()

    def interrupt(self) -> None:
        """End the stream's open connection."""
        conn = self._streaming
        if conn is not None and conn.sock is not None:
            with contextlib.suppress(OSError):
                conn.sock.shutdown(socket.SHUT_RDWR)


class Mirror(Explorer):
    """Directory `name` of `upstream`, mirrored in an index under `cache_root`; `session`, what reaches the upstream
    when something does, closes with it."""

    def __init__(self, upstream: Upstream, cache_root: str | os.PathLike[str], name: str,
                 session: Closeable | None = None) -> None:
        super().__init__(Path("/trex-mirror") / name, cache_root, workers=1)
        self.upstream, self.name, self.session = upstream, name, session
        self.media_dir = self.cache_dir / "media"
        self.media_dir.mkdir(exist_ok=True)
        self.connected = False
        self.error = ""  # why the last sync failed, if it did
        self._listed: dict[str, RunMeta] = {}  # the upstream's runs, as it last told of them
        self._heard: dict[str, float] = {}  # run -> monotonic time the stream last told of it
        self._dirty: set[str] = set()  # runs to compare with the upstream's
        self._dumped_at: dict[str, float] = {}  # run -> monotonic time of its last dump
        self._wake = threading.Event()  # set when there is something to sync
        self._relist = threading.Event()  # set when the whole run list is due
        self.folders = {p: (0, json.loads(info)) for p, info in self._writer.execute("SELECT path, info FROM folders")}

    def info(self) -> dict[str, object]:
        return {"root": self.name, "name": self.name.rstrip("/").rsplit("/", 1)[-1], "cache": self.cache_dir.name,
                "upstream": self.upstream.label}

    # ---- syncing ----

    def start(self) -> Self:
        """Follow the upstream's stream and sync in background threads until `close`."""
        threading.Thread(target=self._follow, name=f"trex-follow-{self.name}", daemon=True).start()
        return super().start()

    def stop(self) -> None:
        super().stop()
        self._wake.set()
        self.upstream.interrupt()

    def close(self) -> None:
        super().close()
        if self.session is not None:
            self.session.close()

    def retarget(self, upstream: Upstream) -> None:
        """Pull from `upstream` from now on."""
        old, self.upstream = self.upstream, upstream
        old.retire()
        self._relist.set()
        self._wake.set()

    def poll_forever(self) -> None:
        """Until `stop`: the run list at first, after the stream reconnects and every LIST_EVERY seconds, and the runs
        that changed whenever the stream tells of one; sets `ready` after the first pass."""
        listed = -math.inf
        while not self._stop.is_set():
            self._wake.clear()
            try:
                if self._relist.is_set() or time.monotonic() - listed >= LIST_EVERY:
                    self._relist.clear()
                    self.rewalk()
                    listed = time.monotonic()
                self.poll()
                self.error = ""
            except Exception as e:
                listed = -math.inf
                if str(e) != self.error:
                    print(f"[trex] mirror {self.name}: {e}", flush=True)
                self.error = str(e)
            self.ready.set()
            self._wake.wait(RETRY if self.error else RUNNING_EVERY)

    def rewalk(self) -> None:
        """The upstream's run list and folder notes; runs it no longer has are dropped."""
        asked = time.monotonic()
        body = json.loads(self.upstream.request("GET", "/api/runs?path="))
        listed: dict[str, RunMeta] = {m["id"]: m for m in body["runs"]}
        with self.lock:
            for p in listed:
                if self._heard.get(p, -math.inf) > asked and p in self._listed:
                    listed[p] = self._listed[p]  # the stream told of it after the list was read
            self._listed = listed
            self._dirty = set(listed)
            gone = [p for p in self.records if p not in listed]
        for p in gone:
            self.drop(p, publish=True)
        self._keep_folders(body["folders"])

    def poll(self) -> list[str]:
        """Dump and apply the runs that differ from the upstream's (a running one at most every RUNNING_EVERY seconds)
        and copy their media files; their paths."""
        with self.lock:
            look, self._dirty = self._dirty, set()
            pending = [(p, self._listed[p]) for p in look if p in self._listed]
        todo: list[str] = []
        later: set[str] = set()
        for p, m in pending:
            due = self._due(p, m)
            if due:
                todo.append(p)
            elif due is not None:
                later.add(p)
        with self.lock:
            self._dirty |= later
        for i in range(0, len(todo), DUMPS_AT_ONCE):
            results = self._dumped(todo[i:i + DUMPS_AT_ONCE])
            self.apply(results)
            with ThreadPoolExecutor(FETCH_THREADS) as pool:
                list(pool.map(lambda m: self._media_file(m.run, m.file), [m for r in results for m in r["media"]]))
        return todo

    def _due(self, path: str, m: RunMeta) -> bool | None:
        """None when the mirror holds the run as listed, else whether to dump it now."""
        if path in self.records:
            mine = self.run_meta(path)
            if all(mine.get(k) == m.get(k) for k in SYNCED):
                return None
        recent = time.monotonic() - self._dumped_at.get(path, -math.inf) < RUNNING_EVERY
        return not (m["state"] == "running" and recent and path in self.records)

    def _dumped(self, paths: Sequence[str]) -> list[ScanResult]:
        """What the upstream's dumps of `paths` change, as scan results; runs it no longer has are dropped."""
        held: list[dict[str, object]] = []
        for p in paths:
            st = self.records.get(p)
            held.append({"path": p} if st is None else {"path": p, "uid": st["uid"], "mseq": st["mseq"],
                                                         "kept": st["kept_seq"], "pyramid": st["pyramid_seq"]})
        bodies = bk.unframe(self.upstream.request("POST", "/api/dumps", dumps({"runs": held}).encode()))
        now = time.monotonic()
        out: list[ScanResult] = []
        for p, body in zip(paths, bodies, strict=True):
            self._dumped_at[p] = now
            if body:
                out.append(self._result(p, body))
            else:
                with self.lock:
                    self._listed.pop(p, None)
                self.drop(p, publish=True)
        return out

    def _result(self, path: str, body: bytes) -> ScanResult:
        """A dump of `path` as the scan result it stands for."""
        parts = bk.unframe(body)
        head, blobs = json.loads(parts[0]), iter(parts[1:])
        rec, st = head["record"], self.records.get(path)
        kept = None if head["kept"] is None else [KeptRecord(k, level, zlib.decompress(next(blobs))) for k, level, _ in head["kept"]]
        pyramid = None
        if head["pyramid"] is not None:
            blocks: dict[str, list[tuple[int, int, bytes]]] = {}
            for key, level, block in head["pyramid"]:
                blocks.setdefault(key, []).append((level, block, next(blobs)))
            pyramid = [Compiled(key, fine, blocks.get(key, [])) for key, fine, _ in head["compiled"]]
        reset = st is not None and st["uid"] != rec["uid"]
        return {"path": path, "sig": rec["sig"], "uid": rec["uid"], "reset": reset, "fresh": st is None or reset,
                "seq": rec["seq"], "mseq": rec["mseq"], "media": [MediaRecord(path, *m) for m in head["media"]],
                "state": rec["state"], "heartbeat": rec["heartbeat"], "public": rec["public"], "keys": rec["keys"],
                "summary": rec["summary"], "kept": kept, "kept_seq": rec["kept_seq"], "pyramid": pyramid,
                "pyramid_seq": rec["pyramid_seq"], "rows": None}

    def _follow(self) -> None:
        """The upstream's stream, the current upstream's after a `retarget`: each event goes to the mirror's own
        stream, and a run it tells of is synced."""
        while not self._stop.is_set():
            for msg in self.upstream.stream(self._stop, self._connected):
                head, _, rest = msg.partition(b"\n")
                if head.startswith(b"event: ") and rest.startswith(b"data: "):
                    self._event(head[7:].decode(), rest[6:].rstrip(b"\n"), msg)

    def _event(self, kind: str, data: bytes, msg: bytes) -> None:
        """Take in an upstream event and pass it on to the mirror's own stream, whose heartbeats are its own."""
        if kind == "hb":
            return self._heartbeat(json.loads(data))
        if kind == "rows":
            path = json.loads(data[:data.index(b',"seq0"')] + b"}")["run"]
        elif kind == "media":
            path = json.loads(data)[0]
        elif kind == "run":
            meta = json.loads(data)
            path = meta["id"]
            self._told(path, meta)
        elif kind == "delete":
            path = json.loads(data)["run"]
            self._told(path, None)
        elif kind == "folder":
            ev = json.loads(data)
            path = ev["path"]
            self._folder(ev)
        else:
            return
        self.hub.publish_msg(path, msg)

    def _told(self, path: str, meta: RunMeta | None) -> None:
        """The stream told of run `path`: its new metadata, or None when it was deleted."""
        with self.lock:
            if meta is None:
                self._listed.pop(path, None)
            else:
                self._listed[path] = meta
                self._dirty.add(path)
            self._heard[path] = time.monotonic()
        if meta is None:
            self.drop(path)
        self._wake.set()

    def _heartbeat(self, seqs: dict[str, list[int]]) -> None:
        """The upstream's rows and media of its running runs."""
        with self.lock:
            for p, (seq, mseq) in seqs.items():
                if p in self._listed and (self._listed[p]["seq"], self._listed[p]["mseq"]) != (seq, mseq):
                    self._listed[p] = {**self._listed[p], "seq": seq, "mseq": mseq}

    def _folder(self, ev: dict[str, JSONValue]) -> None:
        notes = {p: info for p, (_, info) in self.folders.items() if p != ev["path"]}
        if isinstance(info := ev.get("info"), dict):
            notes[str(ev["path"])] = info
        self._keep_folders(notes)

    def _keep_folders(self, notes: dict[str, dict[str, JSONValue]]) -> None:
        """Hold `notes` (folder path -> its notes) as the folders' notes, in memory and in the index."""
        if {p: (0, info) for p, info in notes.items()} == self.folders:
            return
        with self._write_lock:
            if self._closed:
                return
            self._writer.execute("BEGIN IMMEDIATE")
            self._writer.execute("DELETE FROM folders")
            self._writer.executemany("INSERT INTO folders VALUES (?, ?)", [(p, dumps(info)) for p, info in notes.items()])
            self._writer.execute("COMMIT")
        self.folders = {p: (0, info) for p, info in notes.items()}
        self._view_gen += 1

    def _connected(self, up: bool) -> None:
        if up and not self.connected:
            self._relist.set()
            self._wake.set()
        self.connected = up

    # ---- answering what only the upstream has ----

    def live_seqs(self, prefix: str) -> dict[str, tuple[int, int]]:
        """{path: (rows, media)} of running runs, as the upstream last told of them while it is reachable."""
        if not self.connected:
            return super().live_seqs(prefix)
        with self.lock:
            return {p: (m["seq"], m["mseq"]) for p, m in self._listed.items() if in_scope(p, prefix) and m["state"] == "running"}

    def backfill(self, prefix: str) -> list[bytes]:
        """`rows` events of the running runs under `prefix` beyond their kept buckets, from the upstream while it is
        reachable."""
        if not self.connected:
            return []
        with self.lock:
            tails = [(p, st["kept_seq"]) for p, st in self.records.items() if in_scope(p, prefix) and st["state"] == "running"]

        def rows(tail: tuple[str, int]) -> bytes:
            try:
                return sse_text("rows", self.upstream.request("GET", f"/api/rows?path={quote(tail[0])}&from={tail[1]}").decode())
            except (KeyError, Unreachable):
                return b""

        with ThreadPoolExecutor(FETCH_THREADS) as pool:
            return [b for b in pool.map(rows, tails) if b]

    def rows_json(self, path: str, start: int, stop: int | None = None) -> str:
        """The `rows` event JSON of the run's rows from `start`, from the upstream; none while it is unreachable."""
        with self.lock:
            if path not in self.records and path not in self._listed:
                raise KeyError(path)
        try:
            return self.upstream.request("GET", f"/api/rows?path={quote(path)}&from={start}").decode()
        except Unreachable:
            return f'{{"run":{dumps(path)},"seq0":{start},"rows":[]}}'

    def buckets_bodies(self, asks: Sequence[Ask]) -> list[bytes]:
        """Each block as `Explorer.buckets_body` answers it; from the upstream, while it is reachable, for the blocks
        of runs named by id of which one is running."""
        live = [i for i, a in enumerate(asks) if a.runs is not None and any(self._running(r) for r in a.runs)] if self.connected else []
        rest = sorted(set(range(len(asks))) - set(live))
        out = dict(zip(rest, super().buckets_bodies([asks[i] for i in rest]), strict=True))
        if live:
            try:
                got = self._asked([asks[i] for i in live])
            except Unreachable:
                got = super().buckets_bodies([asks[i] for i in live])
            out.update(zip(live, got, strict=True))
        return [out[i] for i in range(len(asks))]

    def _running(self, path: str) -> bool:
        st = self.records.get(path)
        return st is not None and st["state"] == "running"

    def _asked(self, asks: Sequence[Ask]) -> list[bytes]:
        """The upstream's answers to `asks`."""
        blocks = [{"key": a.key, "level": a.level, "index": a.block, "scope": a.scope,
                   "runs": None if a.runs is None else list(a.runs), "which": a.which} for a in asks]
        return bk.unframe(self.upstream.request("POST", "/api/buckets", dumps({"blocks": blocks}).encode()))

    def _build_many(self, paths: list[str], key: str, level: int, index: int) -> list[tuple[bytes, int]]:
        """Block (level, index) of `key` of each run, from the upstream (a mirror has no run files); empty while it is
        unreachable."""
        try:
            a = bk.decode(self._asked([Ask(key, level, index, "", paths, "all")])[0])
        except (Unreachable, ValueError):
            return [(bk.encode(level, index, [""], [0], bk.empty()), 0) for _ in paths]
        row = {p: i for i, p in enumerate(a.paths)}
        out: list[tuple[bytes, int]] = []
        for p in paths:
            i = row.get(p)
            b = bk.empty() if i is None else bk.select(a.buckets, a.buckets.run == i)
            seq = 0 if i is None else int(a.seq[i])
            out.append((bk.encode(level, index, [""], [seq], b._replace(run=b.run * 0)), seq))
        return out

    def media_path(self, path: str, file: str) -> Path:
        """The mirrored copy of run `path`'s media file `file`, copied first when missing; KeyError for one it lacks."""
        name = file.removeprefix("media/")
        if "/" in name or name.startswith(".") or not name:
            raise KeyError(file)
        f = self._media_file(path, file)
        if f is None:
            raise KeyError(file)
        return f

    def _media_file(self, path: str, file: str) -> Path | None:
        """The local copy of media `file` (named for its contents, so one copy serves every run), copied from the
        upstream when missing; None when it cannot be."""
        f = self.media_dir / Path(file).name
        if f.exists():
            return f
        try:
            data = self.upstream.request("GET", f"/m/{quote(path, safe='')}/{file}")
        except (KeyError, Unreachable):
            return None
        tmp = f.with_name(f".{f.name}.{threading.get_ident()}.tmp")
        tmp.write_bytes(data)
        tmp.replace(f)
        return f
