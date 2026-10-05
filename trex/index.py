"""A directory's index, in <cache_root>/<hash of its origin's key>/index.sqlite, kept current by its origin: a runs
directory on this machine (`trex.crawl`) or another trex holding it (`trex.mirror`). It holds each run's metadata, last
values and media, and its buckets of every metric at every level from about one row per bucket up to where the run
lies in two blocks (`trex.buckets`); beside it, in levels/, the finished runs' buckets merged per level. A run is its
path in the directory. Everything a server answers for a directory comes from here: its runs, blocks of buckets, the
live stream, and the dumps other trex pull.
"""

import contextlib
import fcntl
import functools
import hashlib
import itertools
import json
import math
import multiprocessing
import os
import queue
import shutil
import signal
import sqlite3
import struct
import sys
import threading
import time
import zlib
from collections import OrderedDict
from collections.abc import Callable, Generator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from itertools import batched
from multiprocessing.context import BaseContext
from pathlib import Path
from typing import Any, Final, Literal, Protocol, Self, TextIO, cast

import numpy as np
import numpy.typing as npt

from . import buckets as bk
from .buckets import Buckets, Stack
from .format import JSONValue, MediaKind, RunState

type Sig = list[int]
"""[db mtime_ns, db size, wal mtime_ns, wal size] of a run's files; 0s for a missing file."""

type Summary = dict[str, float | str]
"""Last value of every metric (non-finite as strings), plus _step and _runtime."""

type Which = Literal["all", "finished", "running"]
"""The runs of a scope a block request takes: all, or the finished or the running ones."""

type Event = tuple[str, object]
"""(SSE event name, data); data that is a str is already JSON."""


@dataclass(frozen=True, slots=True)
class Ask:
    """Block `block` of `level` of `key`: of the runs `runs` (ids), or else of the runs under `scope` in state `which`."""

    key: str
    level: int
    block: int
    scope: str = ""
    runs: Sequence[str] | None = None
    which: Which = "all"


@dataclass(frozen=True, slots=True)
class MediaRecord:
    """A media item of a run."""

    run: str
    seq: int
    step: float
    key: str
    kind: MediaKind
    file: str

    def row(self) -> tuple[str, int, float, str, MediaKind, str]:
        """Its fields in order, as the index and JSON hold it."""
        return self.run, self.seq, self.step, self.key, self.kind, self.file


@dataclass(frozen=True, slots=True)
class Metric:
    """A metric of a run and how it is compiled."""

    key: str
    span: bk.Span


@dataclass(frozen=True, slots=True)
class Block:
    """A block of a run's metric: a compressed one-run bucket array holding no row count; `since`, the rows compiled when
    it last changed."""

    key: str
    level: int
    block: int
    since: int
    data: bytes


@dataclass(frozen=True, slots=True, kw_only=True)
class Public:
    """Metadata from the run file."""

    name: str | None
    tags: list[str]
    config: dict[str, JSONValue]
    created: float | None
    info: dict[str, JSONValue]
    user_summary: dict[str, JSONValue]

    def wire(self) -> dict[str, Any]:
        return {"name": self.name, "tags": self.tags, "config": self.config, "created": self.created, "info": self.info,
                "user_summary": self.user_summary}

    @classmethod
    def read(cls, d: Mapping[str, Any]) -> Self:
        return cls(name=d["name"], tags=d["tags"], config=d["config"], created=d["created"], info=d["info"],
                   user_summary=d["user_summary"])


@dataclass(frozen=True, slots=True, kw_only=True)
class RunRecord:
    """What the index keeps per run."""

    uid: str
    seq: int
    mseq: int
    keys: list[str]
    summary: Summary
    sig: Sig
    heartbeat: float | None
    public: Public
    state: RunState
    compiled: int  # the rows its levels hold
    compiled_t: float  # when they last changed here
    rebuilt: int  # `compiled` when its levels were last compiled from every row
    ver: int  # bumped by every change where the run is crawled; the same wherever it is held

    def wire(self) -> dict[str, Any]:
        """As JSON holds it: in the index, and in the dumps mirrors take."""
        return {"uid": self.uid, "seq": self.seq, "mseq": self.mseq, "keys": self.keys, "summary": self.summary,
                "sig": self.sig, "heartbeat": self.heartbeat, "public": self.public.wire(), "state": self.state,
                "compiled": self.compiled, "compiled_t": self.compiled_t, "rebuilt": self.rebuilt, "ver": self.ver}

    @classmethod
    def read(cls, d: Mapping[str, Any]) -> Self:
        return cls(uid=d["uid"], seq=d["seq"], mseq=d["mseq"], keys=d["keys"], summary=d["summary"], sig=d["sig"],
                   heartbeat=d["heartbeat"], public=Public.read(d["public"]), state=d["state"], compiled=d["compiled"],
                   compiled_t=d["compiled_t"], rebuilt=d["rebuilt"], ver=d["ver"])


@dataclass(frozen=True, slots=True, kw_only=True)
class Update:
    """What changed in a run, as a scan of its run file or a dump from another trex finds it."""

    path: str
    sig: Sig
    uid: str
    reset: bool  # the run was replaced: what the index holds of it goes first
    fresh: bool  # the index held nothing of it
    seq: int
    mseq: int
    media: list[MediaRecord]
    state: RunState
    heartbeat: float | None
    public: Public
    keys: list[str]
    summary: Summary | None  # None: as it was
    compiled: int | None  # the rows its levels hold once `blocks` are written; None: as they were
    rebuilt: int
    metrics: list[Metric]  # of the metrics compiled
    blocks: list[Block]  # blocks that changed
    replace: bool  # `metrics` and `blocks` are all the run has
    rows: str | None = None  # its `rows` event
    ver: int | None = None  # None: one more than it was


@dataclass(frozen=True, slots=True)
class Have:
    """What a mirror holds of a run: its uid, media items, and the rows and rebuild of its levels."""

    uid: str
    mseq: int
    compiled: int
    rebuilt: int

    def wire(self) -> dict[str, Any]:
        return {"uid": self.uid, "mseq": self.mseq, "compiled": self.compiled, "rebuilt": self.rebuilt}

    @classmethod
    def read(cls, d: Mapping[str, Any]) -> Self:
        """ValueError unless the fields are a string and whole numbers."""
        uid, counts = d["uid"], [d["mseq"], d["compiled"], d["rebuilt"]]
        if not isinstance(uid, str) or not all(isinstance(n, int) and not isinstance(n, bool) for n in counts):
            raise ValueError(f"not what a mirror holds of a run: {d!r}")
        return cls(uid, *counts)


@dataclass(frozen=True, slots=True, kw_only=True)
class RunMeta:
    """A run, as sent to the browser and to mirrors; `dir` names its directory in a view of several."""

    id: str
    uid: str
    name: str
    parent: str
    tags: list[str]
    config: dict[str, JSONValue]
    info: dict[str, JSONValue]
    summary: dict[str, JSONValue]
    state: RunState
    created: float | None
    updated: float | None
    seq: int
    mseq: int
    keys: list[str]
    compiled: int
    ver: int
    dir: str | None = None

    def wire(self) -> dict[str, Any]:
        out = {"id": self.id, "uid": self.uid, "name": self.name, "parent": self.parent, "tags": self.tags,
               "config": self.config, "info": self.info, "summary": self.summary, "state": self.state,
               "created": self.created, "updated": self.updated, "seq": self.seq, "mseq": self.mseq, "keys": self.keys,
               "compiled": self.compiled, "ver": self.ver}
        return out if self.dir is None else {**out, "dir": self.dir}

    @classmethod
    def read(cls, d: Mapping[str, Any]) -> Self:
        return cls(id=d["id"], uid=d["uid"], name=d["name"], parent=d["parent"], tags=d["tags"], config=d["config"],
                   info=d["info"], summary=d["summary"], state=d["state"], created=d["created"], updated=d["updated"],
                   seq=d["seq"], mseq=d["mseq"], keys=d["keys"], compiled=d["compiled"], ver=d["ver"], dir=d.get("dir"))


@dataclass(frozen=True, slots=True)
class RunsView:
    """Runs under a folder, their media, and the notes of related folders."""

    runs: list[RunMeta]
    media: list[MediaRecord]
    folders: dict[str, dict[str, JSONValue]]

    def wire(self) -> dict[str, Any]:
        return {"runs": [m.wire() for m in self.runs], "media": [m.row() for m in self.media], "folders": self.folders}

    @classmethod
    def read(cls, d: Mapping[str, Any]) -> Self:
        return cls([RunMeta.read(m) for m in d["runs"]], [MediaRecord(*m) for m in d["media"]], d["folders"])


@dataclass(frozen=True, slots=True)
class RunView:
    run: RunMeta
    media: list[MediaRecord]

    def wire(self) -> dict[str, Any]:
        return {"run": self.run.wire(), "media": [m.row() for m in self.media]}


@dataclass(frozen=True, slots=True)
class Rows:
    """A `rows` event: rows [seq0, seq0 + len(rows)) of run `run`, each [step, runtime, {metric: value}] with
    non-finite values as text (`wire`)."""

    run: str
    seq0: int
    rows: list[list[Any]]

    @property
    def end(self) -> int:
        return self.seq0 + len(self.rows)

    def since(self, seq: int) -> Self:
        """Its rows from row `seq` on (from `seq0` when that is later)."""
        at = max(seq, self.seq0)
        return type(self)(self.run, at, self.rows[at - self.seq0:])

    def text(self) -> str:
        """Its JSON, `run` first."""
        return f'{{"run":{dumps(self.run)},"seq0":{self.seq0},"rows":{dumps(self.rows)}}}'

    @classmethod
    def read(cls, d: Mapping[str, Any]) -> Self:
        return cls(d["run"], d["seq0"], d["rows"])


@dataclass(frozen=True, slots=True)
class Dump:
    """What a mirror lacks of a run: its record, the media items it lacks, its metrics, and the blocks of its levels
    that changed since the rows the mirror's hold; every block, to take the place of the mirror's, when `replace`."""

    record: RunRecord
    media: list[MediaRecord]
    replace: bool
    metrics: list[Metric]
    blocks: list[Block]

    def encode(self) -> bytes:
        """One body (`buckets.frame`): a JSON header, then each block's data."""
        head = {"record": self.record.wire(), "media": [m.row()[1:] for m in self.media], "replace": self.replace,
                "metrics": [[m.key, m.span.fine, m.span.top, m.span.lo, m.span.hi] for m in self.metrics],
                "blocks": [[b.key, b.level, b.block, b.since] for b in self.blocks]}
        return bk.frame([dumps(head).encode(), *(b.data for b in self.blocks)])

    @classmethod
    def decode(cls, path: str, body: bytes) -> Self:
        """The dump of run `path` that `body` encodes."""
        parts = bk.unframe(body)
        head = json.loads(parts[0])
        return cls(RunRecord.read(head["record"]), [MediaRecord(path, *m) for m in head["media"]], head["replace"],
                   [Metric(key, bk.Span(fine, top, lo, hi)) for key, fine, top, lo, hi in head["metrics"]],
                   [Block(key, level, block, since, data)
                    for (key, level, block, since), data in zip(head["blocks"], parts[1:], strict=True)])


class Origin(Protocol):
    """Where an Explorer's runs come from, and what of them only it has: rows its levels do not hold yet, media files."""

    key: str  # names the index: one cache directory per key

    def attach(self, ex: "Explorer") -> None:
        """Told once, first, of the Explorer it fills."""
        ...

    def run(self, ex: "Explorer") -> None:
        """Keep `ex` current until it stops, setting `ex.ready` after the first pass."""
        ...

    def sync(self, ex: "Explorer") -> list[str]:
        """One pass now; the runs it updated."""
        ...

    def stop(self) -> None:
        """End `run`'s waiting."""
        ...

    def close(self) -> None: ...

    def info(self) -> dict[str, object]:
        """What it is, for /api/info: at least `root` and `name`."""
        ...

    def rows_json(self, path: str, start: int) -> str:
        """The `rows` event JSON of the rows from `start` it has of a run; KeyError for a run it does not have."""
        ...

    def live(self, path: str, rec: RunRecord) -> tuple[int, int]:
        """(rows, media) it has of a running run."""
        ...

    def media_file(self, path: str, file: str) -> Path:
        """A run's media file; KeyError unless it has it."""
        ...


CACHE_VERSION: Final = 15  # bump whenever what the index stores changes; older caches are rebuilt
CLOSE_WAIT: Final = 5.0  # longest `close` waits for a pass in progress
MEMO_BYTES = 1 << 30  # stacks, merged levels and answers an Explorer keeps in memory, least recently used dropped
LEVELS_BYTES = int(os.environ.get("TREX_LEVELS_MB", "4096")) << 20  # saved merged levels an index keeps, least recently used deleted
LEVELS_SAVE_EVERY = 60.0  # seconds between saves of one metric's merged levels
LEVELS_AHEAD: Final = 3  # levels above a metric's coarsest top level merged and saved ahead of requests
MERGE_THREADS: Final = min(8, os.cpu_count() or 1)  # threads merging one level
READ_THREADS: Final = min(8, os.cpu_count() or 1)  # threads reading and decompressing stored blocks
READ_ALONE: Final = 48  # runs whose stored blocks one thread reads
BLOCK_WORKERS: Final = min(4, os.cpu_count() or 1)  # processes reading and joining stored blocks
BLOCK_ALONE: Final = 32  # runs whose stored blocks the asking thread reads and joins itself
SLICE: Final = 512  # runs of a block one block worker reads, at most
STARTING: Final = 0.05  # seconds each block worker is kept busy as the workers start, so that every one starts
PAGE_SIZE: Final = 16384
READ_ERRORS: Final = (sqlite3.Error, OSError, ValueError, KeyError, struct.error)
ROWS_EVENT_MAX: Final = 20_000  # larger catch-ups reach browsers through newly compiled levels instead of rows
HEARTBEAT: Final = 10.0  # seconds of stream silence after which a heartbeat is sent
STREAM_BATCH: Final = 2000  # events sent together at most
STOP_POLL: Final = 0.5  # seconds between a stream's checks of its stop event
PATHS_PER_QUERY: Final = 500  # runs one index query names

TABLES: Final = {
    "cache": "CREATE TABLE IF NOT EXISTS cache(key TEXT PRIMARY KEY, value TEXT)",
    "runs": "CREATE TABLE IF NOT EXISTS runs(path TEXT PRIMARY KEY, record TEXT NOT NULL)",
    "media": "CREATE TABLE IF NOT EXISTS media(path TEXT, seq INTEGER, step REAL, key TEXT, kind TEXT, file TEXT, "
             "PRIMARY KEY(path, seq)) WITHOUT ROWID",
    # per run and metric: the finest and top levels it is compiled at, and the steps it spans
    "metrics": "CREATE TABLE IF NOT EXISTS metrics(path TEXT, key TEXT, fine INTEGER, top INTEGER, lo REAL, hi REAL, "
               "PRIMARY KEY(key, path))",
    "metrics_path": "CREATE INDEX IF NOT EXISTS metrics_path ON metrics(path)",
    # a run's levels of a metric without reading its row
    "metrics_levels": "CREATE INDEX IF NOT EXISTS metrics_levels ON metrics(key, path, fine, top)",
    # a run's buckets of a metric at every level from fine to top, by block (compressed one-run bucket arrays, those
    # holding buckets); since: the rows compiled when the block last changed
    "levels": "CREATE TABLE IF NOT EXISTS levels(key TEXT, level INTEGER, block INTEGER, path TEXT, since INTEGER, data BLOB, "
              "PRIMARY KEY(key, level, block, path))",
    "levels_path": "CREATE INDEX IF NOT EXISTS levels_path ON levels(path)",
    "folders": "CREATE TABLE IF NOT EXISTS folders(path TEXT PRIMARY KEY, info TEXT NOT NULL)",
}


def dumps(obj: object) -> str:
    """Compact JSON (non-finite floats raise; see `wire`)."""
    return json.dumps(obj, separators=(",", ":"), allow_nan=False)


def wire(d: Mapping[str, float]) -> dict[str, float | str]:
    """Metrics for JSON: non-finite values as "nan", "inf", "-inf"."""
    return {k: v if math.isfinite(v) else str(v) for k, v in d.items()}


def sse(event: str, data: object) -> bytes:
    return f"event: {event}\ndata: {dumps(data)}\n\n".encode()


def sse_text(event: str, text: str) -> bytes:
    return f"event: {event}\ndata: {text}\n\n".encode()


def in_scope(path: str, prefix: str) -> bool:
    return not prefix or path == prefix or path.startswith(prefix + "/")


def pack(b: Buckets, level: int, block: int) -> bytes:
    """One run's buckets of a block as the index stores them."""
    return zlib.compress(bk.encode(level, block, [""], [0], b), 1)


def unpack(data: bytes) -> Buckets:
    return bk.decode(zlib.decompress(data)).buckets


class Subscriber:
    """A bounded event queue for runs under a folder; dead once it overflows."""

    def __init__(self, prefix: str, maxsize: int = 20000) -> None:
        self.prefix = prefix
        self.q: queue.Queue[bytes] = queue.Queue(maxsize=maxsize)
        self.dead = False

    def put(self, msg: bytes) -> None:
        try:
            self.q.put_nowait(msg)
        except queue.Full:
            self.dead = True


class Hub:
    """Fan-out of events to subscribers whose folder scope contains the run."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.subs: set[Subscriber] = set()

    def subscribe(self, prefix: str) -> Subscriber:
        s = Subscriber(prefix)
        with self.lock:
            self.subs.add(s)
        return s

    def unsubscribe(self, s: Subscriber) -> None:
        with self.lock:
            self.subs.discard(s)

    def close(self) -> None:
        """End every subscription."""
        with self.lock:
            for s in self.subs:
                s.dead = True

    def watched(self, path: str) -> bool:
        """Whether any subscriber's scope contains the run."""
        with self.lock:
            return any(in_scope(path, s.prefix) for s in self.subs)

    def publish(self, path: str, event: str, data: object) -> None:
        self.publish_msg(path, sse(event, data))

    def publish_msg(self, path: str, msg: bytes) -> None:
        with self.lock:
            subs = [s for s in self.subs if in_scope(path, s.prefix)]
        for s in subs:
            s.put(msg)


def _bound_dir(d: Path, limit: int) -> None:
    """Delete the least recently used entries of `d` (files, or directories of files) until the rest hold at most
    `limit` bytes."""
    entries: list[tuple[float, int, Path]] = []
    for e in d.iterdir():
        try:
            files = [e] if e.is_file() else list(e.iterdir())
            entries.append((max(f.stat().st_mtime for f in files) if files else 0.0, sum(f.stat().st_size for f in files), e))
        except OSError:
            continue
    total = sum(x[1] for x in entries)
    for _, size, e in sorted(entries, key=lambda x: x[0]):
        if total <= limit:
            return
        if e.is_dir():
            shutil.rmtree(e, ignore_errors=True)
        else:
            e.unlink(missing_ok=True)
        total -= size


def _events(r: Update, cur: RunRecord | None, st: RunRecord) -> list[Event]:
    """An update's events: a run event follows the rows and media it counts; a new run's comes first."""
    rows: list[Event] = [("rows", r.rows)] if r.rows is not None else []
    media: list[Event] = [("media", m.row()) for m in r.media]
    if cur is None:
        return [("run", None), *rows, *media]
    changed = r.fresh or (st.compiled, st.state, st.public, st.keys) != (cur.compiled, cur.state, cur.public, cur.keys)
    return [*rows, *media, *([("run", None)] if changed else [])]


def _record(r: Update, cur: RunRecord | None, now: float) -> RunRecord:
    """The run's record after update `r`."""
    compiled = r.compiled if r.compiled is not None else cur.compiled if cur else 0
    return RunRecord(uid=r.uid, seq=r.seq, mseq=r.mseq, keys=r.keys,
                     summary=r.summary if r.summary is not None else cur.summary if cur else {},
                     sig=r.sig, heartbeat=r.heartbeat, public=r.public, state=r.state, compiled=compiled,
                     compiled_t=now if cur is None or compiled != cur.compiled else cur.compiled_t, rebuilt=r.rebuilt,
                     ver=r.ver if r.ver is not None else cur.ver + 1 if cur else 1)


@dataclass(frozen=True, slots=True, eq=False)
class Finished:
    """The finished runs logging a metric, in path order, the rows of each its levels hold, and a digest of them and
    their levels."""

    paths: list[str]
    seq: npt.NDArray[np.uint32]
    sig: bytes


class Memo:
    """Values kept under keys while their generation stays the same, the least recently used dropped beyond `limit`
    bytes; a value is built by one thread while others asking for it wait."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.items: OrderedDict[tuple[object, ...], tuple[int, object, int]] = OrderedDict()  # key -> (gen, value, bytes)
        self.bytes = 0
        self.lock = threading.Lock()
        self.building: dict[tuple[object, ...], threading.Lock] = {}

    def get[T](self, k: tuple[object, ...], gen: int, build: Callable[[], tuple[T, int]]) -> T:
        """The value kept under `k` at generation `gen`, else the value of build() (value, bytes), kept."""
        hit = self._hit(k, gen)
        if hit is None:
            with self.lock:
                lock = self.building.setdefault(k, threading.Lock())
            try:
                with lock:
                    hit = self._hit(k, gen)
                    if hit is None:
                        value, nbytes = build()
                        self._put(k, gen, value, nbytes)
                        hit = (value,)
            finally:
                with self.lock:
                    self.building.pop(k, None)
        return cast(T, hit[0])

    def _hit(self, k: tuple[object, ...], gen: int) -> tuple[object] | None:
        with self.lock:
            item = self.items.get(k)
            if item is None or item[0] != gen:
                return None
            self.items.move_to_end(k)
            return (item[1],)

    def _put(self, k: tuple[object, ...], gen: int, value: object, nbytes: int) -> None:
        with self.lock:
            old = self.items.pop(k, None)
            self.bytes += nbytes - (old[2] if old else 0)
            self.items[k] = (gen, value, nbytes)
            while self.bytes > self.limit and len(self.items) > 1:
                self.bytes -= self.items.popitem(last=False)[1][2]


def _sized(body: bytes) -> tuple[bytes, int]:
    return body, len(body)


# ---- stored blocks: read and joined by the asking thread, or by a block worker ----

_reading = ThreadPoolExecutor(READ_THREADS, thread_name_prefix="trex-read")  # reads of stored blocks, in every process
_readers: dict[str, tuple[int, queue.LifoQueue[sqlite3.Connection]]] = {}  # index file -> (its inode, idle read connections)
_readers_lock = threading.Lock()
_answering = ThreadPoolExecutor(2 * BLOCK_WORKERS, thread_name_prefix="trex-ask")  # the asks of a request being answered


def at_once[A, T](answer: Callable[[A], T], asks: Sequence[A], shared: Callable[[A], bool],
                  threads: ThreadPoolExecutor = _answering) -> list[T]:
    """`answer` of each of `asks`, in order. While the block workers are up, the asks whose blocks they may read
    (`shared`) are answered several at a time on `threads`, each waiting for its workers, and the others meanwhile by
    the calling thread; else one after another."""
    if _workers.pool is None or sum(map(shared, asks)) < 2:
        return [answer(a) for a in asks]
    waiting = {i: threads.submit(answer, a) for i, a in enumerate(asks) if shared(a)}
    return [waiting[i].result() if i in waiting else answer(a) for i, a in enumerate(asks)]


def for_workers(a: Ask) -> bool:
    """Whether the block workers may read ask `a`'s block: one of a scope's runs, or of more runs than BLOCK_ALONE."""
    return a.runs is None or len(a.runs) > BLOCK_ALONE


def worker_init() -> None:
    """Workers leave Ctrl-C to the main process, which stops them, and end when it does."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    threading.Thread(target=_end_with_parent, name="trex-parent", daemon=True).start()


def _end_with_parent() -> None:
    parent = multiprocessing.parent_process()
    if parent is not None:
        parent.join()
        os._exit(0)


def mp_context() -> BaseContext:
    """Fresh worker interpreters that import only trex (the main process runs threads)."""
    if "forkserver" in multiprocessing.get_all_start_methods():
        ctx = multiprocessing.get_context("forkserver")
        ctx.set_forkserver_preload(["trex.crawl"])
        return ctx
    return multiprocessing.get_context("spawn")


def _read_only(db: str) -> str:
    """The SQLite URI that opens file `db` read-only."""
    return "file:" + db.replace("%", "%25").replace("?", "%3f").replace("#", "%23") + "?mode=ro"


@contextlib.contextmanager
def _reader(db: str) -> Generator[sqlite3.Connection]:
    """A read connection to index file `db`, kept for the next reader while the file is the one it was opened on."""
    inode = os.stat(db).st_ino
    with _readers_lock:
        held, stale = _readers.get(db), None
        if held is None or held[0] != inode:
            stale, held = held, (inode, queue.LifoQueue[sqlite3.Connection]())
            _readers[db] = held
    if stale is not None:
        _close_idle(stale[1])
    try:
        c = held[1].get_nowait()
    except queue.Empty:
        c = sqlite3.connect(_read_only(db), uri=True, check_same_thread=False, isolation_level=None, timeout=60)
    try:
        yield c
    finally:
        with _readers_lock:
            kept = _readers.get(db) is held
            if kept:
                held[1].put(c)
        if not kept:
            c.close()


def _forget_readers(db: str) -> None:
    """Close this process's idle read connections to index file `db`."""
    with _readers_lock:
        held = _readers.pop(db, None)
    if held is not None:
        _close_idle(held[1])


def _close_idle(idle: queue.LifoQueue[sqlite3.Connection]) -> None:
    with contextlib.suppress(queue.Empty):
        while True:
            idle.get_nowait().close()


def stored_block(db: str, key: str, level: int, index: int, paths: Sequence[str], seq: bytes) -> bytes:
    """Block `index` of `level` of `key` of the runs `paths` (in path order; `seq` the rows their levels hold, u32) as
    a bucket array, from the blocks index file `db` stores: each run's block as it is stored (`buckets.join`), or,
    below its finest level, that level's buckets refined."""
    fine: dict[str, int] = {}
    with _reader(db) as c:
        for part in batched(paths, PATHS_PER_QUERY):
            fine.update(c.execute(f"SELECT path, fine FROM metrics WHERE key=? AND path IN ({_marks(part)})", (key, *part)))
    rows = {src: found for src in sorted({max(f, level) for f in fine.values()})
            if (found := _read(db, key, src, index >> (src - level), [p for p in paths if p in fine and max(fine[p], level) == src]))}
    held = np.frombuffer(seq, "<u4")
    if set(rows) <= {level}:
        stored = dict(rows.get(level, ()))
        return bk.join(level, index, paths, held, [stored.get(p) for p in paths])
    place = {p: i for i, p in enumerate(paths)}
    parts: list[Buckets] = []
    for src, found in rows.items():
        b = bk.stack([], np.empty(0, np.uint32), np.empty(0, np.int8), [blob for _, blob in found], [place[p] for p, _ in found]).buckets
        parts.append(b if src == level else bk.cut(bk.refine(b, src, level), index * bk.BLOCK, (index + 1) * bk.BLOCK))
    return bk.encode(level, index, paths, held, bk.union(parts))


def _read(db: str, key: str, level: int, block: int, paths: Sequence[str]) -> list[tuple[str, bytes]]:
    """(path, one-run bucket array) of each of `paths` (in path order) storing block `block` of `level` of `key`, in
    path order: read and decompressed on READ_THREADS threads when there are more than READ_ALONE."""
    if len(paths) <= READ_ALONE:
        return _read_some(db, key, level, block, paths)
    size = max(READ_ALONE, -(-len(paths) // READ_THREADS))
    chunks = [paths[i:i + size] for i in range(0, len(paths), size)]
    return [row for rows in _reading.map(functools.partial(_read_some, db, key, level, block), chunks) for row in rows]


def _read_some(db: str, key: str, level: int, block: int, paths: Sequence[str]) -> list[tuple[str, bytes]]:
    out: list[tuple[str, bytes]] = []
    with _reader(db) as c:
        for part in batched(paths, PATHS_PER_QUERY):
            out += [(p, zlib.decompress(data)) for p, data in c.execute(
                f"SELECT path, data FROM levels WHERE key=? AND level=? AND block=? AND path IN ({_marks(part)}) ORDER BY path",
                (key, level, block, *part))]
    return out


class Workers:
    """The block workers: BLOCK_WORKERS processes that read and join stored blocks, started on a thread of their own by
    the first block that is theirs to read, and used once each has answered."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.pool: ProcessPoolExecutor | None = None  # once each has answered
        self.starting = False

    def ready(self) -> ProcessPoolExecutor | None:
        """The workers once they answer; the first call starts them."""
        with self.lock:
            if self.pool is None and not self.starting:
                self.starting = True
                threading.Thread(target=self.start, name="trex-workers", daemon=True).start()
            return self.pool

    def start(self) -> None:
        """Start the workers and wait for each to answer; they stay unused if one does not."""
        pool = ProcessPoolExecutor(BLOCK_WORKERS, mp_context=mp_context(), initializer=worker_init)
        try:
            for started in [pool.submit(time.sleep, STARTING) for _ in range(BLOCK_WORKERS)]:
                started.result()
        except (RuntimeError, OSError) as e:
            print(f"[trex] block workers: {e!r}", file=sys.stderr, flush=True)
            pool.shutdown(wait=False)
            return
        with self.lock:
            self.pool = pool

    def drop(self, pool: ProcessPoolExecutor) -> None:
        """Stop using `pool`, which failed; the next block that is the workers' starts them anew."""
        with self.lock:
            if self.pool is pool:
                self.pool, self.starting = None, False
        pool.shutdown(wait=False)


_workers = Workers()
_reading_here = threading.Lock()  # the asking thread reading a block the workers would


def _stored(db: str, key: str, level: int, index: int, paths: Sequence[str], seq: bytes) -> bytes:
    """`stored_block`: by the block workers when the runs are more than BLOCK_ALONE and the workers are up; else (and
    when they fail) by the asking thread, one block of that many runs at a time."""
    if len(paths) <= BLOCK_ALONE:
        return stored_block(db, key, level, index, paths, seq)
    pool = _workers.ready()
    if pool is not None:
        try:
            return _shared(pool, db, key, level, index, paths, seq)
        except (RuntimeError, OSError) as e:
            print(f"[trex] block worker: {e!r}", file=sys.stderr, flush=True)
            _workers.drop(pool)
    with _reading_here:
        return stored_block(db, key, level, index, paths, seq)


def _shared(pool: ProcessPoolExecutor, db: str, key: str, level: int, index: int, paths: Sequence[str], seq: bytes) -> bytes:
    """`stored_block` by the workers of `pool`, each reading a slice of the runs: a slice for every worker, of at least
    BLOCK_ALONE runs and at most SLICE."""
    slices = max(-(-len(paths) // SLICE), min(BLOCK_WORKERS, len(paths) // max(BLOCK_ALONE, 1)), 1)
    size = -(-len(paths) // slices)
    asked = [pool.submit(stored_block, db, key, level, index, paths[i:i + size], seq[4 * i:4 * (i + size)])
             for i in range(0, len(paths), size)]
    bodies = [f.result() for f in asked]
    return bodies[0] if len(bodies) == 1 else bk.chain(bodies, paths)


class Explorer:
    """The index of what `origin` holds, under `cache_root`, kept current by `start` (or `sync`); `close` releases it.
    An index is held by one Explorer at a time: another of the same origin takes the next one free (`<hash>-1`, ...),
    so each process keeps an index of its own, and a later one takes it up again."""

    def __init__(self, origin: Origin, cache_root: str | os.PathLike[str]) -> None:
        self.origin = origin
        self.cache_dir, self._lock_file = _free_index(Path(cache_root) / hashlib.sha1(origin.key.encode()).hexdigest()[:12])
        (self.cache_dir / "root.txt").write_text(origin.key + "\n")
        self.db_path = self.cache_dir / "index.sqlite"
        self._writer = self._connect()
        self._writer.execute(TABLES["cache"])
        v = self._writer.execute("SELECT value FROM cache WHERE key='version'").fetchone()
        if v is None or int(v[0]) != CACHE_VERSION:
            self._writer.execute("BEGIN IMMEDIATE")
            for t in [r[0] for r in self._writer.execute("SELECT name FROM sqlite_master WHERE type='table' AND name != 'cache'")]:
                self._writer.execute(f"DROP TABLE {t}")
            self._writer.execute("INSERT OR REPLACE INTO cache VALUES ('version', ?)", (str(CACHE_VERSION),))
            self._writer.execute("COMMIT")
            self._writer.execute("VACUUM")  # the dropped tables' pages go back to the disk
            shutil.rmtree(self.cache_dir / "levels", ignore_errors=True)
        for sql in TABLES.values():
            self._writer.execute(sql)
        self._write_lock = threading.Lock()  # serializes every use of `_writer`
        self.lock = threading.Lock()
        self.hub = Hub()
        self.records: dict[str, RunRecord] = {
            path: RunRecord.read(json.loads(s)) for path, s in self._writer.execute("SELECT path, record FROM runs")}
        self.folders: dict[str, dict[str, JSONValue]] = {
            p: json.loads(info) for p, info in self._writer.execute("SELECT path, info FROM folders")}  # folder -> its notes
        self._readers: queue.LifoQueue[sqlite3.Connection] = queue.LifoQueue()
        self._stop = threading.Event()
        self._runner: threading.Thread | None = None
        self._closed = False
        self.ready = threading.Event()
        self._gens: dict[str, int] = {}  # metric -> bumped whenever its finished runs, or their levels, change
        self._memo = Memo(MEMO_BYTES)  # stacks, merged levels, block answers and `runs` answers
        self._saved_at: dict[str, float] = {}  # metric -> when its merged levels were last saved (monotonic)
        self._view_gen = 0  # bumped whenever what `runs` answers may change
        origin.attach(self)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def start(self) -> Self:
        """Have the origin keep the index current, on a thread, until `close`."""
        self._runner = threading.Thread(target=self.origin.run, args=(self,), name=f"trex-{self.origin.key}", daemon=True)
        self._runner.start()
        return self

    def sync(self) -> list[str]:
        """One pass of the origin now; the runs it updated."""
        return self.origin.sync(self)

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    def wait(self, seconds: float) -> bool:
        """Wait `seconds`, or until stopped; whether it was stopped."""
        return self._stop.wait(seconds)

    def stop(self) -> None:
        self._stop.set()
        self.origin.stop()

    def close(self) -> None:
        """Stop the origin, end subscriptions, close the index's connections and let go of the index."""
        self.stop()
        if self._runner is not None and self._runner is not threading.current_thread():
            self._runner.join(CLOSE_WAIT)
        self.hub.close()
        with self.lock, self._write_lock:
            if self._closed:
                return
            self._closed = True
            self._writer.close()
        while not self._readers.empty():
            self._readers.get_nowait().close()
        self._lock_file.close()
        _forget_readers(str(self.db_path))
        self.origin.close()

    def _connect(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.db_path, check_same_thread=False, isolation_level=None, timeout=60)
        c.execute(f"PRAGMA page_size={PAGE_SIZE}")
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA synchronous=NORMAL")
        return c

    def reader(self) -> sqlite3.Connection:
        """Pooled read connection; return it with `release`."""
        try:
            return self._readers.get_nowait()
        except queue.Empty:
            return self._connect()

    def release(self, c: sqlite3.Connection) -> None:
        if self._closed:
            c.close()
        else:
            self._readers.put(c)

    # ---- what the origin writes ----

    def apply(self, updates: Sequence[Update]) -> None:
        """Commit updates in one transaction, then publish their events in order; nothing once closed."""
        staged: dict[str, RunRecord] = {}
        events: list[tuple[str, list[Event]]] = []
        now = time.time()
        with self._write_lock:
            if self._closed:
                return
            self._writer.execute("BEGIN IMMEDIATE")
            try:
                for r in updates:
                    st, ev = self._stage(r, staged.get(r.path) or self.records.get(r.path), now)
                    staged[r.path] = st
                    events.append((r.path, ev))
            except BaseException:
                self._writer.execute("ROLLBACK")
                raise
            self._writer.execute("COMMIT")
        with self.lock:
            self.records.update(staged)
            self._view_gen += 1
        for path, ev in events:
            for kind, data in ev:
                if kind == "run":
                    self.hub.publish(path, "run", self.run_meta(path).wire())
                elif isinstance(data, str):
                    self.hub.publish_msg(path, sse_text(kind, data))
                else:
                    self.hub.publish(path, kind, data)

    def _stage(self, r: Update, cur: RunRecord | None, now: float) -> tuple[RunRecord, list[Event]]:
        """Write one update inside `apply`'s transaction; its record and events."""
        ev: list[Event] = []
        if r.reset and cur is not None:
            self._forget(r.path)
            ev.append(("delete", {"run": r.path}))
            cur = None
        self._writer.executemany("INSERT OR REPLACE INTO media VALUES (?,?,?,?,?,?)", [m.row() for m in r.media])
        st = _record(r, cur, now)
        was, done = cur is not None and cur.state != "running", st.state != "running"  # finished before, and now
        if (r.compiled is not None and done) or was != done:
            self._bump(st.keys)
        if r.compiled is not None:
            if r.replace:
                for t in ("levels", "metrics"):
                    self._writer.execute(f"DELETE FROM {t} WHERE path=?", (r.path,))
            self._writer.executemany("INSERT OR REPLACE INTO metrics VALUES (?,?,?,?,?,?)",
                                     [(r.path, m.key, m.span.fine, m.span.top, m.span.lo, m.span.hi) for m in r.metrics])
            self._writer.executemany("INSERT OR REPLACE INTO levels VALUES (?,?,?,?,?,?)",
                                     [(b.key, b.level, b.block, r.path, b.since, b.data) for b in r.blocks])
        self._writer.execute("INSERT OR REPLACE INTO runs VALUES (?, ?)", (r.path, dumps(st.wire())))
        return st, ev + _events(r, cur, st)

    def drop(self, path: str) -> None:
        """Forget a run the origin no longer has, and tell the stream."""
        with self._write_lock:
            if self._closed:
                return
            self._writer.execute("BEGIN IMMEDIATE")
            try:
                self._forget(path)
            except BaseException:
                self._writer.execute("ROLLBACK")
                raise
            self._writer.execute("COMMIT")
        with self.lock:
            self.records.pop(path, None)
            self._view_gen += 1
        self.hub.publish(path, "delete", {"run": path})

    def keep_folders(self, notes: Mapping[str, dict[str, JSONValue]]) -> None:
        """Hold `notes` (folder -> its notes) as the folders' notes, telling the stream of each that changed."""
        changed = [p for p in {*notes, *self.folders} if notes.get(p) != self.folders.get(p)]
        if not changed:
            return
        with self._write_lock:
            if self._closed:
                return
            self._writer.execute("BEGIN IMMEDIATE")
            self._writer.execute("DELETE FROM folders")
            self._writer.executemany("INSERT INTO folders VALUES (?, ?)", [(p, dumps(info)) for p, info in notes.items()])
            self._writer.execute("COMMIT")
        self.folders = dict(notes)
        self._view_gen += 1
        for p in sorted(changed):
            self.hub.publish(p, "folder", {"path": p, "info": notes.get(p)})

    def _bump(self, keys: Sequence[str]) -> None:
        """Note that the finished runs of `keys`, or their levels, changed."""
        for k in keys:
            self._gens[k] = self._gens.get(k, 0) + 1

    def _forget(self, path: str) -> None:
        """Delete a run's index rows inside the caller's write transaction."""
        rec = self.records.get(path)
        self._bump(rec.keys if rec else list(self._gens))
        for t in ("runs", "media", "metrics", "levels"):
            self._writer.execute(f"DELETE FROM {t} WHERE path=?", (path,))

    # ---- runs ----

    def run_meta(self, path: str) -> RunMeta:
        st = self.records[path]
        p = st.public
        return RunMeta(id=path, uid=st.uid, name=p.name or path.rsplit("/", 1)[-1], parent=path.rpartition("/")[0],
                       tags=p.tags, config=p.config, info=p.info, summary={**p.user_summary, **st.summary}, state=st.state,
                       created=p.created, updated=st.heartbeat, seq=st.seq, mseq=st.mseq, keys=st.keys,
                       compiled=st.compiled, ver=st.ver)

    def info(self) -> dict[str, object]:
        return {**self.origin.info(), "cache": self.cache_dir.name}

    def tree(self) -> list[tuple[str, RunState]]:
        with self.lock:
            return [(p, st.state) for p, st in sorted(self.records.items())]

    def runs(self, prefix: str) -> RunsView:
        with self.lock:
            metas = [self.run_meta(p) for p in sorted(p for p in self.records if in_scope(p, prefix))]
        c = self.reader()
        try:
            media = [MediaRecord(*m) for m in c.execute(
                "SELECT path, seq, step, key, kind, file FROM media WHERE ? = '' OR path = ? OR (path > ? AND path < ?) "
                "ORDER BY path, seq", (prefix, prefix, prefix + "/", prefix + "0"))]  # "0" follows "/"
        finally:
            self.release(c)
        folders = {p: v for p, v in list(self.folders.items()) if in_scope(p, prefix) or in_scope(prefix, p)}
        return RunsView(metas, media, folders)

    def runs_body(self, prefix: str) -> bytes:
        """`runs` as JSON, kept while no run, medium or folder note changes."""
        return self._memo.get(("runs", prefix), self._view_gen, lambda: _sized(dumps(self.runs(prefix).wire()).encode()))

    def run(self, path: str) -> RunView:
        """One run and its media; KeyError for a run the index does not have."""
        view = self.runs(path)
        for m in view.runs:
            if m.id == path:
                return RunView(m, [x for x in view.media if x.run == path])
        raise KeyError(path)

    def media_path(self, path: str, file: str) -> Path:
        """A run's media file; KeyError unless the origin has it."""
        return self.origin.media_file(path, file)

    # ---- the stream ----

    def rows_json(self, path: str, start: int) -> str:
        """The `rows` event JSON of the rows from `start` the origin has of a run; KeyError for an unknown run."""
        with self.lock:
            if path not in self.records:
                raise KeyError(path)
        return self.origin.rows_json(path, start)

    def backfill(self, prefix: str) -> list[bytes]:
        """`rows` events for running runs' rows beyond their levels."""
        out: list[bytes] = []
        for p, (seq, _) in self.live_seqs(prefix).items():
            st = self.records.get(p)
            if st is None or not 0 < seq - st.compiled <= ROWS_EVENT_MAX:
                continue
            with contextlib.suppress(*READ_ERRORS):
                out.append(sse_text("rows", self.origin.rows_json(p, st.compiled)))
        return out

    def messages(self, prefix: str, stop: threading.Event) -> Generator[bytes, None, None]:
        """SSE for runs under `prefix`: rows beyond their levels, then live events and a heartbeat after each quiet
        HEARTBEAT, until the subscription overflows or `stop` is set."""
        sub = self.hub.subscribe(prefix)
        try:
            yield b"".join(self.backfill(prefix))
            quiet = time.monotonic()
            while not sub.dead and not stop.is_set():
                try:
                    msgs = [sub.q.get(timeout=STOP_POLL)]
                except queue.Empty:
                    if time.monotonic() - quiet < HEARTBEAT:
                        continue
                    msgs = [sse("hb", self.live_seqs(prefix))]
                with contextlib.suppress(queue.Empty):
                    while len(msgs) < STREAM_BATCH:
                        msgs.append(sub.q.get_nowait())
                quiet = time.monotonic()
                yield b"".join(msgs)
        finally:
            self.hub.unsubscribe(sub)

    def live_seqs(self, prefix: str) -> dict[str, tuple[int, int]]:
        """{path: (rows, media)} of running runs, as far as the origin has them."""
        with self.lock:
            running = [(p, st) for p, st in self.records.items() if in_scope(p, prefix) and st.state == "running"]
        return {p: self.origin.live(p, st) for p, st in running}

    # ---- dumps for mirrors ----

    def dump(self, path: str, have: Have | None) -> Dump:
        """What a mirror holding `have` of run `path` lacks. A mirror holding another run of that path, or levels of
        another rebuild, gets every block (`replace`); else those that changed since the rows its levels hold.
        KeyError for a run the index does not have."""
        c = self.reader()
        try:
            c.execute("BEGIN")
            row = c.execute("SELECT record FROM runs WHERE path=?", (path,)).fetchone()
            if row is None:
                raise KeyError(path)
            rec = RunRecord.read(json.loads(row[0]))
            same = have is not None and have.uid == rec.uid
            replace = not (have is not None and same and have.rebuilt == rec.rebuilt)
            media = c.execute("SELECT path, seq, step, key, kind, file FROM media WHERE path=? AND seq>=? ORDER BY seq",
                              (path, have.mseq if have is not None and same else 0)).fetchall()
            metrics = c.execute("SELECT key, fine, top, lo, hi FROM metrics WHERE path=? ORDER BY key", (path,)).fetchall()
            blocks = c.execute("SELECT key, level, block, since, data FROM levels WHERE path=? AND since>?",
                               (path, -1 if have is None or replace else have.compiled)).fetchall()
            c.execute("COMMIT")
        finally:
            self.release(c)
        return Dump(rec, [MediaRecord(*m) for m in media], replace,
                    [Metric(key, bk.Span(fine, top, lo, hi)) for key, fine, top, lo, hi in metrics], [Block(*b) for b in blocks])

    def dump_many(self, held: Sequence[tuple[str, Have | None]]) -> list[bytes]:
        """`dump` of each (run, what a mirror holds of it), encoded; empty for a run the index does not have."""
        out: list[bytes] = []
        for path, have in held:
            try:
                out.append(self.dump(path, have).encode())
            except KeyError:
                out.append(b"")
        return out

    # ---- blocks ----

    def buckets_bodies(self, asks: Sequence[Ask]) -> list[bytes]:
        """`buckets_body` of each ask, in order (`at_once`)."""
        return at_once(lambda a: self.buckets_body(a.key, a.level, a.block, a.scope, a.runs, a.which), asks, for_workers)

    def buckets_body(self, key: str, level: int, index: int, scope: str = "", runs: Sequence[str] | None = None,
                     which: Which = "all") -> bytes:
        """Block `index` of `level` of `key` as a bucket array (`trex.buckets`): of the runs `runs`, or else of the runs
        under `scope` in state `which`; each run's buckets hold its compiled rows. A scope's finished runs' blocks are
        kept in memory while those runs' levels stay the same."""
        if not bk.MIN_LEVEL <= level <= bk.MAX_LEVEL:
            raise ValueError(f"level {level} out of range")
        if runs is None and which == "finished":
            return self._memo.get(("finished", key, level, index, scope), self._gens.get(key, 0),
                                  lambda: _sized(self._block(key, level, index, self._chosen(key, scope, None, which))))
        return self._block(key, level, index, self._chosen(key, scope, runs, which))

    def _chosen(self, key: str, scope: str, runs: Sequence[str] | None, which: Which) -> list[str]:
        """The runs a block request names that log `key`, in path order."""
        with self.lock:
            if runs is not None:
                return sorted({p for p in runs if (r := self.records.get(p)) is not None and key in r.keys})
            return sorted(p for p, r in self.records.items() if in_scope(p, scope) and key in r.keys
                          and (which == "all" or (r.state == "running") == (which == "running")))

    def _block(self, key: str, level: int, index: int, paths: list[str]) -> bytes:
        """Block `index` of `level` of `key` of runs `paths` (in path order). A run whose top level is `level` or finer
        is merged from top levels (`_from_tops`); any other run's block is the one stored (`stored_block`), and when
        every run's is, those are the answer as they are."""
        parts, deep = self._from_tops(key, level, index, paths)
        seq = self._compiled(paths)
        if deep:
            body = _stored(str(self.db_path), key, level, index, [paths[i] for i in deep], seq[deep].astype("<u4").tobytes())
            if len(deep) == len(paths):
                return body
            b = bk.decode(body).buckets
            parts.append(b.of(np.array(deep, np.int32)[b.run]))
        return bk.encode(level, index, paths, seq, bk.union(parts))

    def _from_tops(self, key: str, level: int, index: int, paths: list[str]) -> tuple[list[Buckets], list[int]]:
        """Block `index` of `level` of `key` of the runs of `paths` whose top level is `level` or finer, as buckets of their
        places in `paths`: finished ones cut from the finished runs' merged level (`_level`), running ones merged from
        their top levels. And the places of the other runs, in order."""
        lo, hi = index * bk.BLOCK, (index + 1) * bk.BLOCK
        done, tops, _ = self._runs(key)
        positions = self._positions(key)
        at = np.array([positions.get(p, -1) for p in paths], np.int64)  # each run's index in `done`, or -1
        finished = at >= 0
        coarse = finished.copy()
        coarse[finished] = tops[at[finished]] <= level
        parts: list[Buckets] = []
        if coarse.any():
            out = np.full(len(done), -1, np.int32)
            out[at[coarse]] = np.flatnonzero(coarse)
            part = bk.cut(self._level(key, level), lo, hi, out >= 0)
            parts.append(part.of(out[part.run]))
        others = np.flatnonzero(~finished)
        deep = [int(i) for i in np.flatnonzero(finished & ~coarse)]
        if others.size:
            st = self._tops(key, [paths[i] for i in others])
            near = st.level <= level
            part = bk.cut(st.at(level, np.flatnonzero(near).astype(np.int32)), lo, hi)
            parts.append(part.of(others[part.run].astype(np.int32)))
            deep += [int(i) for i in others[~near]]
        return parts, sorted(deep)

    def _tops(self, key: str, paths: Sequence[str]) -> Stack:
        """The top levels of `key` of runs `paths` (in path order), read from the index."""
        tops: dict[str, int] = {}
        rows: list[tuple[str, bytes]] = []
        c = self.reader()
        try:
            for part in batched(paths, PATHS_PER_QUERY):
                tops.update(c.execute(f"SELECT path, top FROM metrics WHERE key=? AND path IN ({_marks(part)})", (key, *part)))
                rows += c.execute("SELECT l.path, l.data FROM levels l JOIN metrics m ON m.key=l.key AND m.path=l.path AND m.top=l.level "
                                  f"WHERE l.key=? AND l.path IN ({_marks(part)}) ORDER BY l.path, l.block", (key, *part)).fetchall()
        finally:
            self.release(c)
        return self._stacked(paths, tops, rows)

    def _stacked(self, paths: Sequence[str], tops: Mapping[str, int], rows: Sequence[tuple[str, bytes]]) -> Stack:
        """The Stack of runs `paths` at their `tops`, of their top levels' stored blocks `rows` (path, data), in path then
        block order."""
        at = {p: i for i, p in enumerate(paths)}
        level = np.array([tops.get(p, bk.MIN_LEVEL) for p in paths], np.int8)
        return bk.stack(paths, self._compiled(paths), level, [zlib.decompress(data) for _, data in rows], [at[p] for p, _ in rows])

    def _compiled(self, paths: Sequence[str]) -> npt.NDArray[np.uint32]:
        """The rows each run's levels hold."""
        with self.lock:
            return np.array([st.compiled if (st := self.records.get(p)) else 0 for p in paths], np.uint32)

    def _runs(self, key: str) -> tuple[list[str], npt.NDArray[np.int8], npt.NDArray[np.uint32]]:
        """The finished runs of `key` in path order, each one's top level and the rows its levels hold: from their saved
        levels when those are current, else from their stack."""
        saved = self._saved(key)
        if saved is not None:
            fin = self._finished(key)
            return fin.paths, np.load(saved / "levels.npy", mmap_mode="r"), fin.seq
        st = self._stack(key)
        return st.paths, st.level, st.seq

    def _positions(self, key: str) -> dict[str, int]:
        """Each finished run of `key` by its index in path order."""
        return self._memo.get(("positions", key), self._gens.get(key, 0),
                              lambda: ({p: i for i, p in enumerate(self._finished(key).paths)}, 0))

    def _level(self, key: str, level: int) -> Buckets:
        """The buckets of every finished run of `key` whose top level is `level` or finer, merged to `level` (on
        threads); kept as stacks are."""
        return self._memo.get(("level", key, level), self._gens.get(key, 0), lambda: self._merge_level(key, level))

    def _merge_level(self, key: str, level: int) -> tuple[Buckets, int]:
        saved = self._saved(key)
        if saved is not None and (saved / f"L{level}-run.npy").exists():
            part = Buckets(*(np.load(saved / f"L{level}-{name}.npy", mmap_mode="r") for name in bk.COLUMNS))
            return part, part.run.nbytes
        st = self._stack(key)
        part = self._merged(st, level, st.level <= level)
        return part, part.nbytes

    def _merged(self, st: Stack, level: int, take: npt.NDArray[np.bool_]) -> Buckets:
        """The buckets of the runs `take` marks merged to `level`, on threads over runs in order (numpy lets go of the
        GIL)."""
        b = st.buckets
        cuts = np.searchsorted(b.run, np.linspace(0, len(st.paths), MERGE_THREADS + 1)[1:-1].astype(np.int32))
        bounds = [0, *cuts.tolist(), b.run.size]

        def one(lo: int, hi: int) -> Buckets:
            part = bk.Buckets(b.run[lo:hi], b.bucket[lo:hi], b.mean[lo:hi], b.soff[lo:hi], b.tmean[lo:hi], b.n[lo:hi])
            part = bk.select(part, take[part.run])
            return bk.merge(part, level - st.level[part.run].astype(np.int64))

        with ThreadPoolExecutor(max_workers=MERGE_THREADS) as pool:
            parts = list(pool.map(one, bounds, bounds[1:]))
        return bk.concat(parts)

    def _stack(self, key: str) -> Stack:
        """The finished runs' top levels of `key` (`buckets.stack`, runs in path order); kept while they stay the same.
        A new stack gets its levels from the coarsest a first view takes to the finest it keeps merged and saved, on
        a thread of its own."""
        return self._memo.get(("stack", key), self._gens.get(key, 0), lambda: self._read_stack(key))

    def _read_stack(self, key: str) -> tuple[Stack, int]:
        fin = self._finished(key)
        done, sig = fin.paths, fin.sig
        c = self.reader()
        try:
            tops: dict[str, int] = dict(c.execute("SELECT path, top FROM metrics WHERE key=?", (key,)))
            rows = c.execute("SELECT l.path, l.data FROM levels l JOIN metrics m ON m.key=l.key AND m.path=l.path AND m.top=l.level "
                             "WHERE l.key=? ORDER BY l.path, l.block", (key,)).fetchall()
        finally:
            self.release(c)
        held = set(done)
        st = self._stacked(done, tops, [r for r in rows if r[0] in held])
        if done:
            top = int(st.level.max())
            levels = range(top + LEVELS_AHEAD, top - 1, -1)  # coarsest first, as views ask
            threading.Thread(target=self._merge_ahead, args=(key, sig, st, levels), name="trex-levels", daemon=True).start()
        return st, st.buckets.nbytes

    def _merge_ahead(self, key: str, sig: bytes, st: Stack, levels: range) -> None:
        """Merge `levels` of `key` and save them (`_save_levels`) while its finished runs stay those of `sig`, at most
        once every LEVELS_SAVE_EVERY seconds."""
        parts = {x: self._level(key, x) for x in levels}
        due = time.monotonic() - self._saved_at.get(key, -math.inf) >= LEVELS_SAVE_EVERY
        if due and self._finished(key).sig == sig and self._saved(key) is None:
            self._saved_at[key] = time.monotonic()
            self._save_levels(key, sig, st, parts)

    def _finished(self, key: str) -> Finished:
        """The finished runs logging `key` (`Finished`)."""
        return self._memo.get(("finished", key), self._gens.get(key, 0), lambda: (self._digest(key), 0))

    def _digest(self, key: str) -> Finished:
        with self.lock:
            done = [(p, rec.compiled, rec.ver) for p, rec in self.records.items() if rec.state != "running" and key in rec.keys]
        done.sort()
        paths = [p for p, _, _ in done]
        h = hashlib.sha1(f"{CACHE_VERSION}\0{key}\0".encode())
        h.update("\0".join(paths).encode())
        h.update(np.array([(seq, ver) for _, seq, ver in done], np.int64).tobytes())
        return Finished(paths, np.array([seq for _, seq, _ in done], np.uint32), h.digest())

    def _levels_dir(self, key: str, sig: bytes) -> Path:
        return self.cache_dir / "levels" / f"{hashlib.sha1(key.encode()).hexdigest()[:20]}-{sig.hex()[:20]}"

    def _saved(self, key: str) -> Path | None:
        """The directory of the merged levels of `key` an earlier build saved (`_save_levels`) for its finished runs and
        their levels as they are."""
        return self._memo.get(("saved", key), self._gens.get(key, 0), lambda: (self._open_saved(key), 0))

    def _open_saved(self, key: str) -> Path | None:
        d = self._levels_dir(key, self._finished(key).sig)
        try:
            os.utime(d / "levels.npy")
            return d
        except OSError:
            return None

    def _save_levels(self, key: str, sig: bytes, st: Stack, parts: Mapping[int, Buckets]) -> None:
        """Write merged levels of `key` beside the index as numpy arrays (`.npy`, memory-mapped when read): each run's
        top level (`levels`) and each level's buckets (`L<level>-<field>`); then delete its levels saved for other runs
        and the least recently used saved levels beyond LEVELS_BYTES."""
        d = self._levels_dir(key, sig)
        tmp = d.with_name(f"{d.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        try:
            tmp.mkdir(parents=True)
            for lv, b in parts.items():
                for name, a in zip(bk.COLUMNS, b.columns, strict=True):
                    np.save(tmp / f"L{lv}-{name}.npy", a)
            np.save(tmp / "levels.npy", st.level)  # last: a directory with it is complete
            tmp.rename(d)
        except OSError:
            shutil.rmtree(tmp, ignore_errors=True)
            return
        for old in d.parent.glob(f"{d.name.split('-')[0]}-*"):
            if old != d:
                shutil.rmtree(old, ignore_errors=True)
        _bound_dir(d.parent, LEVELS_BYTES)


def _free_index(base: Path) -> tuple[Path, TextIO]:
    """The first of `base`, `base`-1, `base`-2, ... that no other Explorer holds, and its held lock file."""
    for slot in itertools.count():
        d = base if slot == 0 else base.with_name(f"{base.name}-{slot}")
        d.mkdir(parents=True, exist_ok=True)
        lock = open(d / "lock", "w")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock.close()
            continue
        except OSError:
            pass  # a filesystem without locks
        return d, lock
    raise AssertionError("unreachable")


def _marks(part: Sequence[str]) -> str:
    return ",".join("?" * len(part))
