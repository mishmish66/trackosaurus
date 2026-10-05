"""Explorer index of a runs directory, in <cache_root>/<hash of root>/index.sqlite: run metadata, last
values, each run's kept buckets of every metric (`trex.buckets`) and media; beside it, in levels/, the finished runs' buckets merged per level. A run is its path relative to the root.
Runs are scanned inline or in worker processes; the main process commits, then publishes events in
order per run.
"""

import contextlib
import hashlib
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
from collections import OrderedDict, deque
from collections.abc import Callable, Generator, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, ThreadPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from multiprocessing.context import BaseContext
from pathlib import Path
from typing import Final, Literal, NamedTuple, Self, TypedDict, cast

import numpy as np
import numpy.typing as npt

from . import buckets as bk, chunks
from .buckets import Buckets, Stack
from .journal import JOURNAL
from .format import (DB, INFO_FILE, JSONValue, MediaKind, RunState, as_dict, as_float, as_run_state, as_str, as_str_list,
                     snapshot)

type Sig = list[int]
"""[db mtime_ns, db size, wal mtime_ns, wal size]; 0s for a missing file."""

type Summary = dict[str, float | str]
"""Last value of every metric (non-finite as strings), plus _step and _runtime."""

type Which = Literal["all", "finished", "running"]


class Ask(NamedTuple):
    """Block `block` of `level` of `key`: of the runs `runs` (ids), or else of the runs under `scope` in state `which`."""

    key: str
    level: int
    block: int
    scope: str = ""
    runs: Sequence[str] | None = None
    which: Which = "all"
"""The runs of a scope a block request takes: all, or the finished or the running ones."""
type Event = tuple[str, object]
"""(SSE event name, data); data that is a str is already JSON."""


class MediaRecord(NamedTuple):
    """A media item of a run."""

    run: str
    seq: int
    step: float
    key: str
    kind: MediaKind
    file: str


class KeptRecord(NamedTuple):
    """A metric of a run as the index keeps it: a one-run bucket array at its kept level."""

    key: str
    level: int
    data: bytes


class Compiled(NamedTuple):
    """A metric of a run at every level below its kept one (`buckets.pyramid`): the finest level, and (level, block,
    compressed one-run bucket array) of each block holding buckets."""

    key: str
    fine: int
    blocks: list[tuple[int, int, bytes]]


class Public(TypedDict):
    """Metadata from the run file."""

    name: str | None
    tags: list[str]
    config: dict[str, JSONValue]
    created: float | None
    info: dict[str, JSONValue]
    user_summary: dict[str, JSONValue]
    state: RunState


class Prev(TypedDict):
    """The index's state of a run when its scan starts."""

    uid: str
    seq: int
    mseq: int
    kept_seq: int
    kept_t: float
    kept_state: RunState | None
    pyramid_seq: int
    pyramid_t: float


class Job(TypedDict):
    """A run to scan and how: `want_rows` when a browser watches it, `kept_refresh` and `pyramid_refresh` seconds
    between rebuilds of a growing run's kept buckets and compiles of its finer levels."""

    path: str
    dir: str
    sig: Sig
    prev: Prev | None
    crash_after: float
    kept_refresh: float
    pyramid_refresh: float
    want_rows: bool


class ScanResult(TypedDict):
    """What changed in a run since its `Prev`."""

    path: str
    sig: Sig
    uid: str
    reset: bool
    fresh: bool
    seq: int
    mseq: int
    media: list[MediaRecord]
    state: RunState
    heartbeat: float | None
    public: Public
    keys: list[str]
    summary: Summary | None
    kept: list[KeptRecord] | None
    kept_seq: int  # the rows `kept` holds
    pyramid: list[Compiled] | None
    pyramid_seq: int  # the rows `pyramid` holds
    rows: str | None


class RunRecord(TypedDict):
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
    kept_seq: int
    kept_t: float
    kept_state: RunState | None
    pyramid_seq: int
    pyramid_t: float


class Have(TypedDict):
    """What a mirror holds of a run: its uid, media items, and the rows its kept buckets and compiled levels hold."""

    uid: str
    mseq: int
    kept_seq: int
    pyramid_seq: int


class RunMeta(TypedDict):
    """A run, as sent to the browser."""

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
    kept_seq: int
    pyramid_seq: int


class RunsView(TypedDict):
    """Runs under a folder, their media, and the notes of related folders."""

    runs: list[RunMeta]
    media: list[MediaRecord]
    folders: dict[str, dict[str, JSONValue]]


class RunView(TypedDict):
    run: RunMeta
    media: list[MediaRecord]

CACHE_VERSION: Final = 14  # bump whenever what the index stores changes; older caches are rebuilt
CRASH_AFTER = 300.0  # seconds without a heartbeat after which a running run shows as crashed
POLL: Final = 1.0  # seconds between polls of known runs
REWALK: Final = 3.0  # seconds between walks of the root for new and removed runs
KEPT_REFRESH = float(os.environ.get("TREX_KEPT_REFRESH", "10"))  # seconds between rebuilds of a growing run's kept buckets
PYRAMID_REFRESH = float(os.environ.get("TREX_PYRAMID_REFRESH", "300"))  # seconds between compiles of a growing run's finer levels
SKIP_DIRS: Final = frozenset({"node_modules", "__pycache__"})
INLINE_BYTES = 5 << 20  # polls whose run files grew by at most this many bytes are read in the main process
BATCH_RUNS: Final = 256  # scan results per index transaction
BATCH_SECONDS: Final = 0.5  # longest wait before committing a partial batch
CLOSE_WAIT: Final = 5.0  # longest `close` waits for a scan in progress
MEMO_BYTES = 1 << 30  # stacks, merged levels and answers an Explorer keeps in memory, least recently used dropped
LEVELS_BYTES = int(os.environ.get("TREX_LEVELS_MB", "4096")) << 20  # saved merged levels an index keeps, least recently used deleted
LEVELS_SAVE_EVERY = 60.0  # seconds between saves of one metric's merged levels
LEVELS_AHEAD: Final = 3  # levels above a metric's coarsest kept level merged and saved ahead of requests
INLINE_BUILDS: Final = 64  # blocks a request builds from run files without the process pool
MERGE_THREADS: Final = min(8, os.cpu_count() or 1)  # threads merging one level
PAGE_SIZE: Final = 16384
READ_ERRORS: Final = (sqlite3.Error, OSError, ValueError, KeyError, struct.error)
ROWS_EVENT_MAX: Final = 20_000  # larger catch-ups reach browsers through new kept buckets instead of rows
HEARTBEAT: Final = 10.0  # seconds of stream silence after which a heartbeat is sent
STREAM_BATCH: Final = 2000  # events sent together at most
STOP_POLL: Final = 0.5  # seconds between a stream's checks of its stop event
NO_BUCKETS: Final = bk.encode(bk.MIN_LEVEL, 0, [""], [0], bk.empty())  # a run's bucket array where it has none

TABLES: Final = {
    "cache": "CREATE TABLE IF NOT EXISTS cache(key TEXT PRIMARY KEY, value TEXT)",
    "runs": "CREATE TABLE IF NOT EXISTS runs(path TEXT PRIMARY KEY, record TEXT NOT NULL)",
    "media": "CREATE TABLE IF NOT EXISTS media(path TEXT, seq INTEGER, step REAL, key TEXT, kind TEXT, file TEXT, "
             "PRIMARY KEY(path, seq)) WITHOUT ROWID",
    # a run's buckets of a metric at the level it keeps them at (a one-run bucket array); seq: the rows they hold
    "kept": "CREATE TABLE IF NOT EXISTS kept(path TEXT, key TEXT, level INTEGER, seq INTEGER, data BLOB, PRIMARY KEY(key, path))",
    "kept_path": "CREATE INDEX IF NOT EXISTS kept_path ON kept(path)",
    # a run's buckets of a metric at every level below its kept one (compressed one-run bucket arrays, those holding
    # buckets), and per run and metric the finest of those levels and the rows they hold
    "pyramid": "CREATE TABLE IF NOT EXISTS pyramid(key TEXT, level INTEGER, block INTEGER, path TEXT, data BLOB, "
               "PRIMARY KEY(key, level, block, path))",
    "pyramid_path": "CREATE INDEX IF NOT EXISTS pyramid_path ON pyramid(path)",
    "compiled": "CREATE TABLE IF NOT EXISTS compiled(path TEXT, key TEXT, fine INTEGER, seq INTEGER, PRIMARY KEY(key, path))",
    # a mirror's folder notes (an Explorer reads its own from trex_info.json files)
    "folders": "CREATE TABLE IF NOT EXISTS folders(path TEXT PRIMARY KEY, info TEXT NOT NULL)",
}


def dumps(obj: object) -> str:
    """Compact JSON (non-finite floats raise; see `wire`)."""
    return json.dumps(obj, separators=(",", ":"), allow_nan=False)


def wire(d: Mapping[str, float]) -> dict[str, float | str]:
    """Metrics for JSON: non-finite values as "nan", "inf", "-inf"."""
    return {k: v if math.isfinite(v) else str(v) for k, v in d.items()}


def sse(event: str, data: object) -> bytes:
    return sse_text(event, dumps(data))


def sse_text(event: str, text: str) -> bytes:
    return f"event: {event}\ndata: {text}\n\n".encode()


def in_scope(path: str, prefix: str) -> bool:
    """Whether `path` is `prefix` or below it ("" is the root)."""
    return not prefix or path == prefix or path.startswith(prefix + "/")


class Subscriber:
    """Events for runs under `prefix`, queued until sent; dead once the queue overflows."""

    def __init__(self, prefix: str, maxsize: int = 20000) -> None:
        self.prefix = prefix
        self.q: queue.Queue[bytes] = queue.Queue(maxsize)
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


def _stat_sig(d: str | os.PathLike[str]) -> Sig:
    """mtime and size of the database, its WAL and its journal; the journal's opened, which makes a network
    filesystem report its current size."""
    sig: Sig = []
    for name in (DB, DB + "-wal", JOURNAL):
        try:
            if name == JOURNAL:
                with open(Path(d) / name, "rb") as f:
                    st = os.fstat(f.fileno())
            else:
                st = os.stat(Path(d) / name)
            sig += [st.st_mtime_ns, st.st_size]
        except FileNotFoundError:
            sig += [0, 0]
    return sig


# ---- per-run reading (inline or in a worker process) ----


def _last_values(c: sqlite3.Connection, names: Mapping[int, str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for kid, name in names.items():
        r = c.execute("SELECT data FROM chunk WHERE key_id = ? ORDER BY seq0 DESC LIMIT 1", (kid,)).fetchone()
        if r:
            vals = chunks.decode(r[0])[1]
            if len(vals):
                out[name] = float(vals[-1])
    return out


def _compile(c: sqlite3.Connection, names: Mapping[int, str], stop: int, kept: bool,
             finer: bool) -> tuple[list[KeptRecord] | None, list[Compiled] | None]:
    """Every metric's kept buckets over rows [0, stop) when `kept`, and its levels below them when `finer`; each metric
    read once."""
    ks: list[KeptRecord] = []
    ps: list[Compiled] = []
    for kid, name in sorted(names.items(), key=lambda kv: kv[1]) if kept or finer else []:
        s, v, t = chunks.metric(c, kid, stop=stop)
        if not s.size:
            continue
        if kept:
            ks.append(KeptRecord(name, bk.level_for(float(s.max() - s.min())), bk.kept(s, v, t, stop)))
        if finer:
            fine, blocks = bk.pyramid(s, v, t, stop)
            ps.append(Compiled(name, fine, [(level, i, zlib.compress(blob, 1)) for level, i, blob in blocks]))
    return (ks if kept else None), (ps if finer else None)


def scan(job: Job) -> ScanResult | None:
    """What changed in a run, or None to retry later (no id yet, or changed while read without a WAL)."""
    path, d, sig, prev = job["path"], Path(job["dir"]), job["sig"], job["prev"]
    with snapshot(d) as c:
        meta: dict[str, JSONValue] = {k: json.loads(v) for k, v in c.execute("SELECT key, value FROM meta")}
        if "id" not in meta:
            return None
        state, heartbeat = _state(meta, job["crash_after"])
        seq = chunks.row_count(c)
        media_count: int = c.execute("SELECT coalesce(max(seq) + 1, 0) FROM media").fetchone()[0]
        reset = prev is not None and _rewritten(prev, meta["id"], seq, media_count)
        prev = None if reset else prev
        names = chunks.key_names(c)
        summary = _summary(c, names, seq) if prev is None or seq != prev["seq"] else None
        kept, pyramid = _compile(c, names, seq, _kept_due(prev, seq, state, job["kept_refresh"]),
                                 _pyramid_due(prev, seq, state, job["pyramid_refresh"]))
        text = (_rows_event(path, prev["seq"], chunks.rows(c, prev["seq"], seq))
                if prev and _rows_wanted(job, prev, seq) else None)
        mseq = prev["mseq"] if prev else 0
        media = _new_media(c, d, path, mseq)
    after = _stat_sig(d)
    if after != sig and not (sig[2] and after[2]):
        return None
    return {"path": path, "sig": sig, "uid": str(meta["id"]), "reset": reset, "fresh": prev is None,
            "seq": seq, "mseq": mseq + len(media), "media": media,
            "state": state, "heartbeat": heartbeat, "public": _public(meta, state), "keys": sorted(names.values()),
            "summary": summary, "kept": kept, "kept_seq": seq, "pyramid": pyramid, "pyramid_seq": seq, "rows": text}


def _state(meta: Mapping[str, JSONValue], crash_after: float) -> tuple[RunState, float | None]:
    """(state, heartbeat); a running run silent for `crash_after` seconds is crashed."""
    state, heartbeat = as_run_state(meta.get("state")), as_float(meta.get("heartbeat"))
    if state == "running" and time.time() - (heartbeat or 0) > crash_after:
        state = "crashed"
    return state, heartbeat


def _summary(c: sqlite3.Connection, names: Mapping[int, str], seq: int) -> Summary:
    out = wire(_last_values(c, names))
    if (last := chunks.last_step_and_time(c, seq)) is not None:
        out["_step"], out["_runtime"] = last
    return out


def _kept_due(prev: Prev | None, seq: int, state: RunState, refresh: float) -> bool:
    """Kept buckets are rebuilt for a new run, and for a changed one when it stopped running or `refresh` passed."""
    if prev is None:
        return True
    stale = prev["kept_seq"] != seq or prev["kept_state"] != state
    return stale and (state != "running" or time.time() - prev["kept_t"] >= refresh)


def _pyramid_due(prev: Prev | None, seq: int, state: RunState, refresh: float) -> bool:
    """Finer levels are compiled for a new run, and for a changed one when it stopped running or `refresh` passed."""
    if prev is None:
        return True
    return prev["pyramid_seq"] != seq and (state != "running" or time.time() - prev["pyramid_t"] >= refresh)


def _refined(data: bytes | None, fine: int, level: int, index: int, seq: int) -> bytes:
    """Block `index` of `level` from compressed block `data` of level `fine` (or none): as it is when `fine` is `level`,
    else its buckets refined to `level` and cut to the block."""
    if data is None:
        return bk.encode(level, index, [""], [seq], bk.empty())
    blob = zlib.decompress(data)
    if fine == level:
        return blob
    b = bk.cut(bk.refine(bk.decode(blob).buckets, fine, level), index * bk.BLOCK, (index + 1) * bk.BLOCK)
    return bk.encode(level, index, [""], [seq], b)


def _lacking(have: Have | None, rec: RunRecord) -> tuple[int, bool, bool]:
    """What a mirror holding `have` of a run lacks: (its first media item to send, whether to send the kept buckets,
    whether to send the compiled levels); everything when it holds another uid."""
    if have is None or have["uid"] != rec["uid"]:
        return 0, True, True
    return have["mseq"], have["kept_seq"] != rec["kept_seq"], have["pyramid_seq"] != rec["pyramid_seq"]


def _rewritten(prev: Prev, uid: JSONValue, rows: int, media: int) -> bool:
    """The run file was replaced: a new id, or fewer rows or media than indexed."""
    return prev["uid"] != uid or rows < prev["seq"] or media < prev["mseq"]


def _rows_wanted(job: Job, prev: Prev, seq: int) -> bool:
    """A rows event goes to watching browsers, unless the catch-up is large (kept buckets serve it)."""
    return job["want_rows"] and 0 < seq - prev["seq"] <= ROWS_EVENT_MAX


def _rows_event(path: str, start: int, rows: Sequence[chunks.Row]) -> str:
    """JSON of the `rows` event for `rows` of run `path`, the first of them row `start`; `run` comes first."""
    text = ",".join(dumps([r.step, r.t, wire(r.values)]) for r in rows)
    return f'{{"run":{dumps(path)},"seq0":{start},"rows":[{text}]}}'


def _new_media(c: sqlite3.Connection, d: Path, path: str, mseq: int) -> list[MediaRecord]:
    """Media from `mseq` on, up to a gap or a file not yet written."""
    out: list[MediaRecord] = []
    for i, step, key, kind, file in c.execute("SELECT seq, step, key, kind, file FROM media WHERE seq >= ? ORDER BY seq", (mseq,)):
        if i != mseq + len(out):
            break
        if not (d / file).is_file():
            break
        out.append(MediaRecord(path, i, step, key, kind, file))
    return out


def _public(meta: Mapping[str, JSONValue], state: RunState) -> Public:
    return {"name": as_str(meta.get("name")), "tags": as_str_list(meta.get("tags")), "config": as_dict(meta.get("config")),
            "created": as_float(meta.get("created")), "info": as_dict(meta.get("info")),
            "user_summary": as_dict(meta.get("summary")), "state": state}


class _Batch:
    """Scan results awaiting one index transaction."""

    def __init__(self, ex: "Explorer") -> None:
        self.ex = ex
        self.results: list[ScanResult] = []
        self.t0 = time.time()

    def add(self, r: ScanResult | None) -> None:
        if r is not None:
            self.results.append(r)

    def full(self) -> bool:
        return len(self.results) >= BATCH_RUNS

    def commit(self) -> None:
        if self.results:
            self.ex.apply(self.results)
        self.results, self.t0 = [], time.time()


def _worker_init() -> None:
    """Index workers leave Ctrl-C to the main process, which stops them."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _mp_context() -> BaseContext:
    """Fresh worker interpreters that import only trex (the main process runs threads)."""
    if "forkserver" in multiprocessing.get_all_start_methods():
        ctx = multiprocessing.get_context("forkserver")
        ctx.set_forkserver_preload([__name__])
        return ctx
    return multiprocessing.get_context("spawn")


def build_block(run_dir: str, key: str, level: int, index: int) -> tuple[bytes, int]:
    """(block `index` of `level` of `key`, as a one-run bucket array, and the rows it holds) from the run file in
    `run_dir`."""
    lo, hi = bk.block_range(level, index)
    with snapshot(Path(run_dir)) as c:
        kid = c.execute("SELECT id FROM keys WHERE name=?", (key,)).fetchone()
        stop = chunks.row_count(c)
        if kid is None:
            return bk.encode(level, index, [""], [stop], bk.empty()), stop
        s, v, t = chunks.metric(c, kid[0], stop=stop, step_lo=lo, step_hi=hi)
    return bk.built(s, v, t, stop, level, index), stop


def _build_block_job(job: tuple[str, str, int, int]) -> tuple[bytes, int]:
    return build_block(*job)


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


def default_workers() -> int:
    """$TREX_WORKERS, else the CPU count up to 32."""
    env = os.environ.get("TREX_WORKERS")
    return max(1, int(env)) if env else min(32, os.cpu_count() or 1)


def _events(r: ScanResult, cur: RunRecord | None, st: RunRecord) -> list[Event]:
    """A scan result's events: a run event follows the rows and media it counts; a new run's comes first."""
    rows: list[Event] = [("rows", r["rows"])] if r["rows"] is not None else []
    media: list[Event] = [("media", m) for m in r["media"]]
    if cur is None:
        return [("run", None), *rows, *media]
    changed = r["fresh"] or r["kept"] is not None or r["public"] != cur["public"] or st["keys"] != cur["keys"]
    return [*rows, *media, *([("run", None)] if changed else [])]


def _needs_scan(st: RunRecord, sig: Sig, now: float) -> bool:
    """Its files changed, it went silent while running, or its kept buckets are due a refresh."""
    silent = st["state"] == "running" and now - (st["heartbeat"] or now) > CRASH_AFTER
    due = st["kept_seq"] != st["seq"] and now - st["kept_t"] >= KEPT_REFRESH
    return sig != st["sig"] or silent or due


def _record(r: ScanResult, cur: RunRecord | None) -> RunRecord:
    """The run's record after scan result `r`; kept-bucket fields carry over from `cur`."""
    return {"uid": r["uid"], "seq": r["seq"], "mseq": r["mseq"], "keys": r["keys"],
            "summary": r["summary"] if r["summary"] is not None else cur["summary"] if cur else {},
            "sig": r["sig"], "heartbeat": r["heartbeat"], "public": r["public"], "state": r["state"],
            "kept_seq": cur["kept_seq"] if cur else -1, "kept_t": cur["kept_t"] if cur else 0.0,
            "kept_state": cur["kept_state"] if cur else None, "pyramid_seq": cur["pyramid_seq"] if cur else -1,
            "pyramid_t": cur["pyramid_t"] if cur else 0.0}


def _bytes_of(sig: Sig) -> int:
    return sig[1] + sig[3]


class Finished(NamedTuple):
    """The finished runs logging a metric, in path order, the rows of each its kept buckets hold, and a digest of them
    and their kept buckets."""

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


class Explorer:
    """Index of a runs directory, kept current by `start` (or `poll_forever`); `close` releases it."""

    def __init__(self, root: str | os.PathLike[str], cache_root: str | os.PathLike[str], workers: int | None = None) -> None:
        self.root = Path(root).resolve()
        self.workers = max(1, int(workers)) if workers is not None else default_workers()
        self.cache_dir = Path(cache_root) / hashlib.sha1(str(self.root).encode()).hexdigest()[:12]
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "root.txt").write_text(str(self.root) + "\n")
        self.db_path = self.cache_dir / "index.sqlite"
        self._writer = self._connect()
        for sql in TABLES.values():
            self._writer.execute(sql)
        v = self._writer.execute("SELECT value FROM cache WHERE key='version'").fetchone()
        if v is None or int(v[0]) != CACHE_VERSION:
            self._writer.execute("BEGIN IMMEDIATE")
            for t in [r[0] for r in self._writer.execute("SELECT name FROM sqlite_master WHERE type='table' AND name != 'cache'")]:
                self._writer.execute(f"DROP TABLE {t}")
            for name, sql in TABLES.items():
                if name != "cache":
                    self._writer.execute(sql)
            self._writer.execute("INSERT OR REPLACE INTO cache VALUES ('version', ?)", (str(CACHE_VERSION),))
            self._writer.execute("COMMIT")
            self._writer.execute("VACUUM")  # the dropped tables' pages go back to the disk
            shutil.rmtree(self.cache_dir / "levels", ignore_errors=True)
        self._write_lock = threading.Lock()  # serializes every use of `_writer`
        self.lock = threading.Lock()
        self.hub = Hub()
        self.records: dict[str, RunRecord] = {path: json.loads(s) for path, s in self._writer.execute("SELECT path, record FROM runs")}
        self.dirs: dict[str, Path] = {}
        self.folders: dict[str, tuple[int, dict[str, JSONValue]]] = {}  # folder path -> (mtime_ns, notes) of trex_info.json files
        self._readers: queue.LifoQueue[sqlite3.Connection] = queue.LifoQueue()
        self._stop = threading.Event()
        self._poller: threading.Thread | None = None
        self._closed = False
        self.ready = threading.Event()
        self._gens: dict[str, int] = {}  # metric -> bumped whenever its finished runs, or their kept buckets, change
        self._memo = Memo(MEMO_BYTES)  # stacks, merged levels, block answers and `runs` answers
        self._saved_at: dict[str, float] = {}  # metric -> when its merged levels were last saved (monotonic)
        self._view_gen = 0  # bumped whenever what `runs` answers may change

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def start(self) -> Self:
        """Poll in a background thread until `close`."""
        self._poller = threading.Thread(target=self.poll_forever, name=f"trex-poll-{self.root.name}", daemon=True)
        self._poller.start()
        return self

    def close(self) -> None:
        """Stop polling, end subscriptions and close the index's connections."""
        self.stop()
        if self._poller is not None and self._poller is not threading.current_thread():
            self._poller.join(CLOSE_WAIT)
        self.hub.close()
        with self.lock, self._write_lock:
            self._closed = True
            self._writer.close()
        while not self._readers.empty():
            self._readers.get_nowait().close()

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

    # ---- crawling ----

    def walk(self) -> tuple[dict[str, Path], dict[str, Path]]:
        """(run directories, trex_info.json files) by path relative to the root; never descends into a run."""
        found: dict[str, Path] = {}
        infos: dict[str, Path] = {}
        stack = [self.root]
        while stack:
            d = stack.pop()
            try:
                entries = list(os.scandir(d))
            except OSError:
                continue
            rel = Path(d).relative_to(self.root).as_posix()
            if any(e.name == DB and e.is_file() for e in entries):
                found[rel] = Path(d)
                continue
            for e in entries:
                if e.name == INFO_FILE and e.is_file():
                    infos["" if rel == "." else rel] = Path(e.path)
                if e.is_dir(follow_symlinks=False) and not e.name.startswith(".") and e.name not in SKIP_DIRS:
                    stack.append(Path(e.path))
        return found, infos

    def run_dir(self, path: str) -> Path:
        d = self.dirs.get(path)
        if d is None:
            raise KeyError(path)
        return d

    def poll_forever(self) -> None:
        """Until `stop`; sets `ready` after the first pass. A failed pass is logged and the next one runs."""
        last_walk = 0.0
        while not self._stop.is_set():
            t0 = time.time()
            try:
                if t0 - last_walk >= REWALK:
                    self.rewalk()
                    last_walk = t0
                self.poll()
            except Exception as e:
                print(f"[trex] {self.root}: index pass failed: {e!r}", file=sys.stderr, flush=True)
            self.ready.set()
            self._stop.wait(max(0.0, POLL - (time.time() - t0)))

    def stop(self) -> None:
        self._stop.set()

    def rewalk(self) -> None:
        found, infos = self.walk()
        with self.lock:
            gone = [p for p in self.records if p not in found]
            self.dirs = found
        for p in gone:
            self.drop(p, publish=True)
        for p in [p for p in self.folders if p not in infos]:
            del self.folders[p]
            self._view_gen += 1
            self.hub.publish(p, "folder", {"path": p, "info": None})
        for p, f in infos.items():
            try:
                mtime = f.stat().st_mtime_ns
                if p in self.folders and self.folders[p][0] == mtime:
                    continue
                info = json.loads(f.read_text())
            except (OSError, ValueError) as e:
                print(f"[trex] {f}: {e!r}", file=sys.stderr, flush=True)
                continue
            self.folders[p] = (mtime, info if isinstance(info, dict) else {"value": info})
            self._view_gen += 1
            self.hub.publish(p, "folder", {"path": p, "info": self.folders[p][1]})

    def poll(self) -> list[str]:
        """Rescan runs that changed, went silent, or are due new kept buckets; returns their paths."""
        now = time.time()
        todo: list[tuple[str, Path, Sig]] = []
        growth = 0
        for path, d in list(self.dirs.items()):
            st, sig = self.records.get(path), _stat_sig(d)
            if st is None or _needs_scan(st, sig, now):
                todo.append((path, d, sig))
                growth += _bytes_of(sig) - (_bytes_of(st["sig"]) if st else 0)
        if self.workers > 1 and len(todo) > 1 and growth > INLINE_BYTES:
            self._sync_pool(todo)
        else:
            self._sync_inline(todo)
        return [p for p, _, _ in todo]

    # ---- per-run sync ----

    def _job(self, path: str, d: Path, sig: Sig) -> Job:
        st = self.records.get(path)
        prev: Prev | None = None
        if st:
            prev = {"uid": st["uid"], "seq": st["seq"], "mseq": st["mseq"], "kept_seq": st["kept_seq"],
                    "kept_t": st["kept_t"], "kept_state": st["kept_state"], "pyramid_seq": st["pyramid_seq"],
                    "pyramid_t": st["pyramid_t"]}
        return {"path": path, "dir": str(d), "sig": sig, "prev": prev, "crash_after": CRASH_AFTER,
                "kept_refresh": KEPT_REFRESH, "pyramid_refresh": PYRAMID_REFRESH, "want_rows": self.hub.watched(path)}

    def _sync_inline(self, todo: Sequence[tuple[str, Path, Sig]]) -> None:
        batch = _Batch(self)
        for t in todo:
            if self._stop.is_set():
                break
            j = self._job(*t)
            try:
                batch.add(scan(j))
            except Exception as e:
                print(f"[trex] {j['path']}: {e!r}", file=sys.stderr, flush=True)
            if batch.full():
                batch.commit()
        batch.commit()

    def _sync_pool(self, todo: Sequence[tuple[str, Path, Sig]]) -> None:
        jobs = deque(self._job(*t) for t in todo)
        batch = _Batch(self)
        pool = ProcessPoolExecutor(max_workers=min(self.workers, len(todo)), mp_context=_mp_context(), initializer=_worker_init)
        pending: dict[Future[ScanResult | None], Job] = {}
        try:
            while (jobs or pending) and not self._stop.is_set():
                while jobs and len(pending) < 2 * self.workers:
                    j = jobs.popleft()
                    pending[pool.submit(scan, j)] = j
                done, _ = wait(pending, timeout=BATCH_SECONDS, return_when=FIRST_COMPLETED)
                for f in done:
                    j = pending.pop(f)
                    try:
                        batch.add(f.result())
                    except BrokenProcessPool:
                        raise
                    except Exception as e:
                        print(f"[trex] {j['path']}: {e!r}", file=sys.stderr, flush=True)
                if batch.full() or not (jobs or pending) or time.time() - batch.t0 >= BATCH_SECONDS:
                    batch.commit()
        except BrokenProcessPool as e:
            print(f"[trex] index worker died: {e!r}", file=sys.stderr, flush=True)
        finally:
            pool.shutdown(wait=not self._stop.is_set(), cancel_futures=True)
        batch.commit()

    def drop(self, path: str, publish: bool = False) -> None:
        with self.lock:
            self.records.pop(path, None)
            self._view_gen += 1
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
        if publish:
            self.hub.publish(path, "delete", {"run": path})

    def apply(self, results: Sequence[ScanResult]) -> None:
        """Commit scan results in one transaction, then publish their events in order; nothing once closed."""
        staged: dict[str, RunRecord] = {}
        events: list[tuple[str, list[Event]]] = []
        now = time.time()
        with self._write_lock:
            if self._closed:
                return
            self._writer.execute("BEGIN IMMEDIATE")
            try:
                for r in results:
                    st, ev = self._stage(r, staged.get(r["path"]) or self.records.get(r["path"]), now)
                    staged[r["path"]] = st
                    events.append((r["path"], ev))
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
                    self.hub.publish(path, "run", self.run_meta(path))
                elif isinstance(data, str):
                    self.hub.publish_msg(path, sse_text(kind, data))
                else:
                    self.hub.publish(path, kind, data)

    def _stage(self, r: ScanResult, cur: RunRecord | None, now: float) -> tuple[RunRecord, list[Event]]:
        """Write one scan result inside `apply`'s transaction; its record and events."""
        path, ev = r["path"], list[Event]()
        if r["reset"] and cur is not None:
            self._forget(path)
            ev.append(("delete", {"run": path}))
            cur = None
        if r["media"]:
            self._writer.executemany("INSERT OR REPLACE INTO media VALUES (?,?,?,?,?,?)", r["media"])
        st = _record(r, cur)
        was, done = cur is not None and cur["state"] != "running", st["state"] != "running"  # finished before, and now
        if (r["kept"] is not None and done) or was != done:
            self._bump(st["keys"])
        if r["kept"] is not None:
            self._writer.execute("DELETE FROM kept WHERE path=?", (path,))
            self._writer.executemany("INSERT INTO kept VALUES (?,?,?,?,?)",
                                     [(path, k.key, k.level, r["kept_seq"], k.data) for k in r["kept"]])
            st.update(kept_seq=r["kept_seq"], kept_t=now, kept_state=r["state"])
        if r["pyramid"] is not None:
            for t in ("pyramid", "compiled"):
                self._writer.execute(f"DELETE FROM {t} WHERE path=?", (path,))
            self._writer.executemany("INSERT INTO pyramid VALUES (?,?,?,?,?)",
                                     [(p.key, level, i, path, data) for p in r["pyramid"] for level, i, data in p.blocks])
            self._writer.executemany("INSERT INTO compiled VALUES (?,?,?,?)",
                                     [(path, p.key, p.fine, r["pyramid_seq"]) for p in r["pyramid"]])
            st.update(pyramid_seq=r["pyramid_seq"], pyramid_t=now)
        self._writer.execute("INSERT OR REPLACE INTO runs VALUES (?, ?)", (path, dumps(st)))
        return st, ev + _events(r, cur, st)

    def _bump(self, keys: Sequence[str]) -> None:
        """Note that the finished runs of `keys`, or their kept buckets, changed."""
        for k in keys:
            self._gens[k] = self._gens.get(k, 0) + 1

    def _forget(self, path: str) -> None:
        """Delete a run's index rows inside the caller's write transaction."""
        rec = self.records.get(path)
        self._bump(rec["keys"] if rec else list(self._gens))
        for t in ("runs", "media", "kept", "pyramid", "compiled"):
            self._writer.execute(f"DELETE FROM {t} WHERE path=?", (path,))

    # ---- queries ----

    def run_meta(self, path: str) -> RunMeta:
        st = self.records[path]
        p = st["public"]
        parent = path.rsplit("/", 1)[0] if "/" in path else ""
        return {
            "id": path, "uid": st["uid"], "name": p.get("name") or path.rsplit("/", 1)[-1], "parent": parent,
            "tags": p["tags"], "config": p["config"], "info": p["info"],
            "summary": {**p["user_summary"], **st["summary"]},
            "state": st["state"], "created": p.get("created"), "updated": st["heartbeat"],
            "seq": st["seq"], "mseq": st["mseq"], "keys": st["keys"], "kept_seq": st["kept_seq"],
            "pyramid_seq": st["pyramid_seq"],
        }

    def dump(self, path: str, have: Have | None) -> bytes:
        """Run `path`'s index rows a mirror holding `have` of it lacks, in one body (`buckets.frame`): a JSON header (its
        record, media rows, and which kept arrays and compiled blocks follow), then those arrays, compressed; all of it
        when `have` is of another uid. KeyError for a run it does not have."""
        c = self.reader()
        try:
            c.execute("BEGIN")
            row = c.execute("SELECT record FROM runs WHERE path=?", (path,)).fetchone()
            if row is None:
                raise KeyError(path)
            rec: RunRecord = json.loads(row[0])
            mseq, with_kept, with_pyramid = _lacking(have, rec)
            media = c.execute("SELECT seq, step, key, kind, file FROM media WHERE path=? AND seq>=? ORDER BY seq", (path, mseq)).fetchall()
            kept = c.execute("SELECT key, level, seq, data FROM kept WHERE path=? ORDER BY key", (path,)).fetchall() if with_kept else None
            compiled = c.execute("SELECT key, fine, seq FROM compiled WHERE path=? ORDER BY key", (path,)).fetchall() if with_pyramid else None
            pyramid = c.execute("SELECT key, level, block, data FROM pyramid WHERE path=?", (path,)).fetchall() if with_pyramid else None
            c.execute("COMMIT")
        finally:
            self.release(c)
        head = {"record": rec, "media": media, "kept": [k[:3] for k in kept] if kept is not None else None,
                "compiled": compiled, "pyramid": [p[:3] for p in pyramid] if pyramid is not None else None}
        return bk.frame([dumps(head).encode(), *(zlib.compress(k[3], 1) for k in kept or []), *(p[3] for p in pyramid or [])])

    def dump_many(self, held: Sequence[tuple[str, Have | None]]) -> list[bytes]:
        """`dump` of each (run, what a mirror holds of it); empty for a run it does not have."""
        out: list[bytes] = []
        for path, have in held:
            try:
                out.append(self.dump(path, have))
            except KeyError:
                out.append(b"")
        return out

    def info(self) -> dict[str, object]:
        return {"root": str(self.root), "name": self.root.name, "cache": self.cache_dir.name}

    def tree(self) -> list[tuple[str, RunState]]:
        with self.lock:
            return [(p, st["state"]) for p, st in sorted(self.records.items())]

    def runs(self, prefix: str) -> RunsView:
        with self.lock:
            paths = [p for p in self.records if in_scope(p, prefix)]
            metas = [self.run_meta(p) for p in sorted(paths)]
        c = self.reader()
        try:
            media = [MediaRecord(*m) for m in c.execute(
                "SELECT path, seq, step, key, kind, file FROM media WHERE ? = '' OR path = ? OR (path > ? AND path < ?) "
                "ORDER BY path, seq", (prefix, prefix, prefix + "/", prefix + "0"))]  # "0" follows "/"
        finally:
            self.release(c)
        folders = {p: v[1] for p, v in list(self.folders.items()) if in_scope(p, prefix) or in_scope(prefix, p)}
        return {"runs": metas, "media": media, "folders": folders}

    def runs_body(self, prefix: str) -> bytes:
        """`runs` as JSON, kept while no run, medium or folder note changes."""
        return self._memo.get(("runs", prefix), self._view_gen, lambda: _sized(dumps(self.runs(prefix)).encode()))

    def run(self, path: str) -> RunView:
        out = self.runs(path)
        out["runs"] = [m for m in out["runs"] if m["id"] == path]
        if not out["runs"]:
            raise KeyError(path)
        out["media"] = [m for m in out["media"] if m[0] == path]
        return {"run": out["runs"][0], "media": out["media"]}

    def _read_rows(self, path: str, start: int, stop: int | None = None) -> list[chunks.Row]:
        """Rows [start, stop) of the run file (to its last row when `stop` is None); KeyError for an unknown run."""
        with self.lock:
            if path not in self.records:
                raise KeyError(path)
        with snapshot(self.run_dir(path)) as c:
            return chunks.rows(c, start, stop if stop is not None else chunks.row_count(c))

    def rows_json(self, path: str, start: int, stop: int | None = None) -> str:
        """The `rows` event JSON of rows [start, stop) of the run file."""
        return _rows_event(path, start, self._read_rows(path, start, stop))

    def backfill(self, prefix: str) -> list[bytes]:
        """`rows` events for running runs' rows beyond their kept buckets."""
        with self.lock:
            tails = [(p, st["kept_seq"], st["seq"]) for p, st in self.records.items()
                     if in_scope(p, prefix) and st["state"] == "running" and 0 < st["seq"] - st["kept_seq"] <= ROWS_EVENT_MAX]
        out: list[bytes] = []
        for p, start, stop in tails:
            try:
                rows = self._read_rows(p, start, stop)
            except READ_ERRORS:
                continue
            if rows:
                out.append(sse_text("rows", _rows_event(p, start, rows)))
        return out

    def messages(self, prefix: str, stop: threading.Event) -> Generator[bytes, None, None]:
        """SSE for runs under `prefix`: rows beyond their kept buckets, then live events and a heartbeat after each
        quiet HEARTBEAT, until the subscription overflows or `stop` is set."""
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
        """{path: (rows, media)} of running runs."""
        with self.lock:
            return {p: (st["seq"], st["mseq"]) for p, st in self.records.items()
                    if in_scope(p, prefix) and st["state"] == "running"}

    # ---- buckets ----

    def buckets_bodies(self, asks: Sequence[Ask]) -> list[bytes]:
        """`buckets_body` of each ask, in order."""
        return [self.buckets_body(*a) for a in asks]

    def buckets_body(self, key: str, level: int, index: int, scope: str = "", runs: Sequence[str] | None = None,
                     which: Which = "all") -> bytes:
        """Block `index` of `level` of `key` as a bucket array (`trex.buckets`): of the runs `runs`, or else of the runs
        under `scope` in state `which`. A scope's finished runs' blocks are kept in memory while those runs' buckets
        stay the same."""
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
                return sorted({p for p in runs if (r := self.records.get(p)) is not None and key in r["keys"]})
            return sorted(p for p, r in self.records.items() if in_scope(p, scope) and key in r["keys"]
                          and (which == "all" or (r["state"] == "running") == (which == "running")))

    def _block(self, key: str, level: int, index: int, paths: list[str]) -> bytes:
        """Block `index` of `level` of `key` of runs `paths` (in path order): cut from the finished runs' merged level
        (`_level`), or merged from the other runs' kept buckets, where a run keeps its buckets at `level` or finer; else
        built from its run file (`_build_many`)."""
        lo, hi = index * bk.BLOCK, (index + 1) * bk.BLOCK
        done, levels, seqs = self._runs(key)
        positions = self._positions(key)
        at = np.array([positions.get(p, -1) for p in paths], np.int64)  # each run's index in `done`, or -1
        known = at >= 0
        coarse = known.copy()
        coarse[known] = levels[at[known]] <= level
        seq = np.zeros(len(paths), np.uint32)
        seq[coarse] = seqs[at[coarse]]
        parts = []
        if coarse.any():
            out = np.full(len(done), -1, np.int32)
            out[at[coarse]] = np.flatnonzero(coarse)
            part = bk.cut(self._level(key, level), lo, hi, out >= 0)
            parts.append(part._replace(run=out[part.run]))
        others = np.flatnonzero(~known)
        fine = [int(i) for i in np.flatnonzero(known & ~coarse)]
        if others.size:
            st = self._kept_of(key, [paths[i] for i in others])
            near = st.level <= level
            seq[others[near]] = st.seq[near]
            part = bk.cut(st.at(level, np.flatnonzero(near).astype(np.int32)), lo, hi)
            parts.append(part._replace(run=others[part.run].astype(np.int32)))
            fine += [int(i) for i in others[~near]]
        if fine:
            fine.sort()
            blobs = self._compiled(key, level, index, [paths[i] for i in fine])
            todo = [paths[i] for i in fine if paths[i] not in blobs]
            for p, (blob, _) in zip(todo, self._build_many(todo, key, level, index) if todo else [], strict=True):
                blobs[p] = blob
            st = bk.stack([paths[i] for i in fine], [blobs[paths[i]] for i in fine])
            pos = np.array(fine, np.int32)
            seq[pos] = st.seq
            parts.append(st.buckets._replace(run=pos[st.buckets.run]))
        return bk.encode(level, index, paths, seq, bk.union(parts))

    def _compiled(self, key: str, level: int, index: int, paths: Sequence[str]) -> dict[str, bytes]:
        """Block `index` of `level` of `key` of each of `paths` whose levels are compiled from every row it has: its one-run
        bucket array, empty where it has no buckets; below the finest level compiled, that level's buckets refined."""
        with self.lock:
            rows_now = {p: self.records[p]["seq"] for p in paths if p in self.records}
        c = self.reader()
        try:
            have = {p: (fine, n) for p, fine, n in c.execute("SELECT path, fine, seq FROM compiled WHERE key=?", (key,))
                    if rows_now.get(p) == n}
            out: dict[str, bytes] = {}
            for fine in sorted({max(fine, level) for fine, _ in have.values()}):
                block = index >> (fine - level)
                rows = dict(c.execute("SELECT path, data FROM pyramid WHERE key=? AND level=? AND block=?", (key, fine, block)))
                for p, (f, n) in have.items():
                    if max(f, level) == fine:
                        out[p] = _refined(rows.get(p), fine, level, index, n)
        finally:
            self.release(c)
        return out

    def _runs(self, key: str) -> tuple[list[str], npt.NDArray[np.int8], npt.NDArray[np.uint32]]:
        """The finished runs of `key` in path order, the level each keeps its buckets at and the rows they hold: from
        their saved levels when those are current, else from their stack."""
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

    def _kept_of(self, key: str, paths: list[str]) -> Stack:
        """The kept buckets of `key` of runs `paths`, read from the index."""
        rows: dict[str, bytes] = {}
        c = self.reader()
        try:
            for i in range(0, len(paths), 500):
                part = paths[i:i + 500]
                rows.update(c.execute(f"SELECT path, data FROM kept WHERE key=? AND path IN ({','.join('?' * len(part))})",
                                      (key, *part)).fetchall())
        finally:
            self.release(c)
        return bk.stack(paths, [rows.get(p, NO_BUCKETS) for p in paths])

    def _level(self, key: str, level: int) -> Buckets:
        """The buckets of every finished run of `key` that keeps its buckets at `level` or finer, merged to `level`
        (on threads); kept as stacks are."""
        return self._memo.get(("level", key, level), self._gens.get(key, 0), lambda: self._merge_level(key, level))

    def _merge_level(self, key: str, level: int) -> tuple[Buckets, int]:
        saved = self._saved(key)
        if saved is not None and (saved / f"L{level}-run.npy").exists():
            part = Buckets(*(np.load(saved / f"L{level}-{name}.npy", mmap_mode="r") for name in Buckets._fields))
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
            parts = list(pool.map(lambda ab: one(*ab), zip(bounds, bounds[1:])))
        return bk.Buckets(*(np.concatenate(field) for field in zip(*parts)))

    def _stack(self, key: str) -> Stack:
        """The finished runs' kept buckets of `key` (`buckets.stack`, runs in path order); kept while they stay the
        same. A new stack gets its levels from the coarsest a first view takes to the finest it keeps merged and
        saved, on a thread of its own."""
        return self._memo.get(("stack", key), self._gens.get(key, 0), lambda: self._read_stack(key))

    def _read_stack(self, key: str) -> tuple[Stack, int]:
        done, _, sig = self._finished(key)
        c = self.reader()
        try:
            rows = dict(c.execute("SELECT path, data FROM kept WHERE key=?", (key,)).fetchall())
        finally:
            self.release(c)
        st = bk.stack(done, [rows.get(p, NO_BUCKETS) for p in done])
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
            done = [(p, rec["kept_seq"], rec["kept_t"]) for p, rec in self.records.items()
                    if rec["state"] != "running" and key in rec["keys"]]
        done.sort()
        paths = [p for p, _, _ in done]
        h = hashlib.sha1(f"{CACHE_VERSION}\0{key}\0".encode())
        h.update("\0".join(paths).encode())
        h.update(np.array([(seq, t) for _, seq, t in done], np.float64).tobytes())
        return Finished(paths, np.array([seq for _, seq, _ in done], np.uint32), h.digest())

    def _levels_dir(self, key: str, sig: bytes) -> Path:
        return self.cache_dir / "levels" / f"{hashlib.sha1(key.encode()).hexdigest()[:20]}-{sig.hex()[:20]}"

    def _saved(self, key: str) -> Path | None:
        """The directory of the merged levels of `key` an earlier build saved (`_save_levels`) for its finished runs and
        their kept buckets as they are."""
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
        kept level (`levels`) and each level's buckets (`L<level>-<field>`); then delete its levels saved for other runs
        and the least recently used saved levels beyond LEVELS_BYTES."""
        d = self._levels_dir(key, sig)
        tmp = d.with_name(f"{d.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        try:
            tmp.mkdir(parents=True)
            for lv, b in parts.items():
                for name, a in zip(Buckets._fields, b, strict=True):
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

    def _build_many(self, paths: list[str], key: str, level: int, index: int) -> list[tuple[bytes, int]]:
        """(block, rows read) of block (level, index) of `key` for each run, from the run files: on a process pool when
        there are more than INLINE_BUILDS."""
        jobs = [(str(self.run_dir(p)), key, level, index) for p in paths]
        if len(jobs) <= INLINE_BUILDS or self.workers == 1:
            return [build_block(*j) for j in jobs]
        with ProcessPoolExecutor(max_workers=self.workers, mp_context=_mp_context(), initializer=_worker_init) as pool:
            return list(pool.map(_build_block_job, jobs, chunksize=max(1, len(jobs) // (4 * self.workers))))

    def media_path(self, path: str, file: str) -> Path:
        """KeyError unless `file` is in the run's media/."""
        d = self.run_dir(path)
        f = (d / file).resolve()
        if f.parent != (d / "media").resolve():
            raise KeyError(file)
        return f
