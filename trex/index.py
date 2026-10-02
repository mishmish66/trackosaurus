"""Explorer index of a runs directory, in <cache_root>/<hash of root>/index.sqlite: run metadata, last
values, kept tiles, a bounded cache of finer tiles, and media. A run is its path relative to the root.
Runs are scanned inline or in worker processes; the main process commits, then publishes events in
order per run.
"""

import hashlib
import json
import math
import multiprocessing
import os
import queue
import sqlite3
import struct
import sys
import threading
import time
import zlib
from collections import deque
from collections.abc import Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from multiprocessing.context import BaseContext
from pathlib import Path
from typing import Final, Literal, NamedTuple, Self, TypedDict

import numpy as np

from . import chunks, tiles
from .journal import JOURNAL
from .format import (DB, INFO_FILE, JSONValue, MediaKind, RunState, as_dict, as_float, as_run_state, as_str, as_str_list,
                     connect_ro)

type Sig = list[int]
"""[db mtime_ns, db size, wal mtime_ns, wal size]; 0s for a missing file."""

type Summary = dict[str, float | str]
"""Last value of every metric (non-finite as strings), plus _step and _runtime."""

type TileKind = Literal["top", "overview"]
type Event = tuple[str, object]
"""(SSE event name, data); data that is a str is already JSON."""


class MediaRecord(NamedTuple):

    run: str
    seq: int
    step: float
    key: str
    kind: MediaKind
    file: str
    crc: int
    size: int


class TileRecord(NamedTuple):

    key: str
    level: int
    idx: int
    data: bytes
    kind: int


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
    tiles_seq: int
    tiles_t: float
    tiles_state: RunState | None


class Job(TypedDict):

    path: str
    dir: str
    sig: Sig
    prev: Prev | None
    crash_after: float
    top_refresh: float
    want_rows: bool
    quiet: bool


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
    top: list[TileRecord] | None
    rows: str | None
    quiet: bool


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
    tiles_seq: int
    tiles_t: float
    tiles_state: RunState | None


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
    tiles_seq: int


class RunsView(TypedDict):
    """Runs under a folder, their media, and the notes of related folders."""

    runs: list[RunMeta]
    media: list[MediaRecord]
    folders: dict[str, dict[str, JSONValue]]


class RunView(TypedDict):
    run: RunMeta
    media: list[MediaRecord]

CACHE_VERSION: Final = 8  # bump whenever what the index stores changes; older caches are rebuilt
CRASH_AFTER = 300.0  # seconds without a heartbeat after which a running run shows as crashed
POLL: Final = 1.0  # seconds between polls of known runs
REWALK: Final = 3.0  # seconds between walks of the root for new and removed runs
CACHED, TOP, OVERVIEW = 0, 1, 2  # tiles.top: what a stored tile is
KINDS: Final[dict[str, int]] = {"top": TOP, "overview": OVERVIEW}  # request names of the kept tile kinds
OVERVIEW_UP: Final = 2  # overview tiles are top tiles merged this many levels coarser
TOP_REFRESH = 10.0  # seconds between top-tile rebuilds of a growing run
SKIP_DIRS: Final = frozenset({"node_modules", "__pycache__"})
INLINE_BYTES = 5 << 20  # polls whose run files grew by at most this many bytes are read in the main process
BATCH_RUNS: Final = 256  # scan results per index transaction
BATCH_SECONDS: Final = 0.5  # longest wait before committing a partial batch
TILE_CACHE_BYTES = int(os.environ.get("TREX_TILE_CACHE_MB", "4096")) << 20
PAGE_SIZE: Final = 16384
READ_ERRORS: Final = (sqlite3.Error, OSError, ValueError, KeyError, struct.error)
ROWS_EVENT_MAX: Final = 20_000  # larger catch-ups reach browsers as refreshed tiles instead of rows

TABLES: Final = {
    "cache": "CREATE TABLE IF NOT EXISTS cache(key TEXT PRIMARY KEY, value TEXT)",
    "runs": "CREATE TABLE IF NOT EXISTS runs(path TEXT PRIMARY KEY, state TEXT NOT NULL)",
    "media": "CREATE TABLE IF NOT EXISTS media(path TEXT, seq INTEGER, step REAL, key TEXT, kind TEXT, file TEXT, "
             "crc INTEGER, size INTEGER, PRIMARY KEY(path, seq)) WITHOUT ROWID",
    # top: TOP for the coarsest tiles of a metric and OVERVIEW for those merged OVERVIEW_UP levels up
    # (both kept); CACHED for a finer tile built on request (evicted by `used`).
    # seq: run rows when built; a tile stays valid while later rows lie beyond its step range.
    "tiles": "CREATE TABLE IF NOT EXISTS tiles(path TEXT, key TEXT, level INTEGER, idx INTEGER, top INTEGER, "
             "seq INTEGER, used REAL, data BLOB, PRIMARY KEY(path, key, level, idx))",
    "tiles_kind": "CREATE INDEX IF NOT EXISTS tiles_kind ON tiles(key, top)",
    "tiles_used": "CREATE INDEX IF NOT EXISTS tiles_used ON tiles(top, used)",
}


def dumps(obj: object) -> str:
    """Compact JSON (non-finite floats raise; see `wire`)."""
    return json.dumps(obj, separators=(",", ":"), allow_nan=False)


def wire(d: Mapping[str, float]) -> dict[str, float | str]:
    """Metrics for JSON: non-finite values as "NaN", "Infinity", "-Infinity"."""
    out: dict[str, float | str] = {}
    for k, v in d.items():
        out[k] = v if math.isfinite(v) else "NaN" if v != v else "Infinity" if v > 0 else "-Infinity"
    return out


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
            vals = chunks.decode(r[0]).values
            if len(vals):
                out[name] = vals[-1]
    return out


def _top_tiles(c: sqlite3.Connection, names: Mapping[int, str], stop: int) -> list[TileRecord]:
    """Top and overview tiles over rows [0, stop) of every metric."""
    out: list[TileRecord] = []
    for kid, name in sorted(names.items(), key=lambda kv: kv[1]):
        s, v, t = chunks.metric(c, kid, stop=stop)
        if not s.size:
            continue
        level, idxs = tiles.top_tiles(float(s.min()), float(s.max()))
        top = [tiles.build(s, v, t, level, i) for i in idxs]
        out += [TileRecord(name, level, i, b, TOP) for i, b in zip(idxs, top, strict=True)]
        parents = sorted({i >> OVERVIEW_UP for i in idxs})
        out += [TileRecord(name, level + OVERVIEW_UP, i, b, OVERVIEW)
                for i, b in zip(parents, tiles.coarsen(top, OVERVIEW_UP), strict=True)]
    return out


def scan(job: Job) -> ScanResult | None:
    """What changed in a run, or None to retry later (no id yet, or changed while read without a WAL)."""
    path, d, sig, prev = job["path"], Path(job["dir"]), job["sig"], job["prev"]
    c = connect_ro(d)
    try:
        c.execute("BEGIN")
        meta: dict[str, JSONValue] = {k: json.loads(v) for k, v in c.execute("SELECT key, value FROM meta")}
        if "id" not in meta:
            return None
        state, heartbeat = _state(meta, job["crash_after"])
        seq = chunks.row_count(c)
        m_file: int = c.execute("SELECT coalesce(max(seq) + 1, 0) FROM media").fetchone()[0]
        reset = prev is not None and _rewritten(prev, meta["id"], seq, m_file)
        prev = None if reset else prev
        names = chunks.key_names(c)
        summary = _summary(c, names, seq) if prev is None or seq != prev["seq"] else None
        top = _top_tiles(c, names, seq) if _top_due(prev, seq, state, job["top_refresh"]) else None
        text = _rows_text(c, path, prev["seq"], seq) if prev and _rows_wanted(job, prev, seq) else None
        mseq = prev["mseq"] if prev else 0
        media = _new_media(c, d, path, mseq)
    finally:
        c.close()
    after = _stat_sig(d)
    if after != sig and not (sig[2] and after[2]):
        return None
    return {"path": path, "sig": sig, "uid": str(meta["id"]), "reset": reset, "fresh": prev is None,
            "seq": seq, "mseq": mseq + len(media), "media": media,
            "state": state, "heartbeat": heartbeat, "public": _public(meta, state), "keys": sorted(names.values()),
            "summary": summary, "top": top, "rows": text, "quiet": job["quiet"]}


def _state(meta: Mapping[str, JSONValue], crash_after: float) -> tuple[RunState, float | None]:
    """(state, heartbeat); a running run silent for `crash_after` seconds is crashed."""
    state, heartbeat = as_run_state(meta.get("state")), as_float(meta.get("heartbeat"))
    if state == "running" and time.time() - (heartbeat or 0) > crash_after:
        state = "crashed"
    return state, heartbeat


def _summary(c: sqlite3.Connection, names: Mapping[int, str], seq: int) -> Summary:
    out = wire(_last_values(c, names))
    if seq:
        n, data = c.execute("SELECT n, data FROM rowmeta WHERE seq0 + n = ?", (seq,)).fetchone()
        mv = memoryview(data).cast("d")
        out["_step"], out["_runtime"] = mv[n - 1], mv[2 * n - 1]
    return out


def _top_due(prev: Prev | None, seq: int, state: RunState, refresh: float) -> bool:
    """Kept tiles are rebuilt for a new run, and for a changed one when it stopped running or `refresh` passed."""
    if prev is None:
        return True
    stale = prev["tiles_seq"] != seq or prev["tiles_state"] != state
    return stale and (state != "running" or time.time() - prev["tiles_t"] >= refresh)


def _rewritten(prev: Prev, uid: JSONValue, rows: int, media: int) -> bool:
    """The run file was replaced: a new id, or fewer rows or media than indexed."""
    return prev["uid"] != uid or rows < prev["seq"] or media < prev["mseq"]


def _rows_wanted(job: Job, prev: Prev, seq: int) -> bool:
    """A rows event goes to watching browsers, unless the catch-up is large (tiles serve it)."""
    return job["want_rows"] and not job["quiet"] and 0 < seq - prev["seq"] <= ROWS_EVENT_MAX


def _rows_text(c: sqlite3.Connection, path: str, start: int, stop: int) -> str:
    rows = ",".join(dumps([r.step, r.t, wire(r.values)]) for r in chunks.rows(c, start, stop))
    return f'{{"run":{dumps(path)},"seq0":{start},"rows":[{rows}]}}'


def _new_media(c: sqlite3.Connection, d: Path, path: str, mseq: int) -> list[MediaRecord]:
    """Media from `mseq` on, up to a gap or a file not yet written."""
    out: list[MediaRecord] = []
    for i, step, key, kind, file in c.execute("SELECT seq, step, key, kind, file FROM media WHERE seq >= ? ORDER BY seq", (mseq,)):
        if i != mseq + len(out):
            break
        f = d / file
        try:
            size, crc = f.stat().st_size, zlib.crc32(f.read_bytes()) if kind == "html" else 0
        except FileNotFoundError:
            break
        out.append(MediaRecord(path, i, step, key, kind, file, crc, size))
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


def _mp_context() -> BaseContext:
    """Fresh worker interpreters that import only trex (the main process runs threads)."""
    if "forkserver" in multiprocessing.get_all_start_methods():
        ctx = multiprocessing.get_context("forkserver")
        ctx.set_forkserver_preload([__name__])
        return ctx
    return multiprocessing.get_context("spawn")


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
    changed = r["fresh"] or r["quiet"] or r["top"] is not None or r["public"] != cur["public"] or st["keys"] != cur["keys"]
    return [*rows, *media, *([("run", None)] if changed else [])]


def _needs_scan(st: RunRecord, sig: Sig, now: float) -> bool:
    """Its files changed, it went silent while running, or its kept tiles are due a refresh."""
    silent = st["state"] == "running" and now - (st["heartbeat"] or now) > CRASH_AFTER
    due = st["tiles_seq"] != st["seq"] and now - st["tiles_t"] >= TOP_REFRESH
    return sig != st["sig"] or silent or due


def _record(r: ScanResult, cur: RunRecord | None) -> RunRecord:
    """The run's record after scan result `r`; kept-tile fields carry over from `cur`."""
    return {"uid": r["uid"], "seq": r["seq"], "mseq": r["mseq"], "keys": r["keys"],
            "summary": r["summary"] if r["summary"] is not None else cur["summary"] if cur else {},
            "sig": r["sig"], "heartbeat": r["heartbeat"], "public": r["public"], "state": r["state"],
            "tiles_seq": cur["tiles_seq"] if cur else -1, "tiles_t": cur["tiles_t"] if cur else 0.0,
            "tiles_state": cur["tiles_state"] if cur else None}


def _bytes_of(sig: Sig) -> int:
    return sig[1] + sig[3]


class TileBudget:
    """Bytes of cached finer tiles shared by Explorers; beyond `limit` (default TILE_CACHE_BYTES) the least recently
    used across all of them are evicted to 90% of it."""

    def __init__(self, limit: int | None = None) -> None:
        self.limit = limit
        self.members: "list[Explorer]" = []
        self.lock = threading.Lock()

    def used(self) -> int:
        return sum(m._tile_bytes for m in list(self.members))

    def enforce(self) -> None:
        limit = self.limit if self.limit is not None else TILE_CACHE_BYTES
        if self.used() <= limit:
            return
        with self.lock:
            while (excess := self.used() - int(0.9 * limit)) > 0:
                members = list(self.members)
                oldest = sorted((t, i) for i, m in enumerate(members) if (t := m._oldest_cached()) is not None)
                bound = oldest[1][0] if len(oldest) > 1 else math.inf
                if not oldest or not members[oldest[0][1]]._evict(bound, excess):
                    return


class Explorer:
    """Index of a runs directory, kept current by `start` (or `poll_forever`); `close` releases it."""

    def __init__(self, root: str | os.PathLike[str], cache_root: str | os.PathLike[str], workers: int | None = None,
                 budget: TileBudget | None = None) -> None:
        self.root = Path(root).resolve()
        self.workers = max(1, int(workers)) if workers is not None else default_workers()
        self.cache_dir = Path(cache_root) / hashlib.sha1(str(self.root).encode()).hexdigest()[:12]
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "root.txt").write_text(str(self.root) + "\n")
        self.db_path = self.cache_dir / "index.sqlite"
        self.w = self._connect()
        for sql in TABLES.values():
            self.w.execute(sql)
        v = self.w.execute("SELECT value FROM cache WHERE key='version'").fetchone()
        if v is None or int(v[0]) != CACHE_VERSION:
            self.w.execute("BEGIN IMMEDIATE")
            for t in [r[0] for r in self.w.execute("SELECT name FROM sqlite_master WHERE type='table' AND name != 'cache'")]:
                self.w.execute(f"DROP TABLE {t}")
            for name, sql in TABLES.items():
                if name != "cache":
                    self.w.execute(sql)
            self.w.execute("INSERT OR REPLACE INTO cache VALUES ('version', ?)", (str(CACHE_VERSION),))
            self.w.execute("COMMIT")
        self.lock = threading.Lock()
        self.hub = Hub()
        self.state: dict[str, RunRecord] = {path: json.loads(s) for path, s in self.w.execute("SELECT path, state FROM runs")}
        self.dirs: dict[str, Path] = {}
        self.folders: dict[str, tuple[int, dict[str, JSONValue]]] = {}  # folder path -> (mtime_ns, notes) of trex_info.json files
        self._readers: queue.LifoQueue[sqlite3.Connection] = queue.LifoQueue()
        self._tile_lock = threading.Lock()
        self._tile_w = self._connect()
        self._tile_bytes: int = self._tile_w.execute("SELECT coalesce(sum(length(data)), 0) FROM tiles WHERE top = ?",
                                                     (CACHED,)).fetchone()[0]
        self._stop = threading.Event()
        self._poller: threading.Thread | None = None
        self._closed = False
        self.ready = threading.Event()
        self.budget = budget or TileBudget()
        self.budget.members.append(self)

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
        """Stop polling, end subscriptions, leave the tile budget and close the index's connections."""
        self.stop()
        if self._poller is not None and self._poller is not threading.current_thread():
            self._poller.join()
        self.hub.close()
        with self.budget.lock:
            if self in self.budget.members:
                self.budget.members.remove(self)
        with self.lock, self._tile_lock:
            self._closed = True
            self.w.close()
            self._tile_w.close()
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
            if any(e.name == DB and e.is_file() for e in entries):
                rel = Path(d).relative_to(self.root).as_posix()
                found["." if rel == "." else rel] = Path(d)
                continue
            for e in entries:
                if e.name == INFO_FILE and e.is_file():
                    rel = Path(d).relative_to(self.root).as_posix()
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
        """Until `stop`; sets `ready` after the first pass."""
        last_walk = 0.0
        while not self._stop.is_set():
            t0 = time.time()
            if t0 - last_walk >= REWALK:
                self.rewalk()
                last_walk = t0
            self.poll()
            self.ready.set()
            self._stop.wait(max(0.0, POLL - (time.time() - t0)))

    def stop(self) -> None:
        self._stop.set()

    def rewalk(self) -> None:
        found, infos = self.walk()
        with self.lock:
            gone = [p for p in self.state if p not in found]
            self.dirs = found
        for p in gone:
            self.drop(p, publish=True)
        for p in [p for p in self.folders if p not in infos]:
            del self.folders[p]
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
            self.hub.publish(p, "folder", {"path": p, "info": self.folders[p][1]})

    def poll(self) -> list[str]:
        """Rescan runs that changed, went silent, or are due new kept tiles; returns their paths."""
        now = time.time()
        todo: list[tuple[str, Path, Sig]] = []
        growth = 0
        for path, d in list(self.dirs.items()):
            st, sig = self.state.get(path), _stat_sig(d)
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
        st = self.state.get(path)
        prev: Prev | None = None
        if st:
            prev = {"uid": st["uid"], "seq": st["seq"], "mseq": st["mseq"], "tiles_seq": st["tiles_seq"],
                    "tiles_t": st["tiles_t"], "tiles_state": st["tiles_state"]}
        return {"path": path, "dir": str(d), "sig": sig, "prev": prev, "crash_after": CRASH_AFTER,
                "top_refresh": TOP_REFRESH, "want_rows": self.hub.watched(path), "quiet": False}

    def _sync_inline(self, todo: Sequence[tuple[str, Path, Sig]]) -> None:
        batch = _Batch(self)
        for t in todo:
            j = self._job(*t)
            try:
                batch.add(scan(j))
            except READ_ERRORS as e:
                print(f"[trex] {j['path']}: {e!r}", file=sys.stderr, flush=True)
            if batch.full():
                batch.commit()
        batch.commit()

    def _sync_pool(self, todo: Sequence[tuple[str, Path, Sig]]) -> None:
        jobs = deque(self._job(*t) for t in todo)
        batch = _Batch(self)
        with ProcessPoolExecutor(max_workers=min(self.workers, len(todo)), mp_context=_mp_context()) as pool:
            pending: dict[Future[ScanResult | None], Job] = {}
            try:
                while jobs or pending:
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
                for f in pending:
                    f.cancel()
        batch.commit()

    def drop(self, path: str, publish: bool = False) -> None:
        with self.lock:
            self.state.pop(path, None)
        with self._tile_lock:
            self.w.execute("BEGIN IMMEDIATE")
            for t in ("runs", "media", "tiles"):
                self.w.execute(f"DELETE FROM {t} WHERE path=?", (path,))
            self.w.execute("COMMIT")
        if publish:
            self.hub.publish(path, "delete", {"run": path})

    def apply(self, results: Sequence[ScanResult]) -> None:
        """Commit scan results in one transaction, then publish their events in order."""
        staged: dict[str, RunRecord] = {}
        events: list[tuple[str, list[Event]]] = []
        now = time.time()
        with self._tile_lock:
            self.w.execute("BEGIN IMMEDIATE")
            try:
                for r in results:
                    st, ev = self._stage(r, staged.get(r["path"]) or self.state.get(r["path"]), now)
                    staged[r["path"]] = st
                    events.append((r["path"], ev))
            except BaseException:
                self.w.execute("ROLLBACK")
                raise
            self.w.execute("COMMIT")
        with self.lock:
            self.state.update(staged)
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
            for t in ("runs", "media", "tiles"):
                self.w.execute(f"DELETE FROM {t} WHERE path=?", (path,))
            ev.append(("delete", {"run": path}))
            cur = None
        if r["media"]:
            self.w.executemany("INSERT OR REPLACE INTO media VALUES (?,?,?,?,?,?,?,?)", r["media"])
        st = _record(r, cur)
        if r["top"] is not None:
            self.w.execute("DELETE FROM tiles WHERE path=? AND top != ?", (path, CACHED))
            self.w.executemany("INSERT OR REPLACE INTO tiles VALUES (?,?,?,?,?,?,?,?)",
                               [(path, t.key, t.level, t.idx, t.kind, r["seq"], now, t.data) for t in r["top"]])
            st.update(tiles_seq=r["seq"], tiles_t=now, tiles_state=r["state"])
        self.w.execute("INSERT OR REPLACE INTO runs VALUES (?, ?)", (path, dumps(st)))
        return st, ev + _events(r, cur, st)

    # ---- queries ----

    def run_meta(self, path: str) -> RunMeta:
        st = self.state[path]
        p = st["public"]
        parent = path.rsplit("/", 1)[0] if "/" in path else ""
        return {
            "id": path, "uid": st["uid"], "name": p.get("name") or path.rsplit("/", 1)[-1], "parent": parent,
            "tags": p.get("tags") or [], "config": p.get("config") or {}, "info": p.get("info") or {},
            "summary": {**p["user_summary"], **(st["summary"] or {})},
            "state": st["state"], "created": p.get("created"), "updated": st["heartbeat"],
            "seq": st["seq"], "mseq": st["mseq"], "keys": st["keys"], "tiles_seq": st["tiles_seq"],
        }

    def tree(self) -> list[tuple[str, RunState]]:
        with self.lock:
            return [(p, st["state"]) for p, st in sorted(self.state.items())]

    def runs(self, prefix: str) -> RunsView:
        with self.lock:
            paths = [p for p in self.state if in_scope(p, prefix)]
            metas = [self.run_meta(p) for p in sorted(paths)]
        c = self.reader()
        try:
            media: list[MediaRecord] = []
            for p in paths:
                media += [MediaRecord(*m) for m in c.execute(
                    "SELECT path, seq, step, key, kind, file, crc, size FROM media WHERE path=? ORDER BY seq", (p,))]
        finally:
            self.release(c)
        folders = {p: v[1] for p, v in list(self.folders.items()) if in_scope(p, prefix) or in_scope(prefix, p)}
        return {"runs": metas, "media": media, "folders": folders}

    def run(self, path: str) -> RunView:
        out = self.runs(path)
        out["runs"] = [m for m in out["runs"] if m["id"] == path]
        if not out["runs"]:
            raise KeyError(path)
        out["media"] = [m for m in out["media"] if m[0] == path]
        return {"run": out["runs"][0], "media": out["media"]}

    def rows_json(self, path: str, frm: int, stop: int | None = None) -> tuple[str, int]:
        """(`rows` event JSON, row count) for rows [frm, stop) of the run file."""
        with self.lock:
            if path not in self.state:
                raise KeyError(path)
        c = connect_ro(self.run_dir(path))
        try:
            c.execute("BEGIN")
            rows = chunks.rows(c, frm, stop if stop is not None else chunks.row_count(c))
        finally:
            c.close()
        parts = [dumps([r.step, r.t, wire(r.values)]) for r in rows]
        return f'{{"run":{dumps(path)},"seq0":{frm},"rows":[{",".join(parts)}]}}', len(parts)

    def backfill(self, prefix: str) -> list[bytes]:
        """`rows` events for running runs' rows beyond their kept tiles."""
        with self.lock:
            tails = [(p, st["tiles_seq"], st["seq"]) for p, st in self.state.items()
                     if in_scope(p, prefix) and st["state"] == "running" and 0 < st["seq"] - st["tiles_seq"] <= ROWS_EVENT_MAX]
        out: list[bytes] = []
        for p, frm, stop in tails:
            try:
                text, n = self.rows_json(p, frm, stop)
            except READ_ERRORS:
                continue
            if n:
                out.append(sse_text("rows", text))
        return out

    def live_seqs(self, prefix: str) -> dict[str, tuple[int, int]]:
        """{path: (rows, media)} of running runs."""
        with self.lock:
            return {p: (st["seq"], st["mseq"]) for p, st in self.state.items()
                    if in_scope(p, prefix) and st["state"] == "running"}

    # ---- tiles ----

    def tiles(self, requests: Sequence[Sequence[object]]) -> list[list[bytes]]:
        """Tiles answering each request [path, key, "top" | "overview"] or [path, key, level, index];
        finer tiles are built on first request and cached."""
        out: list[list[bytes]] = []
        hits: list[tuple[str, str, int, int]] = []
        c = self.reader()
        try:
            for req in requests:
                path, key = str(req[0]), str(req[1])
                with self.lock:
                    st = self.state.get(path)
                if st is None or key not in st["keys"]:
                    out.append([])
                    continue
                if isinstance(req[2], str) and req[2] in KINDS:
                    out.append([r[0] for r in c.execute(
                        "SELECT data FROM tiles WHERE path=? AND key=? AND top=? ORDER BY idx", (path, key, KINDS[req[2]]))])
                    continue
                level, idx = _int(req[2]), _int(req[3])
                if not tiles.MIN_LEVEL <= level <= tiles.MAX_LEVEL:
                    raise ValueError(f"tile level {level} out of range")
                row = c.execute("SELECT seq, data FROM tiles WHERE path=? AND key=? AND level=? AND idx=?",
                                (path, key, level, idx)).fetchone()
                if row and (row[0] == st["seq"] or self._still_valid(path, level, idx, row[0])):
                    out.append([row[1]])
                    hits.append((path, key, level, idx))
                    continue
                blob, seq = self._build(path, key, level, idx)
                out.append([blob])
                self._store(path, key, level, idx, seq, blob)
        finally:
            self.release(c)
        if hits:
            with self._tile_lock:
                self._tile_w.executemany("UPDATE tiles SET used=? WHERE path=? AND key=? AND level=? AND idx=?",
                                         [(time.time(), *h) for h in hits])
        return out

    def tile_bundle(self, key: str, kind: TileKind, scope: str) -> list[tuple[str, list[bytes]]]:
        """[(path, tiles)] of one kept kind of one metric for every run under `scope`."""
        c = self.reader()
        try:
            rows = c.execute("SELECT path, data FROM tiles WHERE key=? AND top=? ORDER BY path, idx", (key, KINDS[kind])).fetchall()
        finally:
            self.release(c)
        out: list[tuple[str, list[bytes]]] = []
        for path, data in rows:
            if not in_scope(path, scope):
                continue
            if out and out[-1][0] == path:
                out[-1][1].append(data)
            else:
                out.append((path, [data]))
        return out

    def _still_valid(self, path: str, level: int, idx: int, seq: int) -> bool:
        """Whether every row after `seq` lies beyond the tile."""
        lo, hi = tiles.tile_range(level, idx)
        c = connect_ro(self.run_dir(path))
        try:
            first = c.execute("SELECT min(step_lo) FROM rowmeta WHERE seq0 + n > ?", (seq,)).fetchone()[0]
        finally:
            c.close()
        return first is None or first >= hi

    def _build(self, path: str, key: str, level: int, idx: int) -> tuple[bytes, int]:
        """(tile, run rows read) from the run file."""
        lo, hi = tiles.tile_range(level, idx)
        c = connect_ro(self.run_dir(path))
        try:
            c.execute("BEGIN")
            kid = c.execute("SELECT id FROM keys WHERE name=?", (key,)).fetchone()
            stop = chunks.row_count(c)
            if kid is None:
                e = np.empty(0)
                return tiles.build(e, e, e, level, idx), stop
            s, v, t = chunks.metric(c, kid[0], stop=stop, step_lo=lo, step_hi=hi)
        finally:
            c.close()
        return tiles.build(s, v, t, level, idx), stop

    def _store(self, path: str, key: str, level: int, idx: int, seq: int, blob: bytes) -> None:
        """Cache a built tile within the tile budget."""
        with self._tile_lock:
            old = self._tile_w.execute("SELECT length(data), top FROM tiles WHERE path=? AND key=? AND level=? AND idx=?",
                                       (path, key, level, idx)).fetchone()
            if old and old[1]:
                return
            self._tile_w.execute("INSERT OR REPLACE INTO tiles VALUES (?,?,?,?,?,?,?,?)",
                                 (path, key, level, idx, CACHED, seq, time.time(), blob))
            self._tile_bytes += len(blob) - (old[0] if old else 0)
        self.budget.enforce()

    def _oldest_cached(self) -> float | None:
        """When the least recently used cached tile was last used."""
        with self._tile_lock:
            return None if self._closed else self._tile_w.execute("SELECT min(used) FROM tiles WHERE top=?", (CACHED,)).fetchone()[0]

    def _evict(self, before: float, nbytes: int) -> int:
        """Evict cached tiles last used at or before `before`, oldest first, until `nbytes` are freed; how many."""
        with self._tile_lock:
            rows = self._tile_w.execute("SELECT path, key, level, idx, length(data) FROM tiles WHERE top=? AND used<=? "
                                        "ORDER BY used LIMIT 512", (CACHED, before)).fetchall()
            n = 0
            for p, k, lv, i, size in rows:
                if nbytes <= 0:
                    break
                self._tile_w.execute("DELETE FROM tiles WHERE path=? AND key=? AND level=? AND idx=?", (p, k, lv, i))
                self._tile_bytes -= size
                nbytes -= size
                n += 1
            return n

    def media_path(self, path: str, file: str) -> Path:
        """KeyError unless `file` is in the run's media/."""
        d = self.run_dir(path)
        f = (d / file).resolve()
        if f.parent != (d / "media").resolve():
            raise KeyError(file)
        return f


def _int(v: object) -> int:
    if isinstance(v, bool) or not isinstance(v, (int, float, str)):
        raise ValueError(f"expected an integer, got {v!r}")
    return int(v)

