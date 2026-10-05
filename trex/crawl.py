"""A runs directory on this machine as an Explorer's origin. It walks the directory for runs, scans each one that
changed (inline, or in worker processes) for what the index lacks, and compiles its metrics' levels: from every row
while the run is short, and from then on only the blocks its new rows change (`buckets.grow`), every REFRESH seconds
while it grows. Rows the levels do not hold yet, and media files, come from the run files.
"""

import contextlib
import json
import os
import sqlite3
import sys
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from . import buckets as bk, chunks
from .buckets import Buckets
from .format import (DB, INFO_FILE, JSONValue, RunState, as_dict, as_float, as_run_state, as_str, as_str_list, snapshot)
from .index import (ROWS_EVENT_MAX, Block, Explorer, MediaRecord, Metric, Public, Rows, RunRecord, Sig, Summary, Update, mp_context, pack,
                    unpack, wire, worker_init)
from .journal import JOURNAL

CRASH_AFTER = 300.0  # seconds without a heartbeat after which a running run shows as crashed
POLL: Final = 1.0  # seconds between polls of known runs
REWALK: Final = 3.0  # seconds between walks of the root for new and removed runs
REFRESH = float(os.environ.get("TREX_REFRESH", "10"))  # seconds between compiles of a growing run's levels
REBUILD_BELOW: Final = 4096  # a run with fewer rows compiled is compiled from every row, so its finest level follows its steps
SKIP_DIRS: Final = frozenset({"node_modules", "__pycache__"})
INLINE_BYTES = 5 << 20  # polls whose run files grew by at most this many bytes are read in the main process
BATCH_RUNS: Final = 256  # updates per index transaction
BATCH_SECONDS: Final = 0.5  # longest wait before committing a partial batch


@dataclass(frozen=True, slots=True)
class Prev:
    """The index's state of a run when its scan starts."""

    uid: str
    seq: int
    mseq: int
    compiled: int
    compiled_t: float
    rebuilt: int


@dataclass(frozen=True, slots=True, kw_only=True)
class Job:
    """A run to scan and how: `want_rows` when a browser watches it, `refresh` seconds between compiles while it grows,
    `index` the index file holding its stored levels."""

    path: str
    dir: str
    sig: Sig
    prev: Prev | None
    crash_after: float
    refresh: float
    want_rows: bool
    index: str


def stat_sig(d: str | os.PathLike[str]) -> Sig:
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


def rows_event(path: str, start: int, rows: Sequence[chunks.Row]) -> str:
    """JSON of the `rows` event for `rows` of run `path`, the first of them row `start`."""
    return Rows(path, start, [[r.step, r.t, wire(r.values)] for r in rows]).text()


# ---- per-run reading (inline or in a worker process) ----


@dataclass(frozen=True, slots=True)
class Compiled:
    """Levels to write for a run: its metrics, their blocks that changed, and whether they are all the run has."""

    metrics: list[Metric]
    blocks: list[Block]
    replace: bool


def scan(job: Job) -> Update | None:
    """What changed in a run since `job.prev`, or None to retry later (no id yet, or changed while read without a WAL)."""
    d = Path(job.dir)
    with snapshot(d) as c:
        meta: dict[str, JSONValue] = {k: json.loads(v) for k, v in c.execute("SELECT key, value FROM meta")}
        update = _changes(c, d, job, meta) if "id" in meta else None
    after = stat_sig(d)
    return update if after == job.sig or (job.sig[2] and after[2]) else None


def _changes(c: sqlite3.Connection, d: Path, job: Job, meta: Mapping[str, JSONValue]) -> Update:
    """What the run file holds that the index, at `job.prev`, does not."""
    state, heartbeat = _state(meta, job.crash_after)
    seq = chunks.row_count(c)
    media_count: int = c.execute("SELECT coalesce(max(seq) + 1, 0) FROM media").fetchone()[0]
    prev = None if job.prev is None or _rewritten(job.prev, meta["id"], seq, media_count) else job.prev
    names = chunks.key_names(c)
    levels = _compile(c, names, job, prev, seq) if _due(prev, seq, state, job.refresh) else None
    mseq = prev.mseq if prev else 0
    media = _new_media(c, d, job.path, mseq)
    return Update(path=job.path, sig=job.sig, uid=str(meta["id"]), reset=job.prev is not None and prev is None,
                  fresh=prev is None, seq=seq, mseq=mseq + len(media), media=media, state=state, heartbeat=heartbeat,
                  public=_public(meta), keys=sorted(names.values()),
                  summary=_summary(c, names, seq) if prev is None or seq != prev.seq else None,
                  compiled=seq if levels else None, rebuilt=seq if prev is None or (levels and levels.replace) else prev.rebuilt,
                  metrics=levels.metrics if levels else [], blocks=levels.blocks if levels else [],
                  replace=bool(levels and levels.replace), rows=_rows(c, job, prev, seq))


def _rows(c: sqlite3.Connection, job: Job, prev: Prev | None, seq: int) -> str | None:
    """The `rows` event of the rows since `prev`, for watching browsers; none for a large catch-up (newly compiled
    levels serve it)."""
    if prev is None or not job.want_rows or not 0 < seq - prev.seq <= ROWS_EVENT_MAX:
        return None
    return rows_event(job.path, prev.seq, chunks.rows(c, prev.seq, seq))


def _state(meta: Mapping[str, JSONValue], crash_after: float) -> tuple[RunState, float | None]:
    """(state, heartbeat); a running run silent for `crash_after` seconds is crashed."""
    state, heartbeat = as_run_state(meta.get("state")), as_float(meta.get("heartbeat"))
    if state == "running" and time.time() - (heartbeat or 0) > crash_after:
        state = "crashed"
    return state, heartbeat


def _last_values(c: sqlite3.Connection, names: Mapping[int, str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for kid, name in names.items():
        r = c.execute("SELECT data FROM chunk WHERE key_id = ? ORDER BY seq0 DESC LIMIT 1", (kid,)).fetchone()
        if r:
            vals = chunks.decode(r[0])[1]
            if len(vals):
                out[name] = float(vals[-1])
    return out


def _summary(c: sqlite3.Connection, names: Mapping[int, str], seq: int) -> Summary:
    out = wire(_last_values(c, names))
    if (last := chunks.last_step_and_time(c, seq)) is not None:
        out["_step"], out["_runtime"] = last
    return out


def _due(prev: Prev | None, seq: int, state: RunState, refresh: float) -> bool:
    """Levels are compiled for a new run, and for one with new rows when it stopped running or `refresh` passed."""
    if prev is None:
        return True
    return prev.compiled != seq and (state != "running" or time.time() - prev.compiled_t >= refresh)


def _compile(c: sqlite3.Connection, names: Mapping[int, str], job: Job, prev: Prev | None, seq: int) -> Compiled:
    """Every metric's levels over rows [0, seq), as index rows: all of them for a run that is new or has fewer than
    REBUILD_BELOW rows compiled; else the blocks that the rows since `prev` change, given the index's."""
    if prev is None or prev.compiled < REBUILD_BELOW:
        return Compiled(*_levels(c, names, seq, lambda key: None, 0), replace=True)
    with contextlib.closing(sqlite3.connect(Path(job.index).as_uri() + "?mode=ro", uri=True)) as index:
        return Compiled(*_levels(c, names, seq, lambda key: _Stored(index, job.path, key), prev.compiled), replace=False)


class _Stored:
    """A run's stored levels of one metric, read from the index."""

    def __init__(self, index: sqlite3.Connection, path: str, key: str) -> None:
        self.index, self.at = index, (key, path)
        row = index.execute("SELECT fine, top, lo, hi FROM metrics WHERE key=? AND path=?", self.at).fetchone()
        self.span = bk.Span(row[0], row[1], row[2], row[3]) if row else None

    def block(self, level: int, block: int) -> Buckets:
        row = self.index.execute("SELECT data FROM levels WHERE key=? AND path=? AND level=? AND block=?", (*self.at, level, block)).fetchone()
        return unpack(row[0]) if row else bk.empty()

    def held(self, level: int) -> list[int]:
        return [b for (b,) in self.index.execute("SELECT block FROM levels WHERE key=? AND path=? AND level=?", (*self.at, level))]


def _levels(c: sqlite3.Connection, names: Mapping[int, str], seq: int, stored: Callable[[str], _Stored | None],
            since: int) -> tuple[list[Metric], list[Block]]:
    """Each metric's levels over rows [0, seq): compiled from every row (`buckets.pyramid`) where `stored` has none of
    it, else the blocks its rows from `since` change (`buckets.grow`); each metric read once."""
    metrics: list[Metric] = []
    blocks: list[Block] = []
    for kid, key in sorted(names.items(), key=lambda kv: kv[1]):
        old = stored(key)
        m = chunks.metric(c, kid, stop=seq, start=since if old and old.span else 0)
        if not m.steps.size:
            continue
        if old and old.span:
            span, parts = bk.grow(old.span, m.steps, m.values, m.times, old.block, old.held)
        else:
            span, parts = bk.pyramid(m.steps, m.values, m.times)
        metrics.append(Metric(key, span))
        blocks += [Block(key, p.level, p.block, seq, pack(p.buckets, p.level, p.block)) for p in parts]
    return metrics, blocks


def _rewritten(prev: Prev, uid: JSONValue, rows: int, media: int) -> bool:
    """The run file was replaced: a new id, or fewer rows or media than indexed."""
    return prev.uid != uid or rows < prev.seq or media < prev.mseq


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


def _public(meta: Mapping[str, JSONValue]) -> Public:
    return Public(name=as_str(meta.get("name")), tags=as_str_list(meta.get("tags")), config=as_dict(meta.get("config")),
                  created=as_float(meta.get("created")), info=as_dict(meta.get("info")),
                  user_summary=as_dict(meta.get("summary")))


def _needs_scan(st: RunRecord, sig: Sig, now: float) -> bool:
    """Its files changed, it went silent while running, or its levels are due a compile."""
    silent = st.state == "running" and now - (st.heartbeat or now) > CRASH_AFTER
    due = st.compiled != st.seq and now - st.compiled_t >= REFRESH
    return sig != st.sig or silent or due


def _bytes_of(sig: Sig) -> int:
    return sig[1] + sig[3]


class _Batch:
    """Updates awaiting one index transaction."""

    def __init__(self, ex: Explorer) -> None:
        self.ex = ex
        self.updates: list[Update] = []
        self.t0 = time.time()

    def add(self, r: Update | None) -> None:
        if r is not None:
            self.updates.append(r)

    def full(self) -> bool:
        return len(self.updates) >= BATCH_RUNS

    def commit(self) -> None:
        if self.updates:
            self.ex.apply(self.updates)
        self.updates, self.t0 = [], time.time()


def default_workers() -> int:
    """$TREX_WORKERS, else the CPU count up to 32."""
    env = os.environ.get("TREX_WORKERS")
    return max(1, int(env)) if env else min(32, os.cpu_count() or 1)


class Crawl:
    """The runs directory `root` as an Explorer's origin, scanning on up to `workers` processes."""

    def __init__(self, root: str | os.PathLike[str], workers: int | None = None) -> None:
        self.root = Path(root).resolve()
        self.key = str(self.root)
        self.workers = max(1, int(workers)) if workers is not None else default_workers()
        self.dirs: dict[str, Path] = {}  # run -> its directory
        self._notes: dict[str, tuple[int, dict[str, JSONValue]]] = {}  # folder -> (mtime_ns, notes) of its trex_info.json

    def attach(self, ex: Explorer) -> None:
        pass

    def info(self) -> dict[str, object]:
        return {"root": str(self.root), "name": self.root.name}

    def run(self, ex: Explorer) -> None:
        """Until `ex` stops: walk every REWALK seconds and poll every POLL; a failed pass is logged and the next one
        runs."""
        last_walk = 0.0
        while not ex.stopped:
            t0 = time.time()
            try:
                if t0 - last_walk >= REWALK:
                    self.rewalk(ex)
                    last_walk = t0
                self.poll(ex)
            except Exception as e:
                print(f"[trex] {self.root}: index pass failed: {e!r}", file=sys.stderr, flush=True)
            ex.ready.set()
            ex.wait(max(0.0, POLL - (time.time() - t0)))

    def sync(self, ex: Explorer) -> list[str]:
        self.rewalk(ex)
        return self.poll(ex)

    def stop(self) -> None:
        pass

    def close(self) -> None:
        pass

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

    def rewalk(self, ex: Explorer) -> None:
        """Walk the root: runs that are gone leave the index, and the folders' notes are those of their trex_info.json
        files (a file that cannot be read keeps the notes it had)."""
        found, infos = self.walk()
        with ex.lock:
            gone = [p for p in ex.records if p not in found]
        self.dirs = found
        for p in gone:
            ex.drop(p)
        notes: dict[str, tuple[int, dict[str, JSONValue]]] = {}
        for p, f in infos.items():
            try:
                mtime = f.stat().st_mtime_ns
                if p in self._notes and self._notes[p][0] == mtime:
                    notes[p] = self._notes[p]
                    continue
                info = json.loads(f.read_text())
                notes[p] = (mtime, info if isinstance(info, dict) else {"value": info})
            except (OSError, ValueError) as e:
                print(f"[trex] {f}: {e!r}", file=sys.stderr, flush=True)
                if p in self._notes:
                    notes[p] = self._notes[p]
        self._notes = notes
        ex.keep_folders({p: info for p, (_, info) in notes.items()})

    def poll(self, ex: Explorer) -> list[str]:
        """Rescan runs that changed, went silent, or are due a compile; their paths."""
        now = time.time()
        todo: list[tuple[str, Path, Sig]] = []
        growth = 0
        for path, d in list(self.dirs.items()):
            st, sig = ex.records.get(path), stat_sig(d)
            if st is None or _needs_scan(st, sig, now):
                todo.append((path, d, sig))
                growth += _bytes_of(sig) - (_bytes_of(st.sig) if st else 0)
        if self.workers > 1 and len(todo) > 1 and growth > INLINE_BYTES:
            self._sync_pool(ex, todo)
        else:
            self._sync_inline(ex, todo)
        return [p for p, _, _ in todo]

    def _job(self, ex: Explorer, path: str, d: Path, sig: Sig) -> Job:
        st = ex.records.get(path)
        prev = Prev(st.uid, st.seq, st.mseq, st.compiled, st.compiled_t, st.rebuilt) if st else None
        return Job(path=path, dir=str(d), sig=sig, prev=prev, crash_after=CRASH_AFTER, refresh=REFRESH,
                   want_rows=ex.hub.watched(path), index=str(ex.db_path))

    def _sync_inline(self, ex: Explorer, todo: Sequence[tuple[str, Path, Sig]]) -> None:
        batch = _Batch(ex)
        for t in todo:
            if ex.stopped:
                break
            j = self._job(ex, *t)
            try:
                batch.add(scan(j))
            except Exception as e:
                print(f"[trex] {j.path}: {e!r}", file=sys.stderr, flush=True)
            if batch.full():
                batch.commit()
        batch.commit()

    def _sync_pool(self, ex: Explorer, todo: Sequence[tuple[str, Path, Sig]]) -> None:
        jobs = deque(self._job(ex, *t) for t in todo)
        batch = _Batch(ex)
        pool = ProcessPoolExecutor(max_workers=min(self.workers, len(todo)), mp_context=mp_context(), initializer=worker_init)
        pending: dict[Future[Update | None], Job] = {}
        try:
            while (jobs or pending) and not ex.stopped:
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
                        print(f"[trex] {j.path}: {e!r}", file=sys.stderr, flush=True)
                if batch.full() or not (jobs or pending) or time.time() - batch.t0 >= BATCH_SECONDS:
                    batch.commit()
        except BrokenProcessPool as e:
            print(f"[trex] index worker died: {e!r}", file=sys.stderr, flush=True)
        finally:
            pool.shutdown(wait=not ex.stopped, cancel_futures=True)
        batch.commit()

    def run_dir(self, path: str) -> Path:
        d = self.dirs.get(path)
        if d is None:
            raise KeyError(path)
        return d

    # ---- what only the run files have ----

    def rows_json(self, path: str, start: int) -> str:
        with snapshot(self.run_dir(path)) as c:
            return rows_event(path, start, chunks.rows(c, start))

    def live(self, path: str, rec: RunRecord) -> tuple[int, int]:
        return rec.seq, rec.mseq

    def media_file(self, path: str, file: str) -> Path:
        d = self.run_dir(path)
        f = (d / file).resolve()
        if f.parent != (d / "media").resolve():
            raise KeyError(file)
        return f
