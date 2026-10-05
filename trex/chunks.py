"""Row storage. A commit of rows [seq0, seq0 + n), n <= MAX_ROWS, is
    rowmeta(seq0, n, step_lo, step_hi, data)   data: f64 step[n], f64 t[n] (seconds since creation)
    chunk(key_id, seq0, data)                  per metric in the commit: u32 dense, u32 m,
                                               u16 pos[m] padded to 8 (unless dense), f64 value[m]
Readers find rows by the commits that overlap them, so a writer may merge adjacent commits (`prepare_merge`,
`apply_merge`).
"""

import itertools
import sqlite3
import struct
import sys
from array import array
from collections.abc import Iterable, Mapping, Sequence
from typing import Final, NamedTuple

import numpy as np
import numpy.typing as npt

type Floats = npt.NDArray[np.float64]

type CommitRow = tuple[float, float, Mapping[str, float]]
"""One row of a commit: (step, seconds since the run was created, {metric name: value})."""

type RowMeta = tuple[int, int, float, float, bytes]
"""(seq0, n, step_lo, step_hi, data)"""

type KeyChunk = tuple[int, bytes]
"""(key id, chunk)"""


class Row(NamedTuple):
    """One logged row read back."""

    seq: int
    step: float
    t: float
    values: dict[str, float]


class Series(NamedTuple):
    """One metric's points in row order."""

    steps: Floats
    values: Floats
    times: Floats


MAX_ROWS: Final = 65535
_HEAD: Final = struct.Struct("<II")  # chunk header: dense, m

SCHEMA = """
CREATE TABLE IF NOT EXISTS keys(id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE);
CREATE TABLE IF NOT EXISTS rowmeta(seq0 INTEGER PRIMARY KEY, n INTEGER NOT NULL, step_lo REAL, step_hi REAL,
                                   data BLOB NOT NULL);
CREATE TABLE IF NOT EXISTS chunk(id INTEGER PRIMARY KEY, key_id INTEGER NOT NULL, seq0 INTEGER NOT NULL,
                                 data BLOB NOT NULL);
CREATE UNIQUE INDEX IF NOT EXISTS chunk_key ON chunk(key_id, seq0);
"""

if sys.byteorder != "little":
    raise ImportError("trex chunks require a little-endian host")


type Insert = tuple[str, tuple[int | float | str | bytes, ...]]
"""(table, values) of one row a commit inserts."""


def inserts(seq0: int, rows: Sequence[CommitRow], ids: dict[str, int]) -> list[Insert]:
    """The keys, rowmeta and chunk rows of a commit; new names are added to `ids`."""
    n, known = len(rows), len(ids)
    if not 0 < n <= MAX_ROWS:
        raise ValueError(f"a commit holds 1..{MAX_ROWS} rows")
    steps = array("d", (float(r[0]) for r in rows))
    times = array("d", (float(r[1]) for r in rows))
    per: dict[int, tuple[array[int], array[float]]] = {}
    for i, (_, _, d) in enumerate(rows):
        for k, v in d.items():
            kid = ids.get(k)
            if kid is None:
                kid = ids[k] = len(ids)
            e = per.get(kid)
            if e is None:
                e = per[kid] = (array("H"), array("d"))
            e[0].append(i)
            e[1].append(v)
    return [*(("keys", (kid, k)) for k, kid in ids.items() if kid >= known),
            ("rowmeta", (seq0, n, min(steps), max(steps), steps.tobytes() + times.tobytes())),
            *(("chunk", (kid, seq0, _blob(None if len(pos) == n else pos.tobytes(), len(pos), vals.tobytes())))
              for kid, (pos, vals) in per.items())]


def _blob(positions: bytes | None, m: int, values: bytes) -> bytes:
    """A chunk of `m` values: dense (one per row of its commit) when `positions` is None."""
    if positions is None:
        return _HEAD.pack(1, m) + values
    return _HEAD.pack(0, m) + positions + b"\0" * (-len(positions) % 8) + values


def _values_offset(dense: bool, m: int) -> int:
    """Offset of the f64 values in a chunk of `m` values."""
    return 8 if dense else 8 + -(-2 * m // 8) * 8


def decode(blob: bytes) -> tuple[npt.NDArray[np.uint16] | None, Floats]:
    """A chunk's positions (rows within its commit; None when dense) and values, as views into `blob`."""
    dense, m = _HEAD.unpack_from(blob)
    pos = None if dense else np.frombuffer(blob, dtype="<u2", count=m, offset=8)
    return pos, np.frombuffer(blob, dtype="<f8", offset=_values_offset(dense, m))


def key_names(c: sqlite3.Connection) -> dict[int, str]:
    """{key id: name} of a run."""
    return {kid: name for kid, name in c.execute("SELECT id, name FROM keys")}


def row_count(c: sqlite3.Connection) -> int:
    """Rows covered by contiguous commits from seq 0."""
    n = 0
    for seq0, k in c.execute("SELECT seq0, n FROM rowmeta ORDER BY seq0"):
        if seq0 != n:
            break
        n += k
    return n


def metric(c: sqlite3.Connection, key_id: int, stop: int | None = None, step_lo: float | None = None,
           step_hi: float | None = None, start: int = 0) -> Series:
    """One metric's points in rows [start, stop), in commits meeting [step_lo, step_hi] if given."""
    q = "SELECT r.seq0, r.n, r.data, k.data FROM chunk k JOIN rowmeta r ON r.seq0 = k.seq0 WHERE k.key_id = ? AND r.seq0 + r.n > ?"
    args: list[float] = [key_id, start]
    if stop is not None:
        q += " AND r.seq0 < ?"
        args.append(stop)
    if step_lo is not None:
        q += " AND r.step_hi >= ?"
        args.append(step_lo)
    if step_hi is not None:
        q += " AND r.step_lo <= ?"
        args.append(step_hi)
    q += " ORDER BY k.seq0"
    S: list[Floats] = []
    V: list[Floats] = []
    T: list[Floats] = []
    for seq0, n, meta, blob in c.execute(q, args):
        rm = np.frombuffer(meta, dtype="<f8")
        s, t = rm[:n], rm[n:]
        idx, v = decode(blob)
        if idx is not None:
            s, t = s[idx], t[idx]
        if (stop is not None and stop - seq0 < n) or start > seq0:
            at: npt.NDArray[np.int64] = np.arange(n, dtype=np.int64) if idx is None else idx.astype(np.int64)
            keep = (at >= start - seq0) & (at < (n if stop is None else stop - seq0))
            s, v, t = s[keep], v[keep], t[keep]
        S.append(s)
        V.append(v)
        T.append(t)
    if not S:
        return Series(np.empty(0), np.empty(0), np.empty(0))
    return Series(np.concatenate(S), np.concatenate(V), np.concatenate(T))


def rows(c: sqlite3.Connection, start: int = 0, stop: int | None = None) -> list[Row]:
    """Rows [start, stop) with their metric values."""
    names = key_names(c)
    q = "SELECT seq0, n, data FROM rowmeta WHERE seq0 + n > ?" + (" AND seq0 < ?" if stop is not None else "") + " ORDER BY seq0"
    out: list[Row] = []
    for seq0, n, meta in c.execute(q, (start,) if stop is None else (start, stop)):
        mv = memoryview(meta).cast("d")
        part = [Row(seq0 + i, mv[i], mv[n + i], {}) for i in range(n)]
        for kid, blob in c.execute("SELECT key_id, data FROM chunk WHERE seq0 = ?", (seq0,)):
            pos, v = decode(blob)
            name = names[kid]
            for i, x in zip(range(n) if pos is None else pos.tolist(), v.tolist()):
                part[i].values[name] = x
        lo, hi = max(start - seq0, 0), (stop - seq0) if stop is not None else n
        out.extend(part[lo:hi])
    return out


def join_rowmeta(parts: Iterable[tuple[int, bytes]]) -> bytes:
    """Rowmeta data of consecutive commits ((rows, data), in row order) as one: every step, then every time."""
    steps: list[bytes] = []
    times: list[bytes] = []
    for k, d in parts:
        steps.append(d[: 8 * k])
        times.append(d[8 * k: 16 * k])
    return b"".join(steps) + b"".join(times)


def merged_rowmeta(metas: Sequence[RowMeta]) -> RowMeta:
    """The rowmeta row of consecutive commits merged into one."""
    return (metas[0][0], sum(m[1] for m in metas), min(m[2] for m in metas), max(m[3] for m in metas),
            join_rowmeta((m[1], m[4]) for m in metas))


def last_step_and_time(c: sqlite3.Connection, rows: int) -> tuple[float, float] | None:
    """Step and time of row `rows - 1`, from the commit that ends there; None when none does."""
    r: tuple[int, bytes] | None = c.execute("SELECT n, data FROM rowmeta WHERE seq0 + n = ?", (rows,)).fetchone()
    if r is None:
        return None
    n, data = r
    mv = memoryview(data).cast("d")
    return mv[n - 1], mv[2 * n - 1]


class Merge(NamedTuple):
    """Commits holding exactly rows [seq0, stop), read and rebuilt as one commit by `prepare_merge`."""

    seq0: int
    stop: int
    commits: list[tuple[int, int]]  # (seq0, rows) of the commits it replaces
    first_id: int  # the replaced commits' chunks have rowids from here on
    replaced_chunks: int
    rowmeta: RowMeta
    chunks: list[KeyChunk]  # of the merged commit
    values: int


def prepare_merge(c: sqlite3.Connection, seq0: int, stop: int, first_id: int = 0) -> Merge:
    """Read the commits holding exactly rows [seq0, stop) (whose chunks have rowids from `first_id` on) and build
    them as one commit; only reads. A metric dense in every commit keeps its value bytes as they are; any other is
    rebuilt, and raises unless it decodes, as readers decode it, to the same values in the same rows."""
    n = stop - seq0
    if not 0 < n <= MAX_ROWS:
        raise ValueError(f"a commit holds 1..{MAX_ROWS} rows")
    metas: list[RowMeta] = c.execute(
        "SELECT seq0, n, step_lo, step_hi, data FROM rowmeta WHERE seq0 >= ? AND seq0 < ? ORDER BY seq0", (seq0, stop)).fetchall()
    at = seq0
    for s, k, *_ in metas:
        if s != at:
            break
        at += k
    if at != stop or not metas:
        raise ValueError(f"rows [{seq0}, {stop}) are not whole contiguous commits")
    parts: list[tuple[int, int, bytes]] = c.execute(
        "SELECT key_id, seq0, data FROM chunk WHERE id >= ? AND seq0 >= ? AND seq0 < ? ORDER BY key_id, seq0",
        (first_id, seq0, stop)).fetchall()
    sizes = {s: k for s, k, *_ in metas}
    merged: list[KeyChunk] = []
    values = 0
    for kid, group in itertools.groupby(parts, key=lambda p: p[0]):
        blob = merge_chunks([(s, b) for _, s, b in group], sizes, seq0, n)
        merged.append((kid, blob))
        values += _HEAD.unpack_from(blob)[1]
    return Merge(seq0, stop, [(s, k) for s, k, *_ in metas], first_id, len(parts), merged_rowmeta(metas), merged, values)


def apply_merge(c: sqlite3.Connection, m: Merge) -> int:
    """Inside the caller's write transaction, replace m's commits with its merged commit if they are still exactly
    the commits it read; returns the rowid of the first chunk it inserts."""
    now = c.execute("SELECT seq0, n FROM rowmeta WHERE seq0 >= ? AND seq0 < ? ORDER BY seq0", (m.seq0, m.stop)).fetchall()
    if now != m.commits:
        raise RuntimeError(f"rows [{m.seq0}, {m.stop}) changed since they were read")
    gone = c.execute("DELETE FROM chunk WHERE id >= ? AND seq0 >= ? AND seq0 < ?", (m.first_id, m.seq0, m.stop)).rowcount
    if gone != m.replaced_chunks:
        raise RuntimeError(f"rows [{m.seq0}, {m.stop}) had {gone} chunks, not the {m.replaced_chunks} read")
    c.execute("DELETE FROM rowmeta WHERE seq0 >= ? AND seq0 < ?", (m.seq0, m.stop))
    c.execute("INSERT INTO rowmeta(seq0, n, step_lo, step_hi, data) VALUES (?, ?, ?, ?, ?)", m.rowmeta)
    first: int = c.execute("SELECT coalesce(max(id), 0) + 1 FROM chunk").fetchone()[0]
    c.executemany("INSERT INTO chunk(key_id, seq0, data) VALUES (?, ?, ?)", [(kid, m.seq0, blob) for kid, blob in m.chunks])
    return first


def merge_chunks(parts: Sequence[tuple[int, bytes]], sizes: Mapping[int, int], seq0: int, n: int) -> bytes:
    """One metric's chunk for the `n` rows from `seq0`, holding `parts` ((commit seq0, chunk), in row order) of
    commits with `sizes` rows: their value bytes as they are when every part is dense and together they fill the
    rows, else rebuilt, raising unless it decodes as readers decode it to the same values in the same rows."""
    blob = _dense(parts, sizes, n)
    if blob is None:
        blob, pos, vals = _merged(parts, seq0, n)
        _check(blob, pos, vals, n)
    return blob


def _dense(parts: Sequence[tuple[int, bytes]], sizes: Mapping[int, int], n: int) -> bytes | None:
    """The merged chunk when every part is dense and together they fill all `n` rows, else None."""
    total = 0
    for s, blob in parts:
        dense, m = _HEAD.unpack_from(blob)
        if not dense or m != sizes[s] or len(blob) != _values_offset(True, m) + 8 * m:
            return None
        total += m
    if total != n:
        return None
    return _HEAD.pack(1, n) + b"".join(blob[8:] for _, blob in parts)


def _merged(parts: Sequence[tuple[int, bytes]], seq0: int, n: int) -> tuple[bytes, npt.NDArray[np.int64], Floats]:
    """One chunk of the `n` rows from `seq0` holding the values of `parts` (commit seq0, chunk), in row order, with
    the rows and values it should decode to."""
    counts: list[int] = []
    sparse: list[bool] = []
    pos_bytes: list[bytes] = []
    val_bytes: list[bytes] = []
    for _, blob in parts:
        dense, m = _HEAD.unpack_from(blob)
        off = _values_offset(dense, m)
        counts.append(m)
        sparse.append(not dense)
        if not dense:
            pos_bytes.append(blob[8: 8 + 2 * m])
        val_bytes.append(blob[off: off + 8 * m])
    k = np.array(counts, dtype=np.int64)
    starts = np.cumsum(k) - k
    p = np.arange(int(k.sum()), dtype=np.int64) - np.repeat(starts, k)  # each part's rows 0..m-1, as if dense
    p[np.repeat(np.array(sparse), k)] = np.frombuffer(b"".join(pos_bytes), dtype="<u2")
    p += np.repeat(np.array([s - seq0 for s, _ in parts], dtype=np.int64), k)
    v = np.frombuffer(b"".join(val_bytes), dtype="<f8")
    return _blob(None if len(p) == n else p.astype("<u2").tobytes(), len(p), v.tobytes()), p, v


def _check(blob: bytes, pos: npt.NDArray[np.int64], vals: Floats, n: int) -> None:
    """Raise unless `blob` decodes to values `vals` at rows `pos` of an `n`-row commit."""
    got_pos, got_vals = decode(blob)
    got = np.arange(n, dtype=np.int64) if got_pos is None else got_pos.astype(np.int64)
    if not (len(got) == len(pos) and np.array_equal(got, pos) and got_vals.tobytes() == vals.tobytes()
            and len(pos) and bool(np.all(np.diff(pos) > 0)) and 0 <= pos[0] and pos[-1] < n):
        raise RuntimeError("a merged chunk would not read back as the chunks it replaces")
