"""Row storage. A commit of rows [seq0, seq0 + n), n <= MAX_ROWS, is
    rowmeta(seq0, n, step_lo, step_hi, data)   data: f64 step[n], f64 t[n] (seconds since creation)
    chunk(key_id, seq0, data)                  per metric in the commit: u32 dense, u32 m,
                                               u16 pos[m] padded to 8 (unless dense), f64 value[m]
Readers find rows by the commits that overlap them, so a writer may merge adjacent commits (`merge`).
"""

import sqlite3
import struct
import sys
from array import array
from collections.abc import Mapping, Sequence
from typing import Final, NamedTuple

import numpy as np
import numpy.typing as npt

type Floats = npt.NDArray[np.float64]

type CommitRow = tuple[float, float, Mapping[str, float]]
"""One row of a commit: (step, seconds since the run was created, {metric name: value})."""


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


type _IntView = memoryview[int]
type _FloatView = memoryview[float]


class Chunk(NamedTuple):
    """One metric's values in one commit; `positions` (rows within the commit) is None when dense."""

    dense: bool
    positions: _IntView | None
    values: _FloatView

MAX_ROWS: Final = 65535

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


def encode(rows: Sequence[CommitRow], ids: dict[str, int]) -> tuple[tuple[int, float, float, bytes], list[tuple[int, bytes]]]:
    """(rowmeta fields, [(key id, chunk)]) of a commit; new names are added to `ids`."""
    n = len(rows)
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
    out = [(kid, _blob(None if len(pos) == n else pos.tobytes(), len(pos), vals.tobytes())) for kid, (pos, vals) in per.items()]
    return (n, min(steps), max(steps), steps.tobytes() + times.tobytes()), out


def _blob(positions: bytes | None, m: int, values: bytes) -> bytes:
    """A chunk of `m` values: dense (one per row of its commit) when `positions` is None."""
    if positions is None:
        return struct.pack("<II", 1, m) + values
    return struct.pack("<II", 0, m) + positions + b"\0" * (-len(positions) % 8) + values


def decode(blob: bytes) -> Chunk:
    """One chunk's layout, as memoryviews into `blob` (positions u16, values f64)."""
    dense, m = struct.unpack_from("<II", blob, 0)
    mv = memoryview(blob)
    if dense:
        return Chunk(True, None, mv[8:].cast("d"))
    pos = mv[8: 8 + 2 * m].cast("H")
    off = 8 + -(-2 * m // 8) * 8
    return Chunk(False, pos, mv[off:].cast("d"))


type Insert = tuple[str, tuple[int | float | str | bytes, ...]]
"""(table, values) of one row a commit inserts."""


def inserts(seq0: int, rows: Sequence[CommitRow], ids: dict[str, int]) -> list[Insert]:
    """The keys, rowmeta and chunk rows of a commit; new names are added to `ids`."""
    known = len(ids)
    (n, lo, hi, meta), parts = encode(rows, ids)
    return [*(("keys", (kid, k)) for k, kid in ids.items() if kid >= known), ("rowmeta", (seq0, n, lo, hi, meta)),
            *(("chunk", (kid, seq0, b)) for kid, b in parts)]


def write(c: sqlite3.Connection, seq0: int, rows: Sequence[CommitRow], ids: dict[str, int]) -> int:
    """Insert a commit inside the caller's transaction; returns its row count."""
    for table, values in inserts(seq0, rows, ids):
        c.execute(f"INSERT INTO {INSERT_INTO[table]} VALUES ({', '.join('?' * len(values))})", values)
    return len(rows)


INSERT_INTO: Final = {"keys": "keys(id, name)", "rowmeta": "rowmeta(seq0, n, step_lo, step_hi, data)",
                      "chunk": "chunk(key_id, seq0, data)"}


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


def metric(c: sqlite3.Connection, key_id: int, start: int = 0, stop: int | None = None,
           step_lo: float | None = None, step_hi: float | None = None) -> Series:
    """One metric's points over rows [start, stop), in commits meeting [step_lo, step_hi] if given."""
    q = ("SELECT r.seq0, r.n, r.data, k.data FROM chunk k JOIN rowmeta r ON r.seq0 = k.seq0 "
         "WHERE k.key_id = ? AND r.seq0 + r.n > ?")
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
        dense, m = struct.unpack_from("<II", blob, 0)
        if dense:
            v = np.frombuffer(blob, dtype="<f8", offset=8)
            idx = None
        else:
            idx = np.frombuffer(blob, dtype="<u2", count=m, offset=8)
            v = np.frombuffer(blob, dtype="<f8", offset=8 + -(-2 * m // 8) * 8)
            s, t = s[idx], t[idx]
        lo, hi = max(start - seq0, 0), (stop - seq0) if stop is not None else n
        if lo > 0 or hi < n:
            at: npt.NDArray[np.int64] = np.arange(n, dtype=np.int64) if idx is None else idx.astype(np.int64)
            keep = (at >= lo) & (at < hi)
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
            ch = decode(blob)
            name = names[kid]
            if ch.positions is None:
                for i in range(n):
                    part[i].values[name] = ch.values[i]
            else:
                for i, p in enumerate(ch.positions):
                    part[p].values[name] = ch.values[i]
        lo, hi = max(start - seq0, 0), (stop - seq0) if stop is not None else n
        out.extend(part[lo:hi])
    return out



class Merge(NamedTuple):
    """Commits holding exactly rows [seq0, stop), read and rebuilt as one commit by `prepare_merge`."""

    seq0: int
    stop: int
    commits: list[tuple[int, int]]  # (seq0, rows) of the commits it replaces
    first_id: int  # the replaced commits' chunks have rowids from here on
    replaced_chunks: int
    rowmeta: tuple[int, int, float, float, bytes]
    chunks: list[tuple[int, bytes]]  # (key id, chunk) of the merged commit
    values: int


def prepare_merge(c: sqlite3.Connection, seq0: int, stop: int, first_id: int = 0) -> Merge:
    """Read the commits holding exactly rows [seq0, stop) (whose chunks have rowids from `first_id` on) and build
    them as one commit; only reads. A metric dense in every commit keeps its value bytes as they are; any other is
    rebuilt, and raises unless it decodes, as readers decode it, to the same values in the same rows."""
    n = stop - seq0
    if not 0 < n <= MAX_ROWS:
        raise ValueError(f"a commit holds 1..{MAX_ROWS} rows")
    metas: list[tuple[int, int, float, float, bytes]] = c.execute(
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
    steps = b"".join(d[: 8 * k] for _, k, _, _, d in metas)
    times = b"".join(d[8 * k: 16 * k] for _, k, _, _, d in metas)
    rowmeta = (seq0, n, min(m[2] for m in metas), max(m[3] for m in metas), steps + times)
    sizes = {s: k for s, k, *_ in metas}
    merged: list[tuple[int, bytes]] = []
    values = 0
    for kid, group in _by_key(parts):
        blob = _dense(group, sizes, n)
        if blob is None:
            blob, pos, vals = _merged(group, seq0, n)
            _check(blob, pos, vals, n)
        merged.append((kid, blob))
        values += struct.unpack_from("<II", blob)[1]
    return Merge(seq0, stop, [(s, k) for s, k, *_ in metas], first_id, len(parts), rowmeta, merged, values)


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


def merge(c: sqlite3.Connection, seq0: int, stop: int) -> int:
    """`prepare_merge` and `apply_merge` inside the caller's write transaction; returns how many commits it replaced."""
    m = prepare_merge(c, seq0, stop)
    apply_merge(c, m)
    return len(m.commits)


def _by_key(parts: Sequence[tuple[int, int, bytes]]) -> list[tuple[int, list[tuple[int, bytes]]]]:
    out: list[tuple[int, list[tuple[int, bytes]]]] = []
    for kid, s, blob in parts:
        if not out or out[-1][0] != kid:
            out.append((kid, []))
        out[-1][1].append((s, blob))
    return out


def _dense(parts: Sequence[tuple[int, bytes]], sizes: Mapping[int, int], n: int) -> bytes | None:
    """The merged chunk when every part is dense and together they fill all `n` rows, else None."""
    total = 0
    for s, blob in parts:
        dense, m = struct.unpack_from("<II", blob)
        if not dense or m != sizes[s] or len(blob) != 8 + 8 * m:
            return None
        total += m
    if total != n:
        return None
    return struct.pack("<II", 1, n) + b"".join(blob[8:] for _, blob in parts)


def _merged(parts: Sequence[tuple[int, bytes]], seq0: int, n: int) -> tuple[bytes, npt.NDArray[np.int64], Floats]:
    """One chunk of the `n` rows from `seq0` holding the values of `parts` (commit seq0, chunk), in row order, with
    the rows and values it should decode to."""
    pos: list[npt.NDArray[np.int64]] = []
    vals: list[Floats] = []
    for s, blob in parts:
        ch = decode(blob)
        at = np.arange(len(ch.values), dtype=np.int64) if ch.positions is None else np.frombuffer(ch.positions, dtype="<u2").astype(np.int64)
        pos.append(at + (s - seq0))
        vals.append(np.frombuffer(ch.values, dtype="<f8"))
    p, v = np.concatenate(pos), np.concatenate(vals)
    return _blob(None if len(p) == n else p.astype("<u2").tobytes(), len(p), v.tobytes()), p, v


def _check(blob: bytes, pos: npt.NDArray[np.int64], vals: Floats, n: int) -> None:
    """Raise unless `blob` decodes to values `vals` at rows `pos` of an `n`-row commit."""
    ch = decode(blob)
    got = np.arange(n, dtype=np.int64) if ch.positions is None else np.frombuffer(ch.positions, dtype="<u2").astype(np.int64)
    if not (len(got) == len(pos) and np.array_equal(got, pos) and bytes(ch.values) == vals.tobytes()
            and len(pos) and bool(np.all(np.diff(pos) > 0)) and 0 <= pos[0] and pos[-1] < n):
        raise RuntimeError("a merged chunk would not read back as the chunks it replaces")
