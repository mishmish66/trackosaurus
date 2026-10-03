"""Row storage. A commit of rows [seq0, seq0 + n), n <= MAX_ROWS, is
    rowmeta(seq0, n, step_lo, step_hi, data)   data: f64 step[n], f64 t[n] (seconds since creation)
    chunk(key_id, seq0, data)                  per metric in the commit: u32 dense, u32 m,
                                               u16 pos[m] padded to 8 (unless dense), f64 value[m]
Readers find rows by the commits that overlap them, so a writer may merge adjacent commits (`merge`).
"""

import hashlib
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


def merge(c: sqlite3.Connection, seq0: int, stop: int) -> int:
    """Rewrite the commits holding exactly rows [seq0, stop) as one commit, inside the caller's transaction; returns
    how many commits it replaced. Raises, leaving the caller to roll back, unless every metric and the rows' steps
    and times read back byte for byte as before."""
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
    kids: list[int] = [kid for (kid,) in c.execute("SELECT id FROM keys")]
    before = _digest(c, kids, seq0, stop)
    steps = np.concatenate([np.frombuffer(d, dtype="<f8", count=k) for _, k, _, _, d in metas])
    times = np.concatenate([np.frombuffer(d, dtype="<f8", count=k, offset=8 * k) for _, k, _, _, d in metas])
    c.execute("DELETE FROM rowmeta WHERE seq0 >= ? AND seq0 < ?", (seq0, stop))
    c.execute("INSERT INTO rowmeta(seq0, n, step_lo, step_hi, data) VALUES (?, ?, ?, ?, ?)",
              (seq0, n, min(m[2] for m in metas), max(m[3] for m in metas), steps.tobytes() + times.tobytes()))
    for kid in kids:
        parts: list[tuple[int, bytes]] = c.execute(
            "SELECT seq0, data FROM chunk WHERE key_id = ? AND seq0 >= ? AND seq0 < ? ORDER BY seq0", (kid, seq0, stop)).fetchall()
        if not parts:
            continue
        c.execute("DELETE FROM chunk WHERE key_id = ? AND seq0 >= ? AND seq0 < ?", (kid, seq0, stop))
        c.execute("INSERT INTO chunk(key_id, seq0, data) VALUES (?, ?, ?)", (kid, seq0, _merged(parts, seq0, n)))
    if _digest(c, kids, seq0, stop) != before:
        raise RuntimeError(f"merging rows [{seq0}, {stop}) would change them")
    return len(metas)


def _merged(parts: Sequence[tuple[int, bytes]], seq0: int, n: int) -> bytes:
    """One chunk of the `n` rows from `seq0` holding the values of `parts` (commit seq0, chunk), in row order."""
    pos: list[npt.NDArray[np.int64]] = []
    vals: list[Floats] = []
    for s, blob in parts:
        dense, m = struct.unpack_from("<II", blob, 0)
        if dense:
            pos.append(np.arange(m, dtype=np.int64) + (s - seq0))
            vals.append(np.frombuffer(blob, dtype="<f8", offset=8))
        else:
            pos.append(np.frombuffer(blob, dtype="<u2", count=m, offset=8).astype(np.int64) + (s - seq0))
            vals.append(np.frombuffer(blob, dtype="<f8", offset=8 + -(-2 * m // 8) * 8))
    p, v = np.concatenate(pos), np.concatenate(vals)
    return _blob(None if len(p) == n else p.astype("<u2").tobytes(), len(p), v.tobytes())


def _digest(c: sqlite3.Connection, kids: Sequence[int], seq0: int, stop: int) -> bytes:
    """Hash of rows [seq0, stop) as readers see them: their steps and times, and every metric's points."""
    h = hashlib.sha256()
    for r in _row_steps(c, seq0, stop):
        h.update(r)
    for kid in kids:
        s, v, t = metric(c, kid, seq0, stop)
        h.update(struct.pack("<q", kid) + s.tobytes() + v.tobytes() + t.tobytes())
    return h.digest()


def _row_steps(c: sqlite3.Connection, start: int, stop: int) -> list[bytes]:
    """Steps then times of rows [start, stop), as f64 bytes."""
    steps: list[Floats] = []
    times: list[Floats] = []
    for seq0, n, data in c.execute("SELECT seq0, n, data FROM rowmeta WHERE seq0 + n > ? AND seq0 < ? ORDER BY seq0", (start, stop)):
        rm = np.frombuffer(data, dtype="<f8")
        lo, hi = max(start - seq0, 0), min(stop - seq0, n)
        steps.append(rm[lo:hi])
        times.append(rm[n + lo: n + hi])
    return [np.concatenate(steps).tobytes() if steps else b"", np.concatenate(times).tobytes() if times else b""]
