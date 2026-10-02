"""Row storage. A commit of rows [seq0, seq0 + n), n <= MAX_ROWS, is
    rowmeta(seq0, n, step_lo, step_hi, data)   data: f64 step[n], f64 t[n] (seconds since creation)
    chunk(key_id, seq0, data)                  per metric in the commit: u32 dense, u32 m,
                                               u16 pos[m] padded to 8 (unless dense), f64 value[m]
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
    out: list[tuple[int, bytes]] = []
    for kid, (pos, vals) in per.items():
        if len(pos) == n:
            out.append((kid, struct.pack("<II", 1, n) + vals.tobytes()))
        else:
            p = pos.tobytes()
            out.append((kid, struct.pack("<II", 0, len(pos)) + p + b"\0" * (-len(p) % 8) + vals.tobytes()))
    return (n, min(steps), max(steps), steps.tobytes() + times.tobytes()), out


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
