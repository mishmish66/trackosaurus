"""Compacting a run no process has open (`trex compact`): its commits merged into as few as fit, written to a new
file that replaces trex.sqlite atomically once it reads back the same. A run being written is refused."""

import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from . import chunks
from .format import DB, SCHEMA

MERGE_BYTES: Final = 8 << 20  # chunk bytes one merged commit holds at most
TMP: Final = DB + ".compacting"
LOCK_TIMEOUT: Final = 1.0  # seconds to wait for other connections to the run to close


@dataclass(frozen=True, slots=True)
class Compacted:
    """Commits and bytes of a run before and after `compact`."""

    commits_before: int
    commits_after: int
    bytes_before: int
    bytes_after: int


class InUse(RuntimeError):
    """Another process has the run open."""


def compact(run_dir: Path) -> Compacted:
    """Rewrite the run in `run_dir` with its commits merged. Holds an exclusive lock on the run throughout, so no
    writer or reader opening it in WAL mode runs meanwhile (InUse if one has it open); immutable readers keep reading
    the old file. The run is replaced only by a complete new file whose rows, metrics, meta, keys and media read back
    the same; anything else leaves it as it was."""
    db, tmp = run_dir / DB, run_dir / TMP
    tmp.unlink(missing_ok=True)
    before = _size(run_dir)
    src = _exclusive(db)
    try:
        src.execute("PRAGMA journal_mode=DELETE")  # folds the WAL in and removes it, so none outlives the old file
        groups, commits = _groups(src)
        dst = sqlite3.connect(tmp, isolation_level=None)
        try:
            _build(src, dst, groups)
            _verify(src, dst)
            _lock(dst)
            _fsync(tmp)
            os.replace(tmp, db)
            _fsync(run_dir)
        finally:
            dst.close()
    except BaseException:
        tmp.unlink(missing_ok=True)
        src.execute("PRAGMA journal_mode=WAL")
        raise
    finally:
        src.close()
    return Compacted(commits, len(groups), before, _size(run_dir))


def _exclusive(db: Path) -> sqlite3.Connection:
    c = sqlite3.connect(db, isolation_level=None, timeout=LOCK_TIMEOUT)
    try:
        _lock(c)
    except sqlite3.OperationalError as e:
        c.close()
        raise InUse(f"{db.parent} is open in another process (a live writer?)") from e
    return c


def _lock(c: sqlite3.Connection) -> None:
    """Hold an exclusive lock on c's database until c closes."""
    c.execute("PRAGMA locking_mode=EXCLUSIVE")
    c.execute("BEGIN EXCLUSIVE")
    c.execute("COMMIT")


def _groups(c: sqlite3.Connection) -> tuple[list[list[tuple[int, int]]], int]:
    """Consecutive commits ((seq0, rows)) to merge, each group within MAX_ROWS rows and MERGE_BYTES of chunks; and
    the number of commits. Raises if the commits do not run from row 0 without gaps or overlaps."""
    sizes: dict[int, int] = dict(c.execute("SELECT seq0, sum(length(data)) FROM chunk GROUP BY seq0").fetchall())
    groups: list[list[tuple[int, int]]] = []
    at = rows = size = 0
    commits = c.execute("SELECT seq0, n FROM rowmeta ORDER BY seq0").fetchall()
    for seq0, n in commits:
        if seq0 != at:
            raise ValueError(f"commits are not contiguous at row {at}")
        at += n
        b = sizes.get(seq0, 0)
        if not groups or rows + n > chunks.MAX_ROWS or size + b > MERGE_BYTES:
            groups.append([])
            rows = size = 0
        groups[-1].append((seq0, n))
        rows, size = rows + n, size + b
    return groups, len(commits)


def _build(src: sqlite3.Connection, dst: sqlite3.Connection, groups: list[list[tuple[int, int]]]) -> None:
    """Write the run into the empty database `dst`, one commit per group."""
    dst.execute("PRAGMA synchronous=FULL")
    dst.executescript(SCHEMA)
    dst.execute("BEGIN")
    _copy_tables(src, dst)
    _write_rowmeta(src, dst, groups)
    _write_chunks(src, dst, groups)
    dst.execute("COMMIT")


def _copy_tables(src: sqlite3.Connection, dst: sqlite3.Connection) -> None:
    """Meta, metric names and media, as they are."""
    for table in ("meta", "keys", "media"):
        rows = src.execute(f"SELECT * FROM {table}").fetchall()
        if rows:
            dst.executemany(f"INSERT INTO {table} VALUES ({', '.join('?' * len(rows[0]))})", rows)


def _write_rowmeta(src: sqlite3.Connection, dst: sqlite3.Connection, groups: list[list[tuple[int, int]]]) -> None:
    """One rowmeta row per group: its rows' steps, then their times."""
    metas = src.execute("SELECT seq0, n, step_lo, step_hi, data FROM rowmeta ORDER BY seq0")
    for members in groups:
        dst.execute("INSERT INTO rowmeta(seq0, n, step_lo, step_hi, data) VALUES (?, ?, ?, ?, ?)",
                    chunks.merged_rowmeta([next(metas) for _ in members]))


def _write_chunks(src: sqlite3.Connection, dst: sqlite3.Connection, groups: list[list[tuple[int, int]]]) -> None:
    """One chunk per metric per group it has values in, read a metric at a time."""
    group_of = {s: g for g, members in enumerate(groups) for s, _ in members}
    sizes = {s: k for members in groups for s, k in members}
    for (kid,) in src.execute("SELECT id FROM keys").fetchall():
        parts: dict[int, list[tuple[int, bytes]]] = {}
        for s, blob in src.execute("SELECT seq0, data FROM chunk WHERE key_id = ? ORDER BY seq0", (kid,)):
            parts.setdefault(group_of[s], []).append((s, blob))
        for g, group in parts.items():
            seq0, n = groups[g][0][0], sum(k for _, k in groups[g])
            dst.execute("INSERT INTO chunk(key_id, seq0, data) VALUES (?, ?, ?)",
                        (kid, seq0, chunks.merge_chunks(group, sizes, seq0, n)))


def _verify(src: sqlite3.Connection, dst: sqlite3.Connection) -> None:
    """Raise unless `dst` reads back as `src`: every table but the commits, the row count, every row's step and
    time, and every metric's points."""
    for table in ("meta", "keys", "media"):
        if sorted(src.execute(f"SELECT * FROM {table}").fetchall()) != sorted(dst.execute(f"SELECT * FROM {table}").fetchall()):
            raise RuntimeError(f"the compacted {table} table differs")
    n = chunks.row_count(src)
    if chunks.row_count(dst) != n or _steps(src) != _steps(dst):
        raise RuntimeError("the compacted rows differ")
    for (kid,) in src.execute("SELECT id FROM keys").fetchall():
        if [a.tobytes() for a in chunks.metric(src, kid, stop=n).columns] != [a.tobytes() for a in chunks.metric(dst, kid, stop=n).columns]:
            raise RuntimeError(f"metric {kid} differs once compacted")


def _steps(c: sqlite3.Connection) -> bytes:
    """Every row's step then time, in row order."""
    return chunks.join_rowmeta(c.execute("SELECT n, data FROM rowmeta ORDER BY seq0"))


def _size(run_dir: Path) -> int:
    return sum((run_dir / f).stat().st_size for f in (DB, DB + "-wal", DB + "-shm") if (run_dir / f).exists())


def _fsync(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
