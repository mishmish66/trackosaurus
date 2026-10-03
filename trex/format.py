"""Run format: a directory holding trex.sqlite (WAL) and media/.
    meta(key, value)                             JSON values: id, name, created, config, tags, info,
                                                 summary, state, heartbeat, format
    keys(id, name)                               metric names
    rowmeta, chunk                               rows, in commits numbered without gaps (chunks.py)
    media(seq, step, t, key, kind, file, size)   file relative to the run directory
Any folder may hold trex_info.json: notes about it.
"""

import os
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Final, Literal

from . import chunks, journal
from .chunks import key_names, row_count

__all__ = ["DB", "FORMAT", "INFO_FILE", "SCHEMA", "JSONValue", "MediaKind", "RunState", "as_dict", "as_float",
           "as_run_state", "as_str", "as_str_list", "connect_ro", "connect_rw", "key_names", "row_count", "snapshot"]

DB: Final = "trex.sqlite"
INFO_FILE: Final = "trex_info.json"
FORMAT: Final = 3

type MediaKind = Literal["image", "video", "html"]
type JSONValue = None | bool | int | float | str | list[JSONValue] | dict[str, JSONValue]

type RunState = Literal["running", "finished", "failed", "crashed"]
"""Readers report a running run whose heartbeat stopped as crashed."""

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS media(seq INTEGER PRIMARY KEY, step REAL NOT NULL, t REAL NOT NULL, key TEXT NOT NULL,
                                 kind TEXT NOT NULL, file TEXT NOT NULL, size INTEGER NOT NULL);
""" + chunks.SCHEMA


def connect_rw(run_dir: str | os.PathLike[str]) -> sqlite3.Connection:
    """Read-write connection, creating the database if missing."""
    c = sqlite3.connect(Path(run_dir) / DB, isolation_level=None, timeout=30)
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA synchronous=NORMAL")
    c.executescript(SCHEMA)
    return c


def connect_ro(run_dir: str | os.PathLike[str]) -> sqlite3.Connection:
    """Read-only connection that creates no files in the run directory: to the local replica of its journal
    while the run is live and journaled (`journal.sync`), through the WAL when a -wal file exists, else
    immutable (callers re-check the file signature after reading)."""
    db = Path(run_dir) / DB
    live = db.with_name(DB + "-wal").exists()
    if live and (Path(run_dir) / journal.JOURNAL).exists():
        db, live = journal.sync(Path(run_dir), SCHEMA), True
    mode = "mode=ro" if live else "mode=ro&immutable=1"
    c = sqlite3.connect(f"file:{db}?{mode}", uri=True, isolation_level=None, timeout=30)
    c.execute("PRAGMA query_only=1")
    return c


@contextmanager
def snapshot(run_dir: str | os.PathLike[str]) -> Generator[sqlite3.Connection, None, None]:
    """A read-only connection inside one read transaction."""
    c = connect_ro(run_dir)
    try:
        c.execute("BEGIN")
        yield c
    finally:
        c.close()


# ---- narrowing stored JSON values ----

def as_str(v: JSONValue) -> str | None:
    return v if isinstance(v, str) else None


def as_float(v: JSONValue) -> float | None:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def as_dict(v: JSONValue) -> dict[str, JSONValue]:
    return v if isinstance(v, dict) else {}


def as_str_list(v: JSONValue) -> list[str]:
    return [str(x) for x in v] if isinstance(v, list) else []


def as_run_state(v: JSONValue) -> RunState:
    """Unknown states read as running."""
    match v:
        case "finished" | "failed" | "crashed":
            return v
        case _:
            return "running"
