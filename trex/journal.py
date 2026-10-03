"""The commit journal: an append-only, fsync'd copy of a run's commits beside trex.sqlite.

SQLite's WAL can be read only on the host that writes it, so a live run on a network filesystem (a cluster's
shared storage) cannot be read from elsewhere. A journal can: records are appended and fsync'd, each one
length-prefixed with a CRC-32 so a reader stops at a record not yet complete. Readers replay it into a
replica on local disk (`sync`), which `format.connect_ro` reads instead of the run's live database.

    record   u32 length, JSON payload, u32 crc32(payload)
    payload  {"journal": id, "run": run id, "host": h}   first record of a journal
             {"session": h}                              a reopening of the run, on host h
             {"ops": [[table, [values]], ...], "seq": rows after, "mseq": media after}
             values are JSON scalars, or {"$b": base64} for bytes
"""

import base64
import contextlib
import fcntl
import hashlib
import json
import os
import socket
import sqlite3
import struct
import tempfile
import uuid
import zlib
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Final, NamedTuple, cast

JOURNAL: Final = "trex.journal"
NETWORK_FS: Final = frozenset({"nfs", "nfs4", "cifs", "smb3", "smbfs", "lustre", "gpfs", "ceph", "beegfs", "glusterfs",
                               "fuse.sshfs", "9p", "afs", "fuse.glusterfs", "fuse.ceph", "wekafs", "panfs"})
TABLES: Final[dict[str, tuple[str, ...]]] = {  # columns of each table, replayed with INSERT OR REPLACE
    "meta": ("key", "value"),
    "keys": ("id", "name"),
    "rowmeta": ("seq0", "n", "step_lo", "step_hi", "data"),
    "chunk": ("key_id", "seq0", "data"),
    "media": ("seq", "step", "t", "key", "kind", "file", "size"),
}
_HEAD: Final = struct.Struct("<I")

type Value = None | int | float | str | bytes
type Op = tuple[str, Sequence[Value]]


class Record(NamedTuple):
    start: int  # offset of the record in the journal
    end: int  # offset just past it
    payload: dict[str, object]


def host() -> str:
    return socket.gethostname()


def on_network_fs(path: Path, mountinfo: str | None = None) -> bool:
    """Whether `path` is on a network filesystem (Linux; elsewhere False). The path is touched first, which mounts
    an automounted filesystem it lies on."""
    if mountinfo is None:
        try:
            os.stat(path)
            mountinfo = Path("/proc/self/mountinfo").read_text()
        except OSError:
            return False
    target, best, kind = os.path.realpath(path), "", ""
    for line in mountinfo.splitlines():
        fields = line.split(" - ")
        if len(fields) != 2:
            continue
        mount, fstype = fields[0].split(" ")[4].replace("\\040", " "), fields[1].split(" ")[0]
        if (target == mount or target.startswith(mount.rstrip("/") + "/")) and len(mount) > len(best):
            best, kind = mount, fstype
    return kind in NETWORK_FS


def wanted(run_dir: Path) -> bool:
    """Whether a writer journals `run_dir`: $TREX_JOURNAL=1 or 0, else when it is on a network filesystem."""
    flag = os.environ.get("TREX_JOURNAL", "")
    return flag == "1" or (flag != "0" and on_network_fs(run_dir))


def _encode(payload: dict[str, object]) -> bytes:
    def default(v: object) -> object:
        if isinstance(v, (bytes, bytearray)):
            return {"$b": base64.b64encode(v).decode()}
        raise TypeError(f"cannot journal {type(v).__name__}")

    body = json.dumps(payload, separators=(",", ":"), default=default).encode()
    return _HEAD.pack(len(body)) + body + _HEAD.pack(zlib.crc32(body))


def _value(v: object) -> object:
    if isinstance(v, dict):
        return base64.b64decode(cast(dict[str, str], v)["$b"])
    return v


def records(data: bytes, offset: int = 0) -> Iterator[Record]:
    """The complete records of `data` from `offset`; stops at a partial or corrupt one."""
    while offset + 8 <= len(data):
        (n,) = _HEAD.unpack_from(data, offset)
        end = offset + 8 + n
        body = data[offset + 4: offset + 4 + n]
        if end > len(data) or zlib.crc32(body) != _HEAD.unpack_from(data, offset + 4 + n)[0]:
            return
        yield Record(offset, end, json.loads(body))
        offset = end


class Writer:
    """Appends a run's commits to its journal, fsync'ing each."""

    def __init__(self, run_dir: Path, run_id: str, seq: int, mseq: int, snapshot: Callable[[], list[Op]]) -> None:
        """Continue the journal if it holds this run up to `seq` rows and `mseq` media, else start a new one
        from `snapshot` (every op that rebuilds the run)."""
        self.path = run_dir / JOURNAL
        data = self.path.read_bytes() if self.path.exists() else b""
        recs = list(records(data))
        head = recs[0].payload if recs else {}
        last = next((r.payload for r in reversed(recs) if "ops" in r.payload), {"seq": 0, "mseq": 0})
        if head.get("run") == run_id and (last.get("seq"), last.get("mseq")) == (seq, mseq):
            self.fd = os.open(self.path, os.O_WRONLY)
            os.ftruncate(self.fd, recs[-1].end)
            os.lseek(self.fd, 0, os.SEEK_END)
            self._append({"session": host()})
            return
        tmp = self.path.with_name(f".{JOURNAL}.{os.getpid()}.tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        os.write(fd, _encode({"journal": uuid.uuid4().hex, "run": run_id, "host": host()}))
        os.write(fd, _encode({"ops": snapshot(), "seq": seq, "mseq": mseq}))
        os.fsync(fd)
        os.close(fd)
        os.replace(tmp, self.path)
        self.fd = os.open(self.path, os.O_WRONLY | os.O_APPEND)

    def append(self, ops: list[Op], seq: int, mseq: int) -> None:
        """One commit, which leaves the run at `seq` rows and `mseq` media."""
        self._append({"ops": ops, "seq": seq, "mseq": mseq})

    def _append(self, payload: dict[str, object]) -> None:
        os.write(self.fd, _encode(payload))
        os.fsync(self.fd)

    def close(self) -> None:
        os.close(self.fd)


def replay(c: sqlite3.Connection, ops: Sequence[Sequence[object]]) -> None:
    for table, values in ops:
        t = str(table)
        vals = [_value(v) for v in cast(Sequence[object], values)]
        c.execute(f"INSERT OR REPLACE INTO {t}({', '.join(TABLES[t])}) VALUES ({', '.join('?' * len(vals))})", vals)


def replica_dir() -> Path:
    """Local directory of replicas: $TREX_REPLICAS, else trex-<uid>/replicas in the temp directory."""
    d = Path(os.environ.get("TREX_REPLICAS") or Path(tempfile.gettempdir()) / f"trex-{os.getuid()}" / "replicas")
    d.mkdir(parents=True, exist_ok=True, mode=0o700)
    return d


def sync(run_dir: Path, schema: str) -> Path:
    """The replica of the run in `run_dir`, brought up to date with its journal."""
    key = hashlib.sha1(os.path.realpath(run_dir).encode()).hexdigest()[:20]
    db = replica_dir() / f"{key}.sqlite"
    with open(db.with_suffix(".lock"), "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with contextlib.closing(sqlite3.connect(db, isolation_level=None, timeout=60)) as c:
            c.execute("PRAGMA journal_mode=WAL")
            c.executescript(schema + "CREATE TABLE IF NOT EXISTS journal_state(journal TEXT, offset INTEGER);")
            _catch_up(c, run_dir / JOURNAL)
    return db


def _catch_up(c: sqlite3.Connection, path: Path) -> None:
    with open(path, "rb") as f:
        data = f.read()
    head = next(records(data), None)
    if head is None:
        return
    state = c.execute("SELECT journal, offset FROM journal_state").fetchone()
    offset = state[1] if state and state[0] == head.payload.get("journal") and state[1] <= len(data) else 0
    c.execute("BEGIN IMMEDIATE")
    if offset == 0:
        for table in TABLES:
            c.execute(f"DELETE FROM {table}")
    for r in records(data, offset):
        if "ops" in r.payload:
            replay(c, cast(Sequence[Sequence[object]], r.payload["ops"]))
        offset = r.end
    c.execute("DELETE FROM journal_state")
    c.execute("INSERT INTO journal_state VALUES (?, ?)", (head.payload.get("journal"), offset))
    c.execute("COMMIT")
