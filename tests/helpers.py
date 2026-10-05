"""Helpers shared by the test suites."""

import json
import sqlite3
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import trex
from trex import chunks, journal, query
from trex.format import snapshot
from trex.index import Explorer
from trex.node import Node

PNG: bytes = b"\x89PNG\r\n\x1a\n" + bytes(range(64))


def loss(i: int) -> dict[str, float]:
    return {"loss": 1.0 / (i + 1)}


def write_run(d: Path, n: int = 5, *, finish: bool = True, image: bool = False,
              metrics: Callable[[int], Mapping[str, float]] = loss, **init: Any) -> trex.Run:
    """A run of `n` rows (`metrics(step)` at steps 0..n-1), with an image at the last step when `image`."""
    init.setdefault("commit_interval", 0.05)
    run = trex.init(d, **init)
    for i in range(n):
        run.log(dict(metrics(i)), step=i)
    if image:
        run.log_image("img", PNG, step=n - 1)
    if finish:
        run.finish()
    return run


def wait_for(cond: Callable[[], object], timeout: float = 20.0) -> bool:
    """Whether `cond()` became true within `timeout` seconds."""
    end = time.time() + timeout
    while not cond() and time.time() < end:
        time.sleep(0.02)
    return bool(cond())


def request(url: str, data: bytes | None = None, headers: Mapping[str, str] | None = None, method: str | None = None,
            timeout: float = 60.0) -> tuple[int, dict[str, str], bytes]:
    """(status, headers, body) for any status."""
    req = urllib.request.Request(url, data=data, headers=dict(headers or {}), method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


def get_json(url: str, timeout: float = 60.0) -> Any:
    """The JSON body of a GET that succeeds."""
    status, _, body = request(url, timeout=timeout)
    assert status == 200, (status, body)
    return json.loads(body)


def post_json(url: str, body: object = None, timeout: float = 60.0) -> tuple[int, Any]:
    """(status, JSON body) of a POST of `body` (default {}) as JSON."""
    status, _, out = request(url, json.dumps({} if body is None else body).encode(), method="POST", timeout=timeout)
    return status, json.loads(out)


def post_bytes(url: str, body: object) -> bytes:
    """The raw body answering a POST of `body` as JSON."""
    status, _, out = request(url, json.dumps(body).encode(), method="POST")
    assert status == 200, (status, out)
    return out


def readback(c: sqlite3.Connection, tables: Sequence[str] = ("meta", "keys", "media")) -> dict[str, Any]:
    """What readers see of a run, NaN-safe: `tables`, the row count, every row and every metric's points; not how its
    commits are laid out."""
    out: dict[str, Any] = {t: sorted(c.execute(f"SELECT * FROM {t}").fetchall()) for t in tables}
    out["row_count"] = chunks.row_count(c)
    out["rows"] = [(r.seq, r.step, r.t, sorted((k, repr(v)) for k, v in r.values.items())) for r in chunks.rows(c)]
    out["points"] = {name: [a.tobytes() for a in chunks.metric(c, kid).columns] for kid, name in chunks.key_names(c).items()}
    return out


def readback_run(d: Path) -> dict[str, Any]:
    """`readback` of the run in directory `d`."""
    with snapshot(d) as c:
        return readback(c)


def commit_count(d: Path) -> int:
    """Commits of the run in directory `d`."""
    with snapshot(d) as c:
        return c.execute("SELECT count(*) FROM rowmeta").fetchone()[0]


def committed_rows(d: Path) -> int:
    """Rows the run in directory `d` has committed."""
    return query.row_count(d)


def write_commit(c: sqlite3.Connection, seq0: int, rows: Sequence[chunks.CommitRow], ids: dict[str, int]) -> int:
    """Insert one commit inside the caller's transaction; its row count."""
    journal.replay(c, chunks.inserts(seq0, rows, ids))
    return len(rows)


def merge(c: sqlite3.Connection, seq0: int, stop: int) -> int:
    """Merge the commits holding rows [seq0, stop) inside the caller's write transaction; how many it replaced."""
    m = chunks.prepare_merge(c, seq0, stop)
    chunks.apply_merge(c, m)
    return len(m.commits)


def node_of(ex: Explorer) -> Node:
    """A node that keeps nothing, serving `ex` alone as its home."""
    node = Node(ex.cache_dir.parent)
    d = f"{node.identity.name}:{ex.origin.key}"
    node.hold(d, ex, ex.origin.key)
    node.home = d
    return node
