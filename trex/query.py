"""Read-side queries for the CLI: run records, filters, sorting, group statistics, and series."""

import json
import math
import os
import re
import sqlite3
from array import array
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Final, Literal, NotRequired, TypedDict, cast

from . import chunks
from .format import (DB, INFO_FILE, JSONValue, MediaKind, RunState, as_dict, as_float, as_run_state, as_str, as_str_list,
                     connect_ro, key_names)
from .index import Explorer

type PathLike = str | os.PathLike[str]
type Center = Literal["median", "mean", "iqm"]
type Reduce = Literal["last", "first", "max", "min", "mean"]


class Record(TypedDict):

    path: str
    name: str
    parent: str
    state: RunState
    step: JSONValue
    runtime: JSONValue
    rows: int
    media: int
    created: float | None
    updated: float | None
    tags: list[str]
    dir: str
    config: dict[str, JSONValue]
    info: dict[str, JSONValue]
    summary: dict[str, JSONValue]


class Stats(TypedDict):
    """All but n absent when n is 0; the CI absent when n is 1."""

    n: int
    mean: NotRequired[float]
    std: NotRequired[float]
    median: NotRequired[float]
    iqm: NotRequired[float]  # interquartile mean: the mean of the middle half
    min: NotRequired[float]
    max: NotRequired[float]
    ci_lo: NotRequired[float]
    ci_hi: NotRequired[float]
    ci_coverage: NotRequired[float]


class MediaItem(TypedDict):
    seq: int
    step: float
    key: str
    kind: MediaKind
    file: str
    size: int


class KeyStats(TypedDict):
    points: int
    first_step: float
    last_step: float
    last: float


class RunSummary(TypedDict):

    dir: str
    id: str | None
    name: str | None
    state: RunState
    created: float | None
    heartbeat: float | None
    rows: int
    step: float | None
    runtime: float | None
    tags: list[str]
    config: dict[str, JSONValue]
    info: dict[str, JSONValue]
    summary: dict[str, JSONValue]
    keys: dict[str, KeyStats]
    media: list[MediaItem]


PLAIN_FIELDS: Final = ("path", "name", "parent", "state", "step", "runtime", "rows", "media", "created", "updated", "tags", "dir")
T95: Final = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.16, 2.145, 2.131,
       2.12, 2.11, 2.101, 2.093, 2.086, 2.08, 2.074, 2.069, 2.064, 2.06, 2.056, 2.052, 2.048, 2.045, 2.042]


def is_run(path: PathLike) -> bool:
    return (Path(path) / DB).is_file()


def num(v: object) -> float | None:
    """Metric value as float; the wire markers "NaN"/"Infinity"/"-Infinity" become floats."""
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            return None
    return None


# ---- run sets (through the explorer index) ----

def open_index(root: PathLike, cache: PathLike) -> Explorer:
    """Explorer index of `root`, brought up to date with the run files."""
    ex = Explorer(root, cache)
    ex.rewalk()
    ex.poll()
    return ex


def records(ex: Explorer, prefix: str = "") -> list[Record]:
    """One flat record per run under folder `prefix` (relative to the index root)."""
    body = ex.runs(prefix)
    media: dict[str, int] = {}
    for m in body["media"]:
        media[m.run] = media.get(m.run, 0) + 1
    out: list[Record] = []
    for m in body["runs"]:
        s = m["summary"]
        out.append({
            "path": m["id"], "name": m["name"], "parent": m["parent"], "state": m["state"],
            "step": s.get("_step"), "runtime": s.get("_runtime"), "rows": m["seq"], "media": media.get(m["id"], 0),
            "created": m["created"], "updated": m["updated"], "tags": m["tags"], "dir": str(ex.dirs[m["id"]]),
            "config": m["config"], "info": m["info"],
            "summary": {k: v for k, v in s.items() if not k.startswith("_")},
        })
    return out


def get(rec: Record, field: str) -> object:
    """A field: a plain one, config.K (c.), summary.K (s., metric., m.), info.A.B, or a bare key
    (config, then summary)."""
    if field in PLAIN_FIELDS:
        return cast(dict[str, object], rec)[field]
    head, _, rest = field.partition(".")
    if head in ("config", "c") and rest:
        return rec["config"].get(rest)
    if head in ("summary", "s", "metric", "m") and rest:
        v = rec["summary"].get(rest)
        return num(v) if v is not None else None
    if head == "info" and rest:
        if rest in rec["info"]:
            return rec["info"][rest]
        v: object = rec["info"]
        for part in rest.split("."):
            v = cast(dict[str, object], v).get(part) if isinstance(v, dict) else None
        return v
    if field in rec["config"]:
        return rec["config"][field]
    if field in rec["summary"]:
        return num(rec["summary"][field])
    return None


def _text(v: object) -> str:
    return v if isinstance(v, str) else json.dumps(v)


# ---- sorting ----

def sort_records[R](recs: Sequence[R], spec: str, getter: Callable[[R, str], object] | None = None) -> list[R]:
    """Sort by comma-separated fields (`-f` or `f:desc` descending); missing values last."""
    look = getter if getter is not None else cast(Callable[[R, str], object], get)
    out = list(recs)
    for field in reversed([f.strip() for f in spec.split(",") if f.strip()]):
        desc = field.startswith("-") or field.endswith(":desc")
        field = re.sub(r":(desc|asc)$", "", field.lstrip("-+"))
        keyed = [(_sortable(look(r, field)), r) for r in out]
        present = sorted([(k, r) for k, r in keyed if k is not None], key=lambda kr: kr[0], reverse=desc)
        out = [r for _, r in present] + [r for k, r in keyed if k is None]
    return out


def _sortable(v: object) -> tuple[int, float, str] | None:
    if v is None:
        return None
    n = num(v) if not isinstance(v, str) else None
    if n is not None:
        return None if math.isnan(n) else (0, n, "")
    return (1, 0.0, _text(v))


# ---- statistics (same definitions as the browser's group bands) ----

def median_ci_rank(n: int) -> int:
    """Largest k with [x_(k), x_(n-k+1)] covering the median with probability >= 0.95 (else 1)."""
    k, cdf, logc = 1, 0.0, 0.0
    for j in range(n // 2):
        if j:
            logc += math.log(n - j + 1) - math.log(j)
        cdf += math.exp(logc + n * math.log(0.5))
        if 1 - 2 * cdf >= 0.95:
            k = j + 1
        else:
            break
    return k


def median_ci_coverage(n: int) -> float:
    k = median_ci_rank(n)
    return 1 - 2 * sum(math.comb(n, j) for j in range(k)) / 2**n


def stats(values: Iterable[float | None], center: Center = "median") -> Stats:
    """With a 95% CI for `center`: order-statistic for the median, Student t for the mean, Yuen's (trimmed-mean t, from
    the winsorized variance) for the interquartile mean."""
    xs = sorted(v for v in values if v is not None and not math.isnan(v))
    n = len(xs)
    if not n:
        return {"n": 0}
    mean = sum(xs) / n
    std = math.sqrt(sum((x - mean) ** 2 for x in xs) / (n - 1)) if n > 1 else 0.0
    h = (n - 1) / 2
    median = (xs[math.floor(h)] + xs[math.ceil(h)]) / 2
    iqm, iqm_se, kept = _iqm(xs)
    out: Stats = {"n": n, "mean": mean, "std": std, "median": median, "iqm": iqm, "min": xs[0], "max": xs[-1]}
    if n > 1:
        if center in ("mean", "iqm"):
            m, se, df = (mean, std / math.sqrt(n), n - 1) if center == "mean" else (iqm, iqm_se, kept - 1)
            t = (T95[df - 1] if df >= 1 else 0.0) if df <= 30 else 1.96
            out.update(ci_lo=m - t * se, ci_hi=m + t * se, ci_coverage=0.95)
        else:
            k = median_ci_rank(n)
            out.update(ci_lo=xs[k - 1], ci_hi=xs[n - k], ci_coverage=median_ci_coverage(n))
    return out


def _iqm(xs: Sequence[float]) -> tuple[float, float, int]:
    """(interquartile mean, Yuen's standard error, values kept) of sorted xs: the mean of ranks [g, n - g),
    g = floor(n / 4), and the winsorized variance's error."""
    n = len(xs)
    g = n // 4
    kept = n - 2 * g
    iqm = sum(xs[g:n - g]) / kept
    w = [min(max(x, xs[g]), xs[n - g - 1]) for x in xs]
    wm = sum(w) / n
    se = math.sqrt(sum((x - wm) ** 2 for x in w) / (kept * (kept - 1))) if kept > 1 else 0.0
    return iqm, se, kept


# ---- single runs (read straight from the run file) ----

@contextmanager
def snapshot(run_dir: PathLike) -> Iterator[sqlite3.Connection]:
    """A connection in one read transaction."""
    c = connect_ro(run_dir)
    try:
        c.execute("BEGIN")
        yield c
    finally:
        c.close()


def _meta(c: sqlite3.Connection) -> dict[str, JSONValue]:
    return {k: json.loads(v) for k, v in c.execute("SELECT key, value FROM meta")}


def _media(c: sqlite3.Connection, run_dir: PathLike) -> list[MediaItem]:
    base = Path(run_dir).resolve()
    return [{"seq": s, "step": st, "key": k, "kind": kind, "file": str(base / f), "size": n}
            for s, st, k, kind, f, n in c.execute("SELECT seq, step, key, kind, file, size FROM media ORDER BY seq")]


def read_meta(run_dir: PathLike) -> dict[str, JSONValue]:
    with snapshot(run_dir) as c:
        return _meta(c)


def read_keys(run_dir: PathLike) -> list[str]:
    with snapshot(run_dir) as c:
        return sorted(key_names(c).values())


def row_count(run_dir: PathLike) -> int:
    with snapshot(run_dir) as c:
        return chunks.row_count(c)


def read_rows(run_dir: PathLike, keys: Iterable[str] | None = None, start: int = 0) -> list[chunks.Row]:
    """Rows from `start`, values restricted to `keys` if given."""
    want = set(keys) if keys else None
    with snapshot(run_dir) as c:
        rows = chunks.rows(c, start, chunks.row_count(c))
    if want is not None:
        rows = [r._replace(values={k: v for k, v in r.values.items() if k in want}) for r in rows]
    return rows


def read_media(run_dir: PathLike) -> list[MediaItem]:
    with snapshot(run_dir) as c:
        return _media(c, run_dir)


def run_summary(run_dir: PathLike) -> RunSummary:
    with snapshot(run_dir) as c:
        meta = _meta(c)
        rows = chunks.row_count(c)
        keys: dict[str, KeyStats] = {}
        for kid, name in key_names(c).items():
            steps, values, _ = chunks.metric(c, kid, stop=rows)
            if steps.size:
                keys[name] = {"points": int(steps.size), "first_step": float(steps[0]), "last_step": float(steps[-1]),
                              "last": float(values[-1])}
        last = c.execute("SELECT n, data FROM rowmeta WHERE seq0 + n = ?", (rows,)).fetchone()
        media = _media(c, run_dir)
    step = runtime = None
    if last:
        n, data = last
        mv = memoryview(data).cast("d")
        step, runtime = mv[n - 1], mv[2 * n - 1]
    return {
        "dir": str(Path(run_dir).resolve()), "id": as_str(meta.get("id")), "name": as_str(meta.get("name")),
        "state": as_run_state(meta.get("state")), "created": as_float(meta.get("created")),
        "heartbeat": as_float(meta.get("heartbeat")), "rows": rows, "step": step, "runtime": runtime,
        "tags": as_str_list(meta.get("tags")), "config": as_dict(meta.get("config")), "info": as_dict(meta.get("info")),
        "summary": as_dict(meta.get("summary")), "keys": dict(sorted(keys.items())), "media": media,
    }


def folder_infos(path: PathLike, root: PathLike | None = None) -> list[tuple[str, JSONValue]]:
    """(folder, notes) of trex_info.json files from `root` down to `path`."""
    p = Path(path).resolve()
    stop = Path(root).resolve() if root else None
    chain: list[tuple[str, JSONValue]] = []
    for d in [p, *p.parents]:
        f = d / INFO_FILE
        if f.is_file():
            try:
                chain.append((str(d), json.loads(f.read_text())))
            except ValueError:
                pass
        if stop is not None and d == stop:
            break
    return chain[::-1]


# ---- series ----

def smooth_scale(span: float) -> float:
    """span / 1000 rounded to a power of two, as in the browser."""
    s = span / 1000 if span > 0 else 1.0
    return 2.0 ** round(math.log2(s))


def twema(xs: Sequence[float], ys: Sequence[float | None], alpha: float, scale: float) -> list[float | None]:
    """Time-weighted debiased EMA, as in the browser; non-finite values pass through."""
    out: list[float | None] = []
    acc, deb, last = 0.0, 0.0, None
    for x, y in zip(xs, ys):
        if y is None or not math.isfinite(y) or x is None:
            out.append(y)
            continue
        w = 0.0 if last is None else alpha ** (max(x - last, 0.0) / scale)
        acc, deb, last = acc * w + y, deb * w + 1.0, x
        out.append(acc / deb)
    return out


def series(run_dir: PathLike, keys: Iterable[str], x: Literal["step", "runtime"] = "step") -> dict[str, tuple[array[float], array[float]]]:
    """{key: (xs, ys)} in row order."""
    out = {k: (array("d"), array("d")) for k in keys}
    with snapshot(run_dir) as c:
        stop = chunks.row_count(c)
        for kid, name in key_names(c).items():
            if name in out:
                steps, values, times = chunks.metric(c, kid, stop=stop)
                out[name] = (array("d", (steps if x == "step" else times).tobytes()), array("d", values.tobytes()))
    return out


def reduce(xs: Sequence[float], ys: Sequence[float | None], how: Reduce, at: float | None = None) -> float | None:
    """One number from a series; with `at`, the value at the last x <= at."""
    pts = [(x, y) for x, y in zip(xs, ys, strict=True) if y is not None and math.isfinite(y)]
    if at is not None:
        pts = [(x, y) for x, y in pts if x <= at]
        return pts[-1][1] if pts else None
    if not pts:
        return None
    vals = [y for _, y in pts]
    return {"last": vals[-1], "first": vals[0], "max": max(vals), "min": min(vals), "mean": sum(vals) / len(vals)}[how]
