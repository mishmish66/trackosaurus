"""Cases of the formats the Python and browser code share (tiles, live tails, slabs, smoothing, group statistics), written from the Python
implementations into shared_cases.json, which tests/shared_cases.test.mjs checks the browser's against.
TREX_WRITE_CASES=1 rewrites the file."""

import base64
import json
import math
import os
from pathlib import Path

import numpy as np

from trex import query, tiles

CASES = Path(__file__).with_name("shared_cases.json")


def plain(v: float | None) -> float | str | None:
    """A number for JSON: non-finite ones as "nan", "inf" or "-inf", as everywhere else."""
    return None if v is None else float(v) if math.isfinite(v) else str(float(v))


def plains(a: np.ndarray) -> list[float | str | None]:
    return [plain(v) for v in a.tolist()]


def spread(n: int, k: int, m: int) -> np.ndarray:
    """n values in [0, 1), the same on every platform."""
    return (np.arange(n) * k % m) / m


def tile_cases() -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for level, index, n, infinite in [(0, 0, 300, False), (-3, 2, 200, False), (5, 1, 1500, True), (2, 7, 0, False)]:
        lo, hi = tiles.tile_range(level, index)
        steps = lo - 10 + spread(n, 7919, 100003) * (hi - lo + 20)
        values = spread(n, 104729, 2003) * 20 - 10
        values[::17] = np.nan
        if infinite:
            values[5::29], values[11::31] = np.inf, -np.inf
        times = steps * 0.25 + spread(n, 5, 13)
        blob = tiles.build(steps, values, times, level, index)
        t = tiles.decode(blob)
        out.append({"blob": base64.b64encode(blob).decode(), "level": t.level, "index": t.index, "count": int(t.bucket.size),
                    "bucket": t.bucket.tolist(), "min": plains(t.min), "max": plains(t.max), "mean": plains(t.mean),
                    "tmean": plains(t.tmean), "soff": plains(t.soff), "n": t.n.tolist(), "mean_step": plains(t.mean_step)})
    return out


def merge_cases() -> list[dict[str, object]]:
    """Tiles with runs of infinities, merged up: buckets merge into the mean of the finite ones."""
    out: list[dict[str, object]] = []
    level, index, n = 2, 3, 900
    lo, hi = tiles.tile_range(level, index)
    steps = lo + spread(n, 7919, 100003) * (hi - lo)
    values = spread(n, 104729, 2003) * 20 - 10
    values[(steps > lo + 100) & (steps < lo + 180)] = np.inf
    values[(steps > lo + 400) & (steps < lo + 405)] = -np.inf
    values[::23] = np.inf
    blob = tiles.build(steps, values, steps * 0.5, level, index)
    for up in (1, 3):
        merged = [tiles.decode(b) for b in tiles.coarsen([blob], up)]
        out.append({"blob": base64.b64encode(blob).decode(), "level": level, "up": up,
                    "mean": [plain(v) for t in merged for v in t.mean.tolist()],
                    "mean_step": [plain(v) for t in merged for v in t.mean_step.tolist()]})
    return out


def tail_cases() -> list[dict[str, object]]:
    """A live run's tile built before and after its newest rows, which split a bucket: the column of the first plus
    those rows is the column of the second, for top tiles merged up and for finer tiles."""
    out: list[dict[str, object]] = []
    level, index, n = 2, 1, 700
    lo, hi = tiles.tile_range(level, index)
    steps: np.ndarray = lo + np.arange(n) * ((hi - lo) / n)
    values = spread(n, 104729, 2003) * 20 - 10
    values[::19] = np.nan
    values[600:640:7] = np.inf
    times: np.ndarray = steps * 0.5 + spread(n, 5, 13)
    k = 532  # inside a bucket at every merge below
    before = tiles.build(steps[:k], values[:k], times[:k], level, index)
    after = tiles.build(steps, values, times, level, index)
    for up, fine in [(0, False), (2, False), (0, True)]:
        out.append({"before": base64.b64encode(before).decode(), "after": base64.b64encode(after).decode(), "level": level,
                    "index": index, "up": up, "fine": fine, "steps": steps[k:].tolist(), "values": plains(values[k:]),
                    "times": times[k:].tolist()})
    return out


def slab_cases() -> list[dict[str, object]]:
    """A slab of runs' top tiles a level up, and each run's mean per bin over its rows, for bins one and two buckets
    wide: the mean of the finite values, infinite only without any."""
    rng = np.random.default_rng(7)
    runs, rows = [], []
    for i, n in enumerate([900, 1300, 700]):
        steps: np.ndarray = np.arange(n) * 1.0 + i
        values = rng.normal(size=n)
        values[::11] = np.nan
        if i == 1:
            values[200:208] = np.inf
        level, idx = tiles.top_tiles(steps[0], steps[-1])
        times: np.ndarray = steps * 0.5
        runs.append((f"r{i}", [tiles.build(steps, values, times, level, k) for k in idx]))
        rows.append((steps, values))
    level = max(tiles.decode(b).level for _, bs in runs for b in bs) + 1
    width = 2.0 ** level
    out: list[dict[str, object]] = []
    for bins in (tiles.TILE, tiles.TILE // 2):  # bins one bucket wide, then two
        lo, w = 0.0, tiles.TILE * width / bins
        means: list[list[float | str | None]] = []
        for steps, values in rows:
            b = np.floor((steps - lo) / w).astype(np.int64)
            row: list[float | str | None] = [None] * bins
            for k in range(bins):
                v = values[(b == k) & ~np.isnan(values)]
                if v.size:
                    f = v[np.isfinite(v)]
                    row[k] = plain(float(f.mean()) if f.size else float(v.mean()))
            means.append(row)
        out.append({"slab": base64.b64encode(tiles.slab(runs, level, 0)).decode(), "level": level, "index": 0,
                    "x0": lo, "x1": lo + bins * w, "bins": bins, "means": means})
    return out


def smoothing_cases() -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for alpha, n, gaps in [(0.6, 50, False), (0.9, 200, True), (0.99, 300, True)]:
        xs = np.cumsum(spread(n, 31, 7) * 4 + 0.25)
        xs[n // 2] = xs[n // 2 - 1]
        ys = spread(n, 7919, 1009) * 10 - 5
        if gaps:
            ys[5::23] = np.nan
        scale = query.smooth_scale(float(xs[-1] - xs[0]))
        smoothed = query.twema(xs.tolist(), ys.tolist(), alpha, scale)
        out.append({"xs": xs.tolist(), "ys": plains(ys), "alpha": alpha, "span": float(xs[-1] - xs[0]),
                    "scale": scale, "smoothed": [plain(y) for y in smoothed]})
    return out


def stats_cases() -> list[dict[str, object]]:
    groups: list[list[float]] = [[], [2.5], [1.0, 4.0], [3.0, 1.0, 2.0], [5.0, 5.0, 5.0, 5.0, 1.0]]
    groups += [(spread(n, 7919, 1009) * 10 - 5).tolist() for n in (7, 8, 13, 40, 101)]
    groups.append([1.0, float("nan"), 3.0, 2.0, float("nan"), 7.0])
    inf = math.inf
    groups += [[*map(float, range(1, 10)), inf], [-inf, *map(float, range(1, 9)), inf], [inf], [1.0, inf], [inf, -inf],
               [inf, inf, inf, 1.0, 2.0], [-inf, -inf, 3.0, float("nan"), inf, 4.0, 5.0, 6.0]]
    out: list[dict[str, object]] = []
    for values in groups:
        s = query.stats(values)
        xs = sorted(v for v in values if not math.isnan(v))
        case: dict[str, object] = {"values": [plain(v) for v in values], "n": s["n"]}
        if xs:
            iqm, se, kept = query._iqm(xs)
            case.update({k: plain(s.get(k)) for k in ("mean", "std", "median", "min", "max")}, iqm=plain(iqm), iqm_se=plain(se),
                        iqm_kept=kept, median_ci=[plain(s.get("ci_lo", xs[0])), plain(s.get("ci_hi", xs[-1]))])
        out.append(case)
    return out


def build_cases() -> dict[str, object]:
    return {"tiles": tile_cases(), "merges": merge_cases(), "tails": tail_cases(), "slabs": slab_cases(),
            "smoothing": smoothing_cases(), "stats": stats_cases()}


def test_shared_cases_match_the_python_implementations():
    cases = json.loads(json.dumps(build_cases()))
    if os.environ.get("TREX_WRITE_CASES") == "1":
        CASES.write_text(json.dumps(cases, indent=None, separators=(",", ":")) + "\n")
    assert json.loads(CASES.read_text()) == cases
