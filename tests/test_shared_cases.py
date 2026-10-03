"""Cases of the formats the Python and browser code share (tiles, smoothing, group statistics), written from the Python
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
    return {"tiles": tile_cases(), "smoothing": smoothing_cases(), "stats": stats_cases()}


def test_shared_cases_match_the_python_implementations():
    cases = json.loads(json.dumps(build_cases()))
    if os.environ.get("TREX_WRITE_CASES") == "1":
        CASES.write_text(json.dumps(cases, indent=None, separators=(",", ":")) + "\n")
    assert json.loads(CASES.read_text()) == cases
