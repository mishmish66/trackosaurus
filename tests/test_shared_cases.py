"""Cases of the formats the Python and browser code share (bucket arrays and their framing, live tails, layered levels, binning,
smoothing, group statistics), written from the Python implementations into shared_cases.json, which tests/shared_cases.test.mjs checks the browser's
against.
TREX_WRITE_CASES=1 rewrites the file."""

import base64
import json
import math
import os
from pathlib import Path

import numpy as np

from trex import buckets as bk, query

CASES = Path(__file__).with_name("shared_cases.json")


def plain(v: float | None) -> float | str | None:
    """A number for JSON: non-finite ones as "nan", "inf" or "-inf", as everywhere else."""
    return None if v is None else float(v) if math.isfinite(v) else str(float(v))


def plains(a: np.ndarray) -> list[float | str | None]:
    return [plain(v) for v in a.tolist()]


def spread(n: int, k: int, m: int) -> np.ndarray:
    """n values in [0, 1), the same on every platform."""
    return (np.arange(n) * k % m) / m


def b64(blob: bytes) -> str:
    return base64.b64encode(blob).decode()


def run_rows(n: int, seed: int, level: int, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """n rows spread over block `index` of `level` and a little beyond, with NaN gaps and runs of infinities."""
    lo, hi = bk.block_range(level, index)
    steps = np.sort(lo - 10 + spread(n, 7919 + seed, 100003) * (hi - lo + 20))
    values = spread(n, 104729 + seed, 2003) * 20 - 10
    values[::17] = np.nan
    values[5::29], values[11::31] = np.inf, -np.inf
    values[(steps > lo + 100) & (steps < lo + 180)] = np.inf
    return steps, values, steps * 0.25 + spread(n, 5, 13)


def array_cases() -> list[dict[str, object]]:
    """Bucket arrays of a few runs: each field as Python wrote it, and each bucket's mean step."""
    out: list[dict[str, object]] = []
    for level, index, sizes in [(0, 0, [300, 0, 120]), (-3, 2, [200]), (5, 1, [1500, 700]), (2, 7, [])]:
        parts = []
        for i, n in enumerate(sizes):
            if n:
                b = bk.cut(bk.bucketize(*run_rows(n, i, level, index), level), index * bk.BLOCK, (index + 1) * bk.BLOCK)
                parts.append(b.of(np.full(b.run.size, i, np.int32)))
        paths, seq = [f"runs/r{i}" for i in range(len(sizes))], [3 * n + 1 for n in sizes]
        blob = bk.encode(level, index, paths, seq, bk.union(parts))
        a = bk.decode(blob)
        out.append({"blob": b64(blob), "level": level, "index": index, "paths": paths, "seq": seq,
                    "first": np.searchsorted(a.buckets.run, np.arange(len(paths) + 1)).tolist(),
                    "offset": (a.buckets.bucket - index * bk.BLOCK).tolist(), "soff": a.buckets.soff.tolist(),
                    "mean": plains(a.buckets.mean), "tmean": plains(a.buckets.tmean), "n": a.buckets.n.tolist(),
                    "mean_step": plains(a.buckets.step(level))})
    return out


def one_run(level: int, index: int, b: bk.Buckets, seq: int) -> str:
    return b64(bk.encode(level, index, ["r"], [seq], bk.cut(b, index * bk.BLOCK, (index + 1) * bk.BLOCK)))


def tail_cases() -> list[dict[str, object]]:
    """A live run's blocks before and after its newest rows (which split a bucket) reach them: its column from the
    first and those rows is its column from the second; at its own level, and two levels coarser."""
    out: list[dict[str, object]] = []
    level, index, n, k = 2, 1, 700, 532  # row k falls inside a bucket at every level below
    lo, hi = bk.block_range(level, index)
    steps: np.ndarray = lo + np.arange(n) * ((hi - lo) / n)
    values = spread(n, 104729, 2003) * 20 - 10
    values[::19] = np.nan
    values[600:640:7] = np.inf
    times: np.ndarray = steps * 0.5 + spread(n, 5, 13)
    for up in (0, 2):
        lv, ix = level + up, index >> up
        before = bk.bucketize(steps[:k], values[:k], times[:k], lv)
        after = bk.bucketize(steps, values, times, lv)
        out.append({"before": one_run(lv, ix, before, k), "after": one_run(lv, ix, after, n), "level": lv,
                    "steps": steps[k:].tolist(), "values": plains(values[k:]), "times": times[k:].tolist(), "seq0": k})
    return out


def layer_cases() -> list[dict[str, object]]:
    """A run's buckets at a coarse level over everything and a finer level over two blocks: its column takes the finer
    buckets inside those blocks and the coarse ones whose mean step lies outside them, in step order."""
    level, n = 1, 4000
    steps: np.ndarray = np.arange(n) * 0.75
    values = spread(n, 104729, 2003) * 20 - 10
    values[::13] = np.nan
    times: np.ndarray = steps * 2.0
    coarse_level, fine_blocks = level + 3, (2, 3)
    coarse = bk.bucketize(steps, values, times, coarse_level)
    fine = bk.bucketize(steps, values, times, level)
    lo, hi = bk.block_range(level, fine_blocks[0])[0], bk.block_range(level, fine_blocks[-1])[1]
    outside = (coarse.step(coarse_level) < lo) | (coarse.step(coarse_level) >= hi)
    inside = bk.cut(fine, fine_blocks[0] * bk.BLOCK, (fine_blocks[-1] + 1) * bk.BLOCK)
    want = sorted([(x, m) for x, m in zip(coarse.step(coarse_level)[outside].tolist(), coarse.mean[outside].tolist())]
                  + list(zip(inside.step(level).tolist(), inside.mean.tolist())))
    return [{"coarse": [one_run(coarse_level, i, coarse, n) for i in bk.blocks(coarse_level, 0, steps[-1])],
             "fine": [one_run(level, i, fine, n) for i in fine_blocks], "steps": [x for x, _ in want],
             "means": [plain(m) for _, m in want]}]


def bin_cases() -> list[dict[str, object]]:
    """A block of runs a level above their rows, and each run's mean per bin over its rows, for bins one and two
    buckets wide: the mean of the finite values, infinite only without any."""
    rng = np.random.default_rng(7)
    parts, rows = [], []
    level = 3
    for i, n in enumerate([900, 1300, 700]):
        steps: np.ndarray = np.arange(n) * 1.0 + i
        values = rng.normal(size=n)
        values[::11] = np.nan
        if i == 1:
            values[200:208] = np.inf
        b = bk.cut(bk.bucketize(steps, values, steps * 0.5, level), 0, bk.BLOCK)
        parts.append(b.of(np.full(b.run.size, i, np.int32)))
        rows.append((steps, values))
    width = 2.0 ** level
    blob = bk.encode(level, 0, ["r0", "r1", "r2"], [900, 1300, 700], bk.union(parts))
    out: list[dict[str, object]] = []
    for bins in (bk.BLOCK, bk.BLOCK // 2):  # bins one bucket wide, then two
        lo, w = 0.0, bk.BLOCK * width / bins
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
        out.append({"blob": b64(blob), "x0": lo, "x1": lo + bins * w, "bins": bins, "means": means})
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


def frame_case(arrays: list[dict[str, object]]) -> dict[str, object]:
    """The case arrays in one body (`buckets.frame`), an empty one among them."""
    bodies = [base64.b64decode(str(c["blob"])) for c in arrays]
    bodies.insert(1, b"")
    return {"body": b64(bk.frame(bodies)), "parts": [b64(b) for b in bodies]}


def build_cases() -> dict[str, object]:
    arrays = array_cases()
    return {"arrays": arrays, "frame": frame_case(arrays), "tails": tail_cases(), "layers": layer_cases(), "bins": bin_cases(),
            "smoothing": smoothing_cases(), "stats": stats_cases()}


def test_shared_cases_match_the_python_implementations():
    cases = json.loads(json.dumps(build_cases()))
    if os.environ.get("TREX_WRITE_CASES") == "1":
        CASES.write_text(json.dumps(cases, indent=None, separators=(",", ":")) + "\n")
    assert json.loads(CASES.read_text()) == cases
