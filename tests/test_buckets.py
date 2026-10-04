import math

import numpy as np
import pytest

from trex import buckets as bk
from trex.buckets import BLOCK, SOFF_SCALE


def rows(n, seed=0, gaps=True):
    rng = np.random.default_rng(seed)
    steps = np.sort(rng.uniform(-300, 9000, n))
    values = rng.normal(size=n)
    if gaps:
        values[::13], values[5::41], values[7::43] = np.nan, np.inf, -np.inf
    return steps, values, steps * 0.5 + 3


def by_bucket(steps, values, times, level):
    """{bucket: (mean, mean step, mean runtime, count)} computed row by row."""
    w = 2.0 ** level
    out = {}
    for b in np.unique(np.floor(steps[~np.isnan(values)] / w)):
        sel = (np.floor(steps / w) == b) & ~np.isnan(values)
        fin = sel & np.isfinite(values)
        use = fin if fin.any() else sel
        with np.errstate(invalid="ignore"):
            out[int(b)] = (values[use].mean(), steps[use].mean(), times[use].mean(), int(use.sum()))
    return out


def same(a, b):
    return all(np.array_equal(x, y, equal_nan=x.dtype.kind == "f") for x, y in zip(a, b, strict=True))


def run_at(b, i):
    return b._replace(run=np.full(b.run.size, i, np.int32))


def assert_buckets(b, want, level):
    assert list(b.bucket) == list(want) and list(b.n) == [w[3] for w in want.values()]
    np.testing.assert_allclose(b.mean, [w[0] for w in want.values()], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(b.step(level), [w[1] for w in want.values()], atol=2.0 ** level / SOFF_SCALE + 1e-9)
    np.testing.assert_allclose(b.tmean, [w[2] for w in want.values()], rtol=1e-6)


@pytest.mark.parametrize("level", [-3, 0, 4, 9])
def test_buckets_hold_the_mean_mean_step_mean_runtime_and_count_of_their_rows(level):
    steps, values, times = rows(3000)
    assert_buckets(bk.bucketize(steps, values, times, level), by_bucket(steps, values, times, level), level)


def test_buckets_average_their_finite_values_and_are_infinite_only_without_any():
    inf, nan = np.inf, np.nan
    steps = np.arange(10.0)
    values = np.array([1.0, nan, inf, inf, -inf, inf, inf, 3.0, 5.0, 7.0])
    b = bk.bucketize(steps, values, steps, 1)
    assert list(b.bucket) == [0, 1, 2, 3, 4] and list(b.n) == [1, 2, 2, 1, 2]
    assert b.mean[0] == 1.0 and b.mean[1] == inf and math.isnan(b.mean[2]) and b.mean[3] == 3.0 and b.mean[4] == 6.0
    np.testing.assert_allclose(b.step(1), [0.0, 2.5, 4.5, 7.0, 8.5], atol=2 / SOFF_SCALE)


def test_rows_out_of_step_order_make_the_same_buckets():
    steps, values, times = rows(500)
    order = np.random.default_rng(1).permutation(500)
    assert same(bk.bucketize(steps, values, times, 3), bk.bucketize(steps[order], values[order], times[order], 3))


@pytest.mark.parametrize("up", [1, 3, 7])
def test_merged_buckets_are_the_buckets_of_the_coarser_level(up):
    steps, values, times = rows(4000)
    got = bk.merge(bk.bucketize(steps, values, times, 2), up)
    assert_buckets(got, by_bucket(steps, values, times, 2 + up), 2 + up)


def test_merged_buckets_average_the_finite_ones_and_are_infinite_only_without_any():
    inf = np.inf
    steps = np.arange(16.0)
    values = np.array([1.0, 1.0, inf, inf, 3.0, 3.0, inf, inf, inf, inf, inf, inf, 2.0, 4.0, -inf, -inf])
    b = bk.bucketize(steps, values, steps, 1)
    up1 = bk.merge(b, 1)
    assert list(up1.mean) == [1.0, 3.0, inf, 3.0] and list(up1.n) == [2, 2, 4, 2]
    np.testing.assert_allclose(up1.step(2), [0.5, 4.5, 9.5, 12.5], atol=4 / SOFF_SCALE)
    up2 = bk.merge(b, 2)
    assert list(up2.mean) == [2.0, 3.0] and list(up2.n) == [4, 2]


def test_merging_keeps_runs_apart_and_merges_each_from_its_own_level():
    parts = [rows(n, seed=i) for i, n in enumerate([300, 1700, 50])]
    levels = [0, 2, -1]
    st = bk.Stack([f"r{i}" for i in range(3)], np.array([300, 1700, 50], np.uint32), np.array(levels, np.int8),
                  bk.union([run_at(bk.bucketize(*p, lv), i) for i, (p, lv) in enumerate(zip(parts, levels))]))
    got = st.at(4)
    for i, p in enumerate(parts):
        assert_buckets(bk.select(got, got.run == i), by_bucket(*p, 4), 4)
    only = st.at(4, np.array([2, 0], np.int32))
    assert sorted(set(only.run.tolist())) == [0, 2]
    with pytest.raises(ValueError, match="finer"):
        st.at(1)


def test_merging_by_no_level_changes_nothing():
    b = bk.bucketize(*rows(800), 3)
    assert same(bk.merge(b, 0), b)


def test_sparse_rows_sit_at_their_mean_step_not_the_bucket_center():
    steps = np.array([1000.0, 2100.0, 2900.0])
    b = bk.bucketize(steps, np.ones(3), steps, 10)
    assert list(b.bucket) == [0, 2] and b.step(10) == pytest.approx([1000.0, 2500.0], abs=2.0 ** 10 / SOFF_SCALE)


def test_a_cut_keeps_the_buckets_of_a_range_and_of_the_runs_taken():
    b = bk.union([run_at(bk.bucketize(*rows(2000, seed=i), 0), i) for i in range(3)])
    c = bk.cut(b, 1000, 3000, np.array([True, False, True]))
    assert set(c.run.tolist()) == {0, 2} and c.bucket.min() >= 1000 and c.bucket.max() < 3000
    assert c.run.size == int(((b.bucket >= 1000) & (b.bucket < 3000) & (b.run != 1)).sum())


def test_a_bucket_array_round_trips_its_runs_rows_and_buckets():
    b = bk.union([run_at(bk.cut(bk.bucketize(*rows(3000, seed=i), 2), 512, 768), i) for i in (0, 2)])
    blob = bk.encode(2, 2, ["a", "b/c", "d"], [10, 0, 7], b)
    a = bk.decode(blob)
    assert (a.level, a.paths, list(a.seq)) == (2, ["a", "b/c", "d"], [10, 0, 7])
    assert same(a.buckets, b)


def test_decoding_rejects_foreign_or_truncated_bytes():
    blob = bk.encode(0, 0, ["r"], [3], bk.bucketize(np.arange(10.0), np.arange(10.0), np.arange(10.0), 0))
    with pytest.raises(ValueError, match="magic"):
        bk.decode(b"XXXX" + blob[4:])
    with pytest.raises(ValueError, match="length"):
        bk.decode(blob[:-8])


def test_an_array_refuses_buckets_beyond_its_range():
    b = bk.bucketize(np.arange(10.0), np.arange(10.0), np.arange(10.0), 0)
    with pytest.raises(ValueError, match="range"):
        bk.encode(0, 1, ["r"], [10], b)


def test_a_stack_reads_many_one_run_arrays_as_decoding_each_would():
    blobs, want = [], []
    for i, n in enumerate([300, 1700, 0, 4000, 5]):
        s, v, t = rows(n, seed=i) if n else (np.empty(0), np.empty(0), np.empty(0))
        blobs.append(bk.kept(s * (i + 1) - 1000 * i, v, t, n))
        want.append(bk.decode(blobs[-1]))
    st = bk.stack([f"r{i}" for i in range(5)], blobs)
    assert st.paths == [f"r{i}" for i in range(5)] and list(st.seq) == [300, 1700, 0, 4000, 5]
    assert list(st.level) == [a.level for a in want]
    for i, a in enumerate(want):
        got = bk.select(st.buckets, st.buckets.run == i)
        assert same(got[1:], a.buckets[1:])


def test_a_stack_refuses_arrays_of_many_runs():
    b = bk.union([run_at(bk.bucketize(*rows(10, seed=i), 9), i) for i in range(2)])
    with pytest.raises(ValueError):
        bk.stack(["x"], [bk.encode(9, 0, ["a", "b"], [10, 10], b)])


def test_a_run_keeps_its_buckets_at_the_finest_level_whose_blocks_its_steps_fit_in_two():
    steps, values, times = rows(5000)
    a = bk.decode(bk.kept(steps, values, times, 5000))
    assert a.level == bk.level_for(steps.max() - steps.min()) and list(a.seq) == [5000]
    assert len({int(x) // BLOCK for x in a.buckets.bucket}) <= 2
    assert_buckets(a.buckets, by_bucket(steps, values, times, a.level), a.level)


def test_a_built_block_holds_the_buckets_of_its_step_range():
    steps, values, times = rows(5000)
    a = bk.decode(bk.built(steps, values, times, 5000, 3, 2))
    lo, hi = bk.block_range(3, 2)
    sel = (steps >= lo) & (steps < hi)
    assert (a.level, list(a.seq)) == (3, [5000])
    assert_buckets(a.buckets, by_bucket(steps[sel], values[sel], times[sel], 3), 3)


@pytest.mark.parametrize("lo,hi", [(0, 100), (250, 260), (255.5, 256.5), (1000, 5000), (-300, 300), (7, 7), (0, 2**40)])
def test_the_span_level_covers_the_steps_in_one_or_two_blocks(lo, hi):
    level = bk.level_for(hi - lo)
    assert 1 <= len(bk.blocks(level, lo, hi)) <= 2 and bk.block_range(level, 0)[1] >= hi - lo


@pytest.mark.parametrize("k", range(-19, 60, 3))
def test_blocks_of_the_span_level_are_the_narrowest_at_least_the_span_wide(k):
    base = 2.0 ** k * BLOCK
    for span in (base, math.nextafter(base, math.inf), math.nextafter(base, 0), base * 1.5):
        level = bk.level_for(span)
        assert 2.0 ** level * BLOCK >= span > 2.0 ** (level - 1) * BLOCK


def test_fractional_and_negative_steps_get_valid_buckets():
    steps = np.linspace(-0.5, 0.25, 50)
    a = bk.decode(bk.kept(steps, steps, steps, 50))
    assert a.level < 0 and a.buckets.n.sum() == 50
    assert np.all(np.diff(a.buckets.bucket) > 0)
