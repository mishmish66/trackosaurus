import math
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import pytest

from trex import buckets as bk
from trex.buckets import BLOCK, SOFF_SCALE

type Floats = npt.NDArray[np.float64]
type ByBucket = dict[int, tuple[float, float, float, int]]


def rows(n: int, seed: int = 0, gaps: bool = True) -> tuple[Floats, Floats, Floats]:
    rng = np.random.default_rng(seed)
    steps = np.sort(rng.uniform(-300, 9000, n))
    values = rng.normal(size=n)
    if gaps:
        values[::13], values[5::41], values[7::43] = np.nan, np.inf, -np.inf
    return steps, values, steps * 0.5 + 3


def by_bucket(steps: Floats, values: Floats, times: Floats, level: int) -> ByBucket:
    """{bucket: (mean, mean step, mean runtime, count)} computed row by row."""
    w = 2.0 ** level
    out: ByBucket = {}
    for b in np.unique(np.floor(steps[~np.isnan(values)] / w)):
        sel = (np.floor(steps / w) == b) & ~np.isnan(values)
        fin = sel & np.isfinite(values)
        use = fin if fin.any() else sel
        with np.errstate(invalid="ignore"):
            out[int(b)] = (values[use].mean(), steps[use].mean(), times[use].mean(), int(use.sum()))
    return out


def same(a: bk.Buckets, b: bk.Buckets, runs: bool = True) -> bool:
    """Whether two Buckets hold the same buckets (leaving out which runs they are of unless `runs`)."""
    return all(np.array_equal(x, y, equal_nan=x.dtype.kind == "f")
               for x, y in zip(a.columns[not runs:], b.columns[not runs:], strict=True))


def run_at(b: bk.Buckets, i: int) -> bk.Buckets:
    return b.of(np.full(b.run.size, i, np.int32))


def assert_buckets(b: bk.Buckets, want: ByBucket, level: int) -> None:
    assert list(b.bucket) == list(want) and list(b.n) == [w[3] for w in want.values()]
    np.testing.assert_allclose(b.mean, [w[0] for w in want.values()], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(b.step(level), [w[1] for w in want.values()], atol=2.0 ** level / SOFF_SCALE + 1e-9)
    np.testing.assert_allclose(b.tmean, [w[2] for w in want.values()], rtol=1e-6)


@pytest.mark.parametrize("level", [-3, 0, 4, 9])
def test_buckets_hold_the_mean_mean_step_mean_runtime_and_count_of_their_rows(level: int) -> None:
    steps, values, times = rows(3000)
    assert_buckets(bk.bucketize(steps, values, times, level), by_bucket(steps, values, times, level), level)


def test_buckets_average_their_finite_values_and_are_infinite_only_without_any() -> None:
    inf, nan = np.inf, np.nan
    steps = np.arange(10.0)
    values = np.array([1.0, nan, inf, inf, -inf, inf, inf, 3.0, 5.0, 7.0])
    b = bk.bucketize(steps, values, steps, 1)
    assert list(b.bucket) == [0, 1, 2, 3, 4] and list(b.n) == [1, 2, 2, 1, 2]
    assert b.mean[0] == 1.0 and b.mean[1] == inf and math.isnan(b.mean[2]) and b.mean[3] == 3.0 and b.mean[4] == 6.0
    np.testing.assert_allclose(b.step(1), [0.0, 2.5, 4.5, 7.0, 8.5], atol=2 / SOFF_SCALE)


def test_rows_out_of_step_order_make_the_same_buckets() -> None:
    steps, values, times = rows(500)
    order = np.random.default_rng(1).permutation(500)
    assert same(bk.bucketize(steps, values, times, 3), bk.bucketize(steps[order], values[order], times[order], 3))


@pytest.mark.parametrize("up", [1, 3, 7])
def test_merged_buckets_are_the_buckets_of_the_coarser_level(up: int) -> None:
    steps, values, times = rows(4000)
    got = bk.merge(bk.bucketize(steps, values, times, 2), up)
    assert_buckets(got, by_bucket(steps, values, times, 2 + up), 2 + up)


def test_merged_buckets_average_the_finite_ones_and_are_infinite_only_without_any() -> None:
    inf = np.inf
    steps = np.arange(16.0)
    values = np.array([1.0, 1.0, inf, inf, 3.0, 3.0, inf, inf, inf, inf, inf, inf, 2.0, 4.0, -inf, -inf])
    b = bk.bucketize(steps, values, steps, 1)
    up1 = bk.merge(b, 1)
    assert list(up1.mean) == [1.0, 3.0, inf, 3.0] and list(up1.n) == [2, 2, 4, 2]
    np.testing.assert_allclose(up1.step(2), [0.5, 4.5, 9.5, 12.5], atol=4 / SOFF_SCALE)
    up2 = bk.merge(b, 2)
    assert list(up2.mean) == [2.0, 3.0] and list(up2.n) == [4, 2]


def test_merging_keeps_runs_apart_and_merges_each_from_its_own_level() -> None:
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


def test_merging_by_no_level_changes_nothing() -> None:
    b = bk.bucketize(*rows(800), 3)
    assert same(bk.merge(b, 0), b)


def test_sparse_rows_sit_at_their_mean_step_not_the_bucket_center() -> None:
    steps = np.array([1000.0, 2100.0, 2900.0])
    b = bk.bucketize(steps, np.ones(3), steps, 10)
    assert list(b.bucket) == [0, 2] and b.step(10) == pytest.approx([1000.0, 2500.0], abs=2.0 ** 10 / SOFF_SCALE)


def test_a_cut_keeps_the_buckets_of_a_range_and_of_the_runs_taken() -> None:
    b = bk.union([run_at(bk.bucketize(*rows(2000, seed=i), 0), i) for i in range(3)])
    c = bk.cut(b, 1000, 3000, np.array([True, False, True]))
    assert set(c.run.tolist()) == {0, 2} and c.bucket.min() >= 1000 and c.bucket.max() < 3000
    assert c.run.size == int(((b.bucket >= 1000) & (b.bucket < 3000) & (b.run != 1)).sum())


def test_a_bucket_array_round_trips_its_runs_rows_and_buckets() -> None:
    b = bk.union([run_at(bk.cut(bk.bucketize(*rows(3000, seed=i), 2), 512, 768), i) for i in (0, 2)])
    blob = bk.encode(2, 2, ["a", "b/c", "d"], [10, 0, 7], b)
    a = bk.decode(blob)
    assert (a.level, a.paths, list(a.seq)) == (2, ["a", "b/c", "d"], [10, 0, 7])
    assert same(a.buckets, b)


def test_framed_arrays_come_back_as_they_went_in_empty_ones_included() -> None:
    arrays = [bk.encode(0, 0, ["r"], [n], bk.bucketize(np.arange(n * 1.0), np.arange(n * 1.0), np.arange(n * 1.0), 0)) for n in (3, 10)]
    bodies = [arrays[0], b"", arrays[1]]
    data = bk.frame(bodies)
    assert len(data) % 8 == 0 and bk.unframe(data) == bodies
    with pytest.raises(ValueError, match="length"):
        bk.unframe(data[:-8])


def test_decoding_rejects_foreign_or_truncated_bytes() -> None:
    blob = bk.encode(0, 0, ["r"], [3], bk.bucketize(np.arange(10.0), np.arange(10.0), np.arange(10.0), 0))
    with pytest.raises(ValueError, match="magic"):
        bk.decode(b"XXXX" + blob[4:])
    with pytest.raises(ValueError, match="length"):
        bk.decode(blob[:-8])


def test_an_array_refuses_buckets_beyond_its_range() -> None:
    b = bk.bucketize(np.arange(10.0), np.arange(10.0), np.arange(10.0), 0)
    with pytest.raises(ValueError, match="range"):
        bk.encode(0, 1, ["r"], [10], b)


def test_a_stack_reads_each_runs_one_run_arrays_as_decoding_them_would() -> None:
    names, counts, levels = [f"r{i}" for i in range(5)], [300, 1700, 0, 4000, 5], [3, 4, 5, 6, 7]
    blobs, owner, want = list[bytes](), list[int](), list[bk.Buckets]()
    for i, n in enumerate(counts):
        s, v, t = rows(n, seed=i) if n else (np.empty(0), np.empty(0), np.empty(0))
        want.append(bk.bucketize(s * (i + 1) - 1000 * i, v, t, levels[i]))
        for block, part in bk.by_block(want[-1]):
            blobs.append(bk.encode(levels[i], block, [""], [0], part))
            owner.append(i)
    st = bk.stack(names, np.array(counts, np.uint32), np.array(levels, np.int8), blobs, owner)
    assert st.paths == names and list(st.seq) == counts and list(st.level) == levels
    assert len(blobs) > len(names)  # some runs lie in several blocks
    for i, b in enumerate(want):
        assert same(bk.select(st.buckets, st.buckets.run == i), b, runs=False)


def test_a_stack_refuses_arrays_of_many_runs() -> None:
    b = bk.union([run_at(bk.bucketize(*rows(10, seed=i), 9), i) for i in range(2)])
    with pytest.raises(ValueError):
        bk.stack(["x"], np.zeros(1, np.uint32), np.zeros(1, np.int8), [bk.encode(9, 0, ["a", "b"], [10, 10], b)], [0])


def level_of(parts: Sequence[bk.Part], level: int) -> tuple[bk.Buckets, list[int]]:
    """The buckets of `level` among `parts`, and the blocks holding them."""
    mine = [p for p in parts if p.level == level]
    return bk.concat([p.buckets for p in mine]), [p.block for p in mine]


def assert_merged(b: bk.Buckets, want: ByBucket, level: int) -> None:
    """Buckets merged up from finer ones hold what their rows give, as closely as merged steps and float32 means do."""
    assert list(b.bucket) == list(want) and list(b.n) == [w[3] for w in want.values()]
    np.testing.assert_allclose(b.mean, [w[0] for w in want.values()], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(b.tmean, [w[2] for w in want.values()], rtol=1e-5)
    np.testing.assert_allclose(b.step(level), [w[1] for w in want.values()], atol=2.0 ** level * 8 / SOFF_SCALE)


@pytest.mark.parametrize("gaps", [False, True])
def test_a_pyramid_holds_every_level_from_about_a_value_a_bucket_to_the_one_whose_blocks_its_steps_fit_in_two(gaps: bool) -> None:
    steps, values, times = rows(3000, gaps=gaps)
    steps = np.round(steps)
    span, parts = bk.pyramid(steps, values, times)
    assert span.top == bk.level_for(float(steps.max() - steps.min())) and (span.lo, span.hi) == (steps.min(), steps.max())
    assert span.fine == int(np.floor(np.log2(float(np.median(np.diff(np.unique(steps).astype(np.float64))))))) < span.top
    assert sorted({p.level for p in parts}) == list(range(span.fine, span.top + 1))
    for level in range(span.fine, span.top + 1):
        b, held = level_of(parts, level)
        assert_merged(b, by_bucket(steps, values, times, level), level)
        assert held == sorted({int(i) for i in np.unique(np.floor(steps[~np.isnan(values)] / 2.0 ** level) // BLOCK)})
    assert len(level_of(parts, span.top)[1]) <= 2


def test_a_metric_with_at_most_a_value_a_bucket_at_its_top_level_has_that_level_alone() -> None:
    steps = np.arange(100.0)
    span, parts = bk.pyramid(steps, steps, steps)
    assert span.fine == span.top == bk.level_for(99.0) < 0 and [(p.level, p.block) for p in parts] == [(span.top, 0)]
    span, parts = bk.pyramid(np.array([7.0]), np.array([1.0]), np.array([0.0]))
    assert span == bk.Span(0, 0, 7.0, 7.0) and int(parts[0].buckets.n.sum()) == 1


class Held:
    """A metric's blocks as an index holds them, for `grow`."""

    def __init__(self, compiled: tuple[bk.Span, list[bk.Part]]) -> None:
        self.span, parts = compiled
        self.blocks = {(p.level, p.block): p.buckets for p in parts}

    def grow(self, steps: Floats, values: Floats, times: Floats) -> list[tuple[int, int]]:
        """Grow by rows; the (level, block) of the blocks that changed."""
        self.span, parts = bk.grow(self.span, steps, values, times, lambda level, block: self.blocks.get((level, block), bk.empty()),
                                   lambda level: [block for at, block in self.blocks if at == level])
        self.blocks.update({(p.level, p.block): p.buckets for p in parts})
        return [(p.level, p.block) for p in parts]

    def level(self, level: int) -> bk.Buckets:
        return bk.concat([self.blocks[k] for k in sorted(self.blocks) if k[0] == level])


@pytest.mark.parametrize("sizes", [[2000, 1, 1, 500, 3498], [600, 5400]])
def test_a_metric_grown_by_rows_holds_what_compiling_every_row_holds(sizes: list[int]) -> None:
    steps, values, times = np.arange(6000.0) * 4, *rows(6000)[1:]
    held, at = Held(bk.pyramid(steps[:sizes[0]], values[:sizes[0]], times[:sizes[0]])), sizes[0]
    for n in sizes[1:]:
        held.grow(steps[at:at + n], values[at:at + n], times[at:at + n])
        at += n
    span, parts = bk.pyramid(steps, values, times)
    assert held.span == span and sorted(held.blocks) == sorted((p.level, p.block) for p in parts)
    for level in range(span.fine, span.top + 1):
        assert_merged(held.level(level), by_bucket(steps, values, times, level), level)


def test_growing_changes_only_the_blocks_the_new_rows_fall_in_and_those_above_them() -> None:
    steps = np.arange(60000.0)
    held = Held(bk.pyramid(steps[:50000], steps[:50000], steps[:50000]))
    before = dict(held.blocks)
    changed = held.grow(steps[50000:50010], steps[50000:50010], steps[50000:50010])
    assert changed == [(level, 50000 >> level >> 8) for level in range(held.span.fine, held.span.top + 1)]
    assert all(same(held.blocks[k], b) for k, b in before.items() if k not in changed)


def test_growing_past_the_top_levels_span_adds_the_levels_above_it_whole() -> None:
    steps = np.arange(5000.0)
    held = Held(bk.pyramid(steps[:1000], steps[:1000], steps[:1000]))
    top = held.span.top
    held.grow(steps[1000:], steps[1000:], steps[1000:])
    assert held.span.top == bk.pyramid(steps, steps, steps)[0].top > top
    for level in range(top + 1, held.span.top + 1):
        assert_merged(held.level(level), by_bucket(steps, steps, steps, level), level)


def test_rows_at_earlier_steps_grow_the_blocks_there() -> None:
    steps = np.arange(0.0, 8000.0, 2.0)
    late, early = steps >= 3000, steps < 3000
    held = Held(bk.pyramid(steps[late], steps[late], steps[late]))
    held.grow(steps[early], steps[early], steps[early])
    for level in range(held.span.fine, held.span.top + 1):
        assert_merged(held.level(level), by_bucket(steps, steps, steps, level), level)


def test_combined_buckets_merge_where_both_hold_one_as_their_rows_together_would() -> None:
    steps, values, times = rows(2000)
    odd = np.arange(2000) % 2 == 1
    got = bk.combine(bk.bucketize(steps[odd], values[odd], times[odd], 4), bk.bucketize(steps[~odd], values[~odd], times[~odd], 4))
    assert_merged(got, by_bucket(steps, values, times, 4), 4)


@pytest.mark.parametrize("finer", [1, 3])
def test_refined_buckets_of_single_rows_are_the_buckets_of_those_rows_at_the_finer_level(finer: int) -> None:
    steps = np.arange(0.0, 4000.0, 2.0)
    values = np.sin(steps)
    b, want = bk.refine(bk.bucketize(steps, values, steps, 1), 1, 1 - finer), by_bucket(steps, values, steps, 1 - finer)
    assert list(b.bucket) == list(want) and list(b.n) == [w[3] for w in want.values()]
    np.testing.assert_allclose(b.mean, [w[0] for w in want.values()], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(b.step(1 - finer), [w[1] for w in want.values()], atol=2.0 / SOFF_SCALE)  # as exact as level 1 holds them


@pytest.mark.parametrize("lo,hi", [(0, 100), (250, 260), (255.5, 256.5), (1000, 5000), (-300, 300), (7, 7), (0, 2**40)])
def test_the_span_level_covers_the_steps_in_one_or_two_blocks(lo: float, hi: float) -> None:
    level = bk.level_for(hi - lo)
    assert 1 <= len(bk.blocks(level, lo, hi)) <= 2 and bk.block_range(level, 0)[1] >= hi - lo


@pytest.mark.parametrize("k", range(-19, 60, 3))
def test_blocks_of_the_span_level_are_the_narrowest_at_least_the_span_wide(k: int) -> None:
    base = 2.0 ** k * BLOCK
    for span in (base, math.nextafter(base, math.inf), math.nextafter(base, 0), base * 1.5):
        level = bk.level_for(span)
        assert 2.0 ** level * BLOCK >= span > 2.0 ** (level - 1) * BLOCK


def test_fractional_and_negative_steps_get_valid_buckets() -> None:
    steps = np.linspace(-0.5, 0.25, 50)
    span, parts = bk.pyramid(steps, steps, steps)
    b, _ = level_of(parts, span.top)
    assert span.top < 0 and b.n.sum() == 50 and np.all(np.diff(b.bucket) > 0)
