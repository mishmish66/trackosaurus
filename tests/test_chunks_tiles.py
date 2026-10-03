import math
import sqlite3

import numpy as np
import pytest

from trex import chunks, tiles


@pytest.fixture
def db(tmp_path):
    c = sqlite3.connect(tmp_path / "r.sqlite", isolation_level=None)
    c.executescript(chunks.SCHEMA)
    yield c
    c.close()


def write_commits(c, commits):
    ids = {}
    seq = 0
    for rows in commits:
        c.execute("BEGIN")
        seq += chunks.write(c, seq, rows, ids)
        c.execute("COMMIT")
    return ids


TRAIN_EVAL = [
    [(0.0, 0.0, {"loss": 1.0, "acc": 0.1}), (0.0, 0.5, {"eval": 7.0, "loss": 1.5}), (1.0, 1.0, {"loss": 0.9, "acc": 0.2})],
    [(2.0, 2.0, {"loss": 0.8, "acc": float("nan")}), (2.0, 2.5, {"eval": 8.0}), (3.0, 3.0, {"loss": 0.7, "acc": 0.4})],
]


def test_metric_columns_follow_row_order_across_commits_and_row_shapes(db):
    ids = write_commits(db, TRAIN_EVAL)
    s, v, t = chunks.metric(db, ids["loss"])
    assert list(s) == [0.0, 0.0, 1.0, 2.0, 3.0] and list(v) == [1.0, 1.5, 0.9, 0.8, 0.7] and list(t) == [0.0, 0.5, 1.0, 2.0, 3.0]
    s, v, _ = chunks.metric(db, ids["eval"])
    assert list(s) == [0.0, 2.0] and list(v) == [7.0, 8.0]
    s, v, _ = chunks.metric(db, ids["acc"])
    assert list(s) == [0.0, 1.0, 2.0, 3.0] and math.isnan(v[2])
    assert chunks.row_count(db) == 6


def test_metric_row_and_step_windows(db):
    ids = write_commits(db, TRAIN_EVAL)
    assert list(chunks.metric(db, ids["loss"], start=2, stop=4)[1]) == [0.9, 0.8]
    assert list(chunks.metric(db, ids["loss"], step_lo=2.5)[0]) == [2.0, 3.0]
    assert chunks.metric(db, 99)[0].size == 0


def test_rows_round_trip(db):
    write_commits(db, TRAIN_EVAL)
    out = chunks.rows(db)
    assert [r[0] for r in out] == list(range(6))
    assert out[1][3] == {"eval": 7.0, "loss": 1.5} and out[5][3] == {"loss": 0.7, "acc": 0.4}
    assert [r[0] for r in chunks.rows(db, start=2, stop=5)] == [2, 3, 4]


def test_commit_size_is_bounded():
    with pytest.raises(ValueError):
        chunks.encode([(0.0, 0.0, {"a": 1})] * (chunks.MAX_ROWS + 1), {})


def readback(c):
    """Every row and every metric's points as readers see them, NaN-safe."""
    names = chunks.key_names(c)
    rows = [(r.seq, r.step, r.t, sorted((k, repr(v)) for k, v in r.values.items())) for r in chunks.rows(c)]
    points = {names[k]: tuple(a.tobytes() for a in chunks.metric(c, k)) for k in names}
    return chunks.row_count(c), rows, points


SPARSE = [[(float(i), i / 2, {"loss": 1 / (i + 1), **({"eval": float(i)} if i % 3 == 0 else {}),
                                **({"acc": math.nan} if i % 5 == 1 else {})}) for i in range(start, start + n)]
          for start, n in ((0, 1), (1, 3), (4, 1), (5, 7), (12, 2), (14, 1), (15, 9))]


def commits_of(c):
    return c.execute("SELECT seq0, n FROM rowmeta ORDER BY seq0").fetchall()


def test_merging_commits_keeps_every_row_and_metric(db):
    write_commits(db, SPARSE)
    before = readback(db)
    for seq0, stop in ((1, 5), (5, 14), (0, 24)):
        db.execute("BEGIN")
        replaced = chunks.merge(db, seq0, stop)
        db.execute("COMMIT")
        assert replaced > 1 and readback(db) == before
    assert commits_of(db) == [(0, 24)]
    assert db.execute("SELECT count(*) FROM chunk").fetchone()[0] == 3


def test_merge_refuses_ranges_that_are_not_whole_contiguous_commits(db):
    write_commits(db, SPARSE)
    before, layout = readback(db), commits_of(db)
    for seq0, stop in ((0, 3), (2, 5), (1, 6), (20, 30), (5, 5)):
        db.execute("BEGIN")
        with pytest.raises(ValueError):
            chunks.merge(db, seq0, stop)
        db.execute("ROLLBACK")
    assert readback(db) == before and commits_of(db) == layout


def test_a_merge_that_would_change_any_value_is_refused(db, monkeypatch):
    write_commits(db, SPARSE)
    before, layout = readback(db), commits_of(db)
    good = chunks._merged

    def flip_last_value(parts, seq0, n):
        blob = bytearray(good(parts, seq0, n))
        blob[-1] ^= 1
        return bytes(blob)

    monkeypatch.setattr(chunks, "_merged", flip_last_value)
    db.execute("BEGIN")
    with pytest.raises(RuntimeError, match="would change"):
        chunks.merge(db, 0, 24)
    db.execute("ROLLBACK")
    assert readback(db) == before and commits_of(db) == layout


def test_tile_buckets_hold_min_max_mean_and_mean_step_of_their_points():
    steps = np.arange(1000, dtype=float)
    vals = np.sin(steps / 50)
    vals[500] = 9.0
    times = steps * 2
    L, (i,) = tiles.top_tiles(0, 999)
    assert tiles.tile_range(L, i)[0] <= 0 and tiles.tile_range(L, i)[1] > 999
    t = tiles.decode(tiles.build(steps, vals, times, L, i))
    w = 2.0 ** L
    for b, mn, mx, mean, tm, so, n in zip(t.bucket, t.min, t.max, t.mean, t.tmean, t.soff, t.n):
        sel = (steps >= b * w) & (steps < (b + 1) * w)
        assert n == sel.sum()
        assert (i * tiles.TILE + b + so) * w == pytest.approx(steps[sel].mean(), abs=1e-3 * w)
        assert mn == pytest.approx(vals[sel].min(), abs=1e-6) and mx == pytest.approx(vals[sel].max(), abs=1e-6)
        assert mean == pytest.approx(vals[sel].mean(), abs=1e-5) and tm == pytest.approx(times[sel].mean(), rel=1e-6)
    assert t.max.max() == pytest.approx(9.0)


def test_parent_envelope_contains_children():
    rng = np.random.default_rng(0)
    steps = np.sort(rng.uniform(0, 5000, 3000))
    vals = rng.normal(size=3000)
    L, (i,) = tiles.top_tiles(steps[0], steps[-1])
    parent = tiles.decode(tiles.build(steps, vals, steps, L, i))
    for cl, ci in ((L - 1, 2 * i), (L - 1, 2 * i + 1)):
        child = tiles.decode(tiles.build(steps, vals, steps, cl, ci))
        for b, mn, mx in zip(child.bucket, child.min, child.max):
            pb = (ci * tiles.TILE + int(b)) // 2 - i * tiles.TILE
            k = int(np.searchsorted(parent.bucket, pb))
            assert parent.bucket[k] == pb and parent.min[k] <= mn and parent.max[k] >= mx


def test_tiles_skip_non_finite_values_and_empty_tiles_are_valid():
    steps = np.array([0.0, 1.0, 2.0])
    vals = np.array([1.0, np.nan, np.inf])
    t = tiles.decode(tiles.build(steps, vals, steps, 0, 0))
    assert list(t.bucket) == [0] and list(t.n) == [1]
    assert tiles.decode(tiles.build(steps, vals, steps, 0, 5)).bucket.size == 0


def test_fractional_and_negative_steps_get_valid_tiles():
    steps = np.linspace(-0.5, 0.25, 50)
    L, idx = tiles.top_tiles(steps[0], steps[-1])
    assert L < 0 and len(idx) == 2
    assert sum(int(tiles.decode(tiles.build(steps, steps, steps, L, i)).n.sum()) for i in idx) == 50


def test_coarsened_tiles_equal_tiles_built_at_the_coarser_level():
    rng = np.random.default_rng(1)
    steps = np.sort(rng.uniform(-300, 9000, 4000))
    vals, times = rng.normal(size=4000), steps * 0.5
    L, idx = tiles.top_tiles(steps[0], steps[-1])
    blobs = [tiles.build(steps, vals, times, L, i) for i in idx]
    for up in (1, 3):
        got = [tiles.decode(b) for b in tiles.coarsen(blobs, up)]
        want = [tiles.decode(tiles.build(steps, vals, times, L + up, i)) for i in tiles.covering(L + up, steps[0], steps[-1])]
        assert [(t.level, t.index) for t in got] == [(t.level, t.index) for t in want]
        for g, w in zip(got, want):
            assert list(g.bucket) == list(w.bucket) and list(g.n) == list(w.n)
            for k in ("min", "max", "mean", "tmean", "soff"):
                assert np.allclose(getattr(g, k), getattr(w, k), rtol=1e-5, atol=1e-5)
    assert tiles.coarsen(blobs, 0) == blobs


def test_coarsening_tiles_without_finite_values_gives_an_empty_tile():
    steps = np.arange(600.0)
    blobs = [tiles.build(steps, np.full(600, np.nan), steps, 0, i) for i in (0, 1, 2)]
    (t,) = [tiles.decode(b) for b in tiles.coarsen(blobs, 2)]
    assert (t.level, t.index, t.bucket.size) == (2, 0, 0)


def test_sparse_points_are_placed_at_their_mean_step_not_the_bucket_center():
    steps = np.array([1000.0, 2100.0, 2900.0])
    t = tiles.decode(tiles.build(steps, np.ones(3), steps, 10, 0))
    w = 2.0 ** 10
    assert list(t.bucket) == [0, 2]
    assert list((t.bucket + t.soff) * w) == pytest.approx([1000.0, 2500.0], abs=1e-3)


def test_row_count_stops_at_the_first_gap_between_commits(db):
    ids = {}
    db.execute("BEGIN")
    chunks.write(db, 0, TRAIN_EVAL[0], ids)
    chunks.write(db, 10, TRAIN_EVAL[1], ids)
    db.execute("COMMIT")
    assert chunks.row_count(db) == len(TRAIN_EVAL[0])


@pytest.mark.parametrize("lo,hi", [(0, 100), (250, 260), (255.5, 256.5), (1000, 5000), (-300, 300), (7, 7), (0, 2**40)])
def test_top_tiles_rise_from_the_span_level_only_until_two_tiles_cover_the_range(lo, hi):
    level, idx = tiles.top_tiles(lo, hi)
    assert 1 <= len(idx) <= 2 and idx == list(tiles.covering(level, lo, hi))
    assert tiles.tile_range(level, idx[0])[0] <= lo and hi < tiles.tile_range(level, idx[-1])[1]
    assert level == tiles.level_for(hi - lo) or len(tiles.covering(level - 1, lo, hi)) > 2


def test_decoding_rejects_foreign_or_truncated_bytes():
    blob = tiles.build(np.arange(10.0), np.arange(10.0), np.arange(10.0), 0, 0)
    with pytest.raises(ValueError, match="magic"):
        tiles.decode(b"XXXX" + blob[4:])
    with pytest.raises(ValueError, match="length"):
        tiles.decode(blob[:-8])


def test_coarsening_past_the_top_level_is_an_error():
    blob = tiles.build(np.arange(10.0), np.arange(10.0), np.arange(10.0), tiles.MAX_LEVEL, 0)
    with pytest.raises(ValueError, match="out of range"):
        tiles.coarsen([blob], 1)


@pytest.mark.parametrize("k", range(-19, 60, 3))
def test_tiles_of_the_span_level_are_the_narrowest_at_least_the_span_wide(k):
    base = 2.0 ** k * tiles.TILE
    for span in (base, math.nextafter(base, math.inf), math.nextafter(base, 0), base * 1.5):
        level = tiles.level_for(span)
        assert 2.0 ** level * tiles.TILE >= span > 2.0 ** (level - 1) * tiles.TILE
        assert len(tiles.covering(level, 3 * base - span / 2, 3 * base + span / 2)) <= 2
