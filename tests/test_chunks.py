import math
import sqlite3

import numpy as np
import pytest

from trex import chunks

from helpers import merge, readback, write_commit


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
        seq += write_commit(c, seq, rows, ids)
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


SPARSE = [[(float(i), i / 2, {"loss": 1 / (i + 1), **({"eval": float(i)} if i % 3 == 0 else {}),
                                **({"acc": math.nan} if i % 5 == 1 else {})}) for i in range(start, start + n)]
          for start, n in ((0, 1), (1, 3), (4, 1), (5, 7), (12, 2), (14, 1), (15, 9))]


def commits_of(c):
    return c.execute("SELECT seq0, n FROM rowmeta ORDER BY seq0").fetchall()


def test_merging_commits_keeps_every_row_and_metric(db):
    write_commits(db, SPARSE)
    before = readback(db, tables=("keys",))
    for seq0, stop in ((1, 5), (5, 14), (0, 24)):
        db.execute("BEGIN")
        replaced = merge(db, seq0, stop)
        db.execute("COMMIT")
        assert replaced > 1 and readback(db, tables=("keys",)) == before
    assert commits_of(db) == [(0, 24)]
    assert db.execute("SELECT count(*) FROM chunk").fetchone()[0] == 3


def test_merge_refuses_ranges_that_are_not_whole_contiguous_commits(db):
    write_commits(db, SPARSE)
    before, layout = readback(db, tables=("keys",)), commits_of(db)
    for seq0, stop in ((0, 3), (2, 5), (1, 6), (20, 30), (5, 5)):
        db.execute("BEGIN")
        with pytest.raises(ValueError):
            merge(db, seq0, stop)
        db.execute("ROLLBACK")
    assert readback(db, tables=("keys",)) == before and commits_of(db) == layout


def test_a_merge_that_would_change_any_value_is_refused(db, monkeypatch):
    write_commits(db, SPARSE)
    before, layout = readback(db, tables=("keys",)), commits_of(db)
    good = chunks._merged

    def flip_last_value(parts, seq0, n):
        blob, pos, vals = good(parts, seq0, n)
        return bytes(blob[:-1]) + bytes([blob[-1] ^ 1]), pos, vals

    monkeypatch.setattr(chunks, "_merged", flip_last_value)
    db.execute("BEGIN")
    with pytest.raises(RuntimeError, match="read back"):
        merge(db, 0, 24)
    db.execute("ROLLBACK")
    assert readback(db, tables=("keys",)) == before and commits_of(db) == layout


def test_a_merge_is_refused_once_its_commits_changed(db):
    write_commits(db, SPARSE)
    db.execute("BEGIN")
    m = chunks.prepare_merge(db, 5, 24)
    db.execute("COMMIT")
    db.execute("BEGIN")
    merge(db, 5, 14)
    db.execute("COMMIT")
    before, layout = readback(db, tables=("keys",)), commits_of(db)
    db.execute("BEGIN")
    with pytest.raises(RuntimeError, match="changed"):
        chunks.apply_merge(db, m)
    db.execute("ROLLBACK")
    assert readback(db, tables=("keys",)) == before and commits_of(db) == layout


def test_row_count_stops_at_the_first_gap_between_commits(db):
    ids = {}
    db.execute("BEGIN")
    write_commit(db, 0, TRAIN_EVAL[0], ids)
    write_commit(db, 10, TRAIN_EVAL[1], ids)
    db.execute("COMMIT")
    assert chunks.row_count(db) == len(TRAIN_EVAL[0])
