import functools
import http.client as http_client
import json
import os
import shutil
import socket
import sqlite3
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import zlib
from types import SimpleNamespace
from typing import Any, NamedTuple

import numpy as np
import pytest

import trex
from trex import buckets as bk, chunks, crawl, server
from trex import index as trex_index
from trex.crawl import Crawl
from trex.format import FORMAT, connect_ro, connect_rw
from trex.index import Explorer
from trex.daemon import resolve_root
from trex.server import bind, serve
from trex.server import urls as server_urls

import helpers
from helpers import committed_rows, get_json, request, wait_for, write_commit


def loss_and_odd(i):
    return {"loss": 1.0 / (i + 1), "odd": i} if i % 2 else {"loss": 1.0 / (i + 1)}


write_run = functools.partial(helpers.write_run, metrics=loss_and_odd)


def drain(sub):
    out = []
    while not sub.q.empty():
        msg = sub.q.get_nowait().decode()
        ev = msg.split("\n")[0][len("event: "):]
        out.append((ev, json.loads(msg.split("\n")[1][len("data: "):])))
    return out


@pytest.fixture
def root(tmp_path):
    r = tmp_path / "runs"
    r.mkdir()
    return r


def explorer(root, tmp_path, workers=None, cache="cache"):
    ex = Explorer(Crawl(root, workers), tmp_path / cache)
    ex.sync()
    return ex


def write_chunked(d, commits, state="finished", created=1000.0):
    """Closed run whose commits are exactly the given lists of rows [(step, t, {key: value})]."""
    d.mkdir(parents=True)
    (d / "media").mkdir()
    c = connect_rw(d)
    meta = {"id": uuid.uuid4().hex, "created": created, "format": FORMAT, "name": d.name, "state": state,
            "heartbeat": time.time(), "config": {}, "tags": [], "summary": {}, "info": {}}
    ids, seq = {}, 0
    c.execute("BEGIN")
    c.executemany("INSERT INTO meta VALUES (?, ?)", [(k, json.dumps(v)) for k, v in meta.items()])
    for rows in commits:
        seq += write_commit(c, seq, rows, ids)
    c.execute("COMMIT")
    c.close()


def mixed_rows(n, seed=0):
    """Rows of three shapes; `loss` is in two of them, `x` is sometimes NaN."""
    out = []
    for i in range(n):
        if i % 7 == 3:
            d = {"eval/r": float(i + seed), "loss": 0.5 * i}
        elif i % 11 == 5:
            d = {"x": float("nan") if i % 2 else -float(i)}
        else:
            d = {"loss": 1.0 / (i + 1 + seed), "lr": 1e-3 * seed}
        out.append((float(i), 0.25 * i, d))
    return out


def chunked(rows, sizes):
    out, i = [], 0
    for n in sizes:
        out.append(rows[i: i + n])
        i += n
    assert i == len(rows)
    return out


def expected(rows, key, level, index=None):
    """The buckets of `key` in `rows` at `level` (in block `index` when given)."""
    pts = [(s, d[key], t) for s, t, d in rows if key in d]
    s, v, t = (np.array(x, dtype=float) for x in zip(*pts))
    b = bk.bucketize(s, v, t, level)
    return b if index is None else bk.cut(b, index * bk.BLOCK, (index + 1) * bk.BLOCK)


def run_points(root, path, key):
    c = connect_ro(root / path)
    try:
        kid = {n: i for i, n in chunks.key_names(c).items()}[key]
        return chunks.metric(c, kid)
    finally:
        c.close()


def index_dump(ex):
    """Everything the index stores, without compile times."""
    c = sqlite3.connect(ex.db_path)
    out = {t: sorted(c.execute(f"SELECT * FROM {t}").fetchall()) for t in ("runs", "media", "metrics", "levels")}
    c.close()
    out["runs"] = [(p, {k: v for k, v in json.loads(s).items() if k != "compiled_t"}) for p, s in out["runs"]]
    return out


class Top(NamedTuple):
    level: int
    buckets: bk.Buckets


def top_of(ex, path, key):
    """Run `path`'s top level of `key`, and its buckets there."""
    c = sqlite3.connect(ex.db_path)
    try:
        (top,) = c.execute("SELECT top FROM metrics WHERE path=? AND key=?", (path, key)).fetchone()
        blobs = c.execute("SELECT data FROM levels WHERE path=? AND key=? AND level=? ORDER BY block", (path, key, top)).fetchall()
    finally:
        c.close()
    return Top(top, bk.concat([trex_index.unpack(d) for (d,) in blobs]))


def top_n(ex, path, key):
    return int(top_of(ex, path, key).buckets.n.sum())


def stored_blocks(ex, path, key):
    """{(level, block): since} of the blocks the index stores of a run's metric."""
    c = sqlite3.connect(ex.db_path)
    try:
        return {(level, block): since for level, block, since in
                c.execute("SELECT level, block, since FROM levels WHERE path=? AND key=?", (path, key))}
    finally:
        c.close()


def block(ex, key, level, index, scope="", runs=None, which="all"):
    return bk.decode(ex.buckets_body(key, level, index, scope, runs, which))


def of_run(a, path):
    """The buckets of run `path` of decoded array `a`, as one run's."""
    b = bk.select(a.buckets, a.buckets.run == a.paths.index(path))
    return b.of(np.zeros(b.run.size, np.int32))


def same(a, b):
    """Whether two runs' buckets are equal (NaN equal to NaN)."""
    return all(np.array_equal(x, y, equal_nan=x.dtype.kind == "f") for x, y in zip(a.columns[1:], b.columns[1:], strict=True))


def close(a, b, level):
    """Whether two runs' buckets of `level` hold the same buckets and counts, with means, steps and runtimes equal to
    float32 rounding (merged from finer buckets, as against bucketized from rows)."""
    return (np.array_equal(a.bucket, b.bucket) and np.array_equal(a.n, b.n)
            and np.allclose(a.mean, b.mean, rtol=1e-6, atol=1e-6, equal_nan=True) and np.allclose(a.tmean, b.tmean, rtol=1e-6)
            and np.allclose(a.step(level), b.step(level), atol=2.0 ** level * 4 / bk.SOFF_SCALE))


def test_a_finished_runs_levels_hold_every_finite_point_of_each_metric_from_its_finest_level_to_its_top(root, tmp_path):
    rows = mixed_rows(7053)
    write_chunked(root / "r", chunked(rows, [700, 1, 2000, 323, 1024, 5, 3000]))
    ex = explorer(root, tmp_path)
    meta = ex.run_meta("r")
    assert (meta.state, meta.seq, meta.compiled) == ("finished", 7053, 7053)
    assert meta.keys == ["eval/r", "loss", "lr", "x"]
    for key in meta.keys:
        a = top_of(ex, "r", key)
        steps = [s for s, _, d in rows if key in d]
        assert a.level == bk.level_for(max(steps) - min(steps)) and len({int(b) // bk.BLOCK for b in a.buckets.bucket}) <= 2
        assert close(a.buckets, expected(rows, key, a.level), a.level)
        assert top_n(ex, "r", key) == sum(1 for _, _, d in rows if key in d and np.isfinite(d[key]))
        levels = sorted({level for level, _ in stored_blocks(ex, "r", key)})
        assert levels == list(range(levels[0], a.level + 1)) and set(stored_blocks(ex, "r", key).values()) == {7053}


def test_a_block_of_a_level_above_a_runs_top_merges_its_top_level(root, tmp_path):
    rows = mixed_rows(7053)
    write_chunked(root / "r", chunked(rows, [700, 1, 2000, 323, 1024, 5, 3000]))
    ex = explorer(root, tmp_path)
    for key in ex.run_meta("r").keys:
        k = top_of(ex, "r", key)
        for up in (0, 2):
            for index in bk.blocks(k.level + up, 0, 7053):
                a = block(ex, key, k.level + up, index, runs=["r"])
                assert a.paths == ["r"] and list(a.seq) == [7053] and a.level == k.level + up
                want = bk.cut(bk.merge(k.buckets, up), index * bk.BLOCK, (index + 1) * bk.BLOCK)
                assert same(of_run(a, "r"), want)


def test_a_scope_block_holds_every_run_under_it_that_logs_the_metric(root, tmp_path):
    for name in ("a/r1", "a/r2", "b/r3"):
        write_run(root / name, 600)
    write_run(root / "a" / "short", 1)
    ex = explorer(root, tmp_path)
    level = top_of(ex, "a/r1", "odd").level
    a = block(ex, "odd", level, 0, "a")
    assert a.paths == ["a/r1", "a/r2"] and list(a.seq) == [600, 600]
    for p in a.paths:
        assert same(of_run(a, p), of_run(block(ex, "odd", level, 0, runs=[p]), p))
    assert block(ex, "odd", level, 0, runs=["a/r2", "nope", "a/short", "a/r2"]).paths == ["a/r2"]


def test_a_scopes_finished_blocks_are_kept_until_its_finished_runs_change(root, tmp_path):
    write_run(root / "a" / "r1", 600)
    ex = explorer(root, tmp_path)
    level = top_of(ex, "a/r1", "odd").level
    body = lambda: ex.buckets_body("odd", level, 0, "", None, "finished")
    assert body() is body() and bk.decode(body()).paths == ["a/r1"]
    write_run(root / "a" / "r2", 600)
    ex.sync()
    assert bk.decode(body()).paths == ["a/r1", "a/r2"]
    shutil.rmtree(root / "a" / "r1")
    ex.sync()
    assert bk.decode(body()).paths == ["a/r2"]


def test_a_memo_builds_a_value_once_while_others_wait_and_drops_the_least_recently_used():
    memo, built, gate = trex_index.Memo(limit=10), [], threading.Event()

    def build(v, nbytes=4):
        gate.wait(5)
        built.append(v)
        return v, nbytes

    got = []
    threads = [threading.Thread(target=lambda: got.append(memo.get(("a",), 0, lambda: build("a")))) for _ in range(4)]
    for th in threads:
        th.start()
    gate.set()
    for th in threads:
        th.join(5)
    assert got == ["a"] * 4 and built == ["a"]
    assert memo.get(("a",), 1, lambda: build("a1")) == "a1" and built == ["a", "a1"]
    memo.get(("b",), 0, lambda: build("b"))
    memo.get(("a",), 1, lambda: build("again"))
    memo.get(("c",), 0, lambda: build("c"))
    assert memo.get(("a",), 1, lambda: build("a2")) == "a1" and memo.get(("b",), 0, lambda: build("b2")) == "b2"


def test_the_runs_body_follows_runs_added_and_folder_notes(root, tmp_path):
    write_run(root / "a" / "r1", 5)
    ex = explorer(root, tmp_path)
    body = lambda: json.loads(ex.runs_body("a"))
    assert body() == json.loads(trex_index.dumps(ex.runs("a").wire())) and [m["id"] for m in body()["runs"]] == ["a/r1"]
    write_run(root / "a" / "r2", 5)
    trex.folder_info(root / "a", note="x")
    ex.sync()
    assert [m["id"] for m in body()["runs"]] == ["a/r1", "a/r2"] and body()["folders"]["a"] == {"note": "x"}


def test_requests_on_a_kept_alive_connection_answer_without_waiting_for_delayed_acks(http):
    _, url = http
    u = urllib.parse.urlsplit(url)
    c = http_client.HTTPConnection(u.hostname, u.port, timeout=5)
    times = []
    for _ in range(6):
        t = time.perf_counter()
        c.request("GET", "/api/info")
        c.getresponse().read()
        times.append(time.perf_counter() - t)
    c.close()
    assert sorted(times[1:])[2] < 0.02, times


def test_a_block_holds_runs_merged_from_their_top_levels_or_as_their_levels_store_it(root, tmp_path):
    write_run(root / "a" / "short", 600)
    write_run(root / "a" / "long", 5000)
    live = write_run(root / "a" / "live", 600, finish=False)
    assert wait_for(lambda: committed_rows(root / "a" / "live") == 600)
    ex = explorer(root, tmp_path)
    level = top_of(ex, "a/short", "loss").level + 1
    assert top_of(ex, "a/long", "loss").level > level and top_of(ex, "a/live", "loss").level < level
    a = block(ex, "loss", level, 0, "a", which="finished")
    assert a.paths == ["a/long", "a/short"] and list(a.seq) == [5000, 600]
    s, v, t = run_points(root, "a/long", "loss")
    assert close(of_run(a, "a/long"), bk.cut(bk.bucketize(s, v, t, level), 0, bk.BLOCK), level)
    assert same(of_run(a, "a/short"), bk.cut(bk.merge(top_of(ex, "a/short", "loss").buckets, 1), 0, bk.BLOCK))
    every = block(ex, "loss", level, 0, "a")
    assert every.paths == ["a/live", "a/long", "a/short"] and list(every.seq) == [600, 5000, 600]
    assert same(of_run(every, "a/live"), bk.cut(bk.merge(top_of(ex, "a/live", "loss").buckets, 1), 0, bk.BLOCK))
    assert block(ex, "loss", level, 0, "a", which="running").paths == ["a/live"]
    assert ex.buckets_body("loss", level, 0, "a", None, "finished") is ex.buckets_body("loss", level, 0, "a", None, "finished")
    live.finish()


def test_a_runs_blocks_below_its_top_level_come_from_its_stored_levels_without_its_run_file(root, tmp_path):
    rows = mixed_rows(5000)
    write_chunked(root / "r", chunked(rows, [1000] * 5))
    ex = explorer(root, tmp_path)
    shutil.rmtree(root / "r")
    for lv, i in [(4, 1), (2, 3)]:
        assert close(of_run(block(ex, "loss", lv, i, runs=["r"]), "r"), expected(rows, "loss", lv, i), lv)


def test_blocks_below_a_runs_finest_level_are_refined_from_it_without_the_run_file(root, tmp_path):
    rows = [(float(i), i / 2, {"loss": 1 / (i + 1)}) for i in range(3000)]  # a step apart: compiled down to level 0
    write_chunked(root / "r", chunked(rows, [1000] * 3))
    ex = explorer(root, tmp_path)
    shutil.rmtree(root / "r")
    for lv, i in [(-1, 3), (-3, 40)]:
        assert close(of_run(block(ex, "loss", lv, i, runs=["r"]), "r"), expected(rows, "loss", lv, i), lv)


def test_a_growing_runs_blocks_hold_its_compiled_rows_and_its_run_file_gives_the_rows_after_them(root, tmp_path):
    run = write_run(root / "a" / "r1", 3000, finish=False)
    assert wait_for(lambda: committed_rows(root / "a" / "r1") == 3000)
    ex = explorer(root, tmp_path)
    for i in range(3000, 3100):
        run.log({"loss": 1.0 / (i + 1)}, step=i)
    assert wait_for(lambda: committed_rows(root / "a" / "r1") == 3100)
    ex.sync()
    assert (ex.run_meta("a/r1").seq, ex.run_meta("a/r1").compiled) == (3100, 3000)
    for level in (0, top_of(ex, "a/r1", "loss").level):
        a = block(ex, "loss", level, 0, "a")
        assert list(a.seq) == [3000] and of_run(a, "a/r1").n.sum() == min(3000, bk.BLOCK << level)
    tail = json.loads(ex.rows_json("a/r1", 3000))
    assert (tail["run"], tail["seq0"], len(tail["rows"]), tail["rows"][0][0]) == ("a/r1", 3000, 100, 3000.0)
    assert ex.live_seqs("a") == {"a/r1": (3100, 0)}
    run.finish()


@pytest.mark.parametrize("workers", [1, 3])
def test_a_long_runs_new_rows_change_only_the_blocks_they_fall_in_and_those_above_them(root, tmp_path, monkeypatch, workers):
    monkeypatch.setattr(crawl, "REFRESH", 0.0)
    monkeypatch.setattr(crawl, "INLINE_BYTES", 0)
    runs = [trex.init(root / name, commit_interval=0.01) for name in ("a", "b")]
    rows = [{"loss": float(np.sin(i / 50)), **({"odd": float(i)} if i % 2 else {})} for i in range(7500)]
    for run in runs:
        for i in range(6000):
            run.log(rows[i], step=i)
    assert wait_for(lambda: [committed_rows(root / p) for p in ("a", "b")] == [6000, 6000])
    ex = explorer(root, tmp_path, workers=workers)
    before = stored_blocks(ex, "a", "loss")
    assert set(before.values()) == {6000} and ex.records["a"].rebuilt == 6000
    for upto in (6010, 6011, 7500):
        for run in runs:
            for i in range(ex.records["a"].seq, upto):
                run.log(rows[i], step=i)
        assert wait_for(lambda: [committed_rows(root / p) for p in ("a", "b")] == [upto, upto])
        assert sorted(ex.sync()) == ["a", "b"]
        assert [(ex.records[p].compiled, ex.records[p].rebuilt) for p in ("a", "b")] == [(upto, 6000)] * 2
    after = stored_blocks(ex, "a", "loss")
    top = top_of(ex, "a", "loss").level
    assert {k for k, since in after.items() if since == 6000} == {k for k in before if (k[1] + 1) * bk.BLOCK << k[0] <= 6000}
    assert top == bk.level_for(7499.0) and set(after.values()) == {6000, 7500}
    for key in ("loss", "odd"):
        for p in ("a", "b"):
            s, v, t = run_points(root, p, key)
            for level, index in [(0, 23), (0, 29), (2, 5), (2, 7), (top - 1, 0), (top, 0), (-2, 100)]:
                want = bk.cut(bk.bucketize(s, v, t, level), index * bk.BLOCK, (index + 1) * bk.BLOCK)
                assert close(of_run(block(ex, key, level, index, runs=[p]), p), want, level)
    for run in runs:
        run.finish()


def test_a_finished_block_stays_while_running_runs_grow_and_changes_once_one_finishes(root, tmp_path, monkeypatch):
    monkeypatch.setattr(crawl, "REFRESH", 0.0)
    write_run(root / "a" / "done", 300)
    live = write_run(root / "a" / "live", 300, finish=False)
    ex = explorer(root, tmp_path)
    level = top_of(ex, "a/done", "loss").level + 1
    finished = lambda: ex.buckets_body("loss", level, 0, "a", None, "finished")
    first = finished()
    for i in range(300, 400):
        live.log({"loss": 1.0 / (i + 1)}, step=i)
    assert wait_for(lambda: (ex.sync(), ex.records["a/live"].compiled >= 400)[1])
    assert finished() is first and bk.decode(first).paths == ["a/done"]
    live.finish()
    assert wait_for(lambda: (ex.sync(), ex.records["a/live"].state != "running")[1])
    assert bk.decode(finished()).paths == ["a/done", "a/live"]


def saved_levels(ex):
    d = ex.cache_dir / "levels"
    return sorted(f.name for f in d.iterdir() if not f.name.endswith(".tmp")) if d.exists() else []


def test_a_new_explorer_cuts_blocks_from_the_levels_an_earlier_one_saved(root, tmp_path, monkeypatch):
    for name in ("a/r1", "a/r2", "b/r3"):
        write_run(root / name, 3000)
    ex = explorer(root, tmp_path)
    level = top_of(ex, "a/r1", "loss").level
    want = {(lv, scope): ex.buckets_body("loss", lv, 0, scope, None, "finished") for lv in (level, level + 2) for scope in ("", "a")}
    assert wait_for(lambda: saved_levels(ex))
    ex.close()

    def unread(*_):
        raise AssertionError("kept buckets read")
    monkeypatch.setattr(bk, "stack", unread)
    again = explorer(root, tmp_path)
    assert {k: again.buckets_body("loss", k[0], 0, k[1], None, "finished") for k in want} == want


def test_saved_levels_are_left_unused_once_the_finished_runs_change(root, tmp_path):
    for name in ("a/r1", "a/r2"):
        write_run(root / name, 3000)
    ex = explorer(root, tmp_path)
    level = top_of(ex, "a/r1", "loss").level + 1
    ex.buckets_body("loss", level, 0, "a", None, "finished")
    assert wait_for(lambda: saved_levels(ex))
    ex.close()
    write_run(root / "a" / "r3", 3000, metrics=lambda i: {"loss": 5.0})
    again = explorer(root, tmp_path)
    a = block(again, "loss", level, 0, "a", which="finished")
    assert a.paths == ["a/r1", "a/r2", "a/r3"] and np.all(of_run(a, "a/r3").mean == 5.0)


def test_a_metric_saves_its_levels_at_most_once_every_levels_save_every(root, tmp_path, monkeypatch):
    monkeypatch.setattr(trex_index, "LEVELS_SAVE_EVERY", 3600.0)
    write_run(root / "a" / "r1", 3000)
    ex = explorer(root, tmp_path)
    level = top_of(ex, "a/r1", "loss").level + 1
    ex.buckets_body("loss", level, 0, "a", None, "finished")
    assert wait_for(lambda: saved_levels(ex))
    d = ex.cache_dir / "levels" / saved_levels(ex)[0]
    before = {f.name: f.read_bytes() for f in d.iterdir()}
    write_run(root / "a" / "r2", 3000)
    ex.sync()
    assert block(ex, "loss", level, 0, "a", which="finished").paths == ["a/r1", "a/r2"]
    time.sleep(0.5)
    assert saved_levels(ex) == [d.name] and {f.name: f.read_bytes() for f in d.iterdir()} == before


def test_saved_levels_beyond_their_budget_go_least_recently_used_first(tmp_path):
    for i, name in enumerate(("old", "mid", "new")):
        (tmp_path / name).mkdir()
        f = tmp_path / name / "levels.npy"
        f.write_bytes(b"x" * 100)
        os.utime(f, (1000 + i, 1000 + i))
    (tmp_path / "loose").write_bytes(b"x" * 10)
    os.utime(tmp_path / "loose", (999, 999))
    trex_index._bound_dir(tmp_path, 250)
    assert sorted(f.name for f in tmp_path.iterdir()) == ["mid", "new"]


def test_a_changed_cache_version_deletes_the_saved_levels(root, tmp_path, monkeypatch):
    write_run(root / "a" / "r1", 3000)
    ex = explorer(root, tmp_path)
    ex.buckets_body("loss", top_of(ex, "a/r1", "loss").level, 0, "a", None, "finished")
    assert wait_for(lambda: saved_levels(ex))
    ex.close()
    monkeypatch.setattr(trex_index, "CACHE_VERSION", trex_index.CACHE_VERSION + 1)
    assert saved_levels(explorer(root, tmp_path)) == []


def test_http_buckets_answers_every_block_asked_in_one_body_and_info_states_the_protocol(http, root):
    ex, url = http
    for name in ("x/r1", "x/r2"):
        write_run(root / name, 300)
    ex.sync()

    def post(*blocks):
        body = json.dumps({"blocks": list(blocks)}).encode()
        return bk.unframe(urllib.request.urlopen(urllib.request.Request(f"{url}/api/buckets", data=body)).read())

    assert post({"key": "loss", "level": 0, "index": 0, "scope": "x"}, {"key": "loss", "level": 0, "index": 1, "runs": ["x/r2"]},
                {"key": "loss", "level": 2, "index": 0, "scope": "x", "which": "finished"}) == [
        ex.buckets_body("loss", 0, 0, "x"), ex.buckets_body("loss", 0, 1, runs=["x/r2"]), ex.buckets_body("loss", 2, 0, "x", None, "finished")]
    ok = {"key": "loss", "level": 0, "index": 0}
    for bad in ({"key": "loss", "level": 0.5, "index": 0}, {"key": "loss", "level": 0, "index": 0, "which": "some"},
                {"key": "loss", "level": bk.MAX_LEVEL + 1, "index": 0}, {"key": "loss", "level": 0, "index": 0, "runs": "x/r1"},
                {"level": 0, "index": 0}):
        with pytest.raises(urllib.error.HTTPError) as e:
            post(ok, bad)
        assert e.value.code == 400, bad
    for blocks in ([], [ok] * (server.MAX_ASKS + 1)):
        with pytest.raises(urllib.error.HTTPError) as e:
            post(*blocks)
        assert e.value.code == 400
    with urllib.request.urlopen(f"{url}/api/info") as r:
        assert json.loads(r.read())["protocol"] == server.PROTOCOL


def test_blocks_finer_than_a_run_keeps_hold_its_rows_at_that_level(root, tmp_path):
    rows = mixed_rows(5000)
    write_chunked(root / "r", chunked(rows, [1000] * 5))
    ex = explorer(root, tmp_path)
    for k, lv, i in [("loss", 2, 3), ("lr", 0, 7), ("x", -1, 0), ("loss", 5, 99), ("loss", -3, 0)]:
        a = block(ex, k, lv, i, runs=["r"])
        assert (a.paths, list(a.seq), a.level) == (["r"], [5000], lv) and close(of_run(a, "r"), expected(rows, k, lv, i), lv)
    assert block(ex, "nope", 0, 0, runs=["r"]).paths == [] and block(ex, "loss", 0, 0, runs=["nope"]).paths == []
    with pytest.raises(ValueError):
        ex.buckets_body("loss", bk.MAX_LEVEL + 1, 0, runs=["r"])


def test_a_running_runs_blocks_hold_its_newest_rows_once_it_is_compiled_again(root, tmp_path, monkeypatch):
    run = trex.init(root / "r", commit_interval=0.01)
    for i in range(3000):
        run.log({"loss": float(i)}, step=i)
    assert wait_for(lambda: committed_rows(root / "r") == 3000)
    ex = explorer(root, tmp_path)
    assert of_run(block(ex, "loss", 0, 11, runs=["r"]), "r").n.sum() == 3000 - 11 * bk.BLOCK
    for i in range(3000, 3500):
        run.log({"loss": float(i)}, step=i)
    assert wait_for(lambda: committed_rows(root / "r") == 3500)
    monkeypatch.setattr(crawl, "REFRESH", 0.0)
    ex.sync()
    a = block(ex, "loss", 0, 11, runs=["r"])
    s, v, t = run_points(root, "r", "loss")
    assert close(of_run(a, "r"), bk.cut(bk.bucketize(s, v, t, 0), 11 * bk.BLOCK, 12 * bk.BLOCK), 0) and list(a.seq) == [3500]
    run.finish()


def test_live_run_streams_contiguous_rows_and_is_compiled_whole_once_it_finishes(root, tmp_path):
    run = trex.init(root / "live", commit_interval=0.05)
    run.log({"x": 0})
    assert wait_for(lambda: committed_rows(root / "live") == 1)
    ex = explorer(root, tmp_path)
    sub = ex.hub.subscribe("")
    seen = ex.run_meta("live").seq
    assert ex.run_meta("live").compiled == seen
    for i in range(1, 30):
        run.log({"x": i})
        if i % 10 == 0:
            time.sleep(0.15)
            ex.sync()
    run.finish()
    ex.sync()
    events = drain(sub)
    for ev, data in events:
        if ev == "rows":
            assert data["seq0"] == seen
            seen += len(data["rows"])
    assert seen == 30
    last = [d for e, d in events if e == "run"][-1]
    assert (last["state"], last["compiled"]) == ("finished", 30)
    assert top_n(ex, "live", "x") == 30


def test_a_growing_runs_levels_are_compiled_again_after_the_refresh_interval(root, tmp_path, monkeypatch):
    run = trex.init(root / "r", commit_interval=0.01)
    run.log({"x": 0.0}, step=0)
    assert wait_for(lambda: committed_rows(root / "r") == 1)
    ex = explorer(root, tmp_path)
    for i in range(1, 100):
        run.log({"x": float(i)}, step=i)
    assert wait_for(lambda: committed_rows(root / "r") == 100)
    ex.sync()
    assert (ex.run_meta("r").seq, ex.run_meta("r").compiled) == (100, 1)
    monkeypatch.setattr(crawl, "REFRESH", 0.0)
    ex.sync()
    assert ex.run_meta("r").compiled == 100 and top_n(ex, "r", "x") == 100
    run.finish()


def test_a_silent_running_run_becomes_crashed_with_every_row_compiled(root, tmp_path, monkeypatch):
    run = write_run(root / "r", 10, finish=False)
    assert wait_for(lambda: committed_rows(root / "r") == 10)
    ex = explorer(root, tmp_path)
    for i in range(10, 20):
        run.log({"loss": 0.0}, step=i)
    assert wait_for(lambda: committed_rows(root / "r") == 20)
    ex.sync()
    assert ex.run_meta("r").state == "running" and ex.run_meta("r").compiled == 10
    monkeypatch.setattr(crawl, "CRASH_AFTER", 0.0)
    ex.sync()
    meta = ex.run_meta("r")
    assert meta.state == "crashed" and meta.compiled == 20 and top_n(ex, "r", "loss") == 20
    run.finish()


def test_pool_and_inline_indexing_produce_identical_index(root, tmp_path, monkeypatch):
    for i in range(6):
        rows = mixed_rows(1500 + 977 * i, seed=i)
        sizes = [333] * (len(rows) // 333) + [len(rows) % 333]
        write_chunked(root / "sweep" / f"chunked{i}", chunked(rows, [n for n in sizes if n]),
                      state="finished" if i % 3 else "running")
    for i in range(6):
        write_run(root / "sweep" / f"writer{i}", 1100 + 400 * i, info={"i": i})
        if i == 1:
            run = trex.init(root / "sweep" / f"writer{i}")
            run.log_html("report", f"<p>{i}</p>")
            run.log_image("img", b"\x89PNG\r\n\x1a\n" + bytes(range(64)), step=3)
            run.finish()
    assert wait_for(lambda: [committed_rows(root / "sweep" / f"writer{i}") for i in range(6)] == [1100 + 400 * i for i in range(6)])
    before = {p: sorted(x.name for x in p.iterdir()) for p in root.glob("sweep/*")}
    inline = explorer(root, tmp_path, workers=1, cache="inline")
    monkeypatch.setattr(crawl, "INLINE_BYTES", 0)
    pooled = []
    real = Crawl._sync_pool
    monkeypatch.setattr(Crawl, "_sync_pool", lambda self, ex, todo: (pooled.append(len(todo)), real(self, ex, todo)))
    pool = explorer(root, tmp_path, workers=4, cache="pool")
    assert pooled == [12]
    assert len(inline.records) == 12
    a, b = index_dump(inline), index_dump(pool)
    assert a == b
    assert {p: sorted(x.name for x in p.iterdir()) for p in root.glob("sweep/*")} == before
    assert [m[4] for m in a["media"]] == ["html", "image"]
    for path, st in a["runs"]:
        assert st["compiled"] == st["seq"]
        assert {k for p, k, *_ in a["metrics"] if p == path} == set(st["keys"]) == {k for k, _, _, p, *_ in a["levels"] if p == path}


def test_live_rows_are_contiguous_while_the_writer_commits(root, tmp_path):
    run = trex.init(root / "live", commit_interval=0.01)
    run.log({"x": 0.0}, step=0)
    assert wait_for(lambda: committed_rows(root / "live") == 1)
    ex = explorer(root, tmp_path)
    sub = ex.hub.subscribe("")
    seen = ex.run_meta("live").seq
    n = 3000

    def produce():
        for i in range(1, n):
            run.log({"x": float(i), "y": -float(i)} if i % 3 else {"x": float(i)}, step=i)
            if i % 50 == 0:
                time.sleep(0.003)

    t = threading.Thread(target=produce)
    t.start()
    polls = 0
    while t.is_alive():
        ex.sync()
        polls += 1
        time.sleep(0.01)
    run.finish()
    ex.sync()
    rows = [d for e, d in drain(sub) if e == "rows"]
    assert polls > 3 and len(rows) > 3
    got = []
    for d in rows:
        assert d["seq0"] == seen + len(got)
        got += d["rows"]
    assert seen + len(got) == n
    assert [r[2]["x"] for r in got] == [float(i) for i in range(seen, n)]
    assert all(("y" in r[2]) == (int(r[0]) % 3 != 0) for r in got)


def test_run_whose_row_count_shrank_under_the_same_id_is_dropped_and_reindexed(root, tmp_path):
    write_run(root / "r", 20)
    old = tmp_path / "old.sqlite"
    shutil.copy(root / "r" / "trex.sqlite", old)
    run = trex.init(root / "r", commit_interval=0.05)
    for i in range(20, 50):
        run.log({"loss": 1.0}, step=i)
    run.finish()
    ex = explorer(root, tmp_path)
    uid = ex.run_meta("r").uid
    assert ex.run_meta("r").seq == 50
    ex.buckets_body("loss", 0, 0, runs=["r"])
    sub = ex.hub.subscribe("")
    shutil.copy(old, root / "r" / "trex.sqlite")
    ex.sync()
    events = drain(sub)
    assert events[0] == ("delete", {"run": "r"})
    meta = ex.run_meta("r")
    assert (meta.uid, meta.seq, meta.compiled) == (uid, 20, 20)
    assert top_n(ex, "r", "loss") == 20 and of_run(block(ex, "loss", 0, 0, runs=["r"]), "r").n.sum() == 20


def test_http_stream_sends_rows_beyond_the_compiled_levels_then_live_rows(http, root):
    ex, url = http
    run = trex.init(root / "s" / "r", commit_interval=0.05)
    run.log({"x": 0})
    assert wait_for(lambda: committed_rows(root / "s" / "r") == 1)
    ex.sync()
    for i in range(1, 3):
        run.log({"x": i})
    assert wait_for(lambda: committed_rows(root / "s" / "r") == 3)
    ex.sync()
    r = urllib.request.urlopen(f"{url}/api/stream?path=s", timeout=5)

    def next_event():
        ev = None
        while True:
            line = r.readline().decode().rstrip("\n")
            if line.startswith("event: "):
                ev = line[7:]
            elif line.startswith("data: "):
                return ev, json.loads(line[6:])

    assert next_event() == ("rows", {"run": "s/r", "seq0": 1, "rows": [[1.0, pytest.approx(0, abs=5), {"x": 1}],
                                                                        [2.0, pytest.approx(0, abs=5), {"x": 2}]]})
    run.log({"x": 3, "bad": float("inf")})
    assert wait_for(lambda: committed_rows(root / "s" / "r") == 4)
    ex.sync()
    ev, data = next_event()
    while ev == "run":
        ev, data = next_event()
    assert ev == "rows" and data["seq0"] == 3 and data["rows"][0][2] == {"x": 3, "bad": "inf"}
    run.finish()
    r.close()


def test_walk_finds_nested_runs_and_skips_hidden_env_and_run_internals(root, tmp_path):
    write_run(root / "a" / "r1", 3)
    write_run(root / "a" / "b" / "r2", 3)
    write_run(root / ".hidden" / "r3", 3)
    write_run(root / "node_modules" / "r4", 3)
    write_run(root / "a" / "r1" / "nested" / "r5", 3)
    (root / "empty" / "dir").mkdir(parents=True)
    ex = explorer(root, tmp_path)
    assert [p for p, _ in ex.tree()] == ["a/b/r2", "a/r1"]
    assert [r.id for r in ex.runs("a/b").runs] == ["a/b/r2"]
    assert [r.id for r in ex.runs("a/b/r").runs] == []


def test_rewritten_run_is_dropped_and_reindexed(root, tmp_path):
    write_run(root / "r", 50)
    ex = explorer(root, tmp_path)
    uid = ex.run_meta("r").uid
    sub = ex.hub.subscribe("")
    shutil.rmtree(root / "r")
    write_run(root / "r", 5)
    ex.sync()
    events = drain(sub)
    assert events[0] == ("delete", {"run": "r"})
    meta = ex.run_meta("r")
    assert meta.uid != uid and meta.seq == 5


def test_an_index_is_held_by_one_explorer_at_a_time_and_a_later_one_reuses_it_without_rereading_runs(root, tmp_path):
    write_run(root / "r", 1500)
    ex = explorer(root, tmp_path)
    before = ex.run_meta("r")
    beside = Explorer(Crawl(root), tmp_path / "cache")
    assert beside.cache_dir == ex.cache_dir.with_name(ex.cache_dir.name + "-1") and beside.sync() == ["r"]
    ex.close()
    ex2 = Explorer(Crawl(root), tmp_path / "cache")
    assert ex2.cache_dir == ex.cache_dir and ex2.sync() == [] and ex2.run_meta("r") == before


def test_refuses_to_crawl_home_or_filesystem_root():
    with pytest.raises(ValueError, match="refusing"):
        resolve_root("~", force=False)
    with pytest.raises(ValueError, match="refusing"):
        resolve_root("/", force=False)


@pytest.fixture
def http(root, tmp_path, http_server):
    ex = Explorer(Crawl(root), tmp_path / "cache")
    return ex, http_server(serve(ex, "127.0.0.1", 0))


def test_http_scopes_runs_by_folder_and_serves_media_with_ranges(http, root):
    ex, url = http
    write_run(root / "a" / "r1", 5)
    run = trex.init(root / "b" / "r2")
    png = b"\x89PNG\r\n\x1a\n" + bytes(range(256))
    run.log_image("img", png, step=2)
    run.finish()
    ex.sync()
    body = get_json(f"{url}/api/runs?path=b")
    assert [r["id"] for r in body["runs"]] == ["b/r2"]
    (m,) = body["media"]
    assert m[:5] == ["b/r2", 0, 2.0, "img", "image"]
    status, headers, data = request(f"{url}/m/{urllib.parse.quote('b/r2', safe='')}/{m[5]}", headers={"Range": "bytes=8-15"})
    assert status == 206 and data == png[8:16] and "immutable" in headers["Cache-Control"]
    assert request(f"{url}/m/{urllib.parse.quote('b/r2', safe='')}/media/..%2F..%2Ftrex.sqlite")[0] == 404
    assert [p for p, _ in get_json(f"{url}/api/tree")] == ["a/r1", "b/r2"]


def test_indexing_never_creates_files_in_closed_run_directories(root, tmp_path):
    write_run(root / "r", 10)
    before = sorted(p.name for p in (root / "r").iterdir())
    ex = explorer(root, tmp_path)
    ex.rows_json("r", 0)
    assert sorted(p.name for p in (root / "r").iterdir()) == before == ["media", "trex.sqlite"]


def test_run_and_folder_info_reach_the_api_for_the_scope_and_its_ancestors(root, tmp_path):
    write_run(root / "sweep" / "a" / "r", 3, info={"notes": "hi"})
    trex.folder_info(root / "sweep", {"question": "q"})
    trex.folder_info(root / "sweep" / "a", {"arm": "a"})
    trex.folder_info(root / "other", {"x": 1})
    ex = explorer(root, tmp_path)
    view = ex.runs("sweep/a")
    assert view.runs[0].info == {"notes": "hi"}
    assert view.folders == {"sweep": {"question": "q"}, "sweep/a": {"arm": "a"}}
    sub = ex.hub.subscribe("sweep")
    trex.folder_info(root / "sweep", {"question": "q2"})
    ex.sync()
    assert drain(sub) == [("folder", {"path": "sweep", "info": {"question": "q2"}})]


def test_cache_from_another_version_is_rebuilt(root, tmp_path, monkeypatch):
    write_run(root / "r", 5)
    explorer(root, tmp_path).close()
    monkeypatch.setattr(trex_index, "CACHE_VERSION", trex_index.CACHE_VERSION + 1)
    ex = Explorer(Crawl(root), tmp_path / "cache")
    assert ex.sync() == ["r"] and ex.run_meta("r").seq == 5


def test_summary_is_the_last_logged_value_of_each_metric_including_non_finite(root, tmp_path):
    run = trex.init(root / "r", commit_interval=0.05)
    run.log({"a": 1.0, "b": float("inf"), "early": 7}, step=0)
    run.log({"a": 2.0, "c": 3.0}, step=1)
    assert wait_for(lambda: committed_rows(root / "r") == 2)
    ex = explorer(root, tmp_path)
    assert {k: ex.run_meta("r").summary[k] for k in ("a", "b", "c", "early", "_step")} == \
        {"a": 2.0, "b": "inf", "c": 3.0, "early": 7.0, "_step": 1.0}
    for i in range(2, 3000):
        run.log({"a": float(i), "c": -float("inf") if i == 2999 else 1.0}, step=i, timestamp=run.created + 0.5 * i)
    run.log({"a": float("nan")}, step=5000, timestamp=run.created + 9000.0)
    run.summary(final=0.5)
    run.finish()
    ex.sync()
    s = ex.run_meta("r").summary
    assert {k: s[k] for k in ("a", "b", "c", "early", "final", "_step", "_runtime")} == \
        {"a": "nan", "b": "inf", "c": "-inf", "early": 7.0, "final": 0.5, "_step": 5000.0, "_runtime": 9000.0}
    fresh = explorer(root, tmp_path, cache="cache2")
    assert fresh.run_meta("r").summary == s

def test_http_static_files_revalidate_by_etag_and_compress_on_request(http):
    _, url = http
    status, headers, body = request(f"{url}/static/app.js")
    assert status == 200 and b"class App" in body and headers["Cache-Control"] == "no-cache"
    assert request(f"{url}/static/app.js", headers={"If-None-Match": headers["ETag"]})[0] == 304
    status, headers, gz = request(f"{url}/static/app.js", headers={"Accept-Encoding": "gzip"})
    assert headers["Content-Encoding"] == "gzip" and zlib.decompress(gz, 31) == body
    assert b'src="/static/app.js"' in request(f"{url}/")[2]
    assert request(f"{url}/static/missing.js")[0] == 404


def test_http_media_ranges_cover_suffixes_open_ends_and_unsatisfiable_starts(http, root):
    ex, url = http
    run = trex.init(root / "r")
    png = b"\x89PNG\r\n\x1a\n" + bytes(range(100))
    run.log_image("img", png, step=0)
    run.finish()
    ex.sync()
    (m,) = ex.runs("").media
    f = f"{url}/m/r/{m.file}"
    assert request(f, headers={"Range": "bytes=-4"})[2] == png[-4:]
    status, headers, body = request(f, headers={"Range": "bytes=100-"})
    assert status == 206 and body == png[100:] and headers["Content-Range"] == f"bytes 100-107/{len(png)}"
    status, headers, _ = request(f, headers={"Range": f"bytes={len(png)}-"})
    assert status == 416 and headers["Content-Range"] == f"bytes */{len(png)}"
    assert request(f)[2] == png


@pytest.mark.parametrize("method,path,body,status", [
    ("GET", "/api/run?path=nope", None, 404),
    ("GET", "/api/run", None, 404),
    ("GET", "/api/rows?path=r&from=x", None, 400),
    ("GET", "/api/nothing", None, 404),
    ("POST", "/api/buckets", b'["not", "an object"]', 400),
    ("POST", "/api/buckets", b"not json", 400),
    ("POST", "/api/buckets", b'{"key": "loss", "level": 0, "index": 0}', 400),
    ("POST", "/api/buckets", b'{"blocks": [{"key": "loss", "level": 0, "index": 0, "which": "fine"}]}', 400),
])
def test_http_bad_requests_are_client_errors_with_a_json_message(http, root, method, path, body, status):
    ex, url = http
    write_run(root / "r", 3)
    ex.sync()
    got, headers, data = request(f"{url}{path}", data=body if method == "POST" else None)
    assert got == status and headers["Content-Type"] == "application/json" and "error" in json.loads(data)


def test_post_bodies_are_consumed_so_a_kept_alive_connection_stays_in_step(http, root):
    _, url = http
    conn = http_client.HTTPConnection(*urllib.parse.urlsplit(url).netloc.split(":"))
    try:
        conn.request("POST", "/api/nothing", body=b'{"ignored": true}')
        r = conn.getresponse()
        assert r.status == 404
        r.read()
        conn.request("GET", "/api/tree")
        r = conn.getresponse()
        assert r.status == 200 and json.loads(r.read()) == []
    finally:
        conn.close()


def test_deleted_run_directory_is_dropped_and_announced(root, tmp_path):
    write_run(root / "a", 3)
    write_run(root / "b", 3)
    ex = explorer(root, tmp_path)
    sub = ex.hub.subscribe("")
    shutil.rmtree(root / "a")
    ex.sync()
    assert drain(sub) == [("delete", {"run": "a"})]
    assert [r.id for r in ex.runs("").runs] == ["b"] and block(ex, "loss", 0, 0, runs=["a"]).paths == []
    with pytest.raises(KeyError):
        ex.run_meta("a")


def test_removed_folder_notes_are_announced_and_unreadable_ones_keep_the_last_good_notes(root, tmp_path, capsys):
    write_run(root / "sweep" / "r", 3)
    trex.folder_info(root / "sweep", question="q")
    trex.folder_info(root, note="top")
    ex = explorer(root, tmp_path)
    sub = ex.hub.subscribe("")
    (root / "sweep" / "trex_info.json").write_text("{not json")
    os.utime(root / "sweep" / "trex_info.json", ns=(1, 1))
    (root / "trex_info.json").unlink()
    ex.sync()
    assert drain(sub) == [("folder", {"path": "", "info": None})]
    assert ex.runs("sweep").folders == {"sweep": {"question": "q"}}
    assert "trex_info.json" in capsys.readouterr().err


def test_subscriber_that_falls_behind_is_dead_and_closing_the_hub_ends_every_subscription():
    slow = trex_index.Subscriber("", maxsize=2)
    for i in range(3):
        slow.put(b"x")
    assert slow.dead
    hub = trex_index.Hub()
    subs = [hub.subscribe(p) for p in ("", "a")]
    hub.close()
    assert all(s.dead for s in subs)


def test_failed_index_batch_changes_nothing_and_the_next_poll_applies_it(root, tmp_path, monkeypatch):
    for name in ("a", "b"):
        write_run(root / name, 3)
    ex = explorer(root, tmp_path)
    for name in ("a", "b"):
        write_run(root / name, 2)
    stage = ex._stage

    def fail_on_b(r, cur, now):
        if r.path == "b":
            raise RuntimeError("disk error")
        return stage(r, cur, now)

    monkeypatch.setattr(ex, "_stage", fail_on_b)
    sub = ex.hub.subscribe("")
    with pytest.raises(RuntimeError):
        ex.sync()
    assert drain(sub) == [] and [ex.run_meta(p).seq for p in ("a", "b")] == [3, 3]
    assert [r.seq for r in ex.runs("").runs] == [3, 3]
    monkeypatch.setattr(ex, "_stage", stage)
    ex.sync()
    assert [ex.run_meta(p).seq for p in ("a", "b")] == [5, 5]


def test_server_urls_bracket_ipv6_hosts():
    servers: list[Any] = [SimpleNamespace(server_address=("127.0.0.1", 13898)), SimpleNamespace(server_address=("::1", 13899, 0, 0))]
    assert server_urls(servers) == ["http://127.0.0.1:13898/", "http://[::1]:13899/"]


def test_binding_an_explicit_port_in_use_is_an_error():
    with socket.socket() as taken:
        taken.bind(("127.0.0.1", 0))
        taken.listen()
        with pytest.raises(OSError):
            bind(None, ["127.0.0.1"], taken.getsockname()[1])


def test_close_stops_polling_ends_subscriptions_and_closes_connections(root, tmp_path):
    write_run(root / "r", 3)
    ex = Explorer(Crawl(root), tmp_path / "cache").start()
    assert ex.ready.wait(10)
    sub = ex.hub.subscribe("")
    c = ex.reader()
    ex.close()
    assert ex._runner is not None and not ex._runner.is_alive() and sub.dead
    ex.release(c)
    with pytest.raises(sqlite3.ProgrammingError):
        c.execute("SELECT 1")
    with pytest.raises(sqlite3.ProgrammingError):
        ex._writer.execute("SELECT 1")


def slow_scans(monkeypatch, seconds):
    """Make every inline scan take `seconds`; the paths scanned so far."""
    real, scanned = crawl.scan, []

    def slow(job):
        scanned.append(job.path)
        time.sleep(seconds)
        return real(job)

    monkeypatch.setattr(crawl, "scan", slow)
    return scanned


def test_closing_in_the_middle_of_a_pass_stops_it_after_the_run_being_scanned(root, tmp_path, monkeypatch):
    for i in range(30):
        write_run(root / f"r{i}", 3)
    scanned = slow_scans(monkeypatch, 0.1)
    ex = Explorer(Crawl(root, 1), tmp_path / "cache").start()
    while len(scanned) < 3:
        time.sleep(0.01)
    t0 = time.time()
    ex.close()
    assert time.time() - t0 < 0.5 and len(scanned) < 6


def test_close_returns_while_a_long_scan_finishes_and_the_scan_writes_nothing(root, tmp_path, monkeypatch, capfd):
    write_run(root / "r0", 3)
    scanned = slow_scans(monkeypatch, 1.5)
    monkeypatch.setattr(trex_index, "CLOSE_WAIT", 0.2)
    errors = []
    monkeypatch.setattr(threading, "excepthook", lambda a: errors.append(a.exc_value))
    ex = Explorer(Crawl(root, 1), tmp_path / "cache").start()
    while not scanned:
        time.sleep(0.01)
    t0 = time.time()
    ex.close()
    assert time.time() - t0 < 0.5
    runner = ex._runner
    assert runner is not None
    runner.join(5)
    assert not runner.is_alive() and not errors and "[trex]" not in capfd.readouterr().err


def test_a_failed_index_pass_is_logged_and_polling_continues(root, tmp_path, monkeypatch, capfd):
    write_run(root / "r", 3)
    apply, calls = Explorer.apply, []

    def fail_once(self, results):
        calls.append(len(results))
        if len(calls) == 1:
            raise sqlite3.OperationalError("database is locked")
        apply(self, results)

    monkeypatch.setattr(Explorer, "apply", fail_once)
    ex = Explorer(Crawl(root), tmp_path / "cache").start()
    assert wait_for(lambda: ex.runs("").runs, timeout=10)
    assert ex.ready.is_set() and "index pass failed" in capfd.readouterr().err
