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
from typing import Any

import numpy as np
import pytest

import trex
from trex import chunks, server, tiles
from trex import index as trex_index
from trex.format import FORMAT, connect_ro, connect_rw
from trex.index import Explorer
from trex.workspace import unframe_bundle
from trex.server import bind, check_root, serve
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
    ex = Explorer(root, tmp_path / cache, workers=workers)
    ex.rewalk()
    ex.poll()
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


def expected_tile(rows, key, level, idx):
    pts = [(s, d[key], t) for s, t, d in rows if key in d]
    s, v, t = (np.array(x, dtype=float) for x in zip(*pts))
    return tiles.build(s, v, t, level, idx)


def run_points(root, path, key):
    c = connect_ro(root / path)
    try:
        kid = {n: i for i, n in chunks.key_names(c).items()}[key]
        return chunks.metric(c, kid)
    finally:
        c.close()


def index_dump(ex):
    """Everything the index stores, without access and build times."""
    c = sqlite3.connect(ex.db_path)
    out = {t: c.execute(f"SELECT * FROM {t} ORDER BY 1, 2").fetchall() for t in ("runs", "media")}
    out["tiles"] = c.execute("SELECT path, key, level, idx, kind, seq, data FROM tiles ORDER BY 1, 2, 3, 4").fetchall()
    c.close()
    out["runs"] = [(p, {k: v for k, v in json.loads(s).items() if k != "tiles_t"}) for p, s in out["runs"]]
    return out


def top_n(ex, path, key):
    return sum(int(tiles.decode(b).n.sum()) for b in ex.tiles([[path, key, "top"]])[0])


def test_finished_run_has_top_tiles_covering_every_finite_point_of_every_metric(root, tmp_path):
    rows = mixed_rows(7053)
    write_chunked(root / "r", chunked(rows, [700, 1, 2000, 323, 1024, 5, 3000]))
    ex = explorer(root, tmp_path)
    meta = ex.run_meta("r")
    assert (meta["state"], meta["seq"], meta["tiles_seq"]) == ("finished", 7053, 7053)
    assert meta["keys"] == ["eval/r", "loss", "lr", "x"]
    for key in meta["keys"]:
        blobs = ex.tiles([["r", key, "top"]])[0]
        assert 1 <= len(blobs) <= 2
        level = {tiles.decode(b).level for b in blobs}
        assert len(level) == 1
        for b in blobs:
            t = tiles.decode(b)
            assert b == expected_tile(rows, key, t.level, t.index)
        finite = sum(1 for _, _, d in rows if key in d and np.isfinite(d[key]))
        assert top_n(ex, "r", key) == finite


def test_overview_tiles_are_the_top_tiles_merged_overview_up_levels(root, tmp_path):
    rows = mixed_rows(7053)
    write_chunked(root / "r", chunked(rows, [700, 1, 2000, 323, 1024, 5, 3000]))
    ex = explorer(root, tmp_path)
    for key in ex.run_meta("r")["keys"]:
        top, overview = ex.tiles([["r", key, "top"], ["r", key, "overview"]])
        assert overview == tiles.coarsen(top, trex_index.OVERVIEW_UP)
        assert {tiles.decode(b).level for b in overview} == {tiles.decode(top[0]).level + trex_index.OVERVIEW_UP}
        assert sum(int(tiles.decode(b).n.sum()) for b in overview) == top_n(ex, "r", key)


def test_tile_bundle_answers_every_run_in_scope_like_per_run_requests(root, tmp_path):
    for name in ("a/r1", "a/r2", "b/r3"):
        write_run(root / name, 600)
    write_run(root / "a" / "short", 1)
    ex = explorer(root, tmp_path)
    for kind in ("top", "overview"):
        got = ex.tile_bundle("odd", kind, "a")
        assert [p for p, _ in got] == ["a/r1", "a/r2"]
        assert [b for _, b in got] == ex.tiles([["a/r1", "odd", kind], ["a/r2", "odd", kind]])


def test_a_bundle_body_follows_runs_added_and_removed(root, tmp_path):
    write_run(root / "a" / "r1", 600)
    ex = explorer(root, tmp_path)
    body = lambda: ex.tile_bundle_body("odd", "top", "")
    assert body() == trex_index.frame_bundle(ex.tile_bundle("odd", "top", "")) and body() is body()
    write_run(root / "a" / "r2", 600)
    ex.rewalk()
    ex.poll()
    assert [p for p, _ in unframe_bundle(body())] == ["a/r1", "a/r2"]
    shutil.rmtree(root / "a" / "r1")
    ex.rewalk()
    ex.poll()
    assert unframe_bundle(body()) == ex.tile_bundle("odd", "top", "") and [p for p, _ in unframe_bundle(body())] == ["a/r2"]


def test_the_runs_body_follows_runs_added_and_folder_notes(root, tmp_path):
    write_run(root / "a" / "r1", 5)
    ex = explorer(root, tmp_path)
    body = lambda: json.loads(ex.runs_body("a"))
    assert body() == json.loads(trex_index.dumps(ex.runs("a"))) and [m["id"] for m in body()["runs"]] == ["a/r1"]
    write_run(root / "a" / "r2", 5)
    trex.folder_info(root / "a", note="x")
    ex.rewalk()
    ex.poll()
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


def test_every_response_isolates_the_page_so_it_may_share_memory_with_its_workers(http):
    _, url = http
    for path in ("/", "/static/worker.js", "/api/runs?path="):
        with urllib.request.urlopen(f"{url}{path}") as r:
            assert (r.headers["Cross-Origin-Opener-Policy"], r.headers["Cross-Origin-Embedder-Policy"]) == ("same-origin", "require-corp"), path


def test_a_slab_holds_the_finished_runs_from_their_top_tiles_or_their_run_files(root, tmp_path):
    write_run(root / "a" / "short", 600)
    write_run(root / "a" / "long", 5000)
    live = write_run(root / "a" / "live", 600, finish=False)
    ex = explorer(root, tmp_path)
    top = lambda p: tiles.decode(ex.tiles([[p, "loss", "top"]])[0][0]).level
    level = top("a/short") + 1
    assert top("a/long") > level
    s = tiles.decode_slab(ex.slab_body("loss", level, 0, "a"))
    assert s.paths == ["a/long", "a/short"]
    want = [tiles.decode(ex.tiles([["a/long", "loss", level, 0]])[0][0]),
            next(t for t in map(tiles.decode, tiles.coarsen(ex.tiles([["a/short", "loss", "top"]])[0], 1)) if t.index == 0)]
    for i, t in enumerate(want):
        got = slice(s.first[i], s.first[i + 1])
        assert np.array_equal(s.bucket[got], t.bucket) and np.array_equal(s.n[got], t.n)
        np.testing.assert_allclose(s.mean[got], t.mean, rtol=1e-6)
    assert ex.slab_body("loss", level, 0, "a") is ex.slab_body("loss", level, 0, "a")
    live.finish()


def test_a_slab_builds_the_tiles_it_lacks_on_the_process_pool_as_the_run_files_hold_them(root, tmp_path, monkeypatch):
    monkeypatch.setattr(trex_index, "INLINE_TILES", 0)
    for name in ("a/r1", "a/r2"):
        write_run(root / name, 3000)
    ex = explorer(root, tmp_path, workers=2)
    level = tiles.decode(ex.tiles([["a/r1", "loss", "top"]])[0][0]).level - 1
    s = tiles.decode_slab(ex.slab_body("loss", level, 0, "a"))
    for i, path in enumerate(s.paths):
        t = tiles.decode(trex_index.build_tile(str(root / path), "loss", level, 0)[0])
        got = slice(s.first[i], s.first[i + 1])
        assert np.array_equal(s.bucket[got], t.bucket) and np.array_equal(s.n[got], t.n)


def test_a_slab_stays_while_running_runs_grow_and_changes_once_one_finishes(root, tmp_path, monkeypatch):
    monkeypatch.setattr(trex_index, "TOP_REFRESH", 0.0)
    write_run(root / "a" / "done", 300)
    live = write_run(root / "a" / "live", 300, finish=False)
    ex = explorer(root, tmp_path)
    level = tiles.decode(ex.tiles([["a/done", "loss", "top"]])[0][0]).level + 1
    first = ex.slab_body("loss", level, 0, "a")
    for i in range(300, 400):
        live.log({"loss": 1.0 / (i + 1)}, step=i)
    assert wait_for(lambda: (ex.poll(), ex.records["a/live"]["tiles_seq"] >= 400)[1])
    assert ex.slab_body("loss", level, 0, "a") is first and tiles.decode_slab(first).paths == ["a/done"]
    live.finish()
    assert wait_for(lambda: (ex.poll(), ex.records["a/live"]["state"] != "running")[1])
    assert tiles.decode_slab(ex.slab_body("loss", level, 0, "a")).paths == ["a/done", "a/live"]


def saved_levels(ex):
    d = ex.cache_dir / "levels"
    return sorted(f.name for f in d.iterdir() if not f.name.endswith(".tmp")) if d.exists() else []


def test_a_new_explorer_cuts_slabs_from_the_levels_an_earlier_one_saved(root, tmp_path, monkeypatch):
    for name in ("a/r1", "a/r2", "b/r3"):
        write_run(root / name, 3000)
    ex = explorer(root, tmp_path)
    level = tiles.decode(ex.tiles([["a/r1", "loss", "overview"]])[0][0]).level
    want = {(lv, scope): ex.slab_body("loss", lv, 0, scope) for lv in (level, level - 1) for scope in ("", "a")}
    assert wait_for(lambda: saved_levels(ex))
    ex.close()

    def unread(*_):
        raise AssertionError("tiles decoded")
    monkeypatch.setattr(tiles, "stack", unread)
    again = explorer(root, tmp_path)
    assert {k: again.slab_body("loss", *k[:1], 0, k[1]) for k in want} == want


def test_saved_levels_are_left_unused_once_the_finished_runs_change(root, tmp_path):
    for name in ("a/r1", "a/r2"):
        write_run(root / name, 3000)
    ex = explorer(root, tmp_path)
    level = tiles.decode(ex.tiles([["a/r1", "loss", "overview"]])[0][0]).level
    ex.slab_body("loss", level, 0, "a")
    assert wait_for(lambda: saved_levels(ex))
    ex.close()
    write_run(root / "a" / "r3", 3000, metrics=lambda i: {"loss": 5.0})
    again = explorer(root, tmp_path)
    s = tiles.decode_slab(again.slab_body("loss", level, 0, "a"))
    assert s.paths == ["a/r1", "a/r2", "a/r3"] and np.all(s.mean[s.first[2]:s.first[3]] == 5.0)


def test_a_metric_saves_its_levels_at_most_once_every_levels_save_every(root, tmp_path, monkeypatch):
    monkeypatch.setattr(trex_index, "LEVELS_SAVE_EVERY", 3600.0)
    write_run(root / "a" / "r1", 3000)
    ex = explorer(root, tmp_path)
    level = tiles.decode(ex.tiles([["a/r1", "loss", "overview"]])[0][0]).level
    ex.slab_body("loss", level, 0, "a")
    assert wait_for(lambda: saved_levels(ex))
    f = ex.cache_dir / "levels" / saved_levels(ex)[0]
    before = f.read_bytes()
    write_run(root / "a" / "r2", 3000)
    ex.rewalk()
    ex.poll()
    assert tiles.decode_slab(ex.slab_body("loss", level, 0, "a")).paths == ["a/r1", "a/r2"]
    time.sleep(0.5)
    assert f.read_bytes() == before


def test_saved_levels_beyond_their_budget_go_least_recently_used_first(tmp_path):
    for i, name in enumerate(("old", "mid", "new")):
        f = tmp_path / name
        f.write_bytes(b"x" * 100)
        os.utime(f, (1000 + i, 1000 + i))
    trex_index._bound_dir(tmp_path, 250)
    assert sorted(f.name for f in tmp_path.iterdir()) == ["mid", "new"]


def test_a_changed_cache_version_deletes_the_saved_levels(root, tmp_path, monkeypatch):
    write_run(root / "a" / "r1", 3000)
    ex = explorer(root, tmp_path)
    ex.slab_body("loss", tiles.decode(ex.tiles([["a/r1", "loss", "overview"]])[0][0]).level, 0, "a")
    assert wait_for(lambda: saved_levels(ex))
    ex.close()
    monkeypatch.setattr(trex_index, "CACHE_VERSION", trex_index.CACHE_VERSION + 1)
    assert saved_levels(explorer(root, tmp_path)) == []


def test_http_slab_answers_the_slab_and_info_states_the_protocol(http, root):
    ex, url = http
    write_run(root / "x" / "r1", 300)
    ex.rewalk()
    ex.poll()
    body = json.dumps({"key": "loss", "level": 0, "index": 0, "scope": "x"}).encode()
    with urllib.request.urlopen(urllib.request.Request(f"{url}/api/tiles/slab", data=body)) as r:
        assert r.read() == ex.slab_body("loss", 0, 0, "x")
    with urllib.request.urlopen(f"{url}/api/info") as r:
        assert json.loads(r.read())["protocol"] == server.PROTOCOL


def test_http_tile_bundle_frames_paths_and_tiles(http, root):
    ex, url = http
    for name in ("x/r1", "x/r2"):
        write_run(root / name, 300)
    ex.rewalk()
    ex.poll()
    req = urllib.request.Request(f"{url}/api/tiles/bundle", data=json.dumps({"key": "loss", "kind": "top", "scope": "x"}).encode())
    with urllib.request.urlopen(req) as r:
        body = r.read()
    n, off, got = int.from_bytes(body[:4], "little"), 4, []
    for _ in range(n):
        plen = int.from_bytes(body[off: off + 4], "little")
        path = body[off + 4: off + 4 + plen].decode()
        off += 4 + plen + (-plen % 4)
        k = int.from_bytes(body[off: off + 4], "little")
        off += 4
        blobs = []
        for _ in range(k):
            m = int.from_bytes(body[off: off + 4], "little")
            blobs.append(body[off + 4: off + 4 + m])
            off += 4 + m
        got.append((path, blobs))
    assert off == len(body) and got == ex.tile_bundle("loss", "top", "x")


def test_finer_tiles_are_built_from_the_run_and_served_from_cache(root, tmp_path, monkeypatch):
    rows = mixed_rows(5000)
    write_chunked(root / "r", chunked(rows, [1000] * 5))
    ex = explorer(root, tmp_path)
    want = [["r", "loss", 2, 3], ["r", "lr", 0, 7], ["r", "x", -1, 0], ["r", "loss", 5, 99]]
    first = ex.tiles(want)
    assert [b for (b,) in first] == [expected_tile(rows, k, L, i) for _, k, L, i in want]
    monkeypatch.setattr(Explorer, "_build", lambda *a: pytest.fail("cached tile rebuilt"))
    assert ex.tiles(want) == first
    assert ex.tiles([["r", "nope", 0, 0], ["nope", "loss", 0, 0], ["nope", "loss", "top"]]) == [[], [], []]
    with pytest.raises(ValueError):
        ex.tiles([["r", "loss", tiles.MAX_LEVEL + 1, 0]])


def test_cached_tile_is_rebuilt_only_when_new_rows_reach_its_step_range(root, tmp_path, monkeypatch):
    run = trex.init(root / "r", commit_interval=0.01)
    for i in range(3000):
        run.log({"loss": float(i)}, step=i)
    assert wait_for(lambda: committed_rows(root / "r") == 3000)
    ex = explorer(root, tmp_path)
    early, edge = ["r", "loss", 0, 0], ["r", "loss", 0, 11]
    ex.tiles([early, edge])
    for i in range(3000, 3500):
        run.log({"loss": float(i)}, step=i)
    run.finish()
    ex.poll()
    built = []
    real = Explorer._build
    monkeypatch.setattr(Explorer, "_build", lambda self, *a: (built.append(list(a)), real(self, *a))[1])
    got = ex.tiles([early, edge])
    assert built == [edge[:2] + edge[2:]]
    s, v, t = run_points(root, "r", "loss")
    assert got == [[tiles.build(s, v, t, 0, 0)], [tiles.build(s, v, t, 0, 11)]]
    assert tiles.decode(got[1][0]).n.sum() == 256


def test_tile_cache_evicts_least_recently_used_finer_tiles_and_keeps_top_tiles(root, tmp_path, monkeypatch):
    write_run(root / "r", 4000)
    ex = explorer(root, tmp_path)
    top = ex.tiles([["r", "loss", "top"]])
    size = len(ex.tiles([["r", "loss", 0, 0]])[0][0])
    monkeypatch.setattr(trex_index, "TILE_CACHE_BYTES", 4 * size)
    for i in range(1, 15):
        ex.tiles([["r", "loss", 0, i]])
    c = sqlite3.connect(ex.db_path)
    cached = c.execute("SELECT level, idx FROM tiles WHERE kind = 0 ORDER BY idx").fetchall()
    stored = c.execute("SELECT coalesce(sum(length(data)), 0) FROM tiles WHERE kind = 0").fetchone()[0]
    c.close()
    assert stored <= 4 * size and (0, 14) in cached and (0, 0) not in cached
    assert ex.tiles([["r", "loss", "top"]]) == top


def test_live_run_streams_contiguous_rows_and_refreshes_top_tiles_on_finish(root, tmp_path):
    run = trex.init(root / "live", commit_interval=0.05)
    run.log({"x": 0})
    assert wait_for(lambda: committed_rows(root / "live") == 1)
    ex = explorer(root, tmp_path)
    sub = ex.hub.subscribe("")
    seen = ex.run_meta("live")["seq"]
    assert ex.run_meta("live")["tiles_seq"] == seen
    for i in range(1, 30):
        run.log({"x": i})
        if i % 10 == 0:
            time.sleep(0.15)
            ex.poll()
    run.finish()
    ex.poll()
    events = drain(sub)
    for ev, data in events:
        if ev == "rows":
            assert data["seq0"] == seen
            seen += len(data["rows"])
    assert seen == 30
    last = [d for e, d in events if e == "run"][-1]
    assert (last["state"], last["tiles_seq"]) == ("finished", 30)
    assert top_n(ex, "live", "x") == 30


def test_growing_run_top_tiles_refresh_after_the_refresh_interval(root, tmp_path, monkeypatch):
    run = trex.init(root / "r", commit_interval=0.01)
    run.log({"x": 0.0}, step=0)
    assert wait_for(lambda: committed_rows(root / "r") == 1)
    ex = explorer(root, tmp_path)
    for i in range(1, 100):
        run.log({"x": float(i)}, step=i)
    assert wait_for(lambda: committed_rows(root / "r") == 100)
    ex.poll()
    assert (ex.run_meta("r")["seq"], ex.run_meta("r")["tiles_seq"]) == (100, 1)
    monkeypatch.setattr(trex_index, "TOP_REFRESH", 0.0)
    ex.poll()
    assert ex.run_meta("r")["tiles_seq"] == 100 and top_n(ex, "r", "x") == 100
    run.finish()


def test_silent_running_run_becomes_crashed_with_complete_top_tiles(root, tmp_path, monkeypatch):
    run = write_run(root / "r", 10, finish=False)
    assert wait_for(lambda: committed_rows(root / "r") == 10)
    ex = explorer(root, tmp_path)
    for i in range(10, 20):
        run.log({"loss": 0.0}, step=i)
    assert wait_for(lambda: committed_rows(root / "r") == 20)
    ex.poll()
    assert ex.run_meta("r")["state"] == "running" and ex.run_meta("r")["tiles_seq"] == 10
    monkeypatch.setattr(trex_index, "CRASH_AFTER", 0.0)
    ex.poll()
    meta = ex.run_meta("r")
    assert meta["state"] == "crashed" and meta["tiles_seq"] == 20 and top_n(ex, "r", "loss") == 20
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
    monkeypatch.setattr(trex_index, "INLINE_BYTES", 0)
    pooled = []
    real = Explorer._sync_pool
    monkeypatch.setattr(Explorer, "_sync_pool", lambda self, todo: (pooled.append(len(todo)), real(self, todo)))
    pool = explorer(root, tmp_path, workers=4, cache="pool")
    assert pooled == [12]
    assert len(inline.records) == 12
    a, b = index_dump(inline), index_dump(pool)
    assert a == b
    assert {p: sorted(x.name for x in p.iterdir()) for p in root.glob("sweep/*")} == before
    assert [m[4] for m in a["media"]] == ["html", "image"] and a["media"][0][6] == zlib.crc32(b"<p>1</p>")
    for path, st in a["runs"]:
        assert st["tiles_seq"] == st["seq"]
        assert {k for p, k, *_ in a["tiles"] if p == path} == set(st["keys"])


def test_live_rows_are_contiguous_while_the_writer_commits(root, tmp_path):
    run = trex.init(root / "live", commit_interval=0.01)
    run.log({"x": 0.0}, step=0)
    assert wait_for(lambda: committed_rows(root / "live") == 1)
    ex = explorer(root, tmp_path)
    sub = ex.hub.subscribe("")
    seen = ex.run_meta("live")["seq"]
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
        ex.poll()
        polls += 1
        time.sleep(0.01)
    run.finish()
    ex.poll()
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
    uid = ex.run_meta("r")["uid"]
    assert ex.run_meta("r")["seq"] == 50
    ex.tiles([["r", "loss", 0, 0]])
    sub = ex.hub.subscribe("")
    shutil.copy(old, root / "r" / "trex.sqlite")
    ex.poll()
    events = drain(sub)
    assert events[0] == ("delete", {"run": "r"})
    meta = ex.run_meta("r")
    assert (meta["uid"], meta["seq"], meta["tiles_seq"]) == (uid, 20, 20)
    assert top_n(ex, "r", "loss") == 20 and tiles.decode(ex.tiles([["r", "loss", 0, 0]])[0][0]).n.sum() == 20


def test_http_stream_sends_rows_beyond_top_tiles_then_live_rows(http, root):
    ex, url = http
    run = trex.init(root / "s" / "r", commit_interval=0.05)
    run.log({"x": 0})
    assert wait_for(lambda: committed_rows(root / "s" / "r") == 1)
    ex.rewalk()
    ex.poll()
    for i in range(1, 3):
        run.log({"x": i})
    assert wait_for(lambda: committed_rows(root / "s" / "r") == 3)
    ex.poll()
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
    ex.poll()
    ev, data = next_event()
    while ev == "run":
        ev, data = next_event()
    assert ev == "rows" and data["seq0"] == 3 and data["rows"][0][2] == {"x": 3, "bad": "inf"}
    run.finish()
    r.close()


def test_http_tiles_frame_each_request_in_order(http, root):
    ex, url = http
    write_run(root / "r", 3000)
    ex.rewalk()
    ex.poll()
    want = [["r", "loss", "top"], ["r", "nope", 0, 0], ["r", "odd", 1, 2], ["r", "loss", 0, 0]]
    expect = ex.tiles(want)
    req = urllib.request.Request(f"{url}/api/tiles", data=json.dumps(want).encode())
    with urllib.request.urlopen(req) as r:
        body = r.read()
    got, off = [], 0
    while off < len(body):
        k = int.from_bytes(body[off: off + 4], "little")
        off += 4
        blobs = []
        for _ in range(k):
            n = int.from_bytes(body[off: off + 4], "little")
            blobs.append(body[off + 4: off + 4 + n])
            off += 4 + n
        got.append(blobs)
    assert got == expect and [len(b) for b in got] == [1, 0, 1, 1]
    bad = urllib.request.Request(f"{url}/api/tiles", data=json.dumps([["r", "loss", 999, 0]]).encode())
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(bad)
    assert e.value.code == 400


def test_walk_finds_nested_runs_and_skips_hidden_env_and_run_internals(root, tmp_path):
    write_run(root / "a" / "r1", 3)
    write_run(root / "a" / "b" / "r2", 3)
    write_run(root / ".hidden" / "r3", 3)
    write_run(root / "node_modules" / "r4", 3)
    write_run(root / "a" / "r1" / "nested" / "r5", 3)
    (root / "empty" / "dir").mkdir(parents=True)
    ex = explorer(root, tmp_path)
    assert [p for p, _ in ex.tree()] == ["a/b/r2", "a/r1"]
    assert [r["id"] for r in ex.runs("a/b")["runs"]] == ["a/b/r2"]
    assert [r["id"] for r in ex.runs("a/b/r")["runs"]] == []


def test_rewritten_run_is_dropped_and_reindexed(root, tmp_path):
    write_run(root / "r", 50)
    ex = explorer(root, tmp_path)
    uid = ex.run_meta("r")["uid"]
    sub = ex.hub.subscribe("")
    shutil.rmtree(root / "r")
    write_run(root / "r", 5)
    ex.poll()
    events = drain(sub)
    assert events[0] == ("delete", {"run": "r"})
    meta = ex.run_meta("r")
    assert meta["uid"] != uid and meta["seq"] == 5


def test_restart_reuses_cache_without_rereading_unchanged_runs(root, tmp_path, monkeypatch):
    write_run(root / "r", 1500)
    ex = explorer(root, tmp_path)
    before = ex.run_meta("r")
    ex2 = Explorer(root, tmp_path / "cache")
    ex2.rewalk()
    assert ex2.poll() == []
    assert ex2.run_meta("r") == before


def test_refuses_to_crawl_home_or_filesystem_root():
    with pytest.raises(SystemExit):
        check_root("~", force=False)
    with pytest.raises(SystemExit):
        check_root("/", force=False)


@pytest.fixture
def http(root, tmp_path, http_server):
    ex = Explorer(root, tmp_path / "cache")
    return ex, http_server(serve(ex, "127.0.0.1", 0))


def test_http_scopes_runs_by_folder_and_serves_media_with_ranges(http, root):
    ex, url = http
    write_run(root / "a" / "r1", 5)
    run = trex.init(root / "b" / "r2")
    png = b"\x89PNG\r\n\x1a\n" + bytes(range(256))
    run.log_image("img", png, step=2)
    run.finish()
    ex.rewalk()
    ex.poll()
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
    body = ex.runs("sweep/a")
    assert body["runs"][0]["info"] == {"notes": "hi"}
    assert body["folders"] == {"sweep": {"question": "q"}, "sweep/a": {"arm": "a"}}
    sub = ex.hub.subscribe("sweep")
    trex.folder_info(root / "sweep", {"question": "q2"})
    ex.rewalk()
    assert drain(sub) == [("folder", {"path": "sweep", "info": {"question": "q2"}})]


def test_cache_from_another_version_is_rebuilt(root, tmp_path, monkeypatch):
    write_run(root / "r", 5)
    explorer(root, tmp_path)
    monkeypatch.setattr(trex_index, "CACHE_VERSION", trex_index.CACHE_VERSION + 1)
    ex = Explorer(root, tmp_path / "cache")
    ex.rewalk()
    assert ex.poll() == ["r"] and ex.run_meta("r")["seq"] == 5


def test_summary_is_the_last_logged_value_of_each_metric_including_non_finite(root, tmp_path):
    run = trex.init(root / "r", commit_interval=0.05)
    run.log({"a": 1.0, "b": float("inf"), "early": 7}, step=0)
    run.log({"a": 2.0, "c": 3.0}, step=1)
    assert wait_for(lambda: committed_rows(root / "r") == 2)
    ex = explorer(root, tmp_path)
    assert {k: ex.run_meta("r")["summary"][k] for k in ("a", "b", "c", "early", "_step")} == \
        {"a": 2.0, "b": "inf", "c": 3.0, "early": 7.0, "_step": 1.0}
    for i in range(2, 3000):
        run.log({"a": float(i), "c": -float("inf") if i == 2999 else 1.0}, step=i, timestamp=run.created + 0.5 * i)
    run.log({"a": float("nan")}, step=5000, timestamp=run.created + 9000.0)
    run.summary(final=0.5)
    run.finish()
    ex.poll()
    s = ex.run_meta("r")["summary"]
    assert {k: s[k] for k in ("a", "b", "c", "early", "final", "_step", "_runtime")} == \
        {"a": "nan", "b": "inf", "c": "-inf", "early": 7.0, "final": 0.5, "_step": 5000.0, "_runtime": 9000.0}
    fresh = explorer(root, tmp_path, cache="cache2")
    assert fresh.run_meta("r")["summary"] == s

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
    ex.rewalk()
    ex.poll()
    (m,) = ex.runs("")["media"]
    f = f"{url}/m/r/{m[5]}"
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
    ("POST", "/api/tiles", b'{"not": "a list"}', 400),
    ("POST", "/api/tiles", b"not json", 400),
    ("POST", "/api/tiles/bundle", b'{"key": "loss", "kind": "fine"}', 400),
])
def test_http_bad_requests_are_client_errors_with_a_json_message(http, root, method, path, body, status):
    ex, url = http
    write_run(root / "r", 3)
    ex.rewalk()
    ex.poll()
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
    ex.rewalk()
    assert drain(sub) == [("delete", {"run": "a"})]
    assert [r["id"] for r in ex.runs("")["runs"]] == ["b"] and ex.tiles([["a", "loss", "top"]]) == [[]]
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
    ex.rewalk()
    assert drain(sub) == [("folder", {"path": "", "info": None})]
    assert ex.runs("sweep")["folders"] == {"sweep": {"question": "q"}}
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
        if r["path"] == "b":
            raise RuntimeError("disk error")
        return stage(r, cur, now)

    monkeypatch.setattr(ex, "_stage", fail_on_b)
    sub = ex.hub.subscribe("")
    with pytest.raises(RuntimeError):
        ex.poll()
    assert drain(sub) == [] and [ex.run_meta(p)["seq"] for p in ("a", "b")] == [3, 3]
    assert [r["seq"] for r in ex.runs("")["runs"]] == [3, 3]
    monkeypatch.setattr(ex, "_stage", stage)
    ex.poll()
    assert [ex.run_meta(p)["seq"] for p in ("a", "b")] == [5, 5]


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
    ex = Explorer(root, tmp_path / "cache").start()
    assert ex.ready.wait(10)
    sub = ex.hub.subscribe("")
    c = ex.reader()
    ex.close()
    assert ex._poller is not None and not ex._poller.is_alive() and sub.dead
    ex.release(c)
    with pytest.raises(sqlite3.ProgrammingError):
        c.execute("SELECT 1")
    with pytest.raises(sqlite3.ProgrammingError):
        ex._writer.execute("SELECT 1")


def test_a_shared_tile_budget_evicts_the_least_recently_used_tiles_of_any_explorer(tmp_path, monkeypatch):
    roots = [tmp_path / "a", tmp_path / "b"]
    for r in roots:
        write_run(r / "r", 300)
    clock = iter(range(1, 10_000))
    monkeypatch.setattr(trex_index.time, "time", lambda: float(next(clock)))
    probe = Explorer(roots[0], tmp_path / "probe")
    probe.rewalk()
    probe.poll()
    size = len(probe.tiles([["r", "loss", -6, 0]])[0][0])
    budget = trex_index.TileBudget(limit=int(4.5 * size))
    a, b = (Explorer(r, tmp_path / "cache", budget=budget) for r in roots)
    for ex in (a, b):
        ex.rewalk()
        ex.poll()
    a.tiles([["r", "loss", -6, i] for i in range(3)])
    b.tiles([["r", "loss", -6, i] for i in range(3)])

    def cached(ex):
        c = sqlite3.connect(ex.db_path)
        try:
            idx = [i for (i,) in c.execute("SELECT idx FROM tiles WHERE kind = 0 ORDER BY idx")]
            return idx, c.execute("SELECT coalesce(sum(length(data)), 0) FROM tiles WHERE kind = 0").fetchone()[0]
        finally:
            c.close()

    (kept_a, bytes_a), (kept_b, bytes_b) = cached(a), cached(b)
    assert kept_a == [2] and kept_b == [0, 1, 2] and budget.used() == bytes_a + bytes_b <= 4.5 * size


def slow_scans(monkeypatch, seconds):
    """Make every inline scan take `seconds`; the paths scanned so far."""
    real, scanned = trex_index.scan, []

    def slow(job):
        scanned.append(job["path"])
        time.sleep(seconds)
        return real(job)

    monkeypatch.setattr(trex_index, "scan", slow)
    return scanned


def test_closing_in_the_middle_of_a_pass_stops_it_after_the_run_being_scanned(root, tmp_path, monkeypatch):
    for i in range(30):
        write_run(root / f"r{i}", 3)
    scanned = slow_scans(monkeypatch, 0.1)
    ex = Explorer(root, tmp_path / "cache", workers=1).start()
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
    ex = Explorer(root, tmp_path / "cache", workers=1).start()
    while not scanned:
        time.sleep(0.01)
    t0 = time.time()
    ex.close()
    assert time.time() - t0 < 0.5
    poller = ex._poller
    assert poller is not None
    poller.join(5)
    assert not poller.is_alive() and not errors and "[trex]" not in capfd.readouterr().err


def test_a_failed_index_pass_is_logged_and_polling_continues(root, tmp_path, monkeypatch, capfd):
    write_run(root / "r", 3)
    apply, calls = Explorer.apply, []

    def fail_once(self, results):
        calls.append(len(results))
        if len(calls) == 1:
            raise sqlite3.OperationalError("database is locked")
        apply(self, results)

    monkeypatch.setattr(Explorer, "apply", fail_once)
    ex = Explorer(root, tmp_path / "cache").start()
    assert wait_for(lambda: ex.runs("")["runs"], timeout=10)
    assert ex.ready.is_set() and "index pass failed" in capfd.readouterr().err
