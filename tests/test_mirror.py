import http.client as http_client
import json
import shutil
import sqlite3
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import trex
from trex import buckets as bk, crawl, index, mirror, server
from trex.crawl import Crawl
from trex.index import KEPT, Ask, Dump, Explorer, Rows, Unfetched
from trex.mirror import Pull, Unreachable, Upstream
from trex.server import Server
from trex.workspace import Member, Workspace

from helpers import PNG, committed_rows, node_of, post_json, wait_for, write_run

type Source = tuple[Explorer, Server, str]


@pytest.fixture
def runs(tmp_path: Path) -> Path:
    d = tmp_path / "runs"
    write_run(d / "a" / "r1", n=300, image=True)
    write_run(d / "a" / "r2", n=50)
    write_run(d / "b" / "r3", n=1000, metrics=lambda i: {"loss": 1 / (i + 1), "acc": i / 1000})
    trex.folder_info(d / "a", trex={"group_by": "run~1"})
    return d


@pytest.fixture
def source(runs: Path, tmp_path: Path, http_server: Callable[[Server], str]) -> Source:
    """The upstream: an Explorer of `runs`, indexed, its server and the URL serving it."""
    ex = Explorer(Crawl(runs, 1), tmp_path / "source-cache")
    ex.sync()
    srv = server.serve(node_of(ex), "127.0.0.1", 0)
    return ex, srv, http_server(srv)


def mirror_of(url: str, cache: Path) -> Explorer:
    return Explorer(Pull(Upstream.at(url), "box:/runs"), cache)


def synced(url: str, cache: Path) -> Explorer:
    m = mirror_of(url, cache)
    m.sync()
    return m


def stop(ex: Explorer, srv: Server) -> None:
    """End the upstream: its server, with the connections it had open, and its index."""
    srv.shutdown()
    srv.server_close()
    ex.close()


def held(ex: Explorer) -> dict[str, list[Any]]:
    """Everything an index stores, without when its levels changed there."""
    c = sqlite3.connect(ex.db_path)
    try:
        out: dict[str, list[Any]] = {t: sorted(c.execute(f"SELECT * FROM {t}").fetchall()) for t in ("media", "metrics", "levels", "folders")}
        out["runs"] = [(p, {k: v for k, v in json.loads(s).items() if k != "compiled_t"})
                       for p, s in c.execute("SELECT path, record FROM runs ORDER BY path")]
    finally:
        c.close()
    return out


def kept(ex: Explorer) -> dict[str, list[Any]]:
    """`held`, with only the levels a mirror keeps: each metric's from its top to KEPT below."""
    out = held(ex)
    top = {(path, key): top for path, key, _, top, _, _ in out["metrics"]}
    out["levels"] = [row for row in out["levels"] if row[1] >= top[row[3], row[0]] - KEPT]
    return out


def asked(pull: object, monkeypatch: pytest.MonkeyPatch, target: str) -> list[bytes]:
    """The answers to the requests for `target` that mirror origin `pull` makes of its upstream from now on."""
    assert isinstance(pull, Pull)
    taken: list[bytes] = []
    request = pull.upstream.request

    def recorded(method: str, to: str, body: bytes | None = None) -> bytes:
        answer = request(method, to, body)
        if to == target:
            taken.append(answer)
        return answer

    monkeypatch.setattr(pull.upstream, "request", recorded)
    return taken


def blocks(ex: Explorer, key: str, paths: list[str]) -> dict[tuple[int, int], tuple[bytes, bytes]]:
    """Every block of `key` of `paths` from below their finest levels to above their top ones, and the scope's finished
    runs' blocks at those levels."""
    out: dict[tuple[int, int], tuple[bytes, bytes]] = {}
    for level in range(-2, 12):
        for i in range(max(1, (1000 >> max(level, 0)) // bk.BLOCK + 1)):
            out[(level, i)] = (ex.buckets_body(key, level, i, runs=paths), ex.buckets_body(key, level, i, "", None, "finished"))
    return out


def test_a_mirror_holds_the_runs_folders_levels_and_media_of_its_upstream(source: Source, runs: Path, tmp_path: Path) -> None:
    ex, _, url = source
    trex.folder_info(runs, note="top")
    ex.sync()
    m = synced(url, tmp_path / "cache")
    assert held(m) == held(ex) and len(held(m)["levels"]) > 10 and held(m)["folders"] != []
    assert m.runs("") == ex.runs("") and m.runs("a") == ex.runs("a") and m.tree() == ex.tree()
    paths = [p for p, _ in ex.tree()]
    for key in ("loss", "acc"):
        assert blocks(m, key, paths) == blocks(ex, key, paths)
    (media,) = ex.runs("a/r1").media
    assert m.media_path("a/r1", media.file).read_bytes() == PNG


def test_a_mirror_answers_from_its_own_index_once_the_upstream_is_gone(source: Source, tmp_path: Path) -> None:
    ex, srv, url = source
    m = synced(url, tmp_path / "cache")
    want, want_blocks = ex.runs(""), blocks(ex, "loss", ["a/r1", "b/r3"])
    stop(ex, srv)
    with pytest.raises(Unreachable):
        m.sync()
    for again in (False, True):
        if again:
            m.close()
            m = mirror_of(url, tmp_path / "cache")
        assert m.runs("") == want and blocks(m, "loss", ["a/r1", "b/r3"]) == want_blocks
        assert m.media_path("a/r1", want.media[0].file).read_bytes() == PNG
        assert m.rows_json("a/r1", 300) == '{"run":"a/r1","seq0":300,"rows":[]}'


def test_a_mirror_takes_only_the_runs_that_changed_and_of_the_levels_it_keeps_the_blocks_that_did(
        source: Source, runs: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(crawl, "REFRESH", 0.0)
    monkeypatch.setattr(mirror, "RUNNING_EVERY", 0.0)
    ex, _, url = source
    live = write_run(runs / "c" / "r4", n=6000, finish=False)
    assert wait_for(lambda: committed_rows(runs / "c" / "r4") == 6000)
    ex.sync()
    m = synced(url, tmp_path / "cache")
    taken = asked(m.origin, monkeypatch, "/api/dumps")
    assert m.sync() == [] and taken == []
    for i in range(6000, 6040):
        live.log({"loss": 1.0}, step=i)
    assert wait_for(lambda: committed_rows(runs / "c" / "r4") == 6040)
    ex.sync()
    assert m.sync() == ["c/r4"] and len(taken) == 1
    (dump,) = (Dump.decode("c/r4", body) for body in bk.unframe(taken[0]))
    mine = [(level, block, since) for _, level, block, path, since, _ in kept(ex)["levels"] if path == "c/r4"]
    assert not dump.replace and sorted((b.level, b.block) for b in dump.blocks) == sorted((lv, i) for lv, i, since in mine if since == 6040)
    assert 0 < len(dump.blocks) < len(mine) and (dump.record.compiled, dump.record.rebuilt) == (6040, 6000)
    assert held(m) == kept(ex) and m.runs("c") == ex.runs("c")
    live.finish()


def test_a_run_gone_from_the_upstream_leaves_the_mirror(source: Source, runs: Path, tmp_path: Path) -> None:
    ex, _, url = source
    m = synced(url, tmp_path / "cache")
    shutil.rmtree(runs / "a" / "r2")
    ex.sync()
    m.sync()
    assert [p for p, _ in m.tree()] == ["a/r1", "b/r3"] and held(m) == held(ex)
    with pytest.raises(KeyError):
        m.run("a/r2")


def test_a_mirror_of_a_mirror_holds_what_the_source_holds(source: Source, tmp_path: Path, http_server: Callable[[Server], str]) -> None:
    ex, _, url = source
    hub = synced(url, tmp_path / "hub-cache")
    leaf = synced(http_server(server.serve(node_of(hub), "127.0.0.1", 0)), tmp_path / "leaf-cache")
    assert held(leaf) == held(ex) and leaf.runs("") == ex.runs("")
    assert blocks(leaf, "acc", ["b/r3"]) == blocks(ex, "acc", ["b/r3"])
    assert leaf.media_path("a/r1", ex.runs("a/r1").media[0].file).read_bytes() == PNG


def test_a_running_runs_rows_reach_the_mirrors_stream_and_stay_with_it_once_the_upstream_is_gone(
        source: Source, runs: Path, tmp_path: Path) -> None:
    ex, srv, url = source
    ex.start()
    m = mirror_of(url, tmp_path / "cache").start()
    pull = m.origin
    assert isinstance(pull, Pull)
    got: list[bytes] = []
    done = threading.Event()
    threading.Thread(target=lambda: got.extend(m.messages("c", done)), daemon=True).start()
    assert wait_for(lambda: pull.connected and m.ready.is_set())
    live = write_run(runs / "c" / "r5", n=10, finish=False)
    assert wait_for(lambda: "c/r5" in m.records and m.records["c/r5"].compiled == 10)
    for i in range(10, 30):
        live.log({"loss": 0.5}, step=i)
    assert wait_for(lambda: m.live_seqs("c") == {"c/r5": (30, 0)})
    rows = json.loads(m.rows_json("c/r5", 10))
    assert (rows["seq0"], [r[0] for r in rows["rows"]]) == (10, [float(i) for i in range(10, 30)])
    assert wait_for(lambda: sum(g.count(b"event: rows") for g in got) > 0) and not any(b'"a/' in g for g in got)
    assert bk.decode(m.buckets_body("loss", 0, 0, runs=["c/r5"])).seq.tolist() == [10]
    stop(ex, srv)
    assert wait_for(lambda: not pull.connected)
    assert json.loads(m.rows_json("c/r5", 10)) == rows and m.live_seqs("c") == {"c/r5": (30, 0)}
    done.set()
    live.finish()


def test_a_running_runs_levels_follow_its_rows_once_its_tail_has_grown_or_its_keys_change(
        source: Source, runs: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(crawl, "REFRESH", 0.0)
    monkeypatch.setattr(mirror, "RUNNING_EVERY", 0.0)
    monkeypatch.setattr(mirror, "TAIL_ROWS", 20)
    ex, _, url = source
    ex.start()
    m = mirror_of(url, tmp_path / "cache").start()
    pull = m.origin
    assert isinstance(pull, Pull)
    live = write_run(runs / "c" / "r7", n=10, finish=False)
    assert wait_for(lambda: "c/r7" in m.records and m.records["c/r7"].compiled == 10 and m.live_seqs("c") == {"c/r7": (10, 0)})
    for i in range(10, 25):
        live.log({"loss": 0.5}, step=i)
    assert wait_for(lambda: m.live_seqs("c") == {"c/r7": (25, 0)} and ex.records["c/r7"].compiled == 25)
    assert m.sync() == [] and m.records["c/r7"].compiled == 10
    assert json.loads(m.rows_json("c/r7", 10))["rows"][-1][0] == 24.0
    for i in range(25, 32):
        live.log({"loss": 0.5}, step=i)
    assert wait_for(lambda: m.records["c/r7"].compiled >= 30)
    live.log({"loss": 0.5, "acc": 1.0}, step=32)
    assert wait_for(lambda: "acc" in m.records["c/r7"].keys)
    live.finish()


def test_a_running_run_whose_tail_begins_past_its_levels_is_due_them_at_once(
        source: Source, runs: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mirror, "RUNNING_EVERY", 0.0)
    ex, _, url = source
    live = write_run(runs / "c" / "r9", n=10, finish=False)
    assert wait_for(lambda: committed_rows(runs / "c" / "r9") == 10)
    ex.sync()
    m = synced(url, tmp_path / "cache")
    pull = m.origin
    assert isinstance(pull, Pull) and m.records["c/r9"].compiled == 10
    due: list[bool] = []
    for seq0 in (10, 11):  # a tail of one row: right after the levels, then past them
        pull._tails["c/r9"] = Rows("c/r9", seq0, [[float(seq0), 0.0, {"loss": 0.5}]])
        due.append(pull._due(m, "c/r9", time.monotonic()))
    assert due == [False, True]
    live.finish()


def test_a_mirror_counts_of_a_running_run_only_the_rows_it_can_serve(
        source: Source, runs: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(crawl, "REFRESH", 1000.0)
    ex, _, url = source
    live = write_run(runs / "c" / "r10", n=10, finish=False)
    assert wait_for(lambda: committed_rows(runs / "c" / "r10") == 10)
    ex.sync()
    for i in range(10, 20):
        live.log({"loss": 0.5}, step=i)
    assert wait_for(lambda: committed_rows(runs / "c" / "r10") == 20)
    ex.sync()
    assert (ex.records["c/r10"].seq, ex.records["c/r10"].compiled) == (20, 10)
    m = synced(url, tmp_path / "cache")  # a mirror not following the stream: no tail beyond the levels
    meta = m.runs("c").runs[0]
    assert (m.records["c/r10"].seq, meta.seq, json.loads(m.rows_json("c/r10", meta.compiled))["rows"]) == (20, 10, [])
    live.finish()


def test_a_run_that_finishes_upstream_is_whole_on_the_mirror_and_keeps_no_rows_apart(
        source: Source, runs: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mirror, "RUNNING_EVERY", 0.0)
    ex, _, url = source
    ex.start()
    m = mirror_of(url, tmp_path / "cache").start()
    live = write_run(runs / "c" / "r6", n=40, finish=False)
    assert wait_for(lambda: "c/r6" in m.records)
    live.finish()
    assert wait_for(lambda: m.records["c/r6"].state == "finished" and m.records["c/r6"].compiled == 40)
    assert wait_for(lambda: m.runs("c") == ex.runs("c")) and m.live_seqs("c") == {}
    assert m.rows_json("c/r6", 40) == '{"run":"c/r6","seq0":40,"rows":[]}'


def test_the_stream_brings_folder_notes_and_deletions(source: Source, runs: Path, tmp_path: Path) -> None:
    ex, _, url = source
    ex.start()
    m = mirror_of(url, tmp_path / "cache").start()
    pull = m.origin
    assert isinstance(pull, Pull)
    assert wait_for(lambda: pull.connected and len(m.records) == 3)
    trex.folder_info(runs / "b", trex={"group_by": "lr"})
    assert wait_for(lambda: m.runs("b").folders == ex.runs("b").folders == {"b": {"trex": {"group_by": "lr"}}})
    shutil.rmtree(runs / "a" / "r2")
    assert wait_for(lambda: "a/r2" not in m.records)
    m.close()
    offline = Explorer(Pull(Upstream.at("http://127.0.0.1:9"), "box:/runs"), tmp_path / "cache")
    assert offline.runs("b").folders == {"b": {"trex": {"group_by": "lr"}}} and "a/r2" not in offline.records


def test_an_upstream_that_refuses_or_is_not_http_is_said_so(source: Source) -> None:
    _, _, url = source
    with pytest.raises(ValueError, match="http://host"):
        Upstream.at("ftp://box/runs")
    with pytest.raises(Unreachable, match="400"):
        Upstream.at(url).request("POST", "/api/dumps", b"{}")
    with pytest.raises(KeyError):
        Upstream.at(url, "/d/elsewhere").request("GET", "/api/runs")


def test_requests_to_an_upstream_share_connections_and_survive_one_that_went_stale(source: Source, monkeypatch: pytest.MonkeyPatch) -> None:
    _, _, url = source
    up = Upstream.at(url)
    opened: list[http_client.HTTPConnection] = []
    connect = up.connect

    def recorded(timeout: float) -> http_client.HTTPConnection:
        opened.append(c := connect(timeout))
        return c

    monkeypatch.setattr(up, "connect", recorded)
    for _ in range(5):
        assert json.loads(up.request("GET", "/api/info"))["protocol"] == server.PROTOCOL
    assert len(opened) == 1
    opened[0].sock.close()
    assert json.loads(up.request("GET", "/api/info"))["protocol"] == server.PROTOCOL and len(opened) == 2


# ---- the levels a mirror keeps, and the blocks below them ----

DEEP = "c/r4"  # 6000 rows: levels 0 to 5, of which a mirror keeps 3 to 5


@pytest.fixture
def deep(source: Source, runs: Path) -> Source:
    """`source`, holding a run of more levels than a mirror keeps."""
    write_run(runs / "c" / "r4", n=6000)
    source[0].sync()
    return source


def fine(paths: list[str], level: int = 0, n: int = 3) -> list[Ask]:
    """Asks for the first `n` blocks of `level` of loss of runs `paths`."""
    return [Ask("loss", level, i, runs=paths) for i in range(n)]


def test_a_mirror_keeps_each_metrics_top_levels_and_fetches_the_blocks_below_them_once(
        deep: Source, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ex, _, url = deep
    m = synced(url, tmp_path / "cache")
    assert held(m) == kept(ex) and {lv for _, lv, _, path, _, _ in held(m)["levels"] if path == DEEP} == {3, 4, 5}
    assert len(held(m)["levels"]) < len(held(ex)["levels"]) // 2
    taken = asked(m.origin, monkeypatch, "/api/buckets")
    for level in (3, 4, 5, 6):
        assert m.buckets_bodies(fine([DEEP], level)) == ex.buckets_bodies(fine([DEEP], level)) and taken == []
    for paths in ([DEEP], ["a/r1", DEEP, "b/r3"]):
        for level in (-1, 0, 2):
            was = len(taken)
            assert m.buckets_bodies(fine(paths, level)) == ex.buckets_bodies(fine(paths, level))
            assert len(taken) == was + (paths == [DEEP])  # one request for all the asks, and none for blocks it has fetched
    assert m.buckets_body("loss", 0, 1, "", None, "finished") == ex.buckets_body("loss", 0, 1, "", None, "finished")
    assert held(m) == kept(ex)


def test_a_mirror_of_a_mirror_fetches_through_it(deep: Source, tmp_path: Path, http_server: Callable[[Server], str]) -> None:
    ex, _, url = deep
    hub = synced(url, tmp_path / "hub-cache")
    leaf = synced(http_server(server.serve(node_of(hub), "127.0.0.1", 0)), tmp_path / "leaf-cache")
    assert held(leaf) == kept(ex)
    assert leaf.buckets_bodies(fine([DEEP, "a/r1"])) == ex.buckets_bodies(fine([DEEP, "a/r1"]))


def test_a_mirror_whose_upstream_is_gone_answers_the_levels_it_keeps_and_the_blocks_it_fetched(
        deep: Source, tmp_path: Path, http_server: Callable[[Server], str]) -> None:
    ex, srv, url = deep
    m = synced(url, tmp_path / "cache")
    here = http_server(server.serve(node_of(m), "127.0.0.1", 0))
    want_kept, want_fine = ex.buckets_bodies(fine([DEEP], 3)), ex.buckets_bodies(fine([DEEP], 0, 2))
    assert m.buckets_bodies(fine([DEEP], 0, 2)) == want_fine
    stop(ex, srv)
    for again in (False, True):
        if again:
            m.close()
            m = mirror_of(url, tmp_path / "cache")
        assert m.buckets_bodies(fine([DEEP], 3)) == want_kept and m.buckets_bodies(fine([DEEP], 0, 2)) == want_fine
        with pytest.raises(Unfetched):
            m.buckets_bodies(fine([DEEP], 0, 3))
        with pytest.raises(Unfetched):
            m.buckets_body("loss", 1, 0, "", None, "finished")
    status, answer = post_json(here + "/api/buckets", {"blocks": [{"key": "loss", "level": 0, "index": 5, "runs": [DEEP]}]})
    assert status == 503 and "not here" in answer["error"]


def test_a_mirror_whose_stream_dropped_does_not_ask_its_upstream_for_blocks(deep: Source, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, _, url = deep
    m = synced(url, tmp_path / "cache")
    pull = m.origin
    assert isinstance(pull, Pull)
    taken = asked(pull, monkeypatch, "/api/buckets")
    pull._connected(True)
    pull._connected(False)
    with pytest.raises(Unfetched):
        m.buckets_bodies(fine([DEEP]))
    pull._connected(True)
    assert len(m.buckets_bodies(fine([DEEP]))) == 3 and len(taken) == 1


def test_a_running_runs_fetched_block_serves_until_its_levels_grow_and_is_fetched_anew_then(
        source: Source, runs: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(crawl, "REFRESH", 0.0)
    monkeypatch.setattr(mirror, "RUNNING_EVERY", 0.0)
    ex, _, url = source
    live = write_run(runs / "c" / "r4", n=6000, finish=False)
    assert wait_for(lambda: committed_rows(runs / "c" / "r4") == 6000)
    ex.sync()
    m = synced(url, tmp_path / "cache")
    last = [Ask("loss", 0, 6000 // bk.BLOCK, runs=[DEEP])]
    first = ex.buckets_bodies(last)
    taken = asked(m.origin, monkeypatch, "/api/buckets")
    assert m.buckets_bodies(last) == first and bk.decode(first[0]).seq.tolist() == [6000]
    for i in range(6000, 8400):  # past the next top level: the mirror keeps one level fewer below
        live.log({"loss": 1.0}, step=i)
    assert wait_for(lambda: committed_rows(runs / "c" / "r4") == 8400)
    ex.sync()
    assert m.buckets_bodies(last) == first and len(taken) == 1  # the rows it held when fetched: the mirror's levels hold no more
    assert m.sync() == [DEEP] and held(m) == kept(ex) and {row[1] for row in held(m)["levels"] if row[3] == DEEP} == {4, 5, 6}
    assert m.buckets_bodies(last) == ex.buckets_bodies(last) != first and len(taken) == 2
    live.finish()


def test_a_run_replaced_upstream_loses_the_blocks_fetched_of_it(deep: Source, runs: Path, tmp_path: Path) -> None:
    ex, _, url = deep
    m = synced(url, tmp_path / "cache")
    m.buckets_bodies(fine([DEEP]))
    shutil.rmtree(runs / "c" / "r4")
    write_run(runs / "c" / "r4", n=6000, metrics=lambda i: {"loss": float(i)})
    ex.sync()
    assert m.sync() == [DEEP]  # another run of the same path, as many updates old, is another version
    assert held(m) == kept(ex) and m.buckets_bodies(fine([DEEP])) == ex.buckets_bodies(fine([DEEP]))


def test_a_mirror_keeps_the_last_blocks_it_fetched(deep: Source, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(index, "FETCHED_BLOCKS", 4)
    ex, _, url = deep
    m = synced(url, tmp_path / "cache")
    for i in range(0, 12, 3):
        m.buckets_bodies([Ask("loss", 0, i + k, runs=[DEEP]) for k in range(3)])
    c = sqlite3.connect(m.db_path)
    try:
        assert sorted(b for b, in c.execute("SELECT block FROM fetched")) == [8, 9, 10, 11]
    finally:
        c.close()
    assert m.buckets_bodies(fine([DEEP])) == ex.buckets_bodies(fine([DEEP]))


def test_a_workspace_has_each_mirror_fetch_the_blocks_of_a_request_at_once(deep: Source, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ex, _, url = deep
    m = synced(url, tmp_path / "cache")
    taken = asked(m.origin, monkeypatch, "/api/buckets")
    ws = Workspace("w", [Member("far", m)])
    assert ws.buckets_bodies(fine([DEEP], 0, 5)) == ex.buckets_bodies(fine([DEEP], 0, 5)) and len(taken) == 1
