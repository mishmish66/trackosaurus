import http.client as http_client
import json
import shutil
import sqlite3
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import trex
from trex import buckets as bk, crawl, mirror, server
from trex.crawl import Crawl
from trex.index import Dump, Explorer
from trex.mirror import Pull, Unreachable, Upstream
from trex.server import Server

from helpers import PNG, committed_rows, node_of, wait_for, write_run

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


def test_a_mirror_takes_only_the_runs_that_changed_and_of_their_levels_the_blocks_that_did(
        source: Source, runs: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(crawl, "REFRESH", 0.0)
    monkeypatch.setattr(mirror, "RUNNING_EVERY", 0.0)
    ex, _, url = source
    live = write_run(runs / "c" / "r4", n=6000, finish=False)
    assert wait_for(lambda: committed_rows(runs / "c" / "r4") == 6000)
    ex.sync()
    m = synced(url, tmp_path / "cache")
    pull = m.origin
    assert isinstance(pull, Pull)
    taken: list[bytes] = []
    request = pull.upstream.request

    def recorded(method: str, target: str, body: bytes | None = None) -> bytes:
        answer = request(method, target, body)
        if target == "/api/dumps":
            taken.extend(bk.unframe(answer))
        return answer

    monkeypatch.setattr(pull.upstream, "request", recorded)
    assert m.sync() == [] and taken == []
    for i in range(6000, 6040):
        live.log({"loss": 1.0}, step=i)
    assert wait_for(lambda: committed_rows(runs / "c" / "r4") == 6040)
    ex.sync()
    assert m.sync() == ["c/r4"] and len(taken) == 1
    dump = Dump.decode("c/r4", taken[0])
    mine = [(level, block, since) for _, level, block, path, since, _ in held(ex)["levels"] if path == "c/r4"]
    assert not dump.replace and sorted((b.level, b.block) for b in dump.blocks) == sorted((lv, i) for lv, i, since in mine if since == 6040)
    assert 0 < len(dump.blocks) < len(mine) // 4 and (dump.record.compiled, dump.record.rebuilt) == (6040, 6000)
    assert held(m) == held(ex) and m.runs("c") == ex.runs("c")
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
