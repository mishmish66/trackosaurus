import shutil
import threading

import pytest

import trex
from trex import buckets as bk, index, mirror, server
from trex.index import Explorer
from trex.mirror import Mirror, Unreachable, Upstream

from helpers import PNG, committed_rows, wait_for, write_run


@pytest.fixture
def runs(tmp_path):
    d = tmp_path / "runs"
    write_run(d / "a" / "r1", n=300, image=True)
    write_run(d / "a" / "r2", n=50)
    write_run(d / "b" / "r3", n=1000, metrics=lambda i: {"loss": 1 / (i + 1), "acc": i / 1000})
    trex.folder_info(d / "a", trex={"group_by": "run~1"})
    return d


@pytest.fixture
def source(runs, tmp_path, http_server):
    """The upstream: an Explorer of `runs`, indexed, and the URL serving it."""
    ex = Explorer(runs, tmp_path / "source-cache", workers=1)
    ex.rewalk()
    ex.poll()
    srv = server.serve(ex, "127.0.0.1", 0)
    return ex, srv, http_server(srv)


def synced(url, cache, name="box:/runs"):
    m = Mirror(Upstream.at(url), cache, name)
    m.rewalk()
    m.poll()
    return m


def blocks(ex, key, paths):
    """Every block of `key` of `paths` from the finest compiled level to above the kept ones, and the scope's finished
    runs' blocks at those levels."""
    out = {}
    for level in range(-2, 12):
        for i in range(max(1, (1000 >> max(level, 0)) // bk.BLOCK + 1)):
            out[(level, i)] = (ex.buckets_body(key, level, i, runs=paths), ex.buckets_body(key, level, i, "", None, "finished"))
    return out


def test_a_mirror_holds_the_runs_folders_blocks_and_media_of_its_upstream(source, tmp_path):
    ex, _, url = source
    m = synced(url, tmp_path / "cache")
    assert m.runs("") == ex.runs("") and m.runs("a") == ex.runs("a") and m.tree() == ex.tree()
    paths = [r[0] for r in ex.tree()]
    for key in ("loss", "acc"):
        assert blocks(m, key, paths) == blocks(ex, key, paths)
    (media,) = ex.runs("a/r1")["media"]
    assert m.media_path("a/r1", media.file).read_bytes() == PNG


def test_a_mirror_answers_from_its_own_index_once_the_upstream_is_gone(source, tmp_path):
    ex, srv, url = source
    m = synced(url, tmp_path / "cache")
    want, want_blocks = ex.runs(""), blocks(ex, "loss", ["a/r1", "b/r3"])
    srv.shutdown()
    srv.server_close()
    with pytest.raises(Unreachable):
        m.rewalk()
    again = Mirror(Upstream.at(url), tmp_path / "cache", "box:/runs")
    for x in (m, again):
        assert x.runs("") == want and blocks(x, "loss", ["a/r1", "b/r3"]) == want_blocks
        assert x.media_path("a/r1", want["media"][0].file).read_bytes() == PNG
        assert x.rows_json("a/r1", 300) == '{"run":"a/r1","seq0":300,"rows":[]}'


def test_a_mirror_dumps_only_the_runs_that_changed_and_only_their_changed_parts(source, runs, tmp_path, monkeypatch):
    monkeypatch.setattr(index, "KEPT_REFRESH", 0.0)
    monkeypatch.setattr(mirror, "RUNNING_EVERY", 0.0)
    ex, _, url = source
    live = write_run(runs / "c" / "r4", n=20, finish=False)
    assert wait_for(lambda: committed_rows(runs / "c" / "r4") == 20)
    ex.rewalk()
    ex.poll()
    m = synced(url, tmp_path / "cache")
    asked: list[bytes] = []
    request = m.upstream.request
    m.upstream.request = lambda method, target, body=None: (asked.append(body or b""), request(method, target, body))[1]
    m.rewalk()
    assert m.poll() == [] and len(asked) == 1
    pyramid_seq = m.records["c/r4"]["pyramid_seq"]
    for i in range(20, 40):
        live.log({"loss": 1.0}, step=i)
    assert wait_for(lambda: committed_rows(runs / "c" / "r4") == 40)
    ex.poll()
    m.rewalk()
    assert m.poll() == ["c/r4"] and b'"path":"c/r4","uid"' in asked[-1]
    assert m.runs("c") == ex.runs("c") and m.records["c/r4"]["kept_seq"] == 40
    assert m.records["c/r4"]["pyramid_seq"] == ex.records["c/r4"]["pyramid_seq"] == pyramid_seq
    live.finish()


def test_a_run_gone_from_the_upstream_is_dropped_from_the_mirror(source, runs, tmp_path):
    ex, _, url = source
    m = synced(url, tmp_path / "cache")
    shutil.rmtree(runs / "a" / "r2")
    ex.rewalk()
    m.rewalk()
    assert [p for p, _ in m.tree()] == ["a/r1", "b/r3"] and m.runs("") == ex.runs("")
    with pytest.raises(KeyError):
        m.run("a/r2")


def test_a_mirror_of_a_mirror_holds_what_the_source_holds(source, tmp_path, http_server):
    ex, _, url = source
    hub = synced(url, tmp_path / "hub-cache")
    leaf = synced(http_server(server.serve(hub, "127.0.0.1", 0)), tmp_path / "leaf-cache")
    assert leaf.runs("") == ex.runs("")
    assert blocks(leaf, "acc", ["b/r3"]) == blocks(ex, "acc", ["b/r3"])
    assert leaf.media_path("a/r1", ex.runs("a/r1")["media"][0].file).read_bytes() == PNG


def test_a_mirror_passes_on_its_upstreams_live_rows_and_holds_the_run(source, runs, tmp_path):
    ex, _, url = source
    ex.start()
    m = Mirror(Upstream.at(url), tmp_path / "cache", "box:/runs").start()
    got: list[bytes] = []
    stop = threading.Event()

    def listen() -> None:
        for msg in m.messages("c", stop):
            got.append(msg)

    threading.Thread(target=listen, daemon=True).start()
    assert wait_for(lambda: m.connected and m.ready.is_set())
    live = write_run(runs / "c" / "r5", n=10, finish=False)
    assert wait_for(lambda: "c/r5" in m.records)
    for i in range(10, 30):
        live.log({"loss": 0.5}, step=i)
    assert wait_for(lambda: any(b"event: rows" in g and b'"run":"c/r5"' in g for g in got))
    assert not any(b'"run":"a/' in g or b'"id":"a/' in g for g in got)
    live.finish()
    assert wait_for(lambda: m.records["c/r5"]["state"] == "finished" and m.runs("c") == ex.runs("c"))
    stop.set()


def test_running_runs_blocks_come_from_the_upstream_while_it_is_reachable(source, runs, tmp_path):
    ex, srv, url = source
    live = write_run(runs / "c" / "r6", n=10, finish=False)
    assert wait_for(lambda: committed_rows(runs / "c" / "r6") == 10)
    ex.rewalk()
    ex.poll()
    m = Mirror(Upstream.at(url), tmp_path / "cache", "box:/runs").start()
    assert wait_for(lambda: m.connected and "c/r6" in m.records)
    for i in range(10, 600):
        live.log({"loss": 0.5}, step=i)
    assert wait_for(lambda: committed_rows(runs / "c" / "r6") == 600)
    ex.poll()
    asks = [index.Ask("loss", level, 0, "", ["c/r6"]) for level in (0, 3)]
    assert m.buckets_bodies(asks) == ex.buckets_bodies(asks) != [ex.buckets_body("loss", lv, 0, runs=[]) for lv in (0, 3)]
    srv.shutdown()
    srv.server_close()
    ex.close()  # ends the stream the server was sending
    assert wait_for(lambda: not m.connected)
    assert [(a.paths, a.seq.tolist()) for a in map(bk.decode, m.buckets_bodies(asks))] == [(["c/r6"], [10])] * 2
    live.finish()


def test_the_stream_brings_folder_notes_deletions_and_the_upstreams_counts(source, runs, tmp_path, monkeypatch):
    monkeypatch.setattr(index, "HEARTBEAT", 0.3)
    ex, _, url = source
    ex.start()
    m = Mirror(Upstream.at(url), tmp_path / "cache", "box:/runs").start()
    assert wait_for(lambda: m.connected and len(m.records) == 3)
    trex.folder_info(runs / "b", trex={"group_by": "lr"})
    assert wait_for(lambda: m.runs("b")["folders"] == ex.runs("b")["folders"] == {"b": {"trex": {"group_by": "lr"}}})
    live = write_run(runs / "c" / "r7", n=10, finish=False)
    assert wait_for(lambda: m.live_seqs("c") == {"c/r7": (10, 0)})
    shutil.rmtree(runs / "a" / "r2")
    assert wait_for(lambda: "a/r2" not in m.records)
    live.finish()
    m.close()
    offline = Mirror(Upstream.at("http://127.0.0.1:9"), tmp_path / "cache", "box:/runs")
    assert offline.runs("b")["folders"] == {"b": {"trex": {"group_by": "lr"}}} and "a/r2" not in offline.records


def test_blocks_a_mirrors_compiled_levels_lack_come_from_the_upstream(source, runs, tmp_path, monkeypatch):
    monkeypatch.setattr(index, "KEPT_REFRESH", 0.0)
    ex, srv, url = source
    live = write_run(runs / "c" / "r8", n=10, finish=False)
    assert wait_for(lambda: committed_rows(runs / "c" / "r8") == 10)
    ex.rewalk()
    ex.poll()
    for i in range(10, 600):
        live.log({"loss": 0.5}, step=i)
    assert wait_for(lambda: committed_rows(runs / "c" / "r8") == 600)
    ex.poll()
    assert ex.records["c/r8"]["pyramid_seq"] == 10 and ex.records["c/r8"]["seq"] == 600
    m = synced(url, tmp_path / "cache")
    asks = [index.Ask("loss", level, 0, "c", None, "running") for level in (0, 1)]
    assert m.buckets_bodies(asks) == ex.buckets_bodies(asks)
    srv.shutdown()
    srv.server_close()
    assert [(a.paths, a.seq.tolist(), a.buckets.run.size) for a in map(bk.decode, m.buckets_bodies(asks))] == [(["c/r8"], [0], 0)] * 2
    live.finish()


def test_an_upstream_that_refuses_or_is_not_http_is_said_so(source):
    _, _, url = source
    with pytest.raises(ValueError, match="http://host"):
        Upstream.at("ftp://box/runs")
    with pytest.raises(Unreachable, match="400"):
        Upstream.at(url).request("POST", "/api/dumps", b"{}")
    with pytest.raises(KeyError):
        Upstream.at(url, "/d/elsewhere").request("GET", "/api/runs")
