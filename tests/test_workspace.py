import functools
import json
import threading
import time
import urllib.parse
import urllib.request
from collections.abc import Callable, Generator, Iterator
from pathlib import Path
from typing import cast

import pytest

import trex
from trex import buckets as bk, server
from trex.format import RunState
from trex.node import Node, dir_base
from trex.index import Explorer
from trex.server import Server
from trex.workspace import Member, Workspace

import helpers
from helpers import PNG, committed_rows, get_json, post_bytes, post_json, request, wait_for

write_run = functools.partial(helpers.write_run, n=6, config={"lr": 0.1}, commit_interval=0.02)


@pytest.fixture
def dirs(tmp_path: Path) -> tuple[Path, Path]:
    """a/runs: sac/r1, sac/r2, shared/x (with an image); b/runs: sac/r3, shared/x."""
    a, b = tmp_path / "a" / "runs", tmp_path / "b" / "runs"
    for d in (a / "sac" / "r1", a / "sac" / "r2", b / "sac" / "r3", b / "shared" / "x"):
        write_run(d)
    write_run(a / "shared" / "x", image=True)
    return a, b


@pytest.fixture
def node(tmp_path: Path, home: Path) -> Iterator[Node]:
    n = Node(tmp_path / "cache", tmp_path / "state" / "roots.json")
    yield n
    n.close()


@pytest.fixture
def http(node: Node, http_server: Callable[[Server], str]) -> str:
    return http_server(server.serve(node, "127.0.0.1", 0))


def blocks(base: str, **body: str | list[str]) -> bk.BucketArray:
    """The bucket array answering a request for every step of `loss` at base/api/buckets."""
    return bk.decode(bk.unframe(post_bytes(f"{base}/api/buckets", {"blocks": [{"key": "loss", "level": 20, "index": 0, **body}]}))[0])


def runs_with_buckets(a: bk.BucketArray) -> list[str]:
    return [p for i, p in enumerate(a.paths) if (a.buckets.run == i).any()]


def tracked(node: Node, *paths: Path) -> list[str]:
    for p in paths:
        assert node.by_id(node.add(p)).ready.wait(10)
    return [d.name for d in node.served()]


def test_a_workspace_merges_its_members_folders_and_tells_apart_runs_at_one_path(node: Node, dirs: tuple[Path, Path], http: str) -> None:
    a, b = tracked(node, *dirs)
    assert (a, b) == ("runs<a>", "runs<b>")
    assert post_json(f"{http}/api/node/workspace", {"name": "both", "members": [a, b]}) == (200, {"url": "/w/both/"})
    body = get_json(f"{http}/w/both/api/runs")
    assert sorted((r["id"], r["dir"]) for r in body["runs"]) == [
        ("sac/r1", a), ("sac/r2", a), ("sac/r3", b), ("shared/x", a), ("shared/x<runs<b>>", b)]
    assert [m[0] for m in body["media"]] == ["shared/x"]
    assert sorted(p for p, _ in get_json(f"{http}/w/both/api/tree")) == [
        "sac/r1", "sac/r2", "sac/r3", "shared/x", "shared/x<runs<b>>"]
    assert sorted(r["id"] for r in get_json(f"{http}/w/both/api/runs?path=sac")["runs"]) == ["sac/r1", "sac/r2", "sac/r3"]
    assert get_json(f"{http}/w/both/api/info")["root"] == "workspace:both"


def test_workspace_buckets_rows_and_media_come_from_the_member_holding_the_run(node: Node, dirs: tuple[Path, Path], http: str) -> None:
    a, b = tracked(node, *dirs)
    node.set_workspace("both", [a, b])
    got = blocks(f"{http}/w/both", runs=["sac/r3", "shared/x", "shared/x<runs<b>>", "missing"])
    direct_b = blocks(f"{http}{dir_base(node.id_of(b))}", runs=["sac/r3", "shared/x"])
    direct_a = blocks(f"{http}{dir_base(node.id_of(a))}", runs=["shared/x"])
    assert got.paths == ["shared/x", "sac/r3", "shared/x<runs<b>>"] and runs_with_buckets(got) == got.paths  # each member's in turn
    for path, direct, there in (("sac/r3", direct_b, "sac/r3"), ("shared/x", direct_a, "shared/x"), ("shared/x<runs<b>>", direct_b, "shared/x")):
        mine, theirs = got.buckets.run == got.paths.index(path), direct.buckets.run == direct.paths.index(there)
        assert all((x[mine] == y[theirs]).all() for x, y in zip(got.buckets.columns[1:], direct.buckets.columns[1:]))
        assert got.seq[got.paths.index(path)] == direct.seq[direct.paths.index(there)]
    rows = get_json(f"{http}/w/both/api/rows?path={urllib.parse.quote('shared/x<runs<b>>')}&from=2")
    assert rows["run"] == "shared/x<runs<b>>" and rows["seq0"] == 2 and len(rows["rows"]) == 4
    one = get_json(f"{http}/w/both/api/run?path=sac/r1")
    assert one["run"]["id"] == "sac/r1" and one["run"]["dir"] == a
    media = get_json(f"{http}/w/both/api/runs?path=shared/x")["media"][0]
    assert request(f"{http}/w/both/m/{urllib.parse.quote('shared/x', safe='')}/{media[5]}")[2] == PNG


def test_a_workspace_block_holds_every_members_runs_under_its_scope(node: Node, dirs: tuple[Path, Path], http: str) -> None:
    a, b = tracked(node, *dirs)
    node.set_workspace("both", [a, b])
    got = blocks(f"{http}/w/both", scope="sac")
    assert got.paths == ["sac/r1", "sac/r2", "sac/r3"] and runs_with_buckets(got) == got.paths


def test_a_workspace_block_of_a_scope_without_runs_names_none(node: Node, dirs: tuple[Path, Path], http: str) -> None:
    a, b = tracked(node, *dirs)
    node.set_workspace("both", [a, b])
    got = blocks(f"{http}/w/both", scope="nowhere")
    assert got.paths == [] and got.buckets.run.size == 0


def test_a_workspace_answers_a_batch_as_each_block_alone(node: Node, dirs: tuple[Path, Path], http: str) -> None:
    a, b = tracked(node, *dirs)
    node.set_workspace("both", [a, b])
    asks = [{"key": "loss", "level": 20, "index": 0, "scope": "sac"}, {"key": "loss", "level": 3, "index": 0, "runs": ["sac/r3", "shared/x"]},
            {"key": "loss", "level": 20, "index": 0, "scope": "shared", "which": "finished"}]
    alone = [post_bytes(f"{http}/w/both/api/buckets", {"blocks": [ask]}) for ask in asks]
    together = bk.unframe(post_bytes(f"{http}/w/both/api/buckets", {"blocks": asks}))
    assert together == [bk.unframe(x)[0] for x in alone]


def test_a_workspace_streams_live_rows_of_every_member(node: Node, dirs: tuple[Path, Path], http: str) -> None:
    a, b = tracked(node, *dirs)
    node.set_workspace("both", [a, b])
    run = trex.init(dirs[1] / "sac" / "live", commit_interval=0.05)
    run.log({"loss": 1.0}, step=0)
    assert wait_for(lambda: committed_rows(dirs[1] / "sac" / "live") == 1)
    got: list[tuple[str, str, str | None]] = []

    def read() -> None:
        with urllib.request.urlopen(f"{http}/w/both/api/stream?path=sac", timeout=20) as r:
            kind = None
            for line in r:
                text = line.decode().rstrip("\n")
                if text.startswith("event: "):
                    kind = text[7:]
                elif text.startswith("data: ") and kind in ("rows", "run"):
                    ev = json.loads(text[6:])
                    got.append((kind, ev.get("run") or ev.get("id"), ev.get("dir")))
                    if kind == "rows":
                        return

    t = threading.Thread(target=read, daemon=True)
    t.start()
    end = time.time() + 15
    while not got and time.time() < end:
        time.sleep(0.05)
    assert got[:1] == [("run", "sac/live", b)]
    for i in range(1, 30):
        run.log({"loss": 1.0 / i}, step=i)
        time.sleep(0.05)
    t.join(15)
    run.finish()
    assert ("rows", "sac/live", None) in got


def test_a_remote_member_merges_like_a_local_one(node: Node, dirs: tuple[Path, Path], http: str, tmp_path: Path) -> None:
    remote_runs = tmp_path / "far" / "runs"
    write_run(remote_runs / "sac" / "r9", image=True)
    tracked(node, dirs[0])
    node.add_remote(f"box:{remote_runs}")
    local, far = (d.name for d in node.served())
    assert (local, far) == ("runs<a>", "runs<box>")
    node.set_workspace("mixed", [local, far])

    def ids() -> list[tuple[str, str]]:
        return sorted((r["id"], r["dir"]) for r in get_json(f"{http}/w/mixed/api/runs?path=sac")["runs"])

    assert wait_for(lambda: ids() == [("sac/r1", local), ("sac/r2", local), ("sac/r9", far)])
    assert runs_with_buckets(blocks(f"{http}/w/mixed", runs=["sac/r9", "sac/r1"])) == ["sac/r1", "sac/r9"]
    assert ["sac/r9", "finished"] in get_json(f"{http}/w/mixed/api/tree")
    assert get_json(f"{http}/w/mixed/api/run?path=sac/r9")["run"]["dir"] == far
    rows = get_json(f"{http}/w/mixed/api/rows?path=sac/r9&from=4")
    assert rows == {"run": "sac/r9", "seq0": 4, "rows": []}  # a mirrored directory's finished runs are whole in their levels
    media = get_json(f"{http}/w/mixed/api/runs?path=sac/r9")["media"][0]
    assert request(f"{http}/w/mixed/m/{urllib.parse.quote('sac/r9', safe='')}/{media[5]}")[2] == PNG
    assert set(blocks(f"{http}/w/mixed", scope="", which="finished").paths) >= {"sac/r1", "sac/r9"}


def test_a_remote_members_live_rows_reach_the_workspace_stream(node: Node, dirs: tuple[Path, Path], http: str, tmp_path: Path) -> None:
    remote_runs = tmp_path / "far" / "runs"
    run = trex.init(remote_runs / "live", commit_interval=0.05)
    run.log({"loss": 1.0}, step=0)
    tracked(node, dirs[0])
    node.add_remote(f"box:{remote_runs}")
    node.set_workspace("mixed", [d.name for d in node.served()])
    assert wait_for(lambda: "live" in [r["id"] for r in get_json(f"{http}/w/mixed/api/runs")["runs"]])
    got: list[str] = []
    opened = threading.Event()

    def read() -> None:
        with urllib.request.urlopen(f"{http}/w/mixed/api/stream", timeout=20) as r:
            kind = None
            for line in r:
                text = line.decode().rstrip("\n")
                if text.startswith("retry:"):
                    opened.set()
                elif text.startswith("event: "):
                    kind = text[7:]
                elif text.startswith("data: ") and kind == "rows":
                    got.append(json.loads(text[6:])["run"])
                    return

    t = threading.Thread(target=read, daemon=True)
    t.start()
    assert opened.wait(15)
    for i in range(1, 40):
        run.log({"loss": 1.0 / i}, step=i)
        time.sleep(0.05)
    t.join(15)
    run.finish()
    assert got == ["live"]


def test_an_unreachable_member_leaves_the_others_working(node: Node, dirs: tuple[Path, Path], http: str) -> None:
    tracked(node, dirs[0])
    node.add_remote("box:/no/such/runs", wait=False)
    a, far = (d.name for d in node.served())
    node.set_workspace("mixed", [a, far])
    assert wait_for(lambda: [d.state for d in node.served() if d.name == far] == ["unreachable"])
    assert sorted(r["id"] for r in get_json(f"{http}/w/mixed/api/runs")["runs"]) == ["sac/r1", "sac/r2", "shared/x"]
    assert runs_with_buckets(blocks(f"{http}/w/mixed", runs=["sac/r1"])) == ["sac/r1"]


def test_workspaces_are_saved_and_lose_members_that_stop_being_tracked(node: Node, dirs: tuple[Path, Path], http: str, tmp_path: Path) -> None:
    a, b = tracked(node, *dirs)
    node.set_workspace("both", [a, b])
    node.set_workspace("just-a", [a])
    again = Node(tmp_path / "cache2", tmp_path / "state" / "roots.json")
    again.load()
    try:
        assert [(w["name"], w["members"]) for w in again.workspace_list()] == [("both", [a, b]), ("just-a", [a])]
    finally:
        again.close()
    node.remove(b)
    assert [(w["name"], w["members"]) for w in node.workspace_list()] == [("both", ["runs"]), ("just-a", ["runs"])]
    assert post_json(f"{http}/api/node/workspace/delete", {"name": "just-a"}) == (200, {"ok": True})
    assert [w["name"] for w in get_json(f"{http}/api/node")["workspaces"]] == ["both"]


def test_a_workspace_can_be_renamed_and_names_are_checked(node: Node, dirs: tuple[Path, Path], http: str) -> None:
    a, b = tracked(node, *dirs)
    node.set_workspace("one", [a])
    assert post_json(f"{http}/api/node/workspace", {"name": "two", "members": [a, b], "old": "one"})[0] == 200
    assert [w["name"] for w in node.workspace_list()] == ["two"]
    for bad in ({"name": "", "members": [a]}, {"name": "x/y", "members": [a]}, {"name": "z", "members": ["nope"]}):
        assert post_json(f"{http}/api/node/workspace", bad)[0] == 400
    node.set_workspace("three", [b])
    assert post_json(f"{http}/api/node/workspace", {"name": "three", "members": [a], "old": "two"})[0] == 400


def test_a_nodes_root_serves_the_page(node: Node, dirs: tuple[Path, Path], http: str) -> None:
    tracked(node, *dirs)
    assert b'/static/app.js' in request(f"{http}/")[2]
    assert request(f"{http}/w/nope/")[2] == request(f"{http}/")[2]


def test_a_nodes_root_shows_every_directory_as_a_top_level_folder(node: Node, tmp_path: Path, http: str) -> None:
    a, b = tmp_path / "x" / "a" / "runs", tmp_path / "y" / "a" / "runs"
    write_run(a / "sac" / "r1", image=True)
    write_run(b / "sac" / "r1")
    trex.folder_info(a / "sac", note="from x")
    names = tracked(node, a, b)
    assert names == ["runs<x/a>", "runs<y/a>"]
    body = get_json(f"{http}/api/runs")
    assert sorted((r["id"], r["dir"]) for r in body["runs"]) == [("runs<x/a>/sac/r1", "runs<x/a>"), ("runs<y/a>/sac/r1", "runs<y/a>")]
    assert body["folders"]["runs<x/a>/sac"] == {"note": "from x"}
    assert [r["id"] for r in get_json(f"{http}/api/runs?path={urllib.parse.quote('runs<y/a>')}")["runs"]] == ["runs<y/a>/sac/r1"]
    assert get_json(f"{http}/api/runs?path=elsewhere")["runs"] == []
    assert runs_with_buckets(blocks(http, runs=["runs<y/a>/sac/r1", "runs<x/a>/sac/r1"])) == ["runs<x/a>/sac/r1", "runs<y/a>/sac/r1"]
    media = body["media"][0]
    assert request(f"{http}/m/{urllib.parse.quote(media[0], safe='')}/{media[5]}")[2] == PNG
    assert get_json(f"{http}/api/info")["root"] == "daemon:/"


def test_an_empty_nodes_root_has_no_runs(node: Node, http: str) -> None:
    assert get_json(f"{http}/api/runs") == {"runs": [], "media": [], "folders": {}}


class Endless:
    """A directory whose stream sends heartbeats as fast as they are taken; `ended` once the stream is closed."""

    def __init__(self) -> None:
        self.sent = 0
        self.ended = threading.Event()

    def tree(self) -> list[tuple[str, RunState]]:
        return []

    def messages(self, prefix: str, stop: threading.Event) -> Generator[bytes, None, None]:
        try:
            while True:
                self.sent += 1
                yield b"event: hb\ndata: {}\n\n"
        finally:
            self.ended.set()


def test_a_workspace_stream_ends_its_member_pumps_when_the_client_leaves() -> None:
    fake, stop = Endless(), threading.Event()
    ws = Workspace("w", [Member("m", cast(Explorer, fake))])
    gen = ws.messages("", stop)
    next(gen)
    assert wait_for(lambda: fake.sent > 20_001)
    stop.set()
    gen.close()
    assert fake.ended.wait(5)
