import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

import pytest

import trex
from trex import server
from trex.daemon import Roots
from trex.workspace import unframe_bundle, unframe_tiles

from test_remote import home  # noqa: F401  (fixture: a fake remote machine)


def wait_for(cond, timeout=20.0):
    end = time.time() + timeout
    while not cond() and time.time() < end:
        time.sleep(0.05)
    return cond()


def write_run(d, n=6, image=False):
    run = trex.init(d, config={"lr": 0.1}, commit_interval=0.02)
    for i in range(n):
        run.log({"loss": 1.0 / (i + 1)}, step=i)
    if image:
        run.log_image("img", b"\x89PNG\r\n\x1a\n" + bytes(range(16)), step=n - 1)
    run.finish()


@pytest.fixture
def dirs(tmp_path):
    """a/runs: sac/r1, sac/r2, shared/x (with an image); b/runs: sac/r3, shared/x."""
    a, b = tmp_path / "a" / "runs", tmp_path / "b" / "runs"
    for d in (a / "sac" / "r1", a / "sac" / "r2", b / "sac" / "r3", b / "shared" / "x"):
        write_run(d)
    write_run(a / "shared" / "x", image=True)
    return a, b


@pytest.fixture
def roots(tmp_path, home):  # noqa: F811
    r = Roots(tmp_path / "cache", tmp_path / "state" / "roots.json")
    yield r
    r.close()


@pytest.fixture
def http(roots):
    srv = server.serve(None, "127.0.0.1", 0, roots)
    threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def tracked(roots, *paths):
    for p in paths:
        roots.add(p)
    for name in (r["name"] for r in roots.served()):
        assert roots.get(name).ready.wait(10)
    return [r["name"] for r in roots.served()]


def get(url):
    with urllib.request.urlopen(url, timeout=30) as r:
        return r.read()


def post(url, body, raw=False):
    try:
        with urllib.request.urlopen(urllib.request.Request(url, data=json.dumps(body).encode(), method="POST"), timeout=30) as r:
            data = r.read()
            return r.status, data if raw else json.loads(data)
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_a_workspace_merges_its_members_folders_and_tells_apart_runs_at_one_path(roots, dirs, http):
    a, b = tracked(roots, *dirs)
    assert (a, b) == ("runs<a>", "runs<b>")
    assert post(f"{http}/api/daemon/workspace", {"name": "both", "members": [a, b]}) == (200, {"url": "/w/both/"})
    body = json.loads(get(f"{http}/w/both/api/runs"))
    assert sorted((r["id"], r["dir"]) for r in body["runs"]) == [
        ("sac/r1", a), ("sac/r2", a), ("sac/r3", b), ("shared/x", a), ("shared/x<runs<b>>", b)]
    assert [m[0] for m in body["media"]] == ["shared/x"]
    assert sorted(p for p, _ in json.loads(get(f"{http}/w/both/api/tree"))) == [
        "sac/r1", "sac/r2", "sac/r3", "shared/x", "shared/x<runs<b>>"]
    assert sorted(r["id"] for r in json.loads(get(f"{http}/w/both/api/runs?path=sac"))["runs"]) == ["sac/r1", "sac/r2", "sac/r3"]
    assert json.loads(get(f"{http}/w/both/api/info"))["root"] == "workspace:both"


def test_workspace_tiles_rows_and_media_come_from_the_member_holding_the_run(roots, dirs, http):
    a, b = tracked(roots, *dirs)
    roots.set_workspace("both", [a, b])
    want = [["sac/r3", "loss", "top"], ["shared/x", "loss", "top"], ["shared/x<runs<b>>", "loss", "top"], ["missing", "loss", "top"]]
    _, buf = post(f"{http}/w/both/api/tiles", want, raw=True)
    got = unframe_tiles(buf, len(want))
    direct_b = unframe_tiles(post(f"{http}/r/{urllib.parse.quote(b)}/api/tiles", [["sac/r3", "loss", "top"], ["shared/x", "loss", "top"]], raw=True)[1], 2)
    direct_a = unframe_tiles(post(f"{http}/r/{urllib.parse.quote(a)}/api/tiles", [["shared/x", "loss", "top"]], raw=True)[1], 1)
    assert got == [direct_b[0], direct_a[0], direct_b[1], []] and got[0]
    rows = json.loads(get(f"{http}/w/both/api/rows?path={urllib.parse.quote('shared/x<runs<b>>')}&from=2"))
    assert rows["run"] == "shared/x<runs<b>>" and rows["seq0"] == 2 and len(rows["rows"]) == 4
    one = json.loads(get(f"{http}/w/both/api/run?path=sac/r1"))
    assert one["run"]["id"] == "sac/r1" and one["run"]["dir"] == a
    media = json.loads(get(f"{http}/w/both/api/runs?path=shared/x"))["media"][0]
    assert get(f"{http}/w/both/m/{urllib.parse.quote('shared/x', safe='')}/{media[5]}") == b"\x89PNG\r\n\x1a\n" + bytes(range(16))


def test_a_workspace_bundle_holds_every_members_runs_under_its_scope(roots, dirs, http):
    a, b = tracked(roots, *dirs)
    roots.set_workspace("both", [a, b])
    _, buf = post(f"{http}/w/both/api/tiles/bundle", {"key": "loss", "kind": "top", "scope": "sac"}, raw=True)
    assert sorted(p for p, tiles in unframe_bundle(buf) if tiles) == ["sac/r1", "sac/r2", "sac/r3"]


def test_a_workspace_streams_live_rows_of_every_member(roots, dirs, http):
    a, b = tracked(roots, *dirs)
    roots.set_workspace("both", [a, b])
    run = trex.init(dirs[1] / "sac" / "live", commit_interval=0.05)
    run.log({"loss": 1.0}, step=0)
    time.sleep(0.3)
    got = []

    def read():
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


def test_a_remote_member_merges_like_a_local_one(roots, dirs, http, tmp_path):
    remote_runs = tmp_path / "far" / "runs"
    write_run(remote_runs / "sac" / "r9", image=True)
    tracked(roots, dirs[0])
    roots.add_remote(f"box:{remote_runs}")
    local, far = (r["name"] for r in roots.served())
    assert (local, far) == ("runs<a>", "runs<box>")
    roots.set_workspace("mixed", [local, far])

    def ids():
        return sorted((r["id"], r["dir"]) for r in json.loads(get(f"{http}/w/mixed/api/runs?path=sac"))["runs"])

    assert wait_for(lambda: ids() == [("sac/r1", local), ("sac/r2", local), ("sac/r9", far)])
    _, buf = post(f"{http}/w/mixed/api/tiles", [["sac/r9", "loss", "top"], ["sac/r1", "loss", "top"]], raw=True)
    assert all(unframe_tiles(buf, 2))
    assert ["sac/r9", "finished"] in json.loads(get(f"{http}/w/mixed/api/tree"))
    assert json.loads(get(f"{http}/w/mixed/api/run?path=sac/r9"))["run"]["dir"] == far
    rows = json.loads(get(f"{http}/w/mixed/api/rows?path=sac/r9&from=4"))
    assert rows["run"] == "sac/r9" and len(rows["rows"]) == 2
    media = json.loads(get(f"{http}/w/mixed/api/runs?path=sac/r9"))["media"][0]
    assert get(f"{http}/w/mixed/m/{urllib.parse.quote('sac/r9', safe='')}/{media[5]}") == b"\x89PNG\r\n\x1a\n" + bytes(range(16))
    _, buf = post(f"{http}/w/mixed/api/tiles/bundle", {"key": "loss", "kind": "overview", "scope": ""}, raw=True)
    assert {p for p, _ in unframe_bundle(buf)} >= {"sac/r1", "sac/r9"}


def test_a_remote_members_live_rows_reach_the_workspace_stream(roots, dirs, http, tmp_path):
    remote_runs = tmp_path / "far" / "runs"
    run = trex.init(remote_runs / "live", commit_interval=0.05)
    run.log({"loss": 1.0}, step=0)
    tracked(roots, dirs[0])
    roots.add_remote(f"box:{remote_runs}")
    roots.set_workspace("mixed", [r["name"] for r in roots.served()])
    assert wait_for(lambda: "live" in [r["id"] for r in json.loads(get(f"{http}/w/mixed/api/runs"))["runs"]])
    got = []

    def read():
        with urllib.request.urlopen(f"{http}/w/mixed/api/stream", timeout=20) as r:
            kind = None
            for line in r:
                text = line.decode().rstrip("\n")
                if text.startswith("event: "):
                    kind = text[7:]
                elif text.startswith("data: ") and kind == "rows":
                    got.append(json.loads(text[6:])["run"])
                    return

    t = threading.Thread(target=read, daemon=True)
    t.start()
    time.sleep(1.0)
    for i in range(1, 40):
        run.log({"loss": 1.0 / i}, step=i)
        time.sleep(0.05)
    t.join(15)
    run.finish()
    assert got == ["live"]


def test_an_unreachable_member_leaves_the_others_working(roots, dirs, http):
    tracked(roots, dirs[0])
    roots.add_remote("box:/no/such/runs", wait=False)
    a, far = (r["name"] for r in roots.served())
    roots.set_workspace("mixed", [a, far])
    assert wait_for(lambda: roots.get(far).state == "unreachable")
    assert sorted(r["id"] for r in json.loads(get(f"{http}/w/mixed/api/runs"))["runs"]) == ["sac/r1", "sac/r2", "shared/x"]
    _, buf = post(f"{http}/w/mixed/api/tiles", [["sac/r1", "loss", "top"]], raw=True)
    assert unframe_tiles(buf, 1)[0]


def test_workspaces_are_saved_and_lose_members_that_stop_being_tracked(roots, dirs, http, tmp_path):
    a, b = tracked(roots, *dirs)
    roots.set_workspace("both", [a, b])
    roots.set_workspace("just-a", [a])
    again = Roots(tmp_path / "cache2", tmp_path / "state" / "roots.json")
    again.load()
    try:
        assert [(w["name"], w["members"]) for w in again.workspace_list()] == [("both", [a, b]), ("just-a", [a])]
    finally:
        again.close()
    roots.remove(b)
    assert [(w["name"], w["members"]) for w in roots.workspace_list()] == [("both", ["runs"]), ("just-a", ["runs"])]
    assert post(f"{http}/api/daemon/workspace/delete", {"name": "just-a"}) == (200, {"ok": True})
    assert [w["name"] for w in json.loads(get(f"{http}/api/daemon"))["workspaces"]] == ["both"]


def test_a_workspace_can_be_renamed_and_names_are_checked(roots, dirs, http):
    a, b = tracked(roots, *dirs)
    roots.set_workspace("one", [a])
    assert post(f"{http}/api/daemon/workspace", {"name": "two", "members": [a, b], "old": "one"})[0] == 200
    assert [w["name"] for w in roots.workspace_list()] == ["two"]
    for bad in ({"name": "", "members": [a]}, {"name": "x/y", "members": [a]}, {"name": "z", "members": ["nope"]}):
        assert post(f"{http}/api/daemon/workspace", bad)[0] == 400
    roots.set_workspace("three", [b])
    assert post(f"{http}/api/daemon/workspace", {"name": "three", "members": [a], "old": "two"})[0] == 400


def test_the_daemon_root_serves_the_page(roots, dirs, http):
    tracked(roots, *dirs)
    assert b'/static/app.js' in get(f"{http}/")
    assert get(f"{http}/w/nope/") == get(f"{http}/")
