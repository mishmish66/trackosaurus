import json
import socket
import threading
import urllib.error
import urllib.request

import pytest

import trex
from trex import daemon, server
from trex.cli import main
from trex.daemon import ControlServer, Roots
from trex.index import Explorer


def write_run(d, n=5):
    run = trex.init(d, commit_interval=0.05)
    for i in range(n):
        run.log({"loss": 1.0 / (i + 1)}, step=i)
    run.finish()


@pytest.fixture
def dirs(tmp_path):
    out = []
    for parent in ("a", "b"):
        d = tmp_path / parent / "runs"
        write_run(d / "r1")
        out.append(d)
    return out


@pytest.fixture
def state(tmp_path, monkeypatch):
    d = tmp_path / "state"
    monkeypatch.setenv("TREX_DAEMON_DIR", str(d))
    return d


@pytest.fixture
def roots(tmp_path, state):
    return Roots(tmp_path / "cache", state / "roots.json")


@pytest.fixture
def http(roots):
    srv = server.serve(None, "127.0.0.1", 0, roots)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


@pytest.fixture
def control(roots, state):
    c = ControlServer(daemon.socket_path(), roots, ["http://127.0.0.1:1/"])
    threading.Thread(target=c.serve_forever, daemon=True).start()
    yield c
    c.shutdown()
    c.server_close()


def get(url):
    with urllib.request.urlopen(url) as r:
        return r.status, r.geturl(), r.read()


def post(url, body):
    with urllib.request.urlopen(urllib.request.Request(url, data=json.dumps(body).encode(), method="POST")) as r:
        return json.loads(r.read())


def ready(roots, name):
    assert roots.get(name).ready.wait(10)


def test_directories_are_named_by_basename_and_keep_their_names_across_restarts(roots, dirs, tmp_path, state):
    assert [roots.add(d) for d in dirs] == ["runs", "runs-2"]
    assert roots.add(dirs[1]) == "runs-2"
    again = Roots(tmp_path / "cache", state / "roots.json")
    again.load()
    assert [(r["name"], r["root"]) for r in again.served()] == [("runs", str(dirs[0])), ("runs-2", str(dirs[1]))]


def test_saved_directories_that_no_longer_exist_are_skipped(roots, dirs, tmp_path, state):
    roots.add(dirs[0])
    (state / "roots.json").write_text(json.dumps([{"name": "gone", "root": str(tmp_path / "gone")},
                                                  {"name": "runs", "root": str(dirs[0])}]))
    again = Roots(tmp_path / "cache", state / "roots.json")
    again.load()
    assert [r["name"] for r in again.served()] == ["runs"]


def test_daemon_serves_each_directory_under_its_prefix(roots, dirs, http):
    names = [roots.add(d) for d in dirs]
    for n in names:
        ready(roots, n)
    info = json.loads(get(f"{http}/api/daemon")[2])
    assert info["daemon"] and [r["url"] for r in info["roots"]] == ["/r/runs/", "/r/runs-2/"]
    for n, d in zip(names, dirs, strict=True):
        assert json.loads(get(f"{http}/r/{n}/api/info")[2])["root"] == str(d)
        assert [r["id"] for r in json.loads(get(f"{http}/r/{n}/api/runs")[2])["runs"]] == ["r1"]
    assert get(f"{http}/r/runs")[1] == f"{http}/r/runs/"
    assert get(f"{http}/r/nope/")[1] == f"{http}/"
    with pytest.raises(urllib.error.HTTPError) as e:
        get(f"{http}/api/runs")
    assert e.value.code == 404


def test_removing_a_directory_stops_serving_it_without_touching_its_files(roots, dirs, http, state):
    before = sorted(p.relative_to(dirs[0]) for p in dirs[0].rglob("*"))
    name = roots.add(dirs[0])
    ready(roots, name)
    assert post(f"{http}/api/daemon/remove", {"name": name}) == {"ok": True}
    with pytest.raises(urllib.error.HTTPError) as e:
        get(f"{http}/r/{name}/api/runs")
    assert e.value.code == 404
    assert json.loads((state / "roots.json").read_text()) == []
    assert sorted(p.relative_to(dirs[0]) for p in dirs[0].rglob("*")) == before


def test_standalone_server_has_no_daemon_routes(tmp_path, dirs):
    ex = Explorer(dirs[0], tmp_path / "cache")
    srv = server.serve(ex, "127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        assert json.loads(get(f"{url}/api/daemon")[2]) == {"daemon": False, "roots": [], "history": [], "install": None, "updates": None}
        with pytest.raises(urllib.error.HTTPError):
            post(f"{url}/api/daemon/remove", {"name": "runs"})
    finally:
        srv.shutdown()


def test_control_socket_adds_directories_and_reports_status(control, roots, dirs):
    reply = daemon.request({"op": "add", "path": str(dirs[0])})
    assert reply == {"name": "runs", "url": "http://127.0.0.1:1/r/runs/"}
    assert [r["root"] for r in roots.served()] == [str(dirs[0])]
    assert daemon.request({"op": "status"}) == {"urls": ["http://127.0.0.1:1/"], "roots": roots.served()}
    assert "error" in (daemon.request({"op": "add", "path": "~"}) or {})
    assert "error" in (daemon.request({"op": "add", "path": str(dirs[0] / "missing")}) or {})


def test_control_socket_is_private_to_the_user(control):
    assert control.path.stat().st_mode & 0o077 == 0


def test_second_daemon_on_the_same_socket_is_refused(control, roots):
    with pytest.raises(RuntimeError):
        ControlServer(control.path, roots, [])


def test_no_daemon_means_no_reply(state):
    assert daemon.request({"op": "status"}) is None
    state.mkdir()
    with socket.socket(socket.AF_UNIX) as s:
        s.bind(str(daemon.socket_path()))
    assert daemon.request({"op": "status"}) is None


def test_serve_adds_to_a_running_daemon_and_returns(control, roots, dirs, capsys):
    main(["serve", str(dirs[1]), "-y"])
    assert [r["root"] for r in roots.served()] == [str(dirs[1])]
    assert "http://127.0.0.1:1/r/runs/" in capsys.readouterr().out


def test_servers_without_a_port_take_the_next_free_one(monkeypatch):
    with socket.socket() as taken:
        taken.bind(("127.0.0.1", 0))
        taken.listen()
        monkeypatch.setattr(server, "DEFAULT_PORT", taken.getsockname()[1])
        (srv,) = server.bind(None, ["127.0.0.1"], None)
        try:
            assert srv.server_address[1] > taken.getsockname()[1]
        finally:
            srv.server_close()


def post_status(url, body):
    """(status, JSON body) for any status."""
    try:
        with urllib.request.urlopen(urllib.request.Request(url, data=json.dumps(body).encode(), method="POST")) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_added_and_removed_directories_are_remembered_most_recent_first_across_restarts(roots, dirs, tmp_path, state):
    a, b = (roots.add(d) for d in dirs)
    roots.remove(a)
    assert roots.history() == [str(dirs[0])]
    roots.remove(b)
    assert Roots(tmp_path / "cache", state / "roots.json").history() == [str(dirs[1]), str(dirs[0])]


def test_history_lists_only_unserved_directories_that_still_exist(roots, dirs, tmp_path):
    gone = tmp_path / "gone"
    gone.mkdir()
    for d in (*dirs, gone):
        roots.remove(roots.add(d))
    roots.add(dirs[0])
    gone.rmdir()
    assert roots.history() == [str(dirs[1])]


def test_clearing_history_keeps_served_directories(roots, dirs, tmp_path, state):
    roots.remove(roots.add(dirs[0]))
    roots.add(dirs[1])
    roots.clear_history()
    assert roots.history() == [] and [r["root"] for r in roots.served()] == [str(dirs[1])]
    roots.remove("runs")
    assert Roots(tmp_path / "cache", state / "roots.json").history() == [str(dirs[1])]


def test_http_add_serves_a_directory_and_history_can_be_cleared(roots, dirs, http):
    assert post_status(f"{http}/api/daemon/add", {"path": str(dirs[0])}) == (200, {"name": "runs", "url": "/r/runs/"})
    ready(roots, "runs")
    assert json.loads(get(f"{http}/r/runs/api/runs")[2])["runs"][0]["id"] == "r1"
    assert post(f"{http}/api/daemon/remove", {"name": "runs"}) == {"ok": True}
    assert json.loads(get(f"{http}/api/daemon")[2])["history"] == [str(dirs[0])]
    assert post(f"{http}/api/daemon/history/clear", {}) == {"ok": True}
    assert json.loads(get(f"{http}/api/daemon")[2])["history"] == []


@pytest.mark.parametrize("path,message", [
    ("runs", "not an absolute path"), ("~", "refusing to crawl"), ("/", "refusing to crawl"), ("/no/such/dir", "not a directory"),
])
def test_http_add_refuses_relative_home_root_and_missing_paths(roots, http, path, message):
    status, body = post_status(f"{http}/api/daemon/add", {"path": path})
    assert status == 400 and message in body["error"] and roots.served() == []


def test_standalone_server_refuses_daemon_changes(tmp_path, dirs):
    ex = Explorer(dirs[0], tmp_path / "cache")
    srv = server.serve(ex, "127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        for path, body in [("add", {"path": str(dirs[1])}), ("history/clear", {})]:
            assert post_status(f"{url}/api/daemon/{path}", body)[0] == 404
    finally:
        srv.shutdown()
