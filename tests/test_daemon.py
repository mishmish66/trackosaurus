import http.client as http_client
import json
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

import pytest

import trex
from trex import daemon, server
from trex.cli import main
from trex.daemon import ControlServer, Roots, unique_names
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
    threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


@pytest.fixture
def control(roots, state):
    c = ControlServer(daemon.socket_path(), roots, ["http://127.0.0.1:1/"])
    threading.Thread(target=c.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
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


def test_colliding_names_are_told_apart_by_their_parents_until_the_collision_ends(roots, dirs, tmp_path, state):
    assert roots.add(dirs[0]) == "runs"
    assert roots.add(dirs[1]) == "runs<b>" == roots.add(dirs[1])
    assert [(r["name"], r["root"]) for r in roots.served()] == [("runs<a>", str(dirs[0])), ("runs<b>", str(dirs[1]))]
    again = Roots(tmp_path / "cache", state / "roots.json")
    again.load()
    assert [r["name"] for r in again.served()] == ["runs<a>", "runs<b>"]
    roots.remove("runs<a>")
    assert [r["name"] for r in roots.served()] == ["runs"]


@pytest.mark.parametrize("specs,want", [
    (["/x/runs", "/y/runs"], ["runs<x>", "runs<y>"]),
    (["/x/a/runs", "/y/a/runs"], ["runs<x/a>", "runs<y/a>"]),
    (["/data/runs", "gpu-box:~/runs", "big.lan:/scratch/runs"], ["runs<data>", "runs<gpu-box>", "runs<big>"]),
    (["/data/runs", "/data/sweeps"], ["runs", "sweeps"]),
])
def test_names_follow_emacs_uniquify(specs, want):
    names = unique_names(specs)
    assert [names[s] for s in specs] == want


def test_saved_directories_that_no_longer_exist_are_skipped(roots, dirs, tmp_path, state):
    roots.add(dirs[0])
    (state / "roots.json").write_text(json.dumps([{"name": "gone", "root": str(tmp_path / "gone")},
                                                  {"name": "runs", "root": str(dirs[0])}]))
    again = Roots(tmp_path / "cache", state / "roots.json")
    again.load()
    assert [r["name"] for r in again.served()] == ["runs"]


def test_daemon_serves_each_directory_under_its_prefix(roots, dirs, http):
    for d in dirs:
        roots.add(d)
    names = [r["name"] for r in roots.served()]
    for n in names:
        ready(roots, n)
    info = json.loads(get(f"{http}/api/daemon")[2])
    assert info["daemon"] and [r["url"] for r in info["roots"]] == ["/r/runs%3Ca%3E/", "/r/runs%3Cb%3E/"]
    for n, d in zip(names, dirs, strict=True):
        assert json.loads(get(f"{http}/r/{urllib.parse.quote(n)}/api/info")[2])["root"] == str(d)
        assert [r["id"] for r in json.loads(get(f"{http}/r/{urllib.parse.quote(n)}/api/runs")[2])["runs"]] == ["r1"]
    assert get(f"{http}/r/runs%3Ca%3E")[1] == f"{http}/r/runs%3Ca%3E/"
    assert get(f"{http}/r/nope/")[1] == f"{http}/"
    assert sorted(r["id"] for r in json.loads(get(f"{http}/api/runs")[2])["runs"]) == ["runs<a>/r1", "runs<b>/r1"]


def test_removing_a_directory_stops_serving_it_without_touching_its_files(roots, dirs, http, state):
    before = sorted(p.relative_to(dirs[0]) for p in dirs[0].rglob("*"))
    name = roots.add(dirs[0])
    ready(roots, name)
    assert post(f"{http}/api/daemon/remove", {"name": name}) == {"ok": True}
    with pytest.raises(urllib.error.HTTPError) as e:
        get(f"{http}/r/{name}/api/runs")
    assert e.value.code == 404
    assert json.loads((state / "roots.json").read_text()) == {"tracked": [], "workspaces": []}
    assert sorted(p.relative_to(dirs[0]) for p in dirs[0].rglob("*")) == before


def test_standalone_server_has_no_daemon_routes(tmp_path, dirs):
    ex = Explorer(dirs[0], tmp_path / "cache")
    srv = server.serve(ex, "127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        assert json.loads(get(f"{url}/api/daemon")[2]) == {"daemon": False, "roots": [], "workspaces": [], "history": [], "install": None, "updates": None}
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
    for d in dirs:
        roots.add(d)
    roots.remove("runs<a>")
    assert roots.history() == [str(dirs[0])]
    roots.remove("runs")
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
    threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        for path, body in [("add", {"path": str(dirs[1])}), ("history/clear", {})]:
            assert post_status(f"{url}/api/daemon/{path}", body)[0] == 404
    finally:
        srv.shutdown()


def raw(url, method="GET", headers=None, body=None):
    """(status, JSON body) of a request with exactly these headers."""
    parts = urllib.parse.urlsplit(url)
    conn = http_client.HTTPConnection(parts.hostname, parts.port)
    try:
        conn.putrequest(method, parts.path, skip_host=True)
        for k, v in (headers or {}).items():
            conn.putheader(k, v)
        conn.putheader("Content-Length", str(len(body or b"")))
        conn.endheaders(body)
        r = conn.getresponse()
        return r.status, json.loads(r.read() or b"null")
    finally:
        conn.close()


@pytest.mark.parametrize("host", ["evil.example.com", "evil.example.com:80", "localhost.evil.example.com"])
def test_requests_addressed_to_unknown_host_names_are_refused(http, host):
    status, body = raw(f"{http}/api/daemon", headers={"Host": host})
    assert status == 403 and "--allow-host" in body["error"]


@pytest.mark.parametrize("host", ["127.0.0.1:{port}", "[::1]:{port}", "localhost:{port}", "app.localhost:{port}",
                                  "{hostname}:{port}", "100.64.0.9:{port}"])
def test_requests_by_address_localhost_or_machine_name_are_served(http, host):
    port = urllib.parse.urlsplit(http).port
    status, body = raw(f"{http}/api/daemon", headers={"Host": host.format(port=port, hostname=socket.gethostname())})
    assert status == 200 and body["daemon"]


def test_allowed_host_names_are_served(roots):
    srv = server.serve(None, "127.0.0.1", 0, roots, allow=["Box.Tailnet.ts.net"])
    threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
    try:
        assert raw(f"http://127.0.0.1:{srv.server_address[1]}/api/daemon", headers={"Host": "box.tailnet.ts.net"})[0] == 200
    finally:
        srv.shutdown()


@pytest.mark.parametrize("origin", ["https://evil.example.com", "http://127.0.0.1:1", "null"])
def test_changes_posted_from_another_origin_are_refused(roots, dirs, http, origin):
    host = urllib.parse.urlsplit(http).netloc
    status, body = raw(f"{http}/api/daemon/add", "POST", {"Host": host, "Origin": origin}, json.dumps({"path": str(dirs[0])}).encode())
    assert status == 403 and roots.served() == []


def test_changes_from_the_ui_itself_or_without_an_origin_are_allowed(roots, dirs, http):
    host = urllib.parse.urlsplit(http).netloc
    body = json.dumps({"path": str(dirs[0])}).encode()
    assert raw(f"{http}/api/daemon/add", "POST", {"Host": host, "Origin": f"http://{host}"}, body)[0] == 200
    assert raw(f"{http}/api/daemon/add", "POST", {"Host": host}, json.dumps({"path": str(dirs[1])}).encode())[0] == 200
    assert len(roots.served()) == 2


def test_daemon_directories_share_one_tile_budget(roots, dirs):
    a, b = (roots.get(roots.add(d)) for d in dirs)
    assert a.budget is b.budget is roots.budget and roots.budget.members == [a, b]
    roots.remove("runs<a>")
    assert wait_closed(a) and roots.budget.members == [b]


def wait_closed(ex, timeout=10.0):
    end = time.time() + timeout
    while not ex._closed and time.time() < end:
        time.sleep(0.02)
    return ex._closed
