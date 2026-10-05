import http.client as http_client
import json
import socket
import threading
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from trex import control, server
from trex.cli import main
from trex.control import ControlServer
from trex.node import Node, dir_base, unique_names
from trex.server import Server

from helpers import get_json, post_json, request, wait_for, write_run


@pytest.fixture
def dirs(tmp_path: Path) -> list[Path]:
    out: list[Path] = []
    for parent in ("a", "b"):
        d = tmp_path / parent / "runs"
        write_run(d / "r1")
        out.append(d)
    return out


@pytest.fixture
def state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    d = tmp_path / "state"
    monkeypatch.setenv("TREX_DAEMON_DIR", str(d))
    return d


@pytest.fixture
def node(tmp_path: Path, state: Path) -> Iterator[Node]:
    n = Node(tmp_path / "cache", state / "roots.json")
    yield n
    n.close()


@pytest.fixture
def http(node: Node, http_server: Callable[[Server], str]) -> str:
    return http_server(server.serve(node, "127.0.0.1", 0))


@pytest.fixture
def ctl(node: Node, state: Path) -> Iterator[ControlServer]:
    c = ControlServer(control.socket_path(), node, ["http://127.0.0.1:1/"])
    threading.Thread(target=c.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
    yield c
    c.shutdown()
    c.server_close()


def final_url(url: str) -> str:
    """The URL a GET ends at after redirects."""
    with urllib.request.urlopen(url) as r:
        return r.geturl()


def added(node: Node, d: Path) -> str:
    """The name of directory `d` once the node crawls it, after its first pass."""
    x = node.add(d)
    assert node.by_id(x).ready.wait(10)
    return node.names()[x]


def served(node: Node) -> list[tuple[str, str]]:
    return [(d.name, d.root) for d in node.served()]


def test_colliding_names_are_told_apart_by_their_parents_until_the_collision_ends(node: Node, dirs: list[Path], tmp_path: Path, state: Path) -> None:
    assert added(node, dirs[0]) == "runs"
    assert added(node, dirs[1]) == "runs<b>" == node.names()[node.add(dirs[1])]
    assert served(node) == [("runs<a>", str(dirs[0])), ("runs<b>", str(dirs[1]))]
    again = Node(tmp_path / "cache2", state / "roots.json")
    again.load()
    assert [d.name for d in again.served()] == ["runs<a>", "runs<b>"]
    again.close()
    node.remove("runs<a>")
    assert [d.name for d in node.served()] == ["runs"]


@pytest.mark.parametrize("specs,want", [
    (["/x/runs", "/y/runs"], ["runs<x>", "runs<y>"]),
    (["/x/a/runs", "/y/a/runs"], ["runs<x/a>", "runs<y/a>"]),
    (["/data/runs", "gpu-box:~/runs", "big.lan:/scratch/runs"], ["runs<data>", "runs<gpu-box>", "runs<big>"]),
    (["/data/runs", "/data/sweeps"], ["runs", "sweeps"]),
    (["box:/x/runs", "box:/y/runs", "/z/runs"], ["runs<x/box>", "runs<y/box>", "runs<z>"]),
    (["box:/x/runs", "box:/y/runs"], ["runs<x>", "runs<y>"]),
])
def test_names_follow_emacs_uniquify(specs: list[str], want: list[str]) -> None:
    names = unique_names(specs)
    assert [names[s] for s in specs] == want


def test_saved_directories_that_no_longer_exist_are_skipped(node: Node, dirs: list[Path], tmp_path: Path, state: Path) -> None:
    node.add(dirs[0])
    (state / "roots.json").write_text(json.dumps({"tracked": [str(tmp_path / "gone"), str(dirs[0])], "workspaces": []}))
    again = Node(tmp_path / "cache2", state / "roots.json")
    again.load()
    assert [d.name for d in again.served()] == ["runs"]
    again.close()


def test_a_node_serves_each_directory_under_its_id_and_every_one_at_its_root(node: Node, dirs: list[Path], http: str) -> None:
    for d in dirs:
        added(node, d)
    info = get_json(f"{http}/api/node")
    ids = [f"{node.identity.name}:{d}" for d in dirs]
    names = ["runs<a>", "runs<b>"]
    assert info["saves"] and info["home"] is None and info["node"] == {"id": node.identity.id, "name": node.identity.name}
    assert [(r["name"], r["id"], r["url"]) for r in info["dirs"]] == [(n, x, dir_base(x) + "/") for n, x in zip(names, ids, strict=True)]
    for x, d in zip(ids, dirs, strict=True):
        assert get_json(f"{http}{dir_base(x)}/api/info")["root"] == str(d)
        assert [r["id"] for r in get_json(f"{http}{dir_base(x)}/api/runs")["runs"]] == ["r1"]
    assert final_url(f"{http}{dir_base(ids[0])}") == f"{http}{dir_base(ids[0])}/"
    assert final_url(f"{http}/d/nope/") == f"{http}/"
    assert sorted(r["id"] for r in get_json(f"{http}/api/runs")["runs"]) == ["runs<a>/r1", "runs<b>/r1"]


def test_a_temporary_node_saves_nothing_and_shows_its_home_directory_at_its_root(tmp_path: Path, dirs: list[Path],
                                                                                http_server: Callable[[Server], str]) -> None:
    node = Node(tmp_path / "cache")
    node.home = node.add(dirs[0])
    url = http_server(server.serve(node, "127.0.0.1", 0))
    assert wait_for(lambda: [r["id"] for r in get_json(f"{url}/api/runs")["runs"]] == ["r1"])
    assert get_json(f"{url}/api/info")["root"] == str(dirs[0])
    info = get_json(f"{url}/api/node")
    assert not info["saves"] and info["home"] == node.home and info["history"] == []
    assert post_json(f"{url}/api/node/add", {"path": str(dirs[1])})[0] == 200
    assert [r["id"] for r in get_json(f"{url}/api/runs")["runs"]] == ["r1"] and len(get_json(f"{url}/api/node")["dirs"]) == 2
    assert post_json(f"{url}/api/node/update")[0] == 404
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a", "b", "cache"]
    node.close()


def test_a_node_asked_to_crawl_a_path_as_an_id_serves_it_by_that_id(node: Node, dirs: list[Path], http: str) -> None:
    status, body = post_json(f"{http}/api/node/add", {"path": str(dirs[0]), "id": "box:~/runs"})
    assert (status, body) == (200, {"name": "runs", "id": "box:~/runs", "url": dir_base("box:~/runs") + "/"})
    assert [o.id for o in node.holdings().dirs] == ["box:~/runs"]
    assert post_json(f"{http}/api/node/remove", {"id": "box:~/runs"}) == (200, {"ok": True})
    assert node.served() == []


def test_removing_a_directory_stops_serving_it_without_touching_its_files(node: Node, dirs: list[Path], http: str, state: Path) -> None:
    before = sorted(p.relative_to(dirs[0]) for p in dirs[0].rglob("*"))
    name = added(node, dirs[0])
    assert post_json(f"{http}/api/node/remove", {"name": name}) == (200, {"ok": True})
    assert request(f"{http}{dir_base(f'{node.identity.name}:{dirs[0]}')}/api/runs")[0] == 404
    assert json.loads((state / "roots.json").read_text()) == {"tracked": [], "links": [], "pulled": {}, "workspaces": []}
    assert sorted(p.relative_to(dirs[0]) for p in dirs[0].rglob("*")) == before


def test_the_control_socket_adds_directories_and_reports_status(ctl: ControlServer, node: Node, dirs: list[Path]) -> None:
    d = f"{node.identity.name}:{dirs[0]}"
    assert control.request({"op": "add", "path": str(dirs[0])}) == {"name": "runs", "url": f"http://127.0.0.1:1{dir_base(d)}/"}
    assert served(node) == [("runs", str(dirs[0]))]
    assert control.request({"op": "status"}) == {"urls": ["http://127.0.0.1:1/"], "dirs": [x.wire() for x in node.served()]}
    assert "error" in (control.request({"op": "add", "path": "~"}) or {})
    assert "error" in (control.request({"op": "add", "path": str(dirs[0] / "missing")}) or {})


def test_the_control_socket_is_private_to_the_user(ctl: ControlServer) -> None:
    assert ctl.path.stat().st_mode & 0o077 == 0


def test_a_second_trex_on_the_same_socket_is_refused(ctl: ControlServer, node: Node) -> None:
    with pytest.raises(RuntimeError, match="already running"):
        ControlServer(ctl.path, node, [])


def test_no_trex_means_no_reply(state: Path) -> None:
    assert control.request({"op": "status"}) is None
    state.mkdir()
    with socket.socket(socket.AF_UNIX) as s:
        s.bind(str(control.socket_path()))
    assert control.request({"op": "status"}) is None


def test_serve_hands_directories_to_the_running_trex_and_returns(ctl: ControlServer, node: Node, dirs: list[Path],
                                                                capsys: pytest.CaptureFixture[str]) -> None:
    main(["serve", str(dirs[1]), "-y"])
    assert served(node) == [("runs", str(dirs[1]))]
    assert f"http://127.0.0.1:1{dir_base(f'{node.identity.name}:{dirs[1]}')}/" in capsys.readouterr().out
    main(["serve"])
    assert "already running at http://127.0.0.1:1/" in capsys.readouterr().out


def test_servers_without_a_port_take_the_next_free_one(node: Node, monkeypatch: pytest.MonkeyPatch) -> None:
    with socket.socket() as taken:
        taken.bind(("127.0.0.1", 0))
        taken.listen()
        monkeypatch.setattr(server, "DEFAULT_PORT", taken.getsockname()[1])
        (srv,) = server.bind(node, ["127.0.0.1"], None)
        try:
            assert srv.server_address[1] > taken.getsockname()[1]
        finally:
            srv.server_close()


def test_added_and_removed_directories_are_remembered_most_recent_first_across_restarts(node: Node, dirs: list[Path], tmp_path: Path,
                                                                                        state: Path) -> None:
    for d in dirs:
        node.add(d)
    node.remove("runs<a>")
    assert node.history() == [str(dirs[0])]
    node.remove("runs")
    assert Node(tmp_path / "cache2", state / "roots.json").history() == [str(dirs[1]), str(dirs[0])]


def test_history_lists_only_unserved_directories_that_still_exist(node: Node, dirs: list[Path], tmp_path: Path) -> None:
    gone = tmp_path / "gone"
    gone.mkdir()
    for d in (*dirs, gone):
        node.remove_dir(node.add(d))
    node.add(dirs[0])
    gone.rmdir()
    assert node.history() == [str(dirs[1])]


def test_clearing_history_keeps_served_directories(node: Node, dirs: list[Path], tmp_path: Path, state: Path) -> None:
    node.remove_dir(node.add(dirs[0]))
    node.add(dirs[1])
    node.clear_history()
    assert node.history() == [] and [d.root for d in node.served()] == [str(dirs[1])]
    node.remove("runs")
    assert Node(tmp_path / "cache2", state / "roots.json").history() == [str(dirs[1])]


def test_http_add_serves_a_directory_and_history_can_be_cleared(node: Node, dirs: list[Path], http: str) -> None:
    d = f"{node.identity.name}:{dirs[0]}"
    assert post_json(f"{http}/api/node/add", {"path": str(dirs[0])}) == (200, {"name": "runs", "id": d, "url": dir_base(d) + "/"})
    assert node.by_id(d).ready.wait(10)
    assert get_json(f"{http}{dir_base(d)}/api/runs")["runs"][0]["id"] == "r1"
    assert post_json(f"{http}/api/node/remove", {"name": "runs"}) == (200, {"ok": True})
    assert get_json(f"{http}/api/node")["history"] == [str(dirs[0])]
    assert post_json(f"{http}/api/node/history/clear") == (200, {"ok": True})
    assert get_json(f"{http}/api/node")["history"] == []


@pytest.mark.parametrize("path,message", [
    ("runs", "not an absolute path"), ("~", "refusing to crawl"), ("/", "refusing to crawl"), ("/no/such/dir", "not a directory"),
])
def test_http_add_refuses_relative_home_root_and_missing_paths(node: Node, http: str, path: str, message: str) -> None:
    status, body = post_json(f"{http}/api/node/add", {"path": path})
    assert status == 400 and message in body["error"] and node.served() == []


def raw(url: str, method: str = "GET", headers: dict[str, str] | None = None, body: bytes | None = None) -> tuple[int, Any]:
    """(status, JSON body) of a request with exactly these headers."""
    parts = urllib.parse.urlsplit(url)
    conn = http_client.HTTPConnection(parts.hostname or "", parts.port)
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


@pytest.mark.parametrize("path", ["/d/nope/", "/w/nope/", "/w/nope"])
def test_unknown_directories_and_workspaces_redirect_home_without_being_cached(node: Node, http: str, path: str) -> None:
    parts = urllib.parse.urlsplit(http)
    conn = http_client.HTTPConnection(parts.hostname or "", parts.port)
    try:
        conn.request("GET", path)
        r = conn.getresponse()
        r.read()
        assert (r.status, r.getheader("Location")) in {(302, "/"), (303, "/"), (307, "/")}
        assert "no-store" in (r.getheader("Cache-Control") or "")
    finally:
        conn.close()
    node.set_workspace("nope", [])
    assert final_url(f"{http}/w/nope/") == f"{http}/w/nope/"


@pytest.mark.parametrize("host", ["evil.example.com", "evil.example.com:80", "localhost.evil.example.com"])
def test_requests_addressed_to_unknown_host_names_are_refused(http: str, host: str) -> None:
    status, body = raw(f"{http}/api/node", headers={"Host": host})
    assert status == 403 and "--allow-host" in body["error"]


@pytest.mark.parametrize("host", ["127.0.0.1:{port}", "[::1]:{port}", "localhost:{port}", "app.localhost:{port}",
                                  "{hostname}:{port}", "100.64.0.9:{port}"])
def test_requests_by_address_localhost_or_machine_name_are_served(http: str, host: str) -> None:
    port = urllib.parse.urlsplit(http).port
    status, body = raw(f"{http}/api/node", headers={"Host": host.format(port=port, hostname=socket.gethostname())})
    assert status == 200 and body["saves"]


def test_allowed_host_names_are_served(node: Node, http_server: Callable[[Server], str]) -> None:
    url = http_server(server.serve(node, "127.0.0.1", 0, allow=["Box.Tailnet.ts.net"]))
    assert raw(f"{url}/api/node", headers={"Host": "box.tailnet.ts.net"})[0] == 200


@pytest.mark.parametrize("origin", ["https://evil.example.com", "http://127.0.0.1:1", "null"])
def test_changes_posted_from_another_origin_are_refused(node: Node, dirs: list[Path], http: str, origin: str) -> None:
    host = urllib.parse.urlsplit(http).netloc
    status, _ = raw(f"{http}/api/node/add", "POST", {"Host": host, "Origin": origin}, json.dumps({"path": str(dirs[0])}).encode())
    assert status == 403 and node.served() == []


def test_changes_from_the_ui_itself_or_without_an_origin_are_allowed(node: Node, dirs: list[Path], http: str) -> None:
    host = urllib.parse.urlsplit(http).netloc
    body = json.dumps({"path": str(dirs[0])}).encode()
    assert raw(f"{http}/api/node/add", "POST", {"Host": host, "Origin": f"http://{host}"}, body)[0] == 200
    assert raw(f"{http}/api/node/add", "POST", {"Host": host}, json.dumps({"path": str(dirs[1])}).encode())[0] == 200
    assert len(node.served()) == 2


def test_removing_a_directory_closes_its_explorer_and_leaves_the_others(node: Node, dirs: list[Path]) -> None:
    a, b = (node.by_id(node.add(d)) for d in dirs)
    node.remove("runs<a>")
    assert wait_for(lambda: a.stopped, 10) and not b.stopped
