import http.client as http_client
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from trex import buckets as bk, control, remote, server, update
from trex.cli import main
from trex.control import ControlServer
from trex.mirror import Pull
from trex.node import Node, dir_base
from trex.remote import Address, Remote
from trex.server import Server

from helpers import post_json, request, wait_for, write_run


def sessions(home: Path) -> list[tuple[int, str]]:
    """(pid, remote socket) of every ssh session started, oldest first."""
    return [(int(pid), sock) for pid, sock in (line.split() for line in (home / "sessions").read_text().splitlines())]


def session(node: Node, host: str = "box") -> Remote:
    """The ssh session to `host`."""
    s = node.links[host].session
    assert s is not None
    return s


def api(url: str) -> Any:
    return json.loads(request(url)[2])


@pytest.fixture
def runs(tmp_path: Path) -> Path:
    d = tmp_path / "remote data" / "my runs"
    write_run(d / "a" / "r1", image=True)
    write_run(d / "b" / "r2", image=True)
    return d


@pytest.fixture
def node(tmp_path: Path, home: Path) -> Iterator[Node]:
    n = Node(tmp_path / "cache", tmp_path / "state" / "roots.json")
    yield n
    n.close()


@pytest.fixture
def http(node: Node, http_server: Callable[[Server], str]) -> str:
    return http_server(server.serve(node, "127.0.0.1", 0))


@pytest.mark.parametrize("spec,want", [
    ("box:/data/runs", Address("box", "/data/runs")),
    ("me@box.lan:~/my runs", Address("me@box.lan", "~/my runs")),
    ("[::1]:/x", Address("[::1]", "/x")),
    ("/data/runs", None), ("~/runs", None), ("runs", None), ("box:", None), ("a/b:c", None), ("http://box:13898", None),
])
def test_scp_style_addresses_name_remote_directories(spec: str, want: Address | None) -> None:
    assert remote.parse(spec) == want


def test_an_existing_local_path_with_a_colon_is_local(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "run:1").mkdir()
    monkeypatch.chdir(tmp_path)
    assert remote.parse("run:1") is None


def test_remote_machines_get_this_trex_over_ssh_once(node: Node, runs: Path, home: Path) -> None:
    node.add_remote(f"box:{runs}")
    s = session(node)
    assert s.state == "connected"
    name, data = remote.wheel()
    args = (home / "uvx.args").read_text().split("\n")
    assert args[args.index("--from") + 1] == str(home / remote.WHEELS / name)
    assert (home / remote.WHEELS / name).read_bytes() == data and sorted(p.name for p in (home / remote.WHEELS).iterdir()) == [name]
    sent = len(sessions(home))  # the session that found no wheel, the copy, the session serving it
    (pid, _) = sessions(home)[-1]
    os.kill(pid, signal.SIGKILL)
    assert wait_for(lambda: len(sessions(home)) == sent + 1 and s.state == "connected")
    assert sent == 3


def test_this_trex_as_a_wheel_installs_with_its_command_and_ui(tmp_path: Path) -> None:
    name, data = remote.wheel()
    whl = tmp_path / name
    whl.write_bytes(data)
    env = tmp_path / "env"
    subprocess.run([update.uv(), "venv", "-q", "--python", sys.executable, str(env)], check=True)
    subprocess.run([update.uv(), "pip", "install", "-q", "--offline", "--no-deps", "--python", str(env / "bin" / "python"), str(whl)], check=True)
    site = next(env.glob("lib/python*/site-packages"))
    assert (env / "bin" / "trex").exists() and (site / "trex" / "static" / "app.js").read_bytes() == (Path(remote.__file__).parent / "static" / "app.js").read_bytes()


def test_a_wheel_is_the_same_for_the_same_package_and_named_for_its_contents(tmp_path: Path) -> None:
    pkg = tmp_path / "trex"
    shutil.copytree(Path(remote.__file__).parent, pkg, ignore=shutil.ignore_patterns("__pycache__"))
    first = remote.build_wheel(pkg)
    assert remote.build_wheel(pkg) == first
    (pkg / "static" / "app.js").write_text("changed")
    assert remote.build_wheel(pkg)[0] != first[0]


def test_a_remote_directory_is_crawled_there_by_a_temporary_trex_and_served_here(node: Node, runs: Path, http: str, home: Path) -> None:
    d = node.add_remote(f"box:{runs}")
    assert d == f"box:{runs}" and node.names()[d] == "my runs" and [x.state for x in node.served()] == ["connected"]
    base = f"{http}{dir_base(d)}"
    args = (home / "uvx.args").read_text().split("\n")
    assert args[args.index("--no-build-package") + 1] == "numpy" and args[args.index("--python") + 1] == "3.12"
    assert "--temporary" in args and args[args.index("--name") + 1] == "box"
    status, _, body = request(f"{base}/")
    assert status == 200 and b"/static/app.js" in body
    assert wait_for(lambda: len(api(f"{base}/api/runs")["runs"]) == 2)
    listing = api(f"{base}/api/runs?path=a")
    assert [r["id"] for r in listing["runs"]] == ["a/r1"]
    media = listing["media"][0]
    status, headers, data = request(f"{base}/m/{urllib.parse.quote('a/r1', safe='')}/{media[5]}", headers={"Range": "bytes=8-11"})
    assert status == 206 and data == bytes(range(4)) and headers["Content-Range"].startswith("bytes 8-11/")
    body = b'{"blocks": [{"key": "loss", "level": 20, "index": 0, "runs": ["a/r1"]}]}'
    answer = bk.decode(bk.unframe(urllib.request.urlopen(urllib.request.Request(f"{base}/api/buckets", data=body, method="POST")).read())[0])
    assert answer.paths == ["a/r1"] and answer.buckets.run.size >= 1


def test_an_add_whose_last_hop_is_an_ssh_link_is_tracked_as_host_path_by_the_node_before(
        node: Node, runs: Path, http: str, tmp_path: Path, home: Path) -> None:
    node.add_remote(f"box:{runs}")
    box = node.links["box"].identity
    assert box is not None and box.name == "box"
    other = tmp_path / "remote data" / "other"
    write_run(other / "r9")
    laptop = Node(tmp_path / "laptop" / "cache", name="laptop")
    try:
        laptop.add_link(http)
        assert wait_for(lambda: len(laptop.peers()) == 2)
        assert laptop.add_at(str(other), [node.identity.id, box.id]) == f"box:{other}"
        assert node.tracked() == [f"box:{runs}", f"box:{other}"] and laptop.tracked() == []
        assert laptop.pulled[f"box:{other}"].via == [box.id, node.identity.id]
        with pytest.raises(ValueError, match="is not a path on box"):
            laptop.add_at(f"elsewhere:{other}", [node.identity.id, box.id])
    finally:
        laptop.close()


def test_every_directory_on_a_host_shares_one_session(node: Node, runs: Path, tmp_path: Path, home: Path) -> None:
    other = tmp_path / "remote data" / "other"
    write_run(other / "r9")
    node.add_remote(f"box:{runs}")
    started = len(sessions(home))
    node.add_remote(f"box:{other}")
    assert len(sessions(home)) == started and sorted(d.name for d in node.served()) == ["my runs", "other"]
    assert wait_for(lambda: [r.id for r in node.by_id(f"box:{other}").runs("").runs] == ["r9"])
    node.remove("other")
    assert wait_for(lambda: [d.name for d in node.served()] == ["my runs"]) and session(node).state == "connected"


def test_the_live_stream_comes_from_the_mirrored_directory(node: Node, runs: Path, http: str) -> None:
    d = node.add_remote(f"box:{runs}")
    with urllib.request.urlopen(f"{http}{dir_base(d)}/api/stream", timeout=30) as r:
        assert r.headers["Content-Type"] == "text/event-stream" and r.readline() == b"retry: 2000\n"


def test_removing_a_hosts_last_directory_ends_its_session(node: Node, runs: Path, home: Path) -> None:
    d = node.add_remote(f"box:{runs}")
    s = session(node)
    _, sock = sessions(home)[-1]
    assert wait_for(lambda: Path(sock).exists())
    node.remove(node.names()[d])
    assert wait_for(lambda: not Path(sock).exists() and not s.local.exists())
    assert node.history() == [f"box:{runs}"] and "box" not in node.links


def test_a_dropped_connection_reconnects_and_the_host_crawls_its_directories_again(node: Node, runs: Path, home: Path) -> None:
    d = node.add_remote(f"box:{runs}")
    s = session(node)
    started = len(sessions(home))
    os.kill(sessions(home)[-1][0], signal.SIGKILL)
    assert wait_for(lambda: len(sessions(home)) == started + 1 and s.state == "connected")
    pull = node.by_id(d).origin
    assert isinstance(pull, Pull) and wait_for(lambda: pull.connected and node.links["box"].offers is not None and d in node.links["box"].offers)


@pytest.mark.parametrize("spec,message", [("nowhere:/runs", "Could not resolve hostname"), ("box:/no/such/dir", "not a directory")])
def test_adding_an_unreachable_remote_fails_with_the_reason(node: Node, spec: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        node.add_remote(spec)
    assert node.served() == [] and node.history() == [] and node.links == {}


def test_adding_a_remote_without_uv_says_how_to_install_it(node: Node, runs: Path, home: Path) -> None:
    (home / ".local" / "bin" / "uvx").unlink()
    with pytest.raises(ValueError, match="uv is not installed .*astral.sh/uv/install.sh"):
        node.add_remote(f"box:{runs}")


def test_a_remote_directory_answers_from_its_copy_while_its_host_is_unreachable(node: Node, runs: Path, http: str, home: Path) -> None:
    d = node.add_remote(f"box:{runs}")
    base = f"{http}{dir_base(d)}"
    assert wait_for(lambda: len(api(f"{base}/api/runs")["runs"]) == 2)
    (home / ".local" / "bin" / "uvx").unlink()
    os.kill(sessions(home)[-1][0], signal.SIGKILL)
    gone = lambda: [(r["state"], "uv is not installed" in r["error"]) for r in api(f"{http}/api/node")["dirs"]]
    pull = node.by_id(d).origin
    assert isinstance(pull, Pull) and wait_for(lambda: gone() == [("unreachable", True)] and not pull.connected)
    assert sorted(r["id"] for r in api(f"{base}/api/runs")["runs"]) == ["a/r1", "b/r2"]


def test_the_add_menu_takes_remote_addresses(node: Node, runs: Path, http: str) -> None:
    d = f"box:{runs}"
    assert post_json(f"{http}/api/node/add", {"path": d}) == (200, {"name": "my runs", "id": d, "url": dir_base(d) + "/"})
    assert [(r["root"], r["state"], r["link"]) for r in api(f"{http}/api/node")["dirs"]] == [(d, "connected", None)]
    assert api(f"{http}/api/node")["links"] == []
    assert post_json(f"{http}/api/node/add", {"path": "nowhere:/x"})[0] == 400


def test_saved_remote_directories_reconnect_after_a_restart(node: Node, runs: Path, tmp_path: Path) -> None:
    node.add_remote(f"box:{runs}")
    node.close()
    again = Node(tmp_path / "cache", tmp_path / "state" / "roots.json")
    try:
        again.load()
        assert wait_for(lambda: session(again).state == "connected" and [d.state for d in again.served()] == ["connected"])
    finally:
        again.close()


def test_serve_hands_remote_addresses_to_the_running_trex(node: Node, runs: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ctl = ControlServer(control.socket_path(), node, ["http://127.0.0.1:1/"])
    threading.Thread(target=ctl.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
    try:
        main(["serve", f"box:{runs}"])
        assert dir_base(f"box:{runs}") in capsys.readouterr().out and [d.state for d in node.served()] == ["connected"]
    finally:
        ctl.shutdown()
        ctl.server_close()


def test_a_remote_directorys_answers_keep_the_connection_usable(node: Node, runs: Path, http: str) -> None:
    d = node.add_remote(f"box:{runs}")
    parts = urllib.parse.urlsplit(http)
    conn = http_client.HTTPConnection(parts.hostname or "", parts.port, timeout=10)
    try:
        for path in ("/api/info", "/", "/api/tree"):
            conn.request("GET", f"{dir_base(d)}{path}")
            r = conn.getresponse()
            assert r.status == 200 and int(r.headers["Content-Length"]) == len(r.read())
    finally:
        conn.close()


def test_closing_the_node_ends_its_sessions_and_keeps_its_directories_saved(node: Node, runs: Path, home: Path, tmp_path: Path) -> None:
    node.add_remote(f"box:{runs}")
    s = session(node)
    _, sock = sessions(home)[-1]
    node.close()
    assert not s.local.exists() and wait_for(lambda: not Path(sock).exists()) and node.served() == []
    assert json.loads((tmp_path / "state" / "roots.json").read_text())["tracked"] == [f"box:{runs}"]


def test_closing_a_remote_while_it_starts_ends_its_session(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(remote, "CLOSE_TIMEOUT", 3.0)
    r = Remote("far")
    popen = subprocess.Popen
    started: list[subprocess.Popen[str]] = []
    closers: list[threading.Thread] = []

    def close_during_start(*a: Any, **kw: Any) -> subprocess.Popen[str]:
        proc = popen(*a, **kw)
        started.append(proc)
        closer = threading.Thread(target=r.close)
        closer.start()
        closers.append(closer)
        assert wait_for(lambda: r.closing, timeout=5)
        return proc

    monkeypatch.setattr(remote.subprocess, "Popen", close_during_start)
    r.start()
    try:
        assert wait_for(lambda: closers, timeout=10)
        closers[0].join(10)
        assert not closers[0].is_alive()
        started[0].wait(timeout=5)
    finally:
        for proc in started:
            proc.kill()
