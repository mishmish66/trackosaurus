import http.client as http_client
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pytest

from trex import buckets as bk, daemon, remote, server, update
from trex.cli import main
from trex.daemon import ControlServer, Roots
from trex.remote import Address, Remote

from helpers import post_json, request, wait_for, write_run


def sessions(home):
    return [(int(pid), sock) for pid, sock in (line.split() for line in (home / "sessions").read_text().splitlines())]


@pytest.fixture
def runs(tmp_path):
    d = tmp_path / "remote data" / "my runs"
    write_run(d / "a" / "r1", image=True)
    write_run(d / "b" / "r2", image=True)
    return d


@pytest.fixture
def roots(tmp_path, home):
    r = Roots(tmp_path / "cache", tmp_path / "state" / "roots.json")
    yield r
    for name in list(r.entries):
        r.entries.pop(name).close()


@pytest.fixture
def http(roots, http_server):
    return http_server(server.serve(None, "127.0.0.1", 0, roots))


@pytest.mark.parametrize("spec,want", [
    ("box:/data/runs", Address("box", "/data/runs")),
    ("me@box.lan:~/my runs", Address("me@box.lan", "~/my runs")),
    ("[::1]:/x", Address("[::1]", "/x")),
    ("/data/runs", None), ("~/runs", None), ("runs", None), ("box:", None), ("a/b:c", None),
])
def test_scp_style_addresses_name_remote_directories(spec, want):
    assert remote.parse(spec) == want


def test_an_existing_local_path_with_a_colon_is_local(tmp_path, monkeypatch):
    (tmp_path / "run:1").mkdir()
    monkeypatch.chdir(tmp_path)
    assert remote.parse("run:1") is None


def test_remote_machines_get_this_trex_over_ssh_once(roots, runs, home):
    entry = roots.get(roots.add_remote(f"box:{runs}"))
    assert isinstance(entry, Remote) and entry.state == "connected"
    name, data = remote.wheel()
    args = (home / "uvx.args").read_text().split("\n")
    assert args[args.index("--from") + 1] == str(home / remote.WHEELS / name)
    assert (home / remote.WHEELS / name).read_bytes() == data and sorted(p.name for p in (home / remote.WHEELS).iterdir()) == [name]
    sent = len(sessions(home))  # the session that found no wheel, the copy, the session serving it
    (pid, _) = sessions(home)[-1]
    os.kill(pid, signal.SIGKILL)
    assert wait_for(lambda: len(sessions(home)) == sent + 1 and entry.state == "connected")
    assert sent == 3


def test_this_trex_as_a_wheel_installs_with_its_command_and_ui(tmp_path):
    name, data = remote.wheel()
    whl = tmp_path / name
    whl.write_bytes(data)
    env = tmp_path / "env"
    subprocess.run([update.uv(), "venv", "-q", "--python", sys.executable, str(env)], check=True)
    subprocess.run([update.uv(), "pip", "install", "-q", "--offline", "--no-deps", "--python", str(env / "bin" / "python"), str(whl)], check=True)
    site = next(env.glob("lib/python*/site-packages"))
    assert (env / "bin" / "trex").exists() and (site / "trex" / "static" / "app.js").read_bytes() == (Path(remote.__file__).parent / "static" / "app.js").read_bytes()


def test_a_wheel_is_the_same_for_the_same_package_and_named_for_its_contents(tmp_path):
    pkg = tmp_path / "trex"
    shutil.copytree(Path(remote.__file__).parent, pkg, ignore=shutil.ignore_patterns("__pycache__"))
    first = remote.build_wheel(pkg)
    assert remote.build_wheel(pkg) == first
    (pkg / "static" / "app.js").write_text("changed")
    assert remote.build_wheel(pkg)[0] != first[0]


def test_remote_directories_are_served_through_the_daemon(roots, runs, http, home):
    name = roots.add_remote(f"box:{runs}")
    assert name == "my runs" and roots.served()[0]["state"] == "connected"
    base = f"{http}/r/{urllib.parse.quote(name)}"
    args = (home / "uvx.args").read_text().split("\n")
    assert args[args.index("--no-build-package") + 1] == "numpy" and args[args.index("--python") + 1] == "3.12"
    status, _, body = request(f"{base}/")
    assert status == 200 and b"/static/app.js" in body
    assert wait_for(lambda: len(json.loads(request(f"{base}/api/runs")[2])["runs"]) == 2)
    listing = json.loads(request(f"{base}/api/runs?path=a")[2])
    assert [r["id"] for r in listing["runs"]] == ["a/r1"]
    media = listing["media"][0]
    status, headers, data = request(f"{base}/m/{urllib.parse.quote('a/r1', safe='')}/{media[5]}", headers={"Range": "bytes=8-11"})
    assert status == 206 and data == bytes(range(4)) and headers["Content-Range"].startswith("bytes 8-11/")
    body = b'{"key": "loss", "level": 20, "index": 0, "runs": ["a/r1"]}'
    answer = bk.decode(urllib.request.urlopen(urllib.request.Request(f"{base}/api/buckets", data=body, method="POST")).read())
    assert answer.paths == ["a/r1"] and answer.buckets.run.size >= 1


def test_the_live_stream_passes_through(roots, runs, http):
    name = roots.add_remote(f"box:{runs}")
    with urllib.request.urlopen(f"{http}/r/{urllib.parse.quote(name)}/api/stream", timeout=30) as r:
        assert r.headers["Content-Type"] == "text/event-stream" and r.readline() == b"retry: 2000\n"


def test_removing_a_remote_directory_ends_its_remote_server(roots, runs, home):
    name = roots.add_remote(f"box:{runs}")
    entry = roots.get(name)
    assert isinstance(entry, Remote)
    _, sock = sessions(home)[-1]
    assert wait_for(lambda: Path(sock).exists())
    roots.remove(name)
    assert wait_for(lambda: not Path(sock).exists() and not entry.local.exists())
    assert roots.history() == [f"box:{runs}"]


def test_a_dropped_connection_reconnects(roots, runs, home):
    entry = roots.get(roots.add_remote(f"box:{runs}"))
    assert isinstance(entry, Remote)
    started = len(sessions(home))
    os.kill(sessions(home)[-1][0], signal.SIGKILL)
    assert wait_for(lambda: len(sessions(home)) == started + 1 and entry.state == "connected")


@pytest.mark.parametrize("spec,message", [("nowhere:/runs", "Could not resolve hostname"), ("box:/no/such/dir", "not a directory")])
def test_adding_an_unreachable_remote_fails_with_the_reason(roots, spec, message):
    with pytest.raises(ValueError, match=message):
        roots.add_remote(spec)
    assert roots.served() == [] and roots.history() == []


def test_adding_a_remote_without_uv_says_how_to_install_it(roots, runs, home):
    (home / ".local" / "bin" / "uvx").unlink()
    with pytest.raises(ValueError, match="uv is not installed .*astral.sh/uv/install.sh"):
        roots.add_remote(f"box:{runs}")


def test_a_remote_that_is_not_connected_answers_with_a_page_that_reloads(roots, http):
    roots.entries["far"] = Remote("far:/runs", Address("far", "/runs"))
    status, headers, body = request(f"{http}/r/far/")
    assert status == 503 and b'http-equiv="refresh"' in body and b"far:/runs" in body
    status, _, body = request(f"{http}/r/far/api/runs")
    assert status == 503 and "starting" in json.loads(body)["error"]


def test_the_add_menu_takes_remote_addresses(roots, runs, http):
    status, body = post_json(f"{http}/api/daemon/add", {"path": f"box:{runs}"})
    assert status == 200 and body["url"] == f"/r/{urllib.parse.quote('my runs', safe='')}/"
    info = json.loads(request(f"{http}/api/daemon")[2])
    assert [(r["root"], r["state"]) for r in info["roots"]] == [(f"box:{runs}", "connected")]
    assert post_json(f"{http}/api/daemon/add", {"path": "nowhere:/x"})[0] == 400


def test_saved_remote_directories_reconnect_after_a_restart(roots, runs, tmp_path, home):
    roots.add_remote(f"box:{runs}")
    again = Roots(tmp_path / "cache2", tmp_path / "state" / "roots.json")
    try:
        again.load()
        entry = again.get("my runs")
        assert isinstance(entry, Remote) and wait_for(lambda: entry.state == "connected")
    finally:
        for name in list(again.entries):
            again.entries.pop(name).close()


def test_serve_hands_remote_addresses_to_the_daemon(roots, runs, capsys):
    control = ControlServer(daemon.socket_path(), roots, ["http://127.0.0.1:1/"])
    threading.Thread(target=control.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
    try:
        main(["serve", f"box:{runs}"])
        assert "/r/my%20runs/" in capsys.readouterr().out and roots.served()[0]["state"] == "connected"
    finally:
        control.shutdown()
        control.server_close()


def test_serving_a_remote_address_needs_a_daemon(home, capsys):
    with pytest.raises(SystemExit) as e:
        main(["serve", "box:/runs"])
    assert e.value.code == 2 and "trex daemon" in capsys.readouterr().err


def test_passed_through_answers_keep_the_connection_usable(roots, runs, http):
    name = urllib.parse.quote(roots.add_remote(f"box:{runs}"))
    parts = urllib.parse.urlsplit(http)
    conn = http_client.HTTPConnection(parts.hostname, parts.port, timeout=10)
    try:
        for path in ("/api/info", "/", "/api/tree"):
            conn.request("GET", f"/r/{name}{path}")
            r = conn.getresponse()
            assert r.status == 200 and int(r.headers["Content-Length"]) == len(r.read())
    finally:
        conn.close()


def test_closing_the_daemons_directories_ends_remote_sessions_and_keeps_them_saved(roots, runs, home, tmp_path):
    entry = roots.get(roots.add_remote(f"box:{runs}"))
    assert isinstance(entry, Remote)
    _, sock = sessions(home)[-1]
    roots.close()
    assert not entry.local.exists() and wait_for(lambda: not Path(sock).exists()) and roots.served() == []
    assert json.loads((tmp_path / "state" / "roots.json").read_text())["tracked"] == [f"box:{runs}"]


def test_closing_a_remote_while_it_starts_ends_its_session(runs, home, monkeypatch):
    monkeypatch.setattr(remote, "CLOSE_TIMEOUT", 3.0)
    r = Remote(f"far:{runs}", Address("far", str(runs)))
    popen, started, closers = subprocess.Popen, [], []

    def close_during_start(*a, **kw):
        proc = popen(*a, **kw)
        started.append(proc)
        closer = threading.Thread(target=r.close)
        closer.start()
        closers.append(closer)
        assert wait_for(r._closed.is_set, timeout=5)
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
