import http.client as http_client
import json
import os
import signal
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pytest

import trex
from trex import daemon, remote, server, update
from trex.cli import main
from trex.daemon import ControlServer, Roots
from trex.remote import Address, Remote

FAKE_SSH = """#!{python}
import os, subprocess, sys
args, rest, local, sock = sys.argv[1:], [], None, None
i = 0
while i < len(args):
    if args[i] == "-o":
        i += 2
    elif args[i] == "-L":
        local, sock = args[i + 1].split(":", 1)
        i += 2
    else:
        rest.append(args[i])
        i += 1
home = os.environ["FAKE_REMOTE_HOME"]
with open(os.path.join(home, "sessions"), "a") as f:
    f.write(f"{{os.getpid()}} {{sock}}\\n")
if rest[0] == "nowhere":
    sys.exit("ssh: Could not resolve hostname nowhere: Name or service not known")
if os.path.lexists(local):
    os.unlink(local)
os.symlink(sock, local)
try:
    sys.exit(subprocess.call(["sh", "-c", " ".join(rest[1:])], env={{**os.environ, "HOME": home, "PATH": "/usr/bin:/bin"}}))
finally:
    os.unlink(local)
"""
FAKE_UVX = """#!/bin/sh
printf '%s\\n' "$@" > "$HOME/uvx.args"
while [ "$1" != trex ]; do shift; done
shift
exec {python} -m trex "$@"
"""


def write_run(d, n=5):
    run = trex.init(d, commit_interval=0.05)
    for i in range(n):
        run.log({"loss": 1.0 / (i + 1)}, step=i)
    run.log_image("img", b"\x89PNG\r\n\x1a\n" + bytes(range(64)), step=n - 1)
    run.finish()


def wait_for(cond, timeout=20.0):
    end = time.time() + timeout
    while not cond() and time.time() < end:
        time.sleep(0.05)
    return cond()


@pytest.fixture
def home(tmp_path, monkeypatch):
    """The fake remote machine's home, reached by host names through a fake ssh, with uv installed."""
    h = tmp_path / "remote-home"
    (h / ".local" / "bin").mkdir(parents=True)
    for path, text in [(tmp_path / "ssh", FAKE_SSH), (h / ".local" / "bin" / "uvx", FAKE_UVX)]:
        path.write_text(text.format(python=sys.executable))
        path.chmod(0o755)
    (h / "sessions").touch()
    monkeypatch.setenv("TREX_SSH", str(tmp_path / "ssh"))
    monkeypatch.setenv("FAKE_REMOTE_HOME", str(h))
    monkeypatch.setenv("TREX_DAEMON_DIR", str(tmp_path / "state"))
    return h


def sessions(home):
    return [(int(pid), sock) for pid, sock in (line.split() for line in (home / "sessions").read_text().splitlines())]


@pytest.fixture
def runs(tmp_path):
    d = tmp_path / "remote data" / "my runs"
    write_run(d / "a" / "r1")
    write_run(d / "b" / "r2")
    return d


@pytest.fixture
def roots(tmp_path, home):
    r = Roots(tmp_path / "cache", tmp_path / "state" / "roots.json")
    yield r
    for name in list(r.entries):
        r.entries.pop(name).close()


@pytest.fixture
def http(roots):
    srv = server.serve(None, "127.0.0.1", 0, roots)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def get(url, headers=None):
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers or {}), timeout=30) as r:
            return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


def post(url, body):
    try:
        req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST")
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


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


@pytest.mark.parametrize("base,commit,want", [
    ("git+https://example.org/u/trex", "abc123", "git+https://example.org/u/trex@abc123"),
    ("git+ssh://git@example.org/u/trex@v1", "abc123", "git+ssh://git@example.org/u/trex@abc123"),
    ("git+https://example.org/u/trex", None, "git+https://example.org/u/trex"),
    ("/src/trex", "abc123", "/src/trex"),
])
def test_remote_machines_run_this_trex_commit_from_the_source(monkeypatch, base, commit, want):
    monkeypatch.setenv("TREX_SOURCE", base)
    monkeypatch.setattr(update, "RUNNING", {"version": "0.1.0", "commit": commit})
    assert remote.source() == want


def test_remote_directories_are_served_through_the_daemon(roots, runs, http, home, monkeypatch):
    monkeypatch.setattr(update, "RUNNING", {"version": "0.1.0", "commit": "abc123"})
    name = roots.add_remote(f"box:{runs}")
    assert name == "my runs@box" and roots.served()[0]["state"] == "connected"
    base = f"{http}/r/{urllib.parse.quote(name)}"
    assert "--from" in (args := (home / "uvx.args").read_text().split("\n")) and args[args.index("--from") + 1].endswith("@abc123")
    status, _, body = get(f"{base}/")
    assert status == 200 and b"/static/app.js" in body
    assert wait_for(lambda: len(json.loads(get(f"{base}/api/runs")[2])["runs"]) == 2)
    listing = json.loads(get(f"{base}/api/runs?path=a")[2])
    assert [r["id"] for r in listing["runs"]] == ["a/r1"]
    media = listing["media"][0]
    status, headers, data = get(f"{base}/m/{urllib.parse.quote('a/r1', safe='')}/{media[5]}", {"Range": "bytes=8-11"})
    assert status == 206 and data == bytes(range(4)) and headers["Content-Range"].startswith("bytes 8-11/")
    tiles = urllib.request.urlopen(urllib.request.Request(f"{base}/api/tiles", data=b'[["a/r1", "loss", "top"]]', method="POST"))
    assert int.from_bytes(tiles.read()[:4], "little") >= 1


def test_the_live_stream_passes_through(roots, runs, http):
    name = roots.add_remote(f"box:{runs}")
    with urllib.request.urlopen(f"{http}/r/{urllib.parse.quote(name)}/api/stream", timeout=30) as r:
        assert r.headers["Content-Type"] == "text/event-stream" and r.readline() == b"retry: 2000\n"


def test_removing_a_remote_directory_ends_its_remote_server(roots, runs, home):
    name = roots.add_remote(f"box:{runs}")
    entry = roots.get(name)
    assert isinstance(entry, Remote)
    (_, sock), = sessions(home)
    assert wait_for(lambda: Path(sock).exists())
    roots.remove(name)
    assert wait_for(lambda: not Path(sock).exists() and not entry.local.exists())
    assert roots.history() == [f"box:{runs}"]


def test_a_dropped_connection_reconnects(roots, runs, home):
    entry = roots.get(roots.add_remote(f"box:{runs}"))
    assert isinstance(entry, Remote)
    (pid, _), = sessions(home)
    os.kill(pid, signal.SIGKILL)
    assert wait_for(lambda: len(sessions(home)) == 2 and entry.state == "connected")


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
    status, headers, body = get(f"{http}/r/far/")
    assert status == 503 and b'http-equiv="refresh"' in body and b"far:/runs" in body
    status, _, body = get(f"{http}/r/far/api/runs")
    assert status == 503 and "starting" in json.loads(body)["error"]


def test_the_add_menu_takes_remote_addresses(roots, runs, http):
    status, body = post(f"{http}/api/daemon/add", {"path": f"box:{runs}"})
    assert status == 200 and body["url"] == f"/r/{urllib.parse.quote('my runs@box', safe='')}/"
    info = json.loads(get(f"{http}/api/daemon")[2])
    assert [(r["root"], r["state"]) for r in info["roots"]] == [(f"box:{runs}", "connected")]
    assert post(f"{http}/api/daemon/add", {"path": "nowhere:/x"})[0] == 400


def test_saved_remote_directories_reconnect_after_a_restart(roots, runs, tmp_path, home):
    roots.add_remote(f"box:{runs}")
    again = Roots(tmp_path / "cache2", tmp_path / "state" / "roots.json")
    try:
        again.load()
        entry = again.get("my runs@box")
        assert isinstance(entry, Remote) and wait_for(lambda: entry.state == "connected")
    finally:
        for name in list(again.entries):
            again.entries.pop(name).close()


def test_serve_hands_remote_addresses_to_the_daemon(roots, runs, capsys):
    control = ControlServer(daemon.socket_path(), roots, ["http://127.0.0.1:1/"])
    threading.Thread(target=control.serve_forever, daemon=True).start()
    try:
        main(["serve", f"box:{runs}"])
        assert "/r/my%20runs%40box/" in capsys.readouterr().out and roots.served()[0]["state"] == "connected"
    finally:
        control.shutdown()
        control.server_close()


def test_serving_a_remote_address_needs_a_daemon(home):
    with pytest.raises(SystemExit, match="trex daemon"):
        main(["serve", "box:/runs"])


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
    (_, sock), = sessions(home)
    roots.close()
    assert not entry.local.exists() and wait_for(lambda: not Path(sock).exists()) and roots.served() == []
    assert [e["root"] for e in json.loads((tmp_path / "state" / "roots.json").read_text())] == [f"box:{runs}"]
