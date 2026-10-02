"""`trex serve` and `trex daemon` as real processes, each on a free port with a private state directory."""

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import urllib.request
from pathlib import Path

import pytest

import trex


def write_run(d, n=5):
    run = trex.init(d, commit_interval=0.05)
    for i in range(n):
        run.log({"x": float(i)}, step=i)
    run.finish()


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def get_json(url):
    with urllib.request.urlopen(url, timeout=10) as r:
        return json.loads(r.read())


@pytest.fixture
def env(tmp_path):
    return {**os.environ, "TREX_DAEMON_DIR": str(tmp_path / "state"), "TREX_CACHE": str(tmp_path / "cache"),
            "PYTHON_COLORS": "0", "NO_COLOR": "1"}


def trex_cmd(*argv):
    return [sys.executable, "-m", "trex", *map(str, argv)]


def start(env, *argv):
    """A trex server process, once it has printed its address."""
    p = subprocess.Popen(trex_cmd(*argv), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert p.stdout and p.stderr
    line = p.stdout.readline()
    assert "http://" in line, p.stderr.read()
    return p


def interrupt(p, sig=signal.SIGINT):
    """Exit code after `sig`."""
    p.send_signal(sig)
    return p.wait(timeout=20)


def test_standalone_server_answers_until_interrupted(tmp_path, env):
    write_run(tmp_path / "runs" / "r")
    port = free_port()
    p = start(env, "serve", tmp_path / "runs", "--standalone", "--port", port)
    try:
        assert get_json(f"http://127.0.0.1:{port}/api/info")["root"] == str(tmp_path / "runs")
    finally:
        assert interrupt(p) == 0


def test_daemon_takes_directories_from_serve_and_removes_its_socket_on_exit(tmp_path, env):
    a, b = tmp_path / "a" / "runs", tmp_path / "b" / "logs"
    write_run(a / "r")
    write_run(b / "r")
    port = free_port()
    p = start(env, "daemon", a, "--port", port)
    try:
        added = subprocess.run(trex_cmd("serve", b), env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=60)
        assert added.returncode == 0 and f"http://127.0.0.1:{port}/r/logs/" in added.stdout
        assert [r["root"] for r in get_json(f"http://127.0.0.1:{port}/api/daemon")["roots"]] == [str(a), str(b)]
        second = subprocess.run(trex_cmd("daemon", "--port", free_port()), env=env, capture_output=True, text=True, timeout=60)
        assert second.returncode != 0 and "already running" in second.stderr
    finally:
        assert interrupt(p) == 0
    assert not (tmp_path / "state" / "daemon.sock").exists()
    assert json.loads((tmp_path / "state" / "roots.json").read_text())["tracked"] == [str(a), str(b)]


def test_restarted_daemon_serves_the_directories_it_had(tmp_path, env):
    write_run(tmp_path / "runs" / "r")
    port = free_port()
    assert interrupt(start(env, "daemon", tmp_path / "runs", "--port", port)) == 0
    p = start(env, "daemon", "--port", port)
    try:
        assert [r["name"] for r in get_json(f"http://127.0.0.1:{port}/api/daemon")["roots"]] == ["runs"]
    finally:
        assert interrupt(p) == 0


def test_output_to_a_closed_pipe_ends_without_a_traceback(tmp_path, env):
    write_run(tmp_path / "r", 20000)
    p = subprocess.Popen(trex_cmd("series", tmp_path / "r", "-k", "x"), env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True)
    assert p.stdout and p.stderr
    p.stdout.readline()
    p.stdout.close()
    assert "Traceback" not in p.stderr.read()
    p.wait(timeout=20)


def test_terminated_daemon_exits_cleanly_and_removes_its_socket(tmp_path, env):
    p = start(env, "daemon", "--port", free_port())
    assert (tmp_path / "state" / "daemon.sock").exists()
    assert interrupt(p, signal.SIGTERM) == 0
    assert not (tmp_path / "state" / "daemon.sock").exists()


GIT = ["git", "-c", "user.name=trex", "-c", "user.email=trex@localhost", "-c", "init.defaultBranch=main"]


@pytest.fixture
def forge(tmp_path):
    """(source, push) of a git forge holding this checkout's files; `push()` adds a commit and returns its id."""
    repo = Path(__file__).resolve().parents[1]
    files = subprocess.run(["git", "ls-files", "-co", "--exclude-standard"], cwd=repo, capture_output=True, text=True,
                           check=True).stdout.splitlines()
    work, bare = tmp_path / "work", tmp_path / "forge.git"
    for f in files:
        if (repo / f).is_file():
            (work / f).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(repo / f, work / f)
    subprocess.run([*GIT, "init", "-q", "--bare", str(bare)], check=True)

    def push():
        subprocess.run([*GIT, "init", "-q"], cwd=work, check=True)
        subprocess.run([*GIT, "add", "-A"], cwd=work, check=True)
        subprocess.run([*GIT, "commit", "-q", "--allow-empty", "-m", "release"], cwd=work, check=True)
        subprocess.run([*GIT, "push", "-q", str(bare), "HEAD:main"], cwd=work, check=True, capture_output=True)
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=work, capture_output=True, text=True, check=True).stdout.strip()

    push()
    return f"git+file://{bare}", push


def post_json(url):
    with urllib.request.urlopen(urllib.request.Request(url, data=b"{}", method="POST"), timeout=600) as r:
        return json.loads(r.read())


@pytest.mark.skipif(shutil.which("uv") is None, reason="needs uv")
def test_update_installs_the_newest_commit_and_the_restarted_daemon_runs_it(tmp_path, env, forge):
    source, push = forge
    env = {**env, "UV_TOOL_DIR": str(tmp_path / "tools"), "UV_TOOL_BIN_DIR": str(tmp_path / "bin"),
           "TREX_SOURCE": source, "INVOCATION_ID": "systemd"}
    subprocess.run(["uv", "tool", "install", source], env=env, check=True, capture_output=True, timeout=600)
    daemon = [str(tmp_path / "tools" / "trex" / "bin" / "python"), "-m", "trex", "daemon", "--port", str(free_port())]
    api = f"http://127.0.0.1:{daemon[-1]}/api/daemon"

    def start_daemon():
        p = subprocess.Popen(daemon, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        assert p.stdout and "http://" in p.stdout.readline()
        return p

    p = start_daemon()
    try:
        info = get_json(api)
        assert info["updates"] == {"source": source, "available": True, "reason": ""}
        new = push()
        body = post_json(f"{api}/update")
        assert body["updated"] and body["to"]["commit"] == new != body["from"]["commit"] == info["install"]["commit"]
        assert p.wait(timeout=60) == 75
        p = start_daemon()
        assert get_json(api)["install"]["commit"] == new
        assert post_json(f"{api}/update")["updated"] is False
    finally:
        if p.poll() is None:
            assert interrupt(p) == 0
