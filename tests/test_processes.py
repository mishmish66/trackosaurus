"""`trex serve` and `trex daemon` as real processes, each on a free port with a private state directory."""

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from trex.cli import RUNIT_AS_USER

from helpers import get_json, post_json, wait_for, write_run

type Forge = tuple[str, Callable[[], str]]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def env(tmp_path: Path) -> dict[str, str]:
    return {**os.environ, "TREX_DAEMON_DIR": str(tmp_path / "state"), "TREX_CACHE": str(tmp_path / "cache"),
            "PYTHON_COLORS": "0", "NO_COLOR": "1"}


def trex_cmd(*argv: str | int | Path) -> list[str]:
    return [sys.executable, "-m", "trex", *map(str, argv)]


def start(env: dict[str, str], *argv: str | int | Path) -> subprocess.Popen[str]:
    """A trex server process, once it has printed its address."""
    p = subprocess.Popen(trex_cmd(*argv), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert p.stdout and p.stderr
    line = p.stdout.readline()
    assert "http://" in line, p.stderr.read()
    return p


def interrupt(p: subprocess.Popen[str], sig: signal.Signals = signal.SIGINT) -> int:
    """Exit code after `sig`."""
    p.send_signal(sig)
    return p.wait(timeout=20)


def test_a_temporary_trex_answers_until_interrupted(tmp_path: Path, env: dict[str, str]) -> None:
    write_run(tmp_path / "runs" / "r")
    port = free_port()
    p = start(env, "serve", tmp_path / "runs", "--temporary", "--port", port)
    try:
        assert get_json(f"http://127.0.0.1:{port}/api/info")["root"] == str(tmp_path / "runs")
    finally:
        assert interrupt(p) == 0


def test_the_machines_trex_takes_directories_from_serve_and_removes_its_socket_on_exit(tmp_path: Path, env: dict[str, str]) -> None:
    a, b = tmp_path / "a" / "runs", tmp_path / "b" / "logs"
    write_run(a / "r")
    write_run(b / "r")
    port = free_port()
    p = start(env, "serve", a, "--port", port)
    try:
        added = subprocess.run(trex_cmd("serve", b), env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=60)
        dirs = get_json(f"http://127.0.0.1:{port}/api/node")["dirs"]
        assert added.returncode == 0 and f"http://127.0.0.1:{port}{dirs[1]['url']}" in added.stdout
        assert [r["root"] for r in dirs] == [str(a), str(b)]
        again = subprocess.run(trex_cmd("serve", "--port", free_port()), env=env, capture_output=True, text=True, timeout=60)
        assert again.returncode == 0 and f"already running at http://127.0.0.1:{port}/" in again.stdout
    finally:
        assert interrupt(p) == 0
    assert not (tmp_path / "state" / "daemon.sock").exists()
    assert json.loads((tmp_path / "state" / "roots.json").read_text())["tracked"] == [str(a), str(b)]


def test_a_runit_run_script_starts_this_machines_trex_as_a_service(tmp_path: Path, env: dict[str, str]) -> None:
    port = free_port()
    script = subprocess.run(trex_cmd("runit-service", "--port", port, "--source", "git+https://x"), env=env, capture_output=True,
                            text=True, check=True).stdout
    (tmp_path / "run").write_text(script.replace(f"exec {RUNIT_AS_USER} ", "exec "))  # changing the user takes root
    p = subprocess.Popen(["sh", str(tmp_path / "run")], env=env, stdout=subprocess.PIPE, text=True)
    try:
        assert p.stdout and any("http://" in line for line in p.stdout)
        updates = get_json(f"http://127.0.0.1:{port}/api/node")["updates"]
        assert updates["source"] == "git+https://x" and "not running as a service" not in updates["reason"]
    finally:
        assert interrupt(p) == 0


def test_a_restarted_trex_serves_the_directories_it_had(tmp_path: Path, env: dict[str, str]) -> None:
    write_run(tmp_path / "runs" / "r")
    port = free_port()
    assert interrupt(start(env, "serve", tmp_path / "runs", "--port", port)) == 0
    p = start(env, "serve", "--port", port)
    try:
        assert [r["name"] for r in get_json(f"http://127.0.0.1:{port}/api/node")["dirs"]] == ["runs"]
    finally:
        assert interrupt(p) == 0


def test_output_to_a_closed_pipe_ends_without_a_traceback(tmp_path: Path, env: dict[str, str]) -> None:
    write_run(tmp_path / "r", 20000, metrics=lambda i: {"x": float(i)})
    p = subprocess.Popen(trex_cmd("series", tmp_path / "r", "-k", "x"), env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True)
    assert p.stdout and p.stderr
    p.stdout.readline()
    p.stdout.close()
    assert "Traceback" not in p.stderr.read()
    p.wait(timeout=20)


def test_a_terminated_trex_exits_cleanly_and_removes_its_socket(tmp_path: Path, env: dict[str, str]) -> None:
    p = start(env, "serve", "--port", free_port())
    assert (tmp_path / "state" / "daemon.sock").exists()
    assert interrupt(p, signal.SIGTERM) == 0
    assert not (tmp_path / "state" / "daemon.sock").exists()


GIT = ["git", "-c", "user.name=trex", "-c", "user.email=trex@localhost", "-c", "init.defaultBranch=main"]


WORKER = """
import os, time
from concurrent.futures import ProcessPoolExecutor
from trex.index import mp_context, worker_init
pool = ProcessPoolExecutor(1, mp_context=mp_context(), initializer=worker_init)
print(pool.submit(os.getpid).result(), flush=True)
time.sleep(120)
"""


def running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_a_worker_ends_when_the_process_that_started_it_is_killed() -> None:
    p = subprocess.Popen([sys.executable, "-c", WORKER], stdout=subprocess.PIPE, text=True)
    try:
        assert p.stdout
        worker = int(p.stdout.readline())
        assert running(worker)
    finally:
        p.kill()
        p.wait()
    assert wait_for(lambda: not running(worker))


@pytest.fixture
def forge(tmp_path: Path) -> Forge:
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

    def push() -> str:
        subprocess.run([*GIT, "init", "-q"], cwd=work, check=True)
        subprocess.run([*GIT, "add", "-A"], cwd=work, check=True)
        subprocess.run([*GIT, "commit", "-q", "--allow-empty", "-m", "release"], cwd=work, check=True)
        subprocess.run([*GIT, "push", "-q", str(bare), "HEAD:main"], cwd=work, check=True, capture_output=True)
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=work, capture_output=True, text=True, check=True).stdout.strip()

    push()
    return f"git+file://{bare}", push


@pytest.mark.skipif(shutil.which("uv") is None, reason="needs uv")
def test_update_installs_the_newest_commit_and_the_restarted_trex_runs_it(tmp_path: Path, env: dict[str, str], forge: Forge) -> None:
    source, push = forge
    env = {**env, "UV_TOOL_DIR": str(tmp_path / "tools"), "UV_TOOL_BIN_DIR": str(tmp_path / "bin"),
           "TREX_SOURCE": source, "INVOCATION_ID": "systemd"}
    subprocess.run(["uv", "tool", "install", source], env=env, check=True, capture_output=True, timeout=600)
    daemon = [str(tmp_path / "tools" / "trex" / "bin" / "python"), "-m", "trex", "serve", "--port", str(free_port())]
    api = f"http://127.0.0.1:{daemon[-1]}/api/node"

    def start_daemon() -> subprocess.Popen[str]:
        p = subprocess.Popen(daemon, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        assert p.stdout and "http://" in p.stdout.readline()
        return p

    p = start_daemon()
    try:
        info = get_json(api)
        assert info["updates"] == {"source": source, "available": True, "reason": ""}
        new = push()
        body = post_json(f"{api}/update", timeout=600)[1]
        assert body["updated"] and body["to"]["commit"] == new != body["from"]["commit"] == info["install"]["commit"]
        assert p.wait(timeout=60) == 75
        p = start_daemon()
        assert get_json(api)["install"]["commit"] == new
        assert post_json(f"{api}/update", timeout=600)[1]["updated"] is False
    finally:
        if p.poll() is None:
            assert interrupt(p) == 0
