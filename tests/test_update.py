import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from trex import server, update
from trex.daemon import Roots


def fake_env(prefix, version="0.1.0", direct=None):
    dist = prefix / "lib" / "python3.12" / "site-packages" / f"trex-{version}.dist-info"
    dist.mkdir(parents=True)
    if direct is not None:
        (dist / "direct_url.json").write_text(json.dumps(direct))
    return prefix


def test_installed_version_and_commit_come_from_the_environment(tmp_path):
    git = fake_env(tmp_path / "git", "0.2.0", {"url": "https://x", "vcs_info": {"vcs": "git", "commit_id": "abc123"}})
    editable = fake_env(tmp_path / "editable", direct={"url": "file:///src", "dir_info": {"editable": True}})
    assert update.installed(git) == {"version": "0.2.0", "commit": "abc123"}
    assert update.installed(editable) == {"version": "0.1.0", "commit": None}
    assert update.installed(tmp_path / "empty") == {"version": "unknown", "commit": None}


def test_updates_need_a_source_systemd_and_the_uv_tool_install(tmp_path, monkeypatch):
    tool = tmp_path / "tools" / "trex"
    tool.mkdir(parents=True)
    monkeypatch.setattr(update, "tool_env", lambda: tool.resolve())
    systemd = {"TREX_SOURCE": "git+https://x", "INVOCATION_ID": "1"}
    assert update.updates(systemd, tool) == {"source": "git+https://x", "available": True, "reason": ""}
    assert "TREX_SOURCE" in update.updates({"INVOCATION_ID": "1"}, tool)["reason"]
    assert "systemd" in update.updates({"TREX_SOURCE": "git+https://x"}, tool)["reason"]
    assert "not the uv tool install" in update.updates(systemd, tmp_path)["reason"]


def test_install_failure_carries_uvs_output(monkeypatch, tmp_path):
    fake = tmp_path / "uv"
    fake.write_text("#!/bin/sh\necho 'error: repository not found' >&2\nexit 2\n")
    fake.chmod(0o755)
    monkeypatch.setattr(update, "uv", lambda: str(fake))
    with pytest.raises(update.UpdateError, match="repository not found"):
        update.install("git+https://nowhere")


@pytest.fixture
def daemon_http(tmp_path, monkeypatch):
    """(url, restarts) of an in-process daemon server whose updates install from a fake source."""
    srv = server.serve(None, "127.0.0.1", 0, Roots(tmp_path / "cache", tmp_path / "state" / "roots.json"))
    restarts = []
    srv.restart = lambda: restarts.append(time.time())
    monkeypatch.setattr(update, "updates", lambda: {"source": "git+https://x", "available": True, "reason": ""})
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", restarts
    srv.shutdown()


def post(url):
    try:
        with urllib.request.urlopen(urllib.request.Request(url, data=b"{}", method="POST")) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def wait_for(cond, timeout=5.0):
    end = time.time() + timeout
    while not cond() and time.time() < end:
        time.sleep(0.02)
    return cond()


def test_an_update_that_installs_a_new_commit_restarts_the_daemon(daemon_http, monkeypatch):
    url, restarts = daemon_http
    commits = iter(["old", "new"])
    monkeypatch.setattr(update, "installed", lambda: {"version": "0.1.0", "commit": next(commits)})
    monkeypatch.setattr(update, "install", lambda source: f"installed {source}")
    status, body = post(f"{url}/api/daemon/update")
    assert status == 200 and body["updated"] and (body["from"]["commit"], body["to"]["commit"]) == ("old", "new")
    assert wait_for(lambda: len(restarts) == 1)


def test_the_daemon_reports_the_trex_it_runs_not_one_installed_since(daemon_http, monkeypatch):
    url, _ = daemon_http
    commits = iter(["old", "new"])
    monkeypatch.setattr(update, "installed", lambda: {"version": "0.1.0", "commit": next(commits)})
    monkeypatch.setattr(update, "install", lambda source: "installed")
    with urllib.request.urlopen(f"{url}/api/daemon") as r:
        running = json.loads(r.read())["install"]
    assert post(f"{url}/api/daemon/update")[1]["to"]["commit"] == "new"
    with urllib.request.urlopen(f"{url}/api/daemon") as r:
        assert json.loads(r.read())["install"] == running == update.RUNNING


def test_an_update_that_changes_nothing_keeps_the_daemon_running(daemon_http, monkeypatch):
    url, restarts = daemon_http
    monkeypatch.setattr(update, "installed", lambda: {"version": "0.1.0", "commit": "same"})
    monkeypatch.setattr(update, "install", lambda source: "already installed")
    assert post(f"{url}/api/daemon/update") == (200, {"updated": False, "from": {"version": "0.1.0", "commit": "same"},
                                                       "to": {"version": "0.1.0", "commit": "same"}, "output": "already installed"})
    assert post(f"{url}/api/daemon/update")[0] == 200
    time.sleep(server.RESTART_DELAY + 0.2)
    assert restarts == []


def test_a_failed_update_reports_uvs_output_and_can_be_retried(daemon_http, monkeypatch):
    url, restarts = daemon_http
    monkeypatch.setattr(update, "installed", lambda: {"version": "0.1.0", "commit": "same"})

    def fail(source):
        raise update.UpdateError("fatal: could not read from remote")

    monkeypatch.setattr(update, "install", fail)
    assert post(f"{url}/api/daemon/update") == (502, {"error": "fatal: could not read from remote"})
    assert post(f"{url}/api/daemon/update")[0] == 502 and restarts == []


def test_a_second_update_during_the_first_is_refused(daemon_http, monkeypatch):
    url, _ = daemon_http
    release = threading.Event()
    monkeypatch.setattr(update, "installed", lambda: {"version": "0.1.0", "commit": "same"})
    monkeypatch.setattr(update, "install", lambda source: release.wait(10) and "done")
    first = threading.Thread(target=post, args=(f"{url}/api/daemon/update",))
    first.start()
    time.sleep(0.2)
    assert post(f"{url}/api/daemon/update")[0] == 409
    release.set()
    first.join()


def test_unavailable_updates_are_refused_with_the_reason(daemon_http, monkeypatch):
    url, restarts = daemon_http
    monkeypatch.setattr(update, "updates", lambda: {"source": None, "available": False, "reason": "TREX_SOURCE is not set"})
    assert post(f"{url}/api/daemon/update") == (400, {"error": "updates are unavailable: TREX_SOURCE is not set"})
    assert restarts == []


def test_a_server_without_a_restart_hook_has_no_update(tmp_path):
    srv = server.serve(None, "127.0.0.1", 0, Roots(tmp_path / "cache", tmp_path / "state" / "roots.json"))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        assert post(f"http://127.0.0.1:{srv.server_address[1]}/api/daemon/update")[0] == 404
    finally:
        srv.shutdown()


@pytest.mark.skipif(not Path(update.uv()).exists(), reason="needs uv")
def test_tool_env_is_a_plain_path_even_when_color_is_forced(tmp_path, monkeypatch):
    monkeypatch.setenv("FORCE_COLOR", "3")
    monkeypatch.setenv("UV_TOOL_DIR", str(tmp_path / "tools"))
    update.tool_env.cache_clear()
    try:
        assert update.tool_env() == (tmp_path / "tools").resolve() / "trex"
    finally:
        update.tool_env.cache_clear()
