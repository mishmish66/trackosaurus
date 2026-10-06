import importlib.metadata
import json
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import NoReturn

import pytest

from trex import server, update
from trex.format import JSONValue
from trex.node import Node

from helpers import get_json, post_json, wait_for


def fake_env(prefix: Path, version: str = "0.1.0", direct: dict[str, JSONValue] | None = None) -> Path:
    dist = prefix / "lib" / "python3.12" / "site-packages" / f"trex-{version}.dist-info"
    dist.mkdir(parents=True)
    if direct is not None:
        (dist / "direct_url.json").write_text(json.dumps(direct))
    return prefix


def test_installed_version_commit_and_source_come_from_the_environment(tmp_path: Path) -> None:
    git = fake_env(tmp_path / "git", "0.2.0", {"url": "https://x", "vcs_info": {"vcs": "git", "commit_id": "abc123"}})
    branch = fake_env(tmp_path / "branch", direct={"url": "https://x/trex.git",
                                                   "vcs_info": {"vcs": "git", "commit_id": "def456", "requested_revision": "mesh"}})
    editable = fake_env(tmp_path / "editable", direct={"url": "file:///src", "dir_info": {"editable": True}})
    assert update.installed(git) == update.Install(version="0.2.0", commit="abc123", source="git+https://x")
    assert update.installed(branch) == update.Install(version="0.1.0", commit="def456", source="git+https://x/trex.git@mesh")
    assert update.installed(editable) == update.Install(version="0.1.0", commit=None)
    assert update.installed(tmp_path / "empty") == update.Install(version="unknown", commit=None)


def test_updates_need_a_source_a_service_manager_and_the_uv_tool_install(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tool = tmp_path / "tools" / "trex"
    tool.mkdir(parents=True)
    monkeypatch.setattr(update, "tool_env", lambda: tool.resolve())
    systemd = {"TREX_SOURCE": "git+https://x", "INVOCATION_ID": "1"}
    assert update.updates(systemd, tool) == update.Updates(source="git+https://x", available=True, reason="")
    assert "TREX_SOURCE" in update.updates({"INVOCATION_ID": "1"}, tool).reason
    assert "systemd" in update.updates({"TREX_SOURCE": "git+https://x"}, tool).reason
    assert "not the uv tool install" in update.updates(systemd, tmp_path).reason
    for manager in ("launchd", "runit"):
        env = {"TREX_SOURCE": "git+https://x", "TREX_SERVICE": manager}
        assert update.updates(env, tool) == update.Updates(source="git+https://x", available=True, reason="")


def test_install_failure_carries_uvs_output(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake = tmp_path / "uv"
    fake.write_text("#!/bin/sh\necho 'error: repository not found' >&2\nexit 2\n")
    fake.chmod(0o755)
    monkeypatch.setattr(update, "uv", lambda: str(fake))
    with pytest.raises(update.UpdateError, match="repository not found"):
        update.install("git+https://nowhere")


@pytest.fixture
def daemon_http(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                http_server: Callable[[server.Server], str]) -> tuple[str, list[float]]:
    """(url, restarts) of an in-process daemon server whose updates install from a fake source."""
    srv = server.serve(Node(tmp_path / "cache", tmp_path / "state" / "roots.json"), "127.0.0.1", 0)
    restarts: list[float] = []
    srv.restart = lambda: restarts.append(time.time())
    monkeypatch.setattr(update, "updates", lambda: update.Updates(source="git+https://x", available=True, reason=""))
    return http_server(srv), restarts


def test_an_update_that_installs_a_new_commit_restarts_the_daemon(daemon_http: tuple[str, list[float]],
                                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    url, restarts = daemon_http
    commits = iter(["old", "new"])
    monkeypatch.setattr(update, "installed", lambda: update.Install(version="0.1.0", commit=next(commits)))

    def install(source: str) -> str:
        return f"installed {source}"

    monkeypatch.setattr(update, "install", install)
    status, body = post_json(f"{url}/api/node/update")
    assert status == 200 and body["updated"] and (body["from"]["commit"], body["to"]["commit"]) == ("old", "new")
    assert wait_for(lambda: len(restarts) == 1)


def test_the_daemon_reports_the_trex_it_runs_not_one_installed_since(daemon_http: tuple[str, list[float]],
                                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    url, _ = daemon_http
    commits = iter(["old", "new"])
    monkeypatch.setattr(update, "installed", lambda: update.Install(version="0.1.0", commit=next(commits)))

    def install(source: str) -> str:
        return "installed"

    monkeypatch.setattr(update, "install", install)
    running = get_json(f"{url}/api/node")["install"]
    assert post_json(f"{url}/api/node/update")[1]["to"]["commit"] == "new"
    assert get_json(f"{url}/api/node")["install"] == running == update.RUNNING.wire()


def test_an_update_that_changes_nothing_keeps_the_daemon_running(daemon_http: tuple[str, list[float]],
                                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    url, restarts = daemon_http
    monkeypatch.setattr(update, "installed", lambda: update.Install(version="0.1.0", commit="same"))

    def install(source: str) -> str:
        return "already installed"

    monkeypatch.setattr(update, "install", install)
    assert post_json(f"{url}/api/node/update") == (200, {"updated": False, "from": {"version": "0.1.0", "commit": "same"},
                                                       "to": {"version": "0.1.0", "commit": "same"}, "output": "already installed"})
    assert post_json(f"{url}/api/node/update")[0] == 200
    time.sleep(server.RESTART_DELAY + 0.2)
    assert restarts == []


def test_a_failed_update_reports_uvs_output_and_can_be_retried(daemon_http: tuple[str, list[float]],
                                                               monkeypatch: pytest.MonkeyPatch) -> None:
    url, restarts = daemon_http
    monkeypatch.setattr(update, "installed", lambda: update.Install(version="0.1.0", commit="same"))

    def fail(source: str) -> NoReturn:
        raise update.UpdateError("fatal: could not read from remote")

    monkeypatch.setattr(update, "install", fail)
    assert post_json(f"{url}/api/node/update") == (502, {"error": "fatal: could not read from remote"})
    assert post_json(f"{url}/api/node/update")[0] == 502 and restarts == []


def test_a_second_update_during_the_first_is_refused(daemon_http: tuple[str, list[float]],
                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    url, _ = daemon_http
    started, release = threading.Event(), threading.Event()

    def install(source: str) -> str:
        started.set()
        release.wait(10)
        return "done"

    monkeypatch.setattr(update, "installed", lambda: update.Install(version="0.1.0", commit="same"))
    monkeypatch.setattr(update, "install", install)
    first = threading.Thread(target=post_json, args=(f"{url}/api/node/update",))
    first.start()
    assert started.wait(10)
    assert post_json(f"{url}/api/node/update")[0] == 409
    release.set()
    first.join()


def test_unavailable_updates_are_refused_with_the_reason(daemon_http: tuple[str, list[float]],
                                                        monkeypatch: pytest.MonkeyPatch) -> None:
    url, restarts = daemon_http
    monkeypatch.setattr(update, "updates", lambda: update.Updates(source=None, available=False, reason="TREX_SOURCE is not set"))
    assert post_json(f"{url}/api/node/update") == (400, {"error": "updates are unavailable: TREX_SOURCE is not set"})
    assert restarts == []


def test_a_server_without_a_restart_hook_has_no_update(tmp_path: Path, http_server: Callable[[server.Server], str]) -> None:
    url = http_server(server.serve(Node(tmp_path / "cache", tmp_path / "state" / "roots.json"), "127.0.0.1", 0))
    assert post_json(f"{url}/api/node/update")[0] == 404


@pytest.mark.skipif(not Path(update.uv()).exists(), reason="needs uv")
def test_tool_env_is_a_plain_path_even_when_color_is_forced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FORCE_COLOR", "3")
    monkeypatch.setenv("UV_TOOL_DIR", str(tmp_path / "tools"))
    update.tool_env.cache_clear()
    try:
        assert update.tool_env() == (tmp_path / "tools").resolve() / "trex"
    finally:
        update.tool_env.cache_clear()


def test_this_processes_trex_is_found_where_it_is_imported_from_when_its_prefix_has_none(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "prefix", str(tmp_path))
    found = update.installed()
    assert found.version != "unknown" and found.version == importlib.metadata.version("trex")
