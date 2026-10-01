"""Updating the daemon's trex from $TREX_SOURCE: `uv tool install $TREX_SOURCE`, then systemd restarts it."""

import functools
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Final, TypedDict

DEFAULT_SOURCE: Final = "git+https://github.com/mishmish66/trackosaurus"
INSTALL_TIMEOUT: Final = 600.0  # seconds
RESTART_STATUS: Final = 75  # exit status asking systemd to start the daemon again (the unit's RestartForceExitStatus)


class Install(TypedDict):
    version: str
    commit: str | None  # git commit installed, for a git source


class Updates(TypedDict):
    source: str | None  # $TREX_SOURCE
    available: bool
    reason: str  # why not, when unavailable


class UpdateError(Exception):
    """`uv tool install` failed; the message is its output."""


def installed(prefix: Path | None = None) -> Install:
    """The trex installed in the environment at `prefix` (default: this one), read fresh from disk."""
    prefix = prefix or Path(sys.prefix)
    dist = next(iter(sorted(prefix.glob("lib/python*/site-packages/trex-*.dist-info"))), None)
    commit = None
    if dist and (dist / "direct_url.json").is_file():
        commit = (json.loads((dist / "direct_url.json").read_text()).get("vcs_info") or {}).get("commit_id")
    return {"version": dist.name.removeprefix("trex-").removesuffix(".dist-info") if dist else "unknown", "commit": commit}


RUNNING: Final = installed()  # the trex this process runs, as installed when it started


def uv() -> str:
    """The uv executable: uv on PATH, else ~/.local/bin/uv."""
    return shutil.which("uv") or str(Path.home() / ".local/bin/uv")


@functools.cache
def tool_env() -> Path | None:
    """Where `uv tool install` puts trex, or None without uv."""
    try:
        out = subprocess.run([uv(), "--color", "never", "tool", "dir"], capture_output=True, text=True, timeout=30, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return Path(out.strip()).resolve() / "trex"


def updates(env: Mapping[str, str] = os.environ, prefix: Path | None = None) -> Updates:
    """Whether this process can update itself: it needs $TREX_SOURCE, systemd to restart it, and to run the trex
    that `uv tool install` replaces."""
    source = env.get("TREX_SOURCE") or None
    prefix = (prefix or Path(sys.prefix)).resolve()
    reason = ("TREX_SOURCE is not set" if source is None else
              "not running under systemd (see `trex systemd-unit`)" if "INVOCATION_ID" not in env else
              f"this trex ({prefix}) is not the uv tool install ({tool_env()})" if tool_env() != prefix else "")
    return {"source": source, "available": not reason, "reason": reason}


def install(source: str) -> str:
    """`uv tool install source`, fetching its newest version; uv's output."""
    try:
        res = subprocess.run([uv(), "--color", "never", "tool", "install", source], capture_output=True, text=True,
                             timeout=INSTALL_TIMEOUT)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise UpdateError(str(e)) from e
    out = (res.stdout + res.stderr).strip()
    if res.returncode:
        raise UpdateError(out)
    return out
