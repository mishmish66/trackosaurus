"""Updating the daemon's trex from $TREX_SOURCE: `uv tool install $TREX_SOURCE`, then systemd, launchd or runit
restarts it."""

import functools
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from .format import JSONValue, as_dict, as_str

DEFAULT_SOURCE: Final = "git+https://github.com/mishmish66/trackosaurus"
INSTALL_TIMEOUT: Final = 600.0  # seconds
RESTART_STATUS: Final = 75  # exit status asking the service manager to start the daemon again (systemd's RestartForceExitStatus)


@dataclass(frozen=True, slots=True)
class Install:
    """A trex installed: its version; for a git source, its commit and the source as `uv tool install` takes it (the
    URL, at the revision asked for)."""

    version: str
    commit: str | None
    source: str | None = None

    def wire(self) -> dict[str, str | None]:
        return {"version": self.version, "commit": self.commit}

    @classmethod
    def read(cls, v: JSONValue) -> "Install":
        d = as_dict(v)
        return cls(as_str(d.get("version")) or "unknown", as_str(d.get("commit")))


@dataclass(frozen=True, slots=True)
class Updates:
    """Whether this process can update itself, from $TREX_SOURCE (`source`), and why not when it cannot."""

    source: str | None
    available: bool
    reason: str

    def wire(self) -> dict[str, Any]:
        return {"source": self.source, "available": self.available, "reason": self.reason}


class UpdateError(Exception):
    """`uv tool install` failed; the message is its output."""


def installed(prefix: Path | None = None) -> Install:
    """The trex installed in the environment at `prefix` (default: this one, or wherever this process imports it
    from), read fresh from disk."""
    found = next(iter(sorted((prefix or Path(sys.prefix)).glob("lib/python*/site-packages/trex-*.dist-info"))), None)
    if found is None and prefix is None:
        try:
            dist = importlib.metadata.distribution("trex")
        except importlib.metadata.PackageNotFoundError:
            return Install("unknown", None)
        version, direct = dist.version, dist.read_text("direct_url.json")
    elif found is None:
        return Install("unknown", None)
    else:
        version = found.name.removeprefix("trex-").removesuffix(".dist-info")
        direct = (found / "direct_url.json").read_text() if (found / "direct_url.json").is_file() else None
    info = as_dict(json.loads(direct)) if direct else {}
    vcs = as_dict(info.get("vcs_info"))
    kind, url, rev = as_str(vcs.get("vcs")), as_str(info.get("url")), as_str(vcs.get("requested_revision"))
    source = f"{kind}+{url}" + (f"@{rev}" if rev else "") if kind and url else None
    return Install(version, as_str(vcs.get("commit_id")), source)


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
    """Whether this process can update itself: it needs $TREX_SOURCE, systemd, launchd or runit to restart it, and to
    run the trex that `uv tool install` replaces."""
    source = env.get("TREX_SOURCE") or None
    prefix = (prefix or Path(sys.prefix)).resolve()
    if source is None:
        reason = "TREX_SOURCE is not set"
    elif not service(env):
        reason = "not running as a service (see `trex systemd-unit`, `trex launchd-plist`, `trex runit-service`)"
    elif tool_env() != prefix:
        reason = f"this trex ({prefix}) is not the uv tool install ({tool_env()})"
    else:
        reason = ""
    return Updates(source, not reason, reason)


def service(env: Mapping[str, str]) -> bool:
    """Whether systemd, launchd (`trex launchd-plist`) or runit (`trex runit-service`) runs this process."""
    return "INVOCATION_ID" in env or env.get("TREX_SERVICE") in ("launchd", "runit")


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
