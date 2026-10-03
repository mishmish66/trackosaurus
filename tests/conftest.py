import sys
import threading
from collections.abc import Callable, Iterator

import pytest

from trex import index
from trex.server import Server

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


@pytest.fixture(autouse=True)
def close_explorers(monkeypatch):
    """Close every Explorer a test opens."""
    opened = []
    init = index.Explorer.__init__

    def tracked(self, *a, **kw):
        init(self, *a, **kw)
        opened.append(self)

    monkeypatch.setattr(index.Explorer, "__init__", tracked)
    yield
    for ex in opened:
        if not ex._closed:
            ex.close()


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


@pytest.fixture
def http_server() -> Iterator[Callable[[Server], str]]:
    """Starts an unstarted server (from `server.serve`) on a thread and returns its base URL; every one is shut down
    at teardown."""
    started: list[Server] = []

    def start(srv: Server) -> str:
        threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
        started.append(srv)
        return f"http://127.0.0.1:{srv.server_address[1]}"

    yield start
    for srv in started:
        srv.shutdown()
        srv.server_close()
