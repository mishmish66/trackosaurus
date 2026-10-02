"""Remote runs directories (`host:path`): the daemon runs `trex serve` there over ssh with uvx, on a Unix socket
that ssh forwards back, and passes the directory's requests through to it."""

import collections
import http.client
import os
import re
import shlex
import socket
import subprocess
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Final, NamedTuple, Self

from . import update

SPEC: Final = re.compile(r"(?P<host>(?:[^@/:\s]+@)?(?:\[[^\]\s]+\]|[^@/:\s\[\]]+)):(?P<path>.+)")
ADD_TIMEOUT: Final = 120.0  # seconds an add waits for the first start (uvx may install trex first)
BACKOFF_MAX: Final = 60.0  # seconds between reconnection attempts, at most
CLOSE_TIMEOUT: Final = 15.0  # seconds `close` waits for the session to end
PROXY_TIMEOUT: Final = 120.0  # seconds a passed-through request may wait for the remote server
SSH_OPTIONS: Final = ["-o", "BatchMode=yes", "-o", "ExitOnForwardFailure=yes", "-o", "StreamLocalBindUnlink=yes",
                      "-o", "StreamLocalBindMask=0177", "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3"]
NO_UV: Final = "uv is not installed on $(hostname); install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"


class Address(NamedTuple):
    host: str  # [user@]host, as ssh takes it
    path: str  # as typed; ~ is the remote home


def parse(spec: str) -> Address | None:
    """The remote address `spec` names in scp form ([user@]host:path), or None for a local path."""
    m = SPEC.fullmatch(spec.strip())
    if m is None or Path(spec).expanduser().exists():
        return None
    return Address(m["host"], m["path"])


def source() -> str:
    """What remote machines run: $TREX_SOURCE (default the GitHub repository), pinned to this trex's commit."""
    base = os.environ.get("TREX_SOURCE") or update.DEFAULT_SOURCE
    commit = update.RUNNING["commit"]
    if not commit or not base.startswith("git+"):
        return base
    head, _, last = base.rpartition("/")
    return f"{head}/{last.split('@')[0]}@{commit}"


def remote_command(addr: Address, sock: str, src: str) -> str:
    """The shell command that serves `addr.path` on the Unix socket `sock`, until its stdin closes, with numpy from a
    wheel (the newest one for the host's glibc, as old cluster systems need), its index
    on the host's local disk (never shared by two hosts, as a network home would be) and at most 4 index workers
    unless $TREX_WORKERS says otherwise there (remote hosts are often shared)."""
    serve = shlex.join(["uvx", "--no-build-package", "numpy", "--from", src, "trex", "serve", addr.path, "--standalone", "--unix", sock,
                        "--exit-on-eof"])
    script = (f'PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"; command -v uvx >/dev/null || '
              f'{{ echo "{NO_UV}" >&2; exit 127; }}; NO_COLOR=1 PYTHONUNBUFFERED=1 TREX_WORKERS="${{TREX_WORKERS:-4}}" exec {serve} --cache "${{TMPDIR:-/tmp}}/trex-cache-$(id -u)"')
    return shlex.join(["sh", "-c", script])


class UnixHTTPConnection(http.client.HTTPConnection):
    def __init__(self, path: Path, timeout: float) -> None:
        super().__init__("localhost", timeout=timeout)
        self.socket_path = path

    def connect(self) -> None:
        s = socket.socket(socket.AF_UNIX)
        s.settimeout(self.timeout)
        s.connect(str(self.socket_path))
        self.sock = s


class Remote:
    """A remote runs directory served over ssh; reconnects with backoff until `close`."""

    def __init__(self, spec: str, addr: Address) -> None:
        self.spec = spec
        self.addr = addr
        self.local = Path(tempfile.gettempdir()) / f"trex-{os.getuid()}-{uuid.uuid4().hex[:12]}.sock"
        self.state = "starting"  # starting, connected or unreachable
        self.error = ""
        self.settled = threading.Event()  # set once the first start has connected or failed
        self._closed = threading.Event()
        self._proc: subprocess.Popen[str] | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> Self:
        self._thread = threading.Thread(target=self._supervise, name=f"trex-remote-{self.addr.host}", daemon=True)
        self._thread.start()
        return self

    def close(self) -> None:
        self._closed.set()
        proc = self._proc
        if proc is not None:
            _end(proc)
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(CLOSE_TIMEOUT)

    def connection(self) -> UnixHTTPConnection:
        return UnixHTTPConnection(self.local, PROXY_TIMEOUT)

    def _supervise(self) -> None:
        delay = 1.0
        while not self._closed.is_set():
            if self._attempt():
                delay = 1.0
            self.settled.set()
            if self._closed.wait(delay):
                break
            delay = min(2 * delay, BACKOFF_MAX)
        self.local.unlink(missing_ok=True)

    def _attempt(self) -> bool:
        """One ssh session; whether it connected."""
        sock = f"/tmp/trex-{uuid.uuid4().hex[:16]}.sock"
        ssh = os.environ.get("TREX_SSH", "ssh")
        cmd = [ssh, *SSH_OPTIONS, "-L", f"{self.local}:{sock}", self.addr.host, remote_command(self.addr, sock, source())]
        self.state, self.error = "starting", ""
        try:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        except OSError as e:
            self.state, self.error = "unreachable", str(e)
            return False
        self._proc = proc
        assert proc.stdout is not None and proc.stderr is not None
        errors: collections.deque[str] = collections.deque(maxlen=20)
        drain = threading.Thread(target=lambda: errors.extend(line.rstrip() for line in proc.stderr or []), daemon=True)
        drain.start()
        connected = "trex serving" in proc.stdout.readline()
        if connected and not self._closed.is_set():
            self.state = "connected"
            self.settled.set()
        proc.wait()
        drain.join(5)
        _end(proc)
        proc.stdout.close()
        proc.stderr.close()
        self._proc = None
        if not self._closed.is_set():
            self.state = "unreachable"
            self.error = "\n".join(errors) or f"ssh exited with status {proc.returncode}"
        return connected


def _end(proc: subprocess.Popen[str]) -> None:
    """Close the session's stdin, which ends the remote server, then end the ssh process."""
    try:
        if proc.stdin is not None:
            proc.stdin.close()
    except OSError:
        pass
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            proc.kill()
