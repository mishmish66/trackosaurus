"""Remote runs directories (`host:path`): the daemon copies this trex there as a wheel over ssh (once per content),
runs it with uvx on a Unix socket that ssh forwards back, and passes the directory's requests through to it."""

import base64
import collections
import functools
import hashlib
import http.client
import importlib.metadata
import io
import os
import re
import shlex
import socket
import subprocess
import tempfile
import threading
import uuid
import zipfile
from pathlib import Path
from typing import Final, NamedTuple, Self

SPEC: Final = re.compile(r"(?P<host>(?:[^@/:\s]+@)?(?:\[[^\]\s]+\]|[^@/:\s\[\]]+)):(?P<path>.+)")
ADD_TIMEOUT: Final = 120.0  # seconds an add waits for the first start (uvx may install trex first)
BACKOFF_MAX: Final = 60.0  # seconds between reconnection attempts, at most
CLOSE_TIMEOUT: Final = 15.0  # seconds `close` waits for the session to end
REMOTE_PYTHON: Final = "3.12"  # the oldest Python trex supports, whose numpy wheels reach the oldest glibc
PROXY_TIMEOUT: Final = 120.0  # seconds a passed-through request may wait for the remote server
SSH_OPTIONS: Final = ["-o", "BatchMode=yes", "-o", "ExitOnForwardFailure=yes", "-o", "StreamLocalBindUnlink=yes",
                      "-o", "StreamLocalBindMask=0177", "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3"]
NO_UV: Final = "uv is not installed on $(hostname); install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"
WHEELS: Final = ".cache/trex/wheels"  # where a remote home keeps the copies of trex it was sent, relative to it
NEEDS_WHEEL: Final = "trex needs its wheel"  # a session's first line when the remote lacks this trex's wheel


class Address(NamedTuple):
    host: str  # [user@]host, as ssh takes it
    path: str  # as typed; ~ is the remote home


def parse(spec: str) -> Address | None:
    """The remote address `spec` names in scp form ([user@]host:path), or None for a local path."""
    m = SPEC.fullmatch(spec.strip())
    if m is None or Path(spec).expanduser().exists():
        return None
    return Address(m["host"], m["path"])


def build_wheel(package: Path) -> tuple[str, bytes]:
    """The trex package directory `package` as a pure-Python wheel: its file name, whose version carries a digest of
    the wheel's contents, and its bytes (the same for the same contents)."""
    files = sorted((f.relative_to(package.parent).as_posix(), f.read_bytes()) for f in package.rglob("*")
                   if f.is_file() and "__pycache__" not in f.parts and f.suffix != ".pyc")
    try:
        meta = importlib.metadata.metadata("trex")
        base, python, needs = meta["Version"], meta["Requires-Python"], importlib.metadata.requires("trex") or []
    except importlib.metadata.PackageNotFoundError:
        base, python, needs = "0", ">=3.12", ["numpy>=1.24", "typer>=0.27.2"]
    digest = hashlib.sha256()
    for name, data in [*files, ("requires", "\n".join([python, *needs]).encode())]:
        digest.update(name.encode() + b"\0" + hashlib.sha256(data).digest())
    version = f"{base}+{digest.hexdigest()[:16]}"
    info = f"trex-{version}.dist-info"
    files += [(f"{info}/METADATA", "".join([f"Metadata-Version: 2.1\nName: trex\nVersion: {version}\nRequires-Python: {python}\n",
                                             *(f"Requires-Dist: {r}\n" for r in needs)]).encode()),
              (f"{info}/WHEEL", b"Wheel-Version: 1.0\nGenerator: trex\nRoot-Is-Purelib: true\nTag: py3-none-any\n"),
              (f"{info}/entry_points.txt", b"[console_scripts]\ntrex = trex.cli:main\n")]
    record = [f"{n},sha256={base64.urlsafe_b64encode(hashlib.sha256(d).digest()).rstrip(b'=').decode()},{len(d)}" for n, d in files]
    files.append((f"{info}/RECORD", "\n".join([*record, f"{info}/RECORD,,", ""]).encode()))
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in files:
            entry = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            entry.external_attr = 0o644 << 16
            z.writestr(entry, data, zipfile.ZIP_DEFLATED)
    return f"trex-{version}-py3-none-any.whl", out.getvalue()


@functools.cache
def wheel() -> tuple[str, bytes]:
    """This trex as remote machines run it (`build_wheel` of this package)."""
    return build_wheel(Path(__file__).parent)


def remote_command(addr: Address, sock: str, wheel_name: str) -> str:
    """The shell command that serves `addr.path` on the Unix socket `sock`, until its stdin closes, from the wheel
    `wheel_name` in WHEELS (or prints NEEDS_WHEEL and ends when that is missing), on Python REMOTE_PYTHON with numpy
    from a wheel (the newest one for the host's glibc, as old cluster systems need), its index on the host's local disk
    (never shared by two hosts, as a network home would be) and at most 4 index workers unless $TREX_WORKERS says
    otherwise there (remote hosts are often shared)."""
    whl = f'"$HOME/{WHEELS}/{wheel_name}"'
    serve = (f"uvx --python {REMOTE_PYTHON} --no-build-package numpy --from {whl} trex serve "
             + shlex.join([addr.path, "--standalone", "--unix", sock, "--exit-on-eof"]))
    script = (f'PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"; command -v uvx >/dev/null || '
              f'{{ echo "{NO_UV}" >&2; exit 127; }}; test -f {whl} || {{ echo "{NEEDS_WHEEL}"; exit 0; }}; '
              f'NO_COLOR=1 PYTHONUNBUFFERED=1 TREX_WORKERS="${{TREX_WORKERS:-4}}" exec {serve} --cache "${{TMPDIR:-/tmp}}/trex-cache-$(id -u)"')
    return shlex.join(["sh", "-c", script])


def upload_command(wheel_name: str) -> str:
    """The shell command that writes its stdin to WHEELS/`wheel_name` (a temporary name, then renamed)."""
    script = (f'd="$HOME/{WHEELS}"; mkdir -p "$d" && t="$d/.{wheel_name}.$$" && cat > "$t" && mv -f "$t" "$d/{wheel_name}" '
              f'|| {{ rm -f "$t"; exit 1; }}')
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

    def _attempt(self, send: bool = True) -> bool:
        """One ssh session, after sending this trex's wheel when the remote lacks it (if `send`); whether it
        connected."""
        sock = f"/tmp/trex-{uuid.uuid4().hex[:16]}.sock"
        cmd = [*ssh(), "-L", f"{self.local}:{sock}", self.addr.host, remote_command(self.addr, sock, wheel()[0])]
        self.state, self.error = "starting", ""
        try:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        except OSError as e:
            self.state, self.error = "unreachable", str(e)
            return False
        self._proc = proc
        if self._closed.is_set():
            _end(proc)
        assert proc.stdout is not None and proc.stderr is not None
        errors: collections.deque[str] = collections.deque(maxlen=20)
        drain = threading.Thread(target=lambda: errors.extend(line.rstrip() for line in proc.stderr or []), daemon=True)
        drain.start()
        first = proc.stdout.readline()
        connected = "trex serving" in first
        if connected and not self._closed.is_set():
            self.state = "connected"
            self.settled.set()
        proc.wait()
        drain.join(5)
        _end(proc)
        proc.stdout.close()
        proc.stderr.close()
        self._proc = None
        if first.strip() == NEEDS_WHEEL and send and not self._closed.is_set():
            sent = self._send()
            if sent is None:
                return self._attempt(send=False)
            errors.append(sent)
        if not self._closed.is_set():
            self.state = "unreachable"
            self.error = "\n".join(errors) or f"ssh exited with status {proc.returncode}"
        return connected

    def _send(self) -> str | None:
        """Copy this trex's wheel to the remote's WHEELS over ssh; None when it arrived, else why not."""
        name, data = wheel()
        cmd = [*ssh(), self.addr.host, upload_command(name)]
        try:
            done = subprocess.run(cmd, input=data, capture_output=True, timeout=ADD_TIMEOUT)
        except (OSError, subprocess.TimeoutExpired) as e:
            return f"sending trex: {e}"
        return None if done.returncode == 0 else f"sending trex: {done.stderr.decode(errors='replace').strip() or done.returncode}"


def ssh() -> list[str]:
    """The ssh command ($TREX_SSH, else ssh) and its options."""
    return [os.environ.get("TREX_SSH", "ssh"), *SSH_OPTIONS]


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
