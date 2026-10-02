"""Logging API. A background thread commits logged rows every `commit_interval` seconds; writer
errors surface from `Run.finish`, never from logging."""

import atexit
import hashlib
import io
import json
import math
import os
import sqlite3
import sys
import threading
import time
import uuid
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from types import TracebackType
from typing import IO, Literal, NamedTuple, Protocol, Self, cast, runtime_checkable

from numpy.typing import ArrayLike

from . import chunks, journal
from .format import FORMAT, INFO_FILE, JSONValue, MediaKind, as_dict, as_float, as_str, connect_rw, key_names, row_count
from .media import encode_mp4, encode_png, sniff_image

__all__ = ["FinalState", "ImageInput", "MetricValue", "Metrics", "Run", "VideoInput", "folder_info", "init"]


class SupportsItem(Protocol):
    """A 0-d array or tensor, logged as `.item()`."""

    def item(self) -> object: ...


@runtime_checkable
class SupportsSavePNG(Protocol):
    """A PIL-like image."""

    def save(self, fp: IO[bytes], format: str) -> None: ...


type MetricValue = int | float | bool | SupportsItem
"""Anything else is skipped."""

type Metrics = Mapping[str, MetricValue | Metrics]
"""Nested mappings flatten to `a/b`."""

type ImageInput = str | os.PathLike[str] | bytes | bytearray | SupportsSavePNG | ArrayLike
"""Path, encoded bytes, PIL image, or HW/HWC/CHW array (uint8, or float in [0, 1])."""

type VideoInput = str | os.PathLike[str] | bytes | bytearray | ArrayLike
"""Path, mp4 bytes, or THW/THWC frames (needs ffmpeg)."""

type FinalState = Literal["finished", "failed"]


class _Row(NamedTuple):
    step: float
    time: float
    values: dict[str, float]


class _Media(NamedTuple):
    key: str
    step: float
    time: float
    kind: MediaKind
    ext: str
    data: bytes


def _flatten(d: Mapping[str, object], prefix: str = "", out: dict[str, object] | None = None) -> dict[str, object]:
    """`d` with nested mappings flattened to `a/b` keys."""
    out = {} if out is None else out
    for k, v in d.items():
        name = f"{prefix}{k}"
        if isinstance(v, Mapping):
            _flatten(cast(Mapping[str, object], v), f"{name}/", out)
        else:
            out[name] = v
    return out


def _number(v: object) -> int | float | None:
    """A metric value as a number (bools as 0/1, `.item()` unwrapped), or None."""
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, float)):
        return v
    item = getattr(v, "item", None)
    if item is None:
        return None
    try:
        v = item()
    except (ValueError, RuntimeError, TypeError):
        return None
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _jsonify(v: object) -> JSONValue:
    """JSON-safe copy: `.item()` unwrapped; non-finite numbers and other objects as strings."""
    if isinstance(v, Mapping):
        return {str(k): _jsonify(x) for k, x in cast(Mapping[object, object], v).items()}
    if isinstance(v, (list, tuple)):
        return [_jsonify(x) for x in cast(Sequence[object], v)]
    if v is None or isinstance(v, (str, bool)):
        return v
    n = _number(v)
    if n is None:
        return str(v)
    return n if math.isfinite(n) else str(n)


def folder_info(path: str | os.PathLike[str], info: Mapping[str, object] | None = None, **kv: object) -> dict[str, JSONValue]:
    """Merge notes into `<path>/trex_info.json`; `trex={"group_by": [...]}` sets the folder's default grouping."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    f = p / INFO_FILE
    cur: dict[str, JSONValue] = json.loads(f.read_text()) if f.exists() else {}
    new = _jsonify({**(info or {}), **kv})
    assert isinstance(new, dict)
    cur.update(new)
    tmp = f.with_name(f".{INFO_FILE}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(cur, indent=2))
    os.replace(tmp, f)
    return cur


def _opened(stored: dict[str, JSONValue], name: str, config: Mapping[str, object] | None, tags: list[str],
            info: Mapping[str, object] | None) -> dict[str, JSONValue]:
    """A run's metadata on (re)opening: a new id if it has none; config and tags replaced when given."""
    now = time.time()
    meta: dict[str, JSONValue] = dict(stored) if "id" in stored else {"id": uuid.uuid4().hex, "created": now, "format": FORMAT,
                                                                     "summary": {}}
    meta["name"], meta["state"], meta["heartbeat"] = name, "running", now
    meta["tags"] = list[JSONValue](tags) or stored.get("tags", [])
    meta["config"] = {k: _jsonify(v) for k, v in _flatten(config).items()} if config is not None else stored.get("config", {})
    meta["info"] = {**as_dict(stored.get("info")), **as_dict(_jsonify(info or {}))}
    return meta


class Run:
    """A run directory being logged (see `init`). Ends on `finish`, context exit or interpreter exit;
    an uncaught exception marks it failed. Media `step` defaults to the last logged row's."""

    dir: Path
    id: str
    name: str
    created: float
    """Unix time; runtimes count from it."""
    commit_interval: float

    def __init__(self, dir: str | os.PathLike[str], *, name: str | None = None, config: Mapping[str, object] | None = None,
                 tags: Iterable[str] = (), info: Mapping[str, object] | None = None, commit_interval: float = 1.0) -> None:
        self.dir = Path(dir)
        (self.dir / "media").mkdir(parents=True, exist_ok=True)
        self.commit_interval = commit_interval
        c = connect_rw(self.dir)
        stored: dict[str, JSONValue] = {k: json.loads(v) for k, v in c.execute("SELECT key, value FROM meta")}
        meta = _opened(stored, name or as_str(stored.get("name")) or self.dir.name, config, list(tags), info)
        c.execute("BEGIN")
        c.executemany("INSERT OR REPLACE INTO meta VALUES (?, ?)", [(k, json.dumps(v)) for k, v in meta.items()])
        c.execute("COMMIT")
        self._seq = row_count(c)
        self._mseq: int = c.execute("SELECT coalesce(max(seq) + 1, 0) FROM media").fetchone()[0]
        self._ids = {key: kid for kid, key in key_names(c).items()}
        last: float | None = c.execute("SELECT max(step_hi) FROM rowmeta").fetchone()[0]
        self.id, self.name, self.created = str(meta["id"]), str(meta["name"]), as_float(meta["created"]) or 0.0
        self._journal: journal.Writer | None = None
        self._journaled: dict[str, str] = {}
        if journal.wanted(self.dir):
            try:
                self._journal = journal.Writer(self.dir, self.id, self._seq, self._mseq, lambda: _snapshot(c))
                self._journal.append(self._meta_ops(meta), self._seq, self._mseq)
            except OSError as e:
                self._journal_failed(e)
        c.close()
        self._next_step: float = int(last) + 1 if last is not None else 0
        self._summary = as_dict(meta.get("summary"))
        self._info = as_dict(meta["info"])
        self._rows: list[_Row] = []
        self._media: list[_Media] = []
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = False
        self._state: FinalState | None = None
        self._failed = False
        self._finished = False
        self._error: Exception | None = None
        self._thread = threading.Thread(target=self._loop, name=f"trex-{self.name}", daemon=True)
        self._thread.start()
        self._prev_hook = sys.excepthook
        sys.excepthook = self._excepthook
        atexit.register(self.finish)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, et: type[BaseException] | None, ev: BaseException | None, tb: TracebackType | None) -> None:
        self.finish("failed" if et else "finished")

    def _excepthook(self, et: type[BaseException], ev: BaseException, tb: TracebackType | None) -> None:
        self._failed = True
        self._prev_hook(et, ev, tb)

    def log(self, data: Metrics, step: float | None = None, timestamp: float | None = None) -> None:
        """Log one row; `step` defaults to the previous one + 1, `timestamp` to now."""
        if step is None:
            step = self._next_step
        self._next_step = step + 1
        row: dict[str, float] = {}
        for k, v in _flatten(data).items():
            n = _number(v)
            if n is not None:
                row[k] = n
        if row:
            with self._lock:
                self._rows.append(_Row(step, time.time() if timestamp is None else timestamp, row))

    def info(self, info: Mapping[str, object] | None = None, **kv: object) -> None:
        """Merge notes into the run's info."""
        new = _jsonify({**(info or {}), **kv})
        assert isinstance(new, dict)
        with self._lock:
            self._info.update(new)

    def summary(self, **kv: object) -> None:
        """Set values shown with the run, e.g. final scores."""
        with self._lock:
            self._summary.update({k: _jsonify(v) for k, v in kv.items()})

    def log_image(self, key: str, image: ImageInput, step: float | None = None) -> None:
        """Log an image."""
        if isinstance(image, (str, os.PathLike)):
            p = Path(image)
            data, ext = p.read_bytes(), p.suffix.lstrip(".").lower().replace("jpeg", "jpg")
        elif isinstance(image, (bytes, bytearray)):
            data, ext = bytes(image), sniff_image(image)
        elif isinstance(image, SupportsSavePNG):
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            data, ext = buf.getvalue(), "png"
        else:
            data, ext = encode_png(image), "png"
        self._add_media(key, step, "image", ext, data)

    def log_video(self, key: str, video: VideoInput, step: float | None = None, fps: float = 30) -> None:
        """Log a video; `fps` applies to frame arrays."""
        if isinstance(video, (str, os.PathLike)):
            p = Path(video)
            data, ext = p.read_bytes(), p.suffix.lstrip(".").lower()
        elif isinstance(video, (bytes, bytearray)):
            data, ext = bytes(video), "mp4"
        else:
            data, ext = encode_mp4(video, fps), "mp4"
        self._add_media(key, step, "video", ext, data)

    def log_html(self, key: str, html: str | Path, step: float | None = None) -> None:
        """Log HTML: its text, or a `Path` to a file."""
        if isinstance(html, Path):
            html = html.read_text()
        self._add_media(key, step, "html", "html", html.encode())

    def finish(self, state: FinalState | None = None, timeout: float = 120.0) -> None:
        """Commit everything and mark the run ended (by default failed after an uncaught exception,
        else finished). Raises RuntimeError if the background writer failed."""
        if self._finished:
            return
        self._finished = True
        self._state = state or ("failed" if self._failed else "finished")
        self._stop = True
        self._wake.set()
        self._thread.join(timeout)
        if sys.excepthook == self._excepthook:
            sys.excepthook = self._prev_hook
        if self._error:
            raise RuntimeError(f"trex writer failed: {self._error!r}") from self._error

    # ---- background commits ----

    def _add_media(self, key: str, step: float | None, kind: MediaKind, ext: str, data: bytes) -> None:
        if step is None:
            step = max(self._next_step - 1, 0)
        with self._lock:
            self._media.append(_Media(key, step, time.time(), kind, ext, data))

    def _write_media_file(self, ext: str, data: bytes) -> str:
        """Write media content-addressed under media/ (temp name, fsync'd when journaling, then rename);
        returns its relative path."""
        name = f"media/{hashlib.sha256(data).hexdigest()}.{ext}"
        path = self.dir / name
        if not path.exists():
            tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
            with open(tmp, "wb") as f:
                f.write(data)
                if self._journal is not None:
                    f.flush()
                    os.fsync(f.fileno())
            os.replace(tmp, path)
        return name

    def _meta_ops(self, meta: Mapping[str, JSONValue]) -> list[journal.Op]:
        """Inserts of the meta values that changed since last journaled."""
        ops: list[journal.Op] = []
        for k, v in meta.items():
            text = json.dumps(v)
            if self._journaled.get(k) != text:
                self._journaled[k] = text
                ops.append(("meta", (k, text)))
        return ops

    def _journal_failed(self, e: OSError) -> None:
        """Stop journaling; the run goes on in trex.sqlite, readable on this host."""
        self._journal = None
        print(f"[trex] journal of {self.dir} failed, so other hosts stop seeing this run's updates: {e!r}", file=sys.stderr)

    def _commit(self, c: sqlite3.Connection, final_state: FinalState | None = None) -> None:
        with self._lock:
            rows, self._rows = self._rows, []
            media, self._media = self._media, []
            summary = dict(self._summary)
            info = dict(self._info)
        files = [(m.key, m.step, m.time, m.kind, self._write_media_file(m.ext, m.data), len(m.data)) for m in media]
        ops: list[journal.Op] = []
        for i in range(0, len(rows), chunks.MAX_ROWS):
            part = [(float(r.step), r.time - self.created, r.values) for r in rows[i: i + chunks.MAX_ROWS]]
            ops += chunks.inserts(self._seq + i, part, self._ids)
        ops += [("media", (self._mseq + i, float(s), t, k, kind, f, n)) for i, (k, s, t, kind, f, n) in enumerate(files)]
        meta: dict[str, JSONValue] = {"heartbeat": time.time(), "summary": summary, "info": info}
        if final_state:
            meta["state"] = final_state
        c.execute("BEGIN")
        journal.replay(c, ops)
        c.executemany("INSERT OR REPLACE INTO meta VALUES (?, ?)", [(k, json.dumps(v)) for k, v in meta.items()])
        c.execute("COMMIT")
        self._seq += len(rows)
        self._mseq += len(files)
        if self._journal is not None:
            try:
                self._journal.append(ops + self._meta_ops(meta), self._seq, self._mseq)
            except OSError as e:
                self._journal_failed(e)

    def _loop(self) -> None:
        c = connect_rw(self.dir)
        try:
            while True:
                self._wake.wait(self.commit_interval)
                self._wake.clear()
                stop = self._stop
                self._commit(c, self._state if stop else None)
                if stop:
                    return
        except Exception as e:  # surfaced by finish(); logging must never crash training
            self._error = e
            print(f"[trex] writer for {self.dir} failed: {e!r}", file=sys.stderr)
        finally:
            c.close()
            if self._journal is not None:
                self._journal.close()


def _snapshot(c: sqlite3.Connection) -> list[journal.Op]:
    """Inserts that rebuild the run in `c`."""
    ops: list[journal.Op] = []
    for table, cols in journal.TABLES.items():
        ops += [(table, tuple(row)) for row in c.execute(f"SELECT {cols[cols.index('(') + 1:-1]} FROM {table}")]
    return ops


def init(dir: str | os.PathLike[str], *, name: str | None = None, config: Mapping[str, object] | None = None,
         tags: Iterable[str] = (), info: Mapping[str, object] | None = None, commit_interval: float = 1.0) -> Run:
    """Open the run in `dir`, or resume the one there; the directory is the run's identity.
    `config` flattens to `a/b` and replaces a resumed run's; `info` merges into it."""
    return Run(dir, name=name, config=config, tags=tags, info=info, commit_interval=commit_interval)
