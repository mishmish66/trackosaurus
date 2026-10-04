"""Workspaces: sets of tracked directories, local or remote, shown as one folder tree.

Merged (a named workspace), a run keeps its path within its tracked directory; a path that two members hold
is shown bare for the first member that has it and as `path<member>` for the others. Nested (the daemon's
root view), each member is a top-level folder named for it. Every run gets a `dir` field: its member's name.
A workspace answers the same requests as an Explorer (`runs`, `buckets_body`, ...), by asking each member.
"""

import contextlib
import http.client
import json
import queue
import threading
import time
from collections.abc import Callable, Generator, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Final

import numpy as np
from urllib.parse import quote

from . import buckets as bk
from .index import Explorer, Which, dumps, sse_text
from .remote import Remote

HEARTBEAT: Final = 10.0  # seconds of stream silence after which a heartbeat is sent
OWNERS_TTL: Final = 2.0  # seconds a workspace trusts its map of which member holds which path
STREAM_READ_TIMEOUT: Final = 3 * HEARTBEAT  # seconds without a byte after which a member's stream is reopened

type SseEvent = tuple[str, str]  # (kind, JSON text)


class Unavailable(Exception):
    """A remote member is not connected."""


class Local:
    def __init__(self, name: str, ex: Explorer) -> None:
        self.name, self.ex = name, ex

    def runs(self, prefix: str) -> dict[str, Any]:
        out = self.ex.runs(prefix)
        return {"runs": out["runs"], "media": [list(m) for m in out["media"]], "folders": out["folders"]}

    def run(self, path: str) -> dict[str, Any]:
        out = self.ex.run(path)
        return {"run": out["run"], "media": [list(m) for m in out["media"]]}

    def tree(self) -> list[list[str]]:
        return [[p, s] for p, s in self.ex.tree()]

    def rows(self, path: str, start: int) -> str:
        return self.ex.rows_json(path, start)

    def buckets(self, key: str, level: int, index: int, scope: str, runs: Sequence[str] | None, which: Which) -> bytes:
        return self.ex.buckets_body(key, level, index, scope, runs, which)

    def events(self, prefix: str, stop: threading.Event) -> Generator[SseEvent, None, None]:
        """Backfill, then live events and a heartbeat after each quiet HEARTBEAT, until `stop`."""
        sub = self.ex.hub.subscribe(prefix)
        try:
            for msg in self.ex.backfill(prefix):
                yield from parse_sse(msg.decode())
            quiet = time.time()
            while not stop.is_set() and not sub.dead:
                try:
                    msg = sub.q.get(timeout=0.5)
                except queue.Empty:
                    if time.time() - quiet >= HEARTBEAT:
                        quiet = time.time()
                        yield "hb", dumps(self.ex.live_seqs(prefix))
                    continue
                quiet = time.time()
                yield from parse_sse(msg.decode())
        finally:
            self.ex.hub.unsubscribe(sub)


class Far:
    """A remote member, through its server's Unix socket."""

    def __init__(self, name: str, remote: Remote) -> None:
        self.name, self.remote = name, remote

    def _call(self, method: str, target: str, body: bytes | None = None) -> bytes:
        if self.remote.state != "connected":
            raise Unavailable(self.name)
        conn = self.remote.connection()
        try:
            conn.request(method, target, body=body, headers={"Host": "localhost"})
            r = conn.getresponse()
            data = r.read()
        except OSError as e:
            raise Unavailable(self.name) from e
        finally:
            conn.close()
        if r.status == 404:
            raise KeyError(target)
        if r.status >= 400:
            raise Unavailable(f"{self.name}: {r.status} {data[:200]!r}")
        return data

    def runs(self, prefix: str) -> dict[str, Any]:
        return json.loads(self._call("GET", f"/api/runs?path={quote(prefix)}"))

    def run(self, path: str) -> dict[str, Any]:
        return json.loads(self._call("GET", f"/api/run?path={quote(path)}"))

    def tree(self) -> list[list[str]]:
        return json.loads(self._call("GET", "/api/tree"))

    def rows(self, path: str, start: int) -> str:
        return self._call("GET", f"/api/rows?path={quote(path)}&from={start}").decode()

    def buckets(self, key: str, level: int, index: int, scope: str, runs: Sequence[str] | None, which: Which) -> bytes:
        body = {"key": key, "level": level, "index": index, "scope": scope, "which": which}
        return self._call("POST", "/api/buckets", json.dumps({**body, "runs": list(runs)} if runs is not None else body).encode())

    def events(self, prefix: str, stop: threading.Event) -> Generator[SseEvent, None, None]:
        """The remote server's stream, reopened when it drops, until `stop`."""
        while not stop.is_set():
            if self.remote.state != "connected":
                stop.wait(1.0)
                continue
            conn = self.remote.connection()
            conn.timeout = STREAM_READ_TIMEOUT
            try:
                conn.request("GET", f"/api/stream?path={quote(prefix)}", headers={"Host": "localhost"})
                r = conn.getresponse()
                kind, data = "", []
                while not stop.is_set():
                    line = r.readline()
                    if not line:
                        break
                    text = line.decode().rstrip("\n")
                    if text.startswith("event: "):
                        kind = text[7:]
                    elif text.startswith("data: "):
                        data.append(text[6:])
                    elif not text and kind:
                        yield kind, "\n".join(data)
                        kind, data = "", []
            except (OSError, http.client.HTTPException):
                stop.wait(1.0)
            finally:
                conn.close()


type Member = Local | Far


def parse_sse(text: str) -> Iterator[SseEvent]:
    """The events of SSE text."""
    for block in text.split("\n\n"):
        kind, data = "", []
        for line in block.split("\n"):
            if line.startswith("event: "):
                kind = line[7:]
            elif line.startswith("data: "):
                data.append(line[6:])
        if kind:
            yield kind, "\n".join(data)


class Workspace:
    def __init__(self, name: str, members: Sequence[Member], nested: bool = False) -> None:
        self.name, self.nested = name, nested
        self.members = list(members)
        self._lock = threading.Lock()
        self._owners: dict[str, list[str]] = {}  # path -> names of the members holding it, in member order
        self._owners_at = 0.0
        self._pool = ThreadPoolExecutor(max_workers=max(1, len(self.members)), thread_name_prefix=f"trex-ws-{name}")

    # ---- run ids ----

    def _refresh(self, force: bool = False) -> None:
        if self.nested or (not force and time.time() - self._owners_at < OWNERS_TTL):
            return
        owners: dict[str, list[str]] = {}
        for m, tree in zip(self.members, self._each(lambda m: m.tree()), strict=True):
            for p, _ in tree or []:
                owners.setdefault(p, []).append(m.name)
        with self._lock:
            self._owners, self._owners_at = owners, time.time()

    def ws_id(self, m: Member, path: str) -> str:
        if self.nested:
            return f"{m.name}/{path}"
        with self._lock:
            owners = self._owners.setdefault(path, [m.name])
            if m.name not in owners:
                owners.append(m.name)
        return path if owners[0] == m.name else f"{path}<{m.name}>"

    def resolve(self, run: str) -> tuple[Member, str]:
        """(member, path within it) of a workspace run id."""
        if self.nested:
            for m in sorted(self.members, key=lambda m: -len(m.name)):
                if run.startswith(m.name + "/"):
                    return m, run[len(m.name) + 1:]
            raise KeyError(run)
        for attempt in range(2):
            for m in self.members:
                suffix = f"<{m.name}>"
                if run.endswith(suffix) and m.name in self._owners.get(run[:-len(suffix)], []):
                    return m, run[:-len(suffix)]
            owners = self._owners.get(run)
            if owners:
                return next(m for m in self.members if m.name == owners[0]), run
            if attempt == 0:
                self._refresh(force=True)
        raise KeyError(run)

    def _each[T](self, fn: Callable[[Member], T]) -> list[T | None]:
        """fn(member) for every member in parallel; None for a member that is unavailable."""
        return list(self._pool.map(lambda m: _try(lambda: fn(m)), self.members))

    def _scope(self, prefix: str) -> list[tuple[Member, str]]:
        """The members and their prefixes that a workspace folder or run covers."""
        if not prefix:
            return [(m, "") for m in self.members]
        if self.nested:
            for m in sorted(self.members, key=lambda m: -len(m.name)):
                if prefix == m.name or prefix.startswith(m.name + "/"):
                    return [(m, prefix[len(m.name) + 1:])]
            return []
        try:
            m, path = self.resolve(prefix)
            if path != prefix:
                return [(m, path)]
        except KeyError:
            pass
        return [(m, prefix) for m in self.members]

    def _folder(self, m: Member, path: str) -> str:
        return (f"{m.name}/{path}" if path else m.name) if self.nested else path

    def _rename(self, m: Member, meta: dict[str, Any]) -> dict[str, Any]:
        return {**meta, "id": self.ws_id(m, str(meta["id"])), "dir": m.name}

    # ---- the Explorer interface ----

    def info(self) -> dict[str, Any]:
        root = "daemon:/" if self.nested else f"workspace:{self.name}"
        return {"root": root, "name": self.name, "cache": "", "workspace": [m.name for m in self.members]}

    def tree(self) -> list[list[str]]:
        self._refresh(force=True)
        out: list[list[str]] = []
        for m, tree in zip(self.members, self._each(lambda m: m.tree()), strict=True):
            out += [[self.ws_id(m, p), s] for p, s in tree or []]
        return sorted(out)

    def runs(self, prefix: str) -> dict[str, Any]:
        self._refresh(force=True)
        scope = self._scope(prefix)
        bodies = list(self._pool.map(lambda mp: _try(lambda: mp[0].runs(mp[1])), scope))
        runs: list[object] = []
        media: list[object] = []
        folders: dict[str, Any] = {}
        for (m, _), body in zip(scope, bodies, strict=True):
            if body is None:
                continue
            runs += [self._rename(m, meta) for meta in body["runs"]]
            media += [[self.ws_id(m, str(r[0])), *r[1:]] for r in body["media"]]
            for p, info in body["folders"].items():
                p = self._folder(m, p)
                folders[p] = {**folders.get(p, {}), **info} if isinstance(info, dict) else info
        return {"runs": runs, "media": media, "folders": folders}

    def run(self, path: str) -> dict[str, Any]:
        m, mp = self.resolve(path)
        out = m.run(mp)
        return {"run": self._rename(m, out["run"]), "media": [[path, *r[1:]] for r in out["media"]]}

    def rows_json(self, path: str, start: int) -> str:
        m, mp = self.resolve(path)
        return _with_run(m.rows(mp, start), mp, path)

    def buckets_body(self, key: str, level: int, index: int, scope: str = "", runs: Sequence[str] | None = None,
                     which: Which = "all") -> bytes:
        """The block as `Explorer.buckets_body` answers it, of every member's runs it names, run ids renamed, in id
        order."""
        if runs is None:
            calls = [(m, mp, None) for m, mp in self._scope(scope)]
        else:
            by_member: dict[str, list[str]] = {}
            for r in runs:
                with contextlib.suppress(KeyError):
                    m, mp = self.resolve(r)
                    by_member.setdefault(m.name, []).append(mp)
            calls = [(m, "", by_member[m.name]) for m in self.members if m.name in by_member]
        bodies = list(self._pool.map(lambda c: _try(lambda: c[0].buckets(key, level, index, c[1], c[2], which)), calls))
        paths: list[str] = []
        seqs: list[np.ndarray] = []
        parts: list[bk.Buckets] = []
        for (m, _, _), body in zip(calls, bodies, strict=True):
            if body is None:
                continue
            a = bk.decode(body)
            parts.append(a.buckets._replace(run=a.buckets.run + len(paths)))
            paths += [self.ws_id(m, p) for p in a.paths]
            seqs.append(a.seq)
        seq = np.concatenate(seqs) if seqs else np.empty(0, np.uint32)
        order = np.argsort(np.array(paths, dtype=object), kind="stable").astype(np.int64)
        rank = np.empty(len(paths), np.int32)
        rank[order] = np.arange(len(paths), dtype=np.int32)
        b = bk.union(parts)
        b = b._replace(run=rank[b.run])
        b = bk.take(b, np.argsort(b.run, kind="stable"))
        return bk.encode(level, index, [paths[i] for i in order], seq[order], b)

    def events(self, prefix: str, stop: threading.Event) -> Generator[bytes, None, None]:
        """Every member's stream under `prefix`, run ids renamed, as SSE messages; until `stop`."""
        self._refresh(force=True)
        q: queue.Queue[bytes] = queue.Queue(maxsize=20000)

        def pump(m: Member, mp: str) -> None:
            try:
                with contextlib.closing(m.events(mp, stop)) as events:
                    for kind, data in events:
                        if not _put(q, sse_text(kind, self._rename_event(m, kind, data)), stop):
                            return
            except (Unavailable, OSError):
                pass

        threads = [threading.Thread(target=pump, args=mp, daemon=True) for mp in self._scope(prefix)]
        for t in threads:
            t.start()
        while not stop.is_set():
            try:
                yield q.get(timeout=0.5)
            except queue.Empty:
                continue

    def _rename_event(self, m: Member, kind: str, data: str) -> str:
        if kind == "rows":
            path = json.loads(data[:data.index(',"seq0"')] + "}")["run"]
            return _with_run(data, path, self.ws_id(m, path))
        if kind == "folder":
            ev = json.loads(data)
            return dumps({**ev, "path": self._folder(m, ev["path"])}) if self.nested else data
        if kind == "hb":
            return dumps({self.ws_id(m, p): v for p, v in json.loads(data).items()})
        ev = json.loads(data)
        if kind == "run":
            return dumps(self._rename(m, ev))
        if kind == "media":
            return dumps([self.ws_id(m, ev[0]), *ev[1:]])
        if kind == "delete":
            return dumps({**ev, "run": self.ws_id(m, ev["run"])})
        return data

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)


def _try[T](fn: Callable[[], T]) -> T | None:
    try:
        return fn()
    except (Unavailable, KeyError):
        return None


def _put(q: queue.Queue[bytes], msg: bytes, stop: threading.Event) -> bool:
    """Queue `msg` unless `stop` is set first; whether it was queued."""
    while not stop.is_set():
        try:
            q.put(msg, timeout=0.5)
            return True
        except queue.Full:
            pass
    return False


def _with_run(text: str, old: str, new: str) -> str:
    """A `rows` event's JSON with its run renamed (the run is its first field)."""
    head = f'{{"run":{dumps(old)}'
    return f'{{"run":{dumps(new)}' + text[len(head):] if text.startswith(head) else text
