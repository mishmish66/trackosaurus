"""Workspaces: sets of directories shown as one folder tree.

Merged (a named workspace), a run keeps its path within its directory; a path that two members hold is shown bare for
the first member that has it and as `path<member>` for the others. Nested (a node's root view), each member is a
top-level folder named for it. Every run gets a `dir` field: its member's name. A workspace answers the same requests
as an Explorer (`runs`, `buckets_bodies`, ...), from its members' Explorers.
"""

import contextlib
import json
import queue
import threading
from collections.abc import Generator, Iterator, Sequence
from dataclasses import dataclass, replace

import numpy as np

from . import buckets as bk
from .format import JSONValue, RunState
from .index import Ask, Explorer, MediaRecord, RunMeta, RunsView, RunView, dumps, sse_text

type SseEvent = tuple[str, str]  # (kind, JSON text)


@dataclass(frozen=True, slots=True)
class Member:
    """A directory of a workspace: the name it is shown by, and the Explorer answering for it."""

    name: str
    src: Explorer


def parse_sse(text: str) -> Iterator[SseEvent]:
    """The events of SSE text."""
    for block in text.split("\n\n"):
        kind, data = "", list[str]()
        for line in block.split("\n"):
            if line.startswith("event: "):
                kind = line[7:]
            elif line.startswith("data: "):
                data.append(line[6:])
        if kind:
            yield kind, "\n".join(data)


class Workspace:
    """Directories `members` as one view named `name`: each a top-level folder when `nested`, else their trees merged."""

    def __init__(self, name: str, members: Sequence[Member], nested: bool = False) -> None:
        self.name, self.nested = name, nested
        self.members = list(members)
        self._lock = threading.Lock()
        self._owners: dict[str, list[str]] = {}  # path -> names of the members holding it, in member order

    # ---- run ids ----

    def _refresh(self) -> None:
        if self.nested:
            return
        owners: dict[str, list[str]] = {}
        for m in self.members:
            for p, _ in m.src.tree():
                owners.setdefault(p, []).append(m.name)
        with self._lock:
            self._owners = owners

    def ws_id(self, m: Member, path: str) -> str:
        if self.nested:
            return f"{m.name}/{path}"
        with self._lock:
            owners = self._owners.setdefault(path, [m.name])
            if m.name not in owners:
                owners.append(m.name)
        return path if owners[0] == m.name else f"{path}<{m.name}>"

    def resolve(self, run: str) -> tuple[Member, str]:
        """(member, path within it) of a workspace run id; KeyError for one no member holds."""
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
                self._refresh()
        raise KeyError(run)

    def _scope(self, prefix: str) -> list[tuple[Member, str]]:
        """The members and their prefixes that a workspace folder or run covers."""
        if not prefix:
            return [(m, "") for m in self.members]
        if self.nested:
            for m in sorted(self.members, key=lambda m: -len(m.name)):
                if prefix == m.name or prefix.startswith(m.name + "/"):
                    return [(m, prefix[len(m.name) + 1:])]
            return []
        with contextlib.suppress(KeyError):
            m, path = self.resolve(prefix)
            if path != prefix:
                return [(m, path)]
        return [(m, prefix) for m in self.members]

    def _folder(self, m: Member, path: str) -> str:
        return (f"{m.name}/{path}" if path else m.name) if self.nested else path

    def _rename(self, m: Member, meta: RunMeta) -> RunMeta:
        return replace(meta, id=self.ws_id(m, meta.id), dir=m.name)

    # ---- the Explorer interface ----

    def info(self) -> dict[str, object]:
        root = "daemon:/" if self.nested else f"workspace:{self.name}"
        return {"root": root, "name": self.name, "cache": "", "workspace": [m.name for m in self.members]}

    def tree(self) -> list[tuple[str, RunState]]:
        self._refresh()
        return sorted((self.ws_id(m, p), state) for m in self.members for p, state in m.src.tree())

    def runs(self, prefix: str) -> RunsView:
        self._refresh()
        runs: list[RunMeta] = []
        media: list[MediaRecord] = []
        folders: dict[str, dict[str, JSONValue]] = {}
        for m, mp in self._scope(prefix):
            part = m.src.runs(mp)
            runs += [self._rename(m, meta) for meta in part.runs]
            media += [replace(r, run=self.ws_id(m, r.run)) for r in part.media]
            for p, info in part.folders.items():
                p = self._folder(m, p)
                folders[p] = {**folders.get(p, {}), **info}
        return RunsView(runs, media, folders)

    def run(self, path: str) -> RunView:
        m, mp = self.resolve(path)
        view = m.src.run(mp)
        return RunView(self._rename(m, view.run), [replace(r, run=path) for r in view.media])

    def rows_json(self, path: str, start: int) -> str:
        m, mp = self.resolve(path)
        return _with_run(m.src.rows_json(mp, start), mp, path)

    def buckets_bodies(self, asks: Sequence[Ask]) -> list[bytes]:
        """Each block as `Explorer.buckets_bodies` answers it, of every member's runs it names, run ids renamed, in id
        order."""
        out: list[bytes] = []
        for a in asks:
            bodies = [(m, m.src.buckets_bodies([part])[0]) for m, part in self._parts(a)]
            out.append(self._join(a, [(m, body) for m, body in bodies if body]))
        return out

    def _parts(self, a: Ask) -> list[tuple[Member, Ask]]:
        """Each member's part of an ask: its folder of the scope, or the runs it holds of those named."""
        if a.runs is None:
            return [(m, replace(a, scope=mp)) for m, mp in self._scope(a.scope)]
        mine: dict[str, list[str]] = {}
        for r in a.runs:
            with contextlib.suppress(KeyError):
                m, mp = self.resolve(r)
                mine.setdefault(m.name, []).append(mp)
        return [(m, replace(a, scope="", runs=mine[m.name])) for m in self.members if m.name in mine]

    def _join(self, ask: Ask, parts: Sequence[tuple[Member, bytes]]) -> bytes:
        """One bucket array of the members' arrays of a block, run ids renamed, in id order."""
        paths: list[str] = []
        seqs: list[np.ndarray] = []
        bs: list[bk.Buckets] = []
        for m, body in parts:
            a = bk.decode(body)
            bs.append(a.buckets.of(a.buckets.run + len(paths)))
            paths += [self.ws_id(m, p) for p in a.paths]
            seqs.append(a.seq)
        seq = np.concatenate(seqs) if seqs else np.empty(0, np.uint32)
        order = np.argsort(np.array(paths, dtype=object), kind="stable").astype(np.int64)
        rank = np.empty(len(paths), np.int32)
        rank[order] = np.arange(len(paths), dtype=np.int32)
        b = bk.concat(bs)
        b = bk.take(b.of(rank[b.run]), np.argsort(rank[b.run], kind="stable"))
        return bk.encode(ask.level, ask.block, [paths[i] for i in order], seq[order], b)

    def messages(self, prefix: str, stop: threading.Event) -> Generator[bytes, None, None]:
        """Every member's stream under `prefix`, run ids renamed, as SSE messages; until `stop`."""
        self._refresh()
        q: queue.Queue[bytes] = queue.Queue(maxsize=20000)

        def pump(m: Member, mp: str) -> None:
            with contextlib.suppress(OSError), contextlib.closing(m.src.messages(mp, stop)) as msgs:
                for msg in msgs:
                    for kind, data in parse_sse(msg.decode()):
                        if not _put(q, sse_text(kind, self._rename_event(m, kind, data)), stop):
                            return

        for mp in self._scope(prefix):
            threading.Thread(target=pump, args=mp, daemon=True).start()
        while not stop.is_set():
            with contextlib.suppress(queue.Empty):
                yield q.get(timeout=0.5)

    def _rename_event(self, m: Member, kind: str, data: str) -> str:
        """An event's JSON with its run (or folder) as the workspace names it."""
        match kind:
            case "rows":
                path = json.loads(data[:data.index(',"seq0"')] + "}")["run"]
                return _with_run(data, path, self.ws_id(m, path))
            case "run":
                return dumps(self._rename(m, RunMeta.read(json.loads(data))).wire())
            case "media":
                run, *rest = json.loads(data)
                return dumps([self.ws_id(m, run), *rest])
            case "delete":
                return dumps({"run": self.ws_id(m, json.loads(data)["run"])})
            case "hb":
                return dumps({self.ws_id(m, p): v for p, v in json.loads(data).items()})
            case "folder" if self.nested:
                ev = json.loads(data)
                return dumps({**ev, "path": self._folder(m, ev["path"])})
            case _:
                return data


def _put(q: queue.Queue[bytes], msg: bytes, stop: threading.Event) -> bool:
    """Queue `msg` unless `stop` is set first; whether it was queued."""
    while not stop.is_set():
        with contextlib.suppress(queue.Full):
            q.put(msg, timeout=0.5)
            return True
    return False


def _with_run(text: str, old: str, new: str) -> str:
    """A `rows` event's JSON with its run renamed (the run is its first field)."""
    head = f'{{"run":{dumps(old)}'
    return f'{{"run":{dumps(new)}' + text[len(head):] if text.startswith(head) else text
