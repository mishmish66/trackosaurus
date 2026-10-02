"""The `trex` command; `trex --help` documents it."""

import csv
import json
import math
import os
import re
import signal
import sys
import threading
import time
from array import array
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Final, Literal, TextIO

import typer

from . import query as Q
from .format import JSONValue, as_dict, as_str
from .server import DEFAULT_PORT
from .where import compile_where

if TYPE_CHECKING:
    from .daemon import Roots
    from .index import Explorer
    from .server import Server

type OutRow = Mapping[str, object]
type Format = Literal["table", "json", "jsonl", "csv", "tsv"]
type Series = dict[str, tuple[array[float], array[float]]]

OPEN: "list[Explorer]" = []  # indexes a command opened, closed when it ends
DEFAULT_COLUMNS: Final = ["path", "state", "step", "runtime"]
STATE_COLORS: Final = {"running": "green", "failed": "red", "crashed": "red", "finished": "bright_black"}


# ---- output ----

def jsonable(v: object) -> object:
    """JSON-safe copy: non-finite floats as strings."""
    if isinstance(v, float) and not math.isfinite(v):
        return "NaN" if v != v else ("Infinity" if v > 0 else "-Infinity")
    if isinstance(v, dict):
        return {str(k): jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    return v


def fmt_num(v: float) -> str:
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, int):
        return str(v)
    if not math.isfinite(v):
        return str(v)
    a = abs(v)
    if a != 0 and (a >= 1e6 or a < 1e-3):
        return f"{v:.3e}"
    return f"{v:.6g}" if a >= 1 else f"{v:.4g}"


def fmt_dur(s: float) -> str:
    s = float(s)
    if s < 60:
        return f"{s:.1f}s"
    if s < 3600:
        return f"{int(s // 60)}m{int(s % 60):02d}s"
    return f"{int(s // 3600)}h{int(s % 3600 // 60):02d}m"


def fmt_cell(col: str, v: object, width: int | None) -> str:
    if v is None:
        return ""
    if col in ("created", "updated", "heartbeat") and isinstance(v, (int, float)):
        return datetime.fromtimestamp(v).strftime("%Y-%m-%d %H:%M")
    if col == "runtime" and isinstance(v, (int, float)):
        return fmt_dur(v)
    if isinstance(v, (int, float)):
        return fmt_num(v)
    s = v if isinstance(v, str) else json.dumps(jsonable(v), separators=(",", ":"))
    return s if width is None or len(s) <= width else s[: width - 1] + "…"


def state(s: object) -> str:
    """A run state, colored on a terminal."""
    return typer.style(str(s), fg=STATE_COLORS.get(str(s)))


def emit(rows: Sequence[OutRow], cols: Sequence[str] | None, fmt: Format, out: TextIO | None = None,
         width: int | None = 48) -> None:
    """Print rows in `fmt`, restricted to `cols` (None: every key)."""
    stream = out if out is not None else sys.stdout
    if fmt in ("json", "jsonl"):
        data = [jsonable(r if cols is None else {c: r.get(c) for c in cols}) for r in rows]
        stream.write(json.dumps(data, indent=1) + "\n" if fmt == "json" else "".join(json.dumps(d, separators=(",", ":")) + "\n" for d in data))
        return
    cols = cols or sorted({k for r in rows for k in r})
    if fmt in ("csv", "tsv"):
        w = csv.writer(stream, delimiter="," if fmt == "csv" else "\t", lineterminator="\n")
        w.writerows([cols, *([fmt_cell(c, r.get(c), None) for c in cols] for r in rows)])
        return
    _table(rows, cols, width, stream)


def _table(rows: Sequence[OutRow], cols: Sequence[str], width: int | None, stream: TextIO) -> None:
    """Aligned columns, numeric ones right-aligned; on a terminal the header is bold and states colored."""
    cells = [[fmt_cell(c, r.get(c), width) for c in cols] for r in rows]
    widths = [max([len(c)] + [len(row[i]) for row in cells]) for i, c in enumerate(cols)]
    numeric = [all(isinstance(r.get(c), (int, float)) or r.get(c) is None for r in rows) for c in cols]

    def line(vals: Sequence[str], styled: bool) -> str:
        padded = [(v.rjust(w) if n else v.ljust(w)) for v, w, n in zip(vals, widths, numeric, strict=True)]
        if styled:
            padded = [p.replace(v, state(v), 1) if c == "state" and v else p for c, v, p in zip(cols, vals, padded, strict=True)]
        return "  ".join(padded).rstrip()

    typer.echo(typer.style(line(cols, False), bold=True), file=stream)
    for row in cells:
        typer.echo(line(row, True), file=stream)


# ---- selecting runs ----

@dataclass(frozen=True)
class Selection:
    """The runs a run-set command works on: PATH (or --root) narrowed by --where."""

    path: str
    where: list[str]
    root: str | None
    cache: str | None
    force: bool

    def scope(self) -> tuple[Path, str]:
        """(index root, folder prefix)."""
        from .server import check_root

        p = Path(self.path).expanduser().resolve()
        if not p.exists():
            sys.exit(f"trex: {self.path} does not exist")
        if self.root:
            root = check_root(self.root, self.force)
            if not p.is_relative_to(root):
                sys.exit(f"trex: {p} is not under --root {root}")
            prefix = p.relative_to(root).as_posix()
            return root, "" if prefix == "." else prefix
        if Q.is_run(p):
            return check_root(p.parent, self.force), p.name
        return check_root(p, self.force), ""

    def index(self) -> tuple[Q.Explorer, Path, str]:
        """(index brought up to date, root, prefix)."""
        root, prefix = self.scope()
        ex = Q.open_index(root, cache_dir(self.cache))
        OPEN.append(ex)
        return ex, root, prefix

    def records(self) -> tuple[list[Q.Record], Q.Explorer, str]:
        """(selected records, index, prefix)."""
        ex, _, prefix = self.index()
        return self.select(Q.records(ex, prefix)), ex, prefix

    def select(self, recs: Iterable[Q.Record]) -> list[Q.Record]:
        tests = [where_test(w) for w in self.where]
        return [r for r in recs if all(t(lambda f, r=r: Q.get(r, f)) for t in tests)]


def cache_dir(cache: str | None) -> str:
    return cache or os.environ.get("TREX_CACHE") or ".trex_cache"


def run_dirs(args: Iterable[str]) -> list[Path]:
    """Run directories; '-' reads them from stdin."""
    paths = [line.strip() for a in args for line in (sys.stdin if a == "-" else [a]) if line.strip()]
    for p in paths:
        if not Q.is_run(Path(p).expanduser()):
            sys.exit(f"trex: {p} is not a run directory (no trex.sqlite)")
    return [Path(p).expanduser() for p in paths]


def where_test(clause: str) -> Callable[[Callable[[str], object]], bool]:
    """The test of a --where clause; ValueError naming the clause when it does not parse."""
    try:
        return compile_where(clause).test
    except (ValueError, re.error) as e:
        raise ValueError(f"bad filter {clause!r}: {e}") from e


def field_columns(where: Iterable[str], sort: str) -> list[str]:
    """Dotted fields that --where clauses read or --sort names."""
    fields = [f for w in where for f in compile_where(w).fields]
    fields += [re.sub(r":(desc|asc)$", "", part.strip().lstrip("-+")) for part in sort.split(",")]
    return list(dict.fromkeys(f for f in fields if f and f not in Q.PLAIN_FIELDS and "." in f))


def _csv(values: Iterable[str]) -> list[str]:
    """Items of repeatable, comma-separated options."""
    return [x.strip() for x in ",".join(values).split(",") if x.strip()]


# ---- options (Typer reads Literal choices, not `type` aliases) ----

SELECT: Final = "Select runs"
OUTPUT: Final = "Output"
PathArg = Annotated[str, typer.Argument(metavar="PATH", help="Runs directory, a folder in it, or one run.", show_default=False)]
Where = Annotated[list[str] | None, typer.Option("--where", "-w", metavar="CLAUSE", rich_help_panel=SELECT,
                  help="SQL WHERE clause over run fields, e.g. \"lr = 0.001 and state = running\"; text with no "
                       "comparison searches names and paths. Repeat for AND.")]
Root = Annotated[str | None, typer.Option(rich_help_panel=SELECT, help="Index root, for paths relative to a larger tree and a shared cache.")]
Cache = Annotated[str | None, typer.Option(rich_help_panel=SELECT, help="Cache directory (default $TREX_CACHE or ./.trex_cache).")]
Force = Annotated[bool, typer.Option("--force", rich_help_panel=SELECT, help="Allow indexing / or $HOME.")]
Fmt = Annotated[Literal["table", "json", "jsonl", "csv", "tsv"], typer.Option("--format", "-o", rich_help_panel=OUTPUT, help="Output format.")]
Json = Annotated[bool, typer.Option("--json", rich_help_panel=OUTPUT, help="Same as --format json.")]
Full = Annotated[bool, typer.Option("--full", rich_help_panel=OUTPUT, help="Do not truncate wide table cells.")]
Limit = Annotated[int | None, typer.Option("--limit", "-n", help="Keep the first N.")]
RunArg = Annotated[str, typer.Argument(metavar="RUN", help="Run directory.", show_default=False)]
RunsArg = Annotated[list[str], typer.Argument(metavar="RUN...", help="Run directories; '-' reads them from stdin.", show_default=False)]
X = Annotated[Literal["step", "runtime"], typer.Option("--x", help="x axis.")]


def out_format(fmt: Format, as_json: bool) -> Format:
    return "json" if as_json else fmt


def width(full: bool) -> int | None:
    return None if full else 48


EPILOG = """
A run is any directory holding trex.sqlite. Query commands take `--json` (or `--format jsonl|csv|tsv`);
non-finite numbers appear as the strings "NaN", "Infinity" and "-Infinity".

**Fields** (for `--where`, `--sort`, `--columns`, `--group-by`)

    path name parent state step runtime rows media created updated tags dir
    config.KEY    config value, e.g. config.lr, config.model/width
    summary.KEY   last logged value of a metric, or a run.summary value
    info.A.B      nested info value
    KEY           bare key: config first, then summary

**Filters**: `-w CLAUSE`, a SQL WHERE clause over the fields above: `= != < <= > >=`, `in (…)`, `like` (`%`, `_`),
`~` (regex search), `is [not] null`, `and or not` and parentheses; repeat `-w` for AND. Numbers compare as numbers
(`lr = 0.001` matches 1e-3 but not 0.0015), a bare word is text, `like` and `~` ignore case, a list matches when
any element does, and a missing value fails every comparison except the negated ones (`!=`, `!~`, `not …`). Text
with no comparison searches names and paths. The UI's filter box takes the same clauses.

**Examples**

    trex tree runs                     # folder tree, run counts by state, notes
    trex keys runs/sweep               # metric keys and spread of last values
    trex ls runs -w "lr = 0.001 and seed in (0, 1)" -s summary.eval/success:desc -n 10
    trex groups runs -g config.lr -m eval/success    # median and 95% CI
    trex series RUN -k train/loss --points 50 --smooth 0.99
    trex ls runs -w "state = running" --paths | trex series - -k loss --last 1
    trex media RUN --latest            # absolute paths of the newest media
"""

app = typer.Typer(name="trex", help="trackosaurus exp: explore and query directories of trex runs.", epilog=EPILOG,
                  rich_markup_mode="markdown", no_args_is_help=True, add_completion=False, pretty_exceptions_enable=False,
                  context_settings={"help_option_names": ["-h", "--help"]})


def _version(show: bool) -> None:
    if show:
        from .update import installed

        inst = installed()
        typer.echo(f"trex {inst['version']}" + (f" ({inst['commit'][:12]})" if inst["commit"] else ""))
        raise typer.Exit()


@app.callback()
def _options(version: Annotated[bool, typer.Option("--version", "-V", callback=_version, is_eager=True,
                                                   help="Print the version (and git commit) and exit.")] = False) -> None:
    """trackosaurus exp: explore and query directories of trex runs."""


def command[**P](name: str, *aliases: str) -> Callable[[Callable[P, None]], Callable[P, None]]:
    """Register a command (and hidden aliases); filter and regex errors become CLI errors."""

    def deco(fn: Callable[P, None]) -> Callable[P, None]:
        @wraps(fn)
        def run(*a: P.args, **kw: P.kwargs) -> None:
            try:
                fn(*a, **kw)
            except (ValueError, re.error) as e:
                typer.echo(f"trex {name}: {e}", err=True)
                raise typer.Exit(2) from e
            finally:
                while OPEN:
                    OPEN.pop().close()

        app.command(name)(run)
        for alias in aliases:
            app.command(alias, hidden=True)(run)
        return run

    return deco


# ---- commands ----

Hosts = Annotated[list[str] | None, typer.Option("--host", help="Bind address; repeat to listen on several (default 127.0.0.1).")]
AllowHosts = Annotated[list[str] | None, typer.Option("--allow-host", metavar="NAME",
                       help="Host name the UI may be opened by besides addresses, localhost and this machine's names (repeatable).")]
Port = Annotated[int | None, typer.Option(help=f"Port (default the first free one from {DEFAULT_PORT}).", show_default=False)]


def listen(explorer: "Explorer | None", hosts: list[str] | None, port: int | None, allow: list[str] | None,
           roots: "Roots | None" = None) -> list["Server"]:
    """`server.bind`, exiting with a message when an address is unavailable."""
    from .server import bind

    try:
        return bind(explorer, hosts or ["127.0.0.1"], port, roots, allow or [])
    except OSError as e:
        sys.exit(f"trex: cannot listen on {', '.join(hosts or ['127.0.0.1'])} port {port or f'from {DEFAULT_PORT}'}: {e.strerror or e}")


def _interrupt(signum: int, frame: object) -> None:
    raise KeyboardInterrupt


def run_servers(servers: Sequence["Server"], banner: Sequence[str]) -> None:
    """Print `banner`, then serve until SIGINT or SIGTERM."""
    signal.signal(signal.SIGTERM, _interrupt)
    try:
        for line in banner:
            typer.echo(line)
        for srv in servers[1:]:
            threading.Thread(target=srv.serve_forever, daemon=True).start()
        servers[0].serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        for srv in servers:
            srv.server_close()


def add_to_daemon(root: str, force: bool, yes: bool) -> bool:
    """Offer to add `root` (a path or host:path) to a running daemon; whether it was added."""
    from . import daemon

    status = daemon.request({"op": "status"})
    if status is None:
        return False
    url = urls[0] if isinstance(urls := status.get("urls"), list) and urls else "?"
    if not (yes or not sys.stdin.isatty() or typer.confirm(f"Add {root} to the trex daemon at {url}?", default=True)):
        return False
    reply = daemon.request({"op": "add", "path": root, "force": force}) or {"error": "the daemon stopped"}
    if "error" in reply:
        sys.exit(f"trex serve: {reply['error']}")
    typer.echo(f"trex daemon serving {typer.style(root, bold=True)} at {reply['url']}")
    return True


@command("serve")
def serve_cmd(runs_dir: PathArg, host: Hosts = None, port: Port = None, allow_host: AllowHosts = None, cache: Cache = None,
              force: Force = False,
              yes: Annotated[bool, typer.Option("--yes", "-y", help="Add to a running daemon without asking.")] = False,
              standalone: Annotated[bool, typer.Option("--standalone", help="Serve on its own even if a daemon runs.")] = False,
              unix: Annotated[str | None, typer.Option("--unix", metavar="SOCKET", help="Listen on this Unix socket instead.")] = None,
              exit_on_eof: Annotated[bool, typer.Option("--exit-on-eof", hidden=True)] = False) -> None:
    """Crawl a runs directory and serve the web UI, or add it to a running `trex daemon` (which also serves
    host:path directories over ssh)."""
    from . import remote
    from .index import Explorer
    from .server import check_root, serve_unix, urls

    if remote.parse(runs_dir):
        if not add_to_daemon(runs_dir, force, yes=True):
            sys.exit("trex serve: host:path directories are served by `trex daemon`; start one first")
        return
    root = check_root(runs_dir, force)
    if not standalone and add_to_daemon(str(root), force, yes):
        return
    ex = Explorer(root, cache_dir(cache)).start()
    servers = [serve_unix(ex, Path(unix))] if unix else listen(ex, host, port, allow_host)
    where = f"unix:{unix}" if unix else "  ".join(urls(servers))
    if exit_on_eof:
        signal.signal(signal.SIGTERM, _interrupt)
        threading.Thread(target=_exit_on_eof, name="trex-stdin", daemon=True).start()
    run_servers(servers, [f"trex serving {typer.style(str(root), bold=True)} on {where}  cache={ex.cache_dir.resolve()}"])


def _exit_on_eof() -> None:
    """End the process as SIGTERM does once stdin closes."""
    while sys.stdin.buffer.read(1 << 16):
        pass
    os.kill(os.getpid(), signal.SIGTERM)


@command("daemon")
def daemon_cmd(dirs: Annotated[list[str] | None, typer.Argument(metavar="[DIR]...", help="Runs directories to add.", show_default=False)] = None,
               host: Hosts = None, port: Port = None, allow_host: AllowHosts = None,
               cache: Annotated[str | None, typer.Option(help="Cache directory (default $TREX_CACHE or ~/.cache/trex).")] = None,
               force: Force = False) -> None:
    """Serve several runs directories from one server; `trex serve DIR` offers to add to it."""
    from . import daemon, remote, update
    from .server import check_root, urls

    roots = daemon.Roots(Path(cache).expanduser() if cache else daemon.default_cache(), daemon.state_dir() / "roots.json")
    servers = listen(None, host, port, allow_host, roots)
    try:
        control = daemon.ControlServer(daemon.socket_path(), roots, urls(servers))
    except RuntimeError as e:
        sys.exit(f"trex daemon: {e}")

    restarting = threading.Event()

    def restart() -> None:
        restarting.set()
        servers[0].shutdown()

    for srv in servers:
        srv.restart = restart
    threading.Thread(target=control.serve_forever, name="trex-control", daemon=True).start()
    roots.load()
    for d in dirs or []:
        if remote.parse(d):
            roots.add_remote(d, wait=False)
        else:
            roots.add(check_root(d, force))
    banner = [f"trex daemon on {'  '.join(urls(servers))}  socket={control.path}  cache={roots.cache}",
              *(f"  {r['name']}  {r['root']}" for r in roots.served())]
    try:
        run_servers(servers, banner)
    finally:
        control.shutdown()
        control.server_close()
        roots.close()
    if restarting.is_set():
        typer.echo(f"trex daemon: updated; exiting with status {update.RESTART_STATUS} for systemd to restart it")
        sys.exit(update.RESTART_STATUS)


UNIT: Final = """\
# trex daemon as a systemd user service:
#   trex systemd-unit{args} > ~/.config/systemd/user/trex.service
#   systemctl --user daemon-reload && systemctl --user enable --now trex
#   loginctl enable-linger "$USER"     # keep it running while logged out
#   journalctl --user -u trex -f       # its log
[Unit]
Description=trex daemon: experiment explorer for several runs directories
StartLimitIntervalSec=0

[Service]
Type=exec
ExecStart={exec_start}
Environment=PATH={path}
Environment=PYTHONUNBUFFERED=1
Environment=NO_COLOR=1
{env}# An address that is not up yet (Tailscale at boot) fails the start; retry until it is.
Restart=on-failure
RestartSec=2
# After an update the daemon exits with {status} to be started again on the new trex.
SuccessExitStatus={status}
RestartForceExitStatus={status}
TimeoutStopSec=30

[Install]
WantedBy=default.target
"""


def systemd_unit(hosts: list[str], port: int, allow: list[str], cache: str | None, source: str | None) -> str:
    """The unit for `trex systemd-unit`."""
    import shlex

    from . import update

    host_args = [*(a for h in hosts for a in ("--host", h)), *(a for n in allow for a in ("--allow-host", n))]
    options = [*(["--port", str(port)] if port != DEFAULT_PORT else []), *(["--cache", cache] if cache else []),
               *(["--source", source] if source else [])]
    env = {"TREX_CACHE": str(Path(cache).expanduser().resolve()) if cache else None, "TREX_SOURCE": source}
    path = [str(Path(update.uv()).parent), str(Path(sys.executable).parent), "/usr/local/bin", "/usr/bin", "/bin"]
    return UNIT.format(args="".join(f" {shlex.quote(a)}" for a in host_args + options), status=update.RESTART_STATUS,
                       env="".join(f"Environment={k}={v}\n" for k, v in env.items() if v), path=":".join(dict.fromkeys(path)),
                       exec_start=shlex.join([sys.executable, "-m", "trex", "daemon", *host_args, "--port", str(port)]))


@command("systemd-unit")
def systemd_unit_cmd(host: Hosts = None, port: Annotated[int, typer.Option(help="Port.")] = DEFAULT_PORT, allow_host: AllowHosts = None,
                     cache: Annotated[str | None, typer.Option(help="Cache directory (default ~/.cache/trex).")] = None,
                     source: Annotated[str | None, typer.Option(envvar="TREX_SOURCE", show_envvar=False,
                                       help="What the UI's update button installs (default $TREX_SOURCE, else "
                                            "git+https://github.com/mishmish66/trackosaurus; '' for no update button).")] = None) -> None:
    """Print a systemd user unit that runs `trex daemon` on this trex; its header says how to install it."""
    from . import update

    source = update.DEFAULT_SOURCE if source is None else source or None
    typer.echo(systemd_unit(host or [], port, allow_host or [], cache, source), nl=False)
    if source and update.tool_env() != Path(sys.prefix).resolve():
        typer.echo(f"trex systemd-unit: this trex ({sys.prefix}) is not the uv tool install, so the update button will "
                   f"not show; run the unit from `uv tool install {source}`", err=True)


@command("ls", "find")
def ls_cmd(path: PathArg = ".", where: Where = None, root: Root = None,
           cache: Cache = None, force: Force = False,
           sort: Annotated[str | None, typer.Option("--sort", "-s", help="Comma-separated fields; FIELD:desc or -FIELD descends.")] = None,
           limit: Limit = None,
           columns: Annotated[list[str] | None, typer.Option("--columns", "-c", help="Extra columns, comma-separated.")] = None,
           paths: Annotated[bool, typer.Option("--paths", help="Print only absolute run directories (to pipe into series/diff -).")] = False,
           fmt: Fmt = "table", as_json: Json = False, full: Full = False) -> None:
    """List runs, with filters, sorting and chosen columns."""
    recs, _, _ = Selection(path, where or [], root, cache, force).records()
    recs = Q.sort_records(recs, sort or "path")[:limit or None]
    fmt = out_format(fmt, as_json)
    if paths:
        typer.echo("".join(r["dir"] + "\n" for r in recs), nl=False)
    elif fmt in ("json", "jsonl") and not columns:
        emit(recs, None, fmt)
    else:
        extra = _csv(columns or [])
        cols = DEFAULT_COLUMNS + [c for c in field_columns(where or [], sort or "") + extra if c not in DEFAULT_COLUMNS]
        emit([{c: Q.get(r, c) for c in cols} for r in recs], cols, fmt, width=width(full))


@dataclass(frozen=True)
class GroupSpec:
    """How `groups` turns each run into one number per metric, and what it reports."""

    metrics: list[str]
    reduce: Q.Reduce
    at: float | None
    x: Literal["step", "runtime"]
    center: Q.Center
    stats: bool

    def value(self, rec: Q.Record, metric: str, cache: dict[str, Series]) -> object:
        """A run's last value, or its series reduced per --reduce/--at (read once per run)."""
        key = metric_key(metric)
        if self.reduce == "last" and self.at is None:
            return Q.get(rec, f"summary.{key}")
        if rec["dir"] not in cache:
            cache[rec["dir"]] = Q.series(rec["dir"], [metric_key(m) for m in self.metrics], x=self.x)
        xs, ys = cache[rec["dir"]].get(key, (array("d"), array("d")))
        return Q.reduce(xs, ys, self.reduce, at=self.at)

    def rows(self, recs: Sequence[Q.Record], fields: Sequence[str], prefix: str) -> list[dict[str, object]]:
        """One row per distinct value of `fields`."""
        groups: dict[tuple[object, ...], list[Q.Record]] = {}
        for r in recs:
            groups.setdefault(tuple(group_key(r, f, prefix) for f in fields), []).append(r)
        series: dict[str, Series] = {}
        return [self.row(dict(zip(fields, key, strict=True)), members, series) for key, members in groups.items()]

    def row(self, row: dict[str, object], members: Sequence[Q.Record], cache: dict[str, Series]) -> dict[str, object]:
        """A group's row: its fields, run count, and per metric the center, CI, and range and n (or full stats)."""
        row["runs"] = len(members)
        for m in self.metrics:
            st = Q.stats([Q.num(self.value(r, m, cache)) for r in members], center=self.center)
            label = metric_key(m)
            row[label] = st.get(self.center)
            row[f"{label}:ci"] = [st["ci_lo"], st["ci_hi"]] if "ci_lo" in st and "ci_hi" in st else None
            if self.stats:
                row[f"{label}:stats"] = st
            else:
                row[f"{label}:range"] = [st["min"], st["max"]] if "min" in st and "max" in st else None
                row[f"{label}:n"] = st["n"]
        row["paths"] = [r["path"] for r in members]
        return row

    def note(self, rows: Sequence[OutRow]) -> str:
        """What the table's center and CI are, with the CI's actual coverage per group size."""
        key = metric_key(self.metrics[0])
        ns = sorted({n for r in rows if isinstance(n := r.get(f"{key}:n"), int) and n > 1})
        cov = ", ".join(f"n={n}: {Q.median_ci_coverage(n):.1%}" for n in ns) if self.center == "median" else "95%"
        what = ("last value" if self.at is None and self.reduce == "last" else
                f"value at {self.x} <= {self.at}" if self.at is not None else f"{self.reduce} over the run")
        return f"\n{self.center} of each run's {what}; ci = {self.center} 95% CI (actual coverage {cov})"


def metric_key(metric: str) -> str:
    return metric.partition(".")[2] if metric.split(".")[0] in ("summary", "s", "metric", "m") else metric


def group_key(rec: Q.Record, field: str, prefix: str) -> object:
    if field == "subfolder":
        rel = rec["path"][len(prefix) + 1:] if prefix else rec["path"]
        return rel.split("/")[0]
    v = Q.get(rec, field)
    return v if v is None or isinstance(v, (str, int, float, bool)) else json.dumps(v)


def _bracketed(v: object) -> object:
    return "[" + ", ".join(fmt_num(x) for x in v) + "]" if isinstance(v, list) else v


@command("groups", "compare")
def groups_cmd(group_by: Annotated[list[str], typer.Option("--group-by", "-g", show_default=False,
                                   help="Fields, comma-separated: subfolder, parent, config.KEY, info.KEY, ...")],
               path: PathArg = ".", where: Where = None, root: Root = None,
               cache: Cache = None, force: Force = False,
               metric: Annotated[list[str] | None, typer.Option("--metric", "-m", help="Metric keys to aggregate (repeatable).")] = None,
               reduce: Annotated[Literal["last", "first", "max", "min", "mean"], typer.Option("--reduce", "-r", help="How each run's series becomes one value.")] = "last",
               at: Annotated[float | None, typer.Option(help="Use each run's value at the last x <= AT instead of --reduce.")] = None,
               x: X = "step", center: Annotated[Literal["median", "mean", "iqm"], typer.Option(help="Group center (iqm: the mean of the middle half).")] = "median",
               sort: Annotated[str | None, typer.Option("--sort", "-s", help="Columns to sort by; default the first metric, descending.")] = None,
               limit: Limit = None, fmt: Fmt = "table", as_json: Json = False, full: Full = False) -> None:
    """Aggregate runs into groups with a median or mean and its 95% CI."""
    recs, _, prefix = Selection(path, where or [], root, cache, force).records()
    fmt, fields = out_format(fmt, as_json), _csv(group_by)
    spec = GroupSpec(_csv(metric or []), reduce, at, x, center, stats=fmt in ("json", "jsonl"))
    keys = [metric_key(m) for m in spec.metrics]
    rows = Q.sort_records(spec.rows(recs, fields, prefix), sort or (f"-{keys[0]}" if keys else ",".join(fields)),
                          getter=lambda r, f: r.get(f))[:limit or None]
    if spec.stats:
        return emit(rows, None, fmt)
    cols = fields + ["runs"] + [f"{k}{part}" for k in keys for part in ("", ":ci", ":range", ":n")]
    emit([{k: _bracketed(v) if k.endswith((":ci", ":range")) else v for k, v in r.items()} for r in rows], cols, fmt, width=width(full))
    if keys and fmt == "table":
        sys.stdout.flush()
        typer.echo(spec.note(rows), err=True)


@command("keys")
def keys_cmd(path: PathArg = ".", where: Where = None, root: Root = None,
             cache: Cache = None, force: Force = False,
             pattern: Annotated[str | None, typer.Option("--pattern", "-p", help="Regex on key names.")] = None,
             fmt: Fmt = "table", as_json: Json = False, full: Full = False) -> None:
    """Metric and media keys across runs, with the spread of last values."""
    recs, ex, prefix = Selection(path, where or [], root, cache, force).records()
    rx = re.compile(pattern) if pattern else None
    last: dict[str, list[float | None]] = {}
    for r in recs:
        for k, v in r["summary"].items():
            if not rx or rx.search(k):
                last.setdefault(k, []).append(Q.num(v))
    rows: list[dict[str, object]] = []
    for k, vals in sorted(last.items()):
        st = Q.stats(vals)
        rows.append({"key": k, "kind": "metric", "runs": len(vals), "last_min": st.get("min"),
                     "last_median": st.get("median"), "last_max": st.get("max")})
    paths = {r["path"] for r in recs}
    media: dict[tuple[str, str], list[str]] = {}
    for m in ex.runs(prefix)["media"]:
        if m.run in paths and (not rx or rx.search(m.key)):
            media.setdefault((m.key, m.kind), []).append(m.run)
    rows += [{"key": k, "kind": kind, "runs": len(set(rs)), "items": len(rs)} for (k, kind), rs in sorted(media.items())]
    fmt = out_format(fmt, as_json)
    emit(rows, ["key", "kind", "runs", "last_min", "last_median", "last_max", "items"] if fmt != "json" else None, fmt, width=width(full))


@dataclass
class Folder:
    name: str
    path: str
    runs: list[Q.Record]
    dirs: dict[str, "Folder"]
    states: dict[str, int]

    def count(self) -> dict[str, int]:
        """Runs below this folder by state."""
        self.states = {}
        for r in self.runs:
            self.states[r["state"]] = self.states.get(r["state"], 0) + 1
        for d in self.dirs.values():
            for s, n in d.count().items():
                self.states[s] = self.states.get(s, 0) + n
        return self.states


def folder_tree(recs: Sequence[Q.Record], prefix: str, name: str) -> Folder:
    tree = Folder(name, prefix, [], {}, {})
    for r in recs:
        node = tree
        for part in (r["path"][len(prefix) + 1:] if prefix else r["path"]).split("/")[:-1]:
            node = node.dirs.setdefault(part, Folder(part, f"{node.path}/{part}" if node.path else part, [], {}, {}))
        node.runs.append(r)
    tree.count()
    return tree


@command("tree")
def tree_cmd(path: PathArg = ".", where: Where = None, root: Root = None,
             cache: Cache = None, force: Force = False,
             depth: Annotated[int, typer.Option("--depth", "-d", help="Folder levels to show.")] = 3,
             runs: Annotated[bool, typer.Option("--runs", help="Also list runs under each shown folder.")] = False,
             fmt: Fmt = "table", as_json: Json = False) -> None:
    """Folder tree with run counts by state and folder notes."""
    sel = Selection(path, where or [], root, cache, force)
    ex, index_root, prefix = sel.index()
    tree = folder_tree(sel.select(Q.records(ex, prefix)), prefix, prefix or index_root.name)
    notes = {p: v[1] for p, v in ex.folders.items()}
    fmt = out_format(fmt, as_json)

    def as_json_(node: Folder, level: int) -> dict[str, object]:
        out: dict[str, object] = {"path": node.path, "runs": sum(node.states.values()), "states": node.states, "info": notes.get(node.path)}
        if level < depth:
            out["dirs"] = [as_json_(d, level + 1) for _, d in sorted(node.dirs.items())]
            if runs:
                out["run_list"] = [{"path": r["path"], "state": r["state"], "step": r["step"]} for r in node.runs]
        return out

    if fmt in ("json", "jsonl"):
        return typer.echo(json.dumps(jsonable(as_json_(tree, 0)), indent=1 if fmt == "json" else None))
    print_tree(tree, 0, depth, runs, notes)


def print_tree(node: Folder, level: int, depth: int, runs: bool, notes: Mapping[str, Mapping[str, JSONValue]]) -> None:
    states = ", ".join(f"{n} {state(s)}" for s, n in sorted(node.states.items()))
    info = {k: v for k, v in (notes.get(node.path) or {}).items() if k != "trex"}
    first = next((v for v in info.values() if isinstance(v, str)), None) or json.dumps(jsonable(info))
    note = typer.style(f"  — {first[:80]}", dim=True) if info else ""
    typer.echo(f"{'  ' * level}{typer.style(node.name + '/', bold=True)}  {sum(node.states.values())} runs ({states}){note}")
    if level >= depth:
        return
    for _, d in sorted(node.dirs.items()):
        print_tree(d, level + 1, depth, runs, notes)
    for r in sorted(node.runs, key=lambda r: r["path"]) if runs else []:
        typer.echo(f"{'  ' * (level + 1)}{r['path'].rsplit('/', 1)[-1]}  [{state(r['state'])}] step {fmt_cell('step', r['step'], None)}")


@command("show")
def show_cmd(run: RunArg, root: Annotated[str | None, typer.Option(help="Stop collecting folder notes at this directory.")] = None,
             fmt: Fmt = "table", as_json: Json = False, full: Full = False) -> None:
    """Everything about one run: info, config, summary, metric keys, media, folder notes."""
    (d,) = run_dirs([run])
    s = Q.run_summary(d)
    folders = [(p, as_dict(i)) for p, i in Q.folder_infos(d.parent, root)]
    fmt = out_format(fmt, as_json)
    if fmt in ("json", "jsonl"):
        out = jsonable({**s, "folders": [{"dir": p, "info": i} for p, i in folders]})
        return typer.echo(json.dumps(out, separators=(",", ":")) if fmt == "jsonl" else json.dumps(out, indent=1))
    typer.echo(f"{typer.style(str(s['name']), bold=True)}  [{state(s['state'])}]  {s['dir']}")
    typer.echo(f"  id {s['id']} · {s['rows']} rows · step {fmt_cell('step', s['step'], None)} · runtime "
               f"{fmt_dur(s['runtime'] or 0)} · created {fmt_cell('created', s['created'], None)}"
               + (f" · tags {', '.join(s['tags'])}" if s["tags"] else ""))
    sections = [("info", s["info"]), ("config", dict(sorted(s["config"].items()))), ("summary", dict(sorted(s["summary"].items()))),
                *((f"folder info {p}", i) for p, i in folders)]
    for title, obj in sections:
        if obj:
            typer.echo(f"\n{typer.style(title, bold=True)}:\n" + "".join(f"  {k}: {fmt_cell(k, v, None)}\n" for k, v in obj.items()), nl=False)
    typer.echo(f"\n{typer.style('metrics', bold=True)}:")
    emit([{"key": k, **v} for k, v in s["keys"].items()], ["key", "points", "first_step", "last_step", "last"], "table", width=width(full))
    steps: dict[tuple[str, str], list[float]] = {}
    for m in s["media"]:
        steps.setdefault((m["key"], m["kind"]), []).append(m["step"])
    if steps:
        typer.echo(f"\n{typer.style('media', bold=True)}:")
    for (k, kind), st in sorted(steps.items()):
        typer.echo(f"  {k} [{kind}] {len(st)} items, steps {fmt_num(min(st))}..{fmt_num(max(st))}")


def thin(n: int, every: int | None, points: int | None, last: int | None) -> list[int]:
    """Indices kept by --last, --every (and the final point) and --points."""
    idx = list(range(n))
    if last:
        idx = idx[-last:]
    if every and every > 1:
        idx = idx[::every] + ([idx[-1]] if idx and (len(idx) - 1) % every else [])
    if points and len(idx) > points:
        step = (len(idx) - 1) / (points - 1) if points > 1 else len(idx)
        idx = sorted({idx[round(i * step)] for i in range(points)})
    return idx


@dataclass(frozen=True)
class SeriesSpec:
    key: list[str]
    pattern: str | None
    x: Literal["step", "runtime"]
    since: float | None
    until: float | None
    last: int | None
    every: int | None
    points: int | None
    smooth: float | None
    paths: bool

    def keys(self, d: Path) -> list[str]:
        """--key keys, plus the run's keys matching --pattern."""
        keys = set(self.key)
        if self.pattern:
            keys |= {k for k in Q.read_keys(d) if re.search(self.pattern, k)}
        if not keys:
            sys.exit("trex series: give --key KEY (repeatable) or --pattern REGEX")
        return sorted(keys)

    def rows(self, d: Path) -> list[dict[str, object]]:
        name = str(d) if self.paths else as_str(Q.read_meta(d).get("name")) or d.name
        rows: list[dict[str, object]] = []
        for k, (xs, ys) in Q.series(d, self.keys(d), x=self.x).items():
            pts = [(a, b) for a, b in zip(xs, ys, strict=True) if (self.since is None or a >= self.since) and (self.until is None or a <= self.until)]
            sx, sy = [a for a, _ in pts], [b for _, b in pts]
            sm = Q.twema(sx, sy, self.smooth, Q.smooth_scale(max(xs) - min(xs) if xs else 0)) if self.smooth else None
            for i in thin(len(sx), self.every, self.points, self.last):
                rows.append({"run": name, "key": k, self.x: sx[i], "value": sy[i], **({"smoothed": sm[i]} if sm else {})})
        return rows


@command("series")
def series_cmd(runs: RunsArg,
               key: Annotated[list[str] | None, typer.Option("--key", "-k", help="Metric key (repeatable).")] = None,
               pattern: Annotated[str | None, typer.Option("--pattern", "-p", help="Regex selecting metric keys.")] = None,
               x: X = "step", since: Annotated[float | None, typer.Option(help="Keep x >= SINCE.")] = None,
               until: Annotated[float | None, typer.Option(help="Keep x <= UNTIL.")] = None,
               last: Annotated[int | None, typer.Option(help="Keep the last N points per key.")] = None,
               every: Annotated[int | None, typer.Option(help="Keep every Nth point (and the last).")] = None,
               points: Annotated[int | None, typer.Option(help="Downsample to about N evenly spaced points per key.")] = None,
               smooth: Annotated[float | None, typer.Option(help="Add a time-weighted EMA column with this weight (0..1, as in the UI).")] = None,
               paths: Annotated[bool, typer.Option("--paths", help="Label rows by run directory instead of name.")] = False,
               fmt: Fmt = "table", as_json: Json = False, full: Full = False) -> None:
    """Metric series of one or more runs, in long format."""
    spec = SeriesSpec(key or [], pattern, x, since, until, last, every, points, smooth, paths)
    rows = [row for d in run_dirs(runs) for row in spec.rows(d)]
    emit(rows, ["run", "key", x, "value"] + (["smoothed"] if smooth else []), out_format(fmt, as_json), width=width(full))


@command("tail")
def tail_cmd(run: RunArg,
             key: Annotated[list[str] | None, typer.Option("--key", "-k", help="Only these metric keys (repeatable).")] = None,
             lines: Annotated[int, typer.Option("--lines", "-n", help="Rows to show.")] = 10,
             follow: Annotated[bool, typer.Option("--follow", "-f", help="Stream new rows until the run ends.")] = False,
             interval: Annotated[float, typer.Option(help="Seconds between polls with --follow.")] = 1.0,
             timeout: Annotated[float | None, typer.Option(help="Stop following after this many seconds.")] = None,
             fmt: Fmt = "table", as_json: Json = False, full: Full = False) -> None:
    """Last rows of a run; --follow streams new ones until it ends."""
    (d,) = run_dirs([run])
    keys, fmt = key or None, out_format(fmt, as_json)
    first = max(0, Q.row_count(d) - lines) if lines > 0 else Q.row_count(d)
    rows = Q.read_rows(d, keys, start=first)
    cols: list[str] = []
    show_rows(rows[-lines:] if lines else [], keys, cols, fmt, width(full), first=True)
    start = rows[-1].seq + 1 if rows else first
    deadline = time.time() + timeout if timeout else None
    while follow and (deadline is None or time.time() < deadline):
        time.sleep(interval)
        new = Q.read_rows(d, keys, start=start)
        start += len(new)
        show_rows(new, keys, cols, fmt, width(full), first=False)
        if not new and (st := Q.read_meta(d).get("state")) != "running":
            return typer.echo(f"[trex] run {d} is {state(st)}", err=True)


def show_rows(batch: Sequence[Q.chunks.Row], keys: list[str] | None, cols: list[str], fmt: Format, w: int | None, first: bool) -> None:
    """The first batch as a table; followed rows as key=value lines (or the chosen format). `cols` grows with new keys."""
    recs = [{"seq": r.seq, "step": r.step, "runtime": r.t, **r.values} for r in batch if keys is not None or r.values]
    cols += dict.fromkeys(k for r in batch for k in r.values if k not in cols)
    if not recs:
        return
    if fmt in ("json", "jsonl"):
        emit(recs, None, "jsonl")
    elif fmt in ("csv", "tsv") or first:
        emit(recs, ["seq", "step", "runtime", *cols], fmt, width=w)
    else:
        for r in recs:
            typer.echo("  ".join(f"{k}={fmt_cell(k, v, None)}" for k, v in r.items()))
    sys.stdout.flush()


@command("media")
def media_cmd(run: RunArg, key: Annotated[str | None, typer.Option("--key", "-k", help="Regex on media key.")] = None,
              kind: Annotated[Literal["image", "video", "html"] | None, typer.Option(help="Only this kind.")] = None,
              latest: Annotated[bool, typer.Option("--latest", help="Only the latest item per key.")] = False,
              fmt: Fmt = "table", as_json: Json = False) -> None:
    """A run's images, videos and HTML, with absolute file paths."""
    (d,) = run_dirs([run])
    items = [m for m in Q.read_media(d) if (not key or re.search(key, m["key"])) and (not kind or m["kind"] == kind)]
    if latest:
        items = list({m["key"]: m for m in items}.values())
    emit(items, ["step", "key", "kind", "size", "file"], out_format(fmt, as_json), width=None)


def flatten(d: Mapping[str, JSONValue], prefix: str = "") -> dict[str, JSONValue]:
    out: dict[str, JSONValue] = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flatten(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


def run_labels(dirs: Sequence[Path], metas: Sequence[Mapping[str, JSONValue]]) -> list[str]:
    """Names, or paths below the common parent when names repeat."""
    names = [as_str(m.get("name")) or d.name for d, m in zip(dirs, metas, strict=True)]
    if len(set(names)) == len(names):
        return names
    common = Path(os.path.commonpath([str(d.resolve()) for d in dirs]))
    return [d.resolve().relative_to(common).as_posix() or d.name for d in dirs]


@command("diff")
def diff_cmd(runs: RunsArg, all_keys: Annotated[bool, typer.Option("--all", help="Include keys that are equal.")] = False,
             info: Annotated[bool, typer.Option("--info", help="Compare info instead of config.")] = False,
             fmt: Fmt = "table", as_json: Json = False, full: Full = False) -> None:
    """Config (or info) differences between runs."""
    dirs = run_dirs(runs)
    if len(dirs) < 2:
        sys.exit("trex diff: give at least two runs")
    metas = [Q.read_meta(d) for d in dirs]
    names = run_labels(dirs, metas)
    vals = [flatten(as_dict(m.get("info" if info else "config"))) for m in metas]
    rows = [{"key": k, **{n: v.get(k) for n, v in zip(names, vals, strict=True)}} for k in sorted(set().union(*vals))
            if all_keys or len({json.dumps(v.get(k), sort_keys=True) for v in vals}) > 1]
    emit(rows, ["key", *names], out_format(fmt, as_json), width=width(full))


@command("index")
def index_cmd(path: PathArg = ".", root: Root = None, cache: Cache = None, force: Force = False,
              fmt: Fmt = "table", as_json: Json = False) -> None:
    """Build or refresh the cache for a runs directory and report counts."""
    t0 = time.time()
    ex, index_root, prefix = Selection(path, [], root, cache, force).index()
    recs = Q.records(ex, prefix)
    states: dict[str, int] = {}
    for r in recs:
        states[r["state"]] = states.get(r["state"], 0) + 1
    out = {"root": str(index_root), "path": prefix, "runs": len(recs), "states": states,
           "cache": str(ex.cache_dir.resolve()), "seconds": round(time.time() - t0, 3)}
    fmt = out_format(fmt, as_json)
    emit([out], None if fmt in ("json", "jsonl") else list(out), fmt, width=None)


def main(argv: Sequence[str] | None = None) -> None:
    """The `trex` command; `argv` defaults to sys.argv. Returns on success, else raises SystemExit."""
    try:
        app(args=None if argv is None else list(argv), prog_name="trex")
    except SystemExit as e:
        if e.code:
            raise
    except BrokenPipeError:
        sys.stderr.close()


if __name__ == "__main__":
    main()
