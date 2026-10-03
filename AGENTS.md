# AGENTS.md

Guidance for coding agents working on **trex** (trackosaurus exp), a file-based experiment
explorer. To *use* trex to explore runs, read `trex --help` and `trex COMMAND --help`.

## Layout

| path | what |
|---|---|
| `trex/format.py` | the run file: `trex.sqlite` schema (format 3), `connect_rw`, `connect_ro`. The contract between writer and readers. |
| `trex/journal.py` | commit journal for runs on network filesystems: `Writer` (append + fsync), `records`, `sync` (replay into a local replica) |
| `trex/chunks.py` | per-metric chunk codec: each commit is one `rowmeta` row (steps, times) plus one `chunk` per metric present |
| `trex/tiles.py` | envelope pyramid tiles (min, max, mean, mean step, mean runtime and count per bucket), `top_tiles`, `build`, `coarsen`, `decode` |
| `trex/writer.py` | `trex.init` / `Run` / `folder_info`: logging API, background commit thread |
| `trex/media.py` | PNG/MP4 encoding for logged arrays (numpy and ffmpeg imported lazily) |
| `trex/index.py` | `Explorer`: crawl, per-run scans (inline or process pool), index cache, top tiles, on-demand finer tiles with a size-bounded cache, event hub |
| `trex/server.py` | read-only HTTP + SSE for the UI: `/api/runs`, `/api/tiles`, `/api/rows`, `/api/stream`, media |
| `trex/daemon.py` | `trex daemon`: `Roots` (tracked directories by spec, `unique_names`, workspaces; `roots.json`, remembered ones in `history.json`), the Unix control socket, its client |
| `trex/workspace.py` | workspaces: `Workspace` merges member directories (`Local`, `Far`) behind the Explorer interface, renaming run ids |
| `trex/remote.py` | `host:path` directories: `parse`, the ssh + `uvx` command, `Remote` (one ssh session, reconnected with backoff) |
| `trex/update.py` | the daemon's update: `uv tool install $TREX_SOURCE`, then exit `RESTART_STATUS` for systemd to restart it |
| `trex/query.py` | read-side queries for the CLI: records, field access, sorting, statistics, series |
| `trex/where.py` | run filters: a SQL WHERE clause (or a name search) compiled to a test over a field getter |
| `trex/cli.py` | `trex` command (Typer): `serve daemon systemd-unit ls groups keys tree show series tail media diff index` |
| `trex/static/` | UI, plain ES modules: `app.js` (page), `data.js` (tile store, scheduler, IndexedDB, stream), `plot.js` (charts), `gl.js` (WebGL2 renderer), `kernel.js` (columns, smoothing, decimation, group stats, CRC-32), `where.js` (run filters, run fields, filter completion); `index.html` |
| `examples/demo.py` | synthetic sweeps and live runs for trying the UI |
| `docs/build.py` | pdoc pages of every module into `site/`; the user guide is the `trex` and `trex.daemon` docstrings (Markdown) |
| `docs/media/` | the docs' video tour and its poster image (left out of the sdist); the README embeds the same video, uploaded to GitHub |
| `tests/` | pytest suites (`test_processes.py` runs real `trex` processes), the node kernel tests, and the browser smoke test |

Runtime dependencies: Python ≥ 3.12, numpy, and Typer for the CLI (ffmpeg only to log frame arrays
as video). The UI loads no external scripts, and the server is standard library.

## Commands

```bash
uv sync && npm install                                    # dev deps: pytest, pytest-cov, pytest-xdist, pyright, radon, pdoc; eslint (UI tests only)
uv run pyright                                            # types: strict for the logging API and formats, standard elsewhere
uv run pytest                                             # all suites, one worker per CPU (-n0: serially)
uv run pytest --cov                                       # the same with branch coverage, subprocesses included; fails below 94%
uv run python docs/build.py                               # pdoc pages into site/
node --test tests/*.test.mjs                              # JS kernel and complexity (node >= 18)
uv run --with playwright python tests/browser_smoke.py    # headless UI on its own throwaway server; fails below 92% UI line coverage
uv run python examples/demo.py /tmp/runs && uv run trex serve /tmp/runs
```

Run pyright and the three suites after any change that touches their area. CI
(`.github/workflows/test.yml`) runs pyright, `pytest --cov` and the node tests with coverage of `kernel.js` and `where.js` (95% of lines, 85% of
branches) on Python 3.12; the smoke test stays local. It also measures which lines of the UI modules run (V8 coverage over every
page it loads), fails below `UI_COVERAGE`, and writes the uncovered lines to `ui_coverage.txt` beside its screenshots. The smoke test starts its own
`trex serve --standalone` and a `trex daemon` on a temporary directory, free ports and a private
`TREX_DAEMON_DIR`. Never point tests at a server or daemon someone is using.

Headless Chromium reaches the GPU only with
`--headless=new --use-gl=angle --use-angle=gl-egl --ignore-gpu-blocklist --enable-gpu`; without
them WebGL falls back to software and timings mean nothing.

## How data flows

- **Run file** (`trex.sqlite`, SQLite in WAL mode, plus `media/`): `meta`, `media`, `keys`,
  `rowmeta(seq0, n, step_lo, step_hi, steps|times)` and `chunk(key_id, seq0, values)`. One commit
  of at most 65535 rows is one `rowmeta` row and one chunk per metric logged in it, so reading one
  metric is an index range scan.
- **Journal** (`trex.journal`, runs on network filesystems, or `TREX_JOURNAL=1`): every commit's inserts,
  appended and fsync'd, because SQLite's WAL is readable only on the writer's host. `connect_ro`
  reads a live journaled run (one with a `-wal`) from a replica in `$TREX_REPLICAS` (default
  `<tmp>/trex-<uid>/replicas`), brought up to date from the journal on each open.
- **Index** (`<cache>/<hash of root>/index.sqlite`): per-run metadata, last values, metric names,
  media, and two kept tile tiers of every metric: the *top tiles* (the coarsest pyramid level
  covering the run in at most two tiles) and *overview tiles* (the top tiles merged `OVERVIEW_UP`
  levels coarser). Growing runs get new kept tiles every `TOP_REFRESH` seconds and when they finish
  or crash. The index never holds a full copy of the data.
- **Server**: `/api/tiles` answers `[run, key, "top" | "overview"]` from the index and
  `[run, key, level, index]` by building the tile from the run file, cached in the index up to
  `TREX_TILE_CACHE_MB` across all of a process's Explorers (`TileBudget`; least recently used
  evicted). A cached tile stays valid while later rows
  lie beyond its step range. `/api/tiles/bundle` answers one kept tier of one metric for every run
  of a folder in one indexed read.
- **Daemon**: one server, one `Explorer` (or `Remote`) per tracked directory. A directory's URLs are its
  standalone URLs under `/r/<name>/`, a workspace's under `/w/<name>/`, and the UI prefixes every request
  with that (`BASE` in `data.js`); `/` is the root view, every tracked directory as a top-level folder
  (`Roots.everything`, a nested `Workspace`), and the trex brand opens the panel that manages them. Names follow Emacs's uniquify (`runs<chush>`)
  and change when a collision appears or ends; specs (paths, `host:path`) are the stable keys.
- **Workspace**: answers the Explorer interface by asking its members in parallel. A run id is the
  member's path, or `path<member>` when an earlier member holds the same path; `resolve` maps it back.
  Its stream merges the members' streams, renaming run ids in every event (`Workspace._rename_event`).
  `/api/daemon` lists the directories, the remembered ones and the running trex; directories are
  added over the control socket (mode 0600) or HTTP and removed over HTTP. An update installs
  `$TREX_SOURCE` and exits with `update.RESTART_STATUS`; the unit's `RestartForceExitStatus` restarts
  it, and the UI reloads once `/api/daemon` reports the new install (`update.RUNNING`, read at start).
  A removed directory's `Explorer` is closed (`Explorer.close`). A `host:path` directory is a `Remote`:
  one `ssh -L <local socket>:<remote socket> host uvx --from <source>@<this commit> trex serve PATH --unix
  <remote socket> --exit-on-eof`, and `Handler._proxy` passes its `/r/<name>/` requests (the stream too)
  to the local socket; the remote server ends when the session's stdin closes. Tests use a fake `ssh`
  and `uvx` (`tests/test_remote.py`).
- **Browser** (`data.js`): visible charts state what they show (`plan`). Each run needs buckets
  about `LINE_PX_PER_BUCKET` wide on screen (`DENSITY_PX_PER_BUCKET` in a density heatmap),
  within a point budget per chart. Charts with many runs start from overview tiles and fetch top
  tiles only where those are too coarse; zooming fetches finer tiles, at most `FINE_TILES` per chart. A chart
  of more than `DENSITY_AUTO` runs (a heatmap, or group statistics) plans coarse buckets
  (`DENSITY_PX_PER_BUCKET`), keeps its bucket merging whatever the zoom, and takes overview tiles until they
  are two levels too coarse. When many runs of a chart
  need a tier, one bundle request fetches it. Buckets are merged locally (`Entry.up`) when that is
  enough. Each (run, metric) column is rebuilt from its best tiles plus the raw rows streamed since
  its kept tiles were built. Top tiles of finished runs are cached in IndexedDB, keyed by metric so
  one range read serves a whole chart.
- **Rendering**: WebGL2 by default (`?gl=0` selects Canvas 2D). Above 300 lines a chart draws a
  density heatmap. No upload overwrites GPU data a queued draw may read: each draw's line table
  takes fresh rows of the table texture (`Renderer.bind`), and a changed column moves to a new slot
  (`LineSet.update`); only appends past a column's drawn points are written in place. Charts then
  never depend on how a driver orders uploads against earlier draws.

## Invariants (things that break silently if ignored)

- **Explorer is read-only.** Nothing under a runs directory is ever created or modified by
  `index.py`, `server.py`, `query.py` or the CLI. `connect_ro` opens runs without a `-wal` file as
  `immutable` so SQLite does not create `-wal`/`-shm`; a test asserts directory contents are
  unchanged. Callers re-check the file signature after reading.
- **Sequence numbers are contiguous.** Rows and media are numbered 0, 1, 2, … per run; readers stop
  at a gap. The browser holds rows `[tiles_seq, seq)` of each running run and resyncs on any gap
  or count mismatch.
- **Events follow commits, in order.** `Explorer._apply` publishes after the index transaction
  commits. A run's `run` event comes after the rows and media it counts (a new run's comes
  first); the browser treats a `run` event whose counts it has not reached as lost data.
- **Cache versions.** Bump `CACHE_VERSION` in `index.py` whenever what the index stores, or how it
  derives it, changes. Bump the IndexedDB version in `data.js` when the stored entries or their
  keys change. Older caches are then rebuilt instead of silently misread.
- **Shared formats across languages.** Change all of these together:
  - tile encoding: `tiles.py` and `decodeTile` in `static/data.js`;
  - tile response framing: `post_tiles` in `server.py` and `unframe` in `static/data.js`;
  - local bucket merging: `tiles.coarsen` and `Data.rebuild` (count-weighted means). Points are drawn
    at their bucket's mean step, never the bucket center;
  - smoothing (time-weighted EMA, scale = span/1000 rounded to a power of two): `kernel.js`
    `Col.ensureSmooth`, `plot.js` `smoothScale`, `query.py` `twema`/`smooth_scale`;
  - group statistics (order-statistic median CI, Student-t mean CI, interquartile mean of ranks
    [floor(n/4), n - floor(n/4)) with Yuen's CI): `kernel.js` `agg` / `medianCiRank` / `iqmStats`, `plot.js` `bandOf`,
    `query.py` `stats` / `_iqm`;
  - run filters and field names: `where.py` and `static/where.js` (`compileWhere`, `runField`), `query.py` `get`;
    both suites run the cases in `tests/where_cases.json`.
- **Other sites cannot use the server.** `Handler._refusal` answers only requests whose Host is an
  IP address, `localhost`, this machine's name or an `--allow-host` name (DNS rebinding), and
  refuses a POST whose Origin is not its Host. The UI sends no cross-origin requests.
- **A live run is read on its writer's host or through its journal.** Never open another host's WAL:
  even read-only readers write SQLite's shared index, which a network filesystem does not keep
  coherent. A journal replayed from its start rebuilds the run exactly (`tests/test_journal.py`).
- **Writer never crashes training.** Errors in the commit thread are recorded and raised from
  `finish()`, not from `log()`. Media files are written (temp name, then rename) before the row
  that references them commits.
- **Run identity is its path** (relative to the served root). A changed `meta.id` or a shrunken
  row count makes the explorer drop and re-index the run.

## UI performance rules

The 10k-run view is the benchmark; interactions should reach the next painted frame in about
20 ms, and no task should block input for more than about 50 ms.
- Per-draw work is proportional to pixels, not points (`prep` decimates per pixel).
- Charts draw first. Sidebar, path bar and info panel update after the frame paints
  (`App.afterPaint`), one task each. Chart drawing stops at `FRAME_BUDGET_MS` per frame and
  continues on the next. Redraws for streamed data come at most 4 times a second, less often when the
  visible charts are expensive to draw, and at once when loading finishes.
- Group statistics (`agg`) never sort a bin: order statistics come from histogram selection, exact.
- Work over all runs is sliced (`Data.rebuildSoon`) or indexed (`ConfigIndex`); nothing that
  touches every run × every key runs on an interaction.
- The sidebar builds only the rows near its scroll position. Off-screen panels skip layout
  (`content-visibility`) and hold no canvas backing store.
- Measure before and after a change (time from the action to the next frame, plus long tasks),
  rather than assuming.

## Conventions

- Cyclomatic complexity is at most 15 per function, Python (radon) and UI JS (eslint's
  `complexity` rule); `tests/test_complexity.py` and `tests/complexity.test.mjs` enforce it. Split
  a function by what it does into named steps; keep hot inner loops in one function.

- Python via `uv run`; never bare `python`. Stdlib and numpy first. Adding a runtime dependency
  needs a very good reason.
- Types: every function is annotated. The logging API (`writer.py`) and the format modules
  (`format`, `chunks`, `tiles`, `media`) pass pyright strict; the rest passes standard. Use PEP 695
  `type` aliases and generics, `X | None`, built-in generics, `TypedDict`/`NamedTuple` for records
  that cross modules or processes, and `Final` for constants. Narrow JSON read from run files with
  the `format.as_*` helpers rather than casts. No `# pyright: ignore` in `trex/`.
- Comments and docstrings state the intended behavior, tersely. The `trex` and `trex.daemon` module
  docstrings are the published user guide, and terse too. No history ("fixed", "now",
  "previously"), no restating the code; prefer a good name to a comment.
- Tests encode the intended contract. If behavior is wrong, the test should fail. Never pin a
  bug or mark it `xfail`. Test names describe the assertion.
- UI: no build step, no frameworks, no CDN, one language. Numeric work belongs in `kernel.js` on
  typed arrays.
- Long-running jobs: print periodic progress lines (not `tqdm`) when output goes to a log file.
- Do not wait on processes with `pgrep -f`/`pkill -f` patterns that can match your own command
  line. Use PIDs (`ss -ltnp` for servers, `$!` for children).
- American English in user-facing text.
