# AGENTS.md

Guidance for coding agents working on **trex** (trackosaurus exp), a file-based experiment
explorer. To *use* trex to explore runs, read `trex --help` and `trex COMMAND --help`.

## Layout

| path | what |
|---|---|
| `trex/format.py` | the run file: `trex.sqlite` schema (format 3), `connect_rw`, `connect_ro`. The contract between writer and readers. |
| `trex/chunks.py` | per-metric chunk codec: each commit is one `rowmeta` row (steps, times) plus one `chunk` per metric present |
| `trex/tiles.py` | envelope pyramid tiles (min, max, mean, mean step, mean runtime and count per bucket), `top_tiles`, `build`, `coarsen`, `decode` |
| `trex/writer.py` | `trex.init` / `Run` / `folder_info`: logging API, background commit thread |
| `trex/media.py` | PNG/MP4 encoding for logged arrays (numpy and ffmpeg imported lazily) |
| `trex/index.py` | `Explorer`: crawl, per-run scans (inline or process pool), index cache, top tiles, on-demand finer tiles with a size-bounded cache, event hub |
| `trex/server.py` | read-only HTTP + SSE for the UI: `/api/runs`, `/api/tiles`, `/api/rows`, `/api/stream`, media |
| `trex/daemon.py` | `trex daemon`: `Roots` (directories by name in `roots.json`, remembered ones in `history.json`), the Unix control socket, its client |
| `trex/update.py` | the daemon's update: `uv tool install $TREX_SOURCE`, then exit `RESTART_STATUS` for systemd to restart it |
| `trex/query.py` | read-side queries for the CLI: records, filters, sorting, statistics, series |
| `trex/cli.py` | `trex` command (Typer): `serve daemon systemd-unit ls groups keys tree show series tail media diff index` |
| `trex/static/` | UI, plain ES modules: `app.js` (page), `data.js` (tile store, scheduler, IndexedDB, stream), `plot.js` (charts), `gl.js` (WebGL2 renderer), `kernel.js` (columns, smoothing, decimation, group stats, CRC-32); `index.html` |
| `examples/demo.py` | synthetic sweeps and live runs for trying the UI |
| `docs/build.py` | pdoc pages of every module into `site/`; the user guide is the `trex` and `trex.daemon` docstrings (Markdown) |
| `tests/` | pytest suites (`test_processes.py` runs real `trex` processes), the node kernel tests, and the browser smoke test |

Runtime dependencies: Python ≥ 3.12, numpy, and Typer for the CLI (ffmpeg only to log frame arrays
as video). The UI loads no external scripts, and the server is standard library.

## Commands

```bash
uv sync && npm install                                    # dev deps: pytest, pytest-cov, pyright, radon, pdoc; eslint (UI tests only)
uv run pyright                                            # types: strict for the logging API and formats, standard elsewhere
uv run pytest                                             # writer, media, chunks/tiles, explorer, HTTP, daemon, CLI, complexity
uv run pytest --cov                                       # the same with branch coverage, subprocesses included; fails below 94%
uv run python docs/build.py                               # pdoc pages into site/
node --test tests/*.test.mjs                              # JS kernel and complexity (node >= 18)
uv run --with playwright python tests/browser_smoke.py    # headless UI on its own throwaway server
uv run python examples/demo.py /tmp/runs && uv run trex serve /tmp/runs
```

Run pyright and the three suites after any change that touches their area. CI
(`.github/workflows/test.yml`) runs pyright, `pytest --cov` and the node tests with coverage on
Python 3.12; the smoke test stays local. The smoke test starts its own
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
- **Index** (`<cache>/<hash of root>/index.sqlite`): per-run metadata, last values, metric names,
  media, and two kept tile tiers of every metric: the *top tiles* (the coarsest pyramid level
  covering the run in at most two tiles) and *overview tiles* (the top tiles merged `OVERVIEW_UP`
  levels coarser). Growing runs get new kept tiles every `TOP_REFRESH` seconds and when they finish
  or crash. The index never holds a full copy of the data.
- **Server**: `/api/tiles` answers `[run, key, "top" | "overview"]` from the index and
  `[run, key, level, index]` by building the tile from the run file, cached in the index up to
  `TREX_TILE_CACHE_MB` (least recently used evicted). A cached tile stays valid while later rows
  lie beyond its step range. `/api/tiles/bundle` answers one kept tier of one metric for every run
  of a folder in one indexed read.
- **Daemon**: one server, one `Explorer` per directory. A directory's URLs are its standalone URLs
  under `/r/<name>/`, and the UI prefixes every request with that (`BASE` in `data.js`).
  `/api/daemon` lists the directories, the remembered ones and the running trex; directories are
  added over the control socket (mode 0600) or HTTP and removed over HTTP. An update installs
  `$TREX_SOURCE` and exits with `update.RESTART_STATUS`; the unit's `RestartForceExitStatus` restarts
  it, and the UI reloads once `/api/daemon` reports the new install (`update.RUNNING`, read at start).
- **Browser** (`data.js`): visible charts state what they show (`plan`). Each run needs buckets
  about `LINE_PX_PER_BUCKET` wide on screen (`DENSITY_PX_PER_BUCKET` in a density heatmap),
  within a point budget per chart. Charts with many runs start from overview tiles and fetch top
  tiles only where those are too coarse; zooming fetches finer tiles. When many runs of a chart
  need a tier, one bundle request fetches it. Buckets are merged locally (`Entry.up`) when that is
  enough. Each (run, metric) column is rebuilt from its best tiles plus the raw rows streamed since
  its kept tiles were built. Top tiles of finished runs are cached in IndexedDB, keyed by metric so
  one range read serves a whole chart.
- **Rendering**: WebGL2 by default (`?gl=0` selects Canvas 2D). Above 300 lines a chart draws a
  density heatmap.

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
  - smoothing (time-weighted EMA, scale = span/1000 quantized to quarter octaves): `kernel.js`
    `Col.ensureSmooth`, `plot.js` `smoothScale`, `query.py` `twema`/`smooth_scale`;
  - group statistics (order-statistic median CI, Student-t mean CI): `kernel.js` `agg` /
    `medianCiRank`, `plot.js` `bandOf`, `query.py` `stats`.
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
  continues on the next.
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
