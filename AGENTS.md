# AGENTS.md

Guidance for coding agents working on **trex** (trackosaurus exp), a file-based experiment
explorer. To *use* trex to explore runs, read `trex --help` and `trex COMMAND --help`.

## Layout

| path | what |
|---|---|
| `trex/format.py` | the run file: `trex.sqlite` schema (format 3), `connect_rw`, `connect_ro`, `snapshot`. The contract between writer and readers. |
| `trex/journal.py` | commit journal for runs on network filesystems: `Writer` (append + fsync), `records`, `sync` (replay into a local replica) |
| `trex/chunks.py` | per-metric chunk codec: each commit is one `rowmeta` row (steps, times) plus one `chunk` per metric present; `prepare_merge` / `apply_merge` rewrite adjacent commits as one |
| `trex/tiles.py` | envelope pyramid tiles (min, max, mean, mean step, mean runtime and count per bucket), `top_tiles`, `build`, `coarsen`, `decode`; slabs (one step range of many runs at one level: mean, mean step and count per bucket): `stack`, `slab_parts`, `encode_slab`, `decode_slab` |
| `trex/writer.py` | `trex.init` / `Run` / `folder_info`: logging API, background commit thread, and a merge thread that merges small commits |
| `trex/media.py` | PNG/MP4 encoding for logged arrays (numpy and ffmpeg imported lazily) |
| `trex/index.py` | `Explorer`: crawl, per-run scans (inline or process pool), index cache, top tiles, on-demand finer tiles with a size-bounded cache, event hub |
| `trex/server.py` | read-only HTTP + SSE for the UI: `/api/runs`, `/api/tiles`, `/api/rows`, `/api/stream`, media |
| `trex/daemon.py` | `trex daemon`: `Roots` (tracked directories by spec, `unique_names`, workspaces; `roots.json`, remembered ones in `history.json`), the Unix control socket, its client |
| `trex/workspace.py` | workspaces: `Workspace` merges member directories (`Local`, `Far`) behind the Explorer interface, renaming run ids |
| `trex/remote.py` | `host:path` directories: `parse`, the ssh + `uvx` command, `Remote` (one ssh session, reconnected with backoff) |
| `trex/compact.py` | `trex compact`: rewrites a run no process has open with its commits merged (exclusive lock, new file, verify, rename) |
| `trex/update.py` | the daemon's update: `uv tool install $TREX_SOURCE`, then exit `RESTART_STATUS` for systemd or launchd to restart it |
| `trex/query.py` | read-side queries for the CLI: records, field access, sorting, statistics, series |
| `trex/where.py` | run filters: a SQL WHERE clause (or a name search) compiled to a test over a field getter |
| `trex/cli.py` | `trex` command (Typer): `serve daemon systemd-unit launchd-plist ls groups keys tree show series tail media diff index compact` |
| `trex/static/` | UI, plain ES modules: `app.js` (page), `data.js` (tile store, scheduler, IndexedDB, stream), `plot.js` (charts), `gl.js` (WebGL2 renderer), `kernel.js` (columns, smoothing, decimation, group stats, CRC-32), `pool.js` and `worker.js` (group stats on workers over shared columns), `where.js` (run filters, run fields, filter completion); `index.html` |
| `examples/demo.py` | synthetic sweeps and live runs for trying the UI |
| `docs/build.py` | pdoc pages of every module into `site/`; the user guide is the `trex` and `trex.daemon` docstrings (Markdown) |
| `docs/media/` | the docs' video tour and its poster image (left out of the sdist); the README embeds the same video, uploaded to GitHub |
| `tests/` | pytest suites (`test_processes.py` runs real `trex` processes; shared helpers in `helpers.py` and `conftest.py`), the node tests, the cross-language cases (`where_cases.json`, `shared_cases.json`), and the browser smoke test |

Runtime dependencies: Python ≥ 3.12, numpy, and Typer for the CLI (ffmpeg only to log frame arrays
as video). The UI loads no external scripts, and the server is standard library.

## Commands

```bash
uv sync && npm install                                    # dev deps: pytest, pytest-cov, pytest-xdist, pyright, radon, pdoc; eslint (UI tests only)
uv run pyright                                            # types: strict for the logging API and formats, standard elsewhere
uv run pytest                                             # all suites, one worker per CPU (-n0: serially)
uv run pytest --cov                                       # the same with branch coverage, subprocesses included; fails below 94%
uv run python docs/build.py                               # pdoc pages into site/
node --test tests/*.test.mjs                              # JS kernel, shared cases, complexity and unused variables (node >= 18)
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
  `rowmeta(seq0, n, step_lo, step_hi, data: steps then times)` and `chunk(key_id, seq0, data: values)`. One commit
  of at most 65535 rows is one `rowmeta` row and one chunk per metric logged in it, so reading one
  metric is an index range scan. A second writer thread merges the session's newest small commits
  (`writer.merge_plan`: `FAN_IN` commits of a value tier at a time, at most `MERGE_VALUES` values, and
  never a `sealed` commit), so a run logging a row per commit stays near 8 bytes per value. It reads and
  builds each merge outside any write lock and swaps it in with one short transaction, so commits wait
  only for the swap. Readers find rows by the commits that overlap them and never depend on where
  commits begin or end.
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
  of a folder in one indexed read. An `Explorer` keeps framed bundles (`BUNDLE_CACHE_BYTES`) and `/api/runs` answers in
  memory until what they hold changes (`_kept_gen`, `_view_gen`).
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
  `$TREX_SOURCE` and exits with `update.RESTART_STATUS`; the systemd unit's `RestartForceExitStatus`
  or the launchd agent's `KeepAlive` restarts it, and the UI reloads once `/api/daemon` reports the new install (`update.RUNNING`, read at start).
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
  enough. Each (run, metric) column is rebuilt from its best tiles plus the rows streamed since they were
  built, in the buckets the tiles will hold them in (`buildColumn`), so a live run looks the same when its tiles
  catch up. Top tiles of finished runs are cached in IndexedDB, keyed by metric so one range read serves a whole
  chart; overview tiles are fetched without waiting for it.
- **Slabs**: `/api/tiles/slab` answers slab (level, index) of one metric over the finished runs of a folder
  (`Explorer.slab_body`). Each metric's finished-run overview and top tiles are decoded once each (`Explorer._stack`),
  and their buckets merged per level (`Explorer._level`, `tiles.level_parts` on threads; from overview tiles where
  they are fine enough), ahead of requests from the overview level down to the finest top level; both are kept up to
  `STACK_BYTES`. Those merged levels are saved beside the index (`<cache>/levels/`, `SavedLevels`: the finished runs'
  digest, each run's overview level, the levels' buckets), at most once a metric every `LEVELS_SAVE_EVERY` seconds and
  up to `TREX_LEVELS_MB` (least recently used deleted), and memory-mapped by a later Explorer whose finished runs and
  their kept tiles are the same (`Explorer._finished`), so a restarted server cuts slabs without reading tiles. A slab is then cut from its merged level (`tiles.cut`) in a few milliseconds; runs whose top tiles
  are coarser than the level use their tile at that level, read from the index in one query or built from the run
  files on a process pool (`build_tile`) and cached. Slabs are kept like bundles until a finished run's tiles of that metric change
  (`_slab_gens`); running runs are left out. A chart of more than `DENSITY_AUTO` runs in step x without smoothing
  (`Chart.canSlab`) draws its finished runs from slabs a bucket per bin wide (`kernel.binSlabRun`: the mean of the
  rows in the bin), its running ones from their columns: group statistics on a worker, or each run's bin means for a
  heatmap (`Chart.slabRows`). Its finished runs then need no tiles of their own; until its slabs come it draws from
  coarser ones here (`Data.bestSlabs`), and its first slabs are guessed from the runs' last steps (`Chart.slabGuess`).
  Workers fetch slabs straight into shared memory (`pool.fetchSlabOnWorker`), which the page takes over
  (`kernel.adoptStore`, `Data.slabs`, up to `SLAB_BYTES`); while idle, the store fetches the slabs one and two levels
  finer than each visible chart draws. `/api/info` states `server.PROTOCOL`; a page of another (`data.js`
  PROTOCOL) says so beside the status.
- **Workers**: every response carries COOP/COEP headers (`server.ISOLATION`), so the page is cross-origin isolated and
  `buildColumn` puts columns in `SharedArrayBuffer` chunks (`kernel.columnStore`). Each chart's group statistics run on
  one worker of the pool (`pool.js`), which keeps that chart's binnings; a draw round waits for its charts' workers and
  draws them together. `?shared=0`, or a page that is not isolated, computes them on the page instead.
- **Fetching ahead**: once the store is idle, `Data.prefetch` fetches one bundle at a time: top tiles of the metrics
  charts show, then overview tiles of charts not loaded yet, then top tiles of the rest (finished runs only). Top
  tiles are held (`Entry.held`, up to `PREFETCH_BYTES`) without changing what charts show, and installed when a plan
  wants them.
- **Rendering**: WebGL2 by default (`?gl=0` selects Canvas 2D). Above 300 lines a chart draws a
  density heatmap. No upload overwrites GPU data a queued draw may read: each draw's line table
  takes fresh rows of the table texture (`Renderer.bind`), and a changed column moves to a new slot
  (`LineSet.update`); only appends past a column's drawn points are written in place. Charts then
  never depend on how a driver orders uploads against earlier draws.

## Invariants (things that break silently if ignored)

- **Explorer is read-only.** Nothing under a runs directory is ever created or modified by
  `index.py`, `server.py`, `query.py` or the CLI, except `trex compact` (`compact.py`), which rewrites runs
  no other process has open: a new file, verified, then atomically renamed over `trex.sqlite`. `connect_ro` opens runs without a `-wal` file as
  `immutable` so SQLite does not create `-wal`/`-shm`; a test asserts directory contents are
  unchanged. Callers re-check the file signature after reading.
- **Sequence numbers are contiguous.** Rows and media are numbered 0, 1, 2, … per run; readers stop
  at a gap. The browser holds rows `[tiles_seq, seq)` of each running run and resyncs on any gap
  or count mismatch.
- **Events follow commits, in order.** `Explorer.apply` publishes after the index transaction
  commits. A run's `run` event comes after the rows and media it counts (a new run's comes
  first); the browser treats a `run` event whose counts it has not reached as lost data.
- **Cache versions.** Bump `CACHE_VERSION` in `index.py` whenever what the index stores, or how it
  derives it, changes. Bump the IndexedDB version in `data.js` when the stored entries or their
  keys change. Older caches are then rebuilt instead of silently misread.
- **Shared formats across languages.** Change all of these together:
  - tile encoding: `tiles.py` and `decodeTile` in `static/data.js`. Buckets leave NaN out; a bucket's mean, mean
    step, mean runtime and count are of its finite values, or of its infinities when it has no finite value (the
    mean then infinite, NaN with both signs); its min and max include the infinities;
  - tile response framing: `post_tiles` in `server.py` and `unframe` in `static/data.js`;
  - slab encoding: `tiles.slab` and `slabViews` in `static/kernel.js`; a slab's run binned (`binSlabRun`) is its rows'
    mean per bin;
  - local bucket merging: `tiles.coarsen` and `buildColumn` in `static/data.js` (count-weighted means over the
    buckets with a finite mean, or all of them when none has one). Points are drawn at their bucket's mean step,
    never the bucket center;
  - smoothing (time-weighted EMA, scale = span/1000 rounded to a power of two): `kernel.js`
    `Col.ensureSmooth`, `plot.js` `smoothScale`, `query.py` `twema`/`smooth_scale`;
  - group statistics (order-statistic median CI, Student-t mean CI, interquartile mean of ranks
    [floor(n/4), n - floor(n/4)) with Yuen's CI): `kernel.js` `agg` / `medianCiRank` / `iqmStats`, `plot.js` `bandOf`,
    `query.py` `stats` / `_iqm`. NaN is no value; ±inf are values (the mean infinite or NaN, the spread NaN, order
    statistics and the IQM finite while the infinities fall outside their ranks). A run's value in a bin is the mean
    of its finite points there, infinite only when it has none (`kernel.js` `binColumn`), as with tile buckets;
  - non-finite numbers as text: "nan", "inf", "-inf" (`index.wire`, `cli.jsonable`, `where`'s markers,
    `where.js` `nonFiniteText`);
  - run filters and field names: `where.py` and `static/where.js` (`compileWhere`, `runField`), `query.py` `get`;
    both suites run the cases in `tests/where_cases.json`.

  Tile encoding, live tails, slabs, smoothing and group statistics are checked across languages by `tests/shared_cases.json`, which
  `tests/test_shared_cases.py` writes from the Python side (`TREX_WRITE_CASES=1`) and `tests/shared_cases.test.mjs`
  reads.
- **Other sites cannot use the server.** `Handler._refusal` answers only requests whose Host is an
  IP address, `localhost`, this machine's name or an `--allow-host` name (DNS rebinding), and
  refuses a POST whose Origin is not its Host. The UI sends no cross-origin requests.
- **A live run is read on its writer's host or through its journal.** Never open another host's WAL:
  even read-only readers write SQLite's shared index, which a network filesystem does not keep
  coherent. A journal replayed from its start rebuilds the run's rows, meta and media exactly
  (`tests/test_journal.py`); its commits stay as written, since merges are not journaled.
- **Writer never crashes training.** Errors in the commit thread are recorded and raised from
  `finish()`, not from `log()`. Media files are written (temp name, then rename) before the row
  that references them commits.
- **A killed writer loses no committed row.** Every commit and every merge's swap is one SQLite
  transaction. A swap applies only if its commits are still exactly the ones it read, and a merged
  chunk either keeps its value bytes verbatim (dense) or must decode as readers decode it to the same
  values in the same rows (`chunks.prepare_merge`). A failed merge stops merging for that run and
  changes nothing; `finish()` abandons a merge not yet swapped in.
- **Run identity is its path** (relative to the served root). A changed `meta.id` or a shrunken
  row count makes the explorer drop and re-index the run.

## UI performance rules

The 10k-run view is the benchmark; interactions should reach the next painted frame in about
20 ms, and no task should block input for more than about 50 ms.
- Per-draw work is proportional to pixels, not points (`prep` decimates per pixel).
- Charts draw first. Sidebar, path bar and info panel update after the frame paints
  (`App.afterPaint`), one task each. Chart drawing stops at `FRAME_BUDGET_MS` per frame and
  continues on the next. Redraws for streamed data come at most 4 times a second, less often when the
  visible charts are expensive to draw, and keep the y axis while the lines fill most of it (`steadyY`). Work the
  view asked for draws at once when the data layer is idle (`Data.busy`), and tiles are planned after the charts
  draw, only when what the view shows or the data changed.
- Group statistics bin each column once per binning (`BinCache`, kept per chart while the column has the same
  points); a chart's groups are summarized in one `aggGroups` call. Order statistics come from histogram selection,
  exact; only bins of at most 64 values are sorted.
- A chart's lines (`App.linesFor`) and their x extent are kept until the runs drawn (`linesSig`) or the data
  (`Data.version`) change.
- Work over all runs is sliced (`Data.rebuildSoon`) or indexed (`ConfigIndex`); nothing that
  touches every run × every key runs on an interaction.
- The sidebar builds only the rows near its scroll position. Off-screen panels skip layout
  (`content-visibility`) and hold no canvas backing store.
- The browser smoke test streams runs while recording every chart draw (`flicker_smoke`); `TREX_TOP_REFRESH` sets
  how often growing runs get new top tiles (10 s; the smoke test uses 1 s).
- Measure before and after a change (time from the action to the next frame, plus long tasks),
  rather than assuming.

## Conventions

- Cyclomatic complexity is at most 15 per function, Python (radon) and UI JS (eslint's
  `complexity` rule); `tests/test_complexity.py` and `tests/complexity.test.mjs` enforce it, the latter also refusing
  unused variables in the UI. Split
  a function by what it does into named steps; keep hot inner loops in one function.

- Python via `uv run`; never bare `python`. Stdlib and numpy first. Adding a runtime dependency
  needs a very good reason.
- Types: every function is annotated. The logging API (`writer.py`) and the format modules
  (`format`, `chunks`, `journal`, `tiles`, `media`) pass pyright strict; the rest passes standard. Use PEP 695
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
