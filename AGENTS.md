# AGENTS.md

Guidance for coding agents working on **trex** (trackosaurus exp), a file-based experiment
explorer. To *use* trex to explore runs, read `trex --help` and `trex COMMAND --help`.

## Layout

| path | what |
|---|---|
| `trex/format.py` | the run file: `trex.sqlite` schema (format 3), `connect_rw`, `connect_ro`, `snapshot`. The contract between writer and readers. |
| `trex/journal.py` | commit journal for runs on network filesystems: `Writer` (append + fsync), `records`, `sync` (replay into a local replica) |
| `trex/chunks.py` | per-metric chunk codec: each commit is one `rowmeta` row (steps, times) plus one `chunk` per metric present; `prepare_merge` / `apply_merge` rewrite adjacent commits as one |
| `trex/buckets.py` | bucket arrays, the one form metric data takes between run files and charts: the `TKB1` format (`encode`, `decode`), `bucketize`, `merge`, `cut`, `union`, and `Stack` (many runs' buckets, each at its own level) |
| `trex/writer.py` | `trex.init` / `Run` / `folder_info`: logging API, background commit thread, and a merge thread that merges small commits |
| `trex/media.py` | PNG/MP4 encoding for logged arrays (MP4 through ffmpeg) |
| `trex/index.py` | `Explorer`: crawl, per-run scans (inline or process pool), each run's kept buckets, finished runs' merged levels (saved, memory-mapped), blocks built from run files, block answers (`buckets_body`) and a size-bounded memo of them (`Memo`), event hub, SSE messages (`messages`) |
| `trex/server.py` | read-only HTTP + SSE for the UI: `/api/runs`, `/api/buckets`, `/api/rows`, `/api/stream`, media |
| `trex/daemon.py` | `trex daemon`: `Roots` (tracked directories by spec, `unique_names`, workspaces; `roots.json`, remembered ones in `history.json`), the Unix control socket, its client |
| `trex/workspace.py` | workspaces: `Workspace` merges its members (`Member`: an `Explorer`, or `Far` for a remote one, which answers as an Explorer does) behind the Explorer interface, renaming run ids |
| `trex/remote.py` | `host:path` directories: `parse`, this trex as a wheel (`build_wheel`), the ssh + `uvx` command, `Remote` (one ssh session, reconnected with backoff) |
| `trex/compact.py` | `trex compact`: rewrites a run no process has open with its commits merged (exclusive lock, new file, verify, rename) |
| `trex/update.py` | the daemon's update: `uv tool install $TREX_SOURCE`, then exit `RESTART_STATUS` for systemd or launchd to restart it |
| `trex/query.py` | read-side queries for the CLI: records, field access, sorting, statistics, series |
| `trex/where.py` | run filters: a SQL WHERE clause (or a name search) compiled to a test over a field getter |
| `trex/cli.py` | `trex` command (Typer): `serve daemon systemd-unit launchd-plist ls groups keys tree show series tail media diff index compact` |
| `trex/static/` | UI, plain ES modules: `app.js` (page), `data.js` (block store, planner, stream), `plot.js` (charts), `gl.js` (WebGL2 renderer), `kernel.js` (bucket arrays, columns, smoothing, decimation, group stats), `pool.js` and `worker.js` (binning on workers over shared columns and bucket arrays, fetching into shared memory), `where.js` (run filters, run fields, filter completion); `index.html` |
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
- **Buckets** (`trex.buckets`): at level L, bucket b holds steps [b·2^L, (b+1)·2^L); block i of a level is its
  buckets [i·BLOCK, (i+1)·BLOCK). A bucket array holds some runs' buckets of one metric at one level: per bucket the
  mean of its finite values (of its infinities when it has none), their mean step, mean runtime and count; per run the
  rows its buckets hold (`seq`). Every view is made of `bucketize` (rows to buckets), `merge` (to a coarser level,
  count-weighted, as `bucketize` would make them) and `cut` (a block, or some runs).
- **Index** (`<cache>/<hash of root>/index.sqlite`): per-run metadata, last values, metric names, media, and each
  run's *kept* buckets of every metric: one array at the finest level whose blocks are as wide as the run
  (`buckets.level_for`), so at most two blocks. Growing runs get new kept buckets every `KEPT_REFRESH` seconds
  (`TREX_KEPT_REFRESH`) and when they finish or crash. The index never holds a full copy of the data.
- **Server**: `POST /api/buckets {key, level, index, scope | runs, which}` answers one block of one metric for a
  list of runs, or for the runs of a folder in state `which` (`all`, `finished`, `running`), as a bucket array
  (`Explorer.buckets_body`). A run that keeps its buckets at the level or finer is merged from them: a finished run
  from its metric's merged level (`Explorer._level`: every finished run's kept buckets decoded at once,
  `buckets.stack`, merged on threads), a running one from its kept buckets read from the index. A run that keeps
  coarser buckets has the block built from its run file (`build_block`, on a process pool beyond `INLINE_BUILDS`).
  The levels from a metric's
  coarsest kept one to `LEVELS_AHEAD` above it are merged ahead of requests and saved beside the index
  (`<cache>/levels/<metric>-<digest>/`, one `.npy` per array, memory-mapped when read), at most once a metric every `LEVELS_SAVE_EVERY` seconds and up to
  `TREX_LEVELS_MB` (least recently used deleted), and memory-mapped by a later Explorer whose finished runs and their
  kept buckets are the same (`Explorer._finished`). A folder's finished-run blocks are kept in memory
  (`Explorer._memo`, a `Memo` of `MEMO_BYTES` that also holds stacks and merged levels) until those runs change (`_gens`), and `/api/runs` answers until what it holds changes
  (`_view_gen`). `/api/info` states `server.PROTOCOL`; a page of another (`data.js` PROTOCOL) says so beside the
  status.
- **Daemon**: one server, one `Explorer` (or `Remote`) per tracked directory. A directory's URLs are its
  standalone URLs under `/r/<name>/`, a workspace's under `/w/<name>/`, and the UI prefixes every request
  with that (`BASE` in `data.js`); `/` is the root view, every tracked directory as a top-level folder
  (`Roots.everything`, a nested `Workspace`), and the trex brand opens the panel that manages them. Names follow Emacs's uniquify (`runs<chush>`)
  and change when a collision appears or ends; specs (paths, `host:path`) are the stable keys.
- **Workspace**: answers the Explorer interface by asking its members in parallel. A run id is the
  member's path, or `path<member>` when an earlier member holds the same path; `resolve` maps it back. Its block
  answers join its members' bucket arrays, run ids renamed (`Workspace.buckets_body`).
  Its stream merges the members' streams, renaming run ids in every event (`Workspace._rename_event`).
  `/api/daemon` lists the directories, the remembered ones and the running trex; directories are
  added over the control socket (mode 0600) or HTTP and removed over HTTP. An update installs
  `$TREX_SOURCE` and exits with `update.RESTART_STATUS`; the systemd unit's `RestartForceExitStatus`
  or the launchd agent's `KeepAlive` restarts it, and the UI reloads once `/api/daemon` reports the new install (`update.RUNNING`, read at start).
  A removed directory's `Explorer` is closed (`Explorer.close`). A `host:path` directory is a `Remote`:
  one `ssh -L <local socket>:<remote socket> host uvx --from ~/.cache/trex/wheels/<wheel> trex serve PATH --unix
  <remote socket> --exit-on-eof`, where `<wheel>` is this trex packaged as a wheel named by a digest of its contents
  (`remote.wheel`); a session that finds it missing says so, and the daemon writes it there over ssh and reconnects. And `Handler._proxy` passes its `/r/<name>/` requests (the stream too)
  to the local socket; the remote server ends when the session's stdin closes. Tests use a fake `ssh`
  and `uvx` (`tests/test_remote.py`).
- **Browser** (`data.js`): visible charts state what they show (`plan`, one demand per metric). A chart wants a coarse
  layer, blocks of one level over every step of its runs, and when zoomed a finer layer over its view (at most
  `FINE_BLOCKS` blocks): levels with buckets about `LINE_PX_PER_BUCKET` wide on screen (`DENSITY_PX_PER_BUCKET` for a
  chart of many runs) within a point budget, rounded so they change only when a span crosses a power of two, as kept
  levels do. When many finished runs lack a block, one request asks for the folder's finished runs; other runs are
  asked for by id, and a running run again once its kept buckets are newer. Answers fill one store, `Data.blocks`
  (block -> run -> its row of a bucket array); a chart shows its wanted layers once every block holds every run, and
  keeps showing the previous ones until then. A run drawn as a line has a column (`kernel.buildColumn`): its buckets in
  the shown blocks, a finer level's where its blocks lie and the coarse level's elsewhere, then the streamed rows those
  blocks do not hold, bucketed as the server would, so a live run looks the same when its buckets catch up. A chart of
  more runs than it draws one by one (`App.coarseAbove`: group statistics, or a heatmap) bins its finished runs from
  their buckets in its finest shown blocks (`Data.partsOf`), its running ones from their columns. Binning weights each
  point by the rows it stands for, so a bin's mean is the mean of the rows in it. Workers fetch blocks straight into
  shared memory (`pool.fetchArrayOnWorker`), which the page takes over (`kernel.adoptStore`, up to `ARRAY_BYTES`;
  blocks no chart uses go first).
- **Workers**: every response carries COOP/COEP headers (`server.ISOLATION`), so the page is cross-origin isolated and
  `buildColumn` puts columns in `SharedArrayBuffer` chunks (`kernel.columnStore`). Each chart's binning runs on one
  worker of the pool (`pool.js`), which keeps that chart's binnings; a run's buckets in one block become a column viewing
  the array (`kernel.runColumn`). A draw round waits for its charts' workers and draws them together. `?shared=0`, or a
  page that is not isolated, computes them on the page instead.
- **Fetching ahead**: once the store is idle, `Data.prefetch` fetches one block at a time (`Data.nextAhead`, while the
  blocks no chart uses hold less than `AHEAD_BYTES`): for each chart, visible ones first, its wanted layers when it
  shows none yet, else the two levels below its finest over the steps it shows. Fetched blocks wait in the store.
- **Rendering**: WebGL2 by default (`?gl=0` selects Canvas 2D). Above 300 lines a chart draws a
  density heatmap. No upload overwrites GPU data a queued draw may read: each draw's line table
  takes fresh rows of the table texture (`Renderer.bind`), and columns, which never change once built, each take a
  slot of their own after the others (`LineSet.update`). Charts then never depend on how a driver orders uploads
  against earlier draws.

## Invariants (things that break silently if ignored)

- **Explorer is read-only.** Nothing under a runs directory is ever created or modified by
  `index.py`, `server.py`, `query.py` or the CLI, except `trex compact` (`compact.py`), which rewrites runs
  no other process has open: a new file, verified, then atomically renamed over `trex.sqlite`. `connect_ro` opens runs without a `-wal` file as
  `immutable` so SQLite does not create `-wal`/`-shm`; a test asserts directory contents are
  unchanged. Callers re-check the file signature after reading.
- **Sequence numbers are contiguous.** Rows and media are numbered 0, 1, 2, … per run; readers stop
  at a gap. The browser holds the rows of each running run its blocks lack and resyncs on any gap
  or count mismatch.
- **Events follow commits, in order.** `Explorer.apply` publishes after the index transaction
  commits. A run's `run` event comes after the rows and media it counts (a new run's comes
  first); the browser treats a `run` event whose counts it has not reached as lost data.
- **Cache version.** Bump `CACHE_VERSION` in `index.py` whenever what the index stores, or how it
  derives it, changes. An older index is then rebuilt instead of silently misread.
- **Shared formats across languages.** Change all of these together:
  - bucket arrays: `buckets.py` and `bucketViews` in `static/kernel.js`. Buckets leave NaN out; a bucket's mean, mean
    step, mean runtime and count are of its finite values, or of its infinities when it has none (the mean then
    infinite, NaN with both signs);
  - columns from buckets and streamed rows: `buckets.bucketize` / `merge` and `buildColumn` in `static/kernel.js`
    (count-weighted means over the buckets with a finite mean, or all of them when none has one). Points are drawn at
    their bucket's mean step, never the bucket center;
  - smoothing (time-weighted EMA, scale = span/1000 rounded to a power of two): `kernel.js`
    `Col.ensureSmooth`, `plot.js` `smoothScale`, `query.py` `twema`/`smooth_scale`;
  - group statistics (order-statistic median CI, Student-t mean CI, interquartile mean of ranks
    [floor(n/4), n - floor(n/4)) with Yuen's CI): `kernel.js` `agg` / `medianCiRank` / `iqmStats`, `plot.js` `bandOf`,
    `query.py` `stats` / `_iqm`. NaN is no value; ±inf are values (the mean infinite or NaN, the spread NaN, order
    statistics and the IQM finite while the infinities fall outside their ranks). A run's value in a bin is the mean
    of its finite points there weighted by the rows each stands for, infinite only when it has none (`kernel.js`
    `binColumn`), as with buckets;
  - non-finite numbers as text: "nan", "inf", "-inf" (`index.wire`, `cli.jsonable`, `where`'s markers,
    `where.js` `nonFiniteText`);
  - run filters and field names: `where.py` and `static/where.js` (`compileWhere`, `runField`), `query.py` `get`;
    both suites run the cases in `tests/where_cases.json`.

  Bucket arrays, live tails, layered levels, binning, smoothing and group statistics are checked across languages by `tests/shared_cases.json`, which
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
  view asked for draws at once when the data layer is idle (`Data.busy`), and blocks are planned after the charts
  draw, only when what the view shows or the data changed; a zoom plans before it draws.
- Group statistics bin each column once per binning (`BinCache`, kept per chart while the column has the same
  points); a chart's groups are summarized in one `aggGroups` call. Order statistics come from histogram selection,
  exact; only bins of at most 64 values are sorted.
- A chart's lines (`App.linesFor`) and their x extent are kept until the runs drawn (`linesSig`) or the data
  (`Data.version`) change.
- Work over all runs is sliced (`Data.rebuildSoon`) or indexed (`ConfigIndex`); nothing that
  touches every run × every key runs on an interaction.
- The sidebar builds only the rows near its scroll position. Off-screen panels skip layout
  (`content-visibility`) and hold no canvas backing store.
- The browser smoke test streams runs while recording every chart draw (`flicker_smoke`); `TREX_KEPT_REFRESH` sets
  how often growing runs get new kept buckets (10 s; the smoke test uses 1 s).
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
  (`format`, `chunks`, `journal`, `buckets`, `media`) pass pyright strict; the rest passes standard. Use PEP 695
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
