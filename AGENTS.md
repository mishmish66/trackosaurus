# AGENTS.md

Guidance for coding agents working on **trex** (trackosaurus exp), a file-based experiment explorer. To *use* trex to
explore runs, read `trex --help` and `trex COMMAND --help`. The user guide is the `trex` and `trex.node` module
docstrings (Markdown, published by `docs/build.py`); the README's "One trex, many machines" covers the same ground.

## Layout

| path | what |
|---|---|
| `trex/format.py` | the run file: `trex.sqlite` schema (format 3), `connect_rw`, `connect_ro`, `snapshot`; `JSONValue` and the `as_*` helpers that narrow JSON. The contract between writer and readers. |
| `trex/journal.py` | commit journal for runs on network filesystems: `Writer` (append + fsync), `records`, `sync` (replay into a local replica) |
| `trex/chunks.py` | per-metric chunk codec: each commit is one `rowmeta` row (steps, times) plus one `chunk` per metric present; `metric` (one metric's `Series`), `rows`; `prepare_merge` / `apply_merge` rewrite adjacent commits as one |
| `trex/buckets.py` | bucket arrays, the one form metric data takes between run files and charts: `TKB1` (`encode`, `decode`, `frame`), `bucketize`, `merge`, `cut`, `refine`, `union`, `Stack`; arrays put together as bytes (`join`, `chain`); one run's levels: `pyramid` (from every row) and `grow` (the blocks new rows change) |
| `trex/writer.py` | `trex.init` / `Run` / `folder_info`: logging API, background commit thread, and a merge thread that merges small commits |
| `trex/media.py` | PNG/MP4 encoding for logged arrays (MP4 through ffmpeg) |
| `trex/index.py` | `Explorer`: a directory's index (`index.sqlite`), kept current by its `Origin`, and everything answered from it: run lists, blocks (`buckets_body`; stored ones read by `stored_block`, on the block workers, `Workers`), rows, the event hub and SSE stream (`messages`), dumps for other trex (`dump`). The records (`RunRecord`, `Update`, `Dump`, `Have`, `RunMeta`, `Rows`, ...); finished runs' merged levels (saved as `.npy`, memory-mapped); `Memo` |
| `trex/crawl.py` | `Crawl`, the origin of a runs directory on this machine: walks it, scans changed runs (inline or on a process pool), compiles their levels |
| `trex/mirror.py` | `Pull`, the origin of a directory another trex holds: its run list, dumps, media copies, its stream, running runs' tails. `Upstream`: that trex's API over http or a Unix socket, connections kept alive |
| `trex/node.py` | `Node`: what one trex process holds and serves: directories by id (crawled here, or pulled through a `Link`), links, workspaces, display names, `holdings`, `reconcile`; saved state (`Saved`, `Identity`). Its docstring is the user guide for running trex. |
| `trex/control.py` | this machine's trex: its state directory (`$TREX_DAEMON_DIR`) and control socket (`ControlServer`, `request`) |
| `trex/remote.py` | `host:path` directories: `parse`, this trex as a wheel (`build_wheel`), the ssh + `uvx` command, `Remote` (one ssh session per host, reconnected with backoff) |
| `trex/workspace.py` | `Workspace`: members (`Member`: a name and an Explorer) behind the Explorer interface, run ids renamed; merged (a named workspace) or nested (a node's home view) |
| `trex/server.py` | HTTP + SSE for a node: the UI, `/api/node*`, `/api/holdings`, and per directory or workspace `/api/runs`, `/api/buckets`, `/api/rows`, `/api/stream`, `/api/dumps`, media |
| `trex/compact.py` | `trex compact`: rewrites a run no process has open with its commits merged (exclusive lock, new file, verify, rename) |
| `trex/update.py` | a node's update: `uv tool install $TREX_SOURCE`, then exit `RESTART_STATUS` for systemd or launchd to restart it |
| `trex/query.py` | read-side queries for the CLI: records, field access, sorting, statistics, series |
| `trex/where.py` | run filters: a SQL WHERE clause (or a name search) compiled to a test over a field getter |
| `trex/cli.py` | `trex` command (Typer): `serve systemd-unit launchd-plist ls groups keys tree show series tail media diff index compact`; `daemon` is a hidden alias of `serve` that service files written by older versions run |
| `trex/static/` | UI, plain ES modules: `app.js` (page, node panel), `data.js` (block store, planner, stream), `plot.js` (charts), `gl.js` (WebGL2 renderer), `kernel.js` (bucket arrays, columns, smoothing, decimation, group stats), `pool.js` and `worker.js` (binning on workers over copies of the columns and bucket arrays they read, fetching), `where.js` (run filters, run fields, filter completion); `index.html` |
| `typings/` | type stubs for untyped dev dependencies (radon) |
| `examples/demo.py` | synthetic sweeps and live runs for trying the UI |
| `docs/build.py` | pdoc pages of every module into `site/` |
| `docs/media/` | the docs' video tour and its poster image (left out of the sdist); the README embeds the same video, uploaded to GitHub |
| `tests/` | pytest suites (`test_processes.py` runs real `trex` processes; `test_node.py`, `test_mesh.py`, `test_remote.py`, `test_mirror.py` cover nodes, links and ssh; shared helpers in `helpers.py` and `conftest.py`, including a fake `ssh` and `uvx`), the node tests (`*.test.mjs`), the cross-language cases (`where_cases.json`, `shared_cases.json`), and the browser smoke test |

Runtime dependencies: Python ≥ 3.12, numpy, and Typer for the CLI (ffmpeg only to log frame arrays as video). The UI
loads no external scripts, and the server is standard library.

## Commands

```bash
uv sync && npm install                                    # dev deps: pytest, pytest-cov, pytest-xdist, pyright, radon, pdoc, playwright; eslint (UI tests only)
uv run pyright                                            # strict, over trex/, tests/, examples/ and docs/
uv run pytest                                             # all suites, one worker per CPU (-n0: serially)
uv run pytest --cov                                       # the same with branch coverage, subprocesses included; fails below 94%
uv run python docs/build.py                               # pdoc pages into site/
node --test tests/*.test.mjs                              # JS kernel, shared cases, complexity and unused variables (node >= 18)
node --test --experimental-test-coverage --test-coverage-include='trex/static/kernel.js' \
  --test-coverage-include='trex/static/where.js' --test-coverage-lines=95 --test-coverage-branches=85 \
  tests/*.test.mjs                                        # the same with CI's coverage floors (node >= 22.8; see below)
uv run python tests/browser_smoke.py                      # headless UI on its own throwaway trex; fails below 92% UI line coverage
                                                          # (Chromium once: uv run playwright install chromium)
uv run python examples/demo.py /tmp/runs && uv run trex serve /tmp/runs --temporary
```

Run pyright and the three suites after any change that touches their area. CI (`.github/workflows/test.yml`) runs
pyright, `pytest --cov` and the node tests with their coverage floors, on Python 3.12 and Node 22; the smoke test stays
local. It measures which lines of the UI modules run (V8 coverage over every page it loads), fails below `UI_COVERAGE`,
and writes the uncovered lines to `ui_coverage.txt` beside its screenshots. It starts its own trex (`trex serve
--temporary`, and nodes with a private `TREX_DAEMON_DIR`) on temporary directories and free ports. Never point tests at
a trex someone is using: tests that start this machine's trex give it a private `TREX_DAEMON_DIR`.

The node tests have coverage floors of their own, which only the second `node` command above checks: of `kernel.js`
and `where.js` together they must run 95% of the lines and 85% of the branches, or the command exits 1 though every
test passes. Its report ends with each file's percentages and the numbers of the lines not run. Only node tests count,
not what the smoke test runs in a browser, so code added to either file needs a node test of what it does
(`tests/kernel.test.mjs`; `tests/shared_cases.test.mjs` for what is shared with Python; `tests/where.test.mjs`).
`--test-coverage-include` needs Node 22.5 and the two floors Node 22.8; no Node 18 or 20 has them (`bad option`). With
an older `node`, a plain `node --test` passing says nothing about the floors: run the command with a Node 22 release
unpacked from nodejs.org (`<dir>/bin/node --test ...`; the tests need nothing else of it), or run `node --test
--experimental-test-coverage tests/*.test.mjs`, which prints the same percentages for every file it loads and never
fails on them, and read its `kernel.js` and `where.js` rows (its `all files` row counts the test files too).

Headless Chromium reaches the GPU only with
`--headless=new --use-gl=angle --use-angle=gl-egl --ignore-gpu-blocklist --enable-gpu`; without them WebGL falls back
to software and timings mean nothing.

## Architecture

### Nodes

Every trex process is a node (`Node`), started by the one command `trex serve`. A node holds runs directories and
serves them, with the UI, to browsers and to other trex.

- **This machine's trex** saves what it holds in `$TREX_DAEMON_DIR` (default `~/.local/state/trex`) and listens on the
  control socket there (`daemon.sock`, mode 0600). `trex serve DIR...` while it runs hands it the directories over the
  socket (asking first unless `-y` or stdin is not a tty) and exits; otherwise it becomes this machine's trex. Run as a
  service (`trex systemd-unit`, `trex launchd-plist`), it updates itself from the panel (`trex.update`).
- **A temporary node** (`trex serve --temporary`) saves nothing and ignores the control socket, so it runs beside this
  machine's trex. Given one local directory, `/` shows that directory alone (`Node.home`).
- **A remote node** is a temporary node that another node starts on a host over ssh (`trex.remote`): `uvx` runs this
  trex's wheel there with `trex serve --temporary --unix SOCK --exit-on-eof --name HOST`, `ssh -L` forwards its Unix
  socket back, and it ends when the session's stdin closes. The wheel is named by a digest of its contents and kept in
  `~/.cache/trex/wheels/` there; a session that finds it missing prints `NEEDS_WHEEL`, and the node uploads it over ssh
  and reconnects. Its index lives on the host's local disk (`$TMPDIR/trex-cache-<uid>`).
- **Several nodes on one machine** share the cache directory but never an index: each Explorer holds its index
  directory's `lock` (flock), and another Explorer of the same origin takes the next free slot (`<hash>-1`, `<hash>-2`,
  ...; `index._free_index`).

A node holds directories by **id**, the same wherever a directory is held: `<node name>:<path>` for one a node crawls
(`--name`, by default the host's, saved in `node.json`), `host:path` for one crawled over ssh. `Node.entries` maps
each id to its Explorer; `crawled` (id -> path) holds the ones crawled here, `pulled` (id -> `Pulled(link, via)`) the
ones pulled through a link. Display names are unique over the paths of crawled directories and the ids of the others,
as Emacs's uniquify makes them (`runs<chush>`, `unique_names`), and change when a collision appears or ends.

### Explorer, index and origins

An `Explorer` is a directory's index plus everything answered from it; its `Origin` keeps the index current and
supplies only what the index does not hold: rows beyond the compiled levels (`rows_json`, `live`) and media files
(`media_file`).

- **`Crawl`** (`trex.crawl`): a runs directory on this machine. It walks the root for runs (`REWALK` s), polls known
  runs' file signatures (`POLL` s), and scans each changed run (`scan`, inline or on a process pool beyond
  `INLINE_BYTES` of growth) into an `Update`: record, new media, last values, and compiled levels. Rows beyond the
  levels and media files come from the run files.
- **`Pull`** (`trex.mirror`): a directory another trex holds, through an `Upstream`. It never forwards a browser's
  request: its Explorer answers from its own index and tails, also while the upstream is unreachable (see Transfer).

`Explorer.apply(updates)` writes updates in one index transaction, then publishes their events. `Explorer.dump(path,
have)` is the other direction: what a mirror holding `have` of a run lacks.

### Storage

- **Run file** (`trex.sqlite`, SQLite in WAL mode, plus `media/`): `meta`, `media`, `keys`, `rowmeta(seq0, n, step_lo,
  step_hi, data: steps then times)` and `chunk(key_id, seq0, data: values)`. One commit of at most 65535 rows is one
  `rowmeta` row and one chunk per metric logged in it, so reading one metric is an index range scan. A second writer
  thread merges the session's newest small commits (`writer.merge_plan`: `FAN_IN` commits of a value tier at a time, at
  most `MERGE_VALUES` values, never a `sealed` commit), so a run logging a row per commit stays near 8 bytes per value.
  It reads and builds each merge outside any write lock and swaps it in with one short transaction, so commits wait
  only for the swap. Readers find rows by the commits that overlap them and never depend on where commits begin or end.
- **Journal** (`trex.journal`, runs on network filesystems, or `TREX_JOURNAL=1`): every commit's inserts, appended and
  fsync'd, because SQLite's WAL is readable only on the writer's host. `connect_ro` reads a live journaled run (one
  with a `-wal`) from a replica in `$TREX_REPLICAS` (default `<tmp>/trex-<uid>/replicas`), brought up to date from the
  journal on each open.
- **Buckets** (`trex.buckets`): at level L, bucket b holds steps [b·2^L, (b+1)·2^L); block i of a level is its buckets
  [i·BLOCK, (i+1)·BLOCK). A bucket array holds some runs' buckets of one metric at one level: per bucket the mean of its
  finite values (of its infinities when it has none), their mean step, mean runtime and count; per run the rows its
  buckets hold (`seq`). Every view is made of `bucketize` (rows to buckets), `merge` (to a coarser level,
  count-weighted, as `bucketize` would make them) and `cut` (a block, or some runs).
- **Index** (`<cache>/<sha1 of the origin's key>/index.sqlite`, WAL; `CACHE_VERSION`): tables
  - `runs(path, record)`: each run's `RunRecord` as JSON (`wire`): uid, seq (rows), mseq (media), keys, summary (last
    values), sig (file signature), heartbeat, public metadata, state, `compiled` (the rows its levels hold),
    `compiled_t`, `rebuilt` (`compiled` when its levels were last compiled from every row) and `ver`;
  - `media(path, seq, step, key, kind, file)`;
  - `metrics(path, key, fine, top, lo, hi)`: per run and metric, its `Span`: the levels it is stored at, from `fine`
    (about one row per bucket, from the median step spacing, `buckets.finest`) to `top` (the finest at which its steps
    [lo, hi] lie in at most two blocks, `buckets.level_for`);
  - `levels(key, level, block, path, since, data)`: the run's buckets of a metric at every level from fine to top, one
    row per block holding buckets: a zlib-compressed one-run TKB1 array (`pack`, `unpack`) and `since`, the rows
    compiled when the block last changed;
  - `folders(path, info)`: folder notes (`trex_info.json`); `cache(key, value)`: the cache version.
- **Compiling** (`crawl._compile`): a new run, or one with fewer than `REBUILD_BELOW` rows compiled, is compiled from
  every row (`buckets.pyramid`), so its finest level follows its steps; after that only the blocks its new rows fall in
  and the blocks above them change (`buckets.grow`, merged upward), plus any new top levels. A growing run is compiled
  every `REFRESH` seconds (`TREX_REFRESH`, 10), and at once when it stops running.
- **Merged levels** (`<index dir>/levels/<sha1 of metric>-<digest>/`, one `.npy` per array, memory-mapped when read):
  the finished runs' top levels of a metric merged to each level from its coarsest top to `LEVELS_AHEAD` above,
  merged ahead of requests on a thread, saved at most once a metric every `LEVELS_SAVE_EVERY` seconds and up to
  `TREX_LEVELS_MB` (least recently used deleted). A later Explorer whose finished runs and their `compiled` and `ver`
  are the same (`Finished.sig`) maps them instead of merging again.
- **In memory**: `Explorer._memo` (`Memo`, `MEMO_BYTES`) holds stacks, merged levels and a folder's finished-run block
  answers until those runs change (`_gens`); `/api/runs` answers are kept until what the index holds changes.
- **Media files**: a crawled run's are its own `media/` files; a pulled directory's are copied into `<index
  dir>/media/`, named by their contents, so one copy serves every run.
- **Node state** (`$TREX_DAEMON_DIR`): `node.json` (`Identity`: id, name), `roots.json` (`Saved`: tracked directories,
  links, pulled directories with their link and via, workspaces), `history.json` (directories and links to offer
  again), `daemon.sock`. Pulled directories are reopened from the cache at start, so they answer before their links do.

### Answering a block

`POST /api/buckets {blocks: [{key, level, index, scope | runs, which}]}` (at most `MAX_ASKS`) answers each block of
one metric for a list of runs, or for the runs of a folder in state `which` (`all`, `finished`, `running`), as a bucket
array, all in one body (`buckets.frame`). Per run (`Explorer._block`):

- a finished run whose top level is `level` or finer: cut from the metric's merged level (`_level`: every finished
  run's top-level blocks decoded at once, `buckets.stack`, merged on `MERGE_THREADS` threads, or the saved `.npy`);
- a running run whose top level is `level` or finer: merged from its top level read from the index (`_tops`);
- any other run: its stored block (`stored_block`), as it is stored (`buckets.join` puts the runs' one-run arrays
  together without decoding them); below its finest level, its finest level's buckets refined (`buckets.refine`: each
  bucket placed at its mean step).

Each run's `seq` in the answer is its `compiled`; the browser adds the rows beyond it from the stream.

Stored blocks of more than `BLOCK_ALONE` runs are read by the **block workers** (`index.Workers`): `BLOCK_WORKERS`
processes, each reading a slice of the runs (at most `SLICE`; rows and decompression on `READ_THREADS` threads) from
its own read-only connections to the index, the slices' arrays chained as bytes (`buckets.chain`). The first such
block starts them on a thread of their own; until every worker has answered, and whenever they fail, the asking thread
reads the block itself, one such block at a time. While the workers are up, a request's asks for such blocks are
answered at once (`at_once`), each waiting for its workers, the smaller asks meanwhile by the request's thread.
Workers end with the process that started them (`worker_init`), also when it is killed.

### Transfer between nodes

- **Links** (`Link`): trex a node pulls every directory of: `Link.http(url)`, or `Link.ssh(host)`, the remote node it
  starts (one per host for all the directories it tracks there). Every `PULL_EVERY` seconds the node asks each link for
  `GET /api/holdings` ({node: {id, name}, dirs: [{id, via}]}, `via` the nodes a directory came through, from the one
  crawling it); an ssh link is first asked to crawl the paths it lacks (`POST /api/node/add {path, id}`).
- **Reconcile** (`Node.reconcile`): each directory the links offer that the node does not crawl is pulled through the
  link offering it by the shortest `via` not containing this node; a directory keeps its link while that link offers it
  (else its `Pull` is retargeted, `Pull.retarget`); a directory goes once its link answers without it, while an
  unreachable link keeps what it offered. So links may form cycles, and each directory is held once per node.
- **Pull** (`trex.mirror`): reads `GET /api/runs?path=` at start, after the stream reconnects and every `LIST_EVERY`
  seconds, and marks every run whose `ver` differs from its own. It dumps marked runs (`POST /api/dumps {runs: [{path,
  uid, mseq, compiled, rebuilt}]}`, `DUMPS_AT_ONCE` a request, a running run at most every `RUNNING_EVERY` seconds),
  copies their new media files, then applies them. A `Dump` holds the record, the media items the mirror lacks, the
  metrics' spans, and blocks: every block (`replace`) when the mirror holds nothing, another uid or another `rebuilt`;
  else the blocks whose `since` is beyond the mirror's `compiled`.
- **The stream**: the Pull follows the upstream's `/api/stream`. A `run` event whose `ver` the index lacks, or a
  `media` event, marks the run; `delete` drops it; `folder` updates folder notes; `rows` events extend running runs'
  **tails** (rows beyond their levels, fetched once by `GET /api/rows?path=&from=<compiled>` while the stream is up)
  and go on to the node's own stream. A tail is trimmed to the rows beyond `compiled` after each apply, ends when a gap
  appears (fetched anew) and is dropped when the run stops.
- **Wire**: `Upstream` keeps up to `IDLE` http connections open; ssh links reach the remote node's Unix socket through
  the forwarded local socket. A server compresses bodies except to loopback clients; a client on the Unix socket counts
  as remote, so what crosses ssh is compressed.

### Server and UI routes

`/` is the node's home view (`Node.home_view`: its home directory when set, else every directory, each a top-level
folder, as a nested `Workspace`), `/d/<id>/` one directory, `/w/<name>/` a workspace. Under each, the same API:
`/api/info` (with `server.PROTOCOL`), `/api/tree`, `/api/runs`, `/api/run`, `/api/rows`, `/api/stream`,
`/api/buckets`, `/api/dumps`, `/m/<run>/media/<file>`. At the node: `/api/node` (identity, whether it saves, its home,
directories, links, workspaces, history, the trex it runs and whether it can update), `/api/node/add`, `/remove`,
`/update`, `/workspace`,
`/workspace/delete`, `/history/clear`, and `/api/holdings`. The UI prefixes every request with its view's base
(`BASE` in `data.js`), and the trex brand opens the panel that manages directories, links, workspaces and updates. A
page whose `PROTOCOL` (in `data.js`) differs from the server's says so beside the status. An update installs
`$TREX_SOURCE` and exits with `update.RESTART_STATUS`; the systemd unit's `RestartForceExitStatus` or the launchd
agent's `KeepAlive` restarts it, and the UI reloads once `/api/node` reports the new install (`update.RUNNING`, read at
start).

### Workspaces

A `Workspace` answers the Explorer interface by asking its members in parallel. Merged, a run id is the member's path,
or `path<member>` when an earlier member holds the same path (`resolve` maps it back); nested, each member is a
top-level folder named for it. For each block it asks the members for their parts (at once, as an Explorer answers
asks) and chains their bucket arrays as bytes, each member's runs in turn, run ids renamed
(`Workspace.buckets_bodies`, `buckets.chain`). Its stream merges the members' streams, renaming run ids in every
event. Every run gets a `dir` field: its member's name.

### Browser

- **Planning** (`data.js`): visible charts state what they show (`plan`, one demand per metric). A chart wants a coarse
  layer, blocks of one level over every step of its runs, and when zoomed a finer layer over its view (at most
  `FINE_BLOCKS` blocks): levels with buckets about `LINE_PX_PER_BUCKET` wide on screen (`DENSITY_PX_PER_BUCKET` for a
  chart of many runs) within a point budget, rounded so they change only when a span crosses a power of two, as top
  levels do. When many finished runs lack a block, one request asks for the folder's finished runs; other runs are
  asked for by id, and a running run again once its `compiled` passes the rows its block holds (`Data.current`).
  Requests go out in batches of at most `BATCH_BLOCKS` blocks, spread over the free request slots (`Data.pump`), the
  chart last pressed first (`App.lead`).
- **The store**: answers fill one store, `Data.blocks` (block -> run -> its row of a bucket array); a chart shows its
  wanted layers once every block holds every run, and keeps showing the previous ones until then. A run drawn as a line
  has a column (`kernel.buildColumn`): its buckets in the shown blocks, a finer level's where its blocks lie and the
  coarse level's elsewhere (in step order without sorting: the finer blocks, when they join into one range, replace
  the coarse buckets inside it, `emitBuckets`), then the streamed rows those blocks do not hold, bucketed as the server
  would, so a live run looks the same when its levels catch up. Columns are rebuilt in tasks (`Data.rebuildSome`:
  `REBUILD_SLICE_MS` at a time, going on up to `REBUILD_WHOLE_MS` to finish a metric), only those whose blocks or
  layers changed, and the UI is told of a metric once none of its columns awaits rebuilding, so a chart draws its runs'
  new columns together. A chart of more runs than it draws one by one (`App.coarseAbove`: group
  statistics, or a heatmap) bins its finished runs from their buckets in its finest shown blocks (`Data.partsOf`), its
  running ones from their columns. Binning weights each point by the rows it stands for, so a bin's mean is the mean
  of the rows in it. Workers fetch blocks, each array of a batch into a buffer of its own handed to the page
  (`pool.fetchArraysOnWorker`), which keeps it (`kernel.adoptStore`, up to `ARRAY_BYTES`; blocks no chart uses go
  first).
- **Workers**: columns and bucket arrays live in located chunks (`kernel.columnStore`, `adoptStore`: chunk,
  generation, offset). Each chart's binning runs on one worker of the pool (`pool.js`), which keeps that chart's
  binnings and is sent a copy of each column and array its jobs read, once (`pool.sendCopies`), dropped when the page
  frees the chunk; a run's buckets in one block become a column viewing the array (`kernel.runColumn`). A draw round
  waits for its charts' workers and draws them together. Nothing needs shared memory, so plain http on any host works
  the same.
- **Fetching ahead**: while no plan's requests are under way, `Data.prefetch` keeps `PREFETCH_PARALLEL` batches in
  flight (`Data.nextAhead`, each request once a page, while the blocks no chart uses hold less than `AHEAD_BYTES`):
  every chart's wanted layers first, visible charts first and then the nearest the view (`App.aheadOf`), then the two
  levels below the finest each shown chart shows over the steps it shows. Fetched blocks wait in the store, so a
  scroll finds the charts' blocks already there. While a zoom is dragged, the blocks the charts would want for it are
  fetched ahead too, for where the drag is every `AIM_MS` (`App.aimZoom`, `Data.fetchFor`), so a zoom mostly finds its
  blocks there at release.
- **Rendering**: WebGL2; a browser without it gets no charts, and a lost context keeps the charts as drawn until it is
  restored. Above 300 lines a chart draws a density heatmap. No upload overwrites GPU data a queued draw may read: each
  draw's line table takes fresh rows of the table texture (`Renderer.bind`), and columns, which never change once
  built, each take a slot of their own after the others (`LineSet.update`; a set whose columns are mostly new is
  uploaded whole). Charts then never depend on how a driver orders uploads against earlier draws. A draw's instances
  are each line's segments in view (`LineSet.tableFor`), not all of its points.

## Invariants (things that break silently if ignored)

- **Explorers are read-only.** Nothing under a runs directory is ever created or modified by `index.py`, `crawl.py`,
  `mirror.py`, `server.py`, `query.py` or the CLI (a node writes only its cache and state), except `trex compact`
  (`compact.py`), which rewrites runs no other process has open: a new file, verified, then atomically renamed over
  `trex.sqlite`. `connect_ro` opens runs without a `-wal` file as `immutable` so SQLite does not create `-wal`/`-shm`;
  a test asserts directory contents are unchanged. Callers re-check the file signature after reading.
- **Sequence numbers are contiguous.** Rows and media are numbered 0, 1, 2, … per run; readers stop at a gap. The
  browser and a Pull hold the rows of each running run beyond its levels, and resync on any gap or count mismatch.
- **Events follow commits, in order.** `Explorer.apply` publishes after the index transaction commits. A run's `run`
  event comes after the rows and media it counts (a new run's comes first); the browser treats a `run` event whose
  counts it has not reached as lost data. It opens the stream before it lists a scope's runs and holds the events until
  the list arrives (`Data.loadScope`), so no run that appears in between is missed.
- **A run's version is set where it is crawled.** `apply` bumps `ver` on every change of a crawled run (`Update.ver`
  None); a Pull copies the upstream's. Mirrors decide what to dump by comparing `ver`, so a change that leaves `ver` as
  it was never reaches them.
- **A stored block changes only with its `since`.** Every compile writes `since` = the rows compiled for each block it
  changes, and a dump sends the blocks whose `since` is beyond what the mirror holds. A full compile sets `rebuilt`, and
  a mirror whose `rebuilt` differs takes every block (`replace`), because a full compile can change any block.
- **Mirrors never forward requests.** Everything a Pull's Explorer answers comes from its index, its media copies and
  its tails; the upstream is reached only to take what changed.
- **One Explorer per index directory** (its `lock`). Two processes of the same origin get separate indexes.
- **Cache version.** Bump `CACHE_VERSION` in `index.py` whenever what the index stores, or how it derives it, changes.
  An older index is then rebuilt instead of silently misread. Bump `server.PROTOCOL` and `data.js` `PROTOCOL` together
  whenever what the UI and the server say to each other changes.
- **Shared formats across languages.** Change all of these together:
  - bucket arrays and their framing: `buckets.py` (`encode`, `frame`) and `bucketViews`, `unframe` in
    `static/kernel.js`. Buckets leave NaN out; a bucket's mean, mean step, mean runtime and count are of its finite
    values, or of its infinities when it has none (the mean then infinite, NaN with both signs);
  - columns from buckets and streamed rows: `buckets.bucketize` / `merge` and `buildColumn` in `static/kernel.js`
    (count-weighted means over the buckets with a finite mean, or all of them when none has one). Points are drawn at
    their bucket's mean step, never the bucket center;
  - smoothing (time-weighted EMA, scale = span/1000 rounded to a power of two): `kernel.js` `Col.ensureSmooth`,
    `plot.js` `smoothScale`, `query.py` `twema`/`smooth_scale`;
  - group statistics (order-statistic median CI, Student-t mean CI, interquartile mean of ranks [floor(n/4), n -
    floor(n/4)) with Yuen's CI): `kernel.js` `agg` / `medianCiRank` / `iqmStats`, `plot.js` `bandOf`, `query.py`
    `stats` / `_iqm`. NaN is no value; ±inf are values (the mean infinite or NaN, the spread NaN, order statistics and
    the IQM finite while the infinities fall outside their ranks). A run's value in a bin is the mean of its finite
    points there weighted by the rows each stands for, infinite only when it has none (`kernel.js` `binColumn`), as
    with buckets;
  - non-finite numbers as text: "nan", "inf", "-inf" (`index.wire`, `cli.jsonable`, `where`'s markers, `where.js`
    `nonFiniteText`);
  - run filters and field names: `where.py` and `static/where.js` (`compileWhere`, `runField`), `query.py` `get`;
    both suites run the cases in `tests/where_cases.json`.

  Bucket arrays and their framing, live tails, layered levels, binning, smoothing and group statistics are checked
  across languages by `tests/shared_cases.json`, which `tests/test_shared_cases.py` writes from the Python side
  (`TREX_WRITE_CASES=1`) and `tests/shared_cases.test.mjs` reads.
- **Other sites cannot use the server.** `Handler._refusal` answers only requests whose Host is an IP address,
  `localhost`, this machine's name or an `--allow-host` name (DNS rebinding), and refuses a POST whose Origin is not its
  Host. The UI sends no cross-origin requests.
- **A live run is read on its writer's host or through its journal.** Never open another host's WAL: even read-only
  readers write SQLite's shared index, which a network filesystem does not keep coherent. A journal replayed from its
  start rebuilds the run's rows, meta and media exactly (`tests/test_journal.py`); its commits stay as written, since
  merges are not journaled. This is why a cluster's runs are crawled by a remote node on a host that sees them.
- **Writer never crashes training.** Errors in the commit thread are recorded and raised from `finish()`, not from
  `log()`. Media files are written (temp name, then rename) before the row that references them commits; a Pull copies
  media files before applying the rows naming them.
- **A killed writer loses no committed row.** Every commit and every merge's swap is one SQLite transaction. A swap
  applies only if its commits are still exactly the ones it read, and a merged chunk either keeps its value bytes
  verbatim (dense) or must decode as readers decode it to the same values in the same rows (`chunks.prepare_merge`). A
  failed merge stops merging for that run and changes nothing; `finish()` abandons a merge not yet swapped in.
- **Run identity is its path** (relative to its directory's root). A changed `meta.id` or a shrunken row count makes
  the crawl drop and re-index the run.

## UI performance rules

The 10k-run view is the benchmark; interactions should reach the next painted frame in about 20 ms, and no task should
block input for more than about 50 ms.
- Per-draw work is proportional to pixels, not points (`prep` decimates per pixel).
- Charts draw first. Sidebar, path bar and info panel update after the frame paints (`App.afterPaint`), one task each.
  Chart drawing stops at `FRAME_BUDGET_MS` per frame and continues on the next. Redraws for streamed data come at most
  4 times a second, less often when the visible charts are expensive to draw, and keep the y axis while the lines fill
  most of it (`steadyY`). Work the view asked for draws on the next frame; after a zoom, a chart that shows lines
  waits up to `HOLD_MS` for the columns being rebuilt for it (`App.due`), so it redraws once, with all of them. Blocks
  are planned after the charts draw, only when what the view shows or the data changed; a zoom plans before it draws
  (`App.setXRange`).
- Group statistics bin each column once per binning (`BinCache`, kept per chart while the column has the same points);
  a chart's groups are summarized in one `aggGroups` call. Order statistics come from histogram selection, exact; only
  bins of at most 64 values are sorted.
- A chart's lines (`App.linesFor`) and their x extent are kept until the runs drawn (`linesSig`) or the data
  (`Data.version`) change.
- Work over all runs is sliced (`Data.rebuildSoon`) or indexed (`ConfigIndex`); nothing that touches every run × every
  key runs on an interaction.
- The sidebar builds only the rows near its scroll position. Off-screen panels skip layout (`content-visibility`) and
  hold no canvas backing store.
- The browser smoke test streams runs while recording every chart draw (`flicker_smoke`); `TREX_REFRESH` sets how
  often growing runs are compiled (10 s; the smoke test uses 1 s).
- Measure before and after a change (time from the action to the next frame, plus long tasks), rather than assuming.
  Measure each kind of chart the change touches, since they take different paths: runs drawn one by one (`group=run`)
  come from the GPU line sets; grouped charts, which a page opens with (`DEFAULT_GROUP`, or the folder's `group_by`),
  bin their runs' columns on workers and redraw when those answer; charts of more runs than `App.coarseAbove` bin from
  buckets. A zoom has two times to measure: its first redraw, and the redraw with its detail.

## Conventions

- **Records are dataclasses**: `@dataclass(frozen=True, slots=True)` (`kw_only=True` past a few fields), with
  `wire()` for their JSON form and a `read()` classmethod from it where they cross a process or the network; never
  `NamedTuple`, `TypedDict` or a bare dict or tuple for a record. A record's fields are accessed by name.
- **Types**: pyright runs in strict mode over everything it checks (`trex/`, `tests/`, `examples/`, `docs/`), and
  tests are typed like the code: every function, parameter and return, fixtures and test functions included
  (`-> None`), empty containers annotated. Tests may use private members (`reportPrivateUsage` is off for `tests/`).
  Parse JSON into `JSONValue` and narrow it with `isinstance` or the `format.as_*` helpers rather than casts; `Any`
  only for decoded JSON used as it is (passed to a `read()`, or indexed by a test's assertions), SQLite rows, and
  wrappers' `*args/**kwargs`. Use PEP 695 `type` aliases and generics, `X | None`, built-in generics, and `Final` for
  constants. No `# pyright: ignore` or `# type: ignore`. An untyped dev dependency gets a stub in `typings/`.
- Cyclomatic complexity is at most 15 per function, Python (radon) and UI JS (eslint's `complexity` rule);
  `tests/test_complexity.py` and `tests/complexity.test.mjs` enforce it, the latter also refusing unused variables in
  the UI. Split a function by what it does into named steps; keep hot inner loops in one function.
- Python via `uv run`; never bare `python`. Stdlib and numpy first. Adding a runtime dependency needs a very good
  reason.
- Comments and docstrings state the intended behavior, tersely. The `trex` and `trex.node` module docstrings are the
  published user guide, and terse too. No history ("fixed", "now", "previously"), no restating the code; prefer a good
  name to a comment.
- Tests encode the intended contract. If behavior is wrong, the test should fail. Never pin a bug or mark it `xfail`.
  Test names describe the assertion.
- UI: no build step, no frameworks, no CDN, one language. Numeric work belongs in `kernel.js` on typed arrays.
- Long-running jobs: print periodic progress lines (not `tqdm`) when output goes to a log file.
- Do not wait on processes with `pgrep -f`/`pkill -f` patterns that can match your own command line. Use PIDs (`ss
  -ltnp` for servers, `$!` for children).
- American English in user-facing text.
