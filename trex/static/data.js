// Data layer: run metadata, the blocks of bucket arrays (buckets.py) the visible charts need (`plan`), and the SSE
// stream. A run's column of a metric is built from its buckets in its chart's blocks (`buildColumn`) plus the rows
// streamed since those buckets were made; a gap or a missed heartbeat resyncs the run.
//# allFunctionsCalledOnLoad

import { BLOCK, adoptStore, bucketStep, bucketViews, buildColumn, freeStore, rowExtents } from "./kernel.js";
import { fetchArraysOnWorker } from "./pool.js";
import { forgetArray } from "./gpustats.js";
import { asNumber } from "./where.js";

const num = (v) => (typeof v === "number" ? v : asNumber(v) ?? NaN);

/** Run f in a task of its own, at once where the browser has no delay for it (nested timers wait 4 ms). */
const soon = (f) => void (globalThis.scheduler?.postTask ? globalThis.scheduler.postTask(f) : setTimeout(f, 0));

/** What this page and the server say to each other (server.PROTOCOL); the page states a mismatch. */
export const PROTOCOL = 8;
/** URL prefix of what the page shows: a directory ("/d/<id>") or a workspace ("/w/<name>"), else "" (the node's home). */
export const BASE = typeof location === "undefined" ? "" : (location.pathname.match(/^\/[wd]\/[^/]+(?=\/)/) || [""])[0];

/** URL of a media record's file; content-addressed, so browsers cache it as immutable. */
export const mediaURL = (rec) => `${BASE}/m/${encodeURIComponent(rec.run)}/${rec.file}`;

const MIN_LEVEL = -20, MAX_LEVEL = 62; // levels a block may have (buckets.MIN_LEVEL, MAX_LEVEL)
const POINT_BUDGET = 1.5e6; // buckets one chart draws across all its runs
const PARALLEL = 5; // requests in flight: the browser's six connections to a host, less the stream's
const PREFETCH_PARALLEL = 2; // of them fetching ahead, at most
const BATCH_BLOCKS = 32; // blocks one request asks for, at most (server.MAX_ASKS)...
const AHEAD_REQUEST_BYTES = 16e6; // ...and about the bytes of them one fetching ahead does: blocks of thousands of runs
// are megabytes each, and a request for many holds up the server and the connection a plan's requests then wait for
const BUCKET_BYTES = 16; // of one bucket in a bucket array
const EXTENT_BY_RUN = 32; // the extent of at most this many runs is found run by run, of more from their arrays' rows
const ARRAY_BYTES = 2e9; // bucket arrays kept, the least recently used dropped beyond
const AHEAD_BYTES = 1.5e9; // arrays fetched ahead of need, at most
const KEEP_MS = 2000; // an array used this recently is not dropped
const FINE_BLOCKS = 8; // blocks of a finer level one chart's view may take
const SCOPE_MIN = 64; // runs of a chart missing a block above which one request asks for the scope's finished runs...
const SCOPE_SHARE = 4; // ...when they are also at least 1 / SCOPE_SHARE of the chart's runs
const RUNS_PER_REQUEST = 2000; // run ids one request names
const REBUILD_SLICE_MS = 8; // column rebuilds per task, in ms...
const REBUILD_WHOLE_MS = 32; // ...or up to this to finish a metric's, so its chart draws them all at once
const PREFETCH_IDLE_MS = 50; // quiet time before fetching ahead
const AHEAD_SCAN_MS = 2; // one task's share of finding what to fetch ahead next (`nextAhead`)
const RETRY_MS = 1000; // a block whose request failed is asked for again after this, twice as long after each failure...
const RETRY_MAX_MS = 30000; // ...up to this
export const LINE_PX_PER_BUCKET = 2; // chart width per bucket a line needs
export const DENSITY_PX_PER_BUCKET = 8; // the same in a density heatmap or group statistics of many runs
const NO_TAIL = Object.freeze({ s: [], v: [], t: [], q: [], n: 0 });

/** Metadata of a run known only by its id until it is resynced. */
const placeholderMeta = (id, seq = 0, mseq = 0) =>
  ({ id, seq, mseq, compiled: 0, keys: [], summary: {}, config: {}, tags: [], name: id, state: "running" });

/** Whether run r logs metric `key` (a set kept in step with r.meta.keys). */
export function hasKey(r, key) {
  if (r.keyList !== r.meta.keys) {
    r.keyList = r.meta.keys;
    r.keySet = new Set(r.meta.keys || []);
  }
  return r.keySet.has(key);
}

/** Id of block (key, level, index). */
const blockId = (key, level, index) => `${key}|${level}|${index}`;

/** What request x asks for: its block, and the scope's runs or its own (how many, and a hash of their run indices: a
 * collision only leaves a block to the plan that needs it). */
export function askId(x) {
  if (x.runs === null) return `${blockId(x.key, x.level, x.index)}|\0${x.scope}`;
  let h = 2166136261;
  for (const r of x.runs) h = Math.imul(h ^ r.idx, 16777619);
  return `${blockId(x.key, x.level, x.index)}|${x.runs.length}:${h >>> 0}`;
}

/** Whether run or folder `id` is folder (or run) `path` or lies under it ("": everything). */
export const within = (id, path) => path === "" || id === path || id.startsWith(`${path}/`);
/** Whether failed request record f (of `Data.failed`) covers run r: it named r, or asked for the scope's finished runs. */
const failedFor = (f, r) => f.runs.has(r.id) || (f.runs.has("") && r.meta.state !== "running");

/** The blocks of `level` covering steps [lo, hi]: {level, indices}. */
function covering(level, lo, hi) {
  const w = BLOCK * 2 ** level, k0 = Math.floor(lo / w), k1 = Math.floor(hi / w);
  return { level, indices: Array.from({ length: k1 - k0 + 1 }, (_, i) => k0 + i) };
}

/** Whether layer x and layer y ({level, indices}, or null) are the same blocks. */
const sameLayer = (x, y) => x === y || (!!x && !!y && x.level === y.level && x.indices.length === y.indices.length
                                        && x.indices.every((k, i) => k === y.indices[i]));

/** Whether layers a and b ({coarse, fine}, or null) are the same blocks. */
export const sameLayers = (a, b) => !!a && !!b && sameLayer(a.coarse, b.coarse) && sameLayer(a.fine, b.fine);

const clampLevel = (l) => Math.min(MAX_LEVEL, Math.max(MIN_LEVEL, l));

/** The level of buckets splitting `span` steps into at most about `buckets` and at least half as many: the finest level
 * whose blocks are as wide as the span (buckets.level_for), coarser by whole levels, so that it changes only when the
 * span crosses a power of two, as a run's top level does. */
const levelFor = (span, buckets) => clampLevel(Math.ceil(Math.log2(Math.max(span, 2 ** MIN_LEVEL) / BLOCK)) + Math.ceil(Math.log2(BLOCK / Math.max(buckets, 1))));

/** The levels a zoom into a view whose finest level is `level` would want: the two below it, of those there are. */
const finerLevels = (level) => [level - 1, level - 2].filter((l) => l >= MIN_LEVEL);

const early = new Map(); // url -> its response, requested by `preload` and not yet taken

/** Request `urls` now, all at once; the next getJSON of each takes its answer. */
export function preload(urls) {
  for (const u of urls) if (!early.has(u)) early.set(u, fetch(u, { cache: "no-store" }));
}

export async function getJSON(url) {
  const pending = early.get(url);
  early.delete(url);
  const r = await (pending ?? fetch(url, { cache: "no-store" }));
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return r.json();
}



// ---- store -----------------------------------------------------------------

export class Data {
  constructor(ui) {
    this.ui = ui; // {runs(), data(keys, streamed), keys(), media(key), status(text), conn(live), idle(), ahead(), protocol(server's), replan(),
    // aimed() (the blocks asked for by `fetchFor` have come), runFields: the fields the UI keeps on each run, with their first values}
    this.info = null; // /api/info of what the page shows
    this.gen = 0; // bumped by `close`; work begun under an older generation is dropped
    this.version = 0; // bumped whenever runs, their metadata, blocks or rows change
    this.metaVer = 0; // bumped whenever a run's metadata is set: a run's `ver` is its value then
    this.listVer = 0; // bumped whenever a run comes or goes
    this.keyVer = new Map(); // metric -> bumped whenever its blocks, layers or columns change, or a run logging it does
    this.extents = new Map(); // metric -> extentOf answers, while its keyVersion stays
    this.runs = new Map();
    this.byIdx = []; // run index -> run: each run's number in the GPU's tables (r.idx), never another's while the scope lasts
    this.keys = new Map(); // metric key -> number of runs having it
    this.keyMasks = new Map(); // metric key -> Uint8Array, 1 at the run index of each run having it
    this.runMasks = new WeakMap(); // run list -> Uint8Array, 1 at the run index of each run of it
    this.lastSteps = new WeakMap(); // run list -> {done (doneVer), hi (the last step of its finished runs)}
    this.keysVer = 0; // bumped whenever `keys` changes
    this.newKeys = false; // a metric appeared since the UI was last told
    this.keyRuns = new WeakMap(); // run list -> {ver (keysVer), by: keySet -> the runs of the list that log a metric of it}
    this.keySets = new Map(); // metric -> {ver (keysVer), id}: `keySet`
    this.every = null; // {ver (doneVer), runs}: every run loaded, as a list
    this.media = new Map(); // media key -> Map(runId -> records sorted by step)
    this.early = null; // stream events that came while the scope's runs were being listed
    this.folders = {}; // folder path -> info dict from its trex_info.json
    this.scope = null; // the folder whose runs are loaded...
    this.listed = false; // ...once they are listed
    this.view = ""; // the folder (or run) under it that the page shows: the one its finished runs are asked for by
    this.rootKey = "";
    this.stream = null; // EventSource of /api/stream
    this.arrays = new Map(); // id -> {key, level, index, v, seq, buf, loc, bytes, refs (block entries naming it), block, rowRun, rowsVer, rowExt}
    this.arrayBytes = 0;
    this.nextArray = 0;
    this.blocks = new Map(); // blockId -> {runs: Map(run id -> {a (array), row}), arrays (those its entries name), used, ver, held, heldAll}
    // ver: bumped whenever its entries change; held, heldAll: `finishedHeld`'s answers
    this.doneVer = 0; // bumped whenever which runs are finished, or a finished run's compiled rows, change
    this.runningLists = new WeakMap(); // run list -> {done (doneVer), out (its running runs)}
    this.charts = new Map(); // metric -> {want ({coarse, fine} layers), ready (the layers shown), runs, many} of the last plan
    this.shownKeys = new Set(); // the metrics of the charts the last plan was for: those in or near the view
    this.queue = []; // requests to send: {key, level, index, runs (null: the finished runs of folder `scope`)}
    this.inflight = new Map(); // blockId -> {scope (the folder whose finished runs are asked for, or null), runs (ids asked for), n (requests)}
    this.failed = new Map(); // blockId -> {runs (ids whose request failed, "" the scope's finished runs), n (failures), until (when due again), error}
    this.retryT = 0;
    this.posts = 0; // requests in flight for plans
    this.planBlocks = 0; // blocks they ask for
    this.aheadPosts = 0; // requests in flight fetching ahead
    this.aimPosts = 0; // those of them for a zoom being dragged
    this.aheadAsked = new Set(); // askId of each request fetching ahead has made
    this.aheadAt = null; // where `nextAhead`'s scan of the charts is (`aheadScan`); null: at the start
    this.aheadMore = false; // that scan stopped before its end
    this.aheadView = null; // the view of the last plan: another one starts the scan over
    this.rebuildQ = new Map(); // "run\0key" -> [run, key] awaiting rebuildSoon
    this.rebuildLeft = new Map(); // metric -> how many of them are its
    this.rebuilding = false; // a task rebuilding them is under way
    this.prefetchT = 0;
    this.planned = null; // inputs of the last plan
    this.touched = new Set();
    this.stats = { blocks: 0, bytes: 0 };
  }

  async init() {
    const info = await getJSON(`${BASE}/api/info`);
    this.info = info;
    this.rootKey = info.root;
    this.ui.protocol?.(info.protocol);
  }

  close() {
    if (this.stream) this.stream.close();
    this.stream = null;
    this.early = null;
    this.listed = false;
    this.runs.clear();
    this.byIdx = [];
    this.keys.clear();
    this.keyMasks.clear();
    this.keySets.clear();
    this.media.clear();
    this.folders = {};
    this.queue = [];
    this.inflight.clear();
    this.failed.clear();
    clearTimeout(this.retryT);
    this.aheadAsked.clear();
    this.aheadAt = null;
    for (const a of this.arrays.values()) freeStore(a.loc);
    this.arrays.clear();
    this.arrayBytes = 0;
    this.blocks.clear();
    this.charts.clear();
    this.shownKeys.clear();
    clearTimeout(this.prefetchT);
    this.rebuildQ.clear();
    this.rebuildLeft.clear();
    this.gen++;
  }

  newRun(meta) {
    const r = { id: meta.id, idx: this.byIdx.length, meta, ver: ++this.metaVer, seq: meta.compiled ?? 0, mseq: meta.mseq, cols: new Map(), built: new Map(),
                tail: [], tailSeq0: meta.compiled ?? 0, pending: [], hbExpect: null, resyncs: 0, resyncInFlight: false, holding: false, keyList: null,
                keySet: null, ...this.ui.runFields };
    // the UI's fields made here, in one order, give every run one shape, so that code reading them over all runs stays fast
    this.byIdx.push(r);
    this.doneVer++;
    this.listVer++;
    // built: key -> {sig, layers, seq} (the inputs of its column, the layers of its chart then, and the rows it had); holding: events wait in
    // `pending` until a resync finishes
    this.runs.set(r.id, r);
    this.version++;
    this.bumpKeys(meta.keys || []);
    this.countKeys(r, meta.keys || [], 1);
    return r;
  }

  /** Note that what metrics `keys` show changed (`keyVersion`). */
  bumpKeys(keys) {
    for (const k of keys) this.keyVer.set(k, (this.keyVer.get(k) || 0) + 1);
  }

  /** A token that changes whenever what metric `key` shows does: its blocks, layers or columns, or a run logging it. */
  keyVersion(key) {
    return `${this.gen}.${this.keyVer.get(key) || 0}`;
  }

  countKeys(r, keys, d) {
    this.keysVer++;
    for (const k of keys) {
      const c = (this.keys.get(k) || 0) + d;
      if (c > 0) this.keys.set(k, c);
      else this.keys.delete(k);
      if (d > 0 && c === 1) this.newKeys = true;
      this.markKey(k, r.idx, d > 0);
    }
  }

  /** Note in metric k's mask whether run index i has it. */
  markKey(k, i, on) {
    let m = this.keyMasks.get(k);
    if (!m || m.length <= i) {
      const grown = new Uint8Array(Math.max(64, 2 * (i + 1)));
      if (m) grown.set(m);
      this.keyMasks.set(k, (m = grown));
    }
    m[i] = on ? 1 : 0;
  }

  /** A mask of the runs of list `runs`: 1 at each one's run index; kept for the list. */
  runMask(runs) {
    let m = this.runMasks.get(runs);
    if (!m) {
      m = new Uint8Array(this.byIdx.length);
      for (const r of runs) m[r.idx] = 1;
      this.runMasks.set(runs, m);
    }
    return m;
  }

  /** Replace a run's metadata, keeping the key counts in step. */
  setMeta(r, meta) {
    const old = r.meta.keys || [], now = meta.keys || old;
    if (meta.state !== r.meta.state || (meta.state !== "running" && meta.compiled !== r.meta.compiled)) this.doneVer++;
    if (old.join("\0") !== now.join("\0")) {
      this.countKeys(r, old, -1);
      this.countKeys(r, now, 1);
      this.bumpKeys(now);
    }
    r.meta = { ...meta, keys: now };
    r.ver = ++this.metaVer;
    this.version++;
    this.bumpKeys(old);
  }

  dropRun(r) {
    this.countKeys(r, r.meta.keys || [], -1);
    for (const b of this.blocks.values()) this.unref(b, r.id);
    this.runs.delete(r.id);
    this.byIdx[r.idx] = undefined;
    this.doneVer++;
    this.listVer++;
    this.version++;
    this.bumpKeys(r.meta.keys || []);
  }

  /** Load every run under folder `path` (relative to the served root; "" = everything). The stream opens first (the
   * server opens it once subscribed) and its events wait until the runs are listed, so a run that appears in between is
   * not missed. */
  async loadScope(path) {
    this.close();
    this.scope = this.view = path;
    this.ui.status("loading runs…");
    this.early = [];
    await this.openStream();
    if (this.scope !== path) return;
    const j = await getJSON(`${BASE}/api/runs?path=${encodeURIComponent(path)}`);
    if (this.scope !== path) return;
    for (const meta of j.runs) this.newRun(meta);
    this.folders = j.folders || {};
    this.setMedia(j.media, null);
    this.flushKeys();
    const early = this.early;
    this.early = null;
    this.listed = true;
    for (const [kind, ev] of early) this.dispatch(kind, ev);
  }

  flushKeys() {
    if (this.newKeys) {
      this.newKeys = false;
      this.ui.keys();
    }
  }

  /** The runs of `runs` that log `key` (`runs` itself when every run does), cached while neither the list nor any
   * run's keys change: one list for all the metrics the same runs log, so that what is kept per list is shared by their
   * charts. */
  runsWith(runs, key) {
    if (this.keys.get(key) === this.runs.size) return runs;
    let kept = this.keyRuns.get(runs);
    if (kept?.ver !== this.keysVer) this.keyRuns.set(runs, (kept = { ver: this.keysVer, by: new Map() }));
    const set = this.keySet(key);
    let out = kept.by.get(set);
    if (!out) {
      const m = this.keyMasks.get(key);
      kept.by.set(set, (out = m ? runs.filter((r) => m[r.idx] === 1) : []));
    }
    return out;
  }

  /** A token of which runs log `key`, the same for metrics the same runs log; kept while no run's keys change. */
  keySet(key) {
    let kept = this.keySets.get(key);
    if (kept?.ver === this.keysVer) return kept.id;
    const m = this.keyMasks.get(key), end = m ? Math.min(m.length, this.byIdx.length) : 0;
    let a = 2166136261, b = 0, n = 0;
    for (let i = 0; i < end; i++) if (m[i]) (a = Math.imul(a ^ i, 16777619)), (b = (b + Math.imul(i + 1, 0x9e3779b1)) | 0), n++;
    this.keySets.set(key, (kept = { ver: this.keysVer, id: `${n}.${a >>> 0}.${b >>> 0}` }));
    return kept.id;
  }

  /** Whether some run of `runs` logs `key`. */
  someLog(runs, key) {
    const m = this.keyMasks.get(key);
    return !!m && runs.some((r) => m[r.idx] === 1);
  }

  /** Every run loaded, as a list kept while they stay. */
  everyRun() {
    if (this.every?.ver !== this.doneVer) this.every = { ver: this.doneVer, runs: [...this.runs.values()] };
    return this.every.runs;
  }

  // ---- planning ----

  /** The blocks the visible charts need and the requests for those not here: one demand per metric, {key, runs,
   * runsSig (a hash of the set of runs), xmode, zoomed, x0, x1, pw, many (its runs are drawn from bins of their
   * buckets)}; the number of requests under way. `part`: the demands are of some of the visible charts only. */
  plan(demands, part = false) {
    if (!part) this.shownKeys.clear();
    for (const d of demands) this.shownKeys.add(d.key);
    const view = demands.map((d) => [d.key, d.xmode, d.zoomed, d.x0, d.x1, d.pw, d.many, d.runsSig].join("|")).join("\n"), sig = `${this.version}\n${view}`;
    if (view !== this.aheadView) (this.aheadView = view), (this.aheadAt = null); // other charts, runs or steps to fetch ahead of
    if (sig === this.planned && !this.queue.length) return this.inflight.size; // the same plan, all of it asked for
    this.planned = sig;
    const queue = [];
    for (const d of demands) this.planChart(d, queue);
    this.queue = queue;
    this.pump();
    this.flush(); // charts whose blocks were all here show them now
    const n = this.queue.length + this.planBlocks;
    this.ui.status(n ? `loading ${n} blocks…` : this.summary());
    return n;
  }

  /** One chart: the layers it wants, the requests for their blocks, and its columns once they are all here. */
  planChart(d, queue) {
    const runs = this.runsWith(d.runs, d.key), ch = this.charts.get(d.key) || { ready: null };
    if (ch.runs === runs && ch.settled === this.planSig(d)) return; // as when it last showed all it wants: nothing to ask for
    const want = this.layersOf(d, runs), n = queue.length, shown = ch.ready;
    Object.assign(ch, { want, runs, many: d.many });
    this.charts.set(d.key, ch);
    for (const L of [want.coarse, want.fine]) for (const index of L ? L.indices : []) this.need(d.key, L.level, index, runs, queue);
    this.settleChart(d.key, ch);
    // settled once it already showed what it wants: what it wants depends on the extent of what it shows (`layersOf`)
    ch.settled = queue.length === n && sameLayers(shown, want) && !this.rebuildLeft.has(d.key) && this.quiet(d.key, want) ? this.planSig(d) : null;
  }

  /** The layers chart d would show at once were its view d's: those it wants (`layersOf`) when their blocks are all
   * here, as a plan of d would have it (`settleChart`), else those it shows. */
  layersIf(d) {
    const runs = this.runsWith(d.runs, d.key), want = this.layersOf(d, runs);
    return this.complete(d.key, want, runs) ? want : this.charts.get(d.key)?.ready ?? null;
  }

  /** What chart d's plan depends on besides its runs: its metric's data, its view, and which runs are finished. */
  planSig(d) {
    return `${this.keyVersion(d.key)}|${this.doneVer}|${d.xmode}|${d.zoomed}|${d.x0}|${d.x1}|${d.pw}|${d.many}`;
  }

  /** Whether no block of `layers` of `key` is being asked for or failing: nothing of it will change unasked. */
  quiet(key, layers) {
    return [layers.coarse, layers.fine].every((L) => !L || L.indices.every((index) => {
      const id = blockId(key, L.level, index);
      return !this.inflight.has(id) && !this.failed.has(id);
    }));
  }

  /** The layers chart d (of runs `runs`) wants: coarse ({level, indices}), its buckets over every step of its runs,
   * as wide as its width and point budget allow; and fine, over the steps it is zoomed into, when finer (else null). */
  layersOf(d, runs) {
    const px = d.many ? DENSITY_PX_PER_BUCKET : LINE_PX_PER_BUCKET, buckets = Math.min(d.pw / px, POINT_BUDGET / Math.max(runs.length, 1));
    let lo = 0, hi = this.lastStep(runs);
    const ext = this.extentOf(runs, d.key);
    if (ext) (lo = Math.min(ext[0], hi)), (hi = Math.max(hi, ext[1]));
    const coarse = covering(levelFor(hi - lo, buckets), lo, hi);
    const view = d.zoomed ? this.stepWindow(d, runs) : null;
    if (!view) return { coarse, fine: null };
    let level = levelFor(view[1] - view[0], buckets);
    while (level < coarse.level && covering(level, view[0], view[1]).indices.length > FINE_BLOCKS) level++;
    return { coarse, fine: level < coarse.level ? covering(level, view[0], view[1]) : null };
  }

  /** The last step any run of `runs` reached (its summary's _step): the finished ones' kept while they stay. */
  lastStep(runs) {
    let kept = this.lastSteps.get(runs);
    if (kept?.done !== this.doneVer) {
      let hi = 0;
      for (const r of runs) if (r.meta.state !== "running") hi = Math.max(hi, num(r.meta.summary?._step ?? 0));
      this.lastSteps.set(runs, (kept = { done: this.doneVer, hi }));
    }
    let hi = kept.hi;
    for (const r of this.runningOf(runs)) hi = Math.max(hi, num(r.meta.summary?._step ?? 0));
    return hi;
  }

  /** The steps chart d is zoomed into: its x range in steps, or found from its runs' columns for a runtime x (none
   * for many runs). */
  stepWindow(d, runs) {
    if (d.xmode === 0) return Number.isFinite(d.x0) && Number.isFinite(d.x1) ? [d.x0, d.x1] : null;
    if (d.many) return null;
    let a = Infinity, b = -Infinity;
    for (const r of runs) {
      const c = r.cols.get(d.key);
      for (let i = 0; c && i < c.n; i++) if (c.t[i] >= d.x0 && c.t[i] <= d.x1) (a = Math.min(a, c.s[i])), (b = Math.max(b, c.s[i]));
    }
    return b >= a ? [a, b] : null;
  }

  /** Queue requests for block (key, level, index) of the runs of `runs` that lack it, or hold a running run's buckets
   * older than its compiled ones, and whose request for it is not failing: the finished runs of the folder shown
   * (`view`) in one request when many lack it, by run ids otherwise. */
  need(key, level, index, runs, queue, touch = true) {
    const id = blockId(key, level, index), have = this.blocks.get(id), asked = this.inflight.get(id);
    if (have && touch) have.used = performance.now();
    const failed = this.failed.get(id), backoff = failed && performance.now() < failed.until ? failed : null;
    // every run lacks a block of which nothing is here, asked for or failing: no run is looked at then
    const missing = !have && !asked && !backoff ? runs : this.lacking(have, asked, backoff, runs, key);
    const running = missing === runs ? this.runningOf(runs) : missing.filter((r) => r.meta.state === "running"), finished = missing.length - running.length;
    const scope = finished >= SCOPE_MIN && finished * SCOPE_SHARE >= runs.length;
    if (scope) queue.push({ key, level, index, runs: null, scope: this.view });
    const byId = scope ? running : missing;
    for (let i = 0; i < byId.length; i += RUNS_PER_REQUEST) queue.push({ key, level, index, runs: byId.slice(i, i + RUNS_PER_REQUEST) });
  }

  /** The runs of `runs` whose buckets block `have` (of metric `key`; undefined: not here) lacks as they now are, and
   * which no request under way (`asked`) or failing (`backoff`) names. */
  lacking(have, asked, backoff, runs, key) {
    const some = have && this.finishedHeld(have, runs, key) ? this.runningOf(runs) : runs;
    return some.filter((r) => !this.current(have?.runs.get(r.id), r) && !asked?.runs.has(r.id)
                              && !(asked && asked.scope !== null && r.meta.state !== "running" && within(r.id, asked.scope))
                              && !(backoff && failedFor(backoff, r)));
  }

  /** Whether block entry e holds run r's buckets as they now are. */
  current(e, r) {
    return !!e && e.a.seq[e.row] >= r.meta.compiled;
  }

  /** Whether block b (of metric `key`) holds every finished run of `runs` as it now is: every finished run that logs
   * the metric, or else those of the list; each answer kept while the block's entries, the list and the finished runs
   * stay. */
  finishedHeld(b, runs, key) {
    const all = this.runsWith(this.everyRun(), key);
    if (this.heldOf(b, "heldAll", all)) return true;
    return runs !== all && this.heldOf(b, "held", runs);
  }

  heldOf(b, slot, runs) {
    const h = b[slot];
    if (h?.runs === runs && h.ver === b.ver && h.done === this.doneVer) return h.ok;
    let ok = true;
    for (const r of runs) {
      if (r.meta.state !== "running" && !this.current(b.runs.get(r.id), r)) {
        ok = false;
        break;
      }
    }
    b[slot] = { runs, ver: b.ver, done: this.doneVer, ok };
    return ok;
  }

  /** The running runs of `runs`, kept while neither the list nor which runs are finished change. */
  runningOf(runs) {
    let kept = this.runningLists.get(runs);
    if (kept?.done !== this.doneVer) this.runningLists.set(runs, (kept = { done: this.doneVer, out: runs.filter((r) => r.meta.state === "running") }));
    return kept.out;
  }

  /** Whether every block of `layers` holds every run of `runs` (any version of a running one's) but those whose request
   * for it failed, so a chart shows the runs that came. */
  complete(key, layers, runs) {
    for (const L of [layers.coarse, layers.fine]) {
      for (const index of L ? L.indices : []) {
        const id = blockId(key, L.level, index), b = this.blocks.get(id), failed = this.failed.get(id);
        const some = b && this.finishedHeld(b, runs, key) ? this.runningOf(runs) : runs;
        if (some.some((r) => !b?.runs.has(r.id) && !(failed && failedFor(failed, r)))) return false;
      }
    }
    return true;
  }

  /** Show chart `key`'s wanted layers once they are all here, and have the columns it draws that were not built for
   * the layers it shows rebuilt. */
  settleChart(key, ch) {
    if (!sameLayers(ch.ready, ch.want) && this.complete(key, ch.want, ch.runs)) (ch.ready = ch.want), this.touched.add(key), this.bumpKeys([key]);
    if (!ch.ready) return;
    for (const r of ch.many ? this.runningOf(ch.runs) : ch.runs) if (this.stale(r, key, ch.ready) && this.drawsColumn(ch, r)) this.rebuildSoon(r, key);
  }

  /** Whether run r's column of `key` was built for other layers than `layers`, or before rows it now has came. */
  stale(r, key, layers) {
    const b = r.built.get(key);
    return !sameLayers(b?.layers, layers) || b.seq !== r.seq;
  }

  /** Whether chart ch draws run r from a column (one of many runs is drawn from its buckets unless it is running). */
  drawsColumn(ch, r) {
    return !ch.many || r.meta.state === "running";
  }

  /** Whether `layers` hold block `index` of `level`. */
  shows(layers, level, index) {
    return !!layers && [layers.coarse, layers.fine].some((L) => L && L.level === level && L.indices.includes(index));
  }

  // ---- blocks ----

  /** Run r's buckets of `key` in the blocks its chart shows ([{v, row, a}] in its layers; only its finest, which
   * covers the steps the chart shows, when `finest`), or null before they are here. */
  partsOf(r, key, finest = false) {
    const ready = this.charts.get(key)?.ready;
    if (!ready) return null;
    const parts = [];
    for (const L of finest ? [ready.fine || ready.coarse] : [ready.coarse, ready.fine]) {
      for (const index of L ? L.indices : []) {
        const e = this.blocks.get(blockId(key, L.level, index))?.runs.get(r.id);
        if (e) parts.push({ v: e.a.v, row: e.row, a: e.a });
      }
    }
    return parts;
  }

  /** The bucket arrays of the blocks of layer L ({level, indices}) of metric `key`: what the GPU bins its runs from. */
  arraysOf(key, L) {
    const out = [];
    for (const index of L.indices) for (const a of this.blocks.get(blockId(key, L.level, index))?.arrays || []) out.push(a);
    return out;
  }

  /** The bucket arrays here of the blocks a zoom into chart `key` would want (`finerAhead`): of the two levels below
   * the finest it shows, over the steps it shows. */
  zoomArrays(key) {
    const ready = this.charts.get(key)?.ready, out = [];
    if (!ready) return out;
    const finest = ready.fine || ready.coarse, steps = this.stepsOf(finest);
    for (const level of finerLevels(finest.level)) out.push(...this.arraysOf(key, covering(level, ...steps)));
    return out;
  }

  /** The coarse level chart `key` shows, or null. */
  levelOf(key) {
    return this.charts.get(key)?.ready?.coarse.level ?? null;
  }

  /** [first, last] x (steps for xmode 0, else runtimes) of the buckets of `runs` in the coarse blocks chart `key`
   * shows, or null. */
  extentOf(runs, key, xmode = 0) {
    const ver = this.keyVersion(key), kept = this.extents.get(key) || [], hit = kept.find((e) => e.runs === runs && e.xmode === xmode);
    if (hit?.ver === ver) return hit.out;
    const out = this.extentNow(runs, key, xmode);
    this.extents.set(key, [{ runs, xmode, ver, out }, ...kept.filter((e) => e !== hit)].slice(0, 4));
    return out;
  }

  extentNow(runs, key, xmode) {
    const ready = this.charts.get(key)?.ready;
    if (!ready) return null;
    const [lo, hi] = runs.length <= EXTENT_BY_RUN ? this.extentByRun(runs, key, ready.coarse, xmode) : this.extentByRow(runs, key, ready.coarse, xmode);
    return hi >= lo ? [lo, hi] : null;
  }

  /** [first, last] x of the buckets of `runs` in the blocks of layer L, each run looked up in each block. */
  extentByRun(runs, key, L, xmode) {
    let lo = Infinity, hi = -Infinity;
    for (const index of L.indices) {
      const b = this.blocks.get(blockId(key, L.level, index));
      for (const r of b ? runs : []) {
        const e = b.runs.get(r.id), v = e?.a.v;
        if (!v || v.first[e.row + 1] <= v.first[e.row]) continue;
        const q0 = v.first[e.row], q1 = v.first[e.row + 1] - 1;
        (lo = Math.min(lo, xmode ? v.tmean[q0] : bucketStep(v, q0))), (hi = Math.max(hi, xmode ? v.tmean[q1] : bucketStep(v, q1)));
      }
    }
    return [lo, hi];
  }

  /** The same from the extents of the rows of the layer's arrays (kept with each array), for many runs. */
  extentByRow(runs, key, L, xmode) {
    const mask = this.runMask(runs);
    let lo = Infinity, hi = -Infinity;
    for (const a of this.arraysOf(key, L)) {
      const ext = ((a.rowExt ||= [])[xmode] ||= rowExtents(a.v, xmode));
      for (let row = 0; row < a.v.runs; row++) {
        if (a.rowRun[row] >= 0 && mask[a.rowRun[row]]) (lo = Math.min(lo, ext[3 * row])), (hi = Math.max(hi, ext[3 * row + 1]));
      }
    }
    return [lo, hi];
  }

  /** Keep answer `got` ({buf, bytes, paths}) of request x: each run it names as its entry of the block. Returns those
   * runs. */
  addArray(x, got) {
    const { loc } = adoptStore(got.buf);
    const v = bucketViews(got.buf);
    const id = blockId(x.key, x.level, x.index);
    let b = this.blocks.get(id);
    if (!b) this.blocks.set(id, (b = { runs: new Map(), arrays: new Set(), used: performance.now(), ver: 0, held: null, heldAll: null }));
    const a = { id: this.nextArray++, key: x.key, level: x.level, index: x.index, v, seq: v.seq, buf: got.buf, loc, bytes: got.bytes, refs: 0,
                block: b, rowRun: new Int32Array(v.runs).fill(-1), rowsVer: 0, rowExt: got.ext ? [got.ext] : [] };
    // rowRun: each row's run index, -1 once another array holds it; rowExt: its rows' extents (rowExtents), by xmode
    this.arrays.set(a.id, a);
    this.arrayBytes += a.bytes;
    b.arrays.add(a);
    const runs = [];
    got.paths.forEach((p, row) => {
      const r = this.runs.get(p);
      if (!r) return;
      this.unref(b, p);
      b.runs.set(p, { a, row });
      b.ver++;
      a.refs++;
      a.rowRun[row] = r.idx;
      runs.push(r);
    });
    if (!a.refs) this.release(a);
    this.dropArrays();
    this.version++;
    this.bumpKeys([x.key]);
    return runs;
  }

  /** Remove run `id`'s entry from block b, releasing its array when nothing else refers to it. */
  unref(b, id) {
    const e = b.runs.get(id);
    if (!e) return;
    b.runs.delete(id);
    b.ver++;
    e.a.rowRun[e.row] = -1;
    e.a.rowsVer++;
    if (--e.a.refs <= 0) this.release(e.a);
  }

  release(a) {
    if (!this.arrays.delete(a.id)) return;
    a.block.arrays.delete(a);
    freeStore(a.loc);
    forgetArray(a);
    this.arrayBytes -= a.bytes;
  }

  /** The blocks the charts show or want. */
  inUse() {
    const out = new Set();
    for (const [key, ch] of this.charts) {
      for (const layers of [ch.want, ch.ready]) {
        for (const L of layers ? [layers.coarse, layers.fine] : []) for (const index of L ? L.indices : []) out.add(blockId(key, L.level, index));
      }
    }
    return out;
  }

  /** Bytes of the arrays a block refers to. */
  blockBytes(b) {
    let n = 0;
    for (const a of b.arrays) n += a.bytes;
    return n;
  }

  /** Drop whole blocks no chart shows or wants, least recently used first, while arrays exceed ARRAY_BYTES. */
  dropArrays() {
    if (this.arrayBytes <= ARRAY_BYTES) return;
    const keep = this.inUse(), now = performance.now();
    const old = [...this.blocks].filter(([id, b]) => !keep.has(id) && now - b.used > KEEP_MS).sort((x, y) => x[1].used - y[1].used);
    for (const [id, b] of old) {
      if (this.arrayBytes <= ARRAY_BYTES) break;
      for (const runId of [...b.runs.keys()]) this.unref(b, runId);
      this.blocks.delete(id);
      this.bumpKeys([id.slice(0, id.lastIndexOf("|", id.lastIndexOf("|") - 1))]);
    }
  }

  // ---- fetching ----

  /** Send the queued requests, spread over the free request slots, BATCH_BLOCKS a request at most. */
  pump() {
    while (this.posts + this.aheadPosts < PARALLEL && this.queue.length) {
      const free = PARALLEL - this.posts - this.aheadPosts, size = Math.min(BATCH_BLOCKS, Math.ceil(this.queue.length / free));
      const batch = [];
      while (batch.length < size && this.queue.length) {
        const x = this.queue.shift(), asked = this.asking(x);
        if (asked) batch.push([x, asked]);
      }
      if (!batch.length) continue;
      this.posts++;
      this.planBlocks += batch.length;
      this.send(batch).finally(() => {
        this.posts--;
        this.planBlocks -= batch.length;
        if (!this.busy) this.ui.status(this.summary());
        this.pump();
        this.settle();
      });
    }
  }

  /** Mark request x under way: its block's record of what is asked for, or null when all x asks for already is. */
  asking(x) {
    const id = blockId(x.key, x.level, x.index);
    let asked = this.inflight.get(id);
    if (!asked) this.inflight.set(id, (asked = { scope: null, runs: new Set(), n: 0 }));
    if (x.runs === null ? asked.scope === x.scope : x.runs.every((r) => asked.runs.has(r.id))) return null;
    if (x.runs === null) asked.scope = x.scope;
    else for (const r of x.runs) asked.runs.add(r.id);
    asked.n++;
    return asked;
  }

  /** Request the blocks of `batch` ([x, its record]) in one request and take their answers in: the charts they complete
   * show their new layers. */
  async send(batch) {
    const gen = this.gen, xs = batch.map(([x]) => x);
    try {
      const got = await this.answers(xs, gen);
      if (!got || gen !== this.gen) return;
      xs.forEach((x, i) => {
        this.answered(x);
        if (got[i]) this.take(x, got[i]);
      });
      this.flush();
      if (!this.busy) this.ui.status(this.summary());
    } finally {
      for (const [x, asked] of batch) if (--asked.n === 0) this.inflight.delete(blockId(x.key, x.level, x.index));
    }
  }

  /** The answers of requests xs (`fetchMany`), or null when the request failed (`fail`). */
  async answers(xs, gen) {
    try {
      return await this.fetchMany(xs);
    } catch (e) {
      if (gen === this.gen) this.fail(xs, e instanceof Error ? e.message : String(e));
      return null;
    }
  }

  /** Requests xs failed with `error`: the runs they named are asked for again once their block is due (RETRY_MS, twice
   * as long after each further failure, at most RETRY_MAX_MS), and meanwhile their charts show the runs that came. */
  fail(xs, error) {
    console.warn("block fetch failed", error);
    const now = performance.now();
    for (const x of xs) {
      const id = blockId(x.key, x.level, x.index), f = this.failed.get(id) ?? { runs: new Set(), n: 0, until: 0, error };
      f.n++;
      f.until = now + Math.min(RETRY_MAX_MS, RETRY_MS * 2 ** (f.n - 1));
      f.error = error;
      for (const run of x.runs ? x.runs.map((r) => r.id) : [""]) f.runs.add(run);
      this.failed.set(id, f);
      this.aheadAsked.delete(askId(x)); // fetched ahead again once due
      this.aheadAt = null;
      const ch = this.charts.get(x.key);
      if (ch) this.settleChart(x.key, ch);
      this.touched.add(x.key);
    }
    this.flush();
    this.retrySoon();
    if (!this.busy) this.ui.status(this.summary());
  }

  /** Request x was answered: the runs it named are no longer failing for its block. */
  answered(x) {
    const id = blockId(x.key, x.level, x.index), f = this.failed.get(id);
    if (!f) return;
    for (const r of x.runs ?? []) f.runs.delete(r.id);
    if (x.runs === null) f.runs.delete("");
    if (!f.runs.size) this.failed.delete(id);
  }

  /** Plan again whenever a failed block falls due; one already due is asked for by the next plan that wants it. */
  retrySoon() {
    clearTimeout(this.retryT);
    const now = performance.now();
    let due = Infinity;
    for (const f of this.failed.values()) if (f.until > now) due = Math.min(due, f.until);
    if (due === Infinity) return;
    this.retryT = setTimeout(() => {
      this.planned = this.aheadAt = null;
      this.ui.replan?.();
      this.retrySoon();
    }, Math.ceil(due - now) + 1); // whole ms, past it: a timer may fire a little early, and then find nothing due
  }

  /** The error of a failing request for a block chart `key` wants, or null. */
  failure(key) {
    const want = this.charts.get(key)?.want;
    for (const L of want ? [want.coarse, want.fine] : []) {
      for (const index of L ? L.indices : []) {
        const f = this.failed.get(blockId(key, L.level, index));
        if (f) return f.error;
      }
    }
    return null;
  }

  /** Keep answer `got` of request x: its chart shows the layers it completes, and when it shows this block, the
   * columns of the runs the answer holds are rebuilt. */
  take(x, got) {
    const runs = this.addArray(x, got), ch = this.charts.get(x.key);
    if (!ch) return;
    this.settleChart(x.key, ch);
    if (!this.shows(ch.ready, x.level, x.index)) return;
    for (const r of runs) if (this.drawsColumn(ch, r)) this.rebuildSoon(r, x.key);
    this.touched.add(x.key);
  }

  /** The answers of requests `xs`, null for a block not answered, fetched by a worker; throws when the request fails. */
  async fetchMany(xs) {
    const url = new URL(`${BASE}/api/buckets`, typeof location === "undefined" ? "http://localhost/" : location.href).href;
    const blocks = xs.map((x) => (x.runs === null ? { key: x.key, level: x.level, index: x.index, scope: x.scope, which: "finished" }
      : { key: x.key, level: x.level, index: x.index, runs: x.runs.map((r) => r.id) }));
    const body = JSON.stringify({ blocks });
    const got = await fetchArraysOnWorker(url, body);
    if (got.error || got.status !== 200) throw new Error(got.error || `status ${got.status}`);
    this.stats.blocks += got.arrays.filter(Boolean).length;
    this.stats.bytes += got.bytes;
    return xs.map((_, i) => got.arrays[i] || null);
  }

  /** Whether blocks are queued or being fetched, or columns await rebuilding. */
  get busy() {
    return this.queue.length > 0 || this.posts > 0 || this.rebuildQ.size > 0;
  }

  /** Tell the UI once the work for its last plan is done, then fetch ahead while nothing else is under way. */
  settle() {
    if (this.busy) return;
    this.ui.idle();
    clearTimeout(this.prefetchT);
    this.prefetchT = setTimeout(() => this.prefetch(), PREFETCH_IDLE_MS);
  }

  // ---- fetching ahead ----

  /** While no plan's requests are under way, fetch what charts may soon show (`nextAhead`), each request once: blocks
   * of about AHEAD_REQUEST_BYTES a request (were every run to fill its block; BATCH_BLOCKS at most), PREFETCH_PARALLEL
   * requests at a time, one after another until nothing is left or AHEAD_BYTES is reached; in another task when the
   * scan for them stopped short with nothing to ask for. */
  prefetch() {
    while (!this.busy && this.aheadPosts < PREFETCH_PARALLEL) {
      const batch = [];
      let bytes = 0;
      for (const x of this.nextAhead(BATCH_BLOCKS)) {
        if (bytes >= AHEAD_REQUEST_BYTES) break;
        bytes += (x.runs ? x.runs.length : this.runs.size) * BLOCK * BUCKET_BYTES;
        const asked = this.asking(x);
        this.aheadAsked.add(askId(x));
        if (asked) batch.push([x, asked]);
      }
      if (!batch.length) {
        clearTimeout(this.prefetchT);
        if (this.aheadMore) this.prefetchT = setTimeout(() => this.prefetch(), 0);
        return;
      }
      this.aheadPosts++;
      this.send(batch).finally(() => {
        this.aheadPosts--;
        this.prefetch();
      });
    }
  }

  /** Fetch ahead, now, the blocks the charts would want were they to show `demands` (a zoom being dragged), spread
   * over the request slots; nothing while an earlier such fetch is under way. */
  fetchFor(demands) {
    if (this.aimPosts) return;
    const out = [];
    for (const d of demands) this.wantedAhead(d, out);
    const size = Math.min(BATCH_BLOCKS, Math.ceil(out.length / PARALLEL));
    for (let i = 0; i < out.length; i += size) {
      const batch = out.slice(i, i + size).map((x) => [x, this.asking(x)]).filter(([, asked]) => asked);
      if (!batch.length) continue;
      this.aheadPosts++;
      this.aimPosts++;
      this.send(batch).finally(() => {
        this.aheadPosts--;
        this.aimPosts--;
        this.pump();
        if (!this.aimPosts) this.ui.aimed?.();
      });
    }
  }

  /** Up to n requests, not made ahead before, for blocks a chart may soon show (`ui.ahead`: demands, nearest the view
   * first): every chart's wanted layers first, then the finer levels a zoom of each would want (`finerAhead`); while
   * the blocks no chart uses hold less than AHEAD_BYTES. The scan of the charts goes on where the last call's found
   * its first request (`aheadScan`: what lies before is asked for) and takes `budget` ms a call (`aheadMore`: it
   * stopped before its end), so that a call costs the charts it gets to, not all of them. */
  nextAhead(n, budget = AHEAD_SCAN_MS) {
    const at = this.aheadScan(), out = [], t0 = performance.now();
    let from = null, looked = 0; // where the first request was found; the demands looked at (one a call at least)
    while (at && at.pass < 2 && out.length < n && (!looked || performance.now() - t0 <= budget)) {
      if (at.i < at.demands.length) {
        out.push(...this.aheadHere(at));
        if (out.length) from ??= [at.pass, at.i];
        at.i++;
        looked++;
      } else (at.pass++), (at.i = 0);
    }
    this.aheadMore = !!at && at.pass < 2;
    if (from) [at.pass, at.i] = from; // its requests may not all be taken: it is looked at again
    return out.slice(0, n);
  }

  /** Where `nextAhead`'s scan is, {demands (`ui.ahead`'s when it began), pass (0: wanted layers, 1: finer levels), i (the
   * demand)}: at the start again when the runs or their metrics changed since it began (and whenever the view planned
   * for does, `plan`, or a request failed); null when nothing is left to scan, or AHEAD_BYTES are held. */
  aheadScan() {
    let at = this.aheadAt;
    if (!at || at.done !== this.doneVer || at.keys !== this.keysVer) at = this.aheadAt = { demands: null, pass: 0, i: 0, done: this.doneVer, keys: this.keysVer };
    if (at.pass > 1 || this.aheadBytes() >= AHEAD_BYTES) return null;
    at.demands ||= this.ui.ahead?.() || [];
    return at;
  }

  /** The requests, not made ahead before, of the demand scan `at` is at. */
  aheadHere(at) {
    const q = [], d = at.demands[at.i];
    if (at.pass) this.finerAhead(d, q);
    else this.wantedAhead(d, q);
    return q.filter((x) => !this.aheadAsked.has(askId(x)));
  }

  /** Bytes of the blocks no chart shows or wants. */
  aheadBytes() {
    const used = this.inUse();
    let n = 0;
    for (const [id, b] of this.blocks) if (!used.has(id)) n += this.blockBytes(b);
    return n;
  }

  /** Requests (into `out`) for the blocks of the layers demand d wants that lack one of its runs. */
  wantedAhead(d, out) {
    const runs = this.runsWith(d.runs, d.key), want = this.layersOf(d, runs);
    for (const L of [want.coarse, want.fine]) for (const index of L ? L.indices : []) this.need(d.key, L.level, index, runs, out, false);
  }

  /** Requests (into `out`) for the blocks of the two levels below the finest the chart of demand d shows (or, not
   * shown yet, would), over the steps it shows: what a zoom into it wants. */
  finerAhead(d, out) {
    const runs = this.runsWith(d.runs, d.key), ready = this.charts.get(d.key)?.ready ?? this.layersOf(d, runs);
    const finest = ready.fine || ready.coarse, steps = this.stepsOf(finest);
    for (const level of finerLevels(finest.level)) {
      const L = covering(level, ...steps);
      for (const index of L.indices.length <= FINE_BLOCKS ? L.indices : []) this.need(d.key, L.level, index, runs, out, false);
    }
  }

  /** [lo, hi] steps of a layer's blocks. */
  stepsOf(L) {
    const w = BLOCK * 2 ** L.level;
    return [L.indices[0] * w, (L.indices[L.indices.length - 1] + 1) * w - 1e-9 * w];
  }

  summary() {
    const s = this.stats, text = `${this.runs.size} runs · ${s.blocks} blocks fetched (${(s.bytes / 1e6).toFixed(1)} MB)`;
    const failing = [...this.failed.values()];
    return failing.length ? `${text} · ${failing.length} failing: ${failing.at(-1).error}` : text;
  }

  // ---- columns ----

  /** Build run r's column of `key` from its buckets in the blocks its chart shows and the tail rows they lack, unless
   * they are what it was built from; whether it changed. A column whose blocks the tail no longer continues (its rows
   * between were dropped while no chart in view drew it, `pruneTail`) stays as it is until those blocks come anew. */
  build(r, key) {
    const parts = this.partsOf(r, key);
    if (!parts || (r.cols.has(key) && parts.some((p) => p.v.seq[p.row] < r.tailSeq0))) return false;
    const tail = this.tailOf(r, key), had = r.built.get(key);
    const sig = `${parts.map((p) => `${p.a.id}:${p.row}`).join()}|${r.tailSeq0}|${tail.n}`;
    r.built.set(key, { sig, layers: this.charts.get(key).ready, seq: r.seq });
    if (r.cols.has(key) && had?.sig === sig) return false;
    r.cols.set(key, buildColumn(parts, tail, this.levelOf(key) ?? 0));
    if (r.tail.length) this.pruneTail(r);
    return true;
  }

  /** Rebuild run r's column of `key` (`build`); the UI is told of the metric at the next `flush`. */
  rebuild(r, key) {
    if (this.build(r, key)) this.touched.add(key);
  }

  /** Have the columns chart `key` draws take in the rows streamed since they were built, which `onRows` leaves out: a
   * chart calls this before it draws, so that rows cost a chart nothing while it is not drawn. Whether a column
   * changed (and with it the metric's version). */
  catchUp(key) {
    const ch = this.charts.get(key);
    if (!ch?.ready || !ch.runs) return false;
    let changed = false;
    for (const r of ch.many ? this.runningOf(ch.runs) : ch.runs) {
      const b = r.built.get(key);
      if (b && b.seq !== r.seq && sameLayers(b.layers, ch.ready) && this.build(r, key)) changed = true;
    }
    if (changed) this.bumpKeys([key]);
    return changed;
  }

  /** Rebuild run r's column of `key` later, in a task of rebuilds, so a view change never blocks input. */
  rebuildSoon(r, key) {
    const id = `${r.id}\0${key}`;
    if (!this.rebuildQ.has(id)) this.rebuildLeft.set(key, (this.rebuildLeft.get(key) || 0) + 1);
    this.rebuildQ.set(id, [r, key]);
    if (this.rebuilding) return;
    this.rebuilding = true;
    soon(() => this.rebuildSome());
  }

  /** Rebuild the queued columns of the metrics `keys` now, as far as one task of `rebuildSome` goes, rather than in a
   * task later. */
  rebuildNow(keys) {
    if (this.rebuildQ.size) this.rebuildSome(keys);
  }

  /** Whether columns of metric `key` await rebuilding. */
  pending(key) {
    return this.rebuildLeft.has(key);
  }

  /** Rebuild queued columns (those of the metrics `only`, when given) for REBUILD_SLICE_MS, going on to finish a
   * metric's within REBUILD_WHOLE_MS; then tell the UI of the metrics done, and go on in another task while some are
   * left (the task already due goes on with those `only` leaves). */
  rebuildSome(only = null) {
    const t0 = performance.now();
    let last = null;
    try {
      for (const [id, [r, key]] of this.rebuildQ) {
        if (only && !only.has(key)) continue;
        const spent = performance.now() - t0;
        if (spent > REBUILD_WHOLE_MS || (spent > REBUILD_SLICE_MS && key !== last)) break;
        last = key;
        this.rebuildQ.delete(id);
        const left = this.rebuildLeft.get(key) - 1;
        if (left) this.rebuildLeft.set(key, left);
        else this.rebuildLeft.delete(key);
        if (this.runs.get(r.id) === r) this.rebuild(r, key);
      }
      this.flush();
    } finally {
      if (!only) this.rebuilding = this.rebuildQ.size > 0;
      if (!only && this.rebuilding) soon(() => this.rebuildSome());
    }
    this.settle();
  }

  /** Drop the first tail rows that the run's compiled levels hold and no chart in or near the view (`shownKeys`) needs
   * for the run's column: rows before the sequence number of every block such a chart draws it from, at steps before
   * the end of those blocks. A chart out of view holds no rows back, or a running run's tail would grow for as long as
   * the page lasts; it draws the run from its blocks fetched anew once it is shown again (`build`). */
  pruneTail(r) {
    if (r.tailSeq0 >= r.meta.compiled) return;
    let keep = r.meta.compiled, end = Infinity;
    for (const key of this.shownKeys) {
      if (!r.cols.has(key)) continue;
      const ready = this.charts.get(key)?.ready;
      if (ready) end = Math.min(end, this.stepsOf(ready.coarse)[1]);
      for (const p of this.partsOf(r, key) || []) keep = Math.min(keep, p.v.seq[p.row]);
    }
    let drop = 0;
    while (drop < r.tail.length && r.tailSeq0 + drop < keep && r.tail[drop][0] < end) drop++;
    if (drop > 0) {
      r.tail.splice(0, drop);
      r.tailSeq0 += drop;
    }
  }

  /** Points of `key` in the run's tail rows from sequence number `from` on: {s, v, t, q (sequence numbers), n}. */
  tailOf(r, key, from = -Infinity) {
    if (!r.tail.length) return NO_TAIL;
    const s = [], v = [], t = [], q = [];
    for (let i = from > -Infinity ? Math.max(0, from - r.tailSeq0) : 0; i < r.tail.length; i++) {
      const [step, rt, d] = r.tail[i];
      if (!(key in d)) continue;
      s.push(step), v.push(num(d[key])), t.push(rt), q.push(r.tailSeq0 + i);
    }
    return { s, v, t, q, n: s.length };
  }

  /** Tell the UI which metrics changed: `streamed` when by rows from live runs; else by work for its view, each once
   * none of its columns awaits rebuilding. */
  flush(streamed = false) {
    const keys = new Set();
    for (const k of this.touched) if (streamed || !this.rebuildLeft.has(k)) keys.add(k);
    if (!keys.size) return;
    for (const k of keys) this.touched.delete(k);
    this.version++;
    this.bumpKeys(keys);
    this.flushKeys();
    this.ui.data(keys, streamed);
  }

  /** Discard a run's rows and reload it from its current server state; while the events that waited for it still
   * leave a gap, again after a delay twice as long each time. */
  async resync(r) {
    if (r.resyncInFlight) return;
    r.resyncInFlight = true;
    r.holding = true;
    const delay = Math.min(30000, 500 * 2 ** r.resyncs++);
    if (r.resyncs > 1) await new Promise((ok) => setTimeout(ok, delay));
    try {
      const j = await getJSON(`${BASE}/api/run?path=${encodeURIComponent(r.id)}`);
      if (this.runs.get(r.id) !== r) return;
      this.setMeta(r, j.run);
      r.mseq = j.run.mseq;
      this.setMedia(j.media, r.id);
      const from = j.run.compiled;
      const rows = from < j.run.seq ? await getJSON(`${BASE}/api/rows?path=${encodeURIComponent(r.id)}&from=${from}`) : { seq0: from, rows: [] };
      if (this.runs.get(r.id) !== r) return;
      r.tail = [];
      r.tailSeq0 = r.seq = rows.seq0; // a mirror's rows may begin past its levels, which then catch up
      this.appendRows(r, rows);
      r.built.clear();
      r.resyncInFlight = false;
      r.holding = false;
      const pending = r.pending;
      r.pending = [];
      for (const [kind, ev] of pending) this.dispatch(kind, ev);
      if (!r.resyncInFlight) r.resyncs = 0;
      this.ui.runs();
      this.ui.data(new Set(r.cols.keys()), true);
    } catch (e) {
      console.warn(`resync ${r.id} failed`, e);
      r.resyncInFlight = false;
      setTimeout(() => this.resync(r), delay);
    }
  }

  // ---- stream ----

  /** Open the stream of the scope; a promise of its first opening (or failing, which the listing then reports). */
  openStream() {
    const es = new EventSource(`${BASE}/api/stream?path=${encodeURIComponent(this.scope)}`);
    this.stream = es;
    for (const kind of ["rows", "run", "media", "delete", "hb", "folder"]) {
      es.addEventListener(kind, (e) => (this.early ? this.early.push([kind, JSON.parse(e.data)]) : this.dispatch(kind, JSON.parse(e.data))));
    }
    return new Promise((opened) => {
      es.onopen = () => {
        opened();
        this.ui.conn(true);
        getJSON(`${BASE}/api/info`).then((i) => this.ui.protocol?.(i.protocol), () => {}); // a restarted server may be another trex
      };
      es.onerror = () => {
        opened();
        this.ui.conn(false);
      };
    });
  }

  dispatch(kind, ev) {
    if (kind === "hb") return this.onHeartbeat(ev);
    if (kind === "folder") {
      if (ev.info) this.folders[ev.path] = ev.info;
      else delete this.folders[ev.path];
      return this.ui.runs();
    }
    if (kind === "run") return this.onRunMeta(ev);
    const id = kind === "media" ? ev[0] : ev.run;
    let r = this.runs.get(id);
    if (kind === "delete") {
      if (r) {
        this.dropRun(r);
        for (const m of this.media.values()) m.delete(id);
        this.ui.runs();
        this.ui.data(null);
      }
      return;
    }
    if (!r) {
      r = this.newRun(placeholderMeta(id));
      r.pending.push([kind, ev]);
      this.resync(r);
      return;
    }
    if (r.holding) {
      r.pending.push([kind, ev]);
      return;
    }
    if (kind === "rows") this.onRows(r, ev);
    else if (kind === "media") this.onMedia(r, ev);
  }

  /** Append streamed rows [seq0, ...) beyond what the run holds; returns the keys they touch. */
  appendRows(r, ev) {
    const skip = r.seq - ev.seq0, rows = ev.rows, keys = new Set();
    if (skip >= rows.length) return keys;
    const summary = r.meta.summary;
    for (let i = Math.max(0, skip); i < rows.length; i++) {
      const row = rows[i];
      r.tail.push(row);
      for (const k in row[2]) {
        summary[k] = num(row[2][k]);
        keys.add(k);
      }
      summary._step = row[0];
      summary._runtime = row[1];
    }
    r.seq = ev.seq0 + rows.length;
    r.ver = ++this.metaVer; // its summary changed
    const known = new Set(r.meta.keys || []), fresh = [...keys].filter((k) => !known.has(k));
    if (fresh.length) {
      this.countKeys(r, fresh, 1);
      r.meta.keys = [...known, ...fresh].sort();
    }
    return keys;
  }

  onRows(r, ev) {
    if (ev.seq0 > r.seq) {
      console.warn(`run ${r.id}: gap (have ${r.seq}, got ${ev.seq0})`);
      r.pending.push(["rows", ev]);
      return this.resync(r);
    }
    for (const k of this.appendRows(r, ev)) {
      const ch = this.charts.get(k);
      if (!ch?.ready || this.drawsNow(ch, r)) this.touched.add(k); // its column takes the rows in when its chart next draws (`catchUp`)
    }
    this.pruneTail(r);
    this.flush(true);
  }

  /** Whether chart ch draws run r from a column: r is one of the runs it was last planned for (`drawsColumn`). A
   * column no chart draws is left as it is while rows stream (what the GPU bins from it then does not change), and
   * rebuilt with them once a chart draws it again (`settleChart`); one a chart draws, when the chart next draws
   * (`catchUp`): the rows of a page's running runs cost a chart nothing while it is out of view. */
  drawsNow(ch, r) {
    return !!ch.runs && this.runMask(ch.runs)[r.idx] === 1 && this.drawsColumn(ch, r);
  }

  onRunMeta(meta) {
    const r = this.runs.get(meta.id);
    if (!r) {
      const n = this.newRun(meta);
      if (meta.seq > meta.compiled || meta.mseq > 0) this.resync(n);
    } else {
      const summary = r.meta.summary;
      const before = r.meta.compiled;
      this.setMeta(r, { ...meta, summary: { ...meta.summary, ...summary } });
      if (meta.compiled !== before) this.ui.data(new Set(r.cols.keys()), true); // its blocks are due again
      // The stream is ordered, so every row and media item this meta counts has already been delivered.
      if (!r.holding && (r.seq < meta.seq || r.mseq < meta.mseq)) {
        console.warn(`run ${r.id}: meta says ${meta.seq},${meta.mseq}, have ${r.seq},${r.mseq}`);
        this.resync(r);
      }
    }
    this.flushKeys();
    this.ui.runs();
  }

  onHeartbeat(seqs) {
    for (const [id, [seq, mseq]] of Object.entries(seqs)) {
      const r = this.runs.get(id);
      if (!r) {
        this.dispatch("run", placeholderMeta(id, seq, mseq));
        continue;
      }
      if (r.holding) continue;
      const e = r.hbExpect;
      if (e && (r.seq < e[0] || r.mseq < e[1])) {
        console.warn(`run ${id}: heartbeat says ${e}, have ${r.seq},${r.mseq}`);
        this.resync(r);
      }
      r.hbExpect = [seq, mseq];
    }
  }

  // ---- media ----

  setMedia(recs, runId) {
    if (runId !== null) for (const m of this.media.values()) m.delete(runId);
    const keys = new Set();
    for (const rec of recs) keys.add(this.addMedia(rec));
    for (const k of keys) this.ui.media(k);
  }

  addMedia(rec) {
    const [run, seq, step, key, kind, file] = rec;
    let m = this.media.get(key);
    if (!m) this.media.set(key, (m = new Map()));
    let list = m.get(run);
    if (!list) m.set(run, (list = []));
    list.push({ run, seq, step, kind, file });
    if (list.length > 1 && list[list.length - 2].step > step) list.sort((a, b) => a.step - b.step);
    return key;
  }

  onMedia(r, rec) {
    if (rec[1] < r.mseq) return;
    if (rec[1] > r.mseq) return this.resync(r);
    r.mseq++;
    this.ui.media(this.addMedia(rec));
  }
}
