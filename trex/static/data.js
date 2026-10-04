// Data layer: run metadata, the blocks of bucket arrays (buckets.py) the visible charts need (`plan`), and the SSE
// stream. A run's column of a metric is built from its buckets in its chart's blocks (`buildColumn`) plus the rows
// streamed since those buckets were made; a gap or a missed heartbeat resyncs the run.

import { BLOCK, SHARED, adoptStore, bucketPaths, bucketStep, bucketViews, buildColumn, crc32, freeStore } from "./kernel.js";
import { PARALLEL as WORKERS, fetchArrayOnWorker } from "./pool.js";
import { asNumber } from "./where.js";

const num = (v) => (typeof v === "number" ? v : asNumber(v) ?? NaN);

/** What this page and the server say to each other (server.PROTOCOL); the page states a mismatch. */
export const PROTOCOL = 4;
/** URL prefix of what the page shows: a daemon's tracked directory ("/r/<name>") or workspace ("/w/<name>"), else "". */
export const BASE = typeof location === "undefined" ? "" : (location.pathname.match(/^\/[rw]\/[^/]+(?=\/)/) || [""])[0];

/** URL of a media record's file; content-addressed, so browsers cache it as immutable. */
export const mediaURL = (rec) => `${BASE}/m/${encodeURIComponent(rec.run)}/${rec.file}`;

const MIN_LEVEL = -20, MAX_LEVEL = 62; // levels a block may have (buckets.MIN_LEVEL, MAX_LEVEL)
const POINT_BUDGET = 1.5e6; // buckets one chart draws across all its runs
const PARALLEL = 5; // requests in flight: the browser's six connections to a host, less the stream's
const ARRAY_BYTES = 384e6; // bucket arrays kept, the least recently used dropped beyond
const AHEAD_BYTES = 256e6; // arrays fetched ahead of need, at most
const KEEP_MS = 2000; // an array used this recently is not dropped
const FINE_BLOCKS = 8; // blocks of a finer level one chart's view may take
const SCOPE_MIN = 64; // runs of a chart missing a block above which one request asks for the scope's finished runs...
const SCOPE_SHARE = 4; // ...when they are also at least 1 / SCOPE_SHARE of the chart's runs
const RUNS_PER_REQUEST = 2000; // run ids one request names
const REBUILD_SLICE_MS = 8; // column rebuilds per task, in ms
const PREFETCH_IDLE_MS = 400; // quiet time before the next block fetched ahead
const IDB_ENTRIES = 2000; // media blobs kept in IndexedDB
export const LINE_PX_PER_BUCKET = 2; // chart width per bucket a line needs
export const DENSITY_PX_PER_BUCKET = 8; // the same in a density heatmap or group statistics of many runs
const NO_TAIL = Object.freeze({ s: [], v: [], t: [], q: [], n: 0 });

/** Metadata of a run known only by its id until it is resynced. */
const placeholderMeta = (id, seq = 0, mseq = 0) =>
  ({ id, seq, mseq, kept_seq: 0, keys: [], summary: {}, config: {}, tags: [], name: id, state: "running" });

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

/** The blocks of `level` covering steps [lo, hi]: {level, indices}. */
function covering(level, lo, hi) {
  const w = BLOCK * 2 ** level, k0 = Math.floor(lo / w), k1 = Math.floor(hi / w);
  return { level, indices: Array.from({ length: k1 - k0 + 1 }, (_, i) => k0 + i) };
}

const clampLevel = (l) => Math.min(MAX_LEVEL, Math.max(MIN_LEVEL, l));

/** The level of buckets splitting `span` steps into at most about `buckets` and at least half as many: the finest level
 * whose blocks are as wide as the span (buckets.level_for), coarser by whole levels, so that it changes only when the
 * span crosses a power of two, as a run's kept level does. */
const levelFor = (span, buckets) => clampLevel(Math.ceil(Math.log2(Math.max(span, 2 ** MIN_LEVEL) / BLOCK)) + Math.ceil(Math.log2(BLOCK / Math.max(buckets, 1))));

// ---- IndexedDB (media) -------------------------------------------------------

const idb = {
  db: null,
  async open() {
    try {
      this.db = await new Promise((ok, bad) => {
        const r = indexedDB.open("trex", 8);
        r.onupgradeneeded = () => {
          for (const s of [...r.result.objectStoreNames]) r.result.deleteObjectStore(s);
          r.result.createObjectStore("blobs");
        };
        r.onsuccess = () => ok(r.result);
        r.onerror = () => bad(r.error);
        r.onblocked = () => bad(new Error("IndexedDB upgrade blocked by another tab"));
      });
    } catch (e) {
      console.warn("IndexedDB unavailable; caching disabled", e);
      this.db = null;
    }
  },
  get(key) {
    if (!this.db) return Promise.resolve(undefined);
    return new Promise((ok) => {
      const r = this.db.transaction("blobs", "readonly").objectStore("blobs").get(key);
      r.onsuccess = () => ok(r.result);
      r.onerror = () => ok(undefined);
    });
  },
  put(key, value) {
    if (!this.db) return;
    this.db.transaction("blobs", "readwrite").objectStore("blobs").put(value, key);
  },
  /** Delete the oldest entries beyond `max`. */
  prune(max) {
    if (!this.db) return;
    const os = this.db.transaction("blobs", "readwrite").objectStore("blobs");
    os.count().onsuccess = (e) => {
      let extra = e.target.result - max;
      if (extra > 0) {
        os.openKeyCursor().onsuccess = (ev) => {
          const c = ev.target.result;
          if (!c || extra-- <= 0) return;
          os.delete(c.primaryKey);
          c.continue();
        };
      }
    };
  },
  clear() {
    if (!this.db) return Promise.resolve();
    return new Promise((ok) => {
      const tx = this.db.transaction("blobs", "readwrite");
      tx.objectStore("blobs").clear();
      tx.oncomplete = tx.onerror = () => ok();
    });
  },
};

export const clearCache = () => idb.clear();

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

/** A bucket array answer fetched on the page (when no worker fetches it): {status, buf, bytes, paths}. */
async function fetchArrayHere(url, body) {
  try {
    const res = await fetch(url, { method: "POST", body });
    if (!res.ok) return { status: res.status, buf: null, bytes: 0 };
    const buf = await res.arrayBuffer(), v = bucketViews(buf);
    return { status: res.status, buf, bytes: buf.byteLength, paths: bucketPaths(buf, v) };
  } catch (e) {
    return { status: 0, buf: null, bytes: 0, error: String(e) };
  }
}

// ---- store -----------------------------------------------------------------

export class Data {
  constructor(ui) {
    this.ui = ui; // {runs(), data(keys, run|null, streamed), keys(), media(key), status(text), conn(live), replan(), idle(), ahead(),
    //   protocol(server's)}
    this.info = null; // /api/info of what the page shows
    this.gen = 0; // bumped by `close`; work begun under an older generation is dropped
    this.version = 0; // bumped whenever runs, their metadata, blocks or rows change
    this.runs = new Map();
    this.keys = new Map(); // metric key -> number of runs having it
    this.keysVer = 0; // bumped whenever `keys` changes
    this.newKeys = false; // a metric appeared since the UI was last told
    this.keyRunsSrc = null; // the run list `keyRuns` was made from...
    this.keyRunsVer = -1; // ...at this `keysVer`
    this.keyRuns = new Map(); // metric -> the runs of `keyRunsSrc` that log it
    this.media = new Map(); // media key -> Map(runId -> records sorted by step)
    this.folders = {}; // folder path -> info dict from its trex_info.json
    this.scope = null;
    this.rootKey = "";
    this.stream = null; // EventSource of /api/stream
    this.arrays = new Map(); // id -> {key, level, index, v, seq, buf, loc, bytes, refs (block entries naming it)}
    this.arrayBytes = 0;
    this.nextArray = 0;
    this.blocks = new Map(); // blockId -> {runs: Map(run id -> {a (array), row}), used}
    this.charts = new Map(); // metric -> {want ({coarse, fine} layers), ready (the layers shown), runs, many} of the last plan
    this.queue = []; // requests to send: {key, level, index, runs (null: the scope's finished runs)}
    this.inflight = new Map(); // blockId -> {scope (asked for the scope's finished runs), runs (ids asked for)}
    this.posts = 0; // requests in flight
    this.rebuildQ = new Map(); // "run\0key" -> [run, key] awaiting rebuildSoon
    this.rebuildT = 0;
    this.prefetchT = 0;
    this.prefetching = false;
    this.planned = null; // inputs of the last plan
    this.touched = new Set();
    this.stats = { requests: 0, bytes: 0 };
  }

  async init() {
    const [info] = await Promise.all([getJSON(`${BASE}/api/info`), idb.open()]);
    this.info = info;
    this.rootKey = info.root;
    this.ui.protocol?.(info.protocol);
    idb.prune(IDB_ENTRIES);
  }

  close() {
    if (this.stream) this.stream.close();
    this.stream = null;
    this.runs.clear();
    this.keys.clear();
    this.media.clear();
    this.folders = {};
    this.queue = [];
    this.inflight.clear();
    for (const a of this.arrays.values()) freeStore(a.loc);
    this.arrays.clear();
    this.arrayBytes = 0;
    this.blocks.clear();
    this.charts.clear();
    clearTimeout(this.prefetchT);
    this.rebuildQ.clear();
    this.gen++;
  }

  newRun(meta) {
    const r = { id: meta.id, meta, seq: meta.kept_seq ?? 0, mseq: meta.mseq, cols: new Map(), built: new Map(), tail: [],
                tailSeq0: meta.kept_seq ?? 0, pending: [], hbExpect: null, resyncs: 0, resyncInFlight: false, holding: false };
    // built: key -> inputs of its column; holding: events wait in `pending` until a resync finishes
    this.runs.set(r.id, r);
    this.version++;
    this.countKeys(r, meta.keys || [], 1);
    return r;
  }

  countKeys(r, keys, d) {
    this.keysVer++;
    for (const k of keys) {
      const c = (this.keys.get(k) || 0) + d;
      if (c > 0) this.keys.set(k, c);
      else this.keys.delete(k);
      if (d > 0 && c === 1) this.newKeys = true;
    }
  }

  /** Replace a run's metadata, keeping the key counts in step. */
  setMeta(r, meta) {
    const old = r.meta.keys || [], now = meta.keys || old;
    if (old.join("\0") !== now.join("\0")) {
      this.countKeys(r, old, -1);
      this.countKeys(r, now, 1);
    }
    r.meta = { ...meta, keys: now };
    this.version++;
  }

  dropRun(r) {
    this.countKeys(r, r.meta.keys || [], -1);
    for (const b of this.blocks.values()) this.unref(b, r.id);
    this.runs.delete(r.id);
    this.version++;
  }

  /** Load every run under folder `path` (relative to the served root; "" = everything). */
  async loadScope(path) {
    this.close();
    this.scope = path;
    this.ui.status("loading runs…");
    const j = await getJSON(`${BASE}/api/runs?path=${encodeURIComponent(path)}`);
    if (this.scope !== path) return;
    for (const meta of j.runs) this.newRun(meta);
    this.folders = j.folders || {};
    this.setMedia(j.media, null);
    this.flushKeys();
    this.openStream();
  }

  flushKeys() {
    if (this.newKeys) {
      this.newKeys = false;
      this.ui.keys();
    }
  }

  /** The runs of `runs` that log `key`, cached while neither the list nor any run's keys change. */
  runsWith(runs, key) {
    if (this.keyRunsSrc !== runs || this.keyRunsVer !== this.keysVer) {
      this.keyRunsSrc = runs;
      this.keyRunsVer = this.keysVer;
      this.keyRuns = new Map();
    }
    let out = this.keyRuns.get(key);
    if (!out) this.keyRuns.set(key, (out = runs.filter((r) => hasKey(r, key))));
    return out;
  }

  // ---- planning ----

  /** The blocks the visible charts need and the requests for those not here: one demand per metric, {key, runs,
   * runsSig (a hash of the set of runs), xmode, zoomed, x0, x1, pw, many (its runs are drawn from bins of their
   * buckets)}; the number of requests under way. */
  plan(demands) {
    const sig = [this.version, ...demands.map((d) => [d.key, d.xmode, d.zoomed, d.x0, d.x1, d.pw, d.many, d.runsSig].join("|"))].join("\n");
    if (sig === this.planned && !this.queue.length) return this.inflight.size; // the same plan, all of it asked for
    this.planned = sig;
    const queue = [];
    for (const d of demands) this.planChart(d, queue);
    this.queue = queue;
    this.pump();
    const n = this.queue.length + this.inflight.size;
    this.ui.status(n ? `loading ${n} blocks…` : this.summary());
    return n;
  }

  /** One chart: the layers it wants, the requests for their blocks, and its columns once they are all here. */
  planChart(d, queue) {
    const runs = this.runsWith(d.runs, d.key), want = this.layersOf(d, runs);
    const ch = this.charts.get(d.key) || { ready: null };
    Object.assign(ch, { want, runs, many: d.many });
    this.charts.set(d.key, ch);
    for (const L of [want.coarse, want.fine]) for (const index of L ? L.indices : []) this.need(d.key, L.level, index, runs, queue);
    this.settleChart(d.key, ch);
  }

  /** The layers chart d (of runs `runs`) wants: coarse ({level, indices}), its buckets over every step of its runs,
   * as wide as its width and point budget allow; and fine, over the steps it is zoomed into, when finer (else null). */
  layersOf(d, runs) {
    const px = d.many ? DENSITY_PX_PER_BUCKET : LINE_PX_PER_BUCKET, buckets = Math.min(d.pw / px, POINT_BUDGET / Math.max(runs.length, 1));
    let lo = 0, hi = 0;
    for (const r of runs) hi = Math.max(hi, num(r.meta.summary?._step ?? 0));
    const ext = this.extentOf(runs, d.key);
    if (ext) (lo = Math.min(ext[0], hi)), (hi = Math.max(hi, ext[1]));
    const coarse = covering(levelFor(hi - lo, buckets), lo, hi);
    const view = d.zoomed ? this.stepWindow(d, runs) : null;
    if (!view) return { coarse, fine: null };
    let level = levelFor(view[1] - view[0], buckets);
    while (level < coarse.level && covering(level, view[0], view[1]).indices.length > FINE_BLOCKS) level++;
    return { coarse, fine: level < coarse.level ? covering(level, view[0], view[1]) : null };
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
   * older than its kept ones: the scope's finished runs in one request when many lack it, by run ids otherwise. */
  need(key, level, index, runs, queue) {
    const id = blockId(key, level, index), have = this.blocks.get(id), asked = this.inflight.get(id);
    if (have) have.used = performance.now();
    const missing = runs.filter((r) => !this.current(have?.runs.get(r.id), r) && !asked?.runs.has(r.id)
                                       && !(asked?.scope && r.meta.state !== "running"));
    const finished = missing.filter((r) => r.meta.state !== "running");
    const scope = finished.length >= SCOPE_MIN && finished.length * SCOPE_SHARE >= runs.length;
    if (scope) queue.push({ key, level, index, runs: null });
    const byId = scope ? missing.filter((r) => r.meta.state === "running") : missing;
    for (let i = 0; i < byId.length; i += RUNS_PER_REQUEST) queue.push({ key, level, index, runs: byId.slice(i, i + RUNS_PER_REQUEST) });
  }

  /** Whether block entry e holds run r's buckets as they now are. */
  current(e, r) {
    return !!e && e.a.seq[e.row] >= r.meta.kept_seq;
  }

  /** Whether every block of `layers` holds every run of `runs` (any version of a running one's). */
  complete(key, layers, runs) {
    for (const L of [layers.coarse, layers.fine]) {
      for (const index of L ? L.indices : []) {
        const b = this.blocks.get(blockId(key, L.level, index));
        if (!b || runs.some((r) => !b.runs.has(r.id))) return false;
      }
    }
    return true;
  }

  /** Show chart `key`'s wanted layers once they are all here, rebuilding the columns it draws. */
  settleChart(key, ch) {
    const same = ch.ready && JSON.stringify(ch.ready) === JSON.stringify(ch.want);
    if (!same && this.complete(key, ch.want, ch.runs)) {
      ch.ready = ch.want;
      this.touched.add(key);
    }
    if (ch.ready) for (const r of ch.runs) if (!ch.many || r.meta.state === "running") this.rebuildSoon(r, key);
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

  /** The coarse level chart `key` shows, or null. */
  levelOf(key) {
    return this.charts.get(key)?.ready?.coarse.level ?? null;
  }

  /** [first, last] x (steps for xmode 0, else runtimes) of the buckets of `runs` in the coarse blocks chart `key`
   * shows, or null. */
  extentOf(runs, key, xmode = 0) {
    const ready = this.charts.get(key)?.ready;
    let lo = Infinity, hi = -Infinity;
    for (const index of ready ? ready.coarse.indices : []) {
      const b = this.blocks.get(blockId(key, ready.coarse.level, index));
      for (const r of b ? runs : []) {
        const e = b.runs.get(r.id), v = e?.a.v;
        if (!v || v.first[e.row + 1] <= v.first[e.row]) continue;
        const q0 = v.first[e.row], q1 = v.first[e.row + 1] - 1;
        (lo = Math.min(lo, xmode ? v.tmean[q0] : bucketStep(v, q0))), (hi = Math.max(hi, xmode ? v.tmean[q1] : bucketStep(v, q1)));
      }
    }
    return hi >= lo ? [lo, hi] : null;
  }

  /** Keep answer `got` ({buf, bytes, paths}) of request x: each run it names as its entry of the block. */
  addArray(x, got) {
    const { loc } = SHARED && got.buf instanceof SharedArrayBuffer ? adoptStore(got.buf) : { loc: null };
    const v = bucketViews(got.buf);
    const a = { id: this.nextArray++, key: x.key, level: x.level, index: x.index, v, seq: v.seq, buf: got.buf, loc, bytes: got.bytes, refs: 0 };
    this.arrays.set(a.id, a);
    this.arrayBytes += a.bytes;
    const id = blockId(x.key, x.level, x.index);
    let b = this.blocks.get(id);
    if (!b) this.blocks.set(id, (b = { runs: new Map(), used: performance.now() }));
    got.paths.forEach((p, row) => {
      if (!this.runs.has(p)) return;
      this.unref(b, p);
      b.runs.set(p, { a, row });
      a.refs++;
    });
    if (!a.refs) this.release(a);
    this.dropArrays();
    this.version++;
  }

  /** Remove run `id`'s entry from block b, releasing its array when nothing else refers to it. */
  unref(b, id) {
    const e = b.runs.get(id);
    if (!e) return;
    b.runs.delete(id);
    if (--e.a.refs <= 0) this.release(e.a);
  }

  release(a) {
    if (!this.arrays.delete(a.id)) return;
    freeStore(a.loc);
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
    const seen = new Set();
    let n = 0;
    for (const e of b.runs.values()) if (!seen.has(e.a)) seen.add(e.a), (n += e.a.bytes);
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
    }
  }

  // ---- fetching ----

  pump() {
    while (this.posts < PARALLEL && this.queue.length) {
      const x = this.queue.shift(), id = blockId(x.key, x.level, x.index);
      let asked = this.inflight.get(id);
      if (!asked) this.inflight.set(id, (asked = { scope: false, runs: new Set(), n: 0 }));
      if (x.runs === null ? asked.scope : x.runs.every((r) => asked.runs.has(r.id))) continue;
      if (x.runs === null) asked.scope = true;
      else for (const r of x.runs) asked.runs.add(r.id);
      asked.n++;
      this.posts++;
      this.fetchBlock(x).finally(() => {
        this.posts--;
        if (--asked.n === 0) this.inflight.delete(id);
        if (!this.busy) this.ui.status(this.summary());
        this.pump();
        this.settle();
      });
    }
  }

  /** Request x's answer: by a worker into shared memory, else here. */
  async fetch(x) {
    const url = new URL(`${BASE}/api/buckets`, typeof location === "undefined" ? "http://localhost/" : location.href).href;
    const body = JSON.stringify(x.runs === null ? { key: x.key, level: x.level, index: x.index, scope: this.scope, which: "finished" }
      : { key: x.key, level: x.level, index: x.index, runs: x.runs.map((r) => r.id) });
    const got = await (WORKERS ? fetchArrayOnWorker(url, body) : fetchArrayHere(url, body));
    if (got.error) console.warn("block fetch failed", got.error);
    this.stats.requests++;
    this.stats.bytes += got.bytes;
    return got.buf ? got : null;
  }

  /** Fetch block request x and take its answer in; the charts it completes show their new layers. */
  async fetchBlock(x) {
    const gen = this.gen, got = await this.fetch(x);
    if (!got || gen !== this.gen) return;
    this.addArray(x, got);
    const ch = this.charts.get(x.key);
    if (ch) this.settleChart(x.key, ch);
    this.touched.add(x.key);
    this.flush();
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

  /** Fetch the next block a view may need (`nextAhead`) when the store is idle; then the one after. */
  async prefetch() {
    if (this.busy || this.prefetching) return;
    const x = this.nextAhead();
    if (!x) return;
    const gen = this.gen;
    this.prefetching = true;
    try {
      const got = await this.fetch(x);
      if (got && gen === this.gen) this.addArray(x, got);
    } finally {
      this.prefetching = false;
    }
    if (gen === this.gen) this.settle();
  }

  /** The first block not here that a chart may soon show (`ui.ahead`: demands, visible charts first): its wanted
   * layers when it shows none yet, else the two levels below its finest, over the steps it shows; while the blocks
   * no chart uses hold less than AHEAD_BYTES. */
  nextAhead() {
    const used = this.inUse();
    let ahead = 0;
    for (const [id, b] of this.blocks) if (!used.has(id)) ahead += this.blockBytes(b);
    if (ahead >= AHEAD_BYTES) return null;
    for (const d of this.ui.ahead?.() || []) {
      const runs = this.runsWith(d.runs, d.key).filter((r) => r.meta.state !== "running");
      if (runs.length) {
        const x = this.aheadOf(d, runs);
        if (x) return x;
      }
    }
    return null;
  }

  /** The first block of demand d's layers (or the two levels below them, when shown) that lacks one of `runs`. */
  aheadOf(d, runs) {
    const want = this.layersOf(d, runs), finest = want.fine || want.coarse, steps = this.stepsOf(finest);
    const next = this.charts.get(d.key)?.ready ? [1, 2].map((up) => covering(finest.level - up, ...steps)) : [want.coarse, want.fine];
    for (const L of next) {
      for (const index of L && L.indices.length <= FINE_BLOCKS ? L.indices : []) {
        const id = blockId(d.key, L.level, index), b = this.blocks.get(id);
        if (!this.inflight.has(id) && (!b || runs.some((r) => !b.runs.has(r.id)))) return { key: d.key, level: L.level, index, runs: runs.length >= SCOPE_MIN ? null : runs };
      }
    }
    return null;
  }

  /** [lo, hi] steps of a layer's blocks. */
  stepsOf(L) {
    const w = BLOCK * 2 ** L.level;
    return [L.indices[0] * w, (L.indices[L.indices.length - 1] + 1) * w - 1e-9 * w];
  }

  summary() {
    const s = this.stats;
    return `${this.runs.size} runs · ${s.requests} blocks fetched (${(s.bytes / 1e6).toFixed(1)} MB)`;
  }

  // ---- columns ----

  /** Rebuild run r's column of `key` from its buckets in the blocks its chart shows and the tail rows they lack,
   * unless they are what it was built from. */
  rebuild(r, key) {
    const parts = this.partsOf(r, key);
    if (!parts) return;
    const tail = this.tailOf(r, key);
    const sig = `${parts.map((p) => `${p.a.id}:${p.row}`).join()}|${r.tailSeq0}|${tail.n}`;
    if (r.cols.has(key) && r.built.get(key) === sig) return;
    r.built.set(key, sig);
    r.cols.set(key, buildColumn(parts, tail, this.levelOf(key) ?? 0));
    this.pruneTail(r);
    this.touched.add(key);
  }

  /** Rebuild columns later, a few milliseconds per task, so a view change never blocks input. */
  rebuildSoon(r, key) {
    this.rebuildQ.set(`${r.id}\0${key}`, [r, key]);
    if (this.rebuildT) return;
    const slice = () => {
      const t0 = performance.now();
      for (const [id, [r, key]] of this.rebuildQ) {
        this.rebuildQ.delete(id);
        if (this.runs.get(r.id) === r) this.rebuild(r, key);
        if (performance.now() - t0 > REBUILD_SLICE_MS) break;
      }
      this.flush();
      this.rebuildT = this.rebuildQ.size ? setTimeout(slice, 0) : 0;
      this.settle();
    };
    this.rebuildT = setTimeout(slice, 0);
  }

  /** Drop the first tail rows while every metric's blocks hold them: rows before every block's (and the kept buckets')
   * sequence number, at steps before the end of the blocks shown. */
  pruneTail(r) {
    let keep = r.meta.kept_seq, end = Infinity;
    for (const key of r.cols.keys()) {
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

  /** Tell the UI which metrics changed: `streamed` when by rows from live runs, else by work for its view. */
  flush(streamed = false) {
    if (!this.touched.size) return;
    this.version++;
    const keys = this.touched;
    this.touched = new Set();
    this.flushKeys();
    this.ui.data(keys, null, streamed);
  }

  /** Discard a run's rows and reload it from its current server state. */
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
      const from = j.run.kept_seq;
      const rows = from < j.run.seq ? await getJSON(`${BASE}/api/rows?path=${encodeURIComponent(r.id)}&from=${from}`) : { seq0: from, rows: [] };
      if (this.runs.get(r.id) !== r) return;
      r.tail = [];
      r.tailSeq0 = r.seq = from;
      this.appendRows(r, rows);
      r.built.clear();
      r.resyncInFlight = false;
      r.holding = false;
      r.resyncs = 0;
      const pending = r.pending;
      r.pending = [];
      for (const [kind, ev] of pending) this.dispatch(kind, ev);
      this.ui.runs();
      this.ui.data(new Set(r.cols.keys()), null, true);
    } catch (e) {
      console.warn(`resync ${r.id} failed`, e);
      r.resyncInFlight = false;
      setTimeout(() => this.resync(r), delay);
    }
  }

  // ---- stream ----

  openStream() {
    const es = new EventSource(`${BASE}/api/stream?path=${encodeURIComponent(this.scope)}`);
    this.stream = es;
    for (const kind of ["rows", "run", "media", "delete", "hb", "folder"]) {
      es.addEventListener(kind, (e) => this.dispatch(kind, JSON.parse(e.data)));
    }
    es.onopen = () => {
      this.ui.conn(true);
      getJSON(`${BASE}/api/info`).then((i) => this.ui.protocol?.(i.protocol), () => {}); // a restarted server may be another trex
    };
    es.onerror = () => this.ui.conn(false);
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
      if (this.charts.get(k)?.ready) this.rebuild(r, k);
      else this.touched.add(k);
    }
    this.flush(true);
  }

  onRunMeta(meta) {
    const r = this.runs.get(meta.id);
    if (!r) {
      const n = this.newRun(meta);
      if (meta.seq > meta.kept_seq || meta.mseq > 0) this.resync(n);
    } else {
      const summary = r.meta.summary;
      const before = r.meta.kept_seq;
      this.setMeta(r, { ...meta, summary: { ...meta.summary, ...summary } });
      if (meta.kept_seq !== before) this.ui.data(new Set(r.cols.keys()), null, true); // its blocks are due again
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
    const [run, seq, step, key, kind, file, crc, size] = rec;
    let m = this.media.get(key);
    if (!m) this.media.set(key, (m = new Map()));
    let list = m.get(run);
    if (!list) m.set(run, (list = []));
    list.push({ run, seq, step, kind, file, crc, size });
    if (list.length > 1 && list[list.length - 2].step > step) list.sort((a, b) => a.step - b.step);
    return key;
  }

  onMedia(r, rec) {
    if (rec[1] < r.mseq) return;
    if (rec[1] > r.mseq) return this.resync(r);
    r.mseq++;
    this.ui.media(this.addMedia(rec));
  }

  /** HTML media bytes, from cache or network, CRC-verified. */
  async blob(rec) {
    const c = await idb.get(rec.file);
    if (c && c.crc === rec.crc && c.buf.byteLength === rec.size) return c.buf;
    const buf = await (await fetch(mediaURL(rec))).arrayBuffer();
    if (buf.byteLength !== rec.size || crc32(buf) !== rec.crc) throw new Error(`media ${rec.file} failed verification`);
    idb.put(rec.file, { crc: rec.crc, buf });
    return buf;
  }
}
