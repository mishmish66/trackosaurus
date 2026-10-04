// Data layer: run metadata, tiles fetched for what the visible charts show (`plan`), an IndexedDB tile
// cache, and the SSE stream. Each (run, metric) column is built from the best tiles present plus the
// rows streamed since its kept tiles [tiles_seq, seq); a gap or a missed heartbeat resyncs the run.

import { Col, X_STEP, adoptStore, columnStore, crc32, freeStore, holdStore, slabViews } from "./kernel.js";
import { fetchSlabOnWorker } from "./pool.js";
import { asNumber } from "./where.js";

const num = (v) => (typeof v === "number" ? v : asNumber(v) ?? NaN);

/** URL prefix of what the page shows: a daemon's tracked directory ("/r/<name>") or workspace ("/w/<name>"), else "". */
/** What this page and the server say to each other (server.PROTOCOL); the page states a mismatch. */
export const PROTOCOL = 3;
export const BASE = typeof location === "undefined" ? "" : (location.pathname.match(/^\/[rw]\/[^/]+(?=\/)/) || [""])[0];

/** URL of a media record's file; content-addressed, so browsers cache it as immutable. */
export const mediaURL = (rec) => `${BASE}/m/${encodeURIComponent(rec.run)}/${rec.file}`;

const TILE = 256;
const MIN_LEVEL = -20;
const POINT_BUDGET = 1.5e6; // points per chart across all its runs
const TOP_BUCKETS = 192; // typical non-empty buckets of a run's top tiles
const BATCH = 512; // tile requests per POST
const PARALLEL = 5; // POSTs in flight: the browser's six connections to a host, less the stream's
const FINE_BYTES = 256e6; // decoded finer tiles kept in memory
const IDB_ENTRIES = 400000; // cached top-tile entries kept in IndexedDB
const IDB_SLICE_MS = 8; // IndexedDB writing per idle period
const REBUILD_SLICE_MS = 8; // column rebuilds per task, in ms
const OVERVIEW_UP = 2; // levels the server's overview tiles sit above the top tiles (index.OVERVIEW_UP)
const OVERVIEW_MIN_RUNS = 300; // charts with more runs load overview tiles before top tiles
const BUNDLE_MIN = 64; // runs of a chart missing a tier above which one bundle request fetches it...
const BUNDLE_SHARE = 4; // ...when they are also at least 1 / BUNDLE_SHARE of the chart's runs
const PREFETCH_BYTES = 256e6; // top tiles fetched ahead of a view's need, held until one asks for them
const SLAB_BYTES = 384e6; // slabs kept in shared memory, the least recently used dropped beyond
const SLAB_AHEAD_BYTES = 256e6; // of which fetched ahead of need
const SLAB_KEEP_MS = 2000; // a slab used this recently is not dropped
const SLAB_UP = 4; // levels coarser a view draws from while its slabs come
const SLAB_TILE = 256; // buckets of a slab's step range (tiles.TILE)
const PREFETCH_IDLE_MS = 400; // quiet time before the next prefetch
const FINE_HOLD_MS = 10000; // tail rows a refetched finer tile needs are kept this long while it is fetched
export const LINE_PX_PER_BUCKET = 2; // chart width per bucket a line needs
export const DENSITY_PX_PER_BUCKET = 8; // the same in a density heatmap or group statistics of many runs
const FINE_TILES = 2048; // finer tiles one chart may need
const NO_TAIL = Object.freeze({ s: [], v: [], t: [], n: 0, s0: Infinity });
const NO_FINE = Object.freeze({ s: [], v: [], t: [], n: 0 });

// ---- tiles -----------------------------------------------------------------

/** Typed views over a whole tile buffer, made once per buffer. */
const viewsOf = new WeakMap();
function views(buf) {
  let v = viewsOf.get(buf);
  if (!v) viewsOf.set(buf, (v = { u16: new Uint16Array(buf, 0, buf.byteLength >> 1), f32: new Float32Array(buf, 0, buf.byteLength >> 2),
                                  u32: new Uint32Array(buf, 0, buf.byteLength >> 2) }));
  return v;
}

/** A tile read in place: bucket i is u16[b + i]; min, max, mean, tmean, soff are f32[f + k * count + i]
 * (k = 0..4); its count is u32[f + 5 * count + i]. */
export function decodeTile(buf, off = 0, len = buf.byteLength - off) {
  const { u16, f32, u32 } = views(buf), o = off >> 2;
  if (len < 24 || off % 4 || u32[o] !== 0x32544b54) throw new Error("bad tile");
  const level = u32[o + 1] | 0, index = (u32[o + 3] | 0) * 4294967296 + u32[o + 2], count = u32[o + 4];
  const fo = off + 24 + Math.ceil((2 * count) / 8) * 8;
  if (fo - off + 24 * count !== len) throw new Error("tile length mismatch");
  return { level, index, count, bytes: len, u16, f32, u32, b: (off + 24) >> 1, f: fo >> 2 };
}

/** [lo, hi) steps covered by tile (level, index). */
const tileRange = (level, index) => [index * 2 ** level * TILE, (index + 1) * 2 ** level * TILE];

/** Step range [lo, hi] of the non-empty buckets of a list of tiles. */
function bucketSpan(tiles) {
  let lo = Infinity, hi = -Infinity;
  for (const t of tiles) {
    if (!t.count) continue;
    const w = 2 ** t.level, base = t.index * TILE;
    lo = Math.min(lo, (base + t.u16[t.b]) * w);
    hi = Math.max(hi, (base + t.u16[t.b + t.count - 1] + 1) * w);
  }
  return [lo, hi];
}

/** Whether demand d draws run r from slabs, which then stand for its tiles: a finished run. */
const slabbed = (d, r) => !!d.slabs && r.meta.state !== "running";

/** Whether tile request x asks for a kept tier ("top" or "overview"), not a finer tile. */
const isKept = (x) => typeof x[2] === "string";

/** Metadata of a run known only by its id until it is resynced. */
/** The next idle period of the main thread ({timeRemaining()}); without requestIdleCallback, a short slot later. */
const idleTime = () => new Promise((ok) => {
  if (typeof requestIdleCallback === "function") return requestIdleCallback(ok, { timeout: 5000 });
  setTimeout(() => {
    const t0 = performance.now();
    ok({ timeRemaining: () => 8 - (performance.now() - t0) });
  }, 200);
});

const placeholderMeta = (id, seq = 0, mseq = 0) =>
  ({ id, seq, mseq, tiles_seq: 0, keys: [], summary: {}, config: {}, tags: [], name: id, state: "running" });

/** Whether run r logs metric `key` (a set kept in step with r.meta.keys). */
export function hasKey(r, key) {
  if (r.keyList !== r.meta.keys) {
    r.keyList = r.meta.keys;
    r.keySet = new Set(r.meta.keys || []);
  }
  return r.keySet.has(key);
}

// ---- IndexedDB -------------------------------------------------------------

const idb = {
  db: null,
  async open() {
    try {
      this.db = await new Promise((ok, bad) => {
        const r = indexedDB.open("trex", 7);
        r.onupgradeneeded = () => {
          for (const s of [...r.result.objectStoreNames]) r.result.deleteObjectStore(s);
          r.result.createObjectStore("tiles").createIndex("at", "at");
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
  get(store, key) {
    if (!this.db) return Promise.resolve(undefined);
    return new Promise((ok) => {
      const r = this.db.transaction(store, "readonly").objectStore(store).get(key);
      r.onsuccess = () => ok(r.result);
      r.onerror = () => ok(undefined);
    });
  },
  /** [[key, value]] of every entry whose key starts with `prefix`, in one transaction. */
  withPrefix(store, prefix) {
    if (!this.db) return Promise.resolve([]);
    return new Promise((ok) => {
      const tx = this.db.transaction(store, "readonly"), os = tx.objectStore(store);
      const range = IDBKeyRange.bound(prefix, prefix + "\uffff");
      let keys, vals;
      os.getAllKeys(range).onsuccess = (e) => (keys = e.target.result);
      os.getAll(range).onsuccess = (e) => (vals = e.target.result);
      tx.oncomplete = () => ok(keys.map((k, i) => [k, vals[i]]));
      tx.onerror = tx.onabort = () => ok([]);
    });
  },
  /** Put the entries `next()` returns ([key, value]) in one transaction, until it returns null. */
  putEach(store, next) {
    if (!this.db) {
      while (next());
      return Promise.resolve();
    }
    return new Promise((ok) => {
      const tx = this.db.transaction(store, "readwrite"), os = tx.objectStore(store);
      for (let e = next(); e; e = next()) os.put(e[1], e[0]);
      tx.oncomplete = tx.onerror = tx.onabort = () => ok();
    });
  },
  putMany(store, entries) {
    if (!this.db || !entries.length) return Promise.resolve();
    return new Promise((ok) => {
      const tx = this.db.transaction(store, "readwrite");
      const os = tx.objectStore(store);
      for (const [k, v] of entries) os.put(v, k);
      tx.oncomplete = tx.onerror = tx.onabort = () => ok();
    });
  },
  /** Delete the oldest tile entries beyond `max`. */
  prune(max) {
    if (!this.db) return;
    const tx = this.db.transaction("tiles", "readwrite");
    const os = tx.objectStore("tiles");
    os.count().onsuccess = (e) => {
      let extra = e.target.result - max;
      if (extra <= 0) return;
      os.index("at").openKeyCursor().onsuccess = (ev) => {
        const c = ev.target.result;
        if (!c || extra-- <= 0) return;
        os.delete(c.primaryKey);
        c.continue();
      };
    };
  },
  clear() {
    if (!this.db) return Promise.resolve();
    return new Promise((ok) => {
      const tx = this.db.transaction(["tiles", "blobs"], "readwrite");
      tx.objectStore("tiles").clear();
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

/** [tile parts, next offset] of "u32 count, (u32 length, tile)*" at `off`; a part is [buf, offset, length]. */
function readTiles(dv, off) {
  const k = dv.getUint32(off, true), parts = [];
  off += 4;
  for (let j = 0; j < k; j++) {
    const n = dv.getUint32(off, true);
    parts.push([dv.buffer, off + 4, n]);
    off += 4 + n;
  }
  return [parts, off];
}

/** Tile parts per request of a /api/tiles response. */
function unframe(buf, nreq) {
  const dv = new DataView(buf), out = [];
  let off = 0, parts;
  for (let i = 0; i < nreq; i++) [parts, off] = readTiles(dv, off), out.push(parts);
  if (off !== buf.byteLength) throw new Error("tile response length mismatch");
  return out;
}

/** [[run path, tile parts]] of a /api/tiles/bundle response. */
function unbundle(buf) {
  const dv = new DataView(buf), dec = new TextDecoder(), out = [];
  let off = 4, parts;
  for (let i = dv.getUint32(0, true); i > 0; i--) {
    const plen = dv.getUint32(off, true), path = dec.decode(new Uint8Array(buf, off + 4, plen));
    [parts, off] = readTiles(dv, off + 4 + plen + (-plen & 3));
    out.push([path, parts]);
  }
  if (off !== buf.byteLength) throw new Error("tile bundle length mismatch");
  return out;
}

// ---- columns from tiles ----------------------------------------------------------

/** Points of finer tiles (in tile order, so by step) before step s0: {s, v, t, n}. */
export function finePoints(fine, s0) {
  if (!fine.length) return NO_FINE;
  let cap = 0;
  for (const f of fine) cap += f.tile.count;
  const s = new Float64Array(cap), v = new Float64Array(cap), t = new Float64Array(cap), k = new Float64Array(cap);
  let n = 0;
  for (const { tile } of fine) {
    const w = 2 ** tile.level, base = tile.index * TILE, c = tile.count, { u16, f32, u32 } = tile;
    for (let i = 0; i < c; i++) {
      const x = (base + u16[tile.b + i] + f32[tile.f + 4 * c + i]) * w;
      if (x < s0) (s[n] = x), (v[n] = f32[tile.f + 2 * c + i]), (t[n] = f32[tile.f + 3 * c + i]), (k[n] = u32[tile.f + 5 * c + i]), n++;
    }
  }
  return { s, v, t, k, n };
}

const merged = new Float64Array(8); // buckets merged into one: value, step and runtime sums and count, of the finite buckets then of the rest

/** A column: `src` tiles' buckets merged f-fold (count-weighted means of value, step and runtime, over the buckets
 * with a finite mean, or all of them when none has one, as `tiles.coarsen`; w0 is their bucket width) outside
 * `ranges`, the finer points `fp` inside them, then the tail rows, which the tiles do not hold, in buckets as tiles
 * have them: merged into the same buckets, or with `fineW`, in buckets that wide after the finer points. */
export function buildColumn(src, f, w0, fp, ranges, tail, fineW = 0) {
  let cap = tail.n + fp.n;
  for (const t of src) cap += t.count;
  const { view: all, loc } = columnStore(3 * cap), s = all.subarray(0, cap), v = all.subarray(cap, 2 * cap), tt = all.subarray(2 * cap);
  const s0 = fineW ? tail.s0 : Infinity; // tile buckets from here on are left to the finer points and the tail
  let n = 0, fi = 0, ri = 0, cur = null;
  const emit = () => {
    const o = merged[3] > 0 ? 0 : 4, cnt = merged[o + 3], x = merged[o + 1] / cnt;
    if (cur === null || !(cnt > 0) || !(x < s0)) return;
    while (ri < ranges.length && ranges[ri][1] <= x) ri++;
    if (ri < ranges.length && x >= ranges[ri][0]) return;
    while (fi < fp.n && fp.s[fi] < x) (s[n] = fp.s[fi]), (v[n] = fp.v[fi]), (tt[n] = fp.t[fi]), n++, fi++;
    (s[n] = x), (v[n] = merged[o] / cnt), (tt[n] = merged[o + 2] / cnt), n++;
  };
  const add = (bk, value, step, runtime, k) => {
    const o = Number.isFinite(value) ? 0 : 4;
    if (bk !== cur) emit(), (cur = bk), merged.fill(0);
    merged[o] += value * k;
    merged[o + 1] += step * k;
    merged[o + 2] += runtime * k;
    merged[o + 3] += k;
  };
  for (const t of src) {
    const base = t.index * TILE, c = t.count, { u16, f32, u32 } = t, mo = t.f + 2 * c, to = t.f + 3 * c, so = t.f + 4 * c, no = t.f + 5 * c;
    for (let i = 0; i < c; i++) {
      const bi = base + u16[t.b + i];
      add(Math.floor(bi / f), f32[mo + i], (bi + f32[so + i]) * w0, f32[to + i], u32[no + i]);
    }
  }
  if (!fineW) for (let i = 0; i < tail.n; i++) if (tail.v[i] === tail.v[i]) add(Math.floor(tail.s[i] / (f * w0)), tail.v[i], tail.s[i], tail.t[i], 1);
  emit();
  const last = fp.n && fi < fp.n ? fp.n - 1 : -1; // the last finer point, when it ends the column
  while (fi < fp.n) (s[n] = fp.s[fi]), (v[n] = fp.v[fi]), (tt[n] = fp.t[fi]), n++, fi++;
  if (fineW) n = bucketRows(tail, fineW, s, v, tt, n, last >= 0 ? fp.k[last] : 0);
  const col = Col.adopt(s, v, tt, n);
  holdStore(col, loc && { ...loc, cap });
  return col;
}

/** Rows (tail: s, v, t) in buckets of width w, as tiles hold them (means of the finite values, or of the infinities
 * when a bucket has none; NaN is no value), written at n onward; the new end. The bucket before n, of `k` values (0:
 * none), takes the rows that fall in it. */
function bucketRows(rows, w, s, v, tt, n, k) {
  let cur = null;
  merged.fill(0);
  if (k > 0 && rows.n && Math.floor(s[n - 1] / w) === Math.floor(rows.s[0] / w)) {
    const o = Number.isFinite(v[n - 1]) ? 0 : 4;
    n--;
    (cur = Math.floor(s[n] / w)), (merged[o] = v[n] * k), (merged[o + 1] = s[n] * k), (merged[o + 2] = tt[n] * k), (merged[o + 3] = k);
  }
  const emit = () => {
    const o = merged[3] > 0 ? 0 : 4, cnt = merged[o + 3];
    if (cnt > 0) (s[n] = merged[o + 1] / cnt), (v[n] = merged[o] / cnt), (tt[n] = merged[o + 2] / cnt), n++;
    merged.fill(0);
  };
  for (let i = 0; i < rows.n; i++) {
    const y = rows.v[i], bk = Math.floor(rows.s[i] / w), o = Number.isFinite(y) ? 0 : 4;
    if (y !== y) continue;
    if (bk !== cur) emit(), (cur = bk);
    (merged[o] += y), (merged[o + 1] += rows.s[i]), (merged[o + 2] += rows.t[i]), (merged[o + 3] += 1);
  }
  emit();
  return n;
}

// ---- per-run, per-metric tile state ------------------------------------------

/** Tiles held for one metric of one run. */
class Entry {
  constructor() {
    this.top = null; // decoded top tiles ([] before the run has any)
    this.ov = null; // decoded overview tiles: the top tiles merged OVERVIEW_UP levels up
    this.ovSeq = -1; // run tiles_seq when `ov` was built
    this.up = 0; // levels above the top level at which the column shows them (merged locally)
    this.topSeq = -1; // run tiles_seq when `top` was built
    this.level = null; // level of the top tiles
    this.span = null; // [lo, hi] steps of this metric
    this.fine = new Map(); // "level|index" -> {tile, seq, used}
    this.held = null; // top tiles fetched ahead, {parts, seq}, until a plan wants them
    this.want = null; // finer level the current view asks for
    this.need = []; // keys of the tiles at `want` the view needs
    this.empty = new Set(); // keys answered with no tile
    this.shown = null; // finer level the column shows: `want` once all of `need` has arrived
    this.sig = null; // inputs of the column last built: [topSeq, up, shown, fine tiles, tail rows, tail start]
  }
}

// ---- store -----------------------------------------------------------------

export class Data {
  constructor(ui) {
    this.ui = ui; // {runs(), data(keys, run|null, streamed), keys(), media(key), status(text), conn(live), replan(), idle(), ahead(),
    //   protocol(server's)}
    this.info = null; // /api/info of what the page shows
    this.gen = 0; // bumped by `close`; work begun under an older generation is dropped
    this.probes = new Map(); // metric -> whether its IndexedDB read finished
    this.idbQueue = []; // top tiles awaiting their IndexedDB write
    this.putTimer = null; // pending `writePuts`
    this.rebuildQ = new Map(); // "run\0key" -> [run, key] awaiting rebuildSoon
    this.version = 0; // bumped whenever runs, their metadata, tiles or rows change
    this.held = 0; // bytes of top tiles held ahead of need (Entry.held)
    this.slabs = new Map(); // "key|level|index" -> {key, level, index, loc, views, rowOf (run id -> row), bytes, used}
    this.slabBytes = 0;
    this.noSlabs = false; // the server answers no slabs (a workspace)
    this.installing = 0; // installs of held tiles under way
    this.prefetchT = 0;
    this.prefetching = false;
    this.planned = null; // inputs of the last plan
    this.rebuildT = 0;
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
    this.queue = [];
    this.inflight = new Set(); // request ids
    this.posts = 0; // POSTs in flight
    this.fineBytes = 0;
    this.touched = new Set();
    this.stats = { requests: 0, tiles: 0, bytes: 0, idbHits: 0, rows: 0 };
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
    this.fineBytes = 0;
    this.held = 0;
    for (const sl of this.slabs.values()) freeStore(sl.loc);
    this.slabs.clear();
    this.slabBytes = 0;
    clearTimeout(this.prefetchT);
    this.probes = new Map();
    this.idbQueue = [];
    this.rebuildQ.clear();
    this.gen++;
  }

  newRun(meta) {
    const r = { id: meta.id, meta, seq: meta.tiles_seq ?? 0, mseq: meta.mseq, cols: new Map(), tiles: new Map(),
                tail: [], tailSeq0: meta.tiles_seq ?? 0, pending: [], hbExpect: null, resyncs: 0, resyncInFlight: false,
                holding: false }; // holding: events wait in `pending` until a resync finishes
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
    for (const e of r.tiles.values()) for (const f of e.fine.values()) this.fineBytes -= f.tile.bytes;
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

  // ---- planning ----

  /** Replace the request queue for what the visible charts show:
   * [{key, runs (by priority), runsSig (a hash of the set of runs), xmode, zoomed, x0, x1, pw, coarseAbove (runs above
   * which each needs only coarse buckets)}]. */
  plan(demands) {
    const sig = [this.version, ...demands.map((d) => [d.key, d.xmode, d.zoomed, d.x0, d.x1, d.pw, d.coarseAbove, d.runsSig,
                                                      d.slabs && `${d.slabs.level}:${d.slabs.indices}`].join("|"))].join("\n");
    if (sig === this.planned && !this.queue.length) return this.inflight.size; // the same plan, all of it requested
    this.planned = sig;
    const tiers = [[], [], []];
    const held = [];
    for (const d of demands) this.needSlabs(d, tiers), held.push(...this.planChart(d, tiers));
    this.installHeld(held);
    const q = [];
    for (const t of tiers) for (const x of t) if (!this.inflight.has(this.reqId(x))) q.push(x);
    this.queue = q;
    this.pump();
    const n = q.length + this.inflight.size;
    this.ui.status(n ? `loading ${n} tiles…` : this.summary());
    return n;
  }

  /** Queue one chart's requests: kept tiers in tiers[0] (overview) and tiers[1] (top), finer tiles in tiers[2]; the
   * held top tiles it wants, as install items. */
  planChart(d, tiers) {
    const all = this.runsWith(d.runs, d.key), runs = d.slabs ? all.filter((r) => !slabbed(d, r)) : all;
    const ranges = runs.map((r) => this.stepRange(r, d));
    let overlap = 0, lo = Infinity, hi = -Infinity;
    for (const g of ranges) if (g) (overlap += g[1] - g[0]), (lo = Math.min(lo, g[0])), (hi = Math.max(hi, g[1]));
    const span = d.zoomed && d.xmode === X_STEP ? d.x1 - d.x0 : hi - lo; // steps across the chart
    const pxPerBucket = runs.length > d.coarseAbove ? DENSITY_PX_PER_BUCKET : LINE_PX_PER_BUCKET;
    const chart = {
      d, span, ovNeed: [], topNeed: [], held: [], fineQ: tiers[2], probed: this.probed(d.key), overview: runs.length > OVERVIEW_MIN_RUNS,
      coarse: runs.length > d.coarseAbove, // merging stays at the budget's, whatever the zoom
      up: Math.max(0, Math.ceil(Math.log2((runs.length * TOP_BUCKETS) / POINT_BUDGET))), // local merging the budget needs
      // the level a run needs: buckets about pxPerBucket of chart width, within the point and finer-tile budgets
      level: Math.max(MIN_LEVEL, Math.floor(Math.log2(Math.max(span / Math.max(d.pw / pxPerBucket, 1), overlap / POINT_BUDGET,
                                                            (runs.length * span) / (TILE * FINE_TILES), 2 ** MIN_LEVEL)))),
    };
    runs.forEach((r, i) => this.planRun(r, ranges[i], chart));
    // many runs missing a tier: one bundle for the whole chart, else per-run requests
    for (const [tier, kind, need] of [[0, "overview", chart.ovNeed], [1, "top", chart.topNeed]]) {
      const bundle = { bundle: true, key: d.key, kind, runs: need };
      if (!need.length || this.inflight.has(this.reqId(bundle))) continue;
      if (need.length >= BUNDLE_MIN && need.length * BUNDLE_SHARE >= runs.length) tiers[tier].push(bundle);
      else for (const r of need) tiers[tier].push([r, d.key, kind]);
    }
    return chart.held;
  }

  /** One run of a chart: which kept tier it needs, its local merging and finer level, and finer tiles to fetch. */
  planRun(r, g, chart) {
    const { d } = chart;
    let e = r.tiles.get(d.key);
    if (!e) r.tiles.set(d.key, (e = new Entry()));
    const L = g && e.level !== null && chart.span > 0 ? chart.level : null;
    this.needKept(r, e, L, chart);
    const wasUp = e.up;
    // coarse buckets keep a merging one level above the budget's, so that views of more or fewer runs share columns
    e.up = L === null ? chart.up : chart.coarse ? Math.min(Math.max(chart.up, wasUp), chart.up + 1)
      : Math.min(chart.up, Math.max(0, L - e.level));
    e.want = L !== null && L < e.level && (d.zoomed || d.pw > 2 * TOP_BUCKETS) ? L : null;
    e.need = [];
    if (e.want !== null) this.needFine(r, e, g, chart);
    if (this.showFine(e) || e.up !== wasUp) this.rebuildSoon(r, d.key);
  }

  needFine(r, e, g, chart) {
    const W = 2 ** e.want * TILE;
    for (let k = Math.floor(g[0] / W); k <= Math.floor(g[1] / W); k++) {
      const key = `${e.want}|${k}`, f = e.fine.get(key);
      e.need.push(key);
      if (f && f.seq >= r.meta.tiles_seq) continue;
      if (f) f.hold = performance.now() + FINE_HOLD_MS;
      chart.fineQ.push([r, chart.d.key, e.want, k]);
    }
  }

  /** Show the wanted finer level once every tile the view needs at it has arrived; whether that changed. */
  showFine(e) {
    if (e.shown === e.want || !e.need.every((k) => e.fine.has(k) || e.empty.has(k))) return false;
    e.shown = e.want;
    return true;
  }

  /** Many-run charts start from overview tiles, fetching top tiles only where those are too coarse: by any level for
   * a line, by two for coarse buckets (a heatmap's or group statistics'). Until IndexedDB has answered for the
   * metric's top tiles, every chart starts from overview tiles. */
  needKept(r, e, L, chart) {
    const slack = chart.coarse ? 1 : 0, ts = r.meta.tiles_seq;
    const wantTop = chart.probed && (!chart.overview || !!e.top || (L !== null && L < e.level + OVERVIEW_UP - slack));
    if (wantTop) this.needTop(r, e, chart);
    else if (!e.top && (!e.ov || e.ovSeq !== ts)) chart.ovNeed.push(r);
  }

  /** Top tiles for run r of a chart: none when it has current ones; the held ones when current (once: they stay held
   * until installed); else a request. */
  needTop(r, e, chart) {
    const ts = r.meta.tiles_seq;
    if (e.top && e.topSeq === ts) return;
    if (e.held?.seq !== ts) return chart.topNeed.push(r);
    if (!e.held.taken) chart.held.push([[r, chart.d.key, "top"], e.held.parts, ts]), (e.held.taken = true);
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

  /** Steps of run r in the demand's view (all when not zoomed), or null. */
  stepRange(r, d) {
    const e = r.tiles.get(d.key);
    if (!e || !e.span) return null;
    let [lo, hi] = e.span;
    if (!d.zoomed) return [lo, hi];
    if (d.xmode === X_STEP) [lo, hi] = [Math.max(lo, d.x0), Math.min(hi, d.x1)];
    else {
      const c = r.cols.get(d.key);
      if (!c || !c.n) return null;
      let a = Infinity, b = -Infinity;
      for (let i = 0; i < c.n; i++) {
        const t = c.t[i];
        if (t >= d.x0 && t <= d.x1) (a = Math.min(a, c.s[i])), (b = Math.max(b, c.s[i]));
      }
      if (!(b >= a)) return null;
      const w = 2 ** (e.level + e.up);
      [lo, hi] = [Math.max(e.span[0], a - w), Math.min(e.span[1], b + w)];
    }
    return hi >= lo ? [lo, hi] : null;
  }

  reqId(x) {
    if (x.bundle) return `\0bundle\0${x.key}\0${x.kind}`;
    if (x.slab) return `\0slab\0${x.key}\0${x.level}\0${x.index}`;
    const [r, key, a, b] = x;
    return `${r.id}\0${key}\0${a}\0${b}`;
  }

  // ---- slabs ----

  /** The slabs (level, indices) of `key` when all are here (each marked used), else null. */
  slabsOf(key, level, indices) {
    const out = indices.map((k) => this.slabs.get(`${key}|${level}|${k}`));
    if (out.some((sl) => !sl)) return null;
    const now = performance.now();
    for (const sl of out) sl.used = now;
    return out;
  }

  /** The finest slabs of `key` here covering steps [x0, x1], at `level` or up to SLAB_UP levels coarser (each marked
   * used): {level, slabs}, or null. */
  bestSlabs(key, level, x0, x1) {
    for (let lv = level; lv <= level + SLAB_UP; lv++) {
      const w = SLAB_TILE * 2 ** lv, k0 = Math.floor(x0 / w), k1 = Math.floor(x1 / w);
      if (k1 - k0 >= SLAB_UP) continue;
      const slabs = this.slabsOf(key, lv, Array.from({ length: k1 - k0 + 1 }, (_, i) => k0 + i));
      if (slabs) return { level: lv, slabs };
    }
    return null;
  }

  /** [first, last] step of `key` in the slabs here, or null. */
  slabExtent(key) {
    let lo = Infinity, hi = -Infinity;
    for (const sl of this.slabs.values()) if (sl.key === key && sl.ext) (lo = Math.min(lo, sl.ext[0])), (hi = Math.max(hi, sl.ext[1]));
    return hi >= lo ? [lo, hi] : null;
  }

  /** Queue the slabs of demand d that are neither here nor requested, in tiers[0]. */
  needSlabs(d, tiers) {
    if (!d.slabs || this.noSlabs) return;
    for (const index of d.slabs.indices) {
      const x = { slab: true, key: d.key, level: d.slabs.level, index };
      if (!this.slabs.has(`${x.key}|${x.level}|${index}`) && !this.inflight.has(this.reqId(x))) tiers[0].push(x);
    }
  }

  /** Fetch slab x ({key, level, index}) into shared memory; `answered` is called once its response has arrived. */
  async fetchSlab(x, answered) {
    const gen = this.gen, got = await this.slabOnWorker(x);
    answered();
    if (!got || gen !== this.gen) return;
    this.addSlab(x, got);
    this.touched.add(x.key);
    this.flush();
  }

  /** Slab x fetched by a worker ({buf, bytes, paths, ext}), or null; a server without slabs turns them off. */
  async slabOnWorker({ key, level, index }) {
    const url = new URL(`${BASE}/api/tiles/slab`, location.href).href;
    const got = await fetchSlabOnWorker(url, JSON.stringify({ key, level, index, scope: this.scope }));
    if (got.status === 404) this.noSlabs = true;
    if (got.error) console.warn("slab fetch failed", got.error);
    this.stats.bytes += got.bytes;
    return got.buf ? got : null;
  }

  /** Keep a slab fetched into shared memory ({buf, bytes, paths, ext}) as slab x, dropping the least recently used
   * beyond SLAB_BYTES. */
  addSlab(x, { buf, bytes, paths, ext }) {
    const { loc } = adoptStore(buf);
    const id = `${x.key}|${x.level}|${x.index}`, old = this.slabs.get(id);
    if (old) freeStore(old.loc), (this.slabBytes -= old.bytes);
    this.slabs.set(id, { key: x.key, level: x.level, index: x.index, loc, views: slabViews(buf, 0), ext,
                         rowOf: new Map(paths.map((p, i) => [p, i])), bytes, used: performance.now() });
    this.slabBytes += bytes;
    this.dropSlabs();
  }

  dropSlabs() {
    const now = performance.now(), old = [...this.slabs].filter(([, sl]) => now - sl.used > SLAB_KEEP_MS).sort((a, b) => a[1].used - b[1].used);
    for (const [id, sl] of old) {
      if (this.slabBytes <= SLAB_BYTES) break;
      freeStore(sl.loc);
      this.slabBytes -= sl.bytes;
      this.slabs.delete(id);
    }
  }

  /** Fetch and install bundle b, one kept tier of one metric for every run of the scope (runs left out of the
   * response have none); `answered` is called once its response has arrived. */
  async fetchBundle(b, answered) {
    const gen = this.gen, seqs = new Map(b.runs.map((r) => [r, r.meta.tiles_seq]));
    const buf = await this.post(`${BASE}/api/tiles/bundle`, { key: b.key, kind: b.kind, scope: this.scope });
    answered();
    const entries = buf && this.parse(() => unbundle(buf));
    if (!entries) return this.retry(gen, [b]);
    if (gen !== this.gen) return;
    this.stats.requests += 1;
    const items = [];
    for (const [path, parts] of entries) {
      const r = this.runs.get(path);
      if (seqs.has(r)) items.push([[r, b.key, b.kind], parts, seqs.get(r)]), seqs.delete(r);
    }
    for (const [r, seq] of seqs) items.push([[r, b.key, b.kind], [], seq]);
    await this.install(gen, items);
  }

  // ---- fetching ----

  async pump() {
    while (this.posts < PARALLEL && this.queue.length) {
      const batch = [];
      if (this.queue[0].bundle || this.queue[0].slab) batch.push(this.queue.shift());
      else while (this.queue.length && !this.queue[0].bundle && !this.queue[0].slab && batch.length < BATCH) batch.push(this.queue.shift());
      for (const x of batch) this.inflight.add(this.reqId(x));
      this.posts++;
      let posted = true;
      const answered = () => { // the next POST goes out while this answer installs
        if (!posted) return;
        posted = false;
        this.posts--;
        this.pump();
      };
      const fetch = batch[0].slab ? this.fetchSlab(batch[0], answered) : batch[0].bundle ? this.fetchBundle(batch[0], answered)
        : this.fetchBatch(batch, answered);
      fetch.finally(() => {
        answered();
        for (const x of batch) this.inflight.delete(this.reqId(x));
        if (!this.busy) this.ui.status(this.summary());
        this.settle();
      });
    }
  }

  /** Whether tiles are queued, being fetched or installed, or columns await rebuilding. */
  get busy() {
    return this.queue.length > 0 || this.inflight.size > 0 || this.rebuildQ.size > 0 || this.installing > 0;
  }

  /** Tell the UI once the work for its last plan is done, then fetch ahead while nothing else is under way. */
  settle() {
    if (this.busy) return;
    this.ui.idle();
    clearTimeout(this.prefetchT);
    this.prefetchT = setTimeout(() => this.prefetch(), PREFETCH_IDLE_MS);
  }

  // ---- fetching ahead ----

  /** Fetch the next bundle a view may need (`nextAhead`), when the store is idle; then the one after. */
  async prefetch() {
    if (this.busy || this.prefetching) return;
    const next = this.nextAhead();
    if (!next) return;
    const gen = this.gen;
    this.prefetching = true;
    try {
      await this.fetchAhead(gen, next);
    } finally {
      this.prefetching = false;
    }
    if (gen === this.gen) this.settle();
  }

  /** What to fetch ahead: slabs zooms of visible charts would draw from, then for metrics without them top tiles of
   * the metrics charts show, overview tiles of charts not loaded yet, and top tiles of the other metrics; tiers of
   * running runs change, and are left to the views. */
  nextAhead() {
    const ahead = this.ui.ahead?.() || [];
    const slab = this.slabAhead(ahead);
    if (slab) return slab;
    const lacking = (fn) => {
      for (const { key, runs, slabs } of ahead) {
        if (slabs.length) continue; // its zooms draw from slabs
        const need = this.runsWith(runs, key).filter((r) => r.meta.state !== "running" && r.meta.tiles_seq > 0 && fn(r, r.tiles.get(key)));
        if (need.length >= BUNDLE_MIN) return { key, runs: need };
      }
      return null;
    };
    const top = (r, e) => this.held < PREFETCH_BYTES && !(e?.top && e.topSeq === r.meta.tiles_seq) && e?.held?.seq !== r.meta.tiles_seq;
    const shown = lacking((r, e) => e && (e.top || e.ov) && top(r, e));
    if (shown) return { ...shown, kind: "top" };
    const bare = lacking((r, e) => !e || !(e.top || e.ov));
    if (bare) return { ...bare, kind: "overview" };
    const rest = lacking(top);
    return rest && { ...rest, kind: "top" };
  }

  /** The first slab a zoom of a visible chart would draw from that is not here, within SLAB_AHEAD_BYTES. */
  slabAhead(ahead) {
    if (this.noSlabs || this.slabBytes >= SLAB_AHEAD_BYTES) return null;
    for (const { key, slabs } of ahead) {
      for (const { level, indices } of slabs) {
        const index = indices.find((k) => !this.slabs.has(`${key}|${level}|${k}`));
        if (index !== undefined) return { slab: true, key, level, index };
      }
    }
    return null;
  }

  /** Fetch one slab or bundle ahead: slabs are kept, overview tiles installed, top tiles held (Entry.held) until a
   * plan wants them. */
  async fetchAhead(gen, { slab, key, kind, runs, level, index }) {
    if (slab) {
      const got = await this.slabOnWorker({ key, level, index });
      if (got && gen === this.gen) this.addSlab({ key, level, index }, got);
      return;
    }
    const seqs = new Map(runs.map((r) => [r, r.meta.tiles_seq]));
    const buf = await this.post(`${BASE}/api/tiles/bundle`, { key, kind, scope: this.scope });
    const entries = buf && this.parse(() => unbundle(buf));
    if (!entries || gen !== this.gen) return;
    const items = [];
    for (const [path, parts] of entries) {
      const r = this.runs.get(path);
      if (!seqs.has(r)) continue;
      items.push([[r, key, kind], parts, seqs.get(r)]);
      seqs.delete(r);
    }
    for (const [r, seq] of seqs) items.push([[r, key, kind], [], seq]);
    if (kind === "overview") return this.install(gen, items);
    for (const [[r], parts, seq] of items) {
      let e = r.tiles.get(key);
      if (!e) r.tiles.set(key, (e = new Entry()));
      this.hold(e, { parts, seq });
    }
  }

  /** Hold top tiles `h` ({parts, seq}) for entry e, or drop the ones it holds (h null). */
  hold(e, h) {
    const bytes = (x) => (x ? x.parts.reduce((n, [, , len]) => n + len, 0) : 0);
    this.held += bytes(h) - bytes(e.held);
    e.held = h;
  }

  /** Install held top tiles a plan wants: [[run, key, "top"], parts, seq] items, in slices as fetched ones are. */
  installHeld(items) {
    if (!items.length) return;
    this.installing++;
    this.install(this.gen, items).finally(() => {
      this.installing--;
      this.settle();
    });
  }

  summary() {
    const s = this.stats;
    return `${this.runs.size} runs · ${s.tiles} tiles fetched (${(s.bytes / 1e6).toFixed(1)} MB), ${s.idbHits} from cache`;
  }

  /** IndexedDB key prefix of a metric's top tiles (followed by run id and uid). */
  idbPrefix(key) {
    return `${this.rootKey}\0${key}\0`;
  }

  /** Whether IndexedDB's top tiles of `key` are installed; the first call starts the read. */
  probed(key) {
    if (!idb.db) return true;
    const st = this.probes.get(key);
    if (st !== undefined) return st;
    this.probes.set(key, false);
    const gen = this.gen, prefix = this.idbPrefix(key);
    idb.withPrefix("tiles", prefix).then((entries) => {
      if (gen !== this.gen) return;
      for (const [k, v] of entries) {
        const rest = k.slice(prefix.length), cut = rest.lastIndexOf("\0");
        const r = this.runs.get(rest.slice(0, cut));
        if (!r || r.meta.uid !== rest.slice(cut + 1) || r.meta.state === "running" || v.seq !== r.meta.tiles_seq) continue;
        const e = r.tiles.get(key);
        if (!e) r.tiles.set(key, new Entry());
        this.stats.idbHits++;
        this.apply([r, key, "top"], v.bufs.map((b) => [b, 0, b.byteLength]), v.seq);
      }
      this.probes.set(key, true);
      this.version++;
      this.flush();
      this.ui.replan();
    });
    return false;
  }

  /** Queue top tiles for IndexedDB. */
  cachePut(r, key, seq, parts) {
    this.idbQueue.push([`${this.idbPrefix(key)}${r.id}\0${r.meta.uid}`, seq, parts]);
    if (!this.putTimer) this.putTimer = setTimeout(() => this.writePuts(), 500);
  }

  /** Write queued tiles in the main thread's idle time, once no fetch is pending: one transaction per idle period. */
  async writePuts() {
    while (this.idbQueue.length) {
      const idle = await idleTime();
      if (this.busy) continue;
      const at = Date.now(), q = this.idbQueue;
      let i = 0;
      const t0 = performance.now();
      const done = idb.putEach("tiles", () => {
        if (i >= q.length || idle.timeRemaining() < 2 || performance.now() - t0 > IDB_SLICE_MS) return null;
        const [k, seq, parts] = q[i++];
        return [k, { seq, bufs: parts.map(([b, o, n]) => b.slice(o, o + n)), at }];
      });
      q.splice(0, i);
      await done;
    }
    this.putTimer = null;
  }

  /** Fetch and install the tiles of requests `want`; `answered` is called once the response has arrived. */
  async fetchBatch(want, answered) {
    const gen = this.gen, seqs = want.map((x) => (isKept(x) ? x[0].meta.tiles_seq : x[0].seq));
    const reqs = want.map((x) => (isKept(x) ? [x[0].id, x[1], x[2]] : [x[0].id, x[1], x[2], x[3]]));
    const buf = await this.post(`${BASE}/api/tiles`, reqs);
    answered();
    const lists = buf && this.parse(() => unframe(buf, want.length));
    if (!lists) return this.retry(gen, want);
    if (gen !== this.gen) return;
    this.stats.requests += want.length;
    await this.install(gen, want.map((x, i) => [x, lists[i], seqs[i]]));
  }

  /** Response bytes of a POST, or null on failure. */
  async post(url, body) {
    try {
      const res = await fetch(url, { method: "POST", body: JSON.stringify(body) });
      if (!res.ok) throw new Error(`${url}: ${res.status}`);
      const buf = await res.arrayBuffer();
      this.stats.bytes += buf.byteLength;
      return buf;
    } catch (e) {
      console.warn("tile fetch failed", e);
      return null;
    }
  }

  parse(fn) {
    try {
      return fn();
    } catch (e) {
      console.warn("bad tile response", e);
      return null;
    }
  }

  /** Queue failed requests again in a second, up to 3 tries each. */
  retry(gen, items) {
    const again = items.filter((x) => (x.tries = (x.tries || 0) + 1) < 3);
    setTimeout(() => {
      if (gen !== this.gen) return;
      this.queue.unshift(...again.filter((x) => !this.inflight.has(this.reqId(x))));
      this.pump();
    }, 1000);
  }

  /** Apply [request, tile parts, seq] items in slices of REBUILD_SLICE_MS, so input is never blocked. Tiles of
   * running runs only are streamed data. */
  async install(gen, items) {
    let t0 = performance.now();
    const live = items.every(([x]) => x[0].meta.state === "running");
    for (const [x, parts, seq] of items) {
      this.stats.tiles += parts.length;
      this.apply(x, parts, seq);
      if (x[2] === "top" && x[0].meta.state !== "running") this.cachePut(x[0], x[1], seq, parts);
      if (performance.now() - t0 > REBUILD_SLICE_MS) {
        this.flush(live);
        await new Promise((ok) => setTimeout(ok, 0));
        if (gen !== this.gen) return;
        t0 = performance.now();
      }
    }
    this.flush(live);
  }

  /** Install the tiles answering request x (parts as from `unframe`). */
  apply(x, parts, seq) {
    const [r, key, a, b] = x, e = this.runs.get(r.id) === r && r.tiles.get(key);
    if (!e) return;
    let tiles;
    try {
      tiles = parts.map(([buf, off, len]) => decodeTile(buf, off, len));
    } catch (err) {
      console.warn(`run ${r.id} ${key}: ${err.message}`);
      return;
    }
    if (a === "top" && e.held) this.hold(e, null);
    const changed = a === "top" ? this.setTop(r, e, tiles, seq) : a === "overview" ? this.setOverview(r, e, tiles, seq)
      : this.setFine(e, `${a}|${b}`, tiles, seq);
    if (!isKept(x)) {
      if (!tiles.length) e.empty.add(`${a}|${b}`);
      this.pruneTail(r);
      if (this.showFine(e) || changed) this.rebuild(r, key);
    } else if (changed) this.rebuild(r, key);
  }

  setTop(r, e, tiles, seq) {
    if (seq < e.topSeq || (seq === e.topSeq && e.top)) return false;
    [e.top, e.topSeq] = [tiles, seq];
    if (tiles.length) [e.level, e.span] = [tiles[0].level, bucketSpan(tiles)];
    this.pruneTail(r);
    return true;
  }

  setOverview(r, e, tiles, seq) {
    if (seq <= e.ovSeq || e.top) return false;
    [e.ov, e.ovSeq] = [tiles, seq];
    if (tiles.length) [e.level, e.span] = [tiles[0].level - OVERVIEW_UP, bucketSpan(tiles)];
    this.pruneTail(r);
    return true;
  }

  setFine(e, k, tiles, seq) {
    const old = e.fine.get(k);
    if (old) this.fineBytes -= old.tile.bytes;
    if (!tiles.length) return e.fine.delete(k);
    e.fine.set(k, { tile: tiles[0], seq, used: performance.now() });
    this.fineBytes += tiles[0].bytes;
    if (this.fineBytes > FINE_BYTES) this.evictFine();
    return true;
  }


  /** Drop least recently used finer tiles down to 3/4 of the budget. */
  evictFine() {
    const all = [];
    for (const r of this.runs.values())
      for (const [key, e] of r.tiles) for (const [k, f] of e.fine) all.push([f.used, r, key, e, k, f]);
    all.sort((a, b) => a[0] - b[0]);
    for (const [, r, key, e, k, f] of all) {
      if (this.fineBytes <= 0.75 * FINE_BYTES) break;
      e.fine.delete(k);
      this.fineBytes -= f.tile.bytes;
      this.rebuild(r, key);
    }
  }

  /** Rebuild the column of one metric of one run from its best tiles and its tail rows. */
  rebuild(r, key) {
    const e = r.tiles.get(key);
    if (!e || !(e.top || e.ov)) return;
    const srcUp = e.top ? 0 : OVERVIEW_UP; // overview tiles sit OVERVIEW_UP levels above the top level
    // a finer tile holds rows [0, f.seq); rows from tailSeq0 on come from the tail
    const fine = e.shown === null ? [] : [...e.fine.values()].filter((f) => f.tile.level === e.shown && f.seq >= r.tailSeq0);
    fine.sort((a, b) => a.tile.index - b.tile.index);
    const tail = this.tailOf(r, key, fine.length ? Math.max(...fine.map((f) => f.seq)) : e.top ? e.topSeq : e.ovSeq); // rows no tile shown holds
    const sig = [e.top ? e.topSeq : e.ovSeq, e.up, e.shown, fine.map((f) => `${f.tile.index}@${f.seq}`).join(), tail.n, tail.s0, srcUp];
    if (r.cols.has(key) && e.sig && sig.every((x, i) => x === e.sig[i])) return;
    e.sig = sig;
    const now = performance.now();
    for (const f of fine) f.used = now;
    const ranges = fine.map((f) => tileRange(f.tile.level, f.tile.index));
    const c = buildColumn(e.top || e.ov, 2 ** Math.max(0, e.up - srcUp), 2 ** (e.level + srcUp), finePoints(fine, tail.s0), ranges, tail,
                          fine.length ? 2 ** e.shown : 0);
    r.cols.set(key, c);
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


  /** Drop tail rows that every metric's top tiles already hold, and that no finer tile being refetched needs. */
  pruneTail(r) {
    let keep = r.meta.tiles_seq;
    const now = performance.now();
    for (const e of r.tiles.values()) {
      if (e.top || e.ov) keep = Math.min(keep, e.top ? e.topSeq : e.ovSeq);
      for (const f of e.fine.values()) if (f.hold > now && f.seq >= r.tailSeq0) keep = Math.min(keep, f.seq);
    }
    const drop = Math.min(r.tail.length, keep - r.tailSeq0);
    if (drop > 0) {
      r.tail.splice(0, drop);
      r.tailSeq0 += drop;
    }
  }

  /** Points of `key` in the run's tail rows from sequence number `from` on: {s, v, t, n, s0 (first step)}. */
  tailOf(r, key, from = -Infinity) {
    if (!r.tail.length) return NO_TAIL;
    const s = [], v = [], t = [];
    let s0 = Infinity;
    for (let i = from > -Infinity ? Math.max(0, from - r.tailSeq0) : 0; i < r.tail.length; i++) {
      const [step, rt, d] = r.tail[i];
      if (!(key in d)) continue;
      s.push(step), v.push(num(d[key])), t.push(rt);
      if (step < s0) s0 = step;
    }
    return { s, v, t, n: s.length, s0 };
  }

  /** Tell the UI which charts changed since the last flush. */
  /** Tell the UI which metrics changed: `streamed` when by rows from live runs, else by work for its view. */
  flush(streamed = false) {
    if (!this.touched.size) return;
    this.version++;
    const keys = this.touched;
    this.touched = new Set();
    this.flushKeys();
    this.ui.data(keys, null, streamed);
  }

  /** Discard a run's rows and tiles and reload it from its current server state. */
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
      const from = j.run.tiles_seq;
      const rows = from < j.run.seq ? await getJSON(`${BASE}/api/rows?path=${encodeURIComponent(r.id)}&from=${from}`) : { seq0: from, rows: [] };
      if (this.runs.get(r.id) !== r) return;
      r.tail = [];
      r.tailSeq0 = r.seq = from;
      this.appendRows(r, rows);
      for (const e of r.tiles.values()) (e.topSeq = -1), (e.sig = null);
      r.resyncInFlight = false;
      r.holding = false;
      r.resyncs = 0;
      const pending = r.pending;
      r.pending = [];
      for (const [kind, ev] of pending) this.dispatch(kind, ev);
      this.ui.runs();
      this.ui.data(new Set(r.tiles.keys()), null, true);
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
    this.stats.rows += rows.length - Math.max(0, skip);
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
      const e = r.tiles.get(k);
      if (e?.top || e?.ov) this.rebuild(r, k);
      else this.touched.add(k);
    }
    this.flush(true);
  }

  onRunMeta(meta) {
    const r = this.runs.get(meta.id);
    if (!r) {
      const n = this.newRun(meta);
      if (meta.seq > meta.tiles_seq || meta.mseq > 0) this.resync(n);
    } else {
      const summary = r.meta.summary;
      const before = r.meta.tiles_seq;
      this.setMeta(r, { ...meta, summary: { ...meta.summary, ...summary } });
      if (meta.tiles_seq !== before) this.ui.data(new Set(r.tiles.keys()), null, true);
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
    const c = await idb.get("blobs", rec.file);
    if (c && c.crc === rec.crc && c.buf.byteLength === rec.size) return c.buf;
    const buf = await (await fetch(mediaURL(rec))).arrayBuffer();
    if (buf.byteLength !== rec.size || crc32(buf) !== rec.crc) throw new Error(`media ${rec.file} failed verification`);
    idb.putMany("blobs", [[rec.file, { crc: rec.crc, buf }]]);
    return buf;
  }
}
