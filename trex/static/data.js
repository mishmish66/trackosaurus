// Data layer: run metadata, tiles fetched for what the visible charts show (`plan`), an IndexedDB tile
// cache, and the SSE stream. Each (run, metric) column is built from the best tiles present plus the
// rows streamed since its kept tiles [tiles_seq, seq); a gap or a missed heartbeat resyncs the run.

import { Col, crc32 } from "./kernel.js";

const num = (v) => (typeof v === "number" ? v : Number(v));

/** URL prefix of what the page shows: a daemon's tracked directory ("/r/<name>") or workspace ("/w/<name>"), else "". */
export const BASE = typeof location === "undefined" ? "" : (location.pathname.match(/^\/[rw]\/[^/]+(?=\/)/) || [""])[0];

/** URL of a media record's file; content-addressed, so browsers cache it as immutable. */
export const mediaURL = (rec) => `${BASE}/m/${encodeURIComponent(rec.run)}/${rec.file}`;

const TILE = 256;
const MIN_LEVEL = -20;
const POINT_BUDGET = 1.5e6; // points per chart across all its runs
const TOP_BUCKETS = 192; // typical non-empty buckets of a run's top tiles
const BATCH = 512; // tile requests per POST
const PARALLEL = 4; // POSTs in flight
const FINE_BYTES = 256e6; // decoded finer tiles kept in memory
const IDB_ENTRIES = 400000; // cached top-tile entries kept in IndexedDB
const IDB_CHUNK = 400; // entries per IndexedDB write
const REBUILD_SLICE_MS = 8; // column rebuilds per task, in ms
const OVERVIEW_UP = 2; // levels the server's overview tiles sit above the top tiles (index.OVERVIEW_UP)
const OVERVIEW_MIN_RUNS = 300; // charts with more runs load overview tiles before top tiles
const BUNDLE_MIN = 64; // runs of a chart missing a tier above which one bundle request fetches it...
const BUNDLE_SHARE = 4; // ...when they are also at least 1 / BUNDLE_SHARE of the chart's runs
const FINE_HOLD_MS = 10000; // tail rows a refetched finer tile needs are kept this long while it is fetched
const LINE_PX_PER_BUCKET = 2; // chart width per bucket a line needs
const DENSITY_PX_PER_BUCKET = 8; // the same in a density heatmap, where many lines overlap
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

/** Whether run r logs metric `key` (a set kept in step with r.meta.keys). */
function hasKey(r, key) {
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
        const r = indexedDB.open("trex", 5);
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

async function getJSON(url) {
  const r = await fetch(url, { cache: "no-store" });
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
function finePoints(fine, s0) {
  if (!fine.length) return NO_FINE;
  let cap = 0;
  for (const f of fine) cap += f.tile.count;
  const s = new Float64Array(cap), v = new Float64Array(cap), t = new Float64Array(cap);
  let n = 0;
  for (const { tile } of fine) {
    const w = 2 ** tile.level, base = tile.index * TILE, c = tile.count, { u16, f32 } = tile;
    for (let i = 0; i < c; i++) {
      const x = (base + u16[tile.b + i] + f32[tile.f + 4 * c + i]) * w;
      if (x < s0) (s[n] = x), (v[n] = f32[tile.f + 2 * c + i]), (t[n] = f32[tile.f + 3 * c + i]), n++;
    }
  }
  return { s, v, t, n };
}

/** A column: `src` tiles' buckets merged f-fold (count-weighted means of value, step and runtime; w0
 * is their bucket width) outside `ranges`, the finer points `fp` inside them, then the tail rows. */
function buildColumn(src, f, w0, fp, ranges, tail) {
  let cap = tail.n + fp.n;
  for (const t of src) cap += t.count;
  const all = new Float64Array(3 * cap), s = all.subarray(0, cap), v = all.subarray(cap, 2 * cap), tt = all.subarray(2 * cap);
  let n = 0, fi = 0, ri = 0, cur = null, sm = 0, ss = 0, st = 0, cnt = 0;
  const emit = () => {
    const x = ss / cnt;
    if (cur === null || !(cnt > 0) || !(x < tail.s0)) return;
    while (ri < ranges.length && ranges[ri][1] <= x) ri++;
    if (ri < ranges.length && x >= ranges[ri][0]) return;
    while (fi < fp.n && fp.s[fi] < x) (s[n] = fp.s[fi]), (v[n] = fp.v[fi]), (tt[n] = fp.t[fi]), n++, fi++;
    (s[n] = x), (v[n] = sm / cnt), (tt[n] = st / cnt), n++;
  };
  for (const t of src) {
    const base = t.index * TILE, c = t.count, { u16, f32, u32 } = t, mo = t.f + 2 * c, to = t.f + 3 * c, so = t.f + 4 * c, no = t.f + 5 * c;
    for (let i = 0; i < c; i++) {
      const bi = base + u16[t.b + i], bk = Math.floor(bi / f), k = u32[no + i];
      if (bk !== cur) emit(), (cur = bk), (sm = ss = st = cnt = 0);
      sm += f32[mo + i] * k;
      ss += (bi + f32[so + i]) * w0 * k;
      st += f32[to + i] * k;
      cnt += k;
    }
  }
  emit();
  while (fi < fp.n) (s[n] = fp.s[fi]), (v[n] = fp.v[fi]), (tt[n] = fp.t[fi]), n++, fi++;
  s.set(tail.s, n), v.set(tail.v, n), tt.set(tail.t, n);
  const col = Col.adopt(s, v, tt, n + tail.n);
  col.tailN = tail.n;
  return col;
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
    this.ui = ui; // {runs(), data(keys, run|null), keys(), media(key), status(text), conn(live), replan()}
    this.probes = new Map();
    this.puts = [];
    this.rebuildQ = new Map(); // "run\0key" -> [run, key] awaiting rebuildSoon
    this.rebuildT = 0;
    this.runs = new Map();
    this.keys = new Map(); // metric key -> number of runs having it
    this.media = new Map(); // media key -> Map(runId -> records sorted by step)
    this.folders = {}; // folder path -> info dict from its trex_info.json
    this.scope = null;
    this.rootKey = "";
    this.es = null;
    this.queue = [];
    this.inflight = new Set(); // request ids
    this.posts = 0;
    this.fineBytes = 0;
    this.touched = new Set();
    this.stats = { requests: 0, tiles: 0, bytes: 0, idbHits: 0, rows: 0 };
  }

  async init() {
    const [info] = await Promise.all([getJSON(`${BASE}/api/info`), idb.open()]);
    this.info = info;
    this.rootKey = info.root;
    idb.prune(IDB_ENTRIES);
  }

  close() {
    if (this.es) this.es.close();
    this.es = null;
    this.runs.clear();
    this.keys.clear();
    this.media.clear();
    this.folders = {};
    this.queue = [];
    this.inflight.clear();
    this.fineBytes = 0;
    this.probes = new Map(); // metric -> whether its IndexedDB read finished
    this.puts = [];
    this.rebuildQ.clear();
    this.gen = (this.gen || 0) + 1;
  }

  newRun(meta) {
    const r = { id: meta.id, meta, seq: meta.tiles_seq ?? 0, mseq: meta.mseq, cols: new Map(), tiles: new Map(),
                tail: [], tailSeq0: meta.tiles_seq ?? 0, syncing: false, pending: [], hbExpect: null, resyncs: 0 };
    this.runs.set(r.id, r);
    this.countKeys(r, meta.keys || [], 1);
    return r;
  }

  countKeys(r, keys, d) {
    this.keysVer = (this.keysVer || 0) + 1;
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
  }

  dropRun(r) {
    this.countKeys(r, r.meta.keys || [], -1);
    for (const e of r.tiles.values()) for (const f of e.fine.values()) this.fineBytes -= f.tile.bytes;
    this.runs.delete(r.id);
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
   * [{key, runs (by priority), xmode, zoomed, x0, x1, pw, densityAbove (lines above which a heatmap)}]. */
  plan(demands) {
    const tiers = [[], [], []];
    for (const d of demands) this.planChart(d, tiers);
    const q = [];
    for (const t of tiers) for (const x of t) if (!this.inflight.has(this.reqId(x))) q.push(x);
    this.queue = q;
    this.pump();
    const n = q.length + this.inflight.size;
    this.ui.status(n ? `loading ${n} tiles…` : this.summary());
    return n;
  }

  /** Queue one chart's requests: kept tiers in tiers[0] (overview) and tiers[1] (top), finer tiles in tiers[2]. */
  planChart(d, tiers) {
    const runs = this.runsWith(d.runs, d.key);
    const ranges = runs.map((r) => this.stepRange(r, d));
    let overlap = 0, lo = Infinity, hi = -Infinity;
    for (const g of ranges) if (g) (overlap += g[1] - g[0]), (lo = Math.min(lo, g[0])), (hi = Math.max(hi, g[1]));
    const span = d.zoomed && d.xmode === 0 ? d.x1 - d.x0 : hi - lo; // steps across the chart
    const pxPerBucket = runs.length > d.densityAbove ? DENSITY_PX_PER_BUCKET : LINE_PX_PER_BUCKET;
    const chart = {
      d, span, ovNeed: [], topNeed: [], fineQ: tiers[2], probed: this.probed(d.key), overview: runs.length > OVERVIEW_MIN_RUNS,
      up: Math.max(0, Math.ceil(Math.log2((runs.length * TOP_BUCKETS) / POINT_BUDGET))), // local merging the budget needs
      // the level a run needs: buckets about pxPerBucket of chart width, within the point budget
      level: Math.max(MIN_LEVEL, Math.floor(Math.log2(Math.max(span / Math.max(d.pw / pxPerBucket, 1), overlap / POINT_BUDGET, 2 ** MIN_LEVEL)))),
    };
    runs.forEach((r, i) => this.planRun(r, ranges[i], chart));
    // many runs missing a tier: one bundle for the whole chart, else per-run requests
    for (const [tier, kind, need] of [[0, "overview", chart.ovNeed], [1, "top", chart.topNeed]]) {
      const bundle = { bundle: true, key: d.key, kind, runs: need };
      if (!need.length || this.inflight.has(this.reqId(bundle))) continue;
      if (need.length >= BUNDLE_MIN && need.length * BUNDLE_SHARE >= runs.length) tiers[tier].push(bundle);
      else for (const r of need) tiers[tier].push([r, d.key, kind]);
    }
  }

  /** One run of a chart: which kept tier it needs, its local merging and finer level, and finer tiles to fetch. */
  planRun(r, g, chart) {
    const { d } = chart;
    let e = r.tiles.get(d.key);
    if (!e) r.tiles.set(d.key, (e = new Entry()));
    const L = g && e.level !== null && chart.span > 0 ? chart.level : null;
    if (chart.probed) this.needKept(r, e, L, chart);
    const wasUp = e.up;
    e.up = L === null ? chart.up : Math.min(chart.up, Math.max(0, L - e.level));
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

  /** Many-run charts start from overview tiles, fetching top tiles only where those are too coarse. */
  needKept(r, e, L, chart) {
    const ts = r.meta.tiles_seq, wantTop = !chart.overview || !!e.top || (L !== null && L < e.level + OVERVIEW_UP);
    if (wantTop && (!e.top || e.topSeq !== ts)) chart.topNeed.push(r);
    else if (!wantTop && (!e.ov || e.ovSeq !== ts)) chart.ovNeed.push(r);
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
    if (d.xmode === 0) [lo, hi] = [Math.max(lo, d.x0), Math.min(hi, d.x1)];
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
    const [r, key, a, b] = x;
    return `${r.id}\0${key}\0${a}\0${b}`;
  }

  /** One kept tier of one metric for every run of the scope; runs left out of the response have none. */
  async fetchBundle(b) {
    const gen = this.gen, seqs = new Map(b.runs.map((r) => [r, r.meta.tiles_seq]));
    const buf = await this.post(`${BASE}/api/tiles/bundle`, { key: b.key, kind: b.kind, scope: this.scope });
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
      if (this.queue[0].bundle) batch.push(this.queue.shift());
      else while (this.queue.length && !this.queue[0].bundle && batch.length < BATCH) batch.push(this.queue.shift());
      for (const x of batch) this.inflight.add(this.reqId(x));
      this.posts++;
      (batch[0].bundle ? this.fetchBundle(batch[0]) : this.fetchBatch(batch)).finally(() => {
        this.posts--;
        for (const x of batch) this.inflight.delete(this.reqId(x));
        if (!this.queue.length && !this.posts) this.ui.status(this.summary());
        this.pump();
      });
    }
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
      this.flush();
      this.ui.replan();
    });
    return false;
  }

  /** Queue top tiles for IndexedDB. */
  cachePut(r, key, seq, parts) {
    this.puts.push([`${this.idbPrefix(key)}${r.id}\0${r.meta.uid}`, seq, parts]);
    if (!this.putTimer) this.putTimer = setTimeout(() => this.writePuts(), 500);
  }

  /** Write queued tiles in small transactions, once no fetch is pending. */
  async writePuts() {
    while (this.puts.length) {
      if (this.posts || this.queue.length) {
        await new Promise((ok) => setTimeout(ok, 300));
        continue;
      }
      const at = Date.now();
      await idb.putMany("tiles", this.puts.splice(0, IDB_CHUNK).map(([k, seq, parts]) =>
        [k, { seq, bufs: parts.map(([b, o, n]) => b.slice(o, o + n)), at }]));
    }
    this.putTimer = null;
  }

  async fetchBatch(want) {
    const gen = this.gen, seqs = want.map(([r, , a]) => (typeof a === "string" ? r.meta.tiles_seq : r.seq));
    const buf = await this.post(`${BASE}/api/tiles`, want.map(([r, k, a, b]) => (typeof a === "string" ? [r.id, k, a] : [r.id, k, a, b])));
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

  /** Apply [request, tile parts, seq] items in slices of REBUILD_SLICE_MS, so input is never blocked. */
  async install(gen, items) {
    let t0 = performance.now();
    for (const [x, parts, seq] of items) {
      this.stats.tiles += parts.length;
      this.apply(x, parts, seq);
      if (x[2] === "top" && x[0].meta.state !== "running") this.cachePut(x[0], x[1], seq, parts);
      if (performance.now() - t0 > REBUILD_SLICE_MS) {
        this.flush();
        await new Promise((ok) => setTimeout(ok, 0));
        if (gen !== this.gen) return;
        t0 = performance.now();
      }
    }
    this.flush();
  }

  /** Install the tiles answering request x (parts as from `unframe`). */
  apply([r, key, a, b], parts, seq) {
    const e = this.runs.get(r.id) === r && r.tiles.get(key);
    if (!e) return;
    let tiles;
    try {
      tiles = parts.map(([buf, off, len]) => decodeTile(buf, off, len));
    } catch (err) {
      console.warn(`run ${r.id} ${key}: ${err.message}`);
      return;
    }
    const changed = a === "top" ? this.setTop(r, e, tiles, seq) : a === "overview" ? this.setOverview(r, e, tiles, seq)
      : this.setFine(e, `${a}|${b}`, tiles, seq);
    if (typeof a !== "string" && (changed || !tiles.length)) {
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
    const tail = this.tailOf(r, key);
    const sig = [e.top ? e.topSeq : e.ovSeq, e.up, e.shown, fine.map((f) => `${f.tile.index}@${f.seq}`).join(), tail.n, tail.s0, srcUp];
    if (r.cols.has(key) && e.sig && sig.every((x, i) => x === e.sig[i])) return;
    e.sig = sig;
    const now = performance.now();
    for (const f of fine) f.used = now;
    const ranges = fine.map((f) => tileRange(f.tile.level, f.tile.index));
    const c = buildColumn(e.top || e.ov, 2 ** Math.max(0, e.up - srcUp), 2 ** (e.level + srcUp), finePoints(fine, tail.s0), ranges, tail);
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

  /** Points of `key` in the run's rows beyond its top tiles: {s, v, t, n, s0 (first step)}. */
  tailOf(r, key) {
    if (!r.tail.length) return NO_TAIL;
    const s = [], v = [], t = [];
    for (const [step, rt, d] of r.tail) if (key in d) s.push(step), v.push(num(d[key])), t.push(rt);
    return { s, v, t, n: s.length, s0: s.length ? Math.min(...s) : Infinity };
  }

  /** Tell the UI which charts changed since the last flush. */
  flush() {
    if (!this.touched.size) return;
    const keys = this.touched;
    this.touched = new Set();
    this.flushKeys();
    this.ui.data(keys, null);
  }

  /** Discard a run's rows and tiles and reload it from its current server state. */
  async resync(r) {
    if (r.resyncing) return;
    r.resyncing = true;
    r.syncing = true;
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
      r.resyncing = false;
      r.syncing = false;
      r.resyncs = 0;
      const pending = r.pending;
      r.pending = [];
      for (const [kind, ev] of pending) this.dispatch(kind, ev);
      this.ui.runs();
      this.ui.data(new Set(r.tiles.keys()), null);
    } catch (e) {
      console.warn(`resync ${r.id} failed`, e);
      r.resyncing = false;
      setTimeout(() => this.resync(r), delay);
    }
  }

  // ---- stream ----

  openStream() {
    const es = new EventSource(`${BASE}/api/stream?path=${encodeURIComponent(this.scope)}`);
    this.es = es;
    for (const kind of ["rows", "run", "media", "delete", "hb", "folder"]) {
      es.addEventListener(kind, (e) => this.dispatch(kind, JSON.parse(e.data)));
    }
    es.onopen = () => this.ui.conn(true);
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
      r = this.newRun({ id, mseq: 0, seq: 0, tiles_seq: 0, keys: [], summary: {}, config: {}, tags: [], name: id, state: "running" });
      r.pending.push([kind, ev]);
      this.resync(r);
      return;
    }
    if (r.syncing) {
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
    this.flush();
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
      if (meta.tiles_seq !== before) this.ui.data(new Set(r.tiles.keys()), null);
      // The stream is ordered, so every row and media item this meta counts has already been delivered.
      if (!r.syncing && (r.seq < meta.seq || r.mseq < meta.mseq)) {
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
        this.dispatch("run", { id, seq, mseq, tiles_seq: 0, keys: [], summary: {}, config: {}, tags: [], name: id, state: "running" });
        continue;
      }
      if (r.syncing) continue;
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
