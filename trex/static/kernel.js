// Numeric kernel for the UI: resident metric columns (in memory shared with workers when the page is cross-origin
// isolated), CRC-32, time-weighted EMA smoothing, per-pixel decimation, group aggregation, and axis quantiles. Pure JS
// on typed arrays; it runs in the page and in its workers.

/** `flags` bits accepted by prep, agg and yrange; IQM (agg only) adds the interquartile mean. */
export const LOGY = 1, RAW = 2, LOGX = 4, IQM = 8;

/** `xmode` values: x is the step, or the runtime. */
export const X_STEP = 0, X_RUNTIME = 1;

/** Per-bin statistics returned by agg, in output order (stat-major). */
export const STATS = ["mean", "std", "n", "min", "max", "median", "q25", "q75", "medlo", "medhi", "iqm", "iqmse", "iqmh"];
export const NSTAT = STATS.length;
/** Row of each statistic in agg's output. */
const S = Object.freeze(Object.fromEntries(STATS.map((k, i) => [k, i])));

/** One metric of one run: points (step, value, runtime, and the count of rows each stands for when `w` is set) in
 * sequence order. */
export class Col {
  constructor() {
    this.n = 0;
    this.s = new Float64Array(64);
    this.v = new Float64Array(64);
    this.t = new Float64Array(64);
    this.w = null;
    this.sorted = [true, true];
    this.ext = [Infinity, -Infinity, Infinity, Infinity, -Infinity, Infinity]; // min, max, min positive: steps then runtimes
    this.sm = null;
    this.smKey = null;
    this.smState = null;
  }

  get len() {
    return this.n;
  }

  /** A column of n points whose steps, values, runtimes and counts are s, v, t, w (Float64Arrays, not copied; w null
   * for one row each), sorted as `sorted` says ([by step, by runtime]); its extents are not tracked. */
  static view(s, v, t, n, sorted, w = null) {
    const c = Object.create(Col.prototype);
    Object.assign(c, { n, s, v, t, w, sorted, ext: null, sm: null, smKey: null, smState: null });
    return c;
  }

  /** A column holding the first n points of s, v, t and w (Float64Arrays, taken over without copying; w null for one
   * row each). */
  static adopt(s, v, t, n, w = null) {
    const c = Object.create(Col.prototype);
    Object.assign(c, { n, s, v, t, w, sorted: [true, true], ext: [Infinity, -Infinity, Infinity, Infinity, -Infinity, Infinity],
                       sm: null, smKey: null, smState: null });
    track(c, s, t, n, -Infinity, -Infinity);
    return c;
  }

  /** Append points; s, v, t are equal-length array-likes. */
  push(s, v, t) {
    const n = s.length, need = this.n + n;
    if (need > this.s.length) {
      const cap = Math.max(need, this.s.length * 2);
      for (const k of ["s", "v", "t"]) {
        const a = new Float64Array(cap);
        a.set(this[k].subarray(0, this.n));
        this[k] = a;
      }
    }
    track(this, s, t, n, this.n ? this.s[this.n - 1] : -Infinity, this.n ? this.t[this.n - 1] : -Infinity);
    this.s.set(s, this.n);
    this.v.set(v, this.n);
    this.t.set(t, this.n);
    this.n = need;
  }

  /** [min, max, smallest positive] of x; null if empty. */
  extent(xmode) {
    if (!this.n) return null;
    const k = xmode === X_STEP ? 0 : 3;
    return [this.ext[k], this.ext[k + 1], this.ext[k + 2]];
  }

  xs(xmode) {
    return xmode === X_STEP ? this.s : this.t;
  }

  /** Debiased EMA decaying by alpha^(dx / scale) per point; incremental; non-finite values pass through. */
  ensureSmooth(alpha, scale, xmode) {
    if (alpha <= 0) return;
    if (!(scale > 0)) scale = 1;
    const key = `${alpha}|${scale}|${xmode}`;
    if (this.smKey !== key || !this.sm) {
      this.sm = new Float64Array(this.s.length);
      this.smKey = key;
      this.smState = { done: 0, acc: 0, deb: 0, last: NaN };
    } else if (this.sm.length < this.n) {
      const a = new Float64Array(this.s.length);
      a.set(this.sm.subarray(0, this.smState.done));
      this.sm = a;
    }
    const st = this.smState, xs = this.xs(xmode), v = this.v, sm = this.sm;
    let { acc, deb, last } = st, lastDx = NaN, lastW = 0;
    for (let i = st.done; i < this.n; i++) {
      const x = xs[i], y = v[i];
      if (!Number.isFinite(y) || !Number.isFinite(x)) {
        sm[i] = y;
        continue;
      }
      let w = 0;
      if (last === last) {
        const dx = Math.max(x - last, 0);
        if (dx !== lastDx) {
          lastDx = dx;
          lastW = Math.pow(alpha, dx / scale);
        }
        w = lastW;
      }
      acc = acc * w + y;
      deb = deb * w + 1;
      last = x;
      sm[i] = acc / deb;
    }
    Object.assign(st, { done: this.n, acc, deb, last });
  }

  ys(alpha, raw) {
    return alpha > 0 && !raw ? this.sm : this.v;
  }
}

// ---- column storage shared with workers ----

/** Whether columns live in memory the page's workers can read: a cross-origin isolated page, unless `?shared=0`. */
export const SHARED = typeof SharedArrayBuffer === "function" && globalThis.crossOriginIsolated === true
  && !/[?&]shared=0(&|$)/.test(globalThis.location?.search ?? "");
const CHUNK = 1 << 22; // floats per shared chunk
const chunks = []; // {id, gen, view (Float64Array over a SharedArrayBuffer), used, live}
const spare = []; // chunks whose columns are all gone, to be reused
let current = null; // the chunk new columns go to
const chunkListeners = new Set();
const finalizer = SHARED ? new FinalizationRegistry(([id, size]) => release(id, size)) : null;

/** n floats for one column: in a shared chunk when SHARED ({view, loc: {chunk, gen, off}}), else a plain array. */
export function columnStore(n) {
  if (!SHARED) return { view: new Float64Array(n), loc: null };
  if (!current || current.used + n > current.view.length) current = takeChunk(n);
  const off = current.used;
  current.used += n;
  current.live += n;
  return { view: current.view.subarray(off, off + n), loc: { chunk: current.id, gen: current.gen, off, size: n } };
}

/** Note that column c holds the storage at `loc`, freed once c is collected. */
export function holdStore(c, loc) {
  c.loc = loc;
  if (loc) finalizer.register(c, [loc.chunk, loc.size]);
}

/** Take shared buffer `buf` (a multiple of 8 bytes, filled elsewhere) as storage of its own: {view, loc}, freed by
 * freeStore(loc). */
export function adoptStore(buf) {
  const view = new Float64Array(buf), ch = { id: chunks.length, gen: 0, view, used: view.length, live: view.length };
  chunks.push(ch);
  for (const fn of chunkListeners) fn(ch.id, buf);
  return { view, loc: { chunk: ch.id, gen: 0, off: 0, size: view.length } };
}

/** Free the storage at `loc` (from columnStore) now. */
export function freeStore(loc) {
  if (loc) release(loc.chunk, loc.size);
}

/** Call fn(id, buffer) for every shared chunk, now and as they are made. */
export function onChunks(fn) {
  for (const ch of chunks) fn(ch.id, ch.view.buffer);
  chunkListeners.add(fn);
}

function takeChunk(n) {
  const i = spare.findIndex((ch) => ch.view.length >= n);
  if (i >= 0) {
    const ch = spare.splice(i, 1)[0];
    (ch.gen += 1), (ch.used = 0), (ch.live = 0);
    return ch;
  }
  const ch = { id: chunks.length, gen: 0, view: new Float64Array(new SharedArrayBuffer(8 * Math.max(CHUNK, n))), used: 0, live: 0 };
  chunks.push(ch);
  for (const fn of chunkListeners) fn(ch.id, ch.view.buffer);
  return ch;
}

function release(id, size) {
  const ch = chunks[id];
  ch.live -= size;
  if (ch.live <= 0 && ch !== current && !spare.includes(ch)) spare.push(ch);
}

/** Extend c's sortedness and extents by n points (steps s, runtimes t) after a last point (ls, lt). */
function track(c, s, t, n, ls, lt) {
  const e = c.ext;
  for (let i = 0; i < n; i++) {
    const x = s[i], y = t[i];
    if (x < ls) c.sorted[0] = false;
    if (y < lt) c.sorted[1] = false;
    ls = x;
    lt = y;
    if (x < e[0]) e[0] = x;
    if (x > e[1]) e[1] = x;
    if (x > 0 && x < e[2]) e[2] = x;
    if (y < e[3]) e[3] = y;
    if (y > e[4]) e[4] = y;
    if (y > 0 && y < e[5]) e[5] = y;
  }
}

const tx = (x, logx) => (!logx ? x : x > 0 ? Math.log10(x) : NaN);

function lowerBound(a, n, x) {
  let lo = 0, hi = n;
  while (lo < hi) {
    const m = (lo + hi) >> 1;
    if (a[m] < x) lo = m + 1;
    else hi = m;
  }
  return lo;
}

function upperBound(a, n, x) {
  let lo = 0, hi = n;
  while (lo < hi) {
    const m = (lo + hi) >> 1;
    if (a[m] <= x) lo = m + 1;
    else hi = m;
  }
  return lo;
}

/** Index range of points that can touch [x0, x1] (transformed x), plus one neighbor per side. */
export function visibleRange(c, xmode, x0, x1, logx) {
  const xs = c.xs(xmode), n = c.n;
  if (!c.sorted[xmode]) return [0, n];
  const a = logx ? 10 ** x0 : x0, b = logx ? 10 ** x1 : x1;
  return [Math.max(0, lowerBound(xs, n, a) - 1), Math.min(n, upperBound(xs, n, b) + 1)];
}

/** Decimate a series over [x0, x1] to ~4 points per pixel column, as (x, y) pairs in `out` (NaN y: a
 * break). Returns {n pairs, ymin, ymax}. */
export function prep(c, xmode, x0, x1, width, flags, alpha, scale, out) {
  const logy = (flags & LOGY) !== 0, raw = (flags & RAW) !== 0, logx = (flags & LOGX) !== 0;
  c.ensureSmooth(alpha, scale, xmode);
  const xs = c.xs(xmode), ys = c.ys(alpha, raw), cap = out.length >> 1;
  const px = width / (x1 - x0);
  const [lo, hi] = visibleRange(c, xmode, x0, x1, logx);
  let ymin = Infinity, ymax = -Infinity, m = 0, broken = true, cur = -Infinity, have = false;
  let f = 0, mn = 0, mx = 0, l = 0;
  const emit = (x, y) => {
    if (m < cap) {
      out[2 * m] = x;
      out[2 * m + 1] = y;
      m++;
    }
  };
  const flush = () => {
    if (!have) return;
    const a = mn < mx ? mn : mx, b = mn < mx ? mx : mn;
    emit(xs[f], ys[f]);
    if (a !== f) emit(xs[a], ys[a]);
    if (b !== a && b !== f) emit(xs[b], ys[b]);
    if (l !== b && l !== a && l !== f) emit(xs[l], ys[l]);
    have = false;
  };
  for (let i = lo; i < hi; i++) {
    const x = xs[i], y = ys[i], xt = tx(x, logx);
    if (!Number.isFinite(y) || (logy && y <= 0) || !Number.isFinite(xt)) {
      flush();
      if (!broken) {
        emit(x, NaN);
        broken = true;
      }
      cur = -Infinity;
      continue;
    }
    broken = false;
    if (xt >= x0 && xt <= x1) {
      if (y < ymin) ymin = y;
      if (y > ymax) ymax = y;
    }
    const b = Math.floor((xt - x0) * px);
    if (b !== cur || !have) {
      flush();
      cur = b;
      f = mn = mx = l = i;
      have = true;
    } else {
      if (y < ys[mn]) mn = i;
      if (y > ys[mx]) mx = i;
      l = i;
    }
  }
  flush();
  return { n: m, ymin, ymax };
}

/** k-th smallest of a[0..n) (reorders a). */
function select(a, n, k) {
  let lo = 0, hi = n - 1;
  while (hi > lo) {
    const pivot = a[(lo + hi) >> 1];
    let i = lo, j = hi;
    while (i <= j) {
      while (a[i] < pivot) i++;
      while (a[j] > pivot) j--;
      if (i <= j) {
        const tmp = a[i];
        a[i] = a[j];
        a[j] = tmp;
        i++;
        j--;
      }
    }
    if (k <= j) hi = j;
    else if (k >= i) lo = i;
    else break;
  }
  return a[k];
}

let scratch = new Float64Array(1 << 16);
function scratchOf(n) {
  if (scratch.length < n) scratch = new Float64Array(Math.max(n, scratch.length * 2));
  return scratch;
}

/** [qlo quantile, qhi quantile] of all finite values across columns with x in [x0, x1]
 * (transformed x); null if there are none. */
export function yrange(cols, xmode, x0, x1, flags, alpha, scale, qlo, qhi) {
  const logy = (flags & LOGY) !== 0, raw = (flags & RAW) !== 0, logx = (flags & LOGX) !== 0;
  let total = 0;
  for (const c of cols) total += c.n;
  const vals = scratchOf(total);
  let n = 0;
  for (const c of cols) {
    c.ensureSmooth(alpha, scale, xmode);
    const xs = c.xs(xmode), ys = c.ys(alpha, raw);
    const [lo, hi] = visibleRange(c, xmode, x0, x1, logx);
    for (let i = lo; i < hi; i++) {
      const xt = tx(xs[i], logx), y = ys[i];
      if (xt >= x0 && xt <= x1 && Number.isFinite(y) && (!logy || y > 0)) vals[n++] = y;
    }
  }
  if (!n) return null;
  const rank = (q) => Math.min(n - 1, Math.round(Math.min(1, Math.max(0, q)) * (n - 1)));
  return [select(vals, n, rank(qlo)), select(vals, n, rank(qhi))];
}

const ciRanks = new Map();

/** Largest lower rank k (1-based) with [x_(k), x_(n-k+1)] covering the median with prob >= 0.95; 1 if none. */
export function medianCiRank(n) {
  let k = ciRanks.get(n);
  if (k === undefined) ciRanks.set(n, (k = ciRankOf(n)));
  return k;
}

function ciRankOf(n) {
  let k = 1, cdf = 0, logc = 0;
  const lnHalfN = n * Math.log(0.5);
  for (let j = 0; j < Math.floor(n / 2); j++) {
    if (j) logc += Math.log(n - j + 1) - Math.log(j);
    cdf += Math.exp(logc + lnHalfN);
    if (1 - 2 * cdf >= 0.95) k = j + 1;
    else break;
  }
  return k;
}

/** Coverage of the order-statistic median interval used for n runs. */
export function medianCiCoverage(n) {
  const k = medianCiRank(n);
  let cdf = 0, logc = 0;
  for (let j = 0; j < k; j++) {
    if (j) logc += Math.log(n - j + 1) - Math.log(j);
    cdf += Math.exp(logc + n * Math.log(0.5));
  }
  return 1 - 2 * cdf;
}

/** Bins of width a power of two at whole multiples of it, covering [x0, x1] with at most `most` + 2 bins:
 * {g0 (first edge), dx (width), bins}. Edges do not move as the range grows. */
export function binGrid(x0, x1, most) {
  const dx = 2 ** Math.ceil(Math.log2((x1 - x0) / most)), b0 = Math.floor(x0 / dx);
  return { g0: b0 * dx, dx, bins: Math.max(1, Math.floor(x1 / dx) + 1 - b0) };
}

/** Group statistics over `bins` x-bins of [x0, x1]: each column is averaged per bin and interpolated
 * across its gaps, then summarized. Returns NSTAT * bins values, stat-major (NaN where n = 0). Columns' bin means are
 * kept in `cache` (a BinCache) for the next call. */
export function agg(cols, xmode, x0, x1, bins, flags, alpha, scale, cache = new BinCache()) {
  return aggGroups([cols], xmode, x0, x1, bins, flags, alpha, scale, cache);
}

/** agg of each list of columns in `groups`, one after another in the result (NSTAT * bins values each). */
export function aggGroups(groups, xmode, x0, x1, bins, flags, alpha, scale, cache = new BinCache()) {
  let R = 0, most = 0;
  for (const g of groups) (R += g.length), (most = Math.max(most, g.length));
  const out = new Float64Array(groups.length * NSTAT * bins).fill(NaN), iqm = (flags & IQM) !== 0;
  const m = cache.get({ xmode, x0, x1, bins, flags: flags & (RAW | LOGX), alpha, scale }, R), buf = scratchOf(most * bins);
  finiteOf(most);
  for (let gi = 0; gi < groups.length; gi++) {
    const n = groups[gi].length, at = m.slots(groups[gi]);
    gatherBins(m.v, at, n, bins, buf);
    for (let b = 0; b < bins; b++) binStats(buf, b * n, n, out, gi * NSTAT * bins, bins, b, iqm);
  }
  return out;
}

/** Each column's per-bin means for binning p ({xmode, x0, x1, bins, flags, alpha, scale}), NaN where
 * it has none: cols.length * p.bins values, column-major; binnings are kept in `cache` as agg's. */
export function binRows(cols, p, cache = new BinCache()) {
  const m = cache.get(p, cols.length), at = m.slots(cols), out = new Float64Array(cols.length * p.bins);
  for (let r = 0; r < cols.length; r++) out.set(m.v.subarray(at[r] * p.bins, (at[r] + 1) * p.bins), r * p.bins);
  return out;
}

/** The bin means of the n columns in slots `at` of v (bins per slot) into buf, bin-major: bin b's at [b * n, (b + 1)
 * * n). Block by block, so the rows read stay in cache. */
function gatherBins(v, at, n, bins, buf) {
  for (let r0 = 0; r0 < n; r0 += GATHER_BLOCK) {
    const r1 = Math.min(n, r0 + GATHER_BLOCK);
    for (let b = 0; b < bins; b++) for (let r = r0, o = b * n; r < r1; r++) buf[o + r] = v[at[r] * bins + b];
  }
}

const BINNINGS = 4; // binnings a BinCache keeps: smoothed and raw, at two zooms
const GATHER_BLOCK = 64; // columns agg gathers at a time

/** Columns' per-bin means for one binning ({xmode, x0, x1, bins, flags, alpha, scale}), one slot per column (bin b of
 * slot k at v[k * bins + b]); a column's slot is reused while it has the same points. */
class Binning {
  constructor(p, cap) {
    this.p = p;
    this.cap = cap;
    this.v = new Float64Array(p.bins * cap);
    this.at = new Map(); // column -> [slot, points binned]
    const at = () => new Float64Array(p.bins);
    this.acc = { sum: at(), cnt: at(), pos: at(), neg: at() };
  }

  /** Slots of `cols`, binning the columns not yet binned with their current points. */
  slots(cols) {
    const out = slotScratch(cols.length);
    for (let r = 0; r < cols.length; r++) {
      const c = cols[r], e = this.at.get(c);
      out[r] = e && e[1] === c.n ? e[0] : this.bin(c, e ? e[0] : this.at.size);
    }
    return out;
  }

  /** Bin column c into slot k. */
  bin(c, k) {
    const p = this.p;
    if (k >= this.cap) this.grow(k + 1);
    const row = this.v.subarray(k * p.bins, (k + 1) * p.bins).fill(NaN);
    c.ensureSmooth(p.alpha, p.scale, p.xmode);
    binColumn(c, p.xmode, c.ys(p.alpha, (p.flags & RAW) !== 0), p.x0, p.x1, p.bins, (p.flags & LOGX) !== 0, this.acc, row);
    this.at.set(c, [k, c.n]);
    return k;
  }

  grow(need) {
    const cap = Math.max(need, 2 * this.cap), v = new Float64Array(this.p.bins * cap);
    v.set(this.v);
    (this.v = v), (this.cap = cap);
  }
}

/** One chart's recent binnings, newest first. */
export class BinCache {
  constructor() {
    this.binnings = [];
    this.most = 0; // most columns one call has binned
  }

  /** The kept binning p, or a new one; one holding many more columns than the chart has drawn (columns since
   * replaced) starts over. */
  get(p, R) {
    const all = this.binnings;
    this.most = Math.max(this.most, R);
    let i = all.findIndex((m) => sameBinning(m.p, p));
    if (i >= 0 && all[i].at.size > 2 * Math.max(R, this.most) + 1024) all.splice(i, 1), (i = -1);
    const m = i >= 0 ? all.splice(i, 1)[0] : new Binning(p, R);
    all.unshift(m);
    if (all.length > BINNINGS) all.pop();
    return m;
  }
}

const sameBinning = (a, b) => a.xmode === b.xmode && a.x0 === b.x0 && a.x1 === b.x1 && a.bins === b.bins
  && a.flags === b.flags && a.alpha === b.alpha && a.scale === b.scale;

let slotBuf = new Int32Array(1024);
const slotScratch = (n) => (slotBuf.length < n ? (slotBuf = new Int32Array(2 * n)) : slotBuf).subarray(0, n);

/** Per-bin means of column c's values in [x0, x1] into row, each point weighted by the rows it stands for,
 * interpolated across empty bins between full ones. A bin's mean is of its finite values; with none, it is infinite
 * (NaN with both signs). */
function binColumn(c, xmode, ys, x0, x1, bins, logx, acc, row) {
  if (!c.sorted[xmode]) return binUnsorted(c, xmode, ys, x0, x1, bins, logx, acc, row);
  const xs = c.xs(xmode), ws = c.w, [lo, hi] = visibleRange(c, xmode, x0, x1, logx), per = bins / (x1 - x0);
  const run = binRun;
  (run.b = -1), (run.prev = -1), (run.sum = 0), (run.cnt = 0), (run.pos = 0), (run.neg = 0);
  for (let i = lo; i < hi; i++) {
    const x = tx(xs[i], logx), y = ys[i];
    if (!(x >= x0 && x <= x1) || y !== y) continue;
    const b = Math.min(bins - 1, Math.floor((x - x0) * per)), k = ws ? ws[i] : 1;
    if (b !== run.b) closeBin(run, row), (run.b = b);
    if (y === Infinity) run.pos = 1;
    else if (y === -Infinity) run.neg = 1;
    else (run.sum += y * k), (run.cnt += k);
  }
  closeBin(run, row);
}

/** The bin binColumn is filling (b, with its finite sum and count and whether it holds +inf or -inf) and the last
 * full bin before it (prev, of mean pv). */
const binRun = { b: -1, sum: 0, cnt: 0, pos: 0, neg: 0, prev: -1, pv: 0 };

/** Write the mean of binRun's bin into row, interpolating across the empty bins since the previous full one. */
function closeBin(run, row) {
  const b = run.b, v = run.cnt ? run.sum / run.cnt : run.pos || run.neg ? (run.pos ? Infinity : 0) + (run.neg ? -Infinity : 0) : undefined;
  (run.sum = 0), (run.cnt = 0), (run.pos = 0), (run.neg = 0);
  if (v === undefined) return;
  row[b] = v;
  for (let k = run.prev + 1, prev = run.prev, pv = run.pv; prev >= 0 && k < b; k++) {
    const w = (k - prev) / (b - prev);
    row[k] = pv * (1 - w) + v * w;
  }
  (run.prev = b), (run.pv = v);
}

// ---- bucket arrays: some runs' buckets of one metric at one level (buckets.py) ----

export const BLOCK = 256; // buckets of a block (buckets.BLOCK)
const BUCKETS_MAGIC = 0x31424b54; // "TKB1"
const SOFF_SCALE = 65536; // step offsets, in fractions of a bucket (buckets.SOFF_SCALE)
const pad8 = (n) => Math.ceil(n / 8) * 8;

/** Views of the bucket array encoded at byte `off` of `buf`: {level, base (its first block), runs, count, first (run
 * i's buckets are [first[i], first[i + 1])), seq (the rows each run's buckets hold), offset (bucket - base * BLOCK),
 * soff (mean step offset times SOFF_SCALE, rounded down), mean, tmean, n, names: [byte offset, length] of its run
 * paths, bytes}. */
export function bucketViews(buf, off = 0) {
  const h = new Uint32Array(buf, off, 8);
  if (h[0] !== BUCKETS_MAGIC) throw new Error("bad bucket array");
  const level = h[1] | 0, base = (h[3] | 0) * 4294967296 + h[2], runs = h[4], count = h[5];
  let at = off + 32 + pad8(h[6]);
  const first = new Uint32Array(buf, at, runs + 1);
  at += pad8(4 * (runs + 1));
  const seq = new Uint32Array(buf, at, runs);
  at += pad8(4 * runs);
  const offset = new Uint16Array(buf, at, count), soff = new Uint16Array(buf, at + pad8(2 * count), count);
  at += 2 * pad8(2 * count);
  return { level, base, runs, count, first, seq, offset, soff, mean: new Float32Array(buf, at, count),
           tmean: new Float32Array(buf, at + 4 * count, count), n: new Uint32Array(buf, at + 8 * count, count),
           names: [off + 32, h[6]], bytes: at + 12 * count - off };
}

/** The run paths of bucket array `v` (bucketViews of `buf`). */
export function bucketPaths(buf, v) {
  return v.runs ? new TextDecoder().decode(new Uint8Array(buf, v.names[0], v.names[1]).slice()).split("\0") : [];
}

/** Mean step of bucket q of bucket array `v`. */
export const bucketStep = (v, q) => (v.base * BLOCK + v.offset[q] + (v.soff[q] + 0.5) / SOFF_SCALE) * 2 ** v.level;

/** Steps [lo, hi) of the block a part ({v, row}) holds. */
const blockSteps = ({ v }) => [v.base * BLOCK * 2 ** v.level, (v.base + 1) * BLOCK * 2 ** v.level];

/** A run's column from its buckets in `parts` ({v (bucketViews), row}, of any levels): each level's buckets where its
 * blocks lie and no finer level's do, as points at their mean steps standing for their counts, in step order; then the
 * rows of `tail` ({s, v, t, q (sequence numbers), n}) the parts do not hold, in buckets as wide as the finest level
 * holding their place (`level` where none does), the first merged into the column's last point when in its bucket.
 * A bucket's mean is of its finite values, of its infinities when it has none. In memory shared with workers when
 * `shared`. */
export function buildColumn(parts, tail, level, shared = true) {
  let cap = tail.n;
  for (const p of parts) cap += p.v.first[p.row + 1] - p.v.first[p.row];
  const { view: all, loc } = shared ? columnStore(4 * cap) : { view: new Float64Array(4 * cap), loc: null };
  const col = { s: all.subarray(0, cap), v: all.subarray(cap, 2 * cap), t: all.subarray(2 * cap, 3 * cap), w: all.subarray(3 * cap), n: 0, lv: 0, b: 0 };
  if (!tail.n && parts.every((p) => p.v.level === parts[0].v.level)) oneLevel(parts, col);
  else {
    const layers = layersOf(parts);
    emitBuckets(layers, col);
    emitTail(layers, tail, level ?? (layers.length ? layers[layers.length - 1].level : 0), col);
  }
  const c = Col.adopt(col.s, col.v, col.t, col.n, col.w);
  if (shared) holdStore(c, loc && { ...loc, cap });
  return c;
}

const arraySteps = new WeakMap(); // bucket array views -> {s (each bucket's mean step), rising (whether each run's runtimes rise)}

/** Run `row` of bucket array `v` as a column viewing the array, its steps computed once per array. */
export function runColumn(v, row) {
  let st = arraySteps.get(v);
  if (!st) arraySteps.set(v, (st = stepsOf(v)));
  const a = v.first[row], e = v.first[row + 1];
  return Col.view(st.s.subarray(a, e), v.mean.subarray(a, e), v.tmean.subarray(a, e), e - a, [true, st.rising[row] === 1], v.n.subarray(a, e));
}

function stepsOf(v) {
  const s = new Float64Array(v.count), rising = new Uint8Array(v.runs), w = 2 ** v.level, base = v.base * BLOCK;
  for (let q = 0; q < v.count; q++) s[q] = (base + v.offset[q] + (v.soff[q] + 0.5) / SOFF_SCALE) * w;
  for (let i = 0; i < v.runs; i++) {
    let q = v.first[i] + 1;
    while (q < v.first[i + 1] && !(v.tmean[q] < v.tmean[q - 1])) q++;
    rising[i] = q >= v.first[i + 1] ? 1 : 0;
  }
  return { s, rising };
}

/** emitBuckets of parts of one level: each part's buckets, parts by block. */
function oneLevel(parts, col) {
  if (parts.length > 1) parts = [...parts].sort((a, b) => a.v.base - b.v.base);
  let n = 0;
  for (const { v, row } of parts) {
    const w = 2 ** v.level, base = v.base * BLOCK, { offset, soff, mean, tmean } = v, cnt = v.n;
    for (let q = v.first[row], end = v.first[row + 1]; q < end; q++) {
      col.s[n] = (base + offset[q] + (soff[q] + 0.5) / SOFF_SCALE) * w;
      (col.v[n] = mean[q]), (col.t[n] = tmean[q]), (col.w[n] = cnt[q]), n++;
    }
  }
  col.n = n;
}

/** `parts` by level, finest first: [{level, parts (by step), ranges ([lo, hi) steps of their blocks)}]. */
function layersOf(parts) {
  const by = new Map();
  for (const p of parts) {
    if (!by.has(p.v.level)) by.set(p.v.level, []);
    by.get(p.v.level).push(p);
  }
  return [...by].sort((a, b) => a[0] - b[0]).map(([level, ps]) => {
    ps.sort((a, b) => a.v.base - b.v.base);
    return { level, parts: ps, ranges: ps.map(blockSteps) };
  });
}

/** Whether step x lies in one of `ranges` ([lo, hi), by lo). */
function inRanges(ranges, x) {
  let lo = 0, hi = ranges.length;
  while (lo < hi) {
    const m = (lo + hi) >> 1;
    if (ranges[m][1] <= x) lo = m + 1;
    else hi = m;
  }
  return lo < ranges.length && ranges[lo][0] <= x;
}

/** The layers' buckets each finer layer leaves them, into col in step order (sorted when there are several layers). */
function emitBuckets(layers, col) {
  for (let k = 0; k < layers.length; k++) {
    const finer = layers.slice(0, k);
    for (const { v, row } of layers[k].parts) {
      for (let q = v.first[row]; q < v.first[row + 1]; q++) {
        const x = bucketStep(v, q);
        if (finer.some((f) => inRanges(f.ranges, x))) continue;
        (col.s[col.n] = x), (col.v[col.n] = v.mean[q]), (col.t[col.n] = v.tmean[q]), (col.w[col.n] = v.n[q]), col.n++;
      }
    }
  }
  if (layers.length > 1) sortPoints(col);
  if (col.n) (col.lv = levelAt(layers, col.s[col.n - 1])), (col.b = Math.floor(col.s[col.n - 1] / 2 ** col.lv));
}

function sortPoints(col) {
  const order = Array.from({ length: col.n }, (_, i) => i).sort((a, b) => col.s[a] - col.s[b]);
  for (const k of ["s", "v", "t", "w"]) {
    const a = col[k].slice(0, col.n);
    order.forEach((j, i) => (col[k][i] = a[j]));
  }
}

/** The level of the finest layer whose blocks hold step x, or null. */
function levelAt(layers, x) {
  for (const L of layers) if (inRanges(L.ranges, x)) return L.level;
  return null;
}

/** The sequence number from which the part holding step x lacks rows (0 when none holds it). */
function heldUpTo(layers, x) {
  for (const L of layers) {
    let lo = 0, hi = L.ranges.length;
    while (lo < hi) {
      const m = (lo + hi) >> 1;
      if (L.ranges[m][1] <= x) lo = m + 1;
      else hi = m;
    }
    if (lo < L.ranges.length && L.ranges[lo][0] <= x) return L.parts[lo].v.seq[L.parts[lo].row];
  }
  return 0;
}

const merged = new Float64Array(8); // one bucket's sums of value, step and runtime and its count: of its finite values, then of the rest

/** The tail rows the parts lack, bucketed and appended to col. */
function emitTail(layers, tail, level, col) {
  let cur = null;
  merged.fill(0);
  const close = () => {
    const o = merged[3] > 0 ? 0 : 4, k = merged[o + 3];
    if (k > 0) (col.s[col.n] = merged[o + 1] / k), (col.v[col.n] = merged[o] / k), (col.t[col.n] = merged[o + 2] / k), (col.w[col.n] = k), col.n++;
    merged.fill(0);
  };
  for (let i = 0; i < tail.n; i++) {
    const y = tail.v[i], x = tail.s[i];
    if (y !== y || tail.q[i] < heldUpTo(layers, x)) continue;
    const lv = levelAt(layers, x) ?? level, b = Math.floor(x / 2 ** lv), key = `${lv}|${b}`;
    if (key !== cur) {
      if (cur !== null) close();
      else if (col.n && col.lv === lv && col.b === b) reopen(col);
      cur = key;
    }
    const o = Number.isFinite(y) ? 0 : 4;
    (merged[o] += y), (merged[o + 1] += x), (merged[o + 2] += tail.t[i]), (merged[o + 3] += 1);
  }
  if (cur !== null) close();
}

/** Take the column's last point back into `merged`, to merge rows into its bucket. */
function reopen(col) {
  const i = --col.n, k = col.w[i], o = Number.isFinite(col.v[i]) ? 0 : 4;
  (merged[o] = col.v[i] * k), (merged[o + 1] = col.s[i] * k), (merged[o + 2] = col.t[i] * k), (merged[o + 3] = k);
}

/** binColumn of a column whose x is not sorted: sums per bin first. */
function binUnsorted(c, xmode, ys, x0, x1, bins, logx, acc, row) {
  const { sum, cnt, pos, neg } = acc;
  const xs = c.xs(xmode), ws = c.w, per = bins / (x1 - x0);
  sum.fill(0), cnt.fill(0), pos.fill(0), neg.fill(0);
  for (let i = 0; i < c.n; i++) {
    const x = tx(xs[i], logx), y = ys[i];
    if (!(x >= x0 && x <= x1) || y !== y) continue;
    const b = Math.min(bins - 1, Math.floor((x - x0) * per)), k = ws ? ws[i] : 1;
    if (y === Infinity) pos[b] = 1;
    else if (y === -Infinity) neg[b] = 1;
    else (sum[b] += y * k), (cnt[b] += k);
  }
  let prev = -1, pv = 0;
  for (let b = 0; b < bins; b++) {
    const v = binMean(acc, b);
    if (v === undefined) continue;
    row[b] = v;
    for (let k = prev + 1; prev >= 0 && k < b; k++) {
      const w = (k - prev) / (b - prev);
      row[k] = pv * (1 - w) + v * w;
    }
    prev = b;
    pv = v;
  }
}

/** Bin b's mean from binColumn's sums: of its finite values, else infinite (NaN with both signs); undefined when
 * the bin is empty. */
function binMean({ sum, cnt, pos, neg }, b) {
  if (cnt[b]) return sum[b] / cnt[b];
  if (pos[b] || neg[b]) return (pos[b] ? Infinity : 0) + (neg[b] ? -Infinity : 0);
  return undefined;
}

/** Counts of a bin's values: finite ones (copied to `fin`), their sum and range, and infinities. */
const part = { m: 0, sum: 0, lo: 0, hi: 0, neg: 0, pos: 0 };
let fin = new Float64Array(64); // the finite values of the bin binStats summarizes

function finiteOf(n) {
  if (fin.length < n) fin = new Float64Array(n);
}

/** Copy the finite values of a[off, off + n) to the front of `fin`, counting them and the infinities into `part`;
 * NaN is no value. */
function partition(a, off, n) {
  let m = 0, sum = 0, lo = Infinity, hi = -Infinity, neg = 0, pos = 0;
  for (let i = off; i < off + n; i++) {
    const v = a[i];
    if (v === Infinity) pos++;
    else if (v === -Infinity) neg++;
    else if (v === v) {
      fin[m++] = v;
      sum += v;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
  }
  (part.m = m), (part.sum = sum), (part.lo = lo), (part.hi = hi), (part.neg = neg), (part.pos = pos);
}

/** STATS of the values a[off, off + n) (NaN: no value) into bin b of the group at `o` of `out`, which holds NaN; the
 * IQM's only when `iqm`. Infinities count as values: the mean is infinite (NaN with both signs), the spread NaN, and
 * order statistics see them at the ends. */
function binStats(a, off, n0, out, o, bins, b, iqm) {
  partition(a, off, n0);
  const { m, neg, pos } = part, n = neg + m + pos;
  out[o + S.n * bins + b] = n;
  if (!n) return;
  moments(out, o, bins, b, n);
  const k = medianCiRank(n), h50 = (n - 1) * 0.5, h25 = (n - 1) * 0.25, h75 = (n - 1) * 0.75, g = Math.floor(n / 4);
  if (m <= SMALL_SORT) insertionSort(fin, m), (sortedVals = fin);
  else (sortedVals = null), resolveRanks(fin, wantRanks(n, [k - 1, n - k, h50, h50 + 1, h25, h25 + 1, h75, h75 + 1, g, n - g - 1]));
  out[o + S.median * bins + b] = quantileOf(h50, n);
  out[o + S.q25 * bins + b] = quantileOf(h25, n);
  out[o + S.q75 * bins + b] = quantileOf(h75, n);
  out[o + S.medlo * bins + b] = rankValue(k - 1);
  out[o + S.medhi * bins + b] = rankValue(n - k);
  if (iqm) iqmStats(fin, g, rankValue(g), rankValue(n - g - 1), out, o, bins, b);
}

/** Mean, std, min and max of the bin `part` describes (n values) into `out`. */
function moments(out, o, bins, b, n) {
  const { m, sum, lo, hi, neg, pos } = part;
  const mean = (sum + (pos ? Infinity : 0) + (neg ? -Infinity : 0)) / n;
  let v2 = 0;
  for (let i = 0; i < m; i++) v2 += (fin[i] - mean) ** 2;
  out[o + S.mean * bins + b] = mean;
  out[o + S.std * bins + b] = n === 1 ? 0 : neg || pos ? NaN : Math.sqrt(v2 / (n - 1));
  out[o + S.min * bins + b] = neg ? -Infinity : lo;
  out[o + S.max * bins + b] = pos ? Infinity : hi;
}

/** The values at ranks want[0, count) of the bin `part` describes: the infinities at either end, the finite values
 * (the first part.m of `s`) by exact order statistics. */
function resolveRanks(s, count) {
  const { m, lo, hi, neg } = part;
  let w0 = 0, w1 = count;
  while (w0 < count && want[w0] < neg) got[w0++] = -Infinity;
  while (w1 > w0 && want[w1 - 1] >= neg + m) got[--w1] = Infinity;
  if (w1 > w0) orderStats(s, m, lo, hi, neg, w0, w1, 0);
}

/** Interquartile mean of the bin `part` describes (the mean of ranks [g, n - g), g = floor(n / 4)), whose values at
 * ranks g and n - g - 1 are lo and hi; with Yuen's standard error (from the winsorized variance) and the count h
 * it keeps, whose t quantile on h - 1 degrees of freedom gives its 95% CI. Infinite (NaN with both signs, error NaN)
 * when the kept ranks reach an infinity. */
function iqmStats(s, g, lo, hi, out, o, bins, b) {
  const { m, neg, pos } = part, n = neg + m + pos, h = n - 2 * g;
  out[o + S.iqmh * bins + b] = h;
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return infiniteIqm(lo, hi, h, out, o, bins, b);
  let le = neg, mid = 0, nmid = 0, wsum = neg * lo + pos * hi;
  for (let i = 0; i < m; i++) {
    const v = s[i];
    if (v <= lo) le++;
    else if (v < hi) (mid += v), nmid++;
    wsum += v < lo ? lo : v > hi ? hi : v;
  }
  const nlo = lo === hi ? h : le - g; // copies of lo, then of hi, fill what is left of the window
  const iqm = lo === hi ? lo : (mid + lo * nlo + hi * (h - nmid - nlo)) / h;
  const wmean = wsum / n;
  let w2 = neg * (lo - wmean) ** 2 + pos * (hi - wmean) ** 2;
  for (let i = 0; i < m; i++) w2 += ((s[i] < lo ? lo : s[i] > hi ? hi : s[i]) - wmean) ** 2;
  out[o + S.iqm * bins + b] = iqm;
  out[o + S.iqmse * bins + b] = h > 1 ? Math.sqrt(w2 / (h * (h - 1))) : 0;
}

/** The IQM of a kept window from lo to hi that reaches an infinity: that infinity (NaN when it reaches both), error
 * NaN. */
function infiniteIqm(lo, hi, h, out, o, bins, b) {
  out[o + S.iqm * bins + b] = lo === -Infinity ? (hi === Infinity ? NaN : -Infinity) : hi;
  out[o + S.iqmse * bins + b] = h > 1 ? NaN : 0;
}

const want = new Int32Array(10), got = new Float64Array(10); // the ranks orderStats resolves, and their values

/** Fill `want` with the distinct ranks (floors of `hs`, below m), ascending; their count. */
function wantRanks(m, hs) {
  let n = 0;
  for (const h of hs) {
    const r = Math.min(m - 1, Math.floor(h));
    let i = n;
    while (i > 0 && want[i - 1] > r) i--;
    if (i > 0 && want[i - 1] === r) continue;
    want.copyWithin(i + 1, i, n);
    want[i] = r;
    n++;
  }
  return n;
}

let sortedVals = null; // when set, the bin's finite values, sorted, which ranks read directly

function rankValue(r) {
  if (sortedVals) return r < part.neg ? -Infinity : r >= part.neg + part.m ? Infinity : sortedVals[r - part.neg];
  let i = 0;
  while (want[i] !== r) i++;
  return got[i];
}

/** The interpolated quantile at fractional rank h of m sorted values, from the resolved ranks. */
function quantileOf(h, m) {
  const i = Math.floor(h), f = h - i;
  return f && i + 1 < m ? rankValue(i) * (1 - f) + rankValue(i + 1) * f : rankValue(i);
}

const HIST = 1024; // most buckets per histogram pass of orderStats: about a quarter of the values
const SMALL_SORT = 64; // at most this many values are sorted outright
const hist = new Int32Array(HIST), slotOf = new Int32Array(HIST).fill(-1);
const gathered = [];
let sortBuf = new Float64Array(SMALL_SORT);

/** The values at the ranks want[w0, w1) (ascending, of the whole set) into got[w0, w1): a holds n values within
 * [lo, hi], of ranks base onward, in any order. Exact: a histogram over [lo, hi] finds the buckets holding the
 * wanted ranks, and only their values are gathered and resolved, by sorting when few. */
function orderStats(a, n, lo, hi, base, w0, w1, depth) {
  if (lo === hi) return got.fill(lo, w0, w1);
  const H = Math.min(HIST, 2 ** Math.ceil(Math.log2(n / 4))), scale = H / (hi - lo);
  if (n <= SMALL_SORT || depth > 6 || !(scale < Infinity)) return sortedRanks(a, n, base, w0, w1);
  histogram(a, n, lo, scale, H);
  const groups = wantedBuckets(base, w0, w1, H);
  const { buf, start, end } = gather(a, n, lo, scale, groups, depth, H);
  for (let g = 0; g < groups.k.length; g++) {
    const sub = buf.subarray(start[g], end[g]);
    let l = Infinity, h = -Infinity;
    for (let i = 0; i < sub.length; i++) {
      if (sub[i] < l) l = sub[i];
      if (sub[i] > h) h = sub[i];
    }
    orderStats(sub, sub.length, l, h, groups.first[g], groups.w[2 * g], groups.w[2 * g + 1], depth + 1);
  }
}

function histogram(a, n, lo, scale, H) {
  hist.fill(0, 0, H);
  for (let i = 0; i < n; i++) {
    const k = ((a[i] - lo) * scale) | 0;
    hist[k < H ? k : H - 1]++;
  }
}

/** The histogram buckets holding want[w0, w1): {k (bucket), first (its first rank), w (want ranges, in pairs)}. */
function wantedBuckets(base, w0, w1, H) {
  const groups = { k: [], first: [], w: [] };
  let cum = base, w = w0;
  for (let k = 0; k < H && w < w1; k++) {
    const from = w;
    while (w < w1 && want[w] < cum + hist[k]) w++;
    if (w > from) groups.k.push(k), groups.first.push(cum), groups.w.push(from, w);
    cum += hist[k];
  }
  return groups;
}

/** The values of a in the wanted buckets, each bucket's together: {buf, start, end} per bucket. */
function gather(a, n, lo, scale, groups, depth, H) {
  let total = 0;
  const start = groups.k.map((k, g) => ((slotOf[k] = g), (total += hist[k]), total - hist[k])), end = start.slice();
  const buf = (gathered[depth] = gathered[depth]?.length >= total ? gathered[depth] : new Float64Array(Math.max(total, 64)));
  for (let i = 0; i < n; i++) {
    const k = ((a[i] - lo) * scale) | 0, g = slotOf[k < H ? k : H - 1];
    if (g >= 0) buf[end[g]++] = a[i];
  }
  for (const k of groups.k) slotOf[k] = -1;
  return { buf, start, end };
}

/** Sort a[0, m) ascending in place. */
function insertionSort(a, m) {
  for (let i = 1; i < m; i++) {
    const v = a[i];
    let j = i - 1;
    while (j >= 0 && a[j] > v) (a[j + 1] = a[j]), j--;
    a[j + 1] = v;
  }
}

function sortedRanks(a, n, base, w0, w1) {
  if (sortBuf.length < n) sortBuf = new Float64Array(n);
  const t = sortBuf.subarray(0, n);
  t.set(a.subarray(0, n));
  t.sort();
  for (let w = w0; w < w1; w++) got[w] = t[want[w] - base];
}

/** Point nearest to data-space x: {i, x, y (smoothed), raw}; null if empty. */
export function nearest(c, xmode, x, alpha, scale) {
  c.ensureSmooth(alpha, scale, xmode);
  const xs = c.xs(xmode), n = c.n;
  if (!n) return null;
  let i;
  if (c.sorted[xmode]) {
    const j = lowerBound(xs, n, x);
    i = j === 0 ? 0 : j === n || x - xs[j - 1] <= xs[j] - x ? j - 1 : j;
  } else {
    i = 0;
    for (let j = 1; j < n; j++) if (Math.abs(xs[j] - x) < Math.abs(xs[i] - x)) i = j;
  }
  return { i, x: xs[i], y: c.ys(alpha, false)[i], raw: c.v[i] };
}

const CRC = (() => {
  const t = Array.from({ length: 8 }, () => new Uint32Array(256));
  for (let i = 0; i < 256; i++) {
    let c = i;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[0][i] = c >>> 0;
  }
  for (let k = 1; k < 8; k++) {
    for (let i = 0; i < 256; i++) {
      const p = t[k - 1][i];
      t[k][i] = ((p >>> 8) ^ t[0][p & 0xff]) >>> 0;
    }
  }
  return t;
})();

/** CRC-32 (IEEE, as zlib.crc32) of an ArrayBuffer or Uint8Array. */
export function crc32(data) {
  const buf = data instanceof Uint8Array ? data : new Uint8Array(data);
  const [t0, t1, t2, t3, t4, t5, t6, t7] = CRC;
  const n = buf.length, n8 = n - (n % 8);
  let c = ~0, i = 0;
  for (; i < n8; i += 8) {
    const a = (c ^ (buf[i] | (buf[i + 1] << 8) | (buf[i + 2] << 16) | (buf[i + 3] << 24))) >>> 0;
    const b = (buf[i + 4] | (buf[i + 5] << 8) | (buf[i + 6] << 16) | (buf[i + 7] << 24)) >>> 0;
    c = t7[a & 0xff] ^ t6[(a >>> 8) & 0xff] ^ t5[(a >>> 16) & 0xff] ^ t4[a >>> 24] ^
        t3[b & 0xff] ^ t2[(b >>> 8) & 0xff] ^ t1[(b >>> 16) & 0xff] ^ t0[b >>> 24];
  }
  for (; i < n; i++) c = t0[(c ^ buf[i]) & 0xff] ^ (c >>> 8);
  return ~c >>> 0;
}
