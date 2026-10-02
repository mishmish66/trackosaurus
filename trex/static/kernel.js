// Numeric kernel for the UI: resident metric columns, CRC-32, time-weighted EMA smoothing,
// per-pixel decimation, group aggregation, and axis quantiles. Pure JS on typed arrays.

/** `flags` bits accepted by prep, agg and yrange; IQM (agg only) adds the interquartile mean. */
export const LOGY = 1, RAW = 2, LOGX = 4, IQM = 8;

/** Per-bin statistics returned by agg, in output order (stat-major). */
export const STATS = ["mean", "std", "n", "min", "max", "median", "q25", "q75", "medlo", "medhi", "iqm", "iqmse", "iqmh"];
export const NSTAT = STATS.length;

/** One metric of one run: points (step, value, runtime) in sequence order. */
export class Col {
  constructor() {
    this.n = 0;
    this.s = new Float64Array(64);
    this.v = new Float64Array(64);
    this.t = new Float64Array(64);
    this.sorted = [true, true];
    this.ext = [Infinity, -Infinity, Infinity, Infinity, -Infinity, Infinity]; // min, max, min positive: steps then runtimes
    this.sm = null;
    this.smKey = null;
    this.smState = null;
  }

  get len() {
    return this.n;
  }

  /** A column holding the first n points of s, v, t (Float64Arrays, taken over without copying). */
  static adopt(s, v, t, n) {
    const c = Object.create(Col.prototype);
    Object.assign(c, { n, s, v, t, sorted: [true, true], ext: [Infinity, -Infinity, Infinity, Infinity, -Infinity, Infinity],
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
    const k = xmode === 0 ? 0 : 3;
    return [this.ext[k], this.ext[k + 1], this.ext[k + 2]];
  }

  xs(xmode) {
    return xmode === 0 ? this.s : this.t;
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
function visibleRange(c, xmode, x0, x1, logx) {
  const xs = c.xs(xmode), n = c.n;
  if (!c.sorted[xmode === 0 ? 0 : 1]) return [0, n];
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

function quantile(s, n, q) {
  const h = (n - 1) * q, i = Math.floor(h), f = h - i;
  return i + 1 < n ? s[i] * (1 - f) + s[i + 1] * f : s[i];
}

/** Bins of width a power of two at whole multiples of it, covering [x0, x1] with at most `most` + 2 bins:
 * {g0 (first edge), dx (width), bins}. Edges do not move as the range grows. */
export function binGrid(x0, x1, most) {
  const dx = 2 ** Math.ceil(Math.log2((x1 - x0) / most)), b0 = Math.floor(x0 / dx);
  return { g0: b0 * dx, dx, bins: Math.max(1, Math.floor(x1 / dx) + 1 - b0) };
}

/** Group statistics over `bins` x-bins of [x0, x1]: each column is averaged per bin and interpolated
 * across its gaps, then summarized. Returns NSTAT * bins values, stat-major (NaN where n = 0). */
export function agg(cols, xmode, x0, x1, bins, flags, alpha, scale) {
  const raw = (flags & RAW) !== 0, logx = (flags & LOGX) !== 0;
  const R = cols.length, out = new Float64Array(NSTAT * bins);
  const buf = scratchOf(R * bins + 2 * bins);
  const vals = buf.subarray(0, R * bins).fill(NaN); // bin-major: bin b's values are vals[b * R, (b + 1) * R)
  const acc = { sum: buf.subarray(R * bins, R * bins + bins), cnt: buf.subarray(R * bins + bins, R * bins + 2 * bins) };
  for (let r = 0; r < R; r++) {
    cols[r].ensureSmooth(alpha, scale, xmode);
    binColumn(cols[r], xmode, cols[r].ys(alpha, raw), x0, x1, bins, logx, acc, vals, r, R);
  }
  const iqm = (flags & IQM) !== 0;
  for (let b = 0; b < bins; b++) binStats(vals.subarray(b * R, (b + 1) * R), out, bins, b, iqm);
  return out;
}

/** Per-bin means of column c's values in [x0, x1] into vals[b * R + r], interpolated across empty bins between full
 * ones. */
function binColumn(c, xmode, ys, x0, x1, bins, logx, { sum, cnt }, vals, r, R) {
  const xs = c.xs(xmode), [lo, hi] = visibleRange(c, xmode, x0, x1, logx), per = bins / (x1 - x0);
  sum.fill(0);
  cnt.fill(0);
  for (let i = lo; i < hi; i++) {
    const x = tx(xs[i], logx), y = ys[i];
    if (!(x >= x0 && x <= x1) || !Number.isFinite(y)) continue;
    const b = Math.min(bins - 1, Math.floor((x - x0) * per));
    sum[b] += y;
    cnt[b] += 1;
  }
  let prev = -1, pv = 0;
  for (let b = 0; b < bins; b++) {
    if (!cnt[b]) continue;
    const v = sum[b] / cnt[b];
    vals[b * R + r] = v;
    for (let k = prev + 1; prev >= 0 && k < b; k++) {
      const w = (k - prev) / (b - prev);
      vals[k * R + r] = pv * (1 - w) + v * w;
    }
    prev = b;
    pv = v;
  }
}

/** STATS of the values in `s` (NaN: no value; reordered) into bin b of `out`; the IQM's only when `iqm`. */
function binStats(s, out, bins, b, iqm) {
  let m = 0, sum = 0, lo = Infinity, hi = -Infinity;
  for (let i = 0; i < s.length; i++) {
    const v = s[i];
    if (v !== v) continue;
    s[m++] = v;
    sum += v;
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  out[2 * bins + b] = m;
  if (!m || !iqm) for (let j = 10; j < NSTAT; j++) out[j * bins + b] = NaN;
  if (!m) {
    for (const j of [0, 1, 3, 4, 5, 6, 7, 8, 9]) out[j * bins + b] = NaN;
    return;
  }
  const mean = sum / m;
  let v2 = 0;
  for (let i = 0; i < m; i++) v2 += (s[i] - mean) ** 2;
  out[b] = mean;
  out[bins + b] = m > 1 ? Math.sqrt(v2 / (m - 1)) : 0;
  out[3 * bins + b] = lo;
  out[4 * bins + b] = hi;
  const k = medianCiRank(m), h50 = (m - 1) * 0.5, h25 = (m - 1) * 0.25, h75 = (m - 1) * 0.75, g = Math.floor(m / 4);
  const n = wantRanks(m, [k - 1, m - k, h50, h50 + 1, h25, h25 + 1, h75, h75 + 1, g, m - g - 1]);
  orderStats(s, m, lo, hi, 0, 0, n, 0);
  out[5 * bins + b] = quantileOf(h50, m);
  out[6 * bins + b] = quantileOf(h25, m);
  out[7 * bins + b] = quantileOf(h75, m);
  out[8 * bins + b] = rankValue(k - 1);
  out[9 * bins + b] = rankValue(m - k);
  if (iqm) iqmStats(s, m, g, rankValue(g), rankValue(m - g - 1), out, bins, b);
}

/** Interquartile mean of the m values in `s` (the mean of ranks [g, m - g), g = floor(m / 4)), whose values at
 * ranks g and m - g - 1 are lo and hi; with Yuen's standard error (from the winsorized variance) and the count h
 * it keeps, whose t quantile on h - 1 degrees of freedom gives its 95% CI. */
function iqmStats(s, m, g, lo, hi, out, bins, b) {
  const h = m - 2 * g;
  let le = 0, mid = 0, nmid = 0, wsum = 0;
  for (let i = 0; i < m; i++) {
    const v = s[i];
    if (v <= lo) le++;
    else if (v < hi) (mid += v), nmid++;
    wsum += v < lo ? lo : v > hi ? hi : v;
  }
  const nlo = lo === hi ? h : le - g; // copies of lo, then of hi, fill what is left of the window
  const iqm = lo === hi ? lo : (mid + lo * nlo + hi * (h - nmid - nlo)) / h;
  const wmean = wsum / m;
  let w2 = 0;
  for (let i = 0; i < m; i++) w2 += ((s[i] < lo ? lo : s[i] > hi ? hi : s[i]) - wmean) ** 2;
  out[10 * bins + b] = iqm;
  out[11 * bins + b] = h > 1 ? Math.sqrt(w2 / (h * (h - 1))) : 0;
  out[12 * bins + b] = h;
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

function rankValue(r) {
  let i = 0;
  while (want[i] !== r) i++;
  return got[i];
}

/** The interpolated quantile at fractional rank h of m sorted values, from the resolved ranks. */
function quantileOf(h, m) {
  const i = Math.floor(h), f = h - i;
  return i + 1 < m ? rankValue(i) * (1 - f) + rankValue(i + 1) * f : rankValue(i);
}

const HIST = 1024; // buckets per histogram pass of orderStats
const SMALL_SORT = 64; // at most this many values are sorted outright
const hist = new Int32Array(HIST), slotOf = new Int32Array(HIST).fill(-1);
const gathered = [];
let sortBuf = new Float64Array(SMALL_SORT);

/** The values at the ranks want[w0, w1) (ascending, of the whole set) into got[w0, w1): a holds n values within
 * [lo, hi], of ranks base onward, in any order. Exact: a histogram over [lo, hi] finds the buckets holding the
 * wanted ranks, and only their values are gathered and resolved, by sorting when few. */
function orderStats(a, n, lo, hi, base, w0, w1, depth) {
  if (lo === hi) return got.fill(lo, w0, w1);
  const scale = HIST / (hi - lo);
  if (n <= SMALL_SORT || depth > 6 || !(scale < Infinity)) return sortedRanks(a, n, base, w0, w1);
  histogram(a, n, lo, scale);
  const groups = wantedBuckets(base, w0, w1);
  const { buf, start, end } = gather(a, n, lo, scale, groups, depth);
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

function histogram(a, n, lo, scale) {
  hist.fill(0);
  for (let i = 0; i < n; i++) {
    const k = ((a[i] - lo) * scale) | 0;
    hist[k < HIST ? k : HIST - 1]++;
  }
}

/** The histogram buckets holding want[w0, w1): {k (bucket), first (its first rank), w (want ranges, in pairs)}. */
function wantedBuckets(base, w0, w1) {
  const groups = { k: [], first: [], w: [] };
  let cum = base, w = w0;
  for (let k = 0; k < HIST && w < w1; k++) {
    const from = w;
    while (w < w1 && want[w] < cum + hist[k]) w++;
    if (w > from) groups.k.push(k), groups.first.push(cum), groups.w.push(from, w);
    cum += hist[k];
  }
  return groups;
}

/** The values of a in the wanted buckets, each bucket's together: {buf, start, end} per bucket. */
function gather(a, n, lo, scale, groups, depth) {
  let total = 0;
  const start = groups.k.map((k, g) => ((slotOf[k] = g), (total += hist[k]), total - hist[k])), end = start.slice();
  const buf = (gathered[depth] = gathered[depth]?.length >= total ? gathered[depth] : new Float64Array(Math.max(total, 64)));
  for (let i = 0; i < n; i++) {
    const k = ((a[i] - lo) * scale) | 0, g = slotOf[k < HIST ? k : HIST - 1];
    if (g >= 0) buf[end[g]++] = a[i];
  }
  for (const k of groups.k) slotOf[k] = -1;
  return { buf, start, end };
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
  if (c.sorted[xmode === 0 ? 0 : 1]) {
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
