// Line charts, drawn with WebGL (gl.js): every run's line stays on the GPU and zoom is a transform;
// group statistics are uploaded per draw. Axes, labels and the hover overlay are Canvas 2D.
import { Col, IQM, LOGX, LOGY, NSTAT, RAW, STATS, X_RUNTIME, binGrid, medianCiCoverage, nearest,
         prep as kprep, visibleRange, yrange } from "./kernel.js";
import { BREAK, Points, Table, pointBuffer, renderer, rgba } from "./gl.js";
import { describe, onWorker } from "./pool.js";
import { DENSITY_PX_PER_BUCKET, LINE_PX_PER_BUCKET } from "./data.js";
import { nonFiniteText } from "./where.js";

const GPU_POINTS = 8e6; // points per line set kept at full resolution (at most half a texture); larger sets are decimated
const EDGE_PX = 4; // px beyond the plot's sides within which a segment's end may still draw inside it
export const DENSITY_AUTO = 300; // "auto" draws a density heatmap above this many lines
const DENSITY_TIP = 8; // runs listed by the density tooltip

const MARGIN = { l: 52, r: 10, t: 6, b: 20 };
export const BAND_LABEL = { ci: "95% CI", iqr: "IQR", minmax: "min/max", std: "±std", stderr: "±stderr", none: "none" };
const CLICK_MAX = 5; // px of movement below which a press is a click
const BOX_MIN = 8; // px of vertical drag that turns an x zoom into a box zoom
const AIM_MIN = 16; // px of horizontal drag from which the zoom's blocks are fetched while it is dragged

// Two-sided 95% Student t critical values for df = 1..30; normal beyond.
const T95 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.16, 2.145, 2.131,
             2.12, 2.11, 2.101, 2.093, 2.086, 2.08, 2.074, 2.069, 2.064, 2.06, 2.056, 2.052, 2.048, 2.045, 2.042];

let outArr = new Float64Array(2 << 14);
/** Reusable decimation output with room for `pairs` points. */
function outBuf(pairs) {
  if (2 * pairs > outArr.length) outArr = new Float64Array(2 * pairs);
  return outArr.subarray(0, 2 * pairs);
}

export function fmt(v) {
  if (!Number.isFinite(v)) return nonFiniteText(v);
  const a = Math.abs(v);
  if (a !== 0 && (a >= 1e5 || a < 1e-3)) return v.toExponential(2);
  return String(+v.toPrecision(4));
}

export function fmtSI(v) {
  const a = Math.abs(v);
  if (a >= 1e9) return +(v / 1e9).toPrecision(3) + "G";
  if (a >= 1e6) return +(v / 1e6).toPrecision(3) + "M";
  if (a >= 1e3) return +(v / 1e3).toPrecision(3) + "k";
  return String(+v.toPrecision(3));
}

export function fmtDur(s) {
  if (s < 60) return `${+s.toFixed(1)}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m${Math.round(s % 60)}s`;
  return `${Math.floor(s / 3600)}h${Math.round((s % 3600) / 60)}m`;
}

function niceTicks(lo, hi, n) {
  const span = hi - lo;
  if (!(span > 0)) return [lo];
  const raw = span / n;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 5, 10].map((m) => m * mag).find((s) => s >= raw);
  const out = [];
  for (let k = Math.ceil(lo / step - 1e-9); k * step <= hi + step * 1e-9; k++) out.push(+(k * step).toPrecision(12));
  return out;
}

/** Log-axis ticks (log10 space) in [lo, hi]: decades, plus 2× and 5× within narrow ranges; at most ~n. */
function logTicks(lo, hi, n) {
  const mults = hi - lo < 2 ? [1, 2, 5] : [1];
  const out = [];
  for (let d = Math.floor(lo); d <= Math.ceil(hi); d++)
    for (const m of mults) {
      const t = d + Math.log10(m);
      if (t >= lo && t <= hi) out.push(t);
    }
  if (out.length < 2) return niceTicks(10 ** lo, 10 ** hi, n).filter((v) => v > 0).map(Math.log10);
  const every = Math.max(1, Math.ceil(out.length / n));
  return out.filter((_, i) => i % every === 0);
}

function durTicks(lo, hi, n) {
  const steps = [1, 5, 10, 30, 60, 300, 600, 1800, 3600, 7200, 21600, 43200, 86400];
  const step = steps.find((s) => (hi - lo) / s <= n) || 86400 * Math.ceil((hi - lo) / n / 86400);
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi; v += step) out.push(v);
  return out;
}

/** Smoothing reference width (x units per EMA step), quantized so streaming growth rarely invalidates caches. */
export function smoothScale(span) {
  const s = span > 0 ? span / 1000 : 1;
  return 2 ** Math.round(Math.log2(s));
}

/** [lo, hi] per bin of `band` around `center`: ci (order-statistic for the median, Student t for the mean, Yuen's for
 * the IQM), iqr, minmax, ±std, ±stderr; none for one run. */
function bandOf(st, center, band, bins) {
  const lo = new Float64Array(bins), hi = new Float64Array(bins);
  const at = (k) => st.a.subarray(st.o + ROW[k] * bins, st.o + (ROW[k] + 1) * bins);
  const c = at(center), n = at("n"), std = at("std"), iqmse = at("iqmse"), iqmh = at("iqmh");
  const [bandLo, bandHi] = { ci: ["medlo", "medhi"], iqr: ["q25", "q75"], minmax: ["min", "max"] }[band] || ["n", "n"];
  const blo = at(bandLo), bhi = at(bandHi);
  for (let i = 0; i < bins; i++) {
    const m = c[i], se = center === "iqm" ? iqmse[i] : std[i] / Math.sqrt(n[i]);
    let a = m, b = m;
    if (n[i] > 1) {
      if (band === "ci" && center !== "median") {
        const df = center === "iqm" ? iqmh[i] - 1 : n[i] - 1, t = df <= 30 ? T95[df - 1] ?? 0 : 1.96;
        (a = m - t * se), (b = m + t * se);
      } else if (band === "ci" || band === "iqr" || band === "minmax") (a = blo[i]), (b = bhi[i]);
      else if (band === "std") (a = m - std[i]), (b = m + std[i]);
      else if (band === "stderr") (a = m - se), (b = m + se);
    }
    (lo[i] = a), (hi[i] = b);
  }
  return [lo, hi];
}

const ROW = Object.fromEntries(STATS.map((k, i) => [k, i])); // row of each statistic in aggGroups' result
const WAITING = Symbol("waiting for a worker"); // what compute returns while a worker computes the chart's statistics
let chartSlots = 0; // charts given a worker so far

/** [first, last, smallest positive] x of columns' extent `e`, widened by `more` ([first, last], or null). */
function withExtent(e, more) {
  if (!more) return e;
  return [Math.min(e[0], more[0]), Math.max(e[1], more[1]), Math.min(e[2], more[0] > 0 ? more[0] : Infinity)];
}

/** Lines drawing each run of `groups` as its bin means `rows` (bins per run, from transformed x g0 in steps of dx):
 * one column over the bin centers. */
function rowLines(groups, rows, g0, dx, bins, logx) {
  const s = Float64Array.from({ length: bins }, (_, i) => (logx ? 10 ** (g0 + (i + 0.5) * dx) : g0 + (i + 0.5) * dx));
  return groups.map((ln, i) => ({ ...ln, cols: [Col.adopt(s, rows.subarray(i * bins, (i + 1) * bins), s, bins)] }));
}

/** A hash of a typed array's 32-bit words. */
function hashWords(a) {
  const w = new Int32Array(a.buffer, a.byteOffset, a.byteLength >> 2);
  let h = 2166136261, g = 0;
  for (let i = 0; i < w.length; i++) (h = Math.imul(h ^ w[i], 16777619)), (g = (g + Math.imul(w[i], 0x9e3779b1)) | 0);
  return `${h >>> 0}.${g >>> 0}`;
}

/** Band name for the tooltip; the median CI states its exact coverage when 95% is unreachable. */
function bandLabel(band, center, n) {
  if (band !== "ci" || center !== "median") return BAND_LABEL[band];
  const cov = medianCiCoverage(n);
  return cov >= 0.95 ? "95% CI" : `${(cov * 100).toFixed(1)}% CI (min–max)`;
}

/** Tooltip heading for x: a runtime or a step. */
const xLabel = (v, x) => (v.xmode === X_RUNTIME ? fmtDur(x) : `step ${fmtSI(Math.round(x))}`);

function dot(ctx, x, y, color) {
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(x, y, 3, 0, 7);
  ctx.fill();
}

/** Tooltip order: highest value first. */
const byValue = (a, b) => (b.val > a.val ? 1 : b.val < a.val ? -1 : 0);

/** Calls grow(y) with the min and max of valid y over points with transformed x in [x0, x1]. */
function visibleY(c, xmode, ys, x0, x1, logx, logy, grow) {
  const xs = c.xs(xmode), [lo, hi] = visibleRange(c, xmode, x0, x1, logx);
  let mn = Infinity, mx = -Infinity;
  for (let i = lo; i < hi; i++) {
    const x = xs[i], xt = logx ? (x > 0 ? Math.log10(x) : NaN) : x, y = ys[i];
    if (xt >= x0 && xt <= x1 && Number.isFinite(y) && (!logy || y > 0)) {
      if (y < mn) mn = y;
      if (y > mx) mx = y;
    }
  }
  if (mn <= mx) grow(mn), grow(mx);
}

const extents = new WeakMap(); // Col -> {key: ext}
/** [xmin, xmax (transformed), ymin, ymax] over a column's valid points, kept per transform key. */
function colExtent(c, xmode, ys, logx, logy, key) {
  let m = extents.get(c);
  if (!m) extents.set(c, (m = {}));
  if (m[key]) return m[key];
  const e = (m[key] = { x0: Infinity, x1: -Infinity, y0: Infinity, y1: -Infinity }), xs = c.xs(xmode);
  for (let i = 0; i < c.n; i++) {
    const x = xs[i], xt = logx ? (x > 0 ? Math.log10(x) : NaN) : x, y = ys[i];
    if (!Number.isFinite(xt) || !Number.isFinite(y) || (logy && !(y > 0))) continue;
    if (xt < e.x0) e.x0 = xt;
    if (xt > e.x1) e.x1 = xt;
    if (y < e.y0) e.y0 = y;
    if (y > e.y1) e.y1 = y;
  }
  return e;
}

/** Writes points (xs[i], ys[i]) for i in [0, n) as f32 relative to (ox, oy) in transformed space at dst point `at`;
 * invalid points become line breaks. */
function fillPoints(xs, xstep, ys, ystep, n, dst, at, ox, oy, logx, logy) {
  let k = 2 * at;
  for (let i = 0; i < n; i++) {
    const x = xs[i * xstep], y = ys[i * ystep];
    const xt = logx ? (x > 0 ? Math.log10(x) : NaN) : x;
    if (Number.isFinite(xt) && Number.isFinite(y) && (!logy || y > 0)) {
      dst[k] = xt - ox;
      dst[k + 1] = (logy ? Math.log10(y) : y) - oy;
    } else dst[k] = dst[k + 1] = BREAK;
    k += 2;
  }
}

/** Running y range of the values a view shows (positive ones only on a log axis). */
class YRange {
  constructor(logy) {
    this.logy = logy;
    this.lo = Infinity;
    this.hi = -Infinity;
  }

  add(v) {
    if (Number.isFinite(v) && (!this.logy || v > 0)) (this.lo = Math.min(this.lo, v)), (this.hi = Math.max(this.hi, v));
  }

  /** Grow toward range `b`, by at most `reach` of this range's span (log span on a log axis) on either side. */
  widen(b, reach) {
    if (!(b.lo <= b.hi) || !(this.lo <= this.hi)) return;
    const t = (y) => (this.logy ? Math.log10(y) : y), u = (y) => (this.logy ? 10 ** y : y);
    const lo = t(this.lo), hi = t(this.hi), pad = reach * (hi - lo || Math.abs(hi) || 1);
    this.lo = Math.min(this.lo, Math.max(b.lo, u(lo - pad)));
    this.hi = Math.max(this.hi, Math.min(b.hi, u(hi + pad)));
  }
}

const BAND_REACH = 0.25; // share of the group lines' y span a band may add to the axis on either side
const Y_FILL = 0.75; // share of a streaming chart's y axis its lines must fill for the axis to keep its range

/** A streaming chart's y range: the shown one grown to hold y, while y fills at least Y_FILL of that; else y. */
function steadyY(y, shown, logy) {
  if (!shown || shown.logy !== logy) return y;
  const y0 = Math.min(y.y0, shown.y0), y1 = Math.max(y.y1, shown.y1);
  return y.y1 - y.y0 >= Y_FILL * (y1 - y0) ? { y0, y1 } : y;
}

/** [x0, x1] in data units: the settings, else the zoom, else the extent (from its smallest positive x on log axes). */
function xRange(o, zoom, e0, e1, epos) {
  let x0 = o.xmin ?? zoom?.[0] ?? e0, x1 = o.xmax ?? zoom?.[1] ?? e1;
  if (o.logx && !(x0 > 0)) x0 = epos;
  if (o.logx && !(x1 > x0)) x1 = x0 * 10;
  if (x0 === x1 && Number.isFinite(x0)) (x0 -= 0.5), (x1 += 0.5);
  return [x0, x1];
}

/** [min, max, smallest positive] x of the columns. */
const xExtents = new WeakMap(); // column list -> [xmode] -> its xExtent

/** [min, max, smallest positive] x of `cols`, kept for the list while it is drawn. */
function xExtent(cols, xmode) {
  let e = xExtents.get(cols);
  if (!e) xExtents.set(cols, (e = []));
  return (e[xmode] ||= colsExtent(cols, xmode));
}

function colsExtent(cols, xmode) {
  let e0 = Infinity, e1 = -Infinity, epos = Infinity;
  for (const c of cols) {
    const r = c.extent(xmode);
    if (r) (e0 = Math.min(e0, r[0])), (e1 = Math.max(e1, r[1])), (epos = Math.min(epos, r[2]));
  }
  return [e0, e1, epos];
}

/** Grow `yr` by column c's values (raw or smoothed) inside the view's x range. */
const colsOfLines = new WeakMap(); // lines -> their columns

/** Every column of `lines` (a run drawn from its buckets may have none), kept for the list while it is drawn. */
function colsOf(lines) {
  let c = colsOfLines.get(lines);
  if (!c) colsOfLines.set(lines, (c = lines.flatMap((g) => g.cols).filter(Boolean)));
  return c;
}

function growVisible(c, v, raw, yr) {
  const ys = c.ys(v.alpha, raw);
  const e = colExtent(c, v.xmode, ys, v.logx, v.logy, `${v.xmode}|${v.logx}|${v.logy}|${raw ? 0 : v.alpha}|${raw ? 0 : v.scale}`);
  if (e.x0 >= v.x0 && e.x1 <= v.x1) {
    if (e.y0 <= e.y1) yr.add(e.y0), yr.add(e.y1);
  } else visibleY(c, v.xmode, ys, v.x0, v.x1, v.logx, v.logy, (y) => yr.add(y));
}

let staging = null, scratchPts = null;

/** A chart's run lines (raw or smoothed) in a GPU texture, uploaded only when columns or the transform change. */
class LineSet {
  constructor(r, raw) {
    this.r = r;
    this.raw = raw;
    this.pts = new Points(r);
    this.slots = new Map(); // Col -> {off, n}
    this.key = null;
    this.win = null; // [w0, w1] transformed x when decimated
    this.table = new Table(64);
  }

  /** Bring every column up to date. p: {xmode, logx, logy, alpha, scale, ex0, ex1, vx0, vx1, pw}.
   * False when the GPU cannot hold the set. */
  sync(cols, p) {
    const alpha = this.raw ? 0 : p.alpha;
    const key = `${p.xmode}|${p.logx}|${p.logy}|${alpha}|${alpha > 0 ? p.scale : 0}`;
    const fresh = this.pts.live && key === this.key && this.inWindow(p) && this.updateStale(cols, p, alpha);
    return fresh || this.build(cols, p, alpha, key);
  }

  /** Whether the view stays inside the decimation window (if any) at enough resolution. */
  inWindow(p) {
    if (!this.win) return true;
    const [w0, w1] = this.win;
    return p.vx0 >= w0 && p.vx1 <= w1 && (this.R * (p.vx1 - p.vx0)) / (w1 - w0) >= p.pw;
  }

  /** Upload the columns not yet here, when they are few of them (a set mostly new goes up whole, in one upload); false
   * if a rebuild is needed instead. */
  updateStale(cols, p, alpha) {
    const stale = cols.filter((c) => !this.slots.has(c));
    return stale.length <= Math.min(256, cols.length >> 1) && stale.every((c) => this.update(c, p, alpha));
  }

  build(cols, p, alpha, key) {
    this.key = key;
    this.slots.clear();
    let total = 0;
    for (const c of cols) total += c.n;
    this.win = null;
    const budget = Math.min(GPU_POINTS, this.r.capacity / 2);
    if (total > budget) {
      this.R = Math.min(16384, Math.max(Math.ceil(2 * p.pw), Math.floor(budget / (4 * cols.length))));
      const span = p.vx1 - p.vx0, head = (p.ex1 - p.ex0) / 8;
      this.win = [Math.max(p.ex0, p.vx0 - span / 2), Math.min(p.ex1 + head, p.vx1 + span / 2)];
      if (!(this.win[1] > this.win[0])) this.win = [p.vx0, p.vx1];
    }
    this.ox = this.win ? this.win[0] : p.ex0;
    this.oy = NaN;
    for (const c of cols) {
      if (this.oy === this.oy) break;
      c.ensureSmooth(alpha, p.scale, p.xmode);
      const ys = c.ys(alpha, false);
      for (let i = 0; i < c.n; i++) if (Number.isFinite(ys[i]) && (!p.logy || ys[i] > 0)) {
        this.oy = p.logy ? Math.log10(ys[i]) : ys[i];
        break;
      }
    }
    if (!(this.oy === this.oy)) this.oy = 0;
    let bound = 0;
    for (const c of cols) bound += this.bound(c);
    staging = pointBuffer(staging, bound);
    let at = 0;
    for (const c of cols) {
      const n = this.convert(c, p, alpha, staging, at);
      this.slots.set(c, { off: at, n });
      at += n;
    }
    this.next = at;
    return this.pts.upload(staging, at, Math.ceil(at * 1.25) + 4096);
  }

  /** Upper bound on points written by convert. */
  bound(c) {
    return this.win ? Math.min(2 * c.n + 2, 4 * this.R + 1024) : c.n;
  }

  /** Writes column c's points (all, or decimated over the window) at `at`; returns the count. */
  convert(c, p, alpha, dst, at) {
    c.ensureSmooth(alpha, p.scale, p.xmode);
    if (!this.win) {
      fillPoints(c.xs(p.xmode), 1, c.ys(alpha, false), 1, c.n, dst, at, this.ox, this.oy, p.logx, p.logy);
      return c.n;
    }
    const out = outBuf(4 * this.R + 1024);
    const flags = (p.logy ? LOGY : 0) | (p.logx ? LOGX : 0) | (alpha > 0 ? 0 : RAW);
    const r = kprep(c, p.xmode, this.win[0], this.win[1], this.R, flags, alpha, p.scale, out);
    fillPoints(out, 2, out.subarray(1), 2, r.n, dst, at, this.ox, this.oy, p.logx, p.logy);
    return r.n;
  }

  /** Upload a new column in a slot after the others, so points a draw may have read are never overwritten. False if
   * a rebuild is needed. */
  update(c, p, alpha) {
    scratchPts = pointBuffer(scratchPts, this.bound(c));
    const n = this.convert(c, p, alpha, scratchPts, 0);
    if (this.next + n > this.pts.cap) return false;
    this.pts.write(this.next, scratchPts, n);
    this.slots.set(c, { off: this.next, n });
    this.next += n;
    return true;
  }

  /** Line table for `lines` (each with cols[0] and color) in draw order: of each line, the points whose segments can
   * show in view v, `pw` px wide (every point of a decimated set). */
  tableFor(lines, v, pw) {
    const t = this.table, pad = (EDGE_PX * (v.x1 - v.x0)) / pw;
    t.clear();
    for (const ln of lines) {
      const s = this.slots.get(ln.cols[0]);
      if (!s || this.win) t.push(s ? s.off : 0, s ? s.n : 0, ln.color);
      else {
        const [lo, hi] = visibleRange(ln.cols[0], v.xmode, v.x0 - pad, v.x1 + pad, v.logx);
        t.push(s.off + lo, Math.max(0, hi - lo), ln.color);
      }
    }
    return t;
  }

  release() {
    this.pts.release();
    this.slots.clear();
    this.key = null;
  }
}

/** View transform for points stored relative to (ox, oy). */
function glView(chart, v, ox, oy) {
  const dpr = devicePixelRatio || 1;
  return {
    off: [v.x0 - ox, v.y0 - oy],
    scale: [(chart.pw * dpr) / (v.x1 - v.x0), (chart.ph * dpr) / (v.y1 - v.y0)],
    org: [MARGIN.l * dpr, (MARGIN.t + chart.ph) * dpr],
  };
}

function isDark() {
  const c = rgba(getComputedStyle(document.documentElement).getPropertyValue("--bg") || "#fff");
  return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2] < 128;
}

export class Chart {
  constructor(app, key) {
    this.app = app;
    this.key = key;
    this.dirty = true;
    this.visible = false;
    this.el = document.createElement("div");
    this.el.className = "panel";
    this.el.innerHTML = `<div class="ptitle"><span class="pname"></span><button class="pin" title="pin to the top"><svg viewBox="0 0 16 16" width="12" height="12" aria-hidden="true"><path fill="currentColor" d="M10.2 1.3l4.5 4.5-1.3 1.3-.8-.4-2.7 2.7.4 3.2-1.3 1.3-2.9-2.9-3.6 3.6H1.8v-.7l3.6-3.6-2.9-2.9 1.3-1.3 3.2.4 2.7-2.7-.4-.8z"/></svg></button><button class="hide" title="hide (its section's header links to it)"><svg viewBox="0 0 16 16" width="12" height="12" aria-hidden="true"><path fill="currentColor" d="M2.1 1.1 1 2.2l2.3 2.3C2.1 5.4 1.1 6.6.5 8c1.2 2.9 4 5 7.5 5 1.3 0 2.5-.3 3.6-.9l2.3 2.3 1.1-1.1L2.1 1.1zM8 11.5A3.5 3.5 0 0 1 4.5 8c0-.6.2-1.2.4-1.7l1.2 1.2V8a1.9 1.9 0 0 0 2.4 1.8l1.2 1.2c-.5.3-1.1.5-1.7.5zm7.5-3.5C14.3 5.1 11.5 3 8 3c-.9 0-1.8.2-2.6.5l1.3 1.3c.4-.2.9-.3 1.3-.3A3.5 3.5 0 0 1 11.5 8c0 .5-.1.9-.3 1.3l1.9 1.9c1-.8 1.9-1.9 2.4-3.2z"/></svg></button><button class="full" title="show this chart large (Esc to go back)">⛶</button><button class="gear" title="chart settings">⚙</button></div>
      <div class="pbody"><canvas></canvas><canvas class="overlay"></canvas></div>`;
    this.el.querySelector(".pname").textContent = key;
    this.gear = this.el.querySelector(".gear");
    this.gear.addEventListener("click", (e) => app.panelSettings(this, e.currentTarget));
    this.el.querySelector(".full").addEventListener("click", () => this.toggleShownAlone());
    this.pinBtn = this.el.querySelector(".pin");
    this.pinBtn.addEventListener("click", () => app.togglePin(this.key));
    this.el.querySelector(".hide").addEventListener("click", () => app.togglePanel(this.key));
    this.body = this.el.querySelector(".pbody");
    [this.canvas, this.overlay] = this.el.querySelectorAll("canvas");
    this.el._chart = this;
    this.drag = null;
    this.yzoom = null; // [y0, y1] in data space from a box drag on this chart
    this.overlay.addEventListener("mousemove", (e) => {
      this.hoverEvent = e;
      this.hoverRaf ||= requestAnimationFrame(() => {
        this.hoverRaf = 0;
        if (this.hoverEvent) this.hover(this.hoverEvent);
      });
    });
    this.overlay.addEventListener("mouseleave", () => {
      this.hoverEvent = null;
      this.unhover();
    });
    this.overlay.addEventListener("mousedown", (e) => {
      if (e.button !== 0) return;
      this.drag = { x: e.offsetX, y: e.offsetY };
      this.app.lead = this.key;
      const up = (ev) => {
        window.removeEventListener("mouseup", up);
        this.endDrag(ev);
      };
      window.addEventListener("mouseup", up);
    });
    new ResizeObserver(() => this.resize()).observe(this.body);
  }

  setPinned(on) {
    this.pinBtn.classList.toggle("on", on);
    this.pinBtn.title = on ? "unpin" : "pin to the top";
  }

  /** Whether this chart is shown alone (and so draws regardless of scroll visibility). */
  get full() {
    return this.app.opts.chart === this.key;
  }

  /** Show this chart alone (or, when it is, go back to all charts). */
  toggleShownAlone() {
    this.app.showChartAlone(this.full ? "" : this.key);
  }

  resize() {
    const w = this.body.clientWidth, h = this.body.clientHeight;
    if (!w || !h) return;
    this.w = w;
    this.h = h;
    this.dirty = true;
    this.app.schedule(true);
  }

  /** Give the canvases backing stores of the chart's size (allocated only for drawn charts). */
  fitCanvases() {
    const dpr = devicePixelRatio || 1, W = Math.round(this.w * dpr), H = Math.round(this.h * dpr);
    for (const c of [this.canvas, this.overlay]) if (c.width !== W || c.height !== H) (c.width = W), (c.height = H);
  }

  /** Free the canvas backing stores of a chart scrolled out of view; it redraws when back. */
  releaseCanvases() {
    for (const c of [this.canvas, this.overlay]) if (c.width) (c.width = 0), (c.height = 0);
    this.dirty = true;
  }

  get pw() {
    return Math.max(10, this.w - MARGIN.l - MARGIN.r);
  }
  get ph() {
    return Math.max(10, this.h - MARGIN.t - MARGIN.b);
  }

  /** Query the kernel for everything this chart draws; WAITING while the GPU context is lost (drawn again once it is
   * back). */
  compute() {
    const app = this.app, o = app.panelOpts(this.key), r = renderer();
    if (!r) return null;
    if (r.lost) return WAITING;
    const groups = app.linesFor(this.key), allCols = colsOf(groups);
    const v = this.xView(o, allCols, groups);
    if (!v) return null;
    const yr = new YRange(o.logy), binned = (this.binned = this.binnedOf(o, groups));
    const rows = !app.grouped && binned ? this.binnedRows(groups, v) : null;
    if (rows === WAITING) return WAITING;
    const gl = app.grouped ? null : this.linesGL(r, rows || groups.filter((ln) => ln.cols[0]), v, yr);
    const lines = gl ? gl.lines : this.linesGrouped(groups, allCols, v, o, yr, binned);
    if (lines === WAITING) return WAITING;
    const y = this.yView(o, yr, allCols, v);
    if (!y) return null;
    return { ...v, ...y, lines, o, lineSets: !!gl, density: !!gl?.density };
  }

  /** x range (transformed) and smoothing of the view: the data's extent, or the zoom; null if empty. */
  xView(o, allCols, groups) {
    const { xmode, logx, logy } = o;
    const [e0, e1, epos] = withExtent(xExtent(allCols, xmode), this.bucketsExtent(groups, xmode));
    if (!(e1 >= e0)) return null;
    const zoom = this.app.xrange && this.app.xrange[2] === xmode ? this.app.xrange : null;
    const [x0, x1] = xRange(o, zoom, e0, e1, epos);
    if (!(x1 > x0) || !Number.isFinite(x0)) return null;
    const t = (x) => (logx ? Math.log10(x) : x);
    return { x0: t(x0), x1: t(x1), ex0: t(logx ? epos : e0), ex1: t(e1), xmode, logx, logy, alpha: o.smooth,
             scale: smoothScale(e1 - e0), flags: (logy ? LOGY : 0) | (logx ? LOGX : 0) };
  }

  /** Lines drawn from the GPU line sets: {lines, density}. */
  linesGL(r, groups, v, yr) {
    const faint = v.alpha > 0;
    const density = this.binned && r.heatmaps;
    const p = { ...v, pw: this.pw, vx0: v.x0, vx1: v.x1 };
    const g = this.glLines(r);
    const cols = groups.map((ln) => ln.cols[0]);
    g.main.sync(cols, p);
    if (faint && !density) g.faint.sync(cols, p);
    for (const c of cols) for (const raw of faint ? [false, true] : [false]) growVisible(c, v, raw, yr);
    return { lines: groups.map((ln) => ({ ...ln })), density };
  }

  /** [first, last] x of the runs of `groups` drawn from their buckets (those without a column), kept while neither
   * the groups nor the data change. */
  bucketsExtent(groups, xmode) {
    const data = this.app.data, sig = `${data.version}|${xmode}`;
    if (this.bucketExt?.groups === groups && this.bucketExt.sig === sig) return this.bucketExt.out;
    const runs = groups.flatMap((g) => (g.runs || [g.run]).filter((_, i) => !g.cols[i]));
    this.bucketExt = { groups, sig, out: data.extentOf(runs, this.key, xmode) };
    return this.bucketExt.out;
  }

  /** One center line with its band per group, from per-bin group statistics; they set the y range, which a band
   * widens by at most BAND_REACH of it. */
  linesGrouped(groups, allCols, v, o, yr, binned) {
    let longest = 0;
    for (const c of allCols) longest = Math.max(longest, c.len);
    let n = 0;
    for (const g of groups) n += g.cols.length; // runs, with or without a column
    const px = n > DENSITY_AUTO ? DENSITY_PX_PER_BUCKET : LINE_PX_PER_BUCKET; // as finely as the data is planned
    const { g0, dx, bins } = binGrid(v.x0, v.x1, Math.max(8, Math.min(600, Math.floor(this.pw / px), binned ? Infinity : longest)));
    const p = { xmode: v.xmode, x0: g0, x1: g0 + bins * dx, bins, flags: (v.logx ? LOGX : 0) | (o.center === "iqm" ? IQM : 0),
                alpha: v.alpha, scale: v.scale };
    const st = this.groupStats(this.sources(groups, binned), p, v.alpha > 0);
    if (!st) return WAITING;
    const { main, raws } = st;
    const row = (a, gi, k) => a.subarray((gi * NSTAT + ROW[k]) * bins, (gi * NSTAT + ROW[k] + 1) * bins);
    const centerXY = (center) => {
      const xy = new Float64Array(2 * bins);
      for (let i = 0; i < bins; i++) {
        const kx = Math.min(v.ex1, g0 + (i + 0.5) * dx);
        xy[2 * i] = v.logx ? 10 ** kx : kx;
        xy[2 * i + 1] = v.logy && !(center[i] > 0) ? NaN : center[i];
      }
      return xy;
    };
    const bands = new YRange(v.logy);
    const lines = groups.map((g, gi) => {
      const center = row(main, gi, o.center);
      const [lo, hi] = bandOf({ a: main, o: gi * NSTAT * bins }, o.center, o.band, bins);
      for (let i = 0; i < bins; i++) {
        if (v.logy && !(lo[i] > 0)) lo[i] = center[i];
        yr.add(center[i]);
        if (o.band !== "none") bands.add(lo[i]), bands.add(hi[i]);
      }
      let raw = null;
      if (raws) {
        const rc = row(raws, gi, o.center);
        rc.forEach((y) => yr.add(y));
        raw = centerXY(rc);
      }
      return { ...g, xy: centerXY(center), raw, lo, hi, center, cnt: row(main, gi, "n"), g0, dx };
    });
    yr.widen(bands, BAND_REACH); // the axis follows the lines; a wide band does not stretch it
    return lines;
  }

  /** Whether this chart (options o, lines `groups`) draws its runs from bins of their buckets: group statistics, or a
   * heatmap, of more runs than it draws one by one (`App.coarseAbove`). */
  binnedOf(o, groups) {
    let n = 0;
    for (const g of groups) n += g.cols.length;
    return n > this.app.coarseAbove(o);
  }

  /** The groups' sources of binning: a finished run's buckets in the finest blocks its chart shows ({parts, level},
   * `Data.partsOf`; they cover the bins) when `binned`, else (and for a running run) its column; runs with neither are
   * left out. Kept while neither the groups nor the data change. */
  sources(groups, binned) {
    const data = this.app.data, sig = `${data.version}|${binned}`;
    if (this.src?.groups === groups && this.src.sig === sig) return this.src.out;
    const level = data.levelOf(this.key);
    const one = (r, c) => {
      if (!binned || r.meta.state === "running") return c;
      const parts = data.partsOf(r, this.key, true);
      return parts?.length ? { parts, level } : null;
    };
    const out = groups.map((g) => (g.runs || [g.run]).map((r, i) => one(r, g.cols[i])).filter(Boolean));
    this.src = { groups, sig, out };
    return out;
  }

  /** aggGroups of `cols` with binning p, and raw when `raw`: {main, raws}, by the chart's worker; null until it answers
   * (the chart then prepares again). */
  groupStats(cols, p, raw) {
    return this.fromWorker("agg", describe(cols), p, raw);
  }

  /** A heatmap's lines: each run as its bin means (`rowLines`), from the chart's worker (WAITING until it answers). */
  binnedRows(groups, v) {
    const { g0, dx, bins } = binGrid(v.x0, v.x1, Math.max(8, Math.floor(this.pw / DENSITY_PX_PER_BUCKET)));
    const src = this.sources(groups, true), lines = groups.filter((_, i) => src[i].length), all = src.filter((s) => s.length);
    const p = { xmode: v.xmode, x0: g0, x1: g0 + bins * dx, bins, flags: v.logx ? LOGX : 0, alpha: v.alpha, scale: v.scale };
    const res = this.fromWorker("rows", describe(all), p, false);
    return res ? rowLines(lines, res.rows, g0, dx, bins, v.logx) : WAITING;
  }

  /** The chart's worker's answer (pool.onWorker `kind`) for described sources d, once it has come; null until then
   * (the chart then prepares again). */
  fromWorker(kind, d, p, raw) {
    const sig = `${kind}|${hashWords(d.desc)}|${hashWords(d.refs)}|${hashWords(d.ends)}|${p.xmode}|${p.x0}|${p.x1}|${p.bins}|${p.flags}|${p.alpha}|${p.scale}|${raw}`;
    if (this.stats?.sig === sig) return this.stats.result;
    const job = (this.stats = { sig, result: null });
    onWorker(kind, (this.slot ||= ++chartSlots), d, p, raw).then((res) => {
      if (this.stats !== job) return;
      job.result = res;
      if (this.waiting) (this.waiting = false), this.app.nextFrame();
    });
    return null;
  }

  /** [y0, y1] in data units: the lines' range or outlier quantiles, overridden by settings, then the box zoom. */
  yBounds(o, yr, allCols, v) {
    let { lo: y0, hi: y1 } = yr;
    const q = o.outliers > 0 && allCols.length ? yrange(allCols, v.xmode, v.x0, v.x1, v.flags, v.alpha, v.scale, o.outliers, 1 - o.outliers) : null;
    if (q) [y0, y1] = q;
    return this.yzoom || [o.ymin ?? y0, o.ymax ?? y1];
  }

  /** y range (transformed) of the view; null if empty. */
  yView(o, yr, allCols, v) {
    let [y0, y1] = this.yBounds(o, yr, allCols, v);
    if (!(y1 >= y0) || (o.logy && !(y0 > 0))) {
      if (!(o.logy && y1 > 0)) return null;
      y0 = y1 / 10;
    }
    if (o.logy) (y0 = Math.log10(y0)), (y1 = Math.log10(y1));
    if (y1 === y0) (y0 -= 0.5), (y1 += 0.5);
    const y = this.padY(o, y0, y1);
    return this.app.paced && !(this.yzoom || o.ymin != null || o.ymax != null) ? steadyY(y, this.view, o.logy) : y;
  }

  /** {y0, y1} padded by 4% on the sides neither the box zoom nor a setting fixes. */
  padY(o, y0, y1) {
    const pad = (y1 - y0) * 0.04;
    return { y0: this.yzoom || o.ymin != null ? y0 : y0 - pad, y1: this.yzoom || o.ymax != null ? y1 : y1 + pad };
  }


  /** This chart's GPU line sets, created on first use. */
  glLines(r) {
    return (this.glState ||= { main: new LineSet(r, false), faint: new LineSet(r, true), tmp: new Points(r) });
  }

  /** Free this chart's GPU data. */
  dispose() {
    const g = this.glState;
    if (g) g.main.release(), g.faint.release(), g.tmp.release();
    this.glState = null;
  }

  /** Draw bands and lines of `view` through WebGL onto ctx (device px, clipped to the plot). */
  drawGL(ctx, view) {
    const r = renderer(), dpr = devicePixelRatio || 1;
    if (r.lost) return;
    r.begin(this.canvas.width, this.canvas.height, [MARGIN.l * dpr, MARGIN.t * dpr, this.pw * dpr, this.ph * dpr]);
    const g = this.glLines(r);
    if (view.lineSets) {
      const { main, faint } = g;
      r.touch(main.pts);
      if (view.density) r.density(main.pts, main.tableFor(view.lines, view, this.pw), glView(this, view, main.ox, main.oy), dpr, isDark());
      else {
        if (view.alpha > 0) {
          r.touch(faint.pts);
          r.lines(faint.pts, faint.tableFor(view.lines, view, this.pw), glView(this, view, faint.ox, faint.oy), dpr, 0.22);
        }
        r.lines(main.pts, main.tableFor(view.lines, view, this.pw), glView(this, view, main.ox, main.oy), 1.25 * dpr, 1);
      }
      r.trim(new Set([main.pts, faint.pts]));
    } else this.drawGroupsGL(r, g.tmp, view, dpr);
    r.copyTo(ctx);
  }

  /** Group bands, faint raw centers and center lines, uploaded per draw. */
  drawGroupsGL(r, pts, view, dpr) {
    const { logx, logy } = view, ox = view.x0, oy = view.y0, band = view.o.band !== "none";
    let n = 0;
    for (const ln of view.lines) n += (ln.xy.length >> 1) * (band && ln.lo ? 4 : 2);
    staging = pointBuffer(staging, n);
    const nl = view.lines.length, lines = new Table(nl), raws = new Table(nl), bands = new Table(2 * nl);
    let at = 0;
    const put = (xs, xstep, ys, ystep, len) => {
      fillPoints(xs, xstep, ys, ystep, len, staging, at, ox, oy, logx, logy);
      at += len;
      return at - len;
    };
    for (const ln of view.lines) {
      const k = ln.xy.length >> 1;
      lines.push(put(ln.xy, 2, ln.xy.subarray(1), 2, k), k, ln.color);
      if (ln.raw) raws.push(put(ln.raw, 2, ln.raw.subarray(1), 2, k), k, ln.color);
      if (band && ln.lo) {
        bands.push(put(ln.xy, 2, ln.hi, 1, k), k, ln.color);
        bands.push(put(ln.xy, 2, ln.lo, 1, k), k, ln.color);
      }
    }
    if (!pts.upload(staging, at, at)) return;
    r.touch(pts);
    const tv = glView(this, view, ox, oy);
    if (bands.n) r.bands(pts, bands, tv, 0.18);
    if (raws.n) r.lines(pts, raws, tv, dpr, 0.22);
    r.lines(pts, lines, tv, 2 * dpr, 1);
  }

  px(x) {
    const v = this.view;
    const fx = v.logx ? Math.log10(x) : x;
    return MARGIN.l + ((fx - v.x0) / (v.x1 - v.x0)) * this.pw;
  }
  py(y) {
    const v = this.view;
    const fy = v.logy ? Math.log10(y) : y;
    return MARGIN.t + this.ph - ((fy - v.y0) / (v.y1 - v.y0)) * this.ph;
  }
  /** Data-space y at a canvas pixel. */
  yAt(py) {
    const v = this.view;
    const fy = v.y0 + ((MARGIN.t + this.ph - py) / this.ph) * (v.y1 - v.y0);
    return v.logy ? 10 ** fy : fy;
  }

  /** Data-space x at a canvas pixel. */
  xAt(px) {
    const v = this.view;
    const fx = v.x0 + ((px - MARGIN.l) / this.pw) * (v.x1 - v.x0);
    return v.logx ? 10 ** fx : fx;
  }

  /** Compute what the chart shows next (the costly part), for `draw` to present. */
  prepare() {
    this.dirty = false;
    const next = this.w ? this.compute() : null;
    this.waiting = next === WAITING; // a worker computes what it shows; it is prepared again once that is done
    if (this.waiting) return;
    this.next = next;
    this.prepared = true;
  }

  /** Present the prepared view, preparing it first if it is not; the shown one stays while a worker computes it. */
  draw() {
    if (!this.prepared) this.prepare();
    if (!this.prepared) return;
    this.prepared = false;
    this.gear.classList.toggle("on", this.app.hasPanelOverrides(this.key));
    if (!this.w) return;
    this.fitCanvases();
    const dpr = devicePixelRatio || 1, ctx = this.canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, this.w, this.h);
    const view = (this.view = this.next);
    const css = getComputedStyle(document.documentElement);
    ctx.font = "10px system-ui, sans-serif";
    ctx.fillStyle = css.getPropertyValue("--muted");
    if (!view) return ctx.fillText(this.emptyText(), MARGIN.l + 4, MARGIN.t + 14);
    this.drawAxes(ctx, view, css.getPropertyValue("--grid"));
    this.drawGL(ctx, view);
  }

  /** Why the chart shows nothing: its blocks are on their way or failed, or its runs have no data in view. */
  emptyText() {
    if (!renderer()) return "charts need WebGL2";
    const data = this.app.data, error = data.failure(this.key);
    if (error) return `failed to load: ${error}`;
    return data.charts.get(this.key)?.ready && !data.pending(this.key) ? "no data" : "loading…";
  }

  /** Grid lines and tick labels. */
  drawAxes(ctx, view, grid) {
    ctx.strokeStyle = grid;
    ctx.lineWidth = 1;
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    const ny = Math.max(2, Math.floor(this.ph / 28));
    for (const t of view.logy ? logTicks(view.y0, view.y1, ny) : niceTicks(view.y0, view.y1, ny)) {
      const y = Math.round(this.py(view.logy ? 10 ** t : t)) + 0.5;
      ctx.beginPath();
      ctx.moveTo(MARGIN.l, y);
      ctx.lineTo(MARGIN.l + this.pw, y);
      ctx.stroke();
      ctx.fillText(fmt(view.logy ? +(10 ** t).toPrecision(6) : t), MARGIN.l - 4, y);
    }
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    const nx = Math.max(2, Math.floor(this.pw / 80));
    const xt = view.logx ? logTicks(view.x0, view.x1, nx).map((d) => +(10 ** d).toPrecision(6))
      : view.xmode === X_RUNTIME ? durTicks(view.x0, view.x1, nx) : niceTicks(view.x0, view.x1, nx);
    for (const t of xt) ctx.fillText(view.xmode === X_RUNTIME ? fmtDur(t) : fmtSI(t), this.px(t), MARGIN.t + this.ph + 4);
  }

  /** Stroke the first n (x, y) pairs of xy (data units); NaN y breaks the line. */
  polyline(ctx, xy, n) {
    ctx.beginPath();
    let pen = false;
    for (let i = 0; i < n; i++) {
      const y = xy[2 * i + 1];
      if (y !== y) {
        pen = false;
        continue;
      }
      const X = this.px(xy[2 * i]), Y = this.py(y);
      if (pen) ctx.lineTo(X, Y);
      else ctx.moveTo(X, Y), (pen = true);
    }
    ctx.stroke();
  }

  hover(e) {
    const v = this.view;
    if (!v || this.app.tipPinned) return;
    this.app.hovered = this;
    this.lastX = e.offsetX;
    const dpr = devicePixelRatio || 1, ctx = this.overlay.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, this.w, this.h);
    if (this.drag) return this.drawDragBox(ctx, e);
    if (e.offsetX < MARGIN.l || e.offsetX > MARGIN.l + this.pw) return this.unhover();
    const x = this.xAt(e.offsetX);
    this.crosshair(ctx, e.offsetX);
    if (v.density) return this.hoverDensity(e, ctx, x);
    const rows = [];
    for (const ln of v.lines) {
      const row = ln.lo ? this.groupRow(ln, x) : this.lineRow(ln, x, e.offsetX);
      if (!row) continue;
      row.py = Number.isFinite(row.val) && (!v.logy || row.val > 0) ? this.py(row.val) : NaN;
      if (row.py === row.py) dot(ctx, row.px, row.py, ln.color);
      rows.push(row);
    }
    const near = this.nearestRow(rows, e.offsetY);
    rows.sort(byValue);
    this.app.tip(e, this, xLabel(v, x), rows, rows.indexOf(near));
  }

  crosshair(ctx, px) {
    ctx.strokeStyle = "rgba(127,127,127,0.6)";
    ctx.beginPath();
    ctx.moveTo(px + 0.5, MARGIN.t);
    ctx.lineTo(px + 0.5, MARGIN.t + this.ph);
    ctx.stroke();
  }

  drawDragBox(ctx, e) {
    const d = this.drag, w = Math.abs(e.offsetX - d.x), h = Math.abs(e.offsetY - d.y);
    ctx.fillStyle = "rgba(127,127,127,0.2)";
    if (h >= BOX_MIN) ctx.fillRect(Math.min(d.x, e.offsetX), Math.min(d.y, e.offsetY), w, h);
    else ctx.fillRect(Math.min(d.x, e.offsetX), MARGIN.t, w, this.ph);
    if (w >= AIM_MIN) this.app.aimZoom([this.xAt(Math.min(d.x, e.offsetX)), this.xAt(Math.max(d.x, e.offsetX)), this.view.xmode]);
  }

  /** Tooltip row of a group line at x: its center, band and count in that bin. */
  groupRow(ln, x) {
    const v = this.view, fx = v.logx ? Math.log10(x) : x;
    const i = Math.min(ln.center.length - 1, Math.max(0, Math.floor((fx - ln.g0) / ln.dx))), val = ln.center[i];
    if (!Number.isFinite(val)) return null;
    let extra = v.o.band !== "none" ? ` ${bandLabel(v.o.band, v.o.center, ln.cnt[i])} [${fmt(ln.lo[i])}, ${fmt(ln.hi[i])}]` : "";
    extra += ` n=${ln.cnt[i]}`;
    if (ln.raw && Number.isFinite(ln.raw[2 * i + 1])) extra += ` (raw ${fmt(ln.raw[2 * i + 1])})`;
    return { ln, val, extra, px: this.px(ln.xy[2 * i]) };
  }

  /** Tooltip row of a run line: its point nearest x, if within 40 px of the pointer. */
  lineRow(ln, x, offsetX) {
    const v = this.view, r = nearest(ln.cols[0], v.xmode, x, v.alpha, v.scale);
    if (!r || Math.abs(this.px(r.x) - offsetX) > 40) return null;
    return { ln, val: r.y, extra: v.alpha > 0 ? ` (raw ${fmt(r.raw)})` : "", px: this.px(r.x) };
  }


  /** The row whose point is vertically nearest the pointer, ringed on the overlay. */
  nearestRow(rows, y) {
    let best = null;
    for (const r of rows) if (Math.abs(r.py - y) < Math.abs((best?.py ?? Infinity) - y)) best = r;
    if (best) {
      const ctx = this.overlay.getContext("2d");
      ctx.strokeStyle = best.ln.color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(best.px, best.py, 6, 0, 7);
      ctx.stroke();
    }
    return best;
  }

  /** Density tooltip: the runs whose value at x is nearest the pointer, the nearest one traced. */
  hoverDensity(e, ctx, x) {
    const v = this.view, near = [];
    for (const ln of v.lines) {
      const r = nearest(ln.cols[0], v.xmode, x, v.alpha, v.scale);
      if (!r || !Number.isFinite(r.y) || (v.logy && !(r.y > 0)) || Math.abs(this.px(r.x) - e.offsetX) > 40) continue;
      const d = Math.abs(this.py(r.y) - e.offsetY);
      if (near.length === DENSITY_TIP && d >= near[DENSITY_TIP - 1].d) continue;
      let i = near.length === DENSITY_TIP ? DENSITY_TIP - 1 : near.length;
      while (i > 0 && near[i - 1].d > d) (near[i] = near[i - 1]), i--;
      near[i] = { d, ln, r };
    }
    if (near.length) this.traceLine(ctx, near[0].ln, 1.5);
    for (const { ln, r } of near) dot(ctx, this.px(r.x), this.py(r.y), ln.color);
    const rows = near.map(({ ln, r }) => ({ ln, val: r.y, extra: v.alpha > 0 ? ` (raw ${fmt(r.raw)})` : "",
                                           px: this.px(r.x), py: this.py(r.y) }));
    const closest = this.nearestRow(rows, e.offsetY);
    rows.sort(byValue);
    this.app.tip(e, this, `${xLabel(v, x)} · ${near.length} nearest of ${v.lines.length}`, rows, rows.indexOf(closest));
  }

  /** Stroke one line of the current view on the overlay: a run's own line, or a group's center. */
  traceLine(ctx, ln, width) {
    const v = this.view;
    let xy, n;
    if (ln.lo) (xy = ln.xy), (n = ln.xy.length >> 1);
    else {
      xy = outBuf(Math.ceil(this.pw) * 4 + 1024);
      n = kprep(ln.cols[0], v.xmode, v.x0, v.x1, this.pw, v.flags, v.alpha, v.scale, xy).n;
    }
    ctx.save();
    ctx.beginPath();
    ctx.rect(MARGIN.l, MARGIN.t, this.pw, this.ph);
    ctx.clip();
    ctx.strokeStyle = ln.color;
    ctx.lineWidth = width;
    ctx.lineJoin = "round";
    this.polyline(ctx, xy, n);
    ctx.restore();
  }

  /** Overlay showing the pinned crosshair with one line emphasized (null: none). */
  highlight(ln) {
    if (!this.view || !this.overlay.width) return;
    const dpr = devicePixelRatio || 1, ctx = this.overlay.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, this.w, this.h);
    this.crosshair(ctx, this.lastX);
    if (ln) this.traceLine(ctx, ln, 3);
  }

  unhover() {
    if (this.app.hovered === this) this.app.hovered = null;
    if (this.app.tipPinned) return;
    if (this.overlay.width) this.overlay.getContext("2d").clearRect(0, 0, this.overlay.width, this.overlay.height);
    this.app.tip(null);
  }

  /** A click resets axes; a drag zooms x (all charts); a box also zooms this chart's y. */
  endDrag(e) {
    const d = this.drag;
    this.drag = null;
    if (!d || !this.view) return;
    const r = this.overlay.getBoundingClientRect();
    const ox = Math.min(Math.max(e.clientX - r.left, MARGIN.l), MARGIN.l + this.pw);
    const oy = Math.min(Math.max(e.clientY - r.top, MARGIN.t), MARGIN.t + this.ph);
    const dx = Math.abs(ox - d.x), dy = Math.abs(oy - d.y);
    this.unhover();
    if (dx < CLICK_MAX && dy < CLICK_MAX) return this.resetAxes();
    if (dy >= BOX_MIN) this.yzoom = [this.yAt(Math.max(d.y, oy)), this.yAt(Math.min(d.y, oy))];
    if (dx >= CLICK_MAX) this.app.setXRange([this.xAt(Math.min(d.x, ox)), this.xAt(Math.max(d.x, ox)), this.view.xmode]);
    else {
      this.dirty = true;
      this.app.schedule(true);
    }
    this.app.updateZoomButton();
  }

  /** Back to automatic axes: this chart's y, and the shared x zoom. */
  resetAxes() {
    this.yzoom = null;
    if (this.app.xrange) this.app.setXRange(null);
    else {
      this.dirty = true;
      this.app.schedule(true);
    }
    this.app.updateZoomButton();
  }
}
