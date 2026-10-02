import test from "node:test";
import assert from "node:assert/strict";
import * as K from "../trex/static/kernel.js";

function column(s, v, t = s) {
  const c = new K.Col();
  c.push(Float64Array.from(s), Float64Array.from(v), Float64Array.from(t));
  return c;
}

function twema(xs, ys, alpha, scale) {
  let acc = 0, deb = 0, last = NaN;
  return ys.map((y, i) => {
    if (!Number.isFinite(y)) return y;
    const w = Number.isNaN(last) ? 0 : alpha ** ((xs[i] - last) / scale);
    acc = acc * w + y;
    deb = deb * w + 1;
    last = xs[i];
    return acc / deb;
  });
}

function crcRef(buf) {
  let c = ~0;
  for (const b of buf) {
    c ^= b;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
  }
  return ~c >>> 0;
}

test("crc32 matches the IEEE reference for all tail lengths", () => {
  for (const n of [0, 1, 7, 8, 9, 1000, 65537]) {
    const buf = Uint8Array.from({ length: n }, (_, i) => (i * 31 + 7) & 255);
    assert.equal(K.crc32(buf), crcRef(buf));
    assert.equal(K.crc32(buf.buffer), crcRef(buf));
  }
});

test("time-weighted EMA matches reference, skips NaN, and extends incrementally", () => {
  const xs = [0, 1, 2, 5, 6], v = [1, 2, NaN, 4, 8];
  const c = column(xs, v);
  const want = twema(xs, v, 0.9, 1);
  for (let i = 0; i < 5; i++) {
    const r = K.nearest(c, 0, xs[i], 0.9, 1);
    if (Number.isNaN(want[i])) assert.ok(Number.isNaN(r.y));
    else assert.ok(Math.abs(r.y - want[i]) < 1e-12, `${i}: ${r.y} vs ${want[i]}`);
    assert.equal(r.raw, v[i]);
  }
  c.push([7], [16], [7]);
  assert.ok(Math.abs(K.nearest(c, 0, 7, 0.9, 1).y - twema([...xs, 7], [...v, 16], 0.9, 1)[5]) < 1e-12);
});

test("incremental smoothing across many growing pushes equals a full recompute", () => {
  const n = 5000, xs = Array.from({ length: n }, (_, i) => i), ys = xs.map((x) => Math.sin(x / 40) + (x % 7) / 10);
  const c = new K.Col();
  for (let i = 0; i < n; i += 137) {
    c.push(xs.slice(i, i + 137), ys.slice(i, i + 137), xs.slice(i, i + 137));
    K.nearest(c, 0, i, 0.95, 3);
  }
  const want = twema(xs, ys, 0.95, 3);
  for (const i of [0, 136, 137, 2500, n - 1]) assert.ok(Math.abs(K.nearest(c, 0, i, 0.95, 3).y - want[i]) < 1e-9);
  assert.equal(c.len, n);
  assert.deepEqual(c.extent(0), [0, n - 1, 1]);
});

test("smoothing window is in x units: sparse and dense series agree", () => {
  const dense = Array.from({ length: 101 }, (_, i) => i);
  const sparse = dense.filter((x) => x % 10 === 0);
  const f = (x) => (x >= 50 ? 1 : 0);
  // 50 x-units after the jump both have caught up; a per-point EMA would leave the sparse one at ~0.68.
  const dv = K.nearest(column(dense, dense.map(f)), 0, 100, 0.9, 1).y;
  const sv = K.nearest(column(sparse, sparse.map(f)), 0, 100, 0.9, 1).y;
  assert.ok(dv > 0.99 && sv > 0.99, `dense ${dv} sparse ${sv}`);
});

test("prep decimates to per-pixel extremes and reports visible y range", () => {
  const n = 100000;
  const s = Array.from({ length: n }, (_, i) => i);
  const v = s.map((i) => Math.sin(i / 50) + (i === 77777 ? 10 : 0));
  const c = column(s, v);
  const out = new Float64Array(2 * (4 * 100 + 16));
  const r = K.prep(c, 0, 0, n - 1, 100, 0, 0, 1, out);
  assert.ok(r.n <= 4 * 101, `${r.n} points`);
  assert.equal(r.ymax, v[77777]);
  assert.ok([...out.subarray(0, 2 * r.n)].some((y, i) => i % 2 === 1 && y === v[77777]), "spike survives decimation");
  assert.ok(K.prep(c, 0, 0, 1000, 100, 0, 0, 1, out).ymax <= 1.0);
});

test("prep breaks lines at NaN and at non-positive values in log mode", () => {
  const c = column([0, 1, 2, 3, 4], [1, NaN, 2, -1, 3]);
  const out = new Float64Array(128);
  const r = K.prep(c, 0, 0, 4, 1000, K.LOGY, 0, 1, out);
  const ys = [...out.subarray(0, 2 * r.n)].filter((_, i) => i % 2 === 1);
  assert.deepEqual(ys.map((y) => (Number.isNaN(y) ? "br" : y)), [1, "br", 2, "br", 3]);
});

const stats = (a, bins) => Object.fromEntries(K.STATS.map((k, i) => [k, [...a.subarray(i * bins, (i + 1) * bins)]]));
const aggOf = (cols, x0, x1, bins, flags = 0, alpha = 0) => stats(K.agg(cols, 0, x0, x1, bins, flags, alpha, 1), bins);

test("agg averages within bins, then summarizes across runs", () => {
  const a = column([0, 1, 2, 3], [1, 3, 10, 10]);
  const b = column([0, 1, 2, 3], [5, 5, 20, 40]);
  const c = column([0, 1, 2, 3], [9, 9, 0, 0]);
  const s = aggOf([a, b, c], 0, 4, 2);
  assert.deepEqual(s.n, [3, 3]);
  assert.deepEqual(s.mean, [(2 + 5 + 9) / 3, (10 + 30 + 0) / 3]);
  assert.deepEqual(s.median, [5, 10]);
  assert.deepEqual([s.min, s.max], [[2, 0], [9, 30]]);
  assert.deepEqual([s.q25, s.q75], [[3.5, 5], [7, 20]]);
  assert.ok(Math.abs(s.std[0] - Math.sqrt(((2 - 16 / 3) ** 2 + (5 - 16 / 3) ** 2 + (9 - 16 / 3) ** 2) / 2)) < 1e-12);
  assert.deepEqual([s.medlo, s.medhi], [[2, 0], [9, 30]]);
});

test("median CI uses the widest order statistics reaching 95% coverage", () => {
  for (const [n, k, cov] of [[5, 1, 0.9375], [9, 2, 0.9609375], [20, 6, 0.9586105346679688], [100, 40, 0.9647997997822952]]) {
    const cols = Array.from({ length: n }, (_, i) => column([0], [i + 1]));
    const s = aggOf(cols, 0, 1, 1);
    assert.deepEqual([s.medlo[0], s.medhi[0]], [k, n - k + 1], `n=${n}`);
    assert.ok(Math.abs(K.medianCiCoverage(n) - cov) < 1e-9, `coverage n=${n}`);
  }
});

test("agg summarizes raw or smoothed values on request", () => {
  const a = column([0, 1], [0, 10]);
  assert.deepEqual(aggOf([a], 0, 2, 2, K.RAW, 0.5).mean, [0, 10]);
  assert.deepEqual(aggOf([a], 0, 2, 2, 0, 0.5).mean, twema([0, 1], [0, 10], 0.5, 1));
});

test("agg interpolates a sparse column across empty bins", () => {
  assert.deepEqual(aggOf([column([0, 10], [0, 10])], 0, 10, 5).mean, [0, 2.5, 5, 7.5, 10]);
});

test("binGrid edges are multiples of a power-of-two width that a growing range keeps", () => {
  const grids = [1000, 1100, 1500, 1600].map((x1) => K.binGrid(3, x1, 400));
  for (const g of grids) assert.ok(g.dx === 4 && g.g0 === 0 && g.g0 + g.bins * g.dx > 3, JSON.stringify(g));
  assert.deepEqual(grids.map((g) => g.bins), [251, 276, 376, 401]);
  for (const most of [8, 100, 600]) for (const x1 of [7, 1000, 12345.5]) assert.ok(K.binGrid(0, x1, most).bins <= most + 2);
  assert.equal(K.binGrid(0, 2001, 400).dx, 8);
});

test("agg over a growing binGrid range leaves every earlier bin unchanged", () => {
  const xs = Array.from({ length: 2000 }, (_, i) => i), noise = (i, k) => (Math.sin(i * 12.9898 + k * 78.233) * 43758.5453) % 1;
  const at = (n) => {
    const cols = [0, 1, 2].map((k) => column(xs.slice(0, n), xs.slice(0, n).map((i) => noise(i, k))));
    const g = K.binGrid(0, n - 1, 400);
    return { g, s: aggOf(cols, g.g0, g.g0 + g.bins * g.dx, g.bins) };
  };
  const a = at(1000), b = at(1500);
  assert.equal(a.g.dx, b.g.dx);
  for (const k of ["mean", "median", "medlo", "medhi"]) assert.deepEqual(b.s[k].slice(0, a.g.bins - 1), a.s[k].slice(0, a.g.bins - 1), k);
});

test("prep and agg bin in log10 x when LOGX is set", () => {
  const xs = [1, 10, 100, 1000], c = column(xs, [1, 2, 3, 4]);
  const out = new Float64Array(128);
  const r = K.prep(c, 0, 0, 3, 3, K.LOGX, 0, 1, out);
  assert.equal(r.n, 4);
  assert.deepEqual([...out.subarray(0, 8)].filter((_, i) => i % 2 === 0), xs);
  assert.deepEqual(aggOf([c], 0, 3, 3, K.LOGX).mean, [1, 2, 3.5]);
});

test("yrange returns visible-value quantiles for outlier-robust axes", () => {
  const xs = Array.from({ length: 101 }, (_, i) => i);
  const c = column(xs, xs.map((x) => (x === 50 ? 1e6 : x)));
  assert.deepEqual(K.yrange([c], 0, 0, 100, 0, 0, 1, 0.05, 0.95), [5, 96]);
  assert.deepEqual(K.yrange([c], 0, 0, 100, 0, 0, 1, 0, 1), [0, 1e6]);
  assert.equal(K.yrange([c], 0, 200, 300, 0, 0, 1, 0, 1), null);
});

test("an adopted column equals one built by push, and keeps growing", () => {
  const s = Float64Array.from([0, 2, 1, 3]), v = Float64Array.from([1, NaN, 3, 4]), t = Float64Array.from([0.5, 0.2, 1, 2]);
  const a = K.Col.adopt(s.slice(), v.slice(), t.slice(), 3), b = new K.Col();
  b.push(s.subarray(0, 3), v.subarray(0, 3), t.subarray(0, 3));
  for (const c of [a, b]) c.push([5], [6], [7]);
  assert.equal(a.n, b.n);
  assert.deepEqual(a.sorted, b.sorted);
  assert.deepEqual(a.ext, b.ext);
  for (const k of ["s", "v", "t"]) assert.deepEqual([...a[k].subarray(0, a.n)], [...b[k].subarray(0, b.n)]);
});
