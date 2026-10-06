import test from "node:test";
import assert from "node:assert/strict";
import * as K from "../trex/static/kernel.js";

function column(s, v, t = s) {
  return K.Col.adopt(Float64Array.from(s), Float64Array.from(v), Float64Array.from(t), s.length);
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

test("time-weighted EMA matches reference and skips NaN, kept per smoothing", () => {
  const xs = [0, 1, 2, 5, 6], v = [1, 2, NaN, 4, 8];
  const c = column(xs, v);
  for (const [alpha, scale] of [[0.9, 1], [0.5, 2], [0.9, 1]]) {
    const want = twema(xs, v, alpha, scale);
    for (let i = 0; i < 5; i++) {
      const r = K.nearest(c, 0, xs[i], alpha, scale);
      if (Number.isNaN(want[i])) assert.ok(Number.isNaN(r.y));
      else assert.ok(Math.abs(r.y - want[i]) < 1e-12, `${i}: ${r.y} vs ${want[i]}`);
      assert.equal(r.raw, v[i]);
    }
  }
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
const aggOf = (cols, x0, x1, bins, flags = 0, alpha = 0) => stats(K.aggGroups([cols], 0, x0, x1, bins, flags, alpha, 1), bins);

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

test("a run's value in a bin is the mean of its finite values, infinite only when it has none", () => {
  const a = column([0, 1, 2, 3], [1, Infinity, 3, Infinity]);
  const b = column([0, 1], [Infinity, Infinity]);
  const c = column([0, 1], [5, -Infinity]);
  const s = aggOf([a, b, c], 0, 4, 1);
  assert.deepEqual([s.n[0], s.mean[0], s.median[0], s.max[0]], [3, Infinity, 5, Infinity]);
  assert.ok(Number.isNaN(s.std[0]));
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

test("binGrid takes no bin narrower than its least width, which a coarser grid leaves as it is", () => {
  assert.deepEqual(K.binGrid(0, 19700, 56), { g0: 0, dx: 512, bins: 39 });
  assert.deepEqual(K.binGrid(0, 19700, 56, 1024), { g0: 0, dx: 1024, bins: 20 });
  assert.deepEqual(K.binGrid(500, 19700, 56, 256), { g0: 0, dx: 512, bins: 39 });
  assert.deepEqual(K.binGrid(3100, 3500, 8, 1024), { g0: 3072, dx: 1024, bins: 1 });
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

test("an adopted column knows its extents and whether its steps and runtimes rise", () => {
  const s = Float64Array.from([0, 2, 1, 3]), v = Float64Array.from([1, NaN, 3, 4]), t = Float64Array.from([0.5, 0.2, 1, 2]);
  const a = K.Col.adopt(s, v, t, 3), b = K.Col.adopt(s, v, Float64Array.from([0.1, 0.2, 0.3, 0.4]), 2);
  assert.deepEqual([a.sorted, a.extent(0), a.extent(1)], [[false, false], [0, 2, 1], [0.2, 1, 0.2]]);
  assert.deepEqual([b.sorted, b.extent(0), b.extent(1)], [[true, true], [0, 2, 2], [0.1, 0.2, 0.1]]);
});

test("agg's order statistics equal a full sort's for large groups with ties, clusters, outliers and gaps", () => {
  const R = 3000, bins = 24, rnd = ((s) => () => ((s = (s * 1103515245 + 12345) % 2147483648) / 2147483648))(7);
  const shapes = [() => rnd(), () => Math.round(rnd() * 5), () => (rnd() < 0.97 ? 1e-9 * rnd() : 1e9 * rnd()), () => 4.25,
                  () => (rnd() < 0.5 ? -rnd() * 1e-300 : 1e300 * rnd())];
  const cols = [], perBin = Array.from({ length: bins }, () => []);
  for (let r = 0; r < R; r++) {
    const xs = [], ys = [];
    for (let b = 0; b < bins; b++) {
      if (r % 7 === 0 && (b === 0 || b === bins - 1)) continue; // runs that start late or end early
      const y = shapes[b % shapes.length]();
      xs.push(b + 0.5), ys.push(y), perBin[b].push(y);
    }
    cols.push(column(xs, ys));
  }
  const s = aggOf(cols, 0, bins, bins);
  for (let b = 0; b < bins; b++) {
    const v = Float64Array.from(perBin[b]).sort(), m = v.length, k = K.medianCiRank(m);
    const q = (p) => { const h = (m - 1) * p, i = Math.floor(h), f = h - i; return i + 1 < m ? v[i] * (1 - f) + v[i + 1] * f : v[i]; };
    const mean = v.reduce((a, x) => a + x, 0) / m, sd = Math.sqrt(v.reduce((a, x) => a + (x - mean) ** 2, 0) / (m - 1));
    assert.deepEqual([s.n[b], s.min[b], s.max[b], s.median[b], s.q25[b], s.q75[b], s.medlo[b], s.medhi[b]],
                     [m, v[0], v[m - 1], q(0.5), q(0.25), q(0.75), v[k - 1], v[m - k]], `bin ${b}`);
    const close = (a, e) => Math.abs(a - e) <= 1e-9 * Math.max(Math.abs(e), 1e-300) || a === e;
    assert.ok(close(s.mean[b], mean) && close(s.std[b], sd), `bin ${b}: mean ${s.mean[b]} vs ${mean}, std ${s.std[b]} vs ${sd}`);
  }
});

test("agg's interquartile mean and Yuen standard error match a sort's, ties and gaps included", () => {
  const rnd = ((s) => () => ((s = (s * 1103515245 + 12345) % 2147483648) / 2147483648))(11);
  const shapes = [() => rnd(), () => Math.round(rnd() * 3), () => (rnd() < 0.9 ? 1 : 100 * rnd()), () => 7];
  for (const R of [1, 2, 3, 5, 17, 400, 2501]) {
    const perBin = shapes.map(() => []), cols = [];
    for (let r = 0; r < R; r++) {
      const xs = [], ys = [];
      shapes.forEach((f, b) => {
        if (r % 9 === 4 && b === 0) return; // runs that start late (agg interpolates gaps inside a run)
        const y = f();
        xs.push(b + 0.5), ys.push(y), perBin[b].push(y);
      });
      cols.push(column(xs, ys));
    }
    const s = stats(K.aggGroups([cols], 0, 0, shapes.length, shapes.length, K.IQM, 0, 1), shapes.length);
    perBin.forEach((vals, b) => {
      const v = Float64Array.from(vals).sort(), m = v.length, g = Math.floor(m / 4), h = m - 2 * g;
      const iqm = v.slice(g, m - g).reduce((a, x) => a + x, 0) / h;
      const w = Array.from(v, (x) => Math.min(Math.max(x, v[g]), v[m - g - 1])), wm = w.reduce((a, x) => a + x, 0) / m;
      const se = h > 1 ? Math.sqrt(w.reduce((a, x) => a + (x - wm) ** 2, 0) / (h * (h - 1))) : 0;
      const close = (a, e) => Math.abs(a - e) <= 1e-9 * Math.max(Math.abs(e), 1);
      assert.ok(close(s.iqm[b], iqm) && close(s.iqmse[b], se) && s.iqmh[b] === h, `R=${R} bin ${b}: ${s.iqm[b]} ${s.iqmse[b]} ${s.iqmh[b]} vs ${iqm} ${se} ${h}`);
    });
  }
  assert.ok(Number.isNaN(aggOf([column([0.5], [1])], 0, 1, 1).iqm[0]), "computed only on request");
});

test("agg's interquartile mean agrees with the CLI's on a hand-checked group", () => {
  const s = stats(K.aggGroups([[1, 2, 3, 4, 5, 6, 7, 8, 100].map((y) => column([0.5], [y]))], 0, 0, 1, 1, K.IQM, 0, 1), 1);
  assert.deepEqual([s.iqm[0], s.iqmh[0]], [5, 5]);
  assert.ok(Math.abs(s.iqmse[0] - Math.sqrt(26 / 20)) < 1e-12);
});

/** A bucket array of one run holding every bucket of block `base` of `level`, each of one row: its mean `mean(q)`, its
 * mean step `frac` of the way through the bucket (buckets.encode). */
function fullBlock(level, base, mean, frac = 0.25) {
  const n = 256, buf = new ArrayBuffer(48 + 2 * 512 + 12 * n);
  new Int32Array(buf, 0, 8).set([0x31424b54, level, base, 0, 1, n, 0, 0]);
  new Uint32Array(buf, 32, 2).set([0, n]);
  new Uint16Array(buf, 48, n).set(Array.from({ length: n }, (_, q) => q));
  new Uint16Array(buf, 48 + 512, n).fill(Math.floor(frac * 65536));
  new Float32Array(buf, 48 + 1024, n).set(Array.from({ length: n }, (_, q) => mean(q)));
  new Uint32Array(buf, 48 + 1024 + 8 * n, n).fill(1);
  return { v: K.bucketViews(buf), row: 0 };
}

/** A bucket array of block `base` of `level` whose row r holds the buckets of `rows[r]`, each [offset, its mean step's
 * offset within the bucket in 1/65536, mean runtime], of one row and mean 0 (buckets.encode). */
function bucketArray(level, base, rows) {
  const runs = rows.length, all = rows.flat(), count = all.length, pad8 = (n) => (n + 7) & ~7;
  const first = 32, offset = first + pad8(4 * (runs + 1)) + pad8(4 * runs), soff = offset + pad8(2 * count), mean = soff + pad8(2 * count);
  const b = new ArrayBuffer(mean + 12 * count);
  new Uint32Array(b, 0, 8).set([0x31424b54, level, base, 0, runs, count, 0, 0]);
  new Uint32Array(b, first, runs + 1).set(rows.reduce((f, r) => [...f, f[f.length - 1] + r.length], [0]));
  new Uint16Array(b, offset, count).set(all.map((q) => q[0]));
  new Uint16Array(b, soff, count).set(all.map((q) => q[1]));
  new Float32Array(b, mean + 4 * count, count).set(all.map((q) => q[2]));
  new Uint32Array(b, mean + 8 * count, count).fill(1);
  return K.bucketViews(b);
}

test("an array's extent spans the buckets of the rows of shown runs binned from their buckets, and nothing else", () => {
  // row 1's run is not shown, row 2's is drawn from its column, row 3 has no run, row 4's is past the table, row 6 is empty
  const v = bucketArray(1, 0, [[[1, 0, 0], [3, 32768, 5]], [[0, 0, -10]], [[100, 0, 50]], [[90, 0, 60]], [[80, 0, 70]], [[2, 0, 1], [50, 0, 9]], []]);
  const rowRun = Int32Array.from([4, 7, 2, -1, 9, 3, 1]), group = Int32Array.from([0, 0, 1, 1, 0, -1, -1, -1]);
  const column = Uint8Array.from([0, 0, 1, 0, 0, 0, 0, 0]);
  const steps = K.rowExtents(v, K.X_STEP), times = K.rowExtents(v, K.X_RUNTIME);
  assert.deepEqual([...steps.subarray(18)], [Infinity, -Infinity, Infinity], "a row without buckets");
  assert.deepEqual(K.arrayExtent(steps, rowRun, group, column), [K.bucketStep(v, 0), K.bucketStep(v, 7), K.bucketStep(v, 0)]);
  assert.deepEqual(K.arrayExtent(times, rowRun, group, column), [0, 9, 1], "the smallest positive runtime of each row");
  assert.deepEqual(K.arrayExtent(steps, rowRun, group.fill(-1), column), [Infinity, -Infinity, Infinity]);
});

test("a column takes each step from the finest level whose blocks hold it, in step order, however the levels' blocks lie", () => {
  const coarse = fullBlock(4, 0, (q) => 4000 + q), fine = fullBlock(0, 9, (q) => q);
  const mid = [0, 1, 2].map((base) => fullBlock(2, base, (q) => 2000 + 256 * base + q));
  const cases = { "three levels": [coarse, mid[0], mid[2], fine], "two levels, blocks apart": [coarse, mid[0], mid[2]],
                  "two levels, blocks side by side": [coarse, mid[1], mid[2]], "one level": [mid[2], mid[0], mid[1]] };
  const blockOf = ({ v }) => [v.base * 256 * 2 ** v.level, (v.base + 1) * 256 * 2 ** v.level];
  for (const [name, parts] of Object.entries(cases)) {
    const held = (level, x) => parts.some((p) => p.v.level < level && blockOf(p)[0] <= x && x < blockOf(p)[1]);
    const want = [];
    for (const { v } of parts) for (let q = 0; q < v.count; q++) if (!held(v.level, K.bucketStep(v, q))) want.push([K.bucketStep(v, q), v.mean[q]]);
    want.sort((a, b) => a[0] - b[0]);
    for (const order of [parts, [...parts].reverse()]) {
      const col = K.buildColumn(order, { s: [], v: [], t: [], q: [], n: 0 }, undefined, false);
      assert.deepEqual(Array.from({ length: col.n }, (_, i) => [col.s[i], col.v[i]]), want, name);
      assert.ok(col.sorted[0] && [...col.w.subarray(0, col.n)].every((w) => w === 1), name);
    }
  }
});

test("a chunk whose columns are all freed is told of, and its floats go to later columns under its next generation", () => {
  const drops = [], whole = (loc) => K.chunkView(loc.chunk).length;
  K.onDrops((id, gen) => drops.push([id, gen]));
  const before = K.columnStore(1), filler = K.columnStore(whole(before.loc)); // the columns after these begin a chunk
  const a = K.columnStore(3), b = K.columnStore(5);
  assert.deepEqual([b.loc.chunk, b.loc.gen, b.loc.off, a.loc.size, b.loc.size], [a.loc.chunk, a.loc.gen, a.loc.off + 3, 3, 5]);
  a.view.set([1, 2, 3]);
  assert.deepEqual([...K.chunkView(a.loc.chunk).subarray(a.loc.off, a.loc.off + 3)], [1, 2, 3]);
  const next = K.columnStore(whole(a.loc)); // more than the chunk has left: another takes the columns from here on
  assert.notEqual(next.loc.chunk, a.loc.chunk);
  K.freeStore(a.loc);
  assert.deepEqual(drops, []);
  K.freeStore(b.loc);
  assert.deepEqual(drops, [[a.loc.chunk, a.loc.gen]]);
  const again = K.columnStore(whole(a.loc));
  assert.deepEqual(again.loc, { chunk: a.loc.chunk, gen: a.loc.gen + 1, off: 0, size: whole(a.loc) });
  for (const x of [before, filler, next, again]) K.freeStore(x.loc);
});

test("a column bins by runtime the same in whatever order its runtimes come", () => {
  // [runtime, value, rows it stands for]: bins of one finite mean, of infinities alone, of both signs, and empty ones
  const points = [[5.5, 1, 2], [0.5, 4, 1], [5.6, 3, 2], [0.6, Infinity, 1], [2.5, Infinity, 1], [9.5, 7, 1], [2.6, -Infinity, 3],
                  [3.5, Infinity, 1], [1.5, NaN, 1], [0.7, 8, 3], [9.1, 9, 1]];
  const of = (pts) => K.Col.adopt(Float64Array.from(pts, (_, i) => i), Float64Array.from(pts, (q) => q[1]), Float64Array.from(pts, (q) => q[0]),
                                  pts.length, Float64Array.from(pts, (q) => q[2]));
  const p = { xmode: K.X_RUNTIME, x0: 0, x1: 10, bins: 10, flags: 0, alpha: 0, scale: 1 };
  const resumed = of(points), inOrder = of([...points].sort((a, b) => a[0] - b[0]));
  assert.deepEqual([resumed.sorted[K.X_RUNTIME], inOrder.sorted[K.X_RUNTIME]], [false, true]);
  assert.deepEqual([...K.binRows([resumed], p)], [7, NaN, NaN, Infinity, Infinity, 2, 3.5, 5, 6.5, 8]);
  assert.deepEqual([...K.binRows([resumed], p)], [...K.binRows([inOrder], p)]);
});

test("the point nearest a runtime is found in whatever order the runtimes come", () => {
  const v = [10, 11, 12, 13, 14], t = [7, 2, 9, 4, 1], c = column([0, 1, 2, 3, 4], v, t);
  assert.equal(c.len, 5);
  for (const [x, i] of [[0, 4], [2.9, 1], [3.1, 3], [8.1, 2], [100, 2]]) {
    assert.deepEqual(K.nearest(c, K.X_RUNTIME, x, 0, 1), { i, x: t[i], y: v[i], raw: v[i] }, `runtime ${x}`);
  }
});

test("a kept binning takes the columns it is given later", () => {
  const cols = Array.from({ length: 5 }, (_, k) => column([0.5, 1.5], [k, 2 * k]));
  const p = { xmode: K.X_STEP, x0: 0, x1: 2, bins: 2, flags: 0, alpha: 0, scale: 1 }, cache = new K.BinCache();
  assert.deepEqual([...K.binRows(cols.slice(0, 1), p, cache)], [0, 0]);
  assert.deepEqual([...K.binRows(cols, p, cache)], cols.flatMap((_, k) => [k, 2 * k]));
});

test("rows streamed at steps no block shown holds are all kept, bucketed at the chart's level", () => {
  const block = fullBlock(2, 0, (q) => q); // steps [0, 1024) in buckets of 4
  const tail = { s: [2000, 2001, 2003, 2004, 2100], v: [1, 3, 5, 7, 9], t: [10, 20, 30, 40, 50], q: [0, 1, 2, 3, 4], n: 5 };
  const col = K.buildColumn([block], tail, 2, false), last = (a) => [...a.subarray(col.n - 3, col.n)];
  assert.equal(col.n, 256 + 3);
  assert.deepEqual([last(col.s), last(col.v), last(col.t), last(col.w)], [[6004 / 3, 2004, 2100], [3, 7, 9], [20, 40, 50], [3, 1, 1]]);
});
