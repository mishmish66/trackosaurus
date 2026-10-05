// The browser's side of the formats it shares with Python, against tests/shared_cases.json (written by
// tests/test_shared_cases.py from the Python implementations).
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { Col, IQM, STATS, X_STEP, aggGroups, binRows, bucketPaths, bucketStep, bucketViews, buildColumn, unframe } from "../trex/static/kernel.js";
import { smoothScale } from "../trex/static/plot.js";
import { asNumber } from "../trex/static/where.js";

const CASES = JSON.parse(readFileSync(new URL("./shared_cases.json", import.meta.url), "utf8"));
/** A case number: null or "nan" for NaN, "inf" and "-inf" for the infinities. */
const nan = (v) => (v === null ? NaN : typeof v === "string" ? asNumber(v) : v);

function close(got, want, what) {
  if (Object.is(got, want)) return;
  if (Number.isNaN(want)) return assert.ok(Number.isNaN(got), `${what}: ${got} is not NaN`);
  assert.ok(Math.abs(got - want) <= 1e-12 * Math.max(1, Math.abs(want)), `${what}: ${got} != ${want}`);
}

/** Views of a case's base64 bucket array, in a buffer of its own. */
const viewsOf = (b64) => {
  const buf = Uint8Array.from(Buffer.from(b64, "base64")).buffer;
  return { buf, v: bucketViews(buf) };
};
const NO_TAIL = { s: [], v: [], t: [], q: [], n: 0 };

function sameColumn(got, want, at, tol) {
  assert.equal(got.n, want.n, `${at}: points`);
  for (let i = 0; i < want.n; i++) {
    const w = want.v[i];
    if (!Number.isFinite(w)) assert.ok(Object.is(got.v[i], w), `${at} point ${i}: ${got.v[i]} != ${w}`);
    else assert.ok(Math.abs(got.v[i] - w) <= 1e-5 * Math.max(1, Math.abs(w)), `${at} point ${i}: ${got.v[i]} != ${w}`);
    assert.ok(Math.abs(got.s[i] - want.s[i]) <= tol, `${at} point ${i} step: ${got.s[i]} != ${want.s[i]}`);
    assert.equal(got.w[i], want.w[i], `${at} point ${i} count`);
  }
}

test("a framed body unframes to the bucket arrays Python framed, an empty one included", () => {
  const buf = Uint8Array.from(Buffer.from(CASES.frame.body, "base64")).buffer;
  const got = unframe(buf).map(({ off, len }) => Buffer.from(new Uint8Array(buf, off, len)).toString("base64"));
  assert.deepEqual(got, CASES.frame.parts);
});

test("bucket arrays decode to the runs and buckets the Python encoder wrote, f32 fields exactly", () => {
  for (const c of CASES.arrays) {
    const { buf, v } = viewsOf(c.blob), at = `array ${c.level}|${c.index}`;
    assert.deepEqual([v.level, v.base, v.runs, bucketPaths(buf, v), [...v.seq], [...v.first]], [c.level, c.index, c.paths.length, c.paths, c.seq, c.first], at);
    for (let q = 0; q < v.count; q++) {
      assert.deepEqual([v.offset[q], v.soff[q], v.n[q]], [c.offset[q], c.soff[q], c.n[q]], `${at} bucket ${q}`);
      ["mean", "tmean"].forEach((f) => assert.equal(v[f][q], Math.fround(nan(c[f][q])), `${at} bucket ${q} ${f}`));
      close(bucketStep(v, q), nan(c.mean_step[q]), `${at} bucket ${q} mean step`);
    }
  }
});

test("a live run's column is the same before and after its buckets take in its newest rows", () => {
  for (const c of CASES.tails) {
    const tail = { s: c.steps, v: c.values.map(nan), t: c.times, q: c.steps.map((_, i) => c.seq0 + i), n: c.steps.length };
    const got = buildColumn([{ v: viewsOf(c.before).v, row: 0 }], tail, c.level, false);
    const want = buildColumn([{ v: viewsOf(c.after).v, row: 0 }], NO_TAIL, c.level, false);
    sameColumn(got, want, `level ${c.level}`, 2 ** c.level / 65536);
  }
});

test("a column takes a finer level's buckets where its blocks lie and a coarser level's elsewhere", () => {
  for (const c of CASES.layers) {
    const parts = [...c.coarse, ...c.fine].map((b) => ({ v: viewsOf(b).v, row: 0 }));
    const col = buildColumn(parts, NO_TAIL, undefined, false);
    assert.equal(col.n, c.steps.length);
    c.steps.forEach((x, i) => {
      assert.ok(Math.abs(col.s[i] - x) <= 1e-9 * Math.max(1, x) + 16 / 65536, `point ${i} step: ${col.s[i]} != ${x}`);
      const w = nan(c.means[i]);
      assert.ok(Number.isFinite(w) ? col.v[i] === Math.fround(w) : Object.is(col.v[i], w), `point ${i}: ${col.v[i]} != ${w}`);
    });
  }
});

test("runs' buckets bin to the mean of their rows per bin, as Python bins the rows", () => {
  for (const c of CASES.bins) {
    const { v } = viewsOf(c.blob);
    const cols = c.means.map((_, i) => buildColumn([{ v, row: i }], NO_TAIL, v.level, false));
    const rows = binRows(cols, { xmode: X_STEP, x0: c.x0, x1: c.x1, bins: c.bins, flags: 0, alpha: 0, scale: 1 });
    c.means.forEach((want, i) => want.forEach((m, b) => {
      if (m === null) return;
      const got = rows[i * c.bins + b], w = nan(m), at = `run ${i} bin ${b} of ${c.bins}`;
      if (!Number.isFinite(w)) return assert.ok(Object.is(got, w), `${at}: ${got} != ${w}`);
      assert.ok(Math.abs(got - w) <= 1e-5 * Math.max(1, Math.abs(w)), `${at}: ${got} != ${w}`);
    }));
  }
});

test("smoothing matches the Python time-weighted EMA, gaps passing through", () => {
  for (const c of CASES.smoothing) {
    assert.equal(smoothScale(c.span), c.scale);
    const col = Col.adopt(Float64Array.from(c.xs), Float64Array.from(c.ys.map(nan)), Float64Array.from(c.xs), c.xs.length);
    col.ensureSmooth(c.alpha, c.scale, X_STEP);
    c.smoothed.forEach((want, i) => close(col.sm[i], nan(want), `alpha ${c.alpha} point ${i}`));
  }
});

test("group statistics of one bin match the Python ones", () => {
  const row = Object.fromEntries(STATS.map((k, i) => [k, i]));
  for (const c of CASES.stats) {
    const cols = c.values.map((v) => Col.adopt(Float64Array.of(0), Float64Array.of(nan(v)), Float64Array.of(0), 1));
    const out = aggGroups([cols], X_STEP, -1, 1, 1, IQM, 0, 1), at = `${c.values.length} values`;
    assert.equal(out[row.n], c.n, at);
    if (!c.n) continue;
    for (const k of ["mean", "std", "median", "min", "max", "iqm"]) close(out[row[k]], nan(c[k]), `${at} ${k}`);
    close(out[row.iqmse], nan(c.iqm_se), `${at} iqm se`);
    assert.deepEqual([out[row.medlo], out[row.medhi], out[row.iqmh]], [...c.median_ci.map(nan), c.iqm_kept], at);
  }
});
