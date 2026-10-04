// The browser's side of the formats it shares with Python, against tests/shared_cases.json (written by
// tests/test_shared_cases.py from the Python implementations).
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { buildColumn, decodeTile, finePoints } from "../trex/static/data.js";
import { Col, IQM, STATS, SlabRun, X_STEP, agg, binRows, slabViews } from "../trex/static/kernel.js";
import { smoothScale } from "../trex/static/plot.js";
import { asNumber } from "../trex/static/where.js";

const CASES = JSON.parse(readFileSync(new URL("./shared_cases.json", import.meta.url), "utf8"));
const TILE = 256;
/** A case number: null or "nan" for NaN, "inf" and "-inf" for the infinities. */
const nan = (v) => (v === null ? NaN : typeof v === "string" ? asNumber(v) : v);

function close(got, want, what) {
  if (Object.is(got, want)) return;
  if (Number.isNaN(want)) return assert.ok(Number.isNaN(got), `${what}: ${got} is not NaN`);
  assert.ok(Math.abs(got - want) <= 1e-12 * Math.max(1, Math.abs(want)), `${what}: ${got} != ${want}`);
}

test("tiles decode to the buckets the Python encoder wrote, f32 fields exactly", () => {
  for (const c of CASES.tiles) {
    const bytes = Uint8Array.from(Buffer.from(c.blob, "base64"));
    const t = decodeTile(bytes.buffer);
    assert.deepEqual([t.level, t.index, t.count], [c.level, c.index, c.count]);
    const k = t.count;
    for (let i = 0; i < k; i++) {
      const at = `tile ${c.level}|${c.index} bucket ${i}`;
      assert.equal(t.u16[t.b + i], c.bucket[i], at);
      ["min", "max", "mean", "tmean", "soff"].forEach((f, j) => assert.equal(t.f32[t.f + j * k + i], Math.fround(nan(c[f][i])), `${at} ${f}`));
      assert.equal(t.u32[t.f + 5 * k + i], c.n[i], `${at} n`);
      close((t.index * TILE + t.u16[t.b + i] + t.f32[t.f + 4 * k + i]) * 2 ** t.level, c.mean_step[i], `${at} mean step`);
    }
  }
});

test("buckets merge locally as tiles.coarsen merges them: the mean of the finite ones, infinite only without any", () => {
  const none = { s: [], v: [], t: [], n: 0, s0: Infinity };
  for (const c of CASES.merges) {
    const tile = decodeTile(Uint8Array.from(Buffer.from(c.blob, "base64")).buffer);
    const col = buildColumn([tile], 2 ** c.up, 2 ** c.level, none, [], none);
    assert.equal(col.n, c.mean.length, `up ${c.up}`);
    c.mean.forEach((m, i) => {
      const want = nan(m), at = `up ${c.up} bucket ${i}`;
      if (!Number.isFinite(want)) return assert.ok(Object.is(col.v[i], want), `${at}: ${col.v[i]} != ${want}`);
      assert.ok(Math.abs(col.v[i] - want) <= 1e-6 * Math.max(1, Math.abs(want)), `${at}: ${col.v[i]} != ${want}`);
      assert.ok(Math.abs(col.s[i] - nan(c.mean_step[i])) <= 1e-3 * 2 ** (c.level + c.up), `${at} step`);
    });
  }
});

test("a live run's column is the same before and after its tiles take in its newest rows", () => {
  const none = { s: [], v: [], t: [], n: 0, s0: Infinity }, tile = (b64) => decodeTile(Uint8Array.from(Buffer.from(b64, "base64")).buffer);
  for (const c of CASES.tails) {
    const tail = { s: c.steps, v: c.values.map(nan), t: c.times, n: c.steps.length, s0: c.steps[0] }, at = `up ${c.up} fine ${c.fine}`;
    const range = [c.index * TILE * 2 ** c.level, (c.index + 1) * TILE * 2 ** c.level];
    const column = (t, rows, s0) => (c.fine
      ? buildColumn([], 1, 1, finePoints([{ tile: t, seq: 0 }], s0), [range], rows, 2 ** c.level)
      : buildColumn([t], 2 ** c.up, 2 ** c.level, none, [], rows));
    const got = column(tile(c.before), tail, tail.s0), want = column(tile(c.after), none, Infinity);
    assert.equal(got.n, want.n, at);
    for (let i = 0; i < want.n; i++) {
      const w = want.v[i];
      if (!Number.isFinite(w)) assert.ok(Object.is(got.v[i], w), `${at} point ${i}: ${got.v[i]} != ${w}`);
      else assert.ok(Math.abs(got.v[i] - w) <= 1e-5 * Math.max(1, Math.abs(w)), `${at} point ${i}: ${got.v[i]} != ${w}`);
      assert.ok(Math.abs(got.s[i] - want.s[i]) <= 1e-3 * 2 ** (c.level + c.up), `${at} point ${i} step`);
    }
  }
});

test("a slab's runs bin to the mean of their rows per bucket, as Python bins the rows", () => {
  for (const c of CASES.slabs) {
    const bytes = Uint8Array.from(Buffer.from(c.slab, "base64")), sl = slabViews(bytes.buffer, 0);
    assert.deepEqual([sl.level, sl.index, sl.runs], [c.level, c.index, c.means.length]);
    const runs = c.means.map((_, i) => new SlabRun([sl], [i]));
    const rows = binRows(runs, { xmode: X_STEP, x0: c.x0, x1: c.x1, bins: c.bins, flags: 0, alpha: 0, scale: 1 });
    c.means.forEach((want, i) => want.forEach((m, b) => {
      if (m === null) return;
      const got = rows[i * c.bins + b], w = nan(m), at = `run ${i} bin ${b}`;
      if (!Number.isFinite(w)) return assert.ok(Object.is(got, w), `${at}: ${got} != ${w}`);
      assert.ok(Math.abs(got - w) <= 1e-5 * Math.max(1, Math.abs(w)), `${at}: ${got} != ${w}`);
    }));
  }
});

test("smoothing matches the Python time-weighted EMA, gaps passing through", () => {
  for (const c of CASES.smoothing) {
    assert.equal(smoothScale(c.span), c.scale);
    const col = new Col();
    col.push(c.xs, c.ys.map(nan), c.xs);
    col.ensureSmooth(c.alpha, c.scale, X_STEP);
    c.smoothed.forEach((want, i) => close(col.sm[i], nan(want), `alpha ${c.alpha} point ${i}`));
  }
});

test("group statistics of one bin match the Python ones", () => {
  const row = Object.fromEntries(STATS.map((k, i) => [k, i]));
  for (const c of CASES.stats) {
    const cols = c.values.map((v) => {
      const col = new Col();
      col.push([0], [nan(v)], [0]);
      return col;
    });
    const out = agg(cols, X_STEP, -1, 1, 1, IQM, 0, 1), at = `${c.values.length} values`;
    assert.equal(out[row.n], c.n, at);
    if (!c.n) continue;
    for (const k of ["mean", "std", "median", "min", "max", "iqm"]) close(out[row[k]], nan(c[k]), `${at} ${k}`);
    close(out[row.iqmse], nan(c.iqm_se), `${at} iqm se`);
    assert.deepEqual([out[row.medlo], out[row.medhi], out[row.iqmh]], [...c.median_ci.map(nan), c.iqm_kept], at);
  }
});
