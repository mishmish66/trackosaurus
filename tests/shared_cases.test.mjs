// The browser's side of the formats it shares with Python, against tests/shared_cases.json (written by
// tests/test_shared_cases.py from the Python implementations).
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { decodeTile } from "../trex/static/data.js";
import { Col, IQM, STATS, X_STEP, agg } from "../trex/static/kernel.js";
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
