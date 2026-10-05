import test from "node:test";
import assert from "node:assert/strict";
import { Data } from "../trex/static/data.js";

const UI = { status() {}, data() {}, runs() {}, keys() {}, media() {}, idle() {}, conn() {}, protocol() {} };

/** A bucket array of no buckets naming runs `paths`, holding rows `seqs` of them (buckets.encode). */
function emptyArray(level, index, paths, seqs) {
  const names = new TextEncoder().encode(paths.join("\0")), pad = (n) => Math.ceil(n / 8) * 8, runs = paths.length;
  const buf = new ArrayBuffer(32 + pad(names.length) + pad(4 * (runs + 1)) + pad(4 * runs));
  const h = new Uint32Array(buf, 0, 8);
  h.set([0x31424b54, level >>> 0, index, 0, runs, 0, names.length, 0]);
  new Uint8Array(buf, 32, names.length).set(names);
  new Uint32Array(buf, 32 + pad(names.length) + pad(4 * (runs + 1)), runs).set(seqs);
  return { buf, bytes: buf.byteLength, paths, ext: null };
}

function withRuns(n, running = 0) {
  const d = new Data(UI);
  d.fetchMany = async (xs) => xs.map(() => null); // no server
  const runs = Array.from({ length: n }, (_, i) => d.newRun({ id: `r${i}`, seq: 10, mseq: 0, kept_seq: 10, keys: ["loss"], summary: { _step: 1000 },
                                                             state: i < running ? "running" : "finished" }));
  return { d, runs };
}

test("many finished runs lacking a block are asked for in one request for the scope, running ones by id", () => {
  const { d, runs } = withRuns(100, 3), queue = [];
  d.need("loss", 2, 0, runs, queue);
  assert.deepEqual(queue.map((x) => (x.runs === null ? "scope" : x.runs.map((r) => r.id))), ["scope", ["r0", "r1", "r2"]]);
});

test("a few runs lacking a block are asked for by id, and none that the block holds as they now are", () => {
  const { d, runs } = withRuns(10, 1), queue = [];
  d.addArray({ key: "loss", level: 2, index: 0 }, emptyArray(2, 0, ["r0", "r5", "r6"], [9, 10, 10]));
  d.need("loss", 2, 0, runs, queue);
  assert.deepEqual(queue.map((x) => x.runs.map((r) => r.id)), [["r0", "r1", "r2", "r3", "r4", "r7", "r8", "r9"]]);
});

test("a chart shows the layers it wants only once every block holds every one of its runs", () => {
  const { d, runs } = withRuns(2);
  d.plan([{ key: "loss", runs, runsSig: "a", xmode: 0, zoomed: false, x0: -Infinity, x1: Infinity, pw: 600, many: false }]);
  const ch = d.charts.get("loss"), { level, indices } = ch.want.coarse;
  assert.equal(ch.ready, null);
  d.addArray({ key: "loss", level, index: indices[0] }, emptyArray(level, indices[0], ["r0"], [10]));
  d.settleChart("loss", ch);
  assert.equal(ch.ready, null);
  for (const index of indices) d.addArray({ key: "loss", level, index }, emptyArray(level, index, ["r0", "r1"], [10, 10]));
  d.settleChart("loss", ch);
  assert.equal(ch.ready, ch.want);
});

test("an array is freed once no block entry names it, and a dropped run leaves every block", () => {
  const { d, runs } = withRuns(2);
  d.addArray({ key: "loss", level: 0, index: 0 }, emptyArray(0, 0, ["r0", "r1"], [10, 10]));
  d.addArray({ key: "loss", level: 0, index: 0 }, emptyArray(0, 0, ["r1", "gone"], [12, 3]));
  assert.equal(d.arrays.size, 2);
  d.dropRun(runs[0]);
  assert.equal(d.arrays.size, 1);
  assert.deepEqual([...d.blocks.get("loss|0|0").runs.keys()], ["r1"]);
});

test("a tail of 200000 rows gives every row of the metric with its sequence number", () => {
  const tail = Array.from({ length: 200000 }, (_, i) => [i + 5, i, i % 2 ? { x: i } : {}]);
  const got = new Data(UI).tailOf({ tail, tailSeq0: 7 }, "x", 9);
  assert.deepEqual([got.n, got.s[0], got.q[0], got.q[got.n - 1]], [99999, 8, 10, 200006]);
});

test("queued requests go out spread over the free request slots, one batch each", () => {
  const { d, runs } = withRuns(4), sent = [];
  d.fetchMany = (xs) => (sent.push(xs.length), new Promise(() => {}));
  for (let index = 0; index < 22; index++) d.need("loss", 0, index, runs, d.queue);
  d.pump();
  assert.deepEqual([sent.length, sent.reduce((a, b) => a + b, 0), d.queue.length], [5, 22, 0]);
});

test("fetching ahead asks for every chart's wanted layers before finer levels of the charts shown", () => {
  const demands = [];
  const d = new Data({ ...UI, ahead: () => demands });
  d.fetchMany = async (xs) => xs.map(() => null);
  const runs = Array.from({ length: 3 }, (_, i) => d.newRun({ id: `r${i}`, seq: 10, mseq: 0, kept_seq: 10, keys: ["loss", "acc"],
                                                               summary: { _step: 1000 }, state: "finished" }));
  const demand = (key) => ({ key, runs, runsSig: "a", xmode: 0, zoomed: false, x0: -Infinity, x1: Infinity, pw: 600, many: false });
  demands.push(demand("loss"), demand("acc"));
  d.plan([demands[0]]);
  const ch = d.charts.get("loss"), { level, indices } = ch.want.coarse;
  for (const index of indices) d.addArray({ key: "loss", level, index }, emptyArray(level, index, ["r0", "r1", "r2"], [10, 10, 10]));
  d.settleChart("loss", ch);
  const asks = d.nextAhead(100), finer = asks.findIndex((x) => x.key === "loss");
  assert.ok(asks.some((x) => x.key === "acc") && finer > asks.findLastIndex((x) => x.key === "acc"));
  assert.ok(asks.filter((x) => x.key === "loss").every((x) => x.level < level));
});
