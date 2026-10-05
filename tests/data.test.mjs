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
  const runs = Array.from({ length: n }, (_, i) => d.newRun({ id: `r${i}`, seq: 10, mseq: 0, compiled: 10, keys: ["loss"], summary: { _step: 1000 },
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
  const runs = Array.from({ length: 3 }, (_, i) => d.newRun({ id: `r${i}`, seq: 10, mseq: 0, compiled: 10, keys: ["loss", "acc"],
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

test("a run that appears while the scope is being listed reaches the page through the stream it opened first", async () => {
  const sources = [], meta = (id) => ({ id, seq: 1, mseq: 0, compiled: 1, keys: [], summary: {}, state: "finished" });
  const realFetch = globalThis.fetch;
  let answer;
  globalThis.EventSource = class {
    constructor(url) { this.url = url; this.on = {}; sources.push(this); }
    addEventListener(kind, fn) { this.on[kind] = fn; }
    close() {}
  };
  globalThis.fetch = () => new Promise((ok) => (answer = () => ok({ ok: true, json: async () => ({ runs: [meta("a")], media: [], folders: {} }) })));
  try {
    const d = new Data(UI), loading = d.loadScope("");
    assert.equal(sources.length, 1);
    sources[0].on.run({ data: JSON.stringify(meta("b")) });
    answer();
    await loading;
    assert.deepEqual([...d.runs.keys()].sort(), ["a", "b"]);
    sources[0].on.run({ data: JSON.stringify(meta("c")) });
    assert.deepEqual([...d.runs.keys()].sort(), ["a", "b", "c"]);
  } finally {
    delete globalThis.EventSource;
    globalThis.fetch = realFetch;
  }
});

/** Let queued tasks (column rebuilds, request answers) run. */
const tasks = () => new Promise((ok) => setTimeout(ok, 5));
const demandOf = (runs, zoom = null) => ({ key: "loss", runs, runsSig: "a", xmode: 0, zoomed: !!zoom, x0: zoom ? zoom[0] : -Infinity,
                                         x1: zoom ? zoom[1] : Infinity, pw: 600, many: false });

/** A Data of n finished runs whose chart of `loss` shows its coarse layer, and what its UI was told of: {d, runs, told}. */
async function shown(n) {
  const told = [], d = new Data({ ...UI, data: (keys) => told.push([...keys]) });
  d.fetchMany = async (xs) => xs.map((x) => emptyArray(x.level, x.index, runs.map((r) => r.id), runs.map(() => 10)));
  const runs = Array.from({ length: n }, (_, i) => d.newRun({ id: `r${i}`, seq: 10, mseq: 0, compiled: 10, keys: ["loss"], summary: { _step: 1000 },
                                                             state: "finished" }));
  d.plan([demandOf(runs)]);
  await tasks();
  return { d, runs, told };
}

test("a chart's columns are built once its layers are here, and again only for the runs whose blocks change", async () => {
  const { d, runs } = await shown(3), ch = d.charts.get("loss"), { level, indices } = ch.ready.coarse;
  const built = runs.map((r) => r.cols.get("loss"));
  assert.ok(built.every(Boolean));
  d.plan([{ ...demandOf(runs), runsSig: "b" }]);
  await tasks();
  assert.deepEqual(runs.map((r) => r.cols.get("loss") === built[runs.indexOf(r)]), [true, true, true]);
  d.take({ key: "loss", level, index: indices[0] }, emptyArray(level, indices[0], ["r1"], [10]));
  await tasks();
  assert.deepEqual(runs.map((r) => r.cols.get("loss") === built[runs.indexOf(r)]), [true, false, true]);
});

test("a run left out while its chart's layers changed gets a column of the layers shown once it is back", async () => {
  const { d, runs } = await shown(2), zoom = [100, 120], fine = d.layersOf(demandOf(runs, zoom), runs).fine;
  for (const index of fine.indices) d.take({ key: "loss", level: fine.level, index }, emptyArray(fine.level, index, ["r0", "r1"], [10, 10])); // fetched ahead
  const coarse = runs[1].cols.get("loss");
  d.plan([demandOf(runs.slice(0, 1), zoom)]);
  await tasks();
  assert.deepEqual([d.charts.get("loss").ready.fine, runs[1].cols.get("loss") === coarse], [fine, true]);
  d.plan([demandOf(runs, zoom)]);
  await tasks();
  assert.ok(runs[1].cols.get("loss") !== coarse);
});

test("the UI hears of a metric once none of its columns awaits rebuilding", async () => {
  const { d, runs, told } = await shown(2);
  told.length = 0;
  for (const r of runs) d.rebuildSoon(r, "loss");
  d.touched.add("loss");
  d.flush();
  assert.deepEqual([told, d.pending("loss")], [[], true]);
  await tasks();
  assert.deepEqual([told, d.pending("loss"), d.busy], [[["loss"]], false, false]);
});

test("rows streamed for a run are told of at once, whatever awaits rebuilding", async () => {
  const { d, runs, told } = await shown(2);
  told.length = 0;
  d.rebuildSoon(runs[1], "loss");
  d.onRows(runs[0], { run: "r0", seq0: 10, rows: [[1001, 5, { loss: 1 }]] });
  assert.deepEqual(told, [["loss"]]);
});

test("the blocks of a zoom being dragged are fetched before it is set, once", async () => {
  const { d, runs } = await shown(3), asked = [], answer = d.fetchMany;
  d.fetchMany = (xs) => (asked.push(...xs.map((x) => `${x.level}|${x.index}`)), answer(xs));
  const zoom = demandOf(runs, [100, 120]), fine = d.layersOf(zoom, runs).fine;
  assert.ok(fine && fine.level < d.charts.get("loss").ready.coarse.level);
  d.fetchFor([zoom]);
  d.fetchFor([zoom]);
  assert.deepEqual(asked, fine.indices.map((i) => `${fine.level}|${i}`));
  await tasks();
  assert.equal(d.plan([zoom]), 0);
  assert.deepEqual([asked.length, d.charts.get("loss").ready.fine], [fine.indices.length, fine]);
  await tasks();
  assert.ok(runs.every((r) => r.cols.has("loss")));
});

test("columns queued behind one whose rebuild throws are still rebuilt", async () => {
  const { d, runs } = await shown(2), rebuild = d.rebuild.bind(d), errors = [], built = [], timer = globalThis.setTimeout;
  d.rebuild = (r, key) => {
    if (r === runs[0]) throw new Error("bad column");
    built.push(r.id);
    rebuild(r, key);
  };
  globalThis.setTimeout = (f, ms) => timer(() => {
    try {
      f();
    } catch (e) {
      errors.push(e.message);
    }
  }, ms);
  try {
    for (const r of runs) d.rebuildSoon(r, "loss");
    await tasks();
    await tasks();
  } finally {
    globalThis.setTimeout = timer;
  }
  assert.deepEqual([errors, built, d.pending("loss"), d.rebuilding], [["bad column"], ["r1"], false, false]);
});
