import test from "node:test";
import assert from "node:assert/strict";
import { Data, askId } from "../trex/static/data.js";

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

/** A bucket array of block `index` naming runs `paths`, each with the buckets `rows` gives it ([offset in the block,
 * value] each, one row a bucket), holding rows `seqs` of them (buckets.encode). */
function bucketArray(level, index, paths, seqs, rows) {
  const names = new TextEncoder().encode(paths.join("\0")), pad = (n) => Math.ceil(n / 8) * 8, runs = paths.length, all = rows.flat(), count = all.length;
  const first = 32 + pad(names.length), seq = first + pad(4 * (runs + 1)), offset = seq + pad(4 * runs), mean = offset + 2 * pad(2 * count);
  const buf = new ArrayBuffer(mean + 12 * count);
  new Uint32Array(buf, 0, 8).set([0x31424b54, level >>> 0, index, 0, runs, count, names.length, 0]);
  new Uint8Array(buf, 32, names.length).set(names);
  new Uint32Array(buf, first, runs + 1).set(rows.reduce((f, r) => [...f, f.at(-1) + r.length], [0]));
  new Uint32Array(buf, seq, runs).set(seqs);
  new Uint16Array(buf, offset, count).set(all.map((q) => q[0]));
  new Float32Array(buf, mean, count).set(all.map((q) => q[1]));
  new Uint32Array(buf, mean + 8 * count, count).fill(1);
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

test("a folder shown from the runs loaded asks for its own finished runs' blocks; a request on its way covers only that folder's", () => {
  const d = new Data(UI), queue = [], again = [], other = [];
  d.fetchMany = async (xs) => xs.map(() => null);
  const runs = ["a", "b"].flatMap((dir) => Array.from({ length: 100 }, (_, i) => d.newRun({ id: `${dir}/r${i}`, seq: 10, mseq: 0, compiled: 10, keys: ["loss"],
                                                                                       summary: {}, state: "finished" })));
  d.view = "a";
  d.need("loss", 2, 0, runs.slice(0, 100), queue);
  assert.deepEqual(queue.map((x) => [x.runs, x.scope]), [[null, "a"]]);
  assert.ok(d.asking(queue[0]), "asked");
  d.need("loss", 2, 0, runs.slice(0, 100), again);
  assert.deepEqual(again, [], "on its way");
  d.view = "b";
  d.need("loss", 2, 0, runs.slice(100), other);
  assert.deepEqual(other.map((x) => [x.runs, x.scope]), [[null, "b"]], "the other folder's runs are not");
});

test("metrics the same runs log share one list of them, which a chart's plan is kept for", () => {
  const d = new Data(UI);
  d.fetchMany = async (xs) => xs.map(() => null);
  const runs = Array.from({ length: 4 }, (_, i) => d.newRun({ id: `r${i}`, seq: 10, mseq: 0, compiled: 10, keys: i < 3 ? ["loss", "acc", "lr"] : ["lr"],
                                                               summary: { _step: 1000 }, state: "finished" }));
  assert.equal(d.runsWith(runs, "loss"), d.runsWith(runs, "acc"));
  assert.deepEqual(d.runsWith(runs, "loss").map((r) => r.id), ["r0", "r1", "r2"]);
  assert.equal(d.runsWith(runs, "lr"), runs, "every run logs it: the list itself");
  assert.ok(d.someLog(runs.slice(3), "lr") && !d.someLog(runs.slice(3), "loss"));
});

test("a chart that shows all its view wants is not planned again until its data or its view changes", () => {
  const { d, runs } = withRuns(3), demand = (pw) => ({ key: "loss", runs, runsSig: "a", xmode: 0, zoomed: false, x0: -Infinity, x1: Infinity, pw, many: false });
  let asked = 0;
  const need = d.need.bind(d);
  d.need = (...a) => (asked++, need(...a));
  d.planChart(demand(600), []);
  const ch = d.charts.get("loss"), { level, indices } = ch.want.coarse;
  for (const index of indices) d.addArray({ key: "loss", level, index }, emptyArray(level, index, ["r0", "r1", "r2"], [10, 10, 10]));
  d.planChart(demand(600), []);
  d.rebuildNow();
  d.planChart(demand(600), []);
  const settled = asked;
  d.planChart(demand(600), []);
  assert.equal(asked, settled, "nothing changed: no block is looked at");
  d.planChart(demand(2400), []);
  assert.ok(asked > settled, "a wider chart is planned anew");
  const wide = asked;
  d.setMeta(runs[0], { ...runs[0].meta, compiled: 11 });
  d.planChart(demand(2400), []);
  assert.ok(asked > wide, "a run's levels grew: its blocks are looked at again");
});

test("the x extent of a chart's runs spans their buckets in the blocks it shows, found run by run or from their arrays' rows", () => {
  const { d, runs } = withRuns(40);
  d.plan([{ key: "loss", runs, runsSig: "a", xmode: 0, zoomed: false, x0: -Infinity, x1: Infinity, pw: 600, many: false }]);
  const ch = d.charts.get("loss"), { level, indices } = ch.want.coarse, ids = runs.map((r) => r.id), seqs = runs.map(() => 10);
  assert.equal(d.extentOf(runs, "loss"), null, "before it shows any block");
  // run i has two buckets in the first block, i and i + 2 buckets in: the first runs' lie inside the others'
  d.addArray({ key: "loss", level, index: indices[0] }, bucketArray(level, indices[0], ids, seqs, runs.map((_, i) => [[i, 1], [i + 2, 2]])));
  for (const index of indices.slice(1)) d.addArray({ key: "loss", level, index }, emptyArray(level, index, ids, seqs));
  d.settleChart("loss", ch);
  const step = (offset) => (indices[0] * 256 + offset + 0.5 / 65536) * 2 ** level;
  assert.deepEqual(d.extentOf(runs, "loss"), [step(0), step(41)], "of many runs");
  assert.deepEqual(d.extentOf(runs.slice(3, 6), "loss"), [step(3), step(7)], "of a few");
  assert.deepEqual(d.extentOf(runs.slice(3, 38), "loss"), [step(3), step(39)], "of many of them");
});

test("the queued columns of the metrics asked for are rebuilt at once, the others' in a task", async () => {
  const d = new Data(UI), keys = ["acc", "loss"];
  const runs = [0, 1].map((i) => d.newRun({ id: `r${i}`, seq: 10, mseq: 0, compiled: 10, keys, summary: { _step: 1000 }, state: "finished" }));
  d.fetchMany = async (xs) => xs.map((x) => emptyArray(x.level, x.index, ["r0", "r1"], [10, 10]));
  d.plan(keys.map((key) => ({ ...demandOf(runs), key })));
  await tasks();
  for (const key of keys) for (const r of runs) d.rebuildSoon(r, key);
  d.rebuildNow(new Set(["loss"]));
  assert.deepEqual([d.pending("loss"), d.pending("acc")], [false, true]);
  await tasks();
  assert.deepEqual([d.pending("acc"), d.busy], [false, false]);
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

test("fetching ahead asks for every chart's wanted layers before the finer levels a zoom of any would want", () => {
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
  const asks = d.nextAhead(100, Infinity), want = d.layersOf(demands[1], runs).coarse.level; // of "acc", not shown yet
  const wanted = asks.findLastIndex((x) => x.key === "acc" && x.level === want), finer = asks.findIndex((x) => x.level < (x.key === "loss" ? level : want));
  assert.ok(wanted >= 0 && finer > wanted);
  assert.ok(asks.filter((x) => x.key === "loss").every((x) => x.level < level), "the shown chart's layers are here: only finer ones are asked for");
  assert.ok(asks.some((x) => x.key === "acc" && x.level < want), "a chart not shown yet gets its finer levels too");
});

test("fetching ahead asks for no level below the lowest there is", () => {
  const demands = [], d = new Data({ ...UI, ahead: () => demands });
  d.fetchMany = async (xs) => xs.map(() => null);
  // steps a millionth apart: the chart's buckets are of the lowest level already
  const runs = Array.from({ length: 3 }, (_, i) => d.newRun({ id: `r${i}`, seq: 10, mseq: 0, compiled: 10, keys: ["loss"], summary: { _step: 1e-6 }, state: "finished" }));
  demands.push({ key: "loss", runs, runsSig: "a", xmode: 0, zoomed: false, x0: -Infinity, x1: Infinity, pw: 600, many: false });
  assert.equal(d.layersOf(demands[0], runs).coarse.level, -20);
  const asks = d.nextAhead(100, Infinity);
  assert.ok(asks.length > 0 && asks.every((x) => x.level >= -20), `levels asked for: ${[...new Set(asks.map((x) => x.level))]}`);
  assert.deepEqual(d.zoomArrays("loss"), []);
});

/** A Data of 3 finished runs logging metrics k0..k(n-1), nothing fetched, whose UI says to fetch ahead for each metric in
 * turn; `looked` lists the metrics whose wanted layers `nextAhead` looks at, `lists` counts its calls of `ui.ahead`. */
function aheadOver(n) {
  const keys = Array.from({ length: n }, (_, i) => `k${i}`), looked = [], lists = { n: 0 };
  const demands = [], d = new Data({ ...UI, ahead: () => (lists.n++, demands) });
  d.fetchMany = async (xs) => xs.map(() => null);
  const runs = Array.from({ length: 3 }, (_, i) => d.newRun({ id: `r${i}`, seq: 10, mseq: 0, compiled: 10, keys, summary: { _step: 1000 }, state: "finished" }));
  demands.push(...keys.map((key) => ({ key, runs, runsSig: "a", xmode: 0, zoomed: false, x0: -Infinity, x1: Infinity, pw: 600, many: false })));
  const wanted = d.wantedAhead.bind(d);
  d.wantedAhead = (dd, q) => (looked.push(dd.key), wanted(dd, q));
  const take = (asks) => asks.forEach((x) => d.aheadAsked.add(askId(x))); // as `prefetch` marks those it sends
  return { d, runs, demands, looked, lists, take };
}

test("fetching ahead scans on from the chart whose requests it last found, not from the first chart again", () => {
  const { d, looked, lists, take } = aheadOver(5);
  const first = d.nextAhead(1, Infinity);
  assert.deepEqual([first.map((x) => x.key), looked], [["k0"], ["k0"]]);
  take(first);
  looked.length = 0;
  const next = d.nextAhead(2, Infinity);
  assert.deepEqual(next.map((x) => x.key), ["k1", "k2"]);
  assert.deepEqual(looked, ["k0", "k1", "k2"], "k0 is looked at once more: its requests might not all have been taken");
  take(next);
  looked.length = 0;
  const rest = d.nextAhead(100, Infinity);
  assert.deepEqual(looked, ["k1", "k2", "k3", "k4"], "k0, found asked for, is behind the scan");
  assert.deepEqual([...new Set(rest.map((x) => x.key))].sort(), ["k0", "k1", "k2", "k3", "k4"], "then every chart's finer levels");
  assert.ok(rest.findIndex((x) => x.key === "k3") < rest.findIndex((x) => x.key === "k0"), "the wanted layers of k3 before any finer level");
  assert.equal(lists.n, 1, "one scan asks the UI for its demands once");
  take(rest);
  assert.deepEqual([d.nextAhead(100, Infinity), d.aheadMore], [[], false]);
  looked.length = 0;
  assert.deepEqual([d.nextAhead(100, Infinity), looked, lists.n], [[], [], 1], "a finished scan looks at nothing");
});

test("requests the scan found and that were not taken are found again", () => {
  const { d, take } = aheadOver(3);
  const two = d.nextAhead(2, Infinity);
  take(two.slice(0, 1));
  assert.deepEqual(d.nextAhead(2, Infinity).map((x) => x.key), ["k1", "k2"]);
});

test("a scan of the charts takes its time a call, a chart at least, and says when it stopped before its end", () => {
  const whole = aheadOver(4), all = whole.d.nextAhead(1000, Infinity).map(askId);
  const { d, take } = aheadOver(4), got = [];
  let calls = 0;
  do {
    const asks = d.nextAhead(1000, -1); // no time: one chart a call
    take(asks);
    got.push(...asks.map(askId));
    calls++;
  } while (d.aheadMore && calls < 100);
  assert.deepEqual(got, all);
  assert.ok(calls >= 8 && calls < 100, `${calls} calls for 4 charts' wanted layers and their finer levels`);
});

test("the scan starts over when the view planned for, the runs or their metrics change, and when a request fails", () => {
  const { d, runs, demands, looked, lists, take } = aheadOver(4);
  const again = (why, change) => {
    take(d.nextAhead(2, Infinity));
    looked.length = 0;
    const before = lists.n;
    change();
    d.nextAhead(1, Infinity);
    assert.deepEqual([looked[0], lists.n], ["k0", before + 1], why);
  };
  again("another view", () => d.plan([{ ...demands[3], pw: 300 }]));
  again("a run more", () => d.newRun({ id: "new", seq: 1, mseq: 0, compiled: 1, keys: ["k0"], summary: { _step: 5 }, state: "finished" }));
  again("a run finished", () => d.setMeta(runs[0], { ...runs[0].meta, state: "crashed" }));
  again("a failed request", () => d.fail([{ key: "k0", level: 0, index: 0, runs }], "down"));
  clearTimeout(d.retryT);
  take(d.nextAhead(2, Infinity));
  looked.length = 0;
  d.plan([{ ...demands[3], pw: 300 }]);
  d.nextAhead(1, Infinity);
  assert.notEqual(looked[0], "k0", "the same view planned again leaves the scan where it is");
});

test("fetching ahead goes on in another task when its scan stopped short with nothing to ask for", async () => {
  const { d } = aheadOver(3), sent = [];
  const scan = d.nextAhead.bind(d);
  d.nextAhead = (n) => scan(n, -1); // no time: one chart a call
  d.fetchMany = (xs) => (sent.push([...new Set(xs.map((x) => x.key))]), new Promise(() => {}));
  d.prefetch();
  assert.deepEqual(sent, [["k0"]], "its second request waits: the call for it looked at k0 again and found nothing new");
  await new Promise((ok) => setTimeout(ok, 30));
  assert.deepEqual(sent, [["k0"], ["k1"]]);
});

test("every run lacks a block of which nothing is here: a few are asked for by id, in the order of the list", () => {
  const { d, runs } = withRuns(10, 1), queue = [];
  d.need("loss", 2, 0, runs, queue);
  assert.deepEqual(queue.map((x) => x.runs.map((r) => r.id)), [runs.map((r) => r.id)]);
});

test("a scope's runs are listed once its stream is open, and a run that appears meanwhile reaches the page through it", async () => {
  const sources = [], meta = (id) => ({ id, seq: 1, mseq: 0, compiled: 1, keys: [], summary: {}, state: "finished" });
  const realFetch = globalThis.fetch;
  let answer = null;
  globalThis.EventSource = class {
    constructor(url) { this.url = url; this.on = {}; sources.push(this); }
    addEventListener(kind, fn) { this.on[kind] = fn; }
    close() {}
  };
  globalThis.fetch = () => new Promise((ok) => (answer = () => ok({ ok: true, json: async () => ({ runs: [meta("a")], media: [], folders: {} }) })));
  try {
    const d = new Data(UI), loading = d.loadScope("");
    assert.equal(sources.length, 1);
    await tasks();
    assert.equal(answer, null);
    sources[0].onopen();
    await tasks();
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

test("rows streamed for a run raise the version of its metadata, whose summary they change", async () => {
  const { d, runs } = await shown(2), was = [runs[0].ver, d.metaVer];
  d.onRows(runs[0], { run: "r0", seq0: 10, rows: [[1001, 5, { loss: 0.5 }]] });
  assert.deepEqual([runs[0].meta.summary.loss, runs[0].ver > was[0], runs[0].ver === d.metaVer, runs[1].ver < was[1] + 1], [0.5, true, true, true]);
});

test("rows streamed for a run its chart does not draw leave its column as it is, until the chart draws it again", async () => {
  const { d, runs } = await shown(2);
  d.plan([{ ...demandOf(runs.slice(0, 1)), runsSig: "r0 alone" }]);
  await tasks();
  const hidden = runs[1].cols.get("loss");
  d.onRows(runs[1], { run: "r1", seq0: 10, rows: [[1001, 5, { loss: 0.5 }]] });
  await tasks();
  assert.equal(runs[1].cols.get("loss"), hidden);
  d.plan([{ ...demandOf(runs), runsSig: "both" }]);
  await tasks();
  const back = runs[1].cols.get("loss");
  assert.ok(back !== hidden && back.n === hidden.n + 1, `${hidden.n} points, then ${back.n}`);
});

test("rows streamed for a run its chart draws are taken into its column when the chart next draws, not as they come", async () => {
  const { d, runs } = await shown(2), was = runs.map((r) => r.cols.get("loss"));
  d.onRows(runs[0], { run: "r0", seq0: 10, rows: [[1001, 5, { loss: 0.5 }]] });
  await tasks();
  assert.equal(runs[0].cols.get("loss"), was[0], "as rows come, a column stays");
  const ver = d.keyVersion("loss");
  assert.equal(d.catchUp("loss"), true);
  const now = runs[0].cols.get("loss");
  assert.ok(now !== was[0] && now.n === was[0].n + 1, `${was[0].n} points, then ${now.n}`);
  assert.ok(d.keyVersion("loss") > ver, "what the metric's charts show changed with the column");
  assert.equal(runs[1].cols.get("loss"), was[1], "a run no row came for keeps its column");
  assert.deepEqual([d.catchUp("loss"), runs[0].cols.get("loss") === now], [false, true], "a column up to its rows is left alone");
});

test("a running run's tail keeps the rows a chart in view lacks, and drops those its levels hold once the chart is out of view", async () => {
  const d = new Data(UI), row = (i) => [1000 + i, i, { loss: i }];
  let held = 10; // the rows the server's blocks hold
  d.fetchMany = async (xs) => xs.map((x) => emptyArray(x.level, x.index, ["r0"], [held]));
  const r = d.newRun({ id: "r0", seq: 10, mseq: 0, compiled: 10, keys: ["loss"], summary: { _step: 1000 }, state: "running" }), demand = demandOf([r]);
  d.plan([demand]);
  await tasks();
  for (let i = 10; i < 15; i++) d.onRows(r, { run: "r0", seq0: i, rows: [row(i)] });
  d.onRunMeta({ ...r.meta, seq: 15, compiled: 13 }); // its levels were compiled anew; the page's blocks hold 10 rows still
  d.onRows(r, { run: "r0", seq0: 15, rows: [row(15)] });
  assert.deepEqual([r.tailSeq0, r.tail.length], [10, 6], "the chart in view draws the run from blocks of 10 rows");
  d.plan([], true); // a plan of the charts in view alone: this one is near the view still
  d.onRows(r, { run: "r0", seq0: 16, rows: [row(16)] });
  assert.deepEqual([r.tailSeq0, r.tail.length], [10, 7], "a plan of some of the charts takes none of the others out of view");
  d.plan([]); // the chart left the view
  d.onRows(r, { run: "r0", seq0: 17, rows: [row(17)] });
  assert.deepEqual([r.tailSeq0, r.tail.length], [13, 5], "no chart in view needs the rows the levels hold");
  const old = r.cols.get("loss");
  assert.deepEqual([d.catchUp("loss"), r.cols.get("loss") === old], [false, true], "rows 10 to 12 are gone: the column waits for blocks that hold them");
  held = 13;
  d.plan([demand]); // back in view: its blocks are asked for anew, and the column is built of them
  await tasks();
  const back = r.cols.get("loss"), rows = Array.from({ length: back.n }, (_, i) => (back.w ? back.w[i] : 1)).reduce((a, b) => a + b, 0);
  assert.ok(back !== old && rows === 5 && back.s[0] >= 1013, `${rows} rows from step ${back.s[0]}: those of rows 13 to 17`);
  assert.deepEqual([r.tailSeq0, r.tail.length], [13, 5]);
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

/** fn's result, with console.warn silenced (a failed request warns). */
async function quietly(fn) {
  const warn = console.warn;
  console.warn = () => {};
  try {
    return await fn();
  } finally {
    console.warn = warn;
  }
}

test("a block whose request failed is asked for again only once it is due, twice as late after each failure", () => quietly(async () => {
  const { d, runs } = withRuns(2), sent = [];
  d.fetchMany = async (xs) => {
    sent.push(xs.map((x) => x.runs.map((r) => r.id)));
    throw new Error("internal error: OSError(24, 'Too many open files')");
  };
  const ask = async () => {
    const before = performance.now();
    d.need("loss", 0, 0, runs, d.queue);
    d.pump();
    await tasks();
    return [before, performance.now()];
  };
  const [t0, t1] = await ask(), f = d.failed.get("loss|0|0");
  assert.ok(f.until >= t0 + 1000 && f.until <= t1 + 1000);
  await ask();
  assert.equal(sent.length, 1);
  f.until = performance.now();
  const [t2, t3] = await ask();
  assert.deepEqual([sent, f.n, f.error], [[[["r0", "r1"]], [["r0", "r1"]]], 2, "internal error: OSError(24, 'Too many open files')"]);
  assert.ok(f.until >= t2 + 2000 && f.until <= t3 + 2000);
  d.close();
}));

test("a chart shows the runs that came while the others' requests fail, and says why until they come", () => quietly(() => {
  const { d, runs } = withRuns(2);
  d.plan([demandOf(runs)]);
  const ch = d.charts.get("loss"), { level, indices } = ch.want.coarse, asks = indices.map((index) => ({ key: "loss", level, index, runs: [runs[1]] }));
  for (const index of indices) d.addArray({ key: "loss", level, index }, emptyArray(level, index, ["r0"], [10]));
  d.settleChart("loss", ch);
  assert.deepEqual([ch.ready, d.failure("loss")], [null, null]);
  d.fail(asks, "boom");
  assert.deepEqual([ch.ready, d.failure("loss")], [ch.want, "boom"]);
  assert.match(d.summary(), new RegExp(`· ${indices.length} failing: boom$`));
  for (const x of asks) d.answered(x);
  assert.deepEqual([d.failure("loss"), d.failed.size, d.summary().includes("failing")], [null, 0, false]);
  d.close();
}));

test("the UI is asked to plan again once a failed block is due", async () => {
  let replans = 0;
  const d = new Data({ ...UI, replan: () => replans++ });
  d.planned = "the last plan";
  d.failed.set("loss|0|0", { runs: new Set(["r0"]), n: 1, until: performance.now() + 100, error: "boom" }); // due once this test has asked
  d.retrySoon();
  assert.equal(replans, 0);
  for (let waited = 0; !replans && waited < 2000; waited += 5) await tasks();
  assert.deepEqual([replans, d.planned], [1, null]);
});

/** A Data holding running run "r" with its levels at row 49, whose server answers /api/run with `run` (fields over the
 * run's) and /api/rows with `rows`, counting the requests (and failing past 20); the delays of resyncs are recorded,
 * not waited for. */
function resyncing(run, rows) {
  const d = new Data(UI), base = { id: "r", seq: 49, mseq: 0, compiled: 49, keys: ["loss"], summary: {}, state: "running" };
  const r = d.newRun(base), asked = [], delays = [], realFetch = globalThis.fetch, timer = globalThis.setTimeout;
  globalThis.fetch = async (url) => {
    asked.push(url.split("?")[0]);
    if (asked.length > 20) throw new Error("resyncing without end");
    return { ok: true, json: async () => (url.includes("/api/rows") ? rows : { run: { ...base, ...run }, media: [] }) };
  };
  globalThis.setTimeout = (f, ms) => (ms >= 500 ? delays.push(ms) : timer(f, ms));
  const restore = () => {
    globalThis.fetch = realFetch;
    globalThis.setTimeout = timer;
  };
  return { d, r, asked, delays, restore };
}

const rowsFrom = (seq0, n) => ({ run: "r", seq0, rows: Array.from({ length: n }, (_, i) => [seq0 + i, 0, { loss: 1 }]) });

test("a resync after which the rows waiting for it still leave a gap is tried again later, twice as late each time", () => quietly(async () => {
  const { d, asked, delays, restore } = resyncing({}, null);
  try {
    d.dispatch("rows", rowsFrom(50, 1));
    await tasks();
    await tasks();
    assert.deepEqual([asked, delays], [["/api/run"], [1000]]);
  } finally {
    restore();
  }
}));

test("rows a resync takes that begin past the run's levels are its tail from their first row, and the rows after follow", () => quietly(async () => {
  const { d, r, asked, delays, restore } = resyncing({ seq: 52 }, rowsFrom(50, 2));
  try {
    d.dispatch("rows", rowsFrom(50, 1));
    await tasks();
    d.dispatch("rows", rowsFrom(52, 1));
    assert.deepEqual([asked, delays, r.seq, d.tailOf(r, "loss").q], [["/api/run", "/api/rows"], [], 53, [50, 51, 52]]);
  } finally {
    restore();
  }
}));

test("a plan finding every block of a binned chart in the store tells the UI of it at once", () => {
  const told = [], d = new Data({ ...UI, data: (keys) => told.push([...keys]) });
  d.fetchMany = async () => assert.fail("nothing to fetch");
  const runs = ["r0", "r1"].map((id) => d.newRun({ id, seq: 10, mseq: 0, compiled: 10, keys: ["loss"], summary: { _step: 1000 }, state: "finished" }));
  const demand = { ...demandOf(runs), many: true }, { level, indices } = d.layersOf(demand, runs).coarse;
  for (const index of indices) d.addArray({ key: "loss", level, index }, emptyArray(level, index, ["r0", "r1"], [10, 10]));
  told.length = 0;
  d.plan([demand]);
  assert.deepEqual([d.charts.get("loss").ready?.coarse.level, told], [level, [["loss"]]]);
});
