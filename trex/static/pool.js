// Workers computing group statistics, or runs' bin means, over columns and slabs in shared memory (kernel.js
// `columnStore`), each chart on one worker, so the charts of a round are computed in parallel and each worker keeps
// its charts' binnings.
import { SHARED, onChunks } from "./kernel.js";

/** Workers in the pool; none when the page is not cross-origin isolated. */
const SIZE = SHARED && typeof Worker === "function" ? Math.max(1, Math.min(8, (navigator.hardwareConcurrency || 4) - 1)) : 0;
const FIELDS = 6; // numbers describing a column to a worker: chunk, generation, offset, capacity, points, sortedness

let workers = null, jobs = 0;
const answers = new Map(); // job -> its promise's resolve

function pool() {
  if (workers) return workers;
  workers = Array.from({ length: SIZE }, () => {
    const w = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
    w.onmessage = ({ data }) => {
      answers.get(data.job)?.(data);
      answers.delete(data.job);
    };
    return w;
  });
  onChunks((chunk, buf) => {
    for (const w of workers) w.postMessage({ chunk, buf });
  });
  return workers;
}

/** Whether workers compute group statistics. */
export const PARALLEL = SIZE > 0;

/** Start the workers, so their modules load while the page does. */
export function startWorkers() {
  if (PARALLEL) pool();
}

/** `groups` (lists of columns, or {slabs, rows} sources: a run's rows of slabs in shared memory, `Data.slabFor`) as a
 * worker reads them: {desc (FIELDS numbers per source: a column's chunk, generation, offset, capacity, points and
 * sortedness, or -1, its first reference, their count and its buckets), refs (per slab reference: chunk, generation,
 * offset in floats, row), ends (where each group ends)}; null when a column is not in shared memory. */
export function describe(groups) {
  let R = 0, S = 0;
  for (const g of groups) for (const c of g) (R += 1), (S += c.slabs ? c.slabs.length : 0);
  const desc = new Float64Array(FIELDS * R), refs = new Float64Array(4 * S), ends = new Int32Array(groups.length);
  let r = 0, q = 0;
  for (let gi = 0; gi < groups.length; gi++) {
    for (const c of groups[gi]) {
      const o = FIELDS * r++;
      if (c.slabs) (desc[o] = -1), (desc[o + 1] = q / 4), (desc[o + 2] = c.slabs.length), (desc[o + 3] = c.n), (q = refer(c, refs, q));
      else if (!c.loc) return null;
      else describeColumn(c, desc, o);
    }
    ends[gi] = r;
  }
  return { desc, refs, ends };
}

function describeColumn(c, desc, o) {
  const l = c.loc;
  (desc[o] = l.chunk), (desc[o + 1] = l.gen), (desc[o + 2] = l.off), (desc[o + 3] = l.cap), (desc[o + 4] = c.n);
  desc[o + 5] = (c.sorted[0] ? 1 : 0) | (c.sorted[1] ? 2 : 0);
}

function refer(c, refs, q) {
  c.slabs.forEach((s, j) => refs.set([s.loc.chunk, s.loc.gen, s.loc.off, c.rows[j]], q + 4 * j));
  return q + 4 * c.slabs.length;
}

let fetches = 0;

/** POST `body` to `url` on a worker, which copies a slab answer into shared memory: a promise of {status, buf
 * (SharedArrayBuffer, null unless the answer was a slab), bytes, paths (its runs), ext ([first, last] step)}. */
export function fetchSlabOnWorker(url, body) {
  const ws = pool(), job = ++jobs;
  return new Promise((ok) => {
    answers.set(job, ok);
    ws[fetches++ % ws.length].postMessage({ kind: "fetch", job, url, body });
  });
}

/** On the worker of chart `slot`, with binning p ({xmode, x0, x1, bins, flags, alpha, scale}): kind "agg", aggGroups
 * (kernel.js) of described groups `d`, and of their raw values too when `raw`, a promise of {main, raws}; kind "rows",
 * binRows of all their sources, a promise of {rows}. */
export function onWorker(kind, slot, d, p, raw) {
  const ws = pool(), job = ++jobs;
  return new Promise((ok) => {
    answers.set(job, ok);
    ws[slot % ws.length].postMessage({ kind, job, chart: slot, desc: d.desc, refs: d.refs, ends: d.ends, p, raw }, [d.desc.buffer, d.refs.buffer, d.ends.buffer]);
  });
}
