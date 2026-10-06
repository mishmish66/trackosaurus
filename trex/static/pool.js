// Workers computing group statistics, or runs' bin means, over columns and bucket arrays in located chunks (kernel.js
// `columnStore`, `adoptStore`), each chart on one worker, so the charts of a round are computed in parallel and each
// worker keeps its charts' binnings; each worker is sent copies of what its jobs read, once. They also fetch bucket
// arrays for the page.
//# allFunctionsCalledOnLoad
import { bucketViews, chunkView, onDrops } from "./kernel.js";

/** Workers in the pool. */
const SIZE = typeof Worker === "function" ? Math.max(1, Math.min(8, (navigator.hardwareConcurrency || 4) - 1)) : 0;
const FIELDS = 6; // numbers describing a column to a worker: chunk, generation, offset, capacity, points, sortedness

let workers = null, jobs = 0;
const answers = new Map(); // job -> its promise's resolve
const copied = []; // per worker: "chunk:generation" -> the offsets of the copies it holds

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
  copied.push(...workers.map(() => new Map()));
  onDrops((chunk, gen) => {
    workers.forEach((w, i) => copied[i].delete(`${chunk}:${gen}`) && w.postMessage({ drop: [chunk, gen] }));
  });
  return workers;
}

/** Send worker i copies of the columns and bucket arrays described by d that it does not hold. */
function sendCopies(i, d) {
  const out = [], moved = [];
  const add = (chunk, gen, off, len) => {
    const key = `${chunk}:${gen}`;
    let offs = copied[i].get(key);
    if (!offs) copied[i].set(key, (offs = new Set()));
    if (offs.has(off)) return;
    offs.add(off);
    const data = chunkView(chunk).slice(off, off + len);
    out.push([chunk, gen, off, data]);
    moved.push(data.buffer);
  };
  const { desc, refs } = d;
  for (let o = 0; o < desc.length; o += FIELDS) {
    if (desc[o] >= 0) add(desc[o], desc[o + 1], desc[o + 2], 4 * desc[o + 3]);
    else for (let j = 4 * desc[o + 1]; j < 4 * (desc[o + 1] + desc[o + 2]); j += 4) add(refs[j], refs[j + 1], refs[j + 2], arrayFloats(refs[j], refs[j + 2]));
  }
  if (out.length) workers[i].postMessage({ copies: out }, moved);
}

/** Floats the bucket array at float `off` of chunk `chunk` takes. */
const arrayFloats = (chunk, off) => Math.ceil(bucketViews(chunkView(chunk).buffer, 8 * off).bytes / 8);

/** Whether workers compute group statistics. */
export const PARALLEL = SIZE > 0;

/** Start the workers, so their modules load while the page does. */
export function startWorkers() {
  if (PARALLEL) pool();
}

/** `groups` (lists of sources: columns, or a run's buckets {parts ([{v, row, a}] of bucket arrays, `Data.partsOf`),
 * level}) as a worker reads them: {desc (FIELDS numbers per source: a column's chunk, generation, offset, capacity,
 * points and sortedness, or -1, its first reference, their count and its level), refs (per part: chunk, generation,
 * offset in floats, row), ends (where each group ends)}. */
export function describe(groups) {
  let R = 0, S = 0;
  for (const g of groups) for (const c of g) (R += 1), (S += c.parts ? c.parts.length : 0);
  const desc = new Float64Array(FIELDS * R), refs = new Float64Array(4 * S), ends = new Int32Array(groups.length);
  let r = 0, q = 0;
  for (let gi = 0; gi < groups.length; gi++) {
    for (const c of groups[gi]) {
      const o = FIELDS * r++;
      if (c.parts) (desc[o] = -1), (desc[o + 1] = q / 4), (desc[o + 2] = c.parts.length), (desc[o + 3] = c.level), (q = refer(c, refs, q));
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
  c.parts.forEach((p, j) => refs.set([p.a.loc.chunk, p.a.loc.gen, p.a.loc.off, p.row], q + 4 * j));
  return q + 4 * c.parts.length;
}

let fetches = 0;

/** POST `body` to `url` on a worker, which reads the bucket arrays it answers each into a buffer of its own, handed
 * over: a promise of {status, arrays ({buf, bytes, paths (its runs), ext (kernel.js rowExtents by step)}, null for a
 * block not answered), bytes}. */
export function fetchArraysOnWorker(url, body) {
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
  const ws = pool(), job = ++jobs, i = slot % ws.length;
  return new Promise((ok) => {
    answers.set(job, ok);
    sendCopies(i, d);
    ws[i].postMessage({ kind, job, chart: slot, desc: d.desc, refs: d.refs, ends: d.ends, p, raw }, [d.desc.buffer, d.refs.buffer, d.ends.buffer]);
  });
}
