// A pool.js worker: group statistics, or each run's bin means, of the page's charts, over copies the page sent of the
// columns and bucket arrays they read; and bucket arrays fetched for the page.
import { BinCache, Col, RAW, aggGroups, binRows, bucketPaths, bucketViews, buildColumn, runColumn, unframe } from "./kernel.js";

const copies = new Map(); // "chunk:generation" -> Map(float offset -> a copy of the page's floats from there)
const arrays = new Map(); // "chunk:generation:offset" -> bucketViews of the bucket array there
const ARRAY_VIEWS = 512; // views kept; more are dropped and made again when used
const NO_TAIL = { s: [], v: [], t: [], q: [], n: 0 };
const charts = new Map(); // chart -> {cache (its binnings), views (its columns, by location)}

onmessage = ({ data: m }) => {
  if (m.copies) return keepCopies(m.copies);
  if (m.drop) return copies.delete(`${m.drop[0]}:${m.drop[1]}`);
  if (m.kind === "fetch") return fetchArrays(m);
  let st = charts.get(m.chart);
  if (!st) charts.set(m.chart, (st = { cache: new BinCache(), views: new Map() }));
  const groups = columns(m, st), p = m.p;
  if (m.kind === "rows") {
    const rows = binRows(groups.flat(), p, st.cache);
    return postMessage({ job: m.job, rows }, [rows.buffer]);
  }
  const main = aggGroups(groups, p.xmode, p.x0, p.x1, p.bins, p.flags, p.alpha, p.scale, st.cache);
  const raws = m.raw ? aggGroups(groups, p.xmode, p.x0, p.x1, p.bins, p.flags | RAW, p.alpha, p.scale, st.cache) : null;
  postMessage({ job: m.job, main, raws }, raws ? [main.buffer, raws.buffer] : [main.buffer]);
};

function keepCopies(list) {
  for (const [chunk, gen, off, data] of list) {
    const key = `${chunk}:${gen}`;
    if (!copies.has(key)) copies.set(key, new Map());
    copies.get(key).set(off, data);
  }
}

/** This worker's copy of the floats from float `off` of chunk `chunk` (generation `gen`). */
const floats = (chunk, gen, off) => copies.get(`${chunk}:${gen}`).get(off);

/** Fetch bucket arrays (POST m.body to m.url, answered as buckets.frame joins them), each into a buffer of its own,
 * handed over to the page. */
async function fetchArrays(m) {
  try {
    const res = await fetch(m.url, { method: "POST", body: m.body });
    if (!res.ok) return postMessage({ job: m.job, status: res.status, arrays: [], bytes: 0 });
    const all = await readBody(res), arrays = [], moved = [];
    for (const { off, len } of unframe(all)) {
      const buf = len ? new ArrayBuffer(Math.ceil(len / 8) * 8) : null;
      if (buf) new Uint8Array(buf).set(new Uint8Array(all, off, len)), moved.push(buf);
      arrays.push(buf && { buf, bytes: len, paths: bucketPaths(buf, bucketViews(buf, 0)) });
    }
    postMessage({ job: m.job, status: res.status, arrays, bytes: all.byteLength }, moved);
  } catch (e) {
    postMessage({ job: m.job, status: 0, arrays: [], bytes: 0, error: String(e) });
  }
}

/** The body of `res`, read as it streams in when its length is known ahead (an uncompressed answer; far faster than
 * arrayBuffer() in Chromium). */
async function readBody(res) {
  const len = res.headers.get("Content-Encoding") ? NaN : Number(res.headers.get("Content-Length"));
  if (!(len >= 0)) return res.arrayBuffer();
  const u8 = new Uint8Array(len), reader = res.body.getReader();
  let at = 0;
  for (let r = await reader.read(); !r.done; r = await reader.read()) u8.set(r.value, at), (at += r.value.byteLength);
  return u8.buffer;
}

/** The job's groups of columns, each the chart's view of the same column when it has one, so its binnings carry
 * over; the chart then keeps only these. */
function columns(m, st) {
  const views = new Map(), groups = [], d = m.desc;
  let r = 0;
  for (const end of m.ends) {
    const g = [];
    for (; r < end; r++) {
      const o = 6 * r, key = d[o] < 0 ? partsKey(m.refs, d[o + 1], d[o + 2]) : d[o + 2] * 4096 + d[o]; // offset, then chunk
      let c = st.views.get(key);
      if (!c || (d[o] >= 0 && (c.gen !== d[o + 1] || c.n !== d[o + 4]))) c = d[o] < 0 ? partsColumn(m.refs, d[o + 1], d[o + 2], d[o + 3]) : view(d, o);
      views.set(key, c);
      g.push(c);
    }
    groups.push(g);
  }
  st.views = views;
  return groups;
}

function partsKey(refs, q, count) {
  let key = "p";
  for (let j = 4 * q; j < 4 * (q + count); j++) key += `:${refs[j]}`;
  return key;
}

/** The column of a run's buckets in the bucket arrays refs[q, q + count) refer to, at `level`: viewing the array when
 * there is one. */
function partsColumn(refs, q, count, level) {
  const parts = [];
  for (let j = 4 * q; j < 4 * (q + count); j += 4) {
    const key = `${refs[j]}:${refs[j + 1]}:${refs[j + 2]}`;
    let v = arrays.get(key);
    if (!v) {
      if (arrays.size >= ARRAY_VIEWS) arrays.clear();
      const mine = floats(refs[j], refs[j + 1], refs[j + 2]);
      arrays.set(key, (v = bucketViews(mine.buffer, mine.byteOffset)));
    }
    parts.push({ v, row: refs[j + 3] });
  }
  return parts.length === 1 ? runColumn(parts[0].v, parts[0].row) : buildColumn(parts, NO_TAIL, level, false);
}

function view(d, o) {
  const cap = d[o + 3], bits = d[o + 5], all = floats(d[o], d[o + 1], d[o + 2]);
  const c = Col.view(all.subarray(0, cap), all.subarray(cap, 2 * cap), all.subarray(2 * cap, 3 * cap),
                     d[o + 4], [(bits & 1) !== 0, (bits & 2) !== 0], all.subarray(3 * cap, 4 * cap));
  c.gen = d[o + 1];
  return c;
}
