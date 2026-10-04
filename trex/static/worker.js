// A pool.js worker: group statistics, or each run's bin means, of the page's charts, over columns and bucket arrays in
// the memory it shares; and bucket arrays fetched into that memory.
import { BinCache, Col, RAW, aggGroups, binRows, bucketExtent, bucketPaths, bucketViews, buildColumn } from "./kernel.js";

const chunks = new Map(); // shared chunk id -> its Float64Array
const arrays = new Map(); // "chunk:generation:offset" -> bucketViews of the bucket array there
const ARRAY_VIEWS = 512; // views kept; more are dropped and made again when used
const NO_TAIL = { s: [], v: [], t: [], q: [], n: 0 };
const charts = new Map(); // chart -> {cache (its binnings), views (its columns, by location)}

onmessage = ({ data: m }) => {
  if (m.buf) return chunks.set(m.chunk, new Float64Array(m.buf));
  if (m.kind === "fetch") return fetchArray(m);
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

/** Fetch a bucket array (POST m.body to m.url) into a SharedArrayBuffer of its own, read as it streams in (far faster
 * than arrayBuffer() in Chromium), so the page only takes it over. */
async function fetchArray(m) {
  try {
    const res = await fetch(m.url, { method: "POST", body: m.body });
    if (!res.ok) return postMessage({ job: m.job, status: res.status, buf: null, bytes: 0 });
    const [buf, bytes] = await readShared(res), v = bucketViews(buf, 0);
    postMessage({ job: m.job, status: res.status, buf, bytes, paths: bucketPaths(buf, v), ext: bucketExtent(v) });
  } catch (e) {
    postMessage({ job: m.job, status: 0, buf: null, bytes: 0, error: String(e) });
  }
}

/** The body of `res` in a SharedArrayBuffer (a multiple of 8 bytes) and its length; filled chunk by chunk when the
 * length is known ahead (an uncompressed answer), else gathered first. */
async function readShared(res) {
  const len = res.headers.get("Content-Encoding") ? NaN : Number(res.headers.get("Content-Length"));
  const reader = res.body.getReader();
  if (!(len >= 0)) {
    const ab = await new Response(new ReadableStream({ start: (c) => pump(reader, (v) => c.enqueue(v)).then(() => c.close()) })).arrayBuffer();
    const buf = new SharedArrayBuffer(Math.ceil(ab.byteLength / 8) * 8);
    new Uint8Array(buf).set(new Uint8Array(ab));
    return [buf, ab.byteLength];
  }
  const buf = new SharedArrayBuffer(Math.ceil(len / 8) * 8), u8 = new Uint8Array(buf);
  let at = 0;
  await pump(reader, (v) => u8.set(v, (at += v.byteLength) - v.byteLength));
  return [buf, at];
}

async function pump(reader, take) {
  for (let r = await reader.read(); !r.done; r = await reader.read()) take(r.value);
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

/** The column of a run's buckets in the bucket arrays refs[q, q + count) refer to, at `level`. */
function partsColumn(refs, q, count, level) {
  const parts = [];
  for (let j = 4 * q; j < 4 * (q + count); j += 4) {
    const key = `${refs[j]}:${refs[j + 1]}:${refs[j + 2]}`;
    let v = arrays.get(key);
    if (!v) {
      if (arrays.size >= ARRAY_VIEWS) arrays.clear();
      arrays.set(key, (v = bucketViews(chunks.get(refs[j]).buffer, 8 * refs[j + 2])));
    }
    parts.push({ v, row: refs[j + 3] });
  }
  return buildColumn(parts, NO_TAIL, level, false);
}

function view(d, o) {
  const all = chunks.get(d[o]), off = d[o + 2], cap = d[o + 3], bits = d[o + 5];
  const c = Col.view(all.subarray(off, off + cap), all.subarray(off + cap, off + 2 * cap), all.subarray(off + 2 * cap, off + 3 * cap),
                     d[o + 4], [(bits & 1) !== 0, (bits & 2) !== 0], all.subarray(off + 3 * cap, off + 4 * cap));
  c.gen = d[o + 1];
  return c;
}
