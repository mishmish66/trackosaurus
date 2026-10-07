// Runs' bin means and group statistics on the GPU, drawn from where they are computed: what kernel.js `binRows` and
// `aggGroups` compute on the workers, as WebGL2 passes over the bucket arrays (uploaded once each) and the running
// runs' columns. The charts binned together make a round: their bins lie side by side as the columns of shared
// textures (a job's bins the columns from its first on), a run a row by its run index, so each step is one pass over
// every chart. Each run's buckets are added up per bin into sums, which become bin means, empty bins between full ones
// interpolated (`binColumn`). For group statistics each group's values in a bin are put in order and summarized
// (`binStats`; infinities as values, NaN none) into its center and band (plot.js `bandOf`). The results stay on the GPU
// as points the charts draw (gl.js `Points` layout); only each chart's y range comes back, all of a call's at once, and
// the values under the pointer when it hovers.
//# allFunctionsCalledOnLoad

import { compile, linked, renderer } from "./gl.js";
import { BLOCK, LOGX, SOFF_SCALE, X_STEP, medianCiRank } from "./kernel.js";

const TW = 2048; // texels per row of the textures of buckets, points and per-run tables (gl.js TW)
const MAX_SLOTS = 16384; // runs (texture rows) a job bins, at most
const MAX_BINS = 4096; // bins one job has, at most
const MAX_JOBS = 32; // jobs of one round, at most...
const MAX_COLS = 8192; // ...and the columns their bins take together
const COLS = 256; // a round's textures are a multiple of this many columns wide, and of ROWS rows of points high, so
const ROWS = 16; // that rounds of about the same size share textures
const BIG = 3.0e38; // beyond any range: what a range starts from
const SCRATCH = 15; // the texture unit textures are made and filled on, so that none a draw reads is replaced
const SMALL = 32; // groups of at most this many runs are put in order where their statistics are computed
const FOLD = 64; // ranges folded into one at a time
const MEANS_BYTES = 128 << 20; // jobs' bin means kept for later rounds of the same data, the least recently used freed beyond
const ROUND_BYTES = 192 << 20; // rounds' means kept whole for a round of the same jobs, likewise (a zoom's rounds take tens of MB)
const ARRAY_BYTES = 768 << 20; // bucket arrays kept on the GPU, the least recently used freed beyond
const CENTERS = { mean: 0, median: 1, iqm: 2 };
const BANDS = { ci: 0, iqr: 1, minmax: 2, std: 3, stderr: 4, none: 5 };
// Two-sided 95% Student t critical values for df = 1..30 (plot.js T95); normal beyond.
const T95 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.16, 2.145, 2.131,
             2.12, 2.11, 2.101, 2.093, 2.086, 2.08, 2.074, 2.069, 2.064, 2.06, 2.056, 2.052, 2.048, 2.045, 2.042];

const COMMON = `#version 300 es
precision highp float;
precision highp int;
float inf() { return uintBitsToFloat(0x7f800000u); }
float ninf() { return uintBitsToFloat(0xff800000u); }
float nan() { return uintBitsToFloat(0x7fc00000u); }
bool isNan(float v) { return (floatBitsToUint(v) & 0x7fffffffu) > 0x7f800000u; }
bool isInf(float v) { return (floatBitsToUint(v) & 0x7fffffffu) == 0x7f800000u; }
ivec2 at(int i) { return ivec2(i % ${TW}, i / ${TW}); }
// a * (1 - w) + b * w as JavaScript makes it of infinities and NaN
float mixed(float a, float b, float w) {
  if (isNan(a) || isNan(b)) return nan();
  bool ai = isInf(a), bi = isInf(b);
  if (!ai && !bi) return a * (1.0 - w) + b * w;
  if (ai && bi) return a == b ? a : nan();
  return ai ? a : b;
}
vec4 cell(int x, int y, int w, int h) {
  return vec4((float(x) + 0.5) / float(w) * 2.0 - 1.0, (float(y) + 0.5) / float(h) * 2.0 - 1.0, 0.0, 1.0);
}
// what a value adds to its bin's sums: (value * count, count) of a finite value, a count of +inf or of -inf
vec4 sums(float y, float n) {
  uint u = floatBitsToUint(y);
  return u == 0x7f800000u ? vec4(0.0, 0.0, 1.0, 0.0) : u == 0xff800000u ? vec4(0.0, 0.0, 0.0, 1.0) : vec4(y * n, n, 0.0, 0.0);
}
// a value as a chart draws it: log10 on a log axis, where it is positive; nothing (false) otherwise
bool drawn(float v, int logy, out float y) {
  y = logy == 1 ? log2(v) * 0.30102999566398120 : v;
  return !isNan(v) && !isInf(v) && (logy == 0 || v > 0.0);
}
`;

// The round's jobs, for the passes over all of them.
const JOBS = `
uniform highp isampler2D u_col; // each column's job, -1 for none
uniform ivec4 u_jx[${MAX_JOBS}]; // per job: its first column, its bins, the first and the last bin in view
uniform ivec4 u_jk[${MAX_JOBS}]; // per job: center, band, whether y is logarithmic, its first point
uniform vec2 u_jf[${MAX_JOBS}]; // per job: the width of a bin, the largest x drawn
int jobAt(int x) { return texelFetch(u_col, ivec2(x, 0), 0).r; }
`;

// A bucket array's buckets, read from its bytes (kernel.js bucketViews: u_first, u_off, u_soff, u_mean, u_tmean and u_n
// are where its sections begin, in 32-bit words), into the sums of their runs (u_rows: each row's run index) that are
// not drawn from their columns (u_info: group, column), shown or not, a point each, in the job's columns (from u_x of
// u_cols). Bit 1 of u_mode: runtimes; bit 2: a log axis.
const SCATTER_VS = `${COMMON}
uniform highp usampler2D u_raw;
uniform highp isampler2D u_rows, u_info;
uniform int u_mode, u_bins, u_x, u_cols, u_slots, u_runs, u_first, u_off, u_soff, u_mean, u_tmean, u_n;
uniform float u_base, u_w, u_x0, u_dx;
flat out vec4 v_val;
uint word(int i) { return texelFetch(u_raw, at(i), 0).r; }
void main() {
  int q = gl_VertexID, lo = 0, hi = u_runs;
  while (hi - lo > 1) { int m = (lo + hi) >> 1; if (int(word(u_first + m)) <= q) lo = m; else hi = m; } // q's run
  int sh = (q & 1) * 16;
  float bx = float((word(u_off + q / 2) >> sh) & 0xffffu) + (float((word(u_soff + q / 2) >> sh) & 0xffffu) + 0.5) / ${SOFF_SCALE}.0;
  int slot = texelFetch(u_rows, at(lo), 0).r;
  if (slot >= u_slots || (slot >= 0 && texelFetch(u_info, at(slot), 0).y != 0)) slot = -1;
  float x = (u_mode & 1) != 0 ? uintBitsToFloat(word(u_tmean + q)) : (u_base + bx) * u_w;
  if ((u_mode & 2) != 0) x = x > 0.0 ? log2(x) * 0.30102999566398120 : -3.0e38;
  float fb = (x - u_x0) / u_dx;
  gl_PointSize = 1.0;
  v_val = sums(uintBitsToFloat(word(u_mean + q)), float(word(u_n + q)));
  gl_Position = slot >= 0 && fb >= 0.0 && fb <= float(u_bins) ? cell(u_x + min(u_bins - 1, int(floor(fb))), slot, u_cols, u_slots) : vec4(2.0, 2.0, 2.0, 1.0);
}`;

// The same sums of a bucket array binned by step, gathered: each bin of each run (u_slot: the run's row in the array)
// adds up the run's buckets that fall in it, found by bisection since a run's buckets are in step order. Far fewer
// primitives than a point a bucket. Mode 0: a linear axis, bin u_a + x * u_b (x within the block, in buckets); else a
// log axis.
const GATHER_SUMS_FS = `${COMMON}
uniform highp usampler2D u_raw;
uniform highp isampler2D u_slot, u_info;
uniform int u_mode, u_bins, u_x, u_first, u_off, u_soff, u_mean, u_n;
uniform float u_a, u_b, u_base, u_w, u_x0, u_dx;
out vec4 o;
uint word(int i) { return texelFetch(u_raw, at(i), 0).r; }
// the bin bucket q falls in: -1 before the first, u_bins after the last
int binOf(int q) {
  int sh = (q & 1) * 16;
  float bx = float((word(u_off + q / 2) >> sh) & 0xffffu) + (float((word(u_soff + q / 2) >> sh) & 0xffffu) + 0.5) / ${SOFF_SCALE}.0;
  float fb = u_a + bx * u_b;
  if (u_mode != 0) {
    float x = (u_base + bx) * u_w;
    fb = ((x > 0.0 ? log2(x) * 0.30102999566398120 : -3.0e38) - u_x0) / u_dx;
  }
  return fb < 0.0 ? -1 : fb > float(u_bins) ? u_bins : min(u_bins - 1, int(floor(fb)));
}
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  int row = texelFetch(u_slot, at(p.y), 0).r, bin = p.x - u_x;
  if (row < 0 || texelFetch(u_info, at(p.y), 0).y != 0) discard;
  int lo = int(word(u_first + row)), end = int(word(u_first + row + 1)), hi = end;
  while (lo < hi) { int m = (lo + hi) >> 1; if (binOf(m) < bin) lo = m + 1; else hi = m; }
  vec4 acc = vec4(0.0);
  int n = 0;
  for (int q = lo; q < end && binOf(q) == bin; q++) acc += sums(uintBitsToFloat(word(u_mean + q)), float(word(u_n + q))), n++;
  if (n == 0) discard;
  o = acc;
}`;

// Column points the page binned (bin, value, count, slot) into their sums, in the job's columns.
const POINTS_VS = `${COMMON}
uniform highp sampler2D u_pts;
uniform int u_x, u_cols, u_slots;
flat out vec4 v_val;
void main() {
  vec4 p = texelFetch(u_pts, at(gl_VertexID), 0);
  gl_PointSize = 1.0;
  v_val = sums(p.y, p.z);
  gl_Position = cell(u_x + int(p.x), int(p.w), u_cols, u_slots);
}`;

const VALUE_FS = `${COMMON}
flat in vec4 v_val;
out vec4 o;
void main() { o = v_val; }`;

const QUAD_VS = `#version 300 es
void main() { gl_Position = vec4(gl_VertexID == 1 ? 3.0 : -1.0, gl_VertexID == 2 ? 3.0 : -1.0, 0.0, 1.0); }`;

// Bin means of each run's row, a job's empty bins between full ones interpolated, those before its first or after its
// last NaN.
const FILL_FS = `${COMMON}${JOBS}
uniform highp sampler2D u_sums;
out vec4 o;
bool full(int x, int s, out float v) {
  vec4 t = texelFetch(u_sums, ivec2(x, s), 0);
  v = t.y > 0.0 ? t.x / t.y : t.z > 0.0 && t.w > 0.0 ? nan() : t.z > 0.0 ? inf() : ninf();
  return t.y > 0.0 || t.z > 0.0 || t.w > 0.0;
}
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  int j = jobAt(p.x);
  float v, lv, rv;
  if (j < 0) { o = vec4(nan()); return; }
  if (full(p.x, p.y, v)) { o = vec4(v); return; }
  int x0 = u_jx[j].x, x1 = x0 + u_jx[j].y, l = p.x - 1, r = p.x + 1;
  while (l >= x0 && !full(l, p.y, lv)) l--;
  while (r < x1 && !full(r, p.y, rv)) r++;
  o = vec4(l < x0 || r >= x1 ? nan() : mixed(lv, rv, float(p.x - l) / float(r - l)));
}`;

// Each group's members' values in its rows (u_members: each place's run), for the sorting steps.
const SPREAD_FS = `${COMMON}
uniform highp sampler2D u_m;
uniform highp isampler2D u_members;
out vec4 o;
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  o = vec4(texelFetch(u_m, ivec2(p.x, texelFetch(u_members, at(p.y), 0).r), 0).r);
}`;

// One step of a sorting network over each group's rows (u_placeGroup: each place's group; u_groups: its start and
// size): of the value at a place and the one at its partner, the first in order (-inf, finite ones ascending, +inf,
// then NaN) goes to the lower place. The places are taken in blocks of u_block: a merge's first step pairs a block's
// places end to end (u_mirror), its later ones those half a block apart. A partner beyond the group counts as the
// last in order, so the same steps order groups of any size.
const SORT_FS = `${COMMON}
uniform highp sampler2D u_v;
uniform highp isampler2D u_groups, u_placeGroup;
uniform int u_block, u_mirror;
out vec4 o;
int kind(float v) { return isNan(v) ? 3 : isInf(v) ? (v > 0.0 ? 2 : 0) : 1; }
bool before(float a, float b) { int j = kind(a), k = kind(b); return j < k || (j == k && j == 1 && a < b); }
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  ivec2 g = texelFetch(u_groups, at(texelFetch(u_placeGroup, at(p.y), 0).r), 0).xy;
  int i = p.y - g.x, first = i - i % u_block, gap = u_block / 2;
  int j = u_mirror == 1 ? 2 * first + u_block - 1 - i : (i - first < gap ? i + gap : i - gap);
  float a = texelFetch(u_v, p, 0).r;
  if (j >= g.y) { o = vec4(a); return; }
  float b = texelFetch(u_v, ivec2(p.x, g.x + j), 0).r;
  o = vec4((i < j ? before(b, a) : before(a, b)) ? b : a);
}`;

// Each group's center and band in each bin (center, lo, hi, n): its statistics as binStats makes them, from its values
// in order, then plot.js bandOf, with the center and band its column's job asks for. The values come as `order` gives
// them: V(i), the i-th of the group (start G.x, size G.y) in column B, after order() has run.
const statsFS = (order) => `${COMMON}${JOBS}
uniform highp isampler2D u_groups, u_ci;
uniform float u_t95[30];
out vec4 o;
int B;
ivec2 G;
${order}
float quantile(float h, int n) {
  int i = int(floor(h));
  float f = h - float(i);
  return f > 0.0 && i + 1 < n ? mixed(V(i), V(i + 1), f) : V(i);
}
// (iqm, its standard error) of the n values (neg -inf, m finite, pos +inf) whose kept ranks are [g, n - g)
vec2 iqmOf(int n, int neg, int m, int pos, int g) {
  float lo = V(g), hi = V(n - g - 1);
  int h = n - 2 * g;
  if (isInf(lo) || isInf(hi)) return vec2(lo == ninf() ? (hi == inf() ? nan() : ninf()) : hi, h > 1 ? nan() : 0.0);
  int le = neg, nmid = 0;
  float mid = 0.0, wsum = float(neg) * lo + float(pos) * hi;
  for (int i = neg; i < neg + m; i++) {
    float v = V(i);
    if (v <= lo) le++;
    else if (v < hi) mid += v, nmid++;
    wsum += clamp(v, lo, hi);
  }
  int nlo = lo == hi ? h : le - g;
  float wmean = wsum / float(n);
  float w2 = float(neg) * (lo - wmean) * (lo - wmean) + float(pos) * (hi - wmean) * (hi - wmean);
  for (int i = neg; i < neg + m; i++) { float d = clamp(V(i), lo, hi) - wmean; w2 += d * d; }
  return vec2(lo == hi ? lo : (mid + lo * float(nlo) + hi * float(h - nmid - nlo)) / float(h),
              h > 1 ? sqrt(w2 / (float(h) * float(h - 1))) : 0.0);
}
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  int j = jobAt(p.x);
  if (j < 0) { o = vec4(nan(), nan(), nan(), 0.0); return; }
  int center = u_jk[j].x, band = u_jk[j].y;
  G = texelFetch(u_groups, at(p.y), 0).xy;
  B = p.x;
  order();
  int n = 0, neg = 0, pos = 0;
  float sum = 0.0;
  for (int i = 0; i < G.y; i++) {
    float v = V(i);
    if (isNan(v)) break;
    n++;
    if (isInf(v)) { if (v > 0.0) pos++; else neg++; } else sum += v;
  }
  if (n == 0) { o = vec4(nan(), nan(), nan(), 0.0); return; }
  int m = n - neg - pos;
  float mean = pos > 0 && neg > 0 ? nan() : pos > 0 ? inf() : neg > 0 ? ninf() : sum / float(n), v2 = 0.0;
  for (int i = 0; i < m && pos + neg == 0; i++) { float d = V(i) - mean; v2 += d * d; }
  float sd = n == 1 ? 0.0 : neg + pos > 0 ? nan() : sqrt(v2 / float(n - 1)), h = float(n - 1);
  vec2 iq = center == 2 ? iqmOf(n, neg, m, pos, n / 4) : vec2(0.0);
  float c = center == 0 ? mean : center == 1 ? quantile(h * 0.5, n) : iq.x;
  float se = center == 2 ? iq.y : sd / sqrt(float(n)), lo = c, hi = c;
  if (n > 1) {
    if (band == 0 && center != 1) {
      int df = center == 2 ? n - 2 * (n / 4) - 1 : n - 1;
      float t = df <= 30 ? (df >= 1 ? u_t95[df - 1] : 0.0) : 1.96;
      lo = c - t * se, hi = c + t * se;
    } else if (band == 0) {
      int k = texelFetch(u_ci, at(n), 0).r;
      lo = V(k - 1), hi = V(n - k);
    } else if (band == 1) lo = quantile(h * 0.25, n), hi = quantile(h * 0.75, n);
    else if (band == 2) lo = V(0), hi = V(n - 1);
    else if (band == 3) lo = c - sd, hi = c + sd;
    else if (band == 4) lo = c - se, hi = c + se;
  }
  o = vec4(c, lo, hi, float(n));
}`;

// The values in order from the texture the sorting steps left them in.
const STATS_FS = statsFS(`
uniform highp sampler2D u_v;
void order() {}
float V(int i) { return texelFetch(u_v, ivec2(B, G.x + i), 0).r; }`);

// The values of a group of at most SMALL runs put in order here (-inf, finite ones ascending, +inf, then NaN), from
// its members' bin means: no rank passes.
const STATS_SMALL_FS = statsFS(`
uniform highp sampler2D u_m;
uniform highp isampler2D u_members;
float S[${SMALL}];
void order() {
  for (int t = 0; t < G.y; t++) {
    float v = texelFetch(u_m, ivec2(B, texelFetch(u_members, at(G.x + t), 0).r), 0).r;
    int j = t;
    while (j > 0 && !isNan(v) && (isNan(S[j - 1]) || v < S[j - 1])) { S[j] = S[j - 1]; j--; }
    S[j] = v;
  }
}
float V(int i) { return S[i]; }`);

// The points the round's charts draw, each job's from its first point on (u_count lines a job): of "agg" jobs, per
// group its center, band top and band bottom over the bins (a break where there is no value); of "rows" jobs, per run
// its bin means, a bin without one at its nearest bin's point (the left one first) so that a line strip draws nothing
// there, and breaks when the run has none. (x from the first bin's edge, y as drawn.)
const LINES_FS = `${COMMON}${JOBS}
uniform highp sampler2D u_src;
uniform int u_jobs, u_count, u_agg;
out vec4 o;
int X, L;
float value(int b) { return texelFetch(u_src, ivec2(X + b, L), 0).r; }
void main() {
  int i = int(gl_FragCoord.y) * ${TW} + int(gl_FragCoord.x), j = 0;
  while (j + 1 < u_jobs && i >= u_jk[j + 1].w) j++;
  i -= u_jk[j].w;
  int bins = u_jx[j].y, logy = u_jk[j].z, per = u_agg == 1 ? 3 * bins : bins, k = (i % per) / bins, b = i % bins;
  float dx = u_jf[j].x, xmax = u_jf[j].y, y;
  X = u_jx[j].x;
  L = i / per;
  if (L >= u_count) { o = vec4(1.0e38); return; }
  if (u_agg == 0) {
    int near = drawn(value(b), logy, y) ? b : -1;
    for (int t = b - 1; t >= 0 && near < 0; t--) if (drawn(value(t), logy, y)) near = t;
    for (int t = b + 1; t < bins && near < 0; t++) if (drawn(value(t), logy, y)) near = t;
    o = near >= 0 ? vec4(min(xmax, (float(near) + 0.5) * dx), y, 0.0, 0.0) : vec4(1.0e38);
    return;
  }
  vec4 s = texelFetch(u_src, ivec2(X + b, L), 0);
  float v = k == 0 ? s.x : k == 1 ? s.z : (logy == 1 && !(s.y > 0.0) ? s.x : s.y);
  o = drawn(v, logy, y) ? vec4(min(xmax, (float(b) + 0.5) * dx), y, 0.0, 0.0) : vec4(1.0e38);
}`;

// Each line's y range, a texel a line (x) of each job (y): (center max, -center min, band max, -band min) of an "agg"
// job's group; (value max, -value min) of a "rows" job's run over the bins whose centers lie in view, when it is shown
// (u_info).
const LINE_RANGE_FS = `${COMMON}${JOBS}
uniform highp sampler2D u_src;
uniform highp isampler2D u_info;
uniform int u_agg;
out vec4 o;
void main() {
  int line = int(gl_FragCoord.x), j = int(gl_FragCoord.y), logy = u_jk[j].z, band = u_jk[j].y;
  ivec4 jx = u_jx[j];
  vec4 r = vec4(-${BIG});
  float y;
  if (u_agg == 0 && texelFetch(u_info, at(line), 0).x < 0) { o = r; return; }
  for (int b = u_agg == 0 ? jx.z : 0; b <= (u_agg == 0 ? jx.w : jx.y - 1); b++) {
    vec4 s = texelFetch(u_src, ivec2(jx.x + b, line), 0);
    if (drawn(s.x, logy, y)) r.xy = max(r.xy, vec2(s.x, -s.x));
    if (u_agg == 0 || band == 5) continue;
    float lo = logy == 1 && !(s.y > 0.0) ? s.x : s.y;
    if (drawn(lo, logy, y)) r.zw = max(r.zw, vec2(lo, -lo));
    if (drawn(s.z, logy, y)) r.zw = max(r.zw, vec2(s.z, -s.z));
  }
  o = r;
}`;

// The largest of each FOLD texels along each row of u_src (u_count of them a row), into the target from u_at on.
const FOLD_FS = `${COMMON}
uniform highp sampler2D u_src;
uniform int u_count;
uniform ivec2 u_at;
out vec4 o;
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy) - u_at;
  vec4 r = vec4(-${BIG});
  for (int k = p.x * ${FOLD}; k < min(u_count, (p.x + 1) * ${FOLD}); k++) r = max(r, texelFetch(u_src, ivec2(k, p.y), 0));
  o = r;
}`;

let ctx = null; // the programs and textures of the renderer's context they were made in
let failed = false; // a round threw (a shader this GPU does not take, say): the workers bin from then on

/** The GPU binning state for the shared renderer, or null when it cannot bin: no WebGL2, a lost context, no float
 * render targets with float blending, or a round that failed. */
function state() {
  const r = renderer();
  if (!r || r.lost || failed) return null;
  if (ctx?.gen === r.gen) return ctx;
  const gl = r.gl;
  if (!gl.getExtension("EXT_color_buffer_float") || !gl.getExtension("EXT_float_blend")) return null;
  const quad = (fs) => compile(gl, QUAD_VS, fs);
  const progs = { scatter: compile(gl, SCATTER_VS, VALUE_FS), gatherSums: quad(GATHER_SUMS_FS), points: compile(gl, POINTS_VS, VALUE_FS), fill: quad(FILL_FS),
                  spread: quad(SPREAD_FS), sort: quad(SORT_FS), stats: quad(STATS_FS), statsSmall: quad(STATS_SMALL_FS),
                  lines: quad(LINES_FS), lineRange: quad(LINE_RANGE_FS), fold: quad(FOLD_FS) };
  ctx = { r, gl, gen: r.gen, most: Math.min(MAX_SLOTS, r.maxRows), progs, pool: new Map(), fbos: new Map(), ci: null, ciN: 0, tab: null,
          arrays: new Map(), arrayBytes: 0, use: 0, // bucket array -> its GPU copy, the least recently used first; their bytes; a count of uses
          means: new Map(), meansBytes: 0, // what a job's means were made of -> their texture, the least recently used first; their bytes
          rounds: new Map(), roundBytes: 0, // the same of whole rounds: what their jobs' means were made of, in order
          ahead: new Set() }; // the keys of the means the last jobs binned ahead made, until a job not ahead uses them
  return ctx;
}

/** Program `name`, linked, in use; its uniforms' locations. */
function use(c, name) {
  const p = linked(c.gl, c.progs[name]);
  c.gl.useProgram(p.p);
  return p.u;
}

/** A texture of internal format `ifmt` (`fmt`, `type`), w × h: one the pool keeps, or a new one. */
function take(c, ifmt, fmt, type, w, h) {
  const key = `${ifmt}|${w}|${h}`, kept = c.pool.get(key)?.pop();
  if (kept) return kept;
  c.gl.activeTexture(c.gl.TEXTURE0 + SCRATCH);
  return { key, tex: c.r.texture(ifmt, w, h, fmt, type, null), w, h };
}

/** Give texture t back to the pool (a few of each size kept; the rest deleted). */
function give(c, t) {
  let free = c.pool.get(t.key);
  if (!free) c.pool.set(t.key, (free = []));
  if (free.length < 4) return free.push(t);
  drop(c, t);
}

function drop(c, t) {
  const f = c.fbos.get(t.tex);
  if (f) c.gl.deleteFramebuffer(f), c.fbos.delete(t.tex);
  c.gl.deleteTexture(t.tex);
}

/** Render into texture t: its whole area, or w × h of it from (x, y) on. */
function target(c, t, x = 0, y = 0, w = t.w, h = t.h) {
  const gl = c.gl;
  let f = c.fbos.get(t.tex);
  if (!f) {
    f = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, f);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, t.tex, 0);
    c.fbos.set(t.tex, f);
  }
  gl.bindFramebuffer(gl.FRAMEBUFFER, f);
  gl.viewport(x, y, w, h);
}

/** Bind texture `tex` to unit `unit` as sampler uniform `loc`. */
function bind(c, unit, tex, loc) {
  const gl = c.gl;
  gl.activeTexture(gl.TEXTURE0 + unit);
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.uniform1i(loc, unit);
}

/** A texture of `data` (`per` numbers a texel), internal format `ifmt`, in rows `width` wide. */
function tableTexture(c, ifmt, fmt, type, data, per, width = TW) {
  const n = data.length / per, h = Math.max(1, Math.ceil(n / width)), t = take(c, ifmt, fmt, type, width, h);
  const full = new data.constructor(per * width * h);
  full.set(data);
  c.gl.activeTexture(c.gl.TEXTURE0 + SCRATCH);
  c.gl.bindTexture(c.gl.TEXTURE_2D, t.tex);
  c.gl.texSubImage2D(c.gl.TEXTURE_2D, 0, 0, 0, width, h, fmt, type, full);
  return t;
}

/** The GPU copy of bucket array `a`: its bytes as 32-bit words, and where its sections begin; made on first use (or
 * ahead of it, `warmArrays`) and kept until `forgetArray`, or until others used since take ARRAY_BYTES. */
function arrayTextures(c, a) {
  let u = c.arrays.get(a);
  if (u) {
    c.arrays.delete(a); // the most recently used last
    c.arrays.set(a, u);
    u.use = c.use;
    return u;
  }
  const v = a.v, gl = c.gl, words = new Uint32Array(v.first.buffer, 0, Math.ceil(v.bytes / 4)), rows = Math.max(1, Math.ceil(words.length / TW));
  gl.activeTexture(gl.TEXTURE0 + SCRATCH);
  const tex = c.r.texture(gl.R32UI, TW, rows, gl.RED_INTEGER, gl.UNSIGNED_INT, null), full = Math.floor(words.length / TW), rest = words.length - full * TW;
  if (full) gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, TW, full, gl.RED_INTEGER, gl.UNSIGNED_INT, words, 0);
  if (rest) gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, full, rest, 1, gl.RED_INTEGER, gl.UNSIGNED_INT, words, full * TW);
  const w = (view) => view.byteOffset / 4;
  u = { tex, bytes: 4 * TW * rows, use: c.use, rows: null, rowsVer: -1, slots: null, slotsSig: "", first: w(v.first), off: w(v.offset), soff: w(v.soff),
        mean: w(v.mean), tmean: w(v.tmean), n: w(v.n) };
  c.arrays.set(a, u);
  c.arrayBytes += u.bytes;
  for (const [old, t] of c.arrays) {
    if (c.arrayBytes <= ARRAY_BYTES || t.use === c.use) break; // none in use now is freed
    freeArray(c, old, t);
  }
  return u;
}

function freeArray(c, a, u) {
  c.gl.deleteTexture(u.tex);
  for (const t of [u.rows, u.slots]) if (t) give(c, t);
  c.arrays.delete(a);
  c.arrayBytes -= u.bytes;
}

/** Copy to the GPU, ahead of their use, those of bucket `arrays` not there yet (with each run index's row for `slots`
 * runs, as binning by step reads them), until about `budget` bytes are copied: whether all of them are there. */
export function warmArrays(arrays, slots, budget) {
  const c = state();
  if (!c) return true;
  c.use++;
  let sent = 0;
  for (const a of arrays) {
    if (c.arrays.has(a)) continue;
    if (sent >= budget) return false;
    arraySlots(c, a, arrayTextures(c, a), Math.max(1, slots));
    sent += a.v.bytes;
  }
  return true;
}

/** The texture of array a's rows' run indices (-1: a row another array holds now), made anew when they change. */
function arrayRows(c, a, u) {
  if (u.rowsVer === a.rowsVer && u.rows) return u.rows;
  if (u.rows) give(c, u.rows);
  u.rows = tableTexture(c, c.gl.R32I, c.gl.RED_INTEGER, c.gl.INT, a.rowRun, 1);
  u.rowsVer = a.rowsVer;
  return u.rows;
}

/** The texture of each run index's row in array a (-1: not in it), for n runs, made anew when its rows' runs change. */
function arraySlots(c, a, u, n) {
  const sig = `${a.rowsVer}|${n}`;
  if (u.slotsSig === sig) return u.slots;
  if (u.slots) give(c, u.slots);
  const row = new Int32Array(Math.max(1, n)).fill(-1);
  for (let r = 0; r < a.rowRun.length; r++) if (a.rowRun[r] >= 0 && a.rowRun[r] < n) row[a.rowRun[r]] = r;
  u.slots = tableTexture(c, c.gl.R32I, c.gl.RED_INTEGER, c.gl.INT, row, 1);
  u.slotsSig = sig;
  return u.slots;
}

/** The run table's textures (app.js buildRunTable): each run's (group, column), the groups' (start, size), their
 * members, each place's group; with how many groups there are and the largest's size (`most`). Made once a version. */
function tabTextures(c, tab) {
  if (c.tab?.ver === tab.ver) return c.tab;
  if (c.tab) for (const t of [c.tab.info, c.tab.starts, c.tab.members, c.tab.placeGroup]) give(c, t);
  const gl = c.gl, info = new Int32Array(2 * Math.max(1, tab.n)), placeGroup = new Int32Array(Math.max(1, tab.members.length));
  for (let i = 0; i < tab.n; i++) (info[2 * i] = tab.group[i]), (info[2 * i + 1] = tab.column[i]);
  let most = 0;
  for (let g = 0; 2 * g < tab.starts.length; g++) {
    const st = tab.starts[2 * g], n = tab.starts[2 * g + 1];
    placeGroup.fill(g, st, st + n);
    most = Math.max(most, n);
  }
  c.tab = { ver: tab.ver, n: tab.n, places: Math.max(1, tab.members.length), groups: Math.max(1, tab.starts.length / 2), most,
            info: tableTexture(c, gl.RG32I, gl.RG_INTEGER, gl.INT, info, 2),
            starts: tableTexture(c, gl.RG32I, gl.RG_INTEGER, gl.INT, tab.starts.length ? tab.starts : new Int32Array(2), 2),
            members: tableTexture(c, gl.R32I, gl.RED_INTEGER, gl.INT, tab.members.length ? tab.members : new Int32Array(1), 1),
            placeGroup: tableTexture(c, gl.R32I, gl.RED_INTEGER, gl.INT, placeGroup, 1) };
  return c.tab;
}

/** Free the GPU copy of bucket array `a` (the page dropped it). */
export function forgetArray(a) {
  const u = ctx?.arrays.get(a);
  if (u) freeArray(ctx, a, u);
}

/** The medianCiRank table for counts up to n, as a texture. */
function ciTable(c, n) {
  if (c.ci && c.ciN >= n) return c.ci;
  const N = Math.max(n, 2 * c.ciN, 64), k = new Int32Array(N + 1);
  for (let i = 1; i <= N; i++) k[i] = medianCiRank(i);
  if (c.ci) drop(c, c.ci);
  c.ci = tableTexture(c, c.gl.R32I, c.gl.RED_INTEGER, c.gl.INT, k, 1);
  c.ciN = N;
  return c.ci;
}

const VALUES_BYTES = 32 << 20; // the values of one binning the CPU keeps a copy of for its tooltips, at most (`GpuLines.values`)

/** A chart's results on the GPU, its share of the round it was binned in: the points it draws (`tex`, as gl.js
 * `Points` hold them, its own from point `first` on), and for its tooltips the statistics (center, lo, hi, n per bin
 * and group) or bin means (per bin and run) they were drawn from, in the round's columns from `x` on. */
export class GpuLines {
  constructor() {
    this.round = null; // {pts, src (textures), refs (the charts sharing them), gen}
    this.first = 0;
    this.x = 0;
    this.copy = null; // {w, h, vals}: its values as the CPU holds them, once read (`values`)
    this.pending = null; // {w, h, buf, sync, gen}: their read into a pixel-pack buffer, under way (`fetch`)
  }

  /** The texture of the points a chart draws, for gl.js draws. */
  get tex() {
    return this.round?.pts.tex ?? null;
  }

  /** Whether its textures still exist (the context was not lost since they were made). */
  get live() {
    return this.round !== null && ctx?.gen === this.round.gen;
  }

  /** Row `line` of the chart's source, `n` bins of it ([center, lo, hi, n] per bin of an "agg" job, a bin mean each of
   * a "rows" job), read back; null when it is gone. */
  row(line, n) {
    return this.read(this.x, line, n, 1);
  }

  /** Bin `b` of every line (`n` of them), read back as row() reads; null when it is gone. */
  column(b, n) {
    return this.read(this.x + b, 0, 1, n);
  }

  read(x, y, w, h) {
    if (!this.live) return null;
    const c = ctx, gl = c.gl, buf = new Float32Array(4 * w * h);
    target(c, this.round.src);
    gl.readPixels(x, y, w, h, gl.RGBA, gl.FLOAT, buf);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return buf;
  }

  /** Begin reading its values (w bins of h lines) into the CPU without waiting for the GPU: into a pixel-pack buffer,
   * behind a fence, which `values` takes them from once the GPU has passed it. */
  fetch(w, h) {
    if (!this.live || this.copy || this.pending || 16 * w * h > VALUES_BYTES) return;
    const c = ctx, gl = c.gl, buf = gl.createBuffer();
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, buf);
    gl.bufferData(gl.PIXEL_PACK_BUFFER, 16 * w * h, gl.STREAM_READ);
    target(c, this.round.src);
    gl.readPixels(this.x, 0, w, h, gl.RGBA, gl.FLOAT, 0);
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    this.pending = { w, h, buf, sync: gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0), gen: c.gen };
    gl.flush();
  }

  /** Its values, w bins of h lines (4 numbers a bin, line after line), as the CPU holds them, read once and never
   * waited for: undefined while their read is under way (`fetch`, begun here when none is); null when they are gone,
   * or more than VALUES_BYTES (read a bin or a line at a time then: `column`, `row`). */
  values(w, h) {
    if (this.copy?.w === w && this.copy.h === h) return this.copy.vals;
    if (!this.live || 16 * w * h > VALUES_BYTES) return null;
    const gl = ctx.gl, p = this.pending;
    if (p?.w !== w || p.h !== h) {
      this.forget();
      this.fetch(w, h);
      return undefined;
    }
    if (![gl.ALREADY_SIGNALED, gl.CONDITION_SATISFIED].includes(gl.clientWaitSync(p.sync, 0, 0))) return undefined;
    const vals = new Float32Array(4 * w * h);
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, p.buf);
    gl.getBufferSubData(gl.PIXEL_PACK_BUFFER, 0, vals);
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
    this.dropFetch();
    this.copy = { w, h, vals };
    return vals;
  }

  /** Give up the read `fetch` began. */
  dropFetch() {
    const p = this.pending;
    this.pending = null;
    if (p && ctx?.gen === p.gen) ctx.gl.deleteBuffer(p.buf), ctx.gl.deleteSync(p.sync);
  }

  /** Give up its values as the CPU holds them, and their read under way: no tooltip reads them any more. */
  forget() {
    this.copy = null;
    this.dropFetch();
  }

  /** Take a share of `round`: its points from `first` on, its columns from `x` on. */
  share(round, first, x) {
    this.release();
    round.refs++;
    Object.assign(this, { round, first, x });
  }

  /** Give its share up; the round's textures go once no chart shares them. */
  release() {
    const r = this.round;
    this.round = null;
    this.forget();
    if (r && --r.refs === 0 && ctx?.gen === r.gen) give(ctx, r.pts), give(ctx, r.src);
  }
}

const queue = []; // jobs queued for the next `runGpuJobs`

/** Whether the GPU bins with binning p ({xmode, x0, x1, bins, flags, alpha, scale}) the runs of run table `tab`:
 * smoothed values and very many runs or bins are left to the workers. */
export function gpuBins(tab, p) {
  const c = p.alpha <= 0 && p.bins <= MAX_BINS ? state() : null;
  return !!c && Math.max(tab.n, tab.starts.length / 2, tab.members.length) <= c.most;
}

/** Queue a binning into `out` (a GpuLines) for the next `runGpuJobs`, of `src` {arrays (bucket arrays, each with its
 * rows' run indices: Data rowRun), cols ([run index, column] of the runs drawn from their columns), tab (the run table,
 * app.js buildRunTable)}: kind "agg", each group's center and band (p.center, p.band) as aggGroups and bandOf make them
 * (one line per group); or "rows", each run's bin means as binRows makes them (one line per run index). Points are
 * drawn from the left edge of the first bin, y as on an axis that is logarithmic when p.logy, x at most p.xmax (from
 * that edge). `ahead`: the binning is of a view the chart may come to show (a zoom being dragged), and its means are
 * kept only until the next such binning's are made. `done({range})` gets the y range ([center lo, hi, band lo, hi] of "agg", [lo, hi] of the bins whose
 * centers lie in [p.b0, p.b1] of "rows"; Infinity and -Infinity when empty), or null when the GPU could not. */
export function queueGpu(kind, src, p, out, done, ahead = false) {
  queue.push({ kind, src, p, out, done, ahead });
}

/** Binnings queued and not yet run. */
export const gpuQueued = () => queue.length;

/** Run every queued binning, in rounds; the y ranges are read back at once and handed to their `done`: null to every
 * job when the GPU cannot bin or a round throws, after which it bins no more (the charts' workers do). */
export function runGpuJobs() {
  const jobs = queue.splice(0), c = state();
  let ranges = null;
  try {
    if (c && jobs.length) ranges = rangesOf(c, jobs);
  } catch (e) {
    failed = true;
    c.gl.bindFramebuffer(c.gl.FRAMEBUFFER, null);
    console.warn("trex: binning on the GPU failed, the workers bin from now on:", e);
  }
  jobs.forEach((j, i) => j.done(ranges && { range: rangeOf(ranges, i, j.kind) }));
}

/** The y ranges of `jobs` (four numbers each, for `rangeOf`), binned in rounds. */
function rangesOf(c, jobs) {
  const gl = c.gl, ranges = take(c, gl.RGBA32F, gl.RGBA, gl.FLOAT, 1, MAX_JOBS * Math.ceil(jobs.length / MAX_JOBS)), temps = [ranges];
  gl.disable(gl.SCISSOR_TEST);
  gl.disable(gl.DEPTH_TEST);
  gl.disable(gl.BLEND);
  gl.bindVertexArray(c.r.vao);
  c.use++;
  const ahead = jobs.filter((j) => j.ahead);
  if (ahead.length) dropAhead(c, ahead.map(meansKey)); // the means of the binnings ahead before these
  let at = 0;
  for (const round of roundsOf(jobs, Math.min(MAX_COLS, c.r.maxRows))) {
    runRound(c, round, at, ranges, temps);
    at += round.length;
  }
  const buf = new Float32Array(4 * jobs.length);
  target(c, ranges);
  gl.readPixels(0, 0, 1, jobs.length, gl.RGBA, gl.FLOAT, buf);
  for (const t of temps) give(c, t);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  return buf;
}

/** `jobs` in order as rounds: runs of jobs of one kind and one run table, all ahead or none, at most MAX_JOBS of them
 * and `most` bins. */
function roundsOf(jobs, most) {
  const out = [];
  let cols = 0;
  for (const j of jobs) {
    const last = out.at(-1), head = last?.[0], alike = head && head.kind === j.kind && head.src.tab === j.src.tab && head.ahead === j.ahead;
    if (alike && last.length < MAX_JOBS && cols + j.p.bins <= most) last.push(j), (cols += j.p.bins);
    else out.push([j]), (cols = j.p.bins);
  }
  return out;
}

/** Job i's y range from the ranges read back. */
function rangeOf(buf, i, kind) {
  const v = buf.subarray(4 * i, 4 * i + 4), end = (x) => (x <= -BIG ? -Infinity : x);
  const r = [-end(v[1]), end(v[0])];
  return kind === "agg" ? [...r, -end(v[3]), end(v[2])] : r;
}

/** One round: its jobs' runs binned into means; for "agg" jobs their groups' statistics; then the points the charts
 * draw, shared by them, and each job's range into its row of `ranges` (from row `at`). */
function runRound(c, jobs, at, ranges, temps) {
  const agg = jobs[0].kind === "agg", ahead = jobs[0].ahead, tab = tabTextures(c, jobs[0].src.tab);
  const lay = layout(c, jobs, agg ? tab.groups : Math.max(1, tab.n), temps);
  const means = meansOf(c, jobs, lay, tab, temps, agg && !ahead, ahead), src = agg ? groupStats(c, lay, tab, means) : means;
  if (agg && ahead) temps.push(means);
  const round = { pts: lines(c, lay, src, agg), src, refs: 0, gen: c.gen };
  jobs.forEach((j, i) => j.out.share(round, lay.jk[4 * i + 3], lay.jx[4 * i]));
  range(c, lay, tab, src, agg, at, ranges, temps);
}

/** Where a round's jobs lie in its textures: `cols` columns in all (job i's bins from column jx[4 i] on), `points`
 * points (job i's from jk[4 i + 3] on, `count` lines each), and the job table for the passes (JOBS: jx, jk, jf, and the
 * texture of each column's job). */
function layout(c, jobs, count, temps) {
  const jx = new Int32Array(4 * MAX_JOBS), jk = new Int32Array(4 * MAX_JOBS), jf = new Float32Array(2 * MAX_JOBS), agg = jobs[0].kind === "agg";
  let x = 0, points = 0;
  jobs.forEach(({ p }, i) => {
    jx.set([x, p.bins, p.b0 ?? 0, p.b1 ?? p.bins - 1], 4 * i);
    jk.set([CENTERS[p.center] ?? 1, BANDS[p.band] ?? 0, p.logy ? 1 : 0, points], 4 * i);
    jf.set([(p.x1 - p.x0) / p.bins, p.xmax ?? 1e38], 2 * i);
    x += p.bins;
    points += count * (agg ? 3 : 1) * p.bins;
  });
  const cols = COLS * Math.ceil(x / COLS), col = new Int32Array(cols).fill(-1);
  jobs.forEach(({ p }, i) => col.fill(i, jx[4 * i], jx[4 * i] + p.bins));
  const colTex = tableTexture(c, c.gl.R32I, c.gl.RED_INTEGER, c.gl.INT, col, 1, cols);
  temps.push(colTex);
  return { n: jobs.length, count, cols, points, jx, jk, jf, col: colTex };
}

/** Hand the round's job table to the program in use (uniforms u), its column texture on `unit`. */
function jobsTo(c, u, lay, unit) {
  const gl = c.gl;
  bind(c, unit, lay.col.tex, u.u_col);
  gl.uniform4iv(u["u_jx[0]"], lay.jx);
  gl.uniform4iv(u["u_jk[0]"], lay.jk);
  gl.uniform2fv(u["u_jf[0]"], lay.jf);
}

/** The points of columns `cols` ([run index, column]) inside the binning's range: (bin, value, count, run index). */
function columnPoints(cols, p) {
  const logx = (p.flags & LOGX) !== 0, per = p.bins / (p.x1 - p.x0);
  let total = 0;
  for (const [, col] of cols) total += col.n;
  const out = new Float32Array(4 * Math.max(1, total));
  let k = 0;
  for (const [slot, col] of cols) {
    const xs = col.xs(p.xmode), ys = col.v, ws = col.w;
    for (let i = 0; i < col.n; i++) {
      const x = logx ? (xs[i] > 0 ? Math.log10(xs[i]) : NaN) : xs[i], y = ys[i];
      if (!(x >= p.x0 && x <= p.x1) || y !== y) continue;
      (out[4 * k] = Math.min(p.bins - 1, Math.floor((x - p.x0) * per))), (out[4 * k + 1] = y), (out[4 * k + 2] = ws ? ws[i] : 1), (out[4 * k + 3] = slot);
      k++;
    }
  }
  return { data: out.subarray(0, 4 * k), n: k };
}

/** The texture of every job's bin means, side by side. A job's are kept, in a texture of their own, for later rounds
 * of the same data and binning (job.src.data, job.p) over the same runs drawn from columns, whichever of them are shown
 * or grouped, and copied from there; only the others are added up. When the round needs the texture no longer than
 * its statistics take (`whole`), the texture is kept too, for a round of the same jobs in the same columns: a filter
 * or a regrouping of the charts in view then copies nothing. The means that jobs binned ahead make (`ahead`) are kept
 * until the next call binning ahead (`dropAhead`), unless a job not ahead uses them first: a drag may bin a view at
 * each rest, and only the last one's means may serve. */
function meansOf(c, jobs, lay, tab, temps, whole, ahead) {
  const gl = c.gl, slots = Math.max(1, tab.n), keys = jobs.map(meansKey);
  const all = `${lay.cols}\n${keys.join("\n")}`, had = whole ? c.rounds.get(all) : undefined;
  if (had) {
    c.rounds.delete(all); // the most recently used last
    c.rounds.set(all, had);
    return had;
  }
  const means = take(c, gl.R32F, gl.RED, gl.FLOAT, lay.cols, slots);
  const kept = keys.map((k) => c.means.get(k)), made = jobs.flatMap((_, i) => (kept[i] ? [] : [i]));
  if (made.length) binned(c, jobs, made, lay, tab, means, temps);
  kept.forEach((t, i) => {
    if (!t) return;
    c.means.delete(keys[i]); // the most recently used last
    c.means.set(keys[i], t);
    if (!ahead) c.ahead.delete(keys[i]);
    copy(c, t, 0, means, lay.jx[4 * i], t.w, slots);
  });
  for (const i of made) {
    const t = take(c, gl.R32F, gl.RED, gl.FLOAT, jobs[i].p.bins, slots);
    copy(c, means, lay.jx[4 * i], t, 0, t.w, slots);
    c.means.set(keys[i], t);
    c.meansBytes += 4 * t.w * t.h;
    if (ahead) c.ahead.add(keys[i]);
    else c.ahead.delete(keys[i]);
  }
  for (const [k, t] of c.means) {
    if (c.meansBytes <= MEANS_BYTES || keys.includes(k)) break;
    c.means.delete(k);
    c.ahead.delete(k);
    c.meansBytes -= 4 * t.w * t.h;
    give(c, t);
  }
  if (whole) keepRound(c, all, means);
  return means;
}

/** What a job's bin means are made of, as text: the runs' slots, which of them are binned from columns, the data and
 * the binning. */
const meansKey = ({ src, p }) => `${Math.max(1, src.tab.n)}|${src.tab.columns}|${src.data}|${p.xmode}|${p.x0}|${p.x1}|${p.bins}|${p.flags & LOGX}`;

/** Give up the means that jobs binned ahead made and no other job has used, but those of `keys`. */
function dropAhead(c, keys) {
  for (const k of c.ahead) {
    if (keys.includes(k)) continue;
    const t = c.means.get(k);
    c.ahead.delete(k);
    if (!t) continue;
    c.means.delete(k);
    c.meansBytes -= 4 * t.w * t.h;
    give(c, t);
  }
}

/** Keep a round's means `t` under what they were made of, freeing the rounds used least recently beyond ROUND_BYTES. */
function keepRound(c, key, t) {
  c.rounds.set(key, t);
  c.roundBytes += 4 * t.w * t.h;
  for (const [k, old] of c.rounds) {
    if (c.roundBytes <= ROUND_BYTES || old === t) break;
    c.rounds.delete(k);
    c.roundBytes -= 4 * old.w * old.h;
    give(c, old);
  }
}

/** Copy columns [sx, sx + w) of texture `from` (h rows) to `to` from column dx on. */
function copy(c, from, sx, to, dx, w, h) {
  const gl = c.gl;
  target(c, to);
  target(c, from);
  gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, c.fbos.get(to.tex));
  gl.blitFramebuffer(sx, 0, sx + w, h, dx, 0, dx + w, h, gl.COLOR_BUFFER_BIT, gl.NEAREST);
}

/** Add up the runs per bin of the jobs `made` (their places among `jobs`) into sums, and make them bin means in
 * their columns of `means`. */
function binned(c, jobs, made, lay, tab, means, temps) {
  const gl = c.gl, slots = Math.max(1, tab.n), sums = take(c, gl.RGBA32F, gl.RGBA, gl.FLOAT, lay.cols, slots);
  temps.push(sums);
  target(c, sums);
  gl.clearColor(0, 0, 0, 0);
  gl.clear(gl.COLOR_BUFFER_BIT);
  gl.enable(gl.BLEND);
  gl.blendEquation(gl.FUNC_ADD);
  gl.blendFunc(gl.ONE, gl.ONE);
  for (const i of made) {
    addArrays(c, jobs[i], lay, i, tab, slots, sums);
    addColumns(c, jobs[i], lay, i, slots, sums, temps);
  }
  gl.disable(gl.BLEND);
  const u = use(c, "fill");
  target(c, means);
  bind(c, 0, sums.tex, u.u_sums);
  jobsTo(c, u, lay, 1);
  gl.drawArrays(gl.TRIANGLES, 0, 3);
}

/** Add the points of job i's columns to their runs' sums. */
function addColumns(c, job, lay, i, slots, sums, temps) {
  const gl = c.gl, points = columnPoints(job.src.cols, job.p);
  if (!points.n) return;
  const pts = tableTexture(c, gl.RGBA32F, gl.RGBA, gl.FLOAT, points.data, 4), u = use(c, "points");
  temps.push(pts);
  target(c, sums);
  bind(c, 0, pts.tex, u.u_pts);
  gl.uniform1i(u.u_x, lay.jx[4 * i]);
  gl.uniform1i(u.u_cols, lay.cols);
  gl.uniform1i(u.u_slots, slots);
  gl.drawArrays(gl.POINTS, 0, points.n);
}

/** Add the buckets of job i's bucket arrays to their runs' sums: gathered per bin when binned by step, else a point
 * each. */
function addArrays(c, job, lay, i, tab, slots, sums) {
  const gl = c.gl, p = job.p, x = lay.jx[4 * i], dx = (p.x1 - p.x0) / p.bins, logx = (p.flags & LOGX) !== 0, runtime = p.xmode !== X_STEP;
  for (const a of job.src.arrays) {
    const t = arrayTextures(c, a), map = runtime ? arrayRows(c, a, t) : arraySlots(c, a, t, slots);
    const w = 2 ** a.v.level, u = use(c, runtime ? "scatter" : "gatherSums"), first = (a.v.base * BLOCK * w - p.x0) / dx;
    target(c, sums);
    bind(c, 0, t.tex, u.u_raw);
    bind(c, 1, map.tex, runtime ? u.u_rows : u.u_slot);
    bind(c, 2, tab.info.tex, u.u_info);
    gl.uniform1i(u.u_first, t.first);
    gl.uniform1i(u.u_off, t.off);
    gl.uniform1i(u.u_soff, t.soff);
    gl.uniform1i(u.u_mean, t.mean);
    gl.uniform1i(u.u_n, t.n);
    gl.uniform1i(u.u_mode, (runtime ? 1 : 0) | (logx ? 2 : 0));
    gl.uniform1i(u.u_bins, p.bins);
    gl.uniform1i(u.u_x, x);
    gl.uniform1f(u.u_x0, p.x0);
    gl.uniform1f(u.u_dx, dx);
    gl.uniform1f(u.u_base, a.v.base * BLOCK);
    gl.uniform1f(u.u_w, w);
    if (runtime) {
      gl.uniform1i(u.u_runs, a.v.runs);
      gl.uniform1i(u.u_tmean, t.tmean);
      gl.uniform1i(u.u_cols, lay.cols);
      gl.uniform1i(u.u_slots, slots);
      gl.drawArrays(gl.POINTS, 0, a.v.count);
      continue;
    }
    // only the bins the block's steps reach, on a linear axis
    const b0 = logx ? 0 : Math.max(0, Math.floor(first)), b1 = logx ? p.bins : Math.min(p.bins, Math.floor(first + BLOCK * (w / dx)) + 1);
    if (b1 <= b0) continue;
    gl.uniform1f(u.u_a, first);
    gl.uniform1f(u.u_b, w / dx);
    gl.enable(gl.SCISSOR_TEST);
    gl.scissor(x + b0, 0, b1 - b0, slots);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.disable(gl.SCISSOR_TEST);
  }
}

/** Summarize each group's bins from its runs' bin means: the texture of (center, lo, hi, n) per column and group.
 * Small groups are put in order where they are summarized; larger ones are put in order first (`inOrder`). */
function groupStats(c, lay, tab, means) {
  const gl = c.gl, small = tab.most <= SMALL, stats = take(c, gl.RGBA32F, gl.RGBA, gl.FLOAT, lay.cols, tab.groups);
  const sorted = small ? null : inOrder(c, lay, tab, means), ci = ciTable(c, tab.most), u = use(c, small ? "statsSmall" : "stats");
  target(c, stats);
  if (small) bind(c, 0, means.tex, u.u_m), bind(c, 3, tab.members.tex, u.u_members);
  else bind(c, 0, sorted.tex, u.u_v);
  bind(c, 1, tab.starts.tex, u.u_groups);
  bind(c, 2, ci.tex, u.u_ci);
  jobsTo(c, u, lay, 4);
  gl.uniform1fv(u["u_t95[0]"], T95);
  gl.drawArrays(gl.TRIANGLES, 0, 3);
  if (sorted) give(c, sorted);
  return stats;
}

/** Each group's values in each column put in order into its rows of a texture: its members' means spread over its
 * rows, then ordered by the steps of a sorting network (a bitonic sort), each one pass over every group and column.
 * Every pass costs a few fetches a texel whatever the groups' sizes, and none scatters points. */
function inOrder(c, lay, tab, means) {
  const gl = c.gl, size = [gl.R32F, gl.RED, gl.FLOAT, lay.cols, tab.places];
  let from = take(c, ...size), to = take(c, ...size), u = use(c, "spread");
  target(c, from);
  bind(c, 0, means.tex, u.u_m);
  bind(c, 1, tab.members.tex, u.u_members);
  gl.drawArrays(gl.TRIANGLES, 0, 3);
  u = use(c, "sort");
  bind(c, 1, tab.starts.tex, u.u_groups);
  bind(c, 2, tab.placeGroup.tex, u.u_placeGroup);
  const step = (block, mirror) => {
    target(c, to);
    bind(c, 0, from.tex, u.u_v);
    gl.uniform1i(u.u_block, block);
    gl.uniform1i(u.u_mirror, mirror);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    [from, to] = [to, from];
  };
  for (let merge = 2; merge < 2 * tab.most; merge *= 2) {
    step(merge, 1);
    for (let block = merge / 2; block >= 2; block /= 2) step(block, 0);
  }
  give(c, to);
  return from;
}

/** The points the round's charts draw, from the statistics (per group: center, band top, band bottom) or the means
 * (per run): their texture. */
function lines(c, lay, src, agg) {
  const gl = c.gl, rows = Math.max(1, Math.ceil(lay.points / TW)), pts = take(c, gl.RG32F, gl.RG, gl.FLOAT, TW, ROWS * Math.ceil(rows / ROWS)), u = use(c, "lines");
  target(c, pts, 0, 0, TW, rows);
  bind(c, 0, src.tex, u.u_src);
  jobsTo(c, u, lay, 1);
  gl.uniform1i(u.u_jobs, lay.n);
  gl.uniform1i(u.u_count, lay.count);
  gl.uniform1i(u.u_agg, agg ? 1 : 0);
  gl.drawArrays(gl.TRIANGLES, 0, 3);
  return pts;
}

/** Each job's y range into its row of `ranges` (from row `at`): each line's range, folded by FOLD along each job's
 * row until one is left. */
function range(c, lay, tab, src, agg, at, ranges, temps) {
  const gl = c.gl;
  let n = lay.count, per = take(c, gl.RGBA32F, gl.RGBA, gl.FLOAT, n, MAX_JOBS), u = use(c, "lineRange");
  temps.push(per);
  target(c, per, 0, 0, n, lay.n);
  bind(c, 0, src.tex, u.u_src);
  bind(c, 1, tab.info.tex, u.u_info);
  jobsTo(c, u, lay, 2);
  gl.uniform1i(u.u_agg, agg ? 1 : 0);
  gl.drawArrays(gl.TRIANGLES, 0, 3);
  u = use(c, "fold");
  for (;;) {
    const next = Math.ceil(n / FOLD), last = next === 1, out = last ? ranges : take(c, gl.RGBA32F, gl.RGBA, gl.FLOAT, next, MAX_JOBS);
    if (!last) temps.push(out);
    target(c, out, 0, last ? at : 0, next, lay.n);
    bind(c, 0, per.tex, u.u_src);
    gl.uniform1i(u.u_count, n);
    gl.uniform2i(u.u_at, 0, last ? at : 0);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    if (last) return;
    (per = out), (n = next);
  }
}
