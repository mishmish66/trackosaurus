// WebGL2 chart renderer. One shared offscreen context draws a chart's lines, group bands and
// density heatmap; the chart copies the result onto its own canvas. Points live in RG32F
// textures (x, y relative to an origin); a line table (offset, count, color) per draw lets one
// instanced multi-draw render every line as antialiased segment quads.
//# allFunctionsCalledOnLoad

const TW = 2048; // points per row of a point texture
const MW = 1024; // lines per row of a line table
const META_ROWS = 64; // rows of a line-table texture
const BREAK = 1e38; // coordinate marking a line break
const MAX_BINS = 16; // texels of the density maximum
export const DOT_PX = 3; // radius, in CSS px, of the dot a point is drawn as when no segment of its line reaches it

const ext = (gl, name) => gl.getExtension(name);

/** A program compiling and linking in the background (KHR_parallel_shader_compile); `linked` checks it. */
export function compile(gl, vs, fs) {
  const p = gl.createProgram(), shaders = [];
  for (const [type, src] of [[gl.VERTEX_SHADER, vs], [gl.FRAGMENT_SHADER, fs]]) {
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    gl.attachShader(p, s);
    shaders.push(s);
  }
  gl.linkProgram(p);
  return { p, shaders, u: null };
}

/** `prog` with its uniforms' locations, once it has linked; throws its compile errors when it failed. */
export function linked(gl, prog) {
  if (prog.u) return prog;
  const p = prog.p;
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
    throw new Error(prog.shaders.map((s) => gl.getShaderInfoLog(s)).filter(Boolean).join("\n") || gl.getProgramInfoLog(p));
  }
  const u = {};
  for (let i = 0, n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS); i < n; i++) {
    const name = gl.getActiveUniform(p, i).name;
    u[name] = gl.getUniformLocation(p, name);
  }
  prog.u = u;
  return prog;
}

const header = (multi) => `#version 300 es
${multi ? "#extension GL_ANGLE_multi_draw : require\n#define DRAW_ID gl_DrawID" : "#define DRAW_ID 0"}
precision highp float;
precision highp int;
uniform highp sampler2D u_pos;
uniform highp isampler2D u_meta;
uniform int u_base, u_row0, u_grid, u_first;
uniform vec2 u_off, u_scale, u_org, u_size;
vec2 fetch(int i) { i += u_first; return texelFetch(u_pos, ivec2(i % ${TW}, i / ${TW}), 0).xy; }
ivec4 meta(int line) { return texelFetch(u_meta, ivec2(line % ${MW}, u_row0 + line / ${MW}), 0); }
bool broken(vec2 p) { return abs(p.y) > 1.0e37 || abs(p.x) > 1.0e37; }
vec2 toPx(vec2 p) { return vec2(u_org.x + (p.x - u_off.x) * u_scale.x, u_org.y - (p.y - u_off.y) * u_scale.y); }
vec4 toClip(vec2 P) { return vec4(P.x / u_size.x * 2.0 - 1.0, 1.0 - P.y / u_size.y * 2.0, 0.0, 1.0); }
vec4 unpack(ivec4 m) { return vec4(float((m.z >> 16) & 255), float((m.z >> 8) & 255), float(m.z & 255), float(m.w)) / 255.0; }
`;

// Whether image point P lies outside the plot (u_inside).
const OUTSIDE = "bool outside(vec2 P) { return any(lessThan(P, u_inside.xy)) || any(greaterThan(P, u_inside.zw)); }";

// Segment quad: two triangles around segment i -> i+1, padded by half width plus the half pixel coverage reaches. Its
// line is the draw's, or with u_grid points a line, the instance's; an instance is a point of its line, drawn with the
// segment to the next one. A point no segment reaches (the only one of its line, or one between breaks) is a dot of
// radius u_dot instead: a segment of no length that wide, since a line of one point would show nothing; none when the
// point lies outside the plot (u_inside: its rectangle in the image, x0, y0, x1, y1).
const LINE_VS = (multi) => `${header(multi)}
uniform float u_half, u_dot, u_count;
uniform vec4 u_inside;
${OUTSIDE}
out vec2 v_uv;
flat out float v_len, v_depth, v_half, v_dot;
flat out vec4 v_color;
void main() {
  int line = u_grid > 0 ? gl_InstanceID / u_grid : DRAW_ID + u_base;
  ivec4 m = meta(line);
  int k = u_grid > 0 ? gl_InstanceID - line * u_grid : gl_InstanceID, i = m.x + k;
  vec2 a = fetch(i), b = k + 1 < m.y ? fetch(i + 1) : vec2(1.0e38);
  bool lone = !broken(a) && broken(b) && (k == 0 || broken(fetch(i - 1)));
  if (broken(a) || (broken(b) && !lone)) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }
  vec2 A = toPx(a), B = lone ? A : toPx(b), d = B - A;
  if (lone && outside(A)) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }
  float len = length(d);
  vec2 dir = len > 1e-4 ? d / len : vec2(1.0, 0.0), nrm = vec2(-dir.y, dir.x);
  int c = gl_VertexID;
  float end = (c == 1 || c == 4 || c == 5) ? 1.0 : 0.0;
  float side = (c == 2 || c == 3 || c == 5) ? 1.0 : -1.0;
  v_half = lone ? u_dot : u_half;
  v_dot = lone ? 1.0 : 0.0;
  float h = v_half + 0.5, s = end * 2.0 - 1.0;
  v_uv = vec2(end * len + s * h, side * h);
  v_len = len;
  v_color = unpack(m);
  v_depth = 1.0 - (float(line) + 1.0) / (u_count + 1.0);
  gl_Position = toClip(mix(A, B, end) + dir * s * h + nrm * side * h);
}`;

// Coverage of a round-capped segment of width 2*v_half, which ends at the plot's sides (u_plot: x0, y0, x1, y1 in the
// target), while a dot on one is drawn whole (`Renderer.lines` lets it reach past them). With `once`, depth decreases
// with draw order and coverage, so each line blends into a pixel about once (as a stroked path would): a translucent
// line's joints then show no beads. Writing depth for each fragment takes the GPU about half again as long, and an
// opaque line's joints blended twice differ only in their edges' few pixels, so opaque lines are drawn without
// (`Renderer.lines`).
const LINE_FS = (once) => `#version 300 es
precision highp float;
uniform float u_alpha, u_count;
uniform vec4 u_plot;
uniform int u_density;
in vec2 v_uv;
flat in float v_len, v_depth, v_half, v_dot;
flat in vec4 v_color;
out vec4 o;
void main() {
  if (v_dot == 0.0 && (any(lessThan(gl_FragCoord.xy, u_plot.xy)) || any(greaterThanEqual(gl_FragCoord.xy, u_plot.zw)))) discard;
  float u = v_uv.x, v = v_uv.y;
  float d = u < 0.0 ? length(vec2(u, v)) : u > v_len ? length(vec2(u - v_len, v)) : abs(v);
  float cov = clamp(v_half + 0.5 - d, 0.0, 1.0);
  if (cov <= 0.0) discard;
  ${once ? "gl_FragDepth = v_depth - cov * 0.5 / (u_count + 1.0);" : ""}
  if (u_density == 1) { o = vec4(cov, 0.0, 0.0, 1.0); return; }
  float al = cov * u_alpha * v_color.a;
  o = vec4(v_color.rgb * al, al);
}`;

// A heatmap's counts: line gl_InstanceID as a 1 px line strip through its points (none broken but all of a line's),
// each pixel it crosses getting one. With u_dots, as a point instead when its points all lie at one place (a run with
// a value in one bin only, whose other bins' points gpustats puts there): its strip has no length and draws nothing,
// so a dot of radius u_dot there counts it, when the place lies in the plot.
const COUNT_VS = `${header(false)}
uniform int u_dots;
uniform float u_dot;
uniform vec4 u_inside;
${OUTSIDE}
void main() {
  ivec4 m = meta(gl_InstanceID);
  vec2 p = fetch(m.x + gl_VertexID);
  bool none = broken(p) || (u_dots == 1 && (p != fetch(m.x + m.y - 1) || outside(toPx(p))));
  gl_Position = none ? vec4(2.0, 2.0, 2.0, 1.0) : toClip(toPx(p));
  gl_PointSize = 2.0 * u_dot;
}`;

// Reading the fragment's place (gl_PointCoord, for the dots) also makes the strips cheaper on a Radeon 760M: the 15
// heatmaps of 2048 runs a window shows take it 7.1 ms, and 12.7 with a shader that reads neither that nor
// gl_FragCoord (a constant, or varyings alone), for the same pixels.
const COUNT_FS = `#version 300 es
precision highp float;
precision highp int;
uniform int u_dots;
out vec4 o;
void main() {
  if (u_dots == 1 && length(gl_PointCoord - 0.5) > 0.5) discard;
  o = vec4(1.0, 0.0, 0.0, 1.0);
}`;

// Band quad between bins i and i+1: line-table entry (hi offset, bins, rgb, alpha) with the lo
// offset in the entry after it.
const BAND_VS = (multi) => `${header(multi)}
flat out vec4 v_color;
void main() {
  int band = u_grid > 0 ? gl_InstanceID / u_grid : DRAW_ID + u_base;
  ivec4 m = meta(2 * band), l = meta(2 * band + 1);
  int k = u_grid > 0 ? gl_InstanceID - band * u_grid : gl_InstanceID;
  vec2 h0 = fetch(m.x + k), h1 = fetch(m.x + k + 1), l0 = fetch(l.x + k), l1 = fetch(l.x + k + 1);
  if (broken(h0) || broken(h1) || broken(l0) || broken(l1)) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }
  int c = gl_VertexID;
  vec2 p = c == 0 ? l0 : (c == 1 || c == 4) ? l1 : (c == 2 || c == 3) ? h0 : h1;
  v_color = unpack(m);
  gl_Position = toClip(toPx(p));
}`;

const BAND_FS = `#version 300 es
precision highp float;
uniform float u_alpha;
flat in vec4 v_color;
out vec4 o;
void main() { float a = u_alpha * v_color.a; o = vec4(v_color.rgb * a, a); }`;

// Maximum of each TILE x TILE tile of the density texture's plot rectangle u_rect.
const TILE = 16;
const TILE_FS = `#version 300 es
precision highp float;
precision highp int;
uniform highp sampler2D u_dens;
uniform ivec4 u_rect;
out vec4 o;
void main() {
  ivec2 t = ivec2(gl_FragCoord.xy) * ${TILE}, end = min(t + ${TILE}, u_rect.zw);
  float m = 0.0;
  for (int y = t.y; y < end.y; y++) for (int x = t.x; x < end.x; x++) m = max(m, texelFetch(u_dens, u_rect.xy + ivec2(x, y), 0).r);
  o = vec4(m, 0.0, 0.0, 1.0);
}`;

// Maximum of the tiles' maxima (u_rect: their texture's), into MAX_BINS texels (MAX blending).
const MAX_VS = `#version 300 es
precision highp float;
precision highp int;
uniform highp sampler2D u_dens;
uniform ivec4 u_rect;
flat out float v_d;
void main() {
  int id = gl_VertexID;
  v_d = texelFetch(u_dens, u_rect.xy + ivec2(id % u_rect.z, id / u_rect.z), 0).r;
  gl_PointSize = 1.0;
  gl_Position = vec4((float(id % ${MAX_BINS}) + 0.5) / ${MAX_BINS}.0 * 2.0 - 1.0, 0.0, 0.0, 1.0);
}`;

const MAX_FS = `#version 300 es
precision highp float;
flat in float v_d;
out vec4 o;
void main() { o = vec4(v_d, 0.0, 0.0, 1.0); }`;

const QUAD_VS = `#version 300 es
void main() {
  vec2 q = vec2(gl_VertexID == 1 ? 3.0 : -1.0, gl_VertexID == 2 ? 3.0 : -1.0);
  gl_Position = vec4(q, 0.0, 1.0);
}`;

// Log-scaled density colormap (viridis; reversed lightness ramp on light backgrounds).
const CMAP_FS = `#version 300 es
precision highp float;
uniform highp sampler2D u_dens, u_max;
uniform ivec2 u_at;
uniform int u_dark;
out vec4 o;
vec3 viridis(float t) {
  const vec3 c0 = vec3(0.2777273272234177, 0.005407344544966578, 0.3340998053353061);
  const vec3 c1 = vec3(0.1050930431085774, 1.404613529898575, 1.384590162594685);
  const vec3 c2 = vec3(-0.3308618287255563, 0.214847559468213, 0.09509516302823659);
  const vec3 c3 = vec3(-4.634230498983486, -5.799100973351585, -19.33244095627987);
  const vec3 c4 = vec3(6.228269936347081, 14.17993336680509, 56.69055260068105);
  const vec3 c5 = vec3(4.776384997670288, -13.74514537774601, -65.35303263337234);
  const vec3 c6 = vec3(-5.435455855934631, 4.645852612178535, 26.3124352495832);
  return c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6)))));
}
void main() {
  float d = texelFetch(u_dens, ivec2(gl_FragCoord.xy) - u_at, 0).r;
  if (d <= 0.004) discard;
  float mx = 0.0;
  for (int i = 0; i < ${MAX_BINS}; i++) mx = max(mx, texelFetch(u_max, ivec2(i, 0), 0).r);
  float t = clamp(log(1.0 + d) / log(1.0 + max(mx, 1.0)), 0.0, 1.0);
  vec3 c = viridis(u_dark == 1 ? t : 1.0 - t);
  float a = clamp(d, 0.0, 1.0) * (u_dark == 1 ? mix(0.55, 1.0, t) : mix(0.45, 1.0, t));
  o = vec4(c * a, a);
}`;

/** A dot's radius in device px. */
const dotRadius = () => DOT_PX * (globalThis.devicePixelRatio || 1);

/** RGBA bytes of a CSS hex or rgb() color. */
const colorCache = new Map();
export function rgba(css) {
  let c = colorCache.get(css);
  if (c) return c;
  const s = css.trim();
  if (s[0] === "#") {
    const hex = s.length <= 5 ? [...s.slice(1)].map((x) => x + x).join("") : s.slice(1);
    c = [0, 2, 4, 6].map((i) => (i < hex.length ? parseInt(hex.slice(i, i + 2), 16) : 255));
  } else {
    const m = s.match(/[\d.]+/g) || [];
    c = [+m[0] || 0, +m[1] || 0, +m[2] || 0, m[3] === undefined ? 255 : Math.round(+m[3] * 255)];
  }
  colorCache.set(css, c);
  return c;
}

/** GPU texture of points (x, y) relative to an origin; BREAK in either coordinate breaks a line. */
export class Points {
  constructor(r) {
    this.r = r;
    this.tex = null;
    this.cap = 0;
    this.gen = r.gen;
  }

  /** Whether the texture still exists (not evicted, context not lost). */
  get live() {
    return this.tex !== null && this.gen === this.r.gen;
  }

  /** Allocate `cap` points and upload `n` from `data` (2 floats each); false if the GPU cannot. */
  upload(data, n, cap) {
    const gl = this.r.gl, rows = Math.max(1, Math.ceil(cap / TW));
    if (rows > this.r.maxRows) return false;
    this.release();
    this.tex = this.r.texture(gl.RG32F, TW, rows, gl.RG, gl.FLOAT, null);
    this.gen = this.r.gen;
    this.cap = rows * TW;
    const full = Math.ceil(n / TW);
    if (full) gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, TW, full, gl.RG, gl.FLOAT, data, 0);
    this.r.account(this, this.cap * 8);
    return true;
  }

  /** Overwrite points [off, off + n) with the first `n` points of `data`. */
  write(off, data, n) {
    const gl = this.r.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.tex);
    let i = 0;
    while (i < n) {
      const p = off + i, row = Math.floor(p / TW), col = p % TW;
      let w, h;
      if (col !== 0 || n - i < TW) (w = Math.min(TW - col, n - i)), (h = 1);
      else (w = TW), (h = Math.floor((n - i) / TW));
      gl.texSubImage2D(gl.TEXTURE_2D, 0, col, row, w, h, gl.RG, gl.FLOAT, data, 2 * i);
      i += w * h;
    }
  }

  /** Free the texture; `live` turns false so the owner rebuilds. */
  release() {
    if (this.tex && this.gen === this.r.gen) this.r.gl.deleteTexture(this.tex);
    this.tex = null;
    this.cap = 0;
    this.r.account(this, 0);
  }
}

/** Float32Array sized for `n` points of Points.upload (whole rows). */
export function pointBuffer(buf, n) {
  const need = 2 * TW * Math.max(1, Math.ceil(n / TW));
  return buf && buf.length >= need ? buf : new Float32Array(need);
}

/** Line table: per line (point offset, point count, color), in whole rows of the table texture. */
export class Table {
  constructor(n) {
    this.n = 0;
    this.a = new Int32Array(4 * MW * Math.max(1, Math.ceil(n / MW)));
    this.translucent = false; // some line's color is
  }
  clear() {
    this.n = 0;
    this.translucent = false;
  }
  push(off, count, color) {
    if (4 * (this.n + 1) > this.a.length) {
      const b = new Int32Array(this.a.length * 2);
      b.set(this.a);
      this.a = b;
    }
    const c = rgba(color), k = 4 * this.n++;
    if (c[3] < 255) this.translucent = true;
    this.a[k] = off;
    this.a[k + 1] = count;
    this.a[k + 2] = (c[0] << 16) | (c[1] << 8) | c[2];
    this.a[k + 3] = c[3];
  }
}

class Renderer {
  constructor() {
    this.canvas = document.createElement("canvas");
    this.canvas.width = this.canvas.height = 1;
    const gl = this.canvas.getContext("webgl2", { antialias: false, depth: true, stencil: false, alpha: true,
      premultipliedAlpha: true, preserveDrawingBuffer: false, powerPreference: "high-performance" });
    if (!gl) throw new Error("WebGL2 unavailable");
    this.gl = gl;
    this.gen = 0;
    this.lost = false;
    this.sized = new Map(); // owner -> bytes on the GPU
    this.bytes = 0;
    this.budget = 1 << 30;
    this.lru = new Map(); // owner -> last use
    this.clock = 0;
    this.canvas.addEventListener("webglcontextlost", (e) => {
      e.preventDefault();
      this.lost = true;
    });
    this.canvas.addEventListener("webglcontextrestored", () => {
      this.init();
      this.lost = false;
      this.onRestore?.();
    });
    this.init();
  }

  init() {
    const gl = this.gl;
    this.gen++;
    this.sized.clear();
    this.lru.clear();
    this.bytes = 0;
    this.multi = ext(gl, "WEBGL_multi_draw");
    const floats = ext(gl, "EXT_color_buffer_float"), blend = floats && ext(gl, "EXT_float_blend");
    ext(gl, "KHR_parallel_shader_compile");
    this.maxRows = Math.min(gl.getParameter(gl.MAX_TEXTURE_SIZE), 32768);
    this.most = Math.min(gl.getParameter(gl.MAX_VIEWPORT_DIMS)[0], 8192); // px the shared canvas is wide or high, at most
    this.progs = this.lineProgs(!!this.multi);
    this.densFormat = null;
    if (floats) {
      this.densFormat = blend ? gl.R32F : gl.R16F;
      this.progs.max = compile(gl, MAX_VS, MAX_FS);
      this.progs.tile = compile(gl, QUAD_VS, TILE_FS);
      this.progs.count = compile(gl, COUNT_VS, COUNT_FS);
      this.progs.cmap = compile(gl, QUAD_VS, CMAP_FS);
      this.maxTex = this.texture(this.densFormat, MAX_BINS, 1, gl.RED, gl.FLOAT, null);
      this.maxFbo = gl.createFramebuffer();
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.maxFbo);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.maxTex, 0);
      if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) this.densFormat = null;
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    }
    this.wakeFbo = gl.createFramebuffer(); // one pixel, for `wake`
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.wakeFbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.texture(gl.RGBA8, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, null), 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    this.dens = null;
    this.metaTex = null;
    this.metaRows = this.metaAt = 0;
    this.vao = gl.createVertexArray();
    this.firsts = this.counts = this.inst = null;
  }

  lineProgs(multi) {
    const gl = this.gl;
    return { ...this.progs, line: compile(gl, LINE_VS(multi), LINE_FS(true)), lineOpaque: compile(gl, LINE_VS(multi), LINE_FS(false)),
             band: compile(gl, BAND_VS(multi), BAND_FS) };
  }

  /** Program `name`, linked; line and band programs drop multi-draw when theirs fail to link. */
  program(name) {
    try {
      return linked(this.gl, this.progs[name]);
    } catch (e) {
      if (!this.multi || !["line", "lineOpaque", "band"].includes(name)) throw e;
      this.multi = null;
      this.progs = this.lineProgs(false);
      return linked(this.gl, this.progs[name]);
    }
  }

  /** Points one point texture holds. */
  get capacity() {
    return this.maxRows * TW;
  }

  /** Whether this GPU draws density heatmaps (float render targets). */
  get heatmaps() {
    return this.densFormat !== null;
  }

  texture(ifmt, w, h, fmt, type, data) {
    const gl = this.gl, t = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, ifmt, w, h, 0, fmt, type, data);
    return t;
  }

  /** Record GPU bytes held by `owner` (0 forgets it). */
  account(owner, bytes) {
    this.bytes += bytes - (this.sized.get(owner) || 0);
    if (bytes) this.sized.set(owner, bytes);
    else this.sized.delete(owner), this.lru.delete(owner);
  }

  /** Mark `owner` used now. */
  touch(owner) {
    this.lru.set(owner, ++this.clock);
  }

  /** Release least recently used Points until under budget, keeping `keep`. */
  trim(keep) {
    if (this.bytes <= this.budget) return;
    const order = [...this.lru].sort((a, b) => a[1] - b[1]);
    for (const [o] of order) {
      if (this.bytes <= this.budget) break;
      if (!keep.has(o)) o.release();
    }
  }

  /** Top-left corners (device px, y down) of a region of the shared canvas for each chart image of `sizes` ([W, H]
   * each), in rows at most `wide` px wide (or one image), the canvas grown to hold them; null for those beyond its
   * largest size, drawn at [0, 0] after the others are copied. */
  place(sizes, wide = Infinity) {
    const most = this.most, end = Math.min(most, wide), out = [];
    let x = 0, y = 0, row = 0, w = 0;
    for (const [W, H] of sizes) {
      if (x && x + W > end) (x = 0), (y += row), (row = 0);
      const fits = x + W <= most && y + H <= most;
      out.push(fits ? [x, y] : null);
      if (!fits) continue;
      (x += W), (row = Math.max(row, H)), (w = Math.max(w, x));
    }
    this.fit(w, y + row);
    return out;
  }

  /** Grow the shared canvas to at least W x H (which clears it, and waits for the GPU). */
  fit(W, H) {
    const c = this.canvas;
    if (c.width < W || c.height < H) (c.width = Math.max(c.width, W)), (c.height = Math.max(c.height, H));
  }

  /** Grow the shared canvas to at least W x H, as far as its largest size allows, ahead of the images that need it. */
  reserve(W, H) {
    this.fit(Math.min(W, this.most), Math.min(H, this.most));
  }

  /** Start a chart image of W x H device px at `at` (its top-left corner on the shared canvas, device px, y down);
   * `clip` = [x, y, w, h] device px, y down. */
  begin(W, H, clip, at = [0, 0]) {
    const gl = this.gl;
    this.fit(at[0] + W, at[1] + H);
    this.W = W;
    this.H = H;
    this.at = [at[0], this.canvas.height - at[1] - H]; // the image's bottom-left corner in GL coordinates
    const x0 = Math.round(clip[0]), y0 = Math.round(clip[1]);
    const x1 = Math.round(clip[0] + clip[2]), y1 = Math.round(clip[1] + clip[3]);
    this.clip = [x0, y0, Math.max(0, x1 - x0), Math.max(0, y1 - y0)];
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(this.at[0], this.at[1], W, H);
    gl.enable(gl.SCISSOR_TEST);
    gl.scissor(this.at[0], this.at[1], W, H);
    gl.clearColor(0, 0, 0, 0);
    gl.clearDepth(1);
    gl.depthMask(true);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    this.scissor(this.at);
    gl.enable(gl.BLEND);
    gl.blendEquation(gl.FUNC_ADD);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    gl.bindVertexArray(this.vao);
  }

  /** Scissor to the plot rectangle of an image whose bottom-left corner is at `base` (GL coordinates). */
  scissor(base) {
    const [x, y, w, h] = this.clip;
    this.scissorTo([base[0] + x, base[1] + this.H - y - h, w, h]);
  }

  /** Tell `prog` the plot's rectangle in the image (u_inside), half a pixel out: a point on a side lies in it. */
  inside(prog) {
    const [x, y, w, h] = this.clip;
    this.gl.uniform4f(prog.u.u_inside, x - 0.5, y - 0.5, x + w + 0.5, y + h + 0.5);
  }

  /** Scissor to `plot` ([x, y, w, h] in the target drawn to): the plot's rectangle there, which `lines` draws to. */
  scissorTo(plot) {
    this.plot = plot;
    this.gl.scissor(...plot);
  }

  /** Bind `pts`, a line table and `view` {off: data at the plot origin, scale: px per unit, org: device px, first: the
   * point the table's offsets count from (0 unless given)}. Each table goes to rows of the table texture no earlier
   * draw reads, so no upload changes what a queued draw sees. */
  bind(prog, pts, table, view) {
    const gl = this.gl, u = prog.u;
    gl.useProgram(prog.p);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, pts.tex);
    gl.uniform1i(u.u_pos, 0);
    gl.activeTexture(gl.TEXTURE1);
    const rows = Math.max(1, Math.ceil(table.n / MW));
    if (this.metaAt + rows > this.metaRows) {
      if (this.metaTex) gl.deleteTexture(this.metaTex);
      this.metaRows = Math.max(META_ROWS, rows);
      this.metaTex = this.texture(gl.RGBA32I, MW, this.metaRows, gl.RGBA_INTEGER, gl.INT, null);
      this.metaAt = 0;
    }
    gl.bindTexture(gl.TEXTURE_2D, this.metaTex);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, this.metaAt, MW, rows, gl.RGBA_INTEGER, gl.INT, table.a, 0);
    gl.uniform1i(u.u_row0, this.metaAt);
    gl.uniform1i(u.u_first, view.first || 0);
    this.metaAt += rows;
    gl.uniform1i(u.u_meta, 1);
    gl.uniform2f(u.u_off, view.off[0], view.off[1]);
    gl.uniform2f(u.u_scale, view.scale[0], view.scale[1]);
    gl.uniform2f(u.u_org, view.org[0], view.org[1]);
    gl.uniform2f(u.u_size, this.W, this.H);
  }

  /** Instanced draws of 6 vertices per instance: `grid` instances each of n lines in one draw, or a draw per line with
   * `inst(i)` instances. */
  draws(prog, n, inst, grid = 0) {
    const gl = this.gl;
    gl.uniform1i(prog.u.u_grid, grid);
    if (!n) return;
    if (grid) gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, n * grid);
    else if (this.multi) {
      if (!this.firsts || this.firsts.length < n) {
        const m = Math.max(n, 2 * (this.firsts?.length || 0));
        this.firsts = new Int32Array(m);
        this.counts = new Int32Array(m).fill(6);
        this.inst = new Int32Array(m);
      }
      for (let i = 0; i < n; i++) this.inst[i] = Math.max(0, inst(i));
      gl.uniform1i(prog.u.u_base, 0);
      this.multi.multiDrawArraysInstancedWEBGL(gl.TRIANGLES, this.firsts, 0, this.counts, 0, this.inst, 0, n);
    } else {
      for (let i = 0; i < n; i++) {
        const k = inst(i);
        if (k <= 0) continue;
        gl.uniform1i(prog.u.u_base, i);
        gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, k);
      }
    }
  }

  /** Every line of `table` as a `width` px (device) stroke at opacity `alpha`, in table order; lines of `grid` + 1
   * points each when grid is set. A point no segment reaches is drawn as a dot (`LINE_VS`), so an instance is a
   * point; a dot on a side of the plot is drawn whole, past the side, except in a heatmap's counts. */
  lines(pts, table, view, width, alpha, density = false, grid = 0) {
    const gl = this.gl, once = density || alpha < 1 || table.translucent, prog = this.program(once ? "line" : "lineOpaque"), a = table.a;
    const [x, y, w, h] = this.plot, dot = dotRadius(), past = density ? 0 : Math.ceil(dot + 0.5);
    this.bind(prog, pts, table, view);
    gl.uniform1f(prog.u.u_half, width / 2);
    gl.uniform1f(prog.u.u_dot, dot);
    this.inside(prog);
    gl.uniform4f(prog.u.u_plot, x, y, x + w, y + h);
    gl.uniform1f(prog.u.u_alpha, alpha);
    gl.uniform1f(prog.u.u_count, table.n);
    gl.uniform1i(prog.u.u_density, density ? 1 : 0);
    if (past) gl.scissor(x - past, y - past, w + 2 * past, h + 2 * past);
    if (once) gl.enable(gl.DEPTH_TEST), gl.depthFunc(gl.LESS), gl.clear(gl.DEPTH_BUFFER_BIT);
    this.draws(prog, table.n, (i) => a[4 * i + 1], grid && grid + 1);
    gl.disable(gl.DEPTH_TEST);
    if (past) gl.scissor(x, y, w, h);
  }

  /** Filled bands: table entries in (hi, lo) pairs with equal counts (`grid` + 1 points each when grid is set). */
  bands(pts, table, view, alpha, grid = 0) {
    const gl = this.gl, prog = this.program("band"), a = table.a;
    this.bind(prog, pts, table, view);
    gl.uniform1f(prog.u.u_alpha, alpha);
    this.draws(prog, table.n >> 1, (i) => a[8 * i + 1] - 1, grid);
  }

  /** Heatmap of every line of `table`: per-pixel line count (each line once), log colormap; lines of `grid` + 1 points
   * each, when grid is set, counted as 1 px lines. */
  density(pts, table, view, width, dark, grid = 0) {
    const gl = this.gl, W = this.W, H = this.H;
    let d = this.dens;
    if (!d || d.w < W || d.h < H) {
      if (d) gl.deleteTexture(d.tex), gl.deleteTexture(d.tiles), gl.deleteRenderbuffer(d.depth), gl.deleteFramebuffer(d.fbo), gl.deleteFramebuffer(d.tileFbo);
      const w = Math.max(W, d?.w || 0), h = Math.max(H, d?.h || 0);
      d = this.dens = { w, h, tex: this.texture(this.densFormat, w, h, gl.RED, gl.FLOAT, null), depth: gl.createRenderbuffer(), fbo: gl.createFramebuffer(),
                        tiles: this.texture(this.densFormat, Math.ceil(w / TILE), Math.ceil(h / TILE), gl.RED, gl.FLOAT, null), tileFbo: gl.createFramebuffer() };
      gl.bindRenderbuffer(gl.RENDERBUFFER, d.depth);
      gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT24, w, h);
      gl.bindFramebuffer(gl.FRAMEBUFFER, d.fbo);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, d.tex, 0);
      gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, d.depth);
      gl.bindFramebuffer(gl.FRAMEBUFFER, d.tileFbo);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, d.tiles, 0);
    }
    const [cx, cy, cw, ch] = this.clip, rect = [cx, H - cy - ch, cw, ch]; // the plot in the density texture
    gl.bindFramebuffer(gl.FRAMEBUFFER, d.fbo);
    gl.viewport(0, 0, W, H);
    this.scissorTo(rect);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.blendFunc(gl.ONE, gl.ONE);
    if (grid) this.countLines(pts, table, view, grid);
    else this.lines(pts, table, view, width, 1, true);
    this.densityMax(d, rect);
    // colormap onto the chart image
    const cp = this.program("cmap");
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.enable(gl.SCISSOR_TEST);
    gl.viewport(this.at[0], this.at[1], W, H);
    this.scissor(this.at);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    gl.useProgram(cp.p);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, d.tex);
    gl.uniform1i(cp.u.u_dens, 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.maxTex);
    gl.uniform1i(cp.u.u_max, 1);
    gl.uniform2i(cp.u.u_at, this.at[0], this.at[1]);
    gl.uniform1i(cp.u.u_dark, dark ? 1 : 0);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  /** Every line of `table` (`grid` + 1 points each, broken only where all are) as a 1 px line strip, adding one to
   * each pixel it crosses; one whose points all lie at one place as a dot there (`COUNT_VS`). */
  countLines(pts, table, view, grid) {
    const gl = this.gl, prog = this.program("count");
    this.bind(prog, pts, table, view);
    gl.uniform1f(prog.u.u_dot, dotRadius());
    this.inside(prog);
    gl.uniform1i(prog.u.u_dots, 0);
    gl.drawArraysInstanced(gl.LINE_STRIP, 0, grid + 1, table.n);
    gl.uniform1i(prog.u.u_dots, 1);
    gl.drawArraysInstanced(gl.POINTS, 0, 1, table.n);
  }

  /** The density texture's maximum over `rect` (the plot's [x, y, w, h] in it) into the maximum texels: each tile's,
   * then theirs. */
  densityMax(d, rect) {
    const gl = this.gl, [, , cw, ch] = rect, tw = Math.ceil(cw / TILE), th = Math.ceil(ch / TILE), tp = this.program("tile"), mp = this.program("max");
    gl.disable(gl.SCISSOR_TEST);
    gl.disable(gl.BLEND);
    gl.bindFramebuffer(gl.FRAMEBUFFER, d.tileFbo);
    gl.viewport(0, 0, Math.max(1, tw), Math.max(1, th));
    gl.useProgram(tp.p);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, d.tex);
    gl.uniform1i(tp.u.u_dens, 0);
    gl.uniform4i(tp.u.u_rect, ...rect);
    if (cw && ch) gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.enable(gl.BLEND);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.maxFbo);
    gl.viewport(0, 0, MAX_BINS, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.blendEquation(gl.MAX);
    gl.useProgram(mp.p);
    gl.bindTexture(gl.TEXTURE_2D, d.tiles);
    gl.uniform1i(mp.u.u_dens, 0);
    gl.uniform4i(mp.u.u_rect, 0, 0, Math.max(1, tw), th);
    if (tw && th) gl.drawArrays(gl.POINTS, 0, tw * th);
    gl.blendEquation(gl.FUNC_ADD);
  }

  /** Give the GPU a command to run now. After a pause its first commands take a millisecond or two longer to be run,
   * which whoever reads results back then waits for; an interaction's handler calls this first, so that the time
   * passes while it does its own work. */
  wake() {
    if (this.lost) return;
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.wakeFbo);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.flush();
  }

  /** Call `then` once the GPU has run every command issued so far: work that reads results back then waits for its
   * own commands only. */
  whenDone(then) {
    const gl = this.gl, sync = this.lost ? null : gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0);
    if (!sync) return void then();
    gl.flush();
    const poll = () => {
      if (!this.lost && gl.clientWaitSync(sync, 0, 0) === gl.TIMEOUT_EXPIRED) return void setTimeout(poll, 4);
      gl.deleteSync(sync);
      then();
    };
    setTimeout(poll, 4);
  }

  /** Copy the W x H chart image at `at` (top-left, device px, y down) onto a 2D context (device px, identity
   * transform). */
  copyTo(ctx, at = [0, 0], W = this.W, H = this.H) {
    ctx.save();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.drawImage(this.canvas, at[0], at[1], W, H, 0, 0, W, H);
    ctx.restore();
  }
}

let shared;
/** The shared renderer (`lost` while its context is lost; `onRestore` runs once it is back), or null without WebGL2. */
export function renderer() {
  if (shared === undefined) {
    try {
      shared = new Renderer();
    } catch (e) {
      console.warn("trex: charts need WebGL2:", e.message);
      shared = null;
    }
  }
  return shared;
}

export { BREAK };
