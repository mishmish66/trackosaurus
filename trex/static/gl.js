// WebGL2 chart renderer. One shared offscreen context draws a chart's lines, group bands and
// density heatmap; the chart copies the result onto its own canvas. Points live in RG32F
// textures (x, y relative to an origin); a line table (offset, count, color) per draw lets one
// instanced multi-draw render every line as antialiased segment quads.

const TW = 2048; // points per row of a point texture
const MW = 1024; // lines per row of a line table
const META_ROWS = 64; // rows of a line-table texture
const BREAK = 1e38; // coordinate marking a line break
const MAX_BINS = 16; // texels of the density maximum

const ext = (gl, name) => gl.getExtension(name);

/** A program compiling and linking in the background (KHR_parallel_shader_compile); `linked` checks it. */
function compile(gl, vs, fs) {
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
function linked(gl, prog) {
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
uniform int u_base, u_row0;
uniform vec2 u_off, u_scale, u_org, u_size;
vec2 fetch(int i) { return texelFetch(u_pos, ivec2(i % ${TW}, i / ${TW}), 0).xy; }
ivec4 meta(int line) { return texelFetch(u_meta, ivec2(line % ${MW}, u_row0 + line / ${MW}), 0); }
bool broken(vec2 p) { return abs(p.y) > 1.0e37 || abs(p.x) > 1.0e37; }
vec2 toPx(vec2 p) { return vec2(u_org.x + (p.x - u_off.x) * u_scale.x, u_org.y - (p.y - u_off.y) * u_scale.y); }
vec4 toClip(vec2 P) { return vec4(P.x / u_size.x * 2.0 - 1.0, 1.0 - P.y / u_size.y * 2.0, 0.0, 1.0); }
vec4 unpack(ivec4 m) { return vec4(float((m.z >> 16) & 255), float((m.z >> 8) & 255), float(m.z & 255), float(m.w)) / 255.0; }
`;

// Segment quad: two triangles around segment i -> i+1, padded by half width plus 1 px of fringe.
const LINE_VS = (multi) => `${header(multi)}
uniform float u_half, u_count;
out vec2 v_uv;
flat out float v_len, v_depth;
flat out vec4 v_color;
void main() {
  int line = DRAW_ID + u_base;
  ivec4 m = meta(line);
  int i = m.x + gl_InstanceID;
  vec2 a = fetch(i), b = fetch(i + 1);
  if (broken(a) || broken(b)) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }
  vec2 A = toPx(a), B = toPx(b), d = B - A;
  float len = length(d);
  vec2 dir = len > 1e-4 ? d / len : vec2(1.0, 0.0), nrm = vec2(-dir.y, dir.x);
  int c = gl_VertexID;
  float end = (c == 1 || c == 4 || c == 5) ? 1.0 : 0.0;
  float side = (c == 2 || c == 3 || c == 5) ? 1.0 : -1.0;
  float h = u_half + 1.0, s = end * 2.0 - 1.0;
  v_uv = vec2(end * len + s * h, side * h);
  v_len = len;
  v_color = unpack(m);
  v_depth = 1.0 - (float(line) + 1.0) / (u_count + 1.0);
  gl_Position = toClip(mix(A, B, end) + dir * s * h + nrm * side * h);
}`;

// Coverage of a round-capped segment of width 2*u_half. Depth decreases with draw order and
// coverage, so each line blends into a pixel about once (as a stroked path would).
const LINE_FS = `#version 300 es
precision highp float;
uniform float u_half, u_alpha, u_count;
uniform int u_density;
in vec2 v_uv;
flat in float v_len, v_depth;
flat in vec4 v_color;
out vec4 o;
void main() {
  float u = v_uv.x, v = v_uv.y;
  float d = u < 0.0 ? length(vec2(u, v)) : u > v_len ? length(vec2(u - v_len, v)) : abs(v);
  float cov = clamp(u_half + 0.5 - d, 0.0, 1.0);
  if (cov <= 0.0) discard;
  gl_FragDepth = v_depth - cov * 0.5 / (u_count + 1.0);
  if (u_density == 1) { o = vec4(cov, 0.0, 0.0, 1.0); return; }
  float al = cov * u_alpha * v_color.a;
  o = vec4(v_color.rgb * al, al);
}`;

// Band quad between bins i and i+1: line-table entry (hi offset, bins, rgb, alpha) with the lo
// offset in the entry after it.
const BAND_VS = (multi) => `${header(multi)}
flat out vec4 v_color;
void main() {
  int band = DRAW_ID + u_base;
  ivec4 m = meta(2 * band), l = meta(2 * band + 1);
  int k = gl_InstanceID;
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

// Maximum of the density texture over the plot rectangle, into MAX_BINS texels (MAX blending).
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
uniform int u_shift, u_dark;
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
  float d = texelFetch(u_dens, ivec2(gl_FragCoord.xy) - ivec2(0, u_shift), 0).r;
  if (d <= 0.004) discard;
  float mx = 0.0;
  for (int i = 0; i < ${MAX_BINS}; i++) mx = max(mx, texelFetch(u_max, ivec2(i, 0), 0).r);
  float t = clamp(log(1.0 + d) / log(1.0 + max(mx, 1.0)), 0.0, 1.0);
  vec3 c = viridis(u_dark == 1 ? t : 1.0 - t);
  float a = clamp(d, 0.0, 1.0) * (u_dark == 1 ? mix(0.55, 1.0, t) : mix(0.45, 1.0, t));
  o = vec4(c * a, a);
}`;

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

/** Line table: per line (point offset, point count, color). */
/** Lines' (offset, count, color), in whole rows of the table texture. */
export class Table {
  constructor(n) {
    this.n = 0;
    this.a = new Int32Array(4 * MW * Math.max(1, Math.ceil(n / MW)));
  }
  clear() {
    this.n = 0;
  }
  push(off, count, color) {
    if (4 * (this.n + 1) > this.a.length) {
      const b = new Int32Array(this.a.length * 2);
      b.set(this.a);
      this.a = b;
    }
    const c = rgba(color), k = 4 * this.n++;
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
    this.progs = this.lineProgs(!!this.multi);
    this.densFormat = null;
    if (floats) {
      this.densFormat = blend ? gl.R32F : gl.R16F;
      this.progs.max = compile(gl, MAX_VS, MAX_FS);
      this.progs.cmap = compile(gl, QUAD_VS, CMAP_FS);
      this.maxTex = this.texture(this.densFormat, MAX_BINS, 1, gl.RED, gl.FLOAT, null);
      this.maxFbo = gl.createFramebuffer();
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.maxFbo);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.maxTex, 0);
      if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) this.densFormat = null;
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    }
    this.dens = null;
    this.metaTex = null;
    this.metaRows = this.metaAt = 0;
    this.vao = gl.createVertexArray();
    this.firsts = this.counts = this.inst = null;
  }

  lineProgs(multi) {
    const gl = this.gl;
    return { ...this.progs, line: compile(gl, LINE_VS(multi), LINE_FS), band: compile(gl, BAND_VS(multi), BAND_FS) };
  }

  /** Program `name`, linked; line and band programs drop multi-draw when theirs fail to link. */
  program(name) {
    try {
      return linked(this.gl, this.progs[name]);
    } catch (e) {
      if (!this.multi || (name !== "line" && name !== "band")) throw e;
      this.multi = null;
      this.progs = this.lineProgs(false);
      return linked(this.gl, this.progs[name]);
    }
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

  /** Start a chart image of W x H device px; `clip` = [x, y, w, h] device px, y down. */
  begin(W, H, clip) {
    const gl = this.gl, c = this.canvas;
    if (c.width < W || c.height < H) {
      c.width = Math.max(c.width, W);
      c.height = Math.max(c.height, H);
    }
    this.W = W;
    this.H = H;
    this.shift = c.height - H;
    const x0 = Math.round(clip[0]), y0 = Math.round(clip[1]);
    const x1 = Math.round(clip[0] + clip[2]), y1 = Math.round(clip[1] + clip[3]);
    this.clip = [x0, y0, Math.max(0, x1 - x0), Math.max(0, y1 - y0)];
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, this.shift, W, H);
    gl.enable(gl.SCISSOR_TEST);
    gl.scissor(0, this.shift, W, H);
    gl.clearColor(0, 0, 0, 0);
    gl.clearDepth(1);
    gl.depthMask(true);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    this.scissor(this.shift);
    gl.enable(gl.BLEND);
    gl.blendEquation(gl.FUNC_ADD);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    gl.bindVertexArray(this.vao);
  }

  scissor(shift) {
    const [x, y, w, h] = this.clip;
    this.gl.scissor(x, shift + this.H - y - h, w, h);
  }

  /** Bind `pts`, a line table and `view` {off: data at the plot origin, scale: px per unit, org: device px}.
   * Each table goes to rows of the table texture no earlier draw reads, so no upload changes what a queued
   * draw sees. */
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
    this.metaAt += rows;
    gl.uniform1i(u.u_meta, 1);
    gl.uniform2f(u.u_off, view.off[0], view.off[1]);
    gl.uniform2f(u.u_scale, view.scale[0], view.scale[1]);
    gl.uniform2f(u.u_org, view.org[0], view.org[1]);
    gl.uniform2f(u.u_size, this.W, this.H);
  }

  /** Instanced draws of 6 vertices per instance; instance counts per draw from `inst(i)`. */
  draws(prog, n, inst) {
    const gl = this.gl;
    if (!n) return;
    if (this.multi) {
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

  /** Every line of `table` as a `width` px (device) stroke at opacity `alpha`, in table order. */
  lines(pts, table, view, width, alpha, density = false) {
    const gl = this.gl, prog = this.program("line"), a = table.a;
    this.bind(prog, pts, table, view);
    gl.uniform1f(prog.u.u_half, width / 2);
    gl.uniform1f(prog.u.u_alpha, alpha);
    gl.uniform1f(prog.u.u_count, table.n);
    gl.uniform1i(prog.u.u_density, density ? 1 : 0);
    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.LESS);
    gl.clear(gl.DEPTH_BUFFER_BIT);
    this.draws(prog, table.n, (i) => a[4 * i + 1] - 1);
    gl.disable(gl.DEPTH_TEST);
  }

  /** Filled bands: table entries in (hi, lo) pairs with equal counts. */
  bands(pts, table, view, alpha) {
    const gl = this.gl, prog = this.program("band"), a = table.a;
    this.bind(prog, pts, table, view);
    gl.uniform1f(prog.u.u_alpha, alpha);
    this.draws(prog, table.n >> 1, (i) => a[8 * i + 1] - 1);
  }

  /** Heatmap of every line of `table`: per-pixel line count (each line once), log colormap. */
  density(pts, table, view, width, dark) {
    const gl = this.gl, W = this.W, H = this.H;
    let d = this.dens;
    if (!d || d.w < W || d.h < H) {
      if (d) gl.deleteTexture(d.tex), gl.deleteRenderbuffer(d.depth), gl.deleteFramebuffer(d.fbo);
      const w = Math.max(W, d?.w || 0), h = Math.max(H, d?.h || 0);
      d = this.dens = { w, h, tex: this.texture(this.densFormat, w, h, gl.RED, gl.FLOAT, null), depth: gl.createRenderbuffer(), fbo: gl.createFramebuffer() };
      gl.bindRenderbuffer(gl.RENDERBUFFER, d.depth);
      gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT24, w, h);
      gl.bindFramebuffer(gl.FRAMEBUFFER, d.fbo);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, d.tex, 0);
      gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, d.depth);
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, d.fbo);
    gl.viewport(0, 0, W, H);
    this.scissor(0);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.blendFunc(gl.ONE, gl.ONE);
    this.lines(pts, table, view, width, 1, true);
    // maximum over the plot rectangle
    const [cx, cy, cw, ch] = this.clip, mp = this.program("max");
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.maxFbo);
    gl.disable(gl.SCISSOR_TEST);
    gl.viewport(0, 0, MAX_BINS, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.blendEquation(gl.MAX);
    gl.useProgram(mp.p);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, d.tex);
    gl.uniform1i(mp.u.u_dens, 0);
    gl.uniform4i(mp.u.u_rect, cx, H - cy - ch, Math.max(1, cw), ch);
    if (cw && ch) gl.drawArrays(gl.POINTS, 0, cw * ch);
    gl.blendEquation(gl.FUNC_ADD);
    // colormap onto the chart image
    const cp = this.program("cmap");
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.enable(gl.SCISSOR_TEST);
    gl.viewport(0, this.shift, W, H);
    this.scissor(this.shift);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    gl.useProgram(cp.p);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, d.tex);
    gl.uniform1i(cp.u.u_dens, 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.maxTex);
    gl.uniform1i(cp.u.u_max, 1);
    gl.uniform1i(cp.u.u_shift, this.shift);
    gl.uniform1i(cp.u.u_dark, dark ? 1 : 0);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  /** Copy the chart image onto a 2D context (device px, identity transform). */
  copyTo(ctx) {
    ctx.save();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.drawImage(this.canvas, 0, 0, this.W, this.H, 0, 0, this.W, this.H);
    ctx.restore();
  }
}

let shared;
/** The shared renderer, or null when WebGL2 is unavailable or the context is lost. */
export function renderer() {
  if (shared === undefined) {
    try {
      shared = new Renderer();
    } catch (e) {
      console.warn("trex: WebGL2 renderer unavailable, using Canvas 2D:", e.message);
      shared = null;
    }
  }
  return shared && !shared.lost ? shared : null;
}

export { BREAK };
