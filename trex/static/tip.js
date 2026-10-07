// The value tooltip of a hovered chart, drawn on one canvas over the page. A hover then changes nothing of the
// document, so the browser has no style, layout or painting of it to do for the frame: a tooltip of elements that
// follows the pointer has it paint the page's layers anew and assign them again at every move, several ms on a page of
// hundreds of charts. Pinned, the tooltip is elements (app.js `pinTip`): a list to scroll and click.
//# allFunctionsCalledOnLoad

import { fmt, theme } from "./plot.js";

export const TIP_ROWS = 14; // value rows in view in the tooltip
export const TIP_ROW_PX = 18; // height of one (#tip .trow in index.html)
const HINT = "hold shift to pin · drag: zoom x · drag a box: zoom x and y · click: reset";
// The tooltip's measures are those of #tip in index.html, which shows the same rows once it is pinned.
const FAMILY = "system-ui, sans-serif", FONT = `12px ${FAMILY}`, BOLD = `700 ${FONT}`, NEAR = `600 ${FONT}`, SMALL = `11px ${FAMILY}`;
const EDGE_X = 7, EDGE_Y = 5; // from the tooltip's edge to its text: its border and padding
const MAX_W = 600; // of the tooltip, edges included
const HEAD_PX = 16.2, HEAD_GAP = 2; // the heading's line, and the space under it
const FOOT_GAP = 3, FOOT_RULE = 1, FOOT_PAD = 2, FOOT_PX = 14.85; // above the hint: space, a rule, space; the hint's line
const SWATCH = 10, SWATCH_GAP = 4; // a row's color, and the space after it
const LABEL_MIN = 120, LABEL_MAX = 220, LABEL_GAP = 6; // a row's label column, and the space after it
const FADE_PX = 16; // the list fades out over this much where rows lie beyond it
const SHADOW = 12; // how far the tooltip's shadow reaches beyond it
const LABELS_KEPT = 4096; // labels kept as they fit their column
const HIDE_MS = 500; // the canvas stays on the page this long after its tooltip went, for the next one

/** `text` as it fits `max` px in the context's font, an ellipsis in place of what does not, and its width. */
function fit(c, text, max) {
  const w = c.measureText(text).width;
  if (w <= max) return [text, w];
  let lo = 0, hi = text.length;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (c.measureText(`${text.slice(0, mid)}…`).width <= max) lo = mid;
    else hi = mid - 1;
  }
  const cut = `${text.slice(0, lo)}…`;
  return [cut, c.measureText(cut).width];
}

export class TipCanvas {
  constructor(el) {
    this.el = el; // the canvas: over the window, hidden while no tooltip is shown
    this.ctx = null; // made, with the canvas's size, at the first tooltip
    this.vw = this.vh = 0; // the window's size, in CSS px
    this.box = null; // [x, y, w, h] of the tooltip drawn, in CSS px
    this.base = new Map(); // font and line height -> `baseline`'s answer
    this.hideT = 0; // the timer that takes the cleared canvas off the page
    this.labels = new Map(); // "n" (the nearest row's weight) or "" and a label -> `label`'s answer
  }

  /** Whether a tooltip is drawn. */
  get shown() {
    return this.box !== null;
  }

  /** The canvas's context, the canvas as large as the window at the display's resolution: also after the window is
   * resized or the browser gave the context back after losing it, which both leave it blank until the next tooltip. */
  context() {
    if (!this.ctx) {
      this.ctx = this.el.getContext("2d");
      addEventListener("resize", () => this.size());
      this.el.addEventListener("contextrestored", () => this.size());
      this.size();
    }
    return this.ctx;
  }

  /** Give the canvas the window's size, which blanks it. */
  size() {
    const dpr = devicePixelRatio || 1;
    [this.vw, this.vh] = [innerWidth, innerHeight];
    this.el.hidden = true;
    this.el.width = Math.round(this.vw * dpr);
    this.el.height = Math.round(this.vh * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.box = null;
    this.base.clear();
  }

  /** Where a line `px` tall has the baseline of text in `font`, from its top, as the browser lays a line out (the font's
   * ascent and descent centered in it, on a whole pixel of the display), and the middle of its small letters, where
   * the browser centers a row's color: [baseline, middle]. */
  baseline(font, px) {
    const key = `${font}|${px}`, c = this.ctx;
    let at = this.base.get(key);
    if (!at) {
      const dpr = devicePixelRatio || 1, was = c.font;
      c.font = font;
      const m = c.measureText("x"), y = Math.round(((px - m.fontBoundingBoxAscent - m.fontBoundingBoxDescent) / 2 + m.fontBoundingBoxAscent) * dpr) / dpr;
      c.font = was;
      this.base.set(key, (at = [y, Math.round((y - m.actualBoundingBoxAscent / 2) * dpr) / dpr]));
    }
    return at;
  }

  /** Label `text` as it fits its column, in the context's font (`near`: the nearest row's), and the column's width, the
   * label's own between LABEL_MIN and LABEL_MAX: [text, column]. */
  label(c, text, near) {
    const key = (near ? "n" : "") + text;
    let got = this.labels.get(key);
    if (!got) {
      if (this.labels.size >= LABELS_KEPT) this.labels.clear();
      const [cut, w] = fit(c, text, LABEL_MAX);
      this.labels.set(key, (got = [cut, cut === text ? Math.max(LABEL_MIN, w) : LABEL_MAX]));
    }
    return got;
  }

  /** What tooltip t ({chart, heading, rows, near, note}: `App.tip`) shows, measured: its heading, the TIP_ROWS rows
   * around the nearest ({row, near, label, col (its label column's width), val, valW, note, noteW (as they fit)}) from
   * row `a` on, the hint, and the width of the widest of them, at most what MAX_W leaves. */
  measure(c, t) {
    const n = t.rows.length, a = Math.max(0, Math.min(t.near - (TIP_ROWS >> 1), n - TIP_ROWS)), b = Math.min(n, a + TIP_ROWS);
    const most = MAX_W - 2 * EDGE_X, lines = [];
    c.font = FONT;
    const head = fit(c, `${t.chart?.key} · ${t.heading}`, most);
    for (let i = a; i < b; i++) {
      const row = t.rows[i], near = i === t.near;
      if (near) c.font = NEAR;
      const [label, col] = this.label(c, row.ln.label, near), note = t.note ? t.note(row) : "";
      lines.push({ row, near, label, col, val: fmt(row.val), valW: 0, note, noteW: c.measureText(note).width });
      if (near) c.font = FONT;
    }
    c.font = BOLD;
    for (const l of lines) l.valW = c.measureText(l.val).width;
    c.font = SMALL;
    const hint = fit(c, HINT, most);
    let w = Math.max(head[1], hint[1]);
    for (const l of lines) w = Math.max(w, SWATCH + SWATCH_GAP + l.col + LABEL_GAP + l.valW + l.noteW);
    return { a, head: head[0], hint: hint[0], lines, w: Math.min(most, Math.ceil(w)), more: [a > 0, b < n] };
  }

  /** Draw tooltip t beside the pointer (on its other side where it would leave the window, and never below the
   * window), in place of the one shown. Returns where: {x, y, w (CSS px), a (the first row shown)}. */
  draw(t) {
    const c = this.context(), m = this.measure(c, t), dpr = devicePixelRatio || 1, snap = (v) => Math.round(v * dpr) / dpr;
    const w = snap(m.w + 2 * EDGE_X), listH = m.lines.length * TIP_ROW_PX;
    const h = snap(2 * EDGE_Y + HEAD_PX + HEAD_GAP + listH + FOOT_GAP + FOOT_RULE + FOOT_PAD + FOOT_PX);
    let x = t.e.clientX + 16, y = t.e.clientY + 12;
    if (x + w > this.vw) x = t.e.clientX - w - 16;
    if (y + h > this.vh) y = Math.max(0, this.vh - h - 4);
    [x, y] = [snap(x), snap(y)];
    this.wipe();
    clearTimeout(this.hideT);
    this.hideT = 0;
    if (this.el.hidden) this.el.hidden = false;
    this.box = [x, y, w, h];
    this.frame(c, x, y, w, h, dpr);
    c.save();
    c.beginPath();
    c.rect(x + 1, y + 1, w - 2, h - 2);
    c.clip();
    const ix = x + EDGE_X, top = y + EDGE_Y + HEAD_PX + HEAD_GAP, th = theme();
    c.font = FONT;
    c.fillStyle = th.muted;
    c.fillText(m.head, ix, y + EDGE_Y + this.baseline(FONT, HEAD_PX)[0]);
    this.rows(c, m, ix, top, th);
    this.fades(c, m, ix, top, listH, th);
    const rule = top + listH + FOOT_GAP;
    c.fillStyle = th.grid;
    c.fillRect(ix, rule, m.w, FOOT_RULE);
    c.font = SMALL;
    c.fillStyle = th.muted;
    c.fillText(m.hint, ix, rule + FOOT_RULE + FOOT_PAD + this.baseline(SMALL, FOOT_PX)[0]);
    c.restore();
    return { x, y, w, a: m.a };
  }

  /** The tooltip's background, with its shadow, and its border. */
  frame(c, x, y, w, h, dpr) {
    const th = theme();
    c.shadowColor = "rgba(0,0,0,.15)";
    c.shadowBlur = 8 * dpr;
    c.shadowOffsetY = 2 * dpr;
    c.fillStyle = th.bg;
    c.beginPath();
    c.roundRect(x, y, w, h, 4);
    c.fill();
    c.shadowColor = "transparent";
    c.strokeStyle = th.line;
    c.lineWidth = 1;
    c.beginPath();
    c.roundRect(x + 0.5, y + 0.5, w - 1, h - 1, 3.5);
    c.stroke();
  }

  /** The measured rows from (ix, top) down: each one's color, label, value and note, a note cut where the tooltip ends;
   * the nearest row marked. Texts of one font are drawn together. */
  rows(c, m, ix, top, th) {
    const rowTop = (k) => top + k * TIP_ROW_PX, labelX = ix + SWATCH + SWATCH_GAP;
    m.lines.forEach((l, k) => {
      if (l.near) (c.fillStyle = th.grid), c.fillRect(ix, rowTop(k), m.w, TIP_ROW_PX);
      c.fillStyle = l.row.ln.color;
      c.beginPath();
      c.roundRect(ix, rowTop(k) + this.baseline(l.near ? NEAR : FONT, TIP_ROW_PX)[1] - SWATCH / 2, SWATCH, SWATCH, 2);
      c.fill();
    });
    const texts = (font, near) => {
      const [base] = this.baseline(font, TIP_ROW_PX);
      c.font = font;
      m.lines.forEach((l, k) => {
        if (l.near !== near) return;
        const noteX = labelX + l.col + LABEL_GAP + l.valW, room = ix + m.w - noteX;
        c.fillStyle = th.fg;
        c.fillText(l.label, labelX, rowTop(k) + base);
        c.fillStyle = th.muted;
        c.fillText(l.noteW > room ? fit(c, l.note, Math.max(0, room))[0] : l.note, noteX, rowTop(k) + base);
      });
    };
    texts(FONT, false);
    texts(NEAR, true);
    const [base] = this.baseline(BOLD, TIP_ROW_PX);
    c.font = BOLD;
    c.fillStyle = th.fg;
    m.lines.forEach((l, k) => c.fillText(l.val, labelX + l.col + LABEL_GAP, rowTop(k) + base));
  }

  /** Fade the list out toward its top and its bottom where rows lie beyond them. */
  fades(c, m, ix, top, listH, th) {
    const [up, down] = m.more, clear = `rgba(${th.rgb}, 0)`;
    for (const [on, y0, y1] of [[up, top, top + FADE_PX], [down, top + listH, top + listH - FADE_PX]]) {
      if (!on || listH < FADE_PX) continue;
      const g = c.createLinearGradient(0, y0, 0, y1);
      g.addColorStop(0, th.bg);
      g.addColorStop(1, clear);
      c.fillStyle = g;
      c.fillRect(ix, Math.min(y0, y1), m.w, FADE_PX);
    }
  }

  /** Clear the pixels of the tooltip drawn, its shadow's with them. */
  wipe() {
    const b = this.box;
    if (b) this.ctx.clearRect(b[0] - SHADOW, b[1] - SHADOW, b[2] + 2 * SHADOW, b[3] + 2 * SHADOW);
    this.box = null;
  }

  /** Clear the tooltip drawn. The canvas leaves the page HIDE_MS later unless another tooltip is drawn by then: showing
   * and hiding it change the document, which a pointer going from chart to chart is then spared. */
  clear() {
    if (!this.ctx || this.el.hidden) return;
    this.wipe();
    this.hideT ||= setTimeout(() => this.hide(), HIDE_MS);
  }

  /** Take the tooltip and its canvas off the page. */
  hide() {
    clearTimeout(this.hideT);
    this.hideT = 0;
    if (!this.ctx || this.el.hidden) return;
    this.wipe();
    this.el.hidden = true;
  }
}
