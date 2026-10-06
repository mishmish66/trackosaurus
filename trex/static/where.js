// Run filters: a SQL WHERE clause over run fields, the same as the CLI's (`trex/where.py`, whose docstring
// states the rules). compileWhere(text) -> {test(get), fields}, where get(field) returns a value or null; runField
// reads a field of a UI run.
//# allFunctionsCalledOnLoad

const TOKEN = /\s*(?:('(?:[^']|'')*')|("(?:[^"]|"")*")|(>=|<=|!=|<>|==|!~|[=<>~(),])|([^\s'"=<>!~(),]+)|(\S))/y;
const NUMBER = /^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$/;
const MARKERS = { nan: NaN, inf: Infinity, "-inf": -Infinity };
const KEYWORDS = new Set(["and", "or", "not", "in", "like", "is", "null", "true", "false"]);
const IDENTIFIER = /^[A-Za-z_][\w./:-]*$/;
const COMPARE = new Set(["=", "==", "!=", "<>", "<", "<=", ">", ">=", "~", "!~"]);

const PLAIN_FIELDS = {
  path: (r) => r.meta.id, step: (r) => r.meta.summary?._step, runtime: (r) => r.meta.summary?._runtime,
  rows: (r) => r.seq, media: (r) => r.mseq, visible: (r) => r.visible ?? true,
  ...Object.fromEntries(["name", "parent", "state", "created", "updated", "tags", "dir"].map((k) => [k, (r) => r.meta[k]])),
};
const FIELD_PREFIXES = { config: "config", c: "config", summary: "summary", s: "summary", metric: "summary", m: "summary", info: "info" };

/** A field of a UI run ({meta, seq, mseq, visible: its sidebar checkbox}), as the CLI reads it: a plain one, config.K (c.), summary.K (s., metric., m.), info.A.B, or a
 * bare key (config, then summary); null when missing. */
export function runField(r, field) {
  const v = rawField(r, field);
  return v === undefined ? null : v;
}

function rawField(r, field) {
  if (Object.hasOwn(PLAIN_FIELDS, field)) return PLAIN_FIELDS[field](r);
  const m = r.meta, cfg = m.config || {}, sum = m.summary || {};
  const dot = field.indexOf("."), kind = dot > 0 ? FIELD_PREFIXES[field.slice(0, dot)] : undefined, rest = field.slice(dot + 1);
  if (kind && rest) return kind === "info" ? infoField(m.info || {}, rest) : (kind === "config" ? cfg : sum)[rest];
  return Object.hasOwn(cfg, field) ? cfg[field] : sum[field];
}

function infoField(info, path) {
  if (path in info) return info[path];
  let v = info;
  for (const part of path.split(".")) v = v && typeof v === "object" ? v[part] : undefined;
  return v ?? null;
}

/** The filter `text` means; throws an Error for a clause that does not parse. */
export function compileWhere(text) {
  const toks = tokens(text);
  if (!toks || !toks.some((t) => (t.kind === "op" && COMPARE.has(t.text)) || isKw(t, "in", "like", "is"))) return search(text);
  const p = new Parser(toks), test = p.expr();
  if (p.i < toks.length) throw new Error(`unexpected '${toks[p.i].text}'`);
  return { test, fields: p.fields };
}

function search(text) {
  let rx;
  try {
    rx = new RegExp(text.trim(), "i");
  } catch {
    rx = new RegExp(text.trim().replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "i");
  }
  return { test: (g) => ["name", "path"].some((f) => g(f) != null && rx.test(textOf(g(f)))), fields: ["name", "path"] };
}

/** Tokens of `text`, {kind, text (quotes unescaped), start, end}; null where it does not tokenize (an open quote, a
 * stray character). */
function tokens(text) {
  const out = [];
  TOKEN.lastIndex = 0;
  while (TOKEN.lastIndex < text.length && text.slice(TOKEN.lastIndex).trim()) {
    const m = TOKEN.exec(text);
    if (!m || m[5] !== undefined) return null;
    const start = m.index + m[0].length - m[0].trimStart().length, end = m.index + m[0].length;
    if (m[1] !== undefined) out.push({ kind: "str", text: m[1].slice(1, -1).replaceAll("''", "'"), start, end });
    else if (m[2] !== undefined) out.push({ kind: "id", text: m[2].slice(1, -1).replaceAll('""', '"'), start, end });
    else out.push({ kind: m[3] !== undefined ? "op" : "word", text: m[3] ?? m[4], start, end });
  }
  return out;
}

const isKw = (t, ...words) => !!t && t.kind === "word" && words.includes(t.text.toLowerCase());

const VALUE_AFTER = new Set([...COMPARE, "("]);

/** What the word at `caret` in a filter is, for completion: {kind: "field" | "operator" | "value" | "joiner" | null,
 * field (for a value), from, to (the span the completion replaces), prefix (its text so far)}. */
export function completionContext(text, caret) {
  const toks = tokens(text.slice(0, caret));
  if (!toks) return { kind: null, from: caret, to: caret, prefix: "" };
  const last = toks.at(-1), partial = last && last.end === caret && (last.kind === "word" || last.kind === "id") ? last : null;
  const prev = partial ? toks.slice(0, -1) : toks;
  const at = { from: partial ? partial.start : caret, to: caret, prefix: partial ? partial.text : "" };
  if (fieldNext(prev)) return { kind: "field", ...at };
  if (valueNext(prev.at(-1))) return { kind: "value", field: fieldBefore(prev), ...at };
  if (termEnds(prev.at(-1))) return { kind: valueEnds(prev) ? "joiner" : "operator", ...at };
  return { kind: null, ...at };
}

/** After nothing, `and`, `or`, `not` or an opening parenthesis (not an `in` list's), a field comes next. */
function fieldNext(prev) {
  const p = prev.at(-1);
  if (!p || isKw(p, "and", "or")) return true;
  if (isKw(p, "not")) return !isField(prev.slice(0, -1)); // `field not` takes in or like
  return p.kind === "op" && p.text === "(" && !isKw(prev.at(-2), "in");
}

/** Whether `toks` ends with a field: a term that is neither a keyword nor a compared value. */
function isField(toks) {
  const q = toks.at(-1);
  return !!q && termEnds(q) && !isKw(q, "and", "or", "not") && !valueEnds(toks);
}

const valueNext = (p) => (p.kind === "op" && (VALUE_AFTER.has(p.text) || p.text === ",")) || isKw(p, "like");

const termEnds = (p) => p.kind === "word" || p.kind === "str" || p.kind === "id" || (p.kind === "op" && p.text === ")");

/** Whether the term ending `prev` is a value (after an operator, `like`, `is` or a list comma), not a field. */
function valueEnds(prev) {
  const q = prev.at(-2);
  return !!q && (COMPARE.has(q.text) || q.text === "," || isKw(q, "like", "is"));
}

/** The field a value at the end of `toks` compares: before its operator, `like`, or `in (` list. */
function fieldBefore(toks) {
  let i = toks.length - 1;
  while (i >= 0 && !(COMPARE.has(toks[i].text) || isKw(toks[i], "like", "in"))) i--;
  let j = i - 1;
  if (isKw(toks[j], "not")) j--;
  return toks[j] && (toks[j].kind === "word" || toks[j].kind === "id") ? toks[j].text : null;
}

/** A value as filter text: numbers and plain words bare, anything else in single quotes. */
export function literal(v) {
  const t = textOf(v);
  if (typeof v === "number" || typeof v === "boolean") return t;
  return IDENTIFIER.test(t) && !KEYWORDS.has(t.toLowerCase()) && !NUMBER.test(t) ? t : `'${t.replaceAll("'", "''")}'`;
}

/** A field as the filter language writes it: bare when it is a plain word, else double-quoted. */
export function fieldText(name) {
  return IDENTIFIER.test(name) ? name : `"${name.replaceAll('"', '""')}"`;
}

/** A number, or the text of one (including the markers "nan", "inf", "-inf"); else null. */
export function asNumber(v) {
  if (typeof v === "number") return v;
  if (typeof v !== "string") return null;
  if (Object.hasOwn(MARKERS, v)) return MARKERS[v];
  return NUMBER.test(v.trim()) ? Number(v) : null;
}

/** "nan", "inf" or "-inf": how a non-finite number is written everywhere. */
export const nonFiniteText = (v) => (v !== v ? "nan" : v > 0 ? "inf" : "-inf");

/** The text a value is matched as: strings as they are, true/false, integers without a point, JSON otherwise. */
export function textOf(v) {
  if (typeof v === "string") return v;
  if (typeof v === "boolean") return v ? "true" : "false";
  if (typeof v === "number") return Number.isFinite(v) ? String(v) : nonFiniteText(v);
  return JSON.stringify(v);
}

const any = (v, f) => (v == null ? false : Array.isArray(v) ? v.some((x) => x != null && f(x)) : f(v));

function equal(v, lit) {
  if (typeof v === "boolean" && typeof lit === "boolean") return v === lit;
  const a = asNumber(v), b = asNumber(lit);
  if (a !== null && b !== null) return a === b || (Number.isFinite(a - b) && Math.abs(a - b) <= 1e-12 * Math.max(Math.abs(a), Math.abs(b)));
  return textOf(v) === textOf(lit);
}

function order(v, lit, op) {
  const a = asNumber(v), b = asNumber(lit);
  const [x, y] = a !== null && b !== null ? [a, b] : typeof v === "string" && typeof lit === "string" ? [v, lit] : [null, null];
  if (x === null) return false;
  return op === "<" ? x < y : op === "<=" ? x <= y : op === ">" ? x > y : x >= y;
}

function like(pattern) {
  const src = [...pattern].map((c) => (c === "%" ? ".*" : c === "_" ? "." : c.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"))).join("");
  return new RegExp(`^(?:${src})$`, "is");
}

class Parser {
  constructor(toks) {
    this.toks = toks;
    this.i = 0;
    this.fields = [];
  }

  peek() {
    return this.toks[this.i] ?? null;
  }

  take() {
    const t = this.peek();
    if (!t) throw new Error("unexpected end");
    this.i++;
    return t;
  }

  expect(text) {
    const t = this.take();
    if (t.text.toLowerCase() !== text || !(t.kind === "op" || t.kind === "word")) throw new Error(`expected '${text}', got '${t.text}'`);
  }

  expr() {
    let left = this.conj();
    while (isKw(this.peek(), "or")) {
      this.i++;
      const a = left, b = this.conj();
      left = (g) => a(g) || b(g);
    }
    return left;
  }

  conj() {
    let left = this.unary();
    while (isKw(this.peek(), "and")) {
      this.i++;
      const a = left, b = this.unary();
      left = (g) => a(g) && b(g);
    }
    return left;
  }

  unary() {
    if (isKw(this.peek(), "not")) {
      this.i++;
      const e = this.unary();
      return (g) => !e(g);
    }
    const t = this.peek();
    if (t && t.kind === "op" && t.text === "(") {
      this.i++;
      const e = this.expr();
      this.expect(")");
      return e;
    }
    return this.comparison();
  }

  field() {
    const t = this.take();
    if (t.kind === "id" || (t.kind === "word" && !KEYWORDS.has(t.text.toLowerCase()))) {
      this.fields.push(t.text);
      return t.text;
    }
    throw new Error(`expected a field, got '${t.text}'`);
  }

  value() {
    const t = this.take();
    if (t.kind === "str") return t.text;
    const low = t.text.toLowerCase();
    if (t.kind !== "word" || (KEYWORDS.has(low) && !["true", "false", "null"].includes(low))) throw new Error(`expected a value, got '${t.text}'`);
    if (low === "true" || low === "false") return low === "true";
    if (low === "null") return null;
    return NUMBER.test(t.text) ? Number(t.text) : t.text;
  }

  comparison() {
    const f = this.field();
    if (isKw(this.peek(), "not")) {
      this.i++;
      if (!isKw(this.peek(), "in", "like")) throw new Error("expected 'in' or 'like' after 'not'");
      const test = this.membership(f);
      return (g) => !test(g);
    }
    if (isKw(this.peek(), "in", "like")) return this.membership(f);
    if (isKw(this.peek(), "is")) return this.isNull(f);
    const t = this.take();
    if (t.kind !== "op" || !COMPARE.has(t.text)) throw new Error(`expected an operator after '${f}', got '${t.text}'`);
    return operator(f, t.text, this.value());
  }

  isNull(f) {
    this.i++;
    const wantNull = !isKw(this.peek(), "not");
    if (!wantNull) this.i++;
    this.expect("null");
    return (g) => (g(f) == null) === wantNull;
  }

  membership(f) {
    if (isKw(this.take(), "like")) {
      const rx = like(textOf(this.value()));
      return (g) => any(g(f), (v) => rx.test(textOf(v)));
    }
    this.expect("(");
    const values = [this.value()];
    while (this.peek()?.kind === "op" && this.peek().text === ",") {
      this.i++;
      values.push(this.value());
    }
    this.expect(")");
    return (g) => any(g(f), (v) => values.some((x) => equal(v, x)));
  }
}

function operator(f, op, lit) {
  if (lit === null && ["=", "==", "!=", "<>"].includes(op)) return (g) => (g(f) == null) === (op === "=" || op === "==");
  if (op === "=" || op === "==") return (g) => any(g(f), (v) => equal(v, lit));
  if (op === "!=" || op === "<>") return (g) => !any(g(f), (v) => equal(v, lit));
  if (op === "~" || op === "!~") {
    const rx = new RegExp(textOf(lit), "i"), hit = (g) => any(g(f), (v) => rx.test(textOf(v)));
    return op === "~" ? hit : (g) => !hit(g);
  }
  return (g) => any(g(f), (v) => order(v, lit, op));
}
