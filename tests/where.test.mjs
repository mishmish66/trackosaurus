import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { asNumber, compileWhere, completionContext, literal, runField, textOf } from "../trex/static/where.js";

const CASES = JSON.parse(readFileSync(new URL("./where_cases.json", import.meta.url), "utf8"));

test("where selects the runs the shared cases name", () => {
  for (const [expr, want] of CASES.cases) {
    const { test: t } = compileWhere(expr);
    const got = Object.entries(CASES.runs).filter(([, run]) => t((f) => run[f] ?? null)).map(([k]) => k);
    assert.deepEqual(got, want, expr);
  }
});

test("where refuses clauses that do not parse", () => {
  for (const expr of CASES.errors) assert.throws(() => compileWhere(expr), Error, expr);
});

test("where reports the fields it reads", () => {
  assert.deepEqual(compileWhere("lr = 1 and (summary.loss < 2 or tags is null)").fields, ["lr", "summary.loss", "tags"]);
  assert.deepEqual(compileWhere("ppo").fields, ["name", "path"]);
});

test("values are matched as their text, and only numbers and their text count as numbers", () => {
  assert.deepEqual([1, 0.001, true, "x", [1, "a"], NaN].map(textOf), ["1", "0.001", "true", "x", '[1,"a"]', "NaN"]);
  assert.deepEqual(["1e-3", " 2 ", "Infinity", "inf", true, "1_000"].map(asNumber), [0.001, 2, Infinity, null, null, null]);
});

test("UI runs resolve fields as the CLI does: config first, then summary, with prefixes and nested info", () => {
  const r = { seq: 7, mseq: 2, meta: { id: "a/r", name: "r", state: "finished", tags: ["x"],
    config: { lr: 0.1, both: "config", "opt/name": "adam", none: null }, summary: { loss: "NaN", both: 2.0, acc: 0.5, _step: 9 },
    info: { git: { sha: "abc" }, "a.b": 1 } } };
  const want = { lr: 0.1, both: "config", acc: 0.5, missing: null, "opt/name": "adam", none: null, "config.lr": 0.1, "c.lr": 0.1,
    "summary.acc": 0.5, "s.both": 2.0, "m.missing": null, "info.git.sha": "abc", "info.a.b": 1, "info.git.nope": null,
    state: "finished", path: "a/r", step: 9, rows: 7, media: 2, tags: ["x"], dir: null };
  for (const [f, v] of Object.entries(want)) assert.deepEqual(runField(r, f), v, f);
});

test("completion knows whether a field, operator, value or joiner comes next, and which field a value is for", () => {
  const at = (text) => { const c = compileContext(text); return [c.kind, c.field ?? null, c.prefix]; };
  const compileContext = (t) => completionContext(t, t.length);
  assert.deepEqual(at(""), ["field", null, ""]);
  assert.deepEqual(at("l"), ["field", null, "l"]);
  assert.deepEqual(at("lr "), ["operator", null, ""]);
  assert.deepEqual(at("lr = "), ["value", "lr", ""]);
  assert.deepEqual(at("lr = 0.0"), ["value", "lr", "0.0"]);
  assert.deepEqual(at("lr = 0.001 "), ["joiner", null, ""]);
  assert.deepEqual(at("lr = 0.001 and se"), ["field", null, "se"]);
  assert.deepEqual(at("seed not in (0, "), ["value", "seed", ""]);
  assert.deepEqual(at("seed in (0"), ["value", "seed", "0"]);
  assert.deepEqual(at("algo like "), ["value", "algo", ""]);
  assert.deepEqual(at("not ("), ["field", null, ""]);
  assert.deepEqual(at("seed not "), ["operator", null, ""]);
  assert.deepEqual(at("lr = 1 and not "), ["field", null, ""]);
  assert.deepEqual(at("\"opt/name\" = "), ["value", "opt/name", ""]);
  assert.equal(completionContext("name = 'it", 10).kind, null);
  const c = completionContext("lr = 0.0 and seed = 1", 8);
  assert.deepEqual([c.kind, c.from, c.to], ["value", 5, 8]);
});

test("completion writes values as filter text", () => {
  assert.deepEqual([0.001, "PPO", "push-t:v2", "two words", "and", "12", "it's", true].map(literal),
                   ["0.001", "PPO", "push-t:v2", "'two words'", "'and'", "'12'", "'it''s'", "true"]);
});
