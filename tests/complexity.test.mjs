// Cyclomatic complexity and unused variables of the UI, by eslint (`npm install` provides eslint).
import test from "node:test";
import assert from "node:assert/strict";
import { readdirSync, readFileSync } from "node:fs";
import { Linter } from "eslint";

const MAX_COMPLEXITY = 15;
const dir = new URL("../trex/static/", import.meta.url);

test("every UI function stays within the complexity limit and declares no unused variables", () => {
  const linter = new Linter();
  const config = { languageOptions: { ecmaVersion: "latest", sourceType: "module" }, rules: { complexity: ["error", MAX_COMPLEXITY], "no-unused-vars": "error" } };
  const over = [];
  for (const f of readdirSync(dir).filter((f) => f.endsWith(".js")))
    for (const m of linter.verify(readFileSync(new URL(f, dir), "utf8"), config)) over.push(`${f}:${m.line} ${m.message}`);
  assert.deepEqual(over, []);
});
