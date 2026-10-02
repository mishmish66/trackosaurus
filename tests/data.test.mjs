import test from "node:test";
import assert from "node:assert/strict";
import { Data } from "../trex/static/data.js";

test("an empty answer for a finer tile drops the tile it replaces and frees its bytes", () => {
  const d = new Data({}), e = { fine: new Map() };
  assert.equal(d.setFine(e, "3|0", [{ bytes: 100 }], 10), true);
  assert.equal(d.setFine(e, "3|1", [{ bytes: 40 }], 10), true);
  assert.equal(d.fineBytes, 140);
  assert.equal(d.setFine(e, "3|0", [], 20), true);
  assert.deepEqual([d.fineBytes, [...e.fine.keys()]], [40, ["3|1"]]);
  assert.equal(d.setFine(e, "3|2", [], 20), false);
  assert.equal(d.fineBytes, 40);
});
