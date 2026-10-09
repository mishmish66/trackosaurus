"""Headless-browser smoke test against a throwaway trex server on a temporary runs directory.

Checks cold and warm loads, grouping, opening groups as path levels, nested chart sections and pinning, panels of
hidden runs, the x range of hidden runs, charts while the server refuses blocks, the filter box, the canvases' colors in a
page that is dark, the page without WebGL2, dots where a line has one point, console errors, UI line coverage (at least UI_COVERAGE of the modules' code lines run),
and that a client dropping every 5th stream event still converges to the run files: every row and media item, and columns whose
points' counts add up to each metric's finite values. Then a trex pulling another's runs through a link added in its panel,
and, as this machine's trex (with a private TREX_DAEMON_DIR): adding a directory with `trex serve -y`, the root view of
both, making a workspace of them in the panel the trex brand opens, and from that panel removing a directory and
re-adding it from the remembered ones.

    uv run playwright install chromium   # once
    uv run python tests/browser_smoke.py [screenshot_dir]
"""

import base64
import functools
import json
import math
import os
import re
import shutil
import socket
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.request
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt
from playwright.sync_api import ConsoleMessage, Error, Page, Request, Response, Route, sync_playwright

from trex import chunks
from trex.format import connect_ro

REPO = Path(__file__).resolve().parents[1]
LIVE_WRITER = """
import sys, time, trex
runs = [trex.init(f"{sys.argv[1]}/live/r{i}", commit_interval=0.1) for i in range(3)]
for step in range(300):
    for i, r in enumerate(runs):
        r.log({"loss": 1 / (step + 1) + i, "acc": step / 300}, step=step)
        if step % 100 == 0:
            r.log_html("report", f"<b>{i} @ {step}</b>", step=step)
    time.sleep(0.03)
for r in runs:
    r.finish()
"""


FLICKER_WRITER = """
import math, random, sys, time, trex
runs = [trex.init(f"{sys.argv[1]}/flicker/{g}/r{i}", commit_interval=0.1) for g in "ab" for i in range(3)]
rngs = [random.Random(i) for i in range(len(runs))]
def log(step):
    for r, rng in zip(runs, rngs):
        r.log({"loss": math.exp(-step / 200) + rng.gauss(0, 0.05)}, step=step)
for step in range(260):
    log(step)
print("ready", flush=True)
for step in range(260, 500):  # no power of two between: the top tiles keep their level
    log(step)
    time.sleep(0.05)
for r in runs:
    r.finish()
"""

HIDDEN_LIVE_WRITER = """
import math, sys, time, trex
runs = [trex.init(f"{sys.argv[1]}/hiddenlive/r{i}", commit_interval=0.05) for i in range(2)]
for r in runs:
    r.log({"loss": 1.0}, step=0)
print("ready", flush=True)
for step in range(1, 500):
    for r in runs:
        r.log({"loss": math.exp(-step / 100)}, step=step)
    time.sleep(0.02)
for r in runs:
    r.finish()
"""

KEPT_WRITER = """
import sys, trex
def run(name, lr):
    r = trex.init(f"{sys.argv[1]}/kept/{name}", config={"lr": lr}, commit_interval=0.05)
    for step in range(40):
        r.log({"loss": lr / (step + 1)}, step=step)
    return r
for name, lr in (("a0", 0.1), ("a1", 0.1), ("b0", 0.2)):
    run(name, lr).finish()
live = run("live", 0.2)
print("ready", flush=True)
sys.stdin.readline()
live.finish()
print("finished", flush=True)
sys.stdin.readline()
run("late", 0.1).finish()
print("added", flush=True)
"""

EXTENT_WRITER = """
import sys, trex
for name, n in (("short", 100), ("long", 1000)):
    r = trex.init(f"{sys.argv[1]}/extent/{name}")
    for i in range(n):
        r.log({"loss": 1.0 / (i + 1)}, step=i)
    r.finish()
"""

MANY_WRITER = """
import math, sys, trex
from concurrent.futures import ThreadPoolExecutor
def one(gi):
    g, i = gi
    r = trex.init(f"{sys.argv[1]}/many/{g}/r{i}", commit_interval=0.01)
    for s in range(200):
        r.log({"loss": math.exp(-s / 50) + (i % 7) * 0.01 + (0.1 if g == "b" else 0) + 0.02 * math.sin(s * (i + 1))}, step=s)
    r.finish()
with ThreadPoolExecutor(16) as pool:
    list(pool.map(one, [(g, i) for g in "ab" for i in range(160)]))
"""

RECORD_DRAWS = """() => {
  window.draws = [];
  const C = [...app.charts.values()][0].constructor;
  if (C.prototype.recorded) return;
  C.prototype.recorded = true;
  const draw = C.prototype.draw;
  C.prototype.draw = function (...args) {
    draw.apply(this, args);
    const v = this.view;
    if (!v || !this.w) return;
    const lines = v.lines.map((ln) => {
      if (v.gpu) return [ln.group ?? ln.id, v.gpu.bins, this.binX(ln, v.gpu.bins - 1, v.gpu.agg)]; // binned on the GPU: a point a bin
      const c = ln.cols?.[0], n = ln.xy ? ln.xy.length / 2 : c ? c.n : 0;
      return [ln.run?.id ?? ln.group ?? ln.label, n, ln.xy ? ln.xy[ln.xy.length - 2] : c && c.n ? c.s[c.n - 1] : null];
    });
    draws.push({ key: this.key, paced: app.paced, y0: v.y0, y1: v.y1, lines });
  };
}"""


# Count the page's reads from the GPU in window.reads, from 0.
READS_HOOK = """(async () => { const gl = (await import('/static/gl.js')).renderer().gl;
    if (!gl.__counted) { const read = gl.readPixels; gl.readPixels = function (...a) { window.reads++; return read.apply(this, a); }; gl.__counted = true; }
    window.reads = 0; })()"""

# [bin width, width of the finest buckets shown, whether the bins the chart's width asks for are narrower] of the loss
# chart's view binned from its buckets, on the GPU or on its worker (a heatmap's lines then lie at the bins'
# centers); null when it shows none.
BIN_FLOOR = ("async () => { const { binGrid } = await import('/static/kernel.js'), c = app.charts.get('loss'), v = c.view;"
             " const L = app.data.charts.get('loss')?.ready, f = L && (L.fine || L.coarse);"
             " const s = v?.lines?.find((l) => l.cols?.[0]?.n >= 2)?.cols[0].s, g = v?.gpu;"
             " const dx = g ? g.dx : v?.lines?.[0]?.dx ?? (s ? s[1] - s[0] : null);"
             " return dx && f ? [dx, 2 ** f.level, binGrid(v.x0, v.x1, Math.max(8, Math.floor(c.pw / 8))).dx < 2 ** f.level] : null; }")

# Group line l of chart c as its bins: [center, band low, band high, runs, raw center] each, however it was binned (by a
# worker: the line's own arrays; on the GPU: its row of the binning, read back).
BINS = ("((c, l) => { const g = c.view.gpu, row = g && g.out.row(l.gi, g.bins);"
        " return Array.from({ length: c.gridOf(l).bins }, (_, i) => (g ? [row[4 * i], c.view.logy && !(row[4 * i + 1] > 0) ? row[4 * i] : row[4 * i + 1],"
        " row[4 * i + 2], row[4 * i + 3], NaN] : [l.center[i], l.lo[i], l.hi[i], l.cnt[i], l.raw ? l.raw[2 * i + 1] : NaN])); })")

type Draw = dict[str, Any]  # a chart draw RECORD_DRAWS records: key, paced, y0, y1, lines ([id, points, end])
type At = Callable[[float, float], tuple[float, float]]  # a point of a chart's plot area by its fractions of it


def first_line(proc: subprocess.Popen[str]) -> str:
    """The first line `proc` prints on its piped stdout."""
    assert proc.stdout is not None
    return proc.stdout.readline()


def flicker_faults(draws: list[Draw]) -> list[str]:
    """What a viewer would see flicker between consecutive draws of a chart: a line with fewer points or its end
    further back, a line gone for a draw, and a streamed redraw shrinking the y axis."""
    faults: list[str] = []
    by: dict[str, list[Draw]] = {}
    for d in draws:
        by.setdefault(d["key"], []).append(d)
    for key, ds in by.items():
        for a, b in zip(ds, ds[1:]):
            was = {l[0]: l for l in a["lines"]}
            for k, n, tip in b["lines"]:
                if k in was and (n < was[k][1] or (tip is not None and was[k][2] is not None and tip < was[k][2])):
                    faults.append(f"{key} {k}: {was[k][1]} points to {n}, end {was[k][2]} to {tip}")
            if b["paced"] and (b["y0"] > a["y0"] or b["y1"] < a["y1"]):
                faults.append(f"{key}: y axis [{a['y0']:.4g}, {a['y1']:.4g}] to [{b['y0']:.4g}, {b['y1']:.4g}]")
        for a, b, c in zip(ds, ds[1:], ds[2:]):
            gone = {l[0] for l in a["lines"]} & {l[0] for l in c["lines"]} - {l[0] for l in b["lines"]}
            faults += [f"{key} {k}: gone for a draw" for k in gone]
    return faults


# Move the pointer to and fro over a chart for 800 ms, a move a frame, then let it rest on the chart: the streamed
# redraws RECORD_DRAWS recorded [while it moved, by 700 ms after it came to rest], and the elements of the page changed
# [once it was moving (its first frames show the tooltip), by then]; the data layer gives a status while it moves.
POINTER_MOVES = """async () => {
  const c = [...app.charts.values()].find((c) => c.view && c.inView), r = c.overlay.getBoundingClientRect(), t0 = performance.now();
  const move = (k) => c.overlay.dispatchEvent(new MouseEvent("mousemove", { clientX: r.left + 70 + (k % 30) * 6, clientY: r.top + r.height / 2, bubbles: true }));
  const changed = [], seen = new MutationObserver((ms) => { if (performance.now() - t0 > 150) for (const m of ms) changed.push((m.target.nodeType === 1 ? m.target : m.target.parentElement)?.id || m.target.nodeName); });
  seen.observe(document.documentElement, { subtree: true, childList: true, attributes: true, characterData: true });
  draws = [];
  let k = 0;
  await new Promise((done) => { const t = setInterval(() => (performance.now() - t0 < 800 ? (move(k++), k === 20 && app.data.ui.status("given while it moved")) : (clearInterval(t), done())), 16); });
  const moving = [draws.filter((d) => d.paced).length, [...new Set(changed)]];
  await new Promise((ok) => setTimeout(ok, 700));
  seen.disconnect();
  const out = [moving[0], draws.filter((d) => d.paced).length, moving[1], changed.length];
  app.data.ui.status(app.data.summary());
  c.overlay.dispatchEvent(new MouseEvent("mouseleave"));
  return out;
}"""


def flicker_smoke(page: Page, url: str, runs: Path) -> bool:
    """Whether charts drawing runs that stream, line by line and grouped, never show a line shrink, its end move back
    or vanish for a draw, nor a streamed redraw shrink the y axis; and whether streamed redraws wait for a pointer
    moving over a chart to rest."""
    writer = subprocess.Popen([sys.executable, "-c", FLICKER_WRITER, str(runs)], stdout=subprocess.PIPE, text=True)
    try:
        first_line(writer)
        page.goto(f"{url}/?flicker#path=flicker&group=run")
        page.wait_for_function("window.app && app.data.runs.size === 6 && [...app.charts.values()].some((c) => c.view)", timeout=30000)
        page.evaluate(RECORD_DRAWS)
        page.wait_for_timeout(3500)
        lines = page.evaluate("draws")
        moving, rested, changed, changes = page.evaluate(POINTER_MOVES)
        page.evaluate("draws = []; app.setGroup('run~1')")
        page.wait_for_timeout(3500)
        groups = page.evaluate("draws")
    finally:
        writer.wait()
    faults = flicker_faults(lines) + flicker_faults(groups)
    print(f"flicker: {len(lines)} draws of lines, {len(groups)} of groups while 6 runs streamed; "
          + (f"faults {faults[:5]}" if faults else "no line shrank, moved back or vanished, no streamed redraw shrank an axis")
          + f"; {moving} streamed redraws while the pointer moved over a chart, {rested} once it rested; elements changed while it moved"
          + f" {changed}, {changes} changes by the time it had rested")
    return len(lines) > 4 and len(groups) > 4 and not faults and moving == 0 and rested > 0 and not changed and changes > 0


def many_value(group: str, i: int, step: int) -> float:
    """What MANY_WRITER logs."""
    return math.exp(-step / 50) + (i % 7) * 0.01 + (0.1 if group == "b" else 0) + 0.02 * math.sin(step * (i + 1))


def group_medians(group: str, g0: float, dx: float, bins: int) -> list[float | None]:
    """Per bin of a grid, the median over a MANY_WRITER group's runs of each run's mean of its rows in the bin."""
    out: list[float | None] = []
    for k in range(bins):
        steps = [s for s in range(200) if g0 + k * dx <= s < g0 + (k + 1) * dx]
        out.append(statistics.median(statistics.fmean(many_value(group, i, s) for s in steps) for i in range(160)) if steps else None)
    return out


def hidden_extent_smoke(page: Page, url: str, runs: Path) -> bool:
    """Whether hiding a run takes its steps out of a chart's x range, and showing it again puts them back."""
    subprocess.run([sys.executable, "-c", EXTENT_WRITER, str(runs)], check=True)
    page.goto(f"{url}/?extent#path=extent&group=run")
    page.wait_for_function("window.app && app.data.runs.size === 2", timeout=60000)
    page.wait_for_function(SETTLED + " && app.charts.get('loss')?.view", timeout=60000)
    end = "(() => { const v = app.charts.get('loss').view; return [v.lines.length, Math.round(v.ex1)]; })()"

    def toggle_long() -> list[int]:
        page.click("tr:has-text('long') input[type=checkbox]")
        page.wait_for_timeout(300)
        page.wait_for_function(SETTLED, timeout=60000)
        return page.evaluate(end)

    both = page.evaluate(end)
    hidden = toggle_long()
    back = toggle_long()
    print(f"hidden extent: lines and x range end with both runs {both}, the long one hidden {hidden}, shown again {back}")
    return both[0] == 2 and both[1] > 900 and hidden[0] == 1 and hidden[1] < 110 and back == both


LINE_PIXELS = """(key) => {
  const cv = app.charts.get(key).canvas, d = cv.getContext('2d').getImageData(0, 0, cv.width, cv.height).data;
  let n = 0;
  for (let i = 0; i < d.length; i += 4) if (d[i + 3] && Math.max(d[i], d[i + 1], d[i + 2]) - Math.min(d[i], d[i + 1], d[i + 2]) > 60) n++;
  return n;
}"""


def line_pixels_smoke(page: Page, url: str) -> bool:
    """Whether charts put their lines on their canvases: grouped, one per run in a folder, and grouped again after
    going back, drawn as at first whatever the folder's lines drew before."""
    chart = "app.charts.get('train/loss')"

    def drawn() -> int:
        page.wait_for_function(f"{chart}?.el.isConnected", timeout=60000)
        page.evaluate(f"{chart}.el.scrollIntoView()")
        page.wait_for_function(f"{SETTLED} && {chart}.view?.lines.length", timeout=60000)
        page.wait_for_timeout(400)
        return page.evaluate(LINE_PIXELS, "train/loss")

    page.goto(f"{url}/?pixels#path=sweep&group=lr")
    page.wait_for_function("window.app && app.data.runs.size > 0 && app.grouped", timeout=60000)
    grouped = drawn()
    page.evaluate("app.setPath('sweep/width512/lr0.01')")
    page.wait_for_function("!app.grouped", timeout=60000)
    lines = drawn()
    page.evaluate("history.back()")
    page.wait_for_function("app.grouped", timeout=60000)
    again = drawn()
    print(f"line pixels: grouped {grouped}, a folder's runs {lines}, grouped again {again}")
    return grouped > 300 and lines > 300 and abs(again - grouped) <= 0.02 * grouped


def near(off: list[float]) -> bool:
    """Whether there are differences, all below 1e-5."""
    return bool(off) and max(off) < 1e-5


def off_medians(lines: list[Any]) -> list[float]:
    """How far each bin's center of group lines `lines` ([group, g0, dx, centers] each) lies from its exact median."""
    return [abs(c - w) for g, g0, dx, center in lines for c, w in zip(center, group_medians(g, g0, dx, len(center))) if w is not None]


def zoom_many(page: Page, url: str, hash_: str) -> None:
    """The many folder at `hash_`, zoomed to steps [60, 140] once settled."""
    page.goto(f"{url}/?binned#path=many&{hash_}")
    page.wait_for_function("window.app && app.data.runs.size === 320", timeout=60000)
    page.wait_for_function(SETTLED, timeout=60000)
    page.evaluate("app.setXRange([60, 140, 0])")
    page.wait_for_timeout(300)
    page.wait_for_function(SETTLED, timeout=60000)


def bin_floors(page: Page) -> list[Any]:
    """BIN_FLOOR of the loss chart's view, then of zooms whose spans the planner rounds to a level coarser than the
    bins; back at [60, 140] after."""
    out = [page.evaluate(BIN_FLOOR)]
    for r in ([60, 130], [40, 175], [0, 140], [60, 140]):
        page.evaluate(f"app.setXRange([{r[0]}, {r[1]}, 0])")
        page.wait_for_timeout(300)
        page.wait_for_function(SETTLED, timeout=60000)
        out.append(page.evaluate(BIN_FLOOR))
    return out[:-1]


def narrow_of(floors: list[Any]) -> list[Any]:
    """The readings of BIN_FLOOR whose bins are narrower than their buckets, or that found no view."""
    return [f for f in floors if not f or f[0] < f[1]]


def rested_drag(page: Page, x0: float, x1: float, key: str = "loss") -> tuple[int, int]:
    """Drag across chart `key` from x0 to x1 (fractions of its canvas's width; it extends past the plot by the axes'
    margins), rest, release: how often the page read the GPU (READS_HOOK) while the drag rested, and at the release."""
    page.evaluate("app.setXRange(null)")
    page.wait_for_function(SETTLED, timeout=60000)
    canvas = page.locator(f".panel:has(.pname:text-is('{key}')) canvas").nth(1)
    canvas.scroll_into_view_if_needed()
    page.wait_for_function(SETTLED, timeout=60000)
    box = canvas.bounding_box()
    assert box is not None
    y = box["y"] + box["height"] / 2
    page.mouse.move(box["x"] + x0 * box["width"], y)
    page.mouse.down()
    page.evaluate("window.reads = 0")  # a tooltip's reads before the drag are not the drag's
    page.mouse.move(box["x"] + x1 * box["width"], y, steps=4)
    page.wait_for_timeout(500)
    rested = page.evaluate("window.reads")
    page.mouse.up()
    page.wait_for_function("!!app.xrange", timeout=5000)
    page.wait_for_function(SETTLED, timeout=60000)
    page.mouse.move(5, 5)
    return rested, page.evaluate("window.reads") - rested


def binned_smoke(page: Page, url: str, runs: Path) -> bool:
    """Whether a zoom of more runs than a chart draws one by one draws them from bins of their buckets, as group
    statistics (each group's median per bin of its runs' means of their rows) binned on the GPU, while its drag rests
    when it is dragged, so that its release reads nothing back from the GPU (also when it ends in the axis' margin),
    and on the chart's worker once a pass on the GPU fails, and as a heatmap; whether no such view takes bins narrower
    than the buckets it bins; and whether a server of another protocol is stated."""
    subprocess.run([sys.executable, "-c", MANY_WRITER, str(runs)], check=True)
    zoom_many(page, url, "group=run~1")
    floors = bin_floors(page)
    centers = f"c.view.lines.map((l) => [l.label.split(' ')[0], c.gridOf(l).g0, c.gridOf(l).dx, {BINS}(c, l).map((b) => b[0])])"
    binned, lines, worker, on_gpu = page.evaluate(f"(() => {{ const c = app.charts.get('loss'); return [c.binned, {centers}, !!c.stats, !!c.view.gpu]; }})()")
    page.evaluate(READS_HOOK)
    dragged = rested_drag(page, 0.4, 0.7)
    dragged_off = off_medians(page.evaluate(f"(() => {{ const c = app.charts.get('loss'); return {centers}; }})()"))
    margin = rested_drag(page, 0.6, 0.02)  # its release lies in the y axis' margin: the zoom starts where the plot does
    page.evaluate("""(async () => { const gl = (await import('/static/gl.js')).renderer().gl, draw = gl.drawArrays;
        gl.drawArrays = () => { gl.drawArrays = draw; throw new Error('a pass this GPU cannot run'); };
        app.setXRange([60, 139, 0]); })()""")
    page.wait_for_timeout(300)
    page.wait_for_function(SETTLED, timeout=60000)
    fell_back, by_worker = page.evaluate(f"(() => {{ const c = app.charts.get('loss'); return [!c.view.gpu, {centers}]; }})()")
    floors += bin_floors(page)  # on the worker
    zoom_many(page, url, "group=run")
    heat = page.evaluate("(() => { const c = app.charts.get('loss'); return [c.binned, c.view.density, c.view.lines.length]; })()")
    floors += bin_floors(page)
    page.evaluate("app.showProtocol(1)")
    stated = page.evaluate("[!document.querySelector('#mismatch').hidden, document.querySelector('#mismatch').title]")
    off, groups = off_medians(lines), sorted(l[0] for l in lines)
    checks = {
        "a zoom of many runs draws bins of their buckets, binned on the GPU": all([binned is True, worker, on_gpu]),
        "each group's center in a bin is its runs' exact median": all([groups == ["a", "b"], near(off)]),
        "a dragged zoom is binned while it rests, and its release reads nothing back": all([dragged[0] >= 1, dragged[1] == 0, near(dragged_off)]),
        "so is one released in the y axis' margin": all([margin[0] >= 1, margin[1] == 0]),
        "after a failed pass on the GPU, the chart's worker bins it": all([fell_back, near(off_medians(by_worker))]),
        "a heatmap of them is drawn": heat == [True, True, 320],
        "no binned view takes bins narrower than its buckets, though some would have": all([not narrow_of(floors), any(f[2] for f in floors)]),
        "a server of another protocol is stated": all([stated[0], "the server 1" in stated[1]]),
    }
    failed = [name for name, passed in checks.items() if not passed]
    print(f"binned: {len(checks) - len(failed)}/{len(checks)} as intended; groups {groups} of {len(lines[0][3]) if lines else 0} bins, "
          f"at most {max(off, default=1):.2g} off their medians; reads of a rested drag {dragged}, of one released in the margin "
          f"{margin}; heatmap {heat}; bins and buckets {floors}" + (f"; not: {failed}" if failed else ""))
    return not failed


class Cdp(Protocol):
    """The part of a Chrome DevTools Protocol session the coverage uses."""

    def send(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]: ...


def cdp_session(page: Page) -> Cdp:
    """A DevTools Protocol session of `page`."""
    return page.context.new_cdp_session(page)


class JsCoverage:
    """Which lines of the UI's modules ran, from Chromium's V8 coverage and the node tests'. A page load discards the previous page's
    counts, so `take` runs before every navigation (`watch` makes the page do so) and takes are merged per file."""

    def __init__(self, page: Page) -> None:
        self.cdp = cdp_session(page)
        self.cdp.send("Profiler.enable")
        self.cdp.send("Profiler.startPreciseCoverage", {"callCount": True, "detailed": True})
        self.ran: dict[str, npt.NDArray[np.bool_]] = {}  # module -> per UTF-16 unit: whether it ran

    def watch(self, page: Page) -> None:
        for name in ("goto", "go_back", "go_forward", "reload", "click"):
            setattr(page, name, self._taking_first(getattr(page, name)))

    def _taking_first[**P, R](self, step: Callable[P, R]) -> Callable[P, R]:
        """`step`, taking the coverage first."""
        def taking(*a: P.args, **kw: P.kwargs) -> R:
            self.take()
            return step(*a, **kw)
        return taking

    def take(self) -> None:
        for script in self.cdp.send("Profiler.takePreciseCoverage")["result"]:
            self.merge(script)

    def add_node_tests(self) -> None:
        """Merge in the node tests' coverage of the same modules (V8's, through NODE_V8_COVERAGE)."""
        with tempfile.TemporaryDirectory() as d:
            subprocess.run(["node", "--test", *map(str, sorted((REPO / "tests").glob("*.test.mjs")))], cwd=REPO, check=True,
                           capture_output=True, env={**os.environ, "NODE_V8_COVERAGE": d})
            for f in Path(d).glob("*.json"):
                for script in json.loads(f.read_text())["result"]:
                    self.merge(script)

    def merge(self, script: dict[str, Any]) -> None:
        """Add one script's V8 coverage, if it is a UI module."""
        url = script["url"].split("?")[0]
        name = url.rsplit("/", 1)[-1]
        if "/static/" not in url or not (STATIC / name).is_file():
            return
        ran = np.zeros(len(units(name)), bool)
        # outer ranges first, so a nested block's count overrides its function's
        for r in sorted((r for f in script["functions"] for r in f["ranges"]), key=lambda r: (r["startOffset"], -r["endOffset"])):
            ran[r["startOffset"]:r["endOffset"]] = r["count"] > 0
        self.ran[name] = self.ran.get(name, ran) | ran

    def report(self) -> dict[str, tuple[int, int, list[int]]]:
        """{module: (covered lines, lines with code, uncovered line numbers)}; a line is covered when any
        non-blank character on it ran."""
        out: dict[str, tuple[int, int, list[int]]] = {}
        for name, ran in sorted(self.ran.items()):
            line, blank = units(name).T
            code = ~blank.astype(bool)
            has = np.bincount(line[code], minlength=line.max() + 1) > 0
            hit = np.bincount(line[code & ran], minlength=line.max() + 1) > 0
            out[name] = (int(hit.sum()), int(has.sum()), [int(i) + 1 for i in np.flatnonzero(has & ~hit)])
        return out


STATIC = REPO / "trex/static"
UI_COVERAGE = 0.92  # share of the UI modules' code lines the smoke test must run


@functools.cache
def units(name: str) -> npt.NDArray[np.int64]:
    """(line, is blank) per UTF-16 code unit of a UI module, the unit V8 counts offsets in."""
    out: list[tuple[int, bool]] = []
    for i, text in enumerate((STATIC / name).read_text().split("\n")):
        for ch in text + "\n":
            out.extend([(i, ch.isspace())] * (2 if ord(ch) > 0xFFFF else 1))
    return np.array(out[:-1] if out else [(0, True)], np.int64)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def file_columns(run_dir: Path) -> tuple[int, dict[str, int], int]:
    """(rows, {key: finite values}, media items) of a run file."""
    c = connect_ro(run_dir)
    rows = chunks.rows(c)
    mseq: int = c.execute("SELECT count(*) FROM media").fetchone()[0]
    c.close()
    finite = Counter(k for r in rows for k, v in r.values.items() if math.isfinite(v))
    return len(rows), dict(finite), mseq


READY = ("window.app && app.data.runs.size > 0 && app.charts.size > 0 && !app.data.queue.length && !app.data.posts"
         " && [...app.charts.values()].some((c) => c.view)")
# every run loaded under the path shown is listed
LISTED = ("app.runList.length === [...app.data.runs.keys()].filter((id) => !app.opts.path || id === app.opts.path"
          " || id.startsWith(app.opts.path + '/')).length")
SETTLED = READY + f" && {LISTED} && !app.data.busy && !app.round && !app.raf && !app.soon && !app.planTimer"


def group_levels_smoke(page: Page, url: str) -> bool:
    """Whether opening a group narrows the view to its runs, drawn as lines, a group nested inside it opens a deeper
    level, and the path bar and back button return to each level, the group-by staying as set."""
    state = """() => [app.opts.focus.length, app.opts.group, app.grouped, app.runList.filter((r) => r.shown).length,
        [...document.querySelectorAll('#crumbPath .crumb')].map((e) => e.textContent)]"""
    def ends(text: str) -> None:
        page.wait_for_function(f"[...document.querySelectorAll('#crumbPath .crumb')].at(-1)?.textContent === {json.dumps(text)}")

    page.goto(f"{url}/?levels#path=sweep&group=lr")
    page.wait_for_function(READY, timeout=30000)
    sizes = page.evaluate("Object.fromEntries([...app.groups.values()].map((g) => [g.values.join(), g.runs.length]))")
    page.click("button.gfocus[title='open this group']")
    page.wait_for_function("app.opts.focus.length === 1")
    lr = page.evaluate("app.opts.focus[0][1][0]")
    ends(f"lr: {lr}")
    opened = page.evaluate(state)
    page.evaluate("app.setGroup('lr / seed')")
    page.click("button.gfocus[title='open this group']")
    page.wait_for_function("app.opts.focus.length === 2")
    seed = page.evaluate("app.opts.focus[1][1][0]")
    ends(f"seed: {seed}")
    nested = page.evaluate(state)
    both = page.evaluate(f"app.runList.filter((r) => String(r.meta.config.lr) === {json.dumps(lr)} && String(r.meta.config.seed) === {json.dumps(seed)}).length")
    page.click(f"#crumbPath .crumb:text-is('lr: {lr}')")
    up = page.evaluate(state)
    page.go_back()
    page.wait_for_function("app.opts.focus.length === 2")
    back = page.evaluate(state)[:2]
    page.click("#crumbPath .crumb:text-is('sweep')")
    home = page.evaluate(state)[:3]
    print(f"group levels: opened lr {lr} {opened}; nested seed {seed} {nested}; up {up}; back {back}; folder {home}")
    return (opened[:4] == [1, "lr", False, sizes[lr]] and opened[4][-2:] == ["sweep", f"lr: {lr}"]
            and nested[:4] == [2, "lr / seed", False, both] and nested[4][-3:] == ["sweep", f"lr: {lr}", f"seed: {seed}"]
            and up[:3] == [1, "lr / seed", True] and back == [2, "lr / seed"] and home == [0, "lr / seed", True])


# Whether the tooltip's canvas is shown and holds the tooltip where the app says it drew it: opaque inside its box, clear
# outside.
TIP_DRAWN = """(() => { const el = document.getElementById("tipCanvas"), at = app.tipAt, b = app.tipCanvas.box, dpr = devicePixelRatio;
  if (el.hidden || !at || !b) return false;
  const alpha = (x, y) => el.getContext("2d").getImageData(Math.round(x * dpr), Math.round(y * dpr), 1, 1).data[3];
  return b[0] === at.x && b[1] === at.y && alpha(b[0] + b[2] / 2, b[1] + 8) === 255 && alpha(b[0] + b[2] / 2, b[1] + b[3] + 40) === 0; })()"""

# Whether the pinned tooltip's elements lie where the canvas drew it, the canvas hidden, showing the rows it showed.
TIP_PINNED = """(() => { const t = document.getElementById("tip"), r = t.getBoundingClientRect(), at = app.tipAt, rows = [...t.querySelectorAll(".trow")];
  return !t.hidden && t.classList.contains("pinned") && document.getElementById("tipCanvas").hidden && Math.abs(r.left - at.x) < 1 && Math.abs(r.top - at.y) < 1
    && rows.some((el) => el.classList.contains("near") && el._ln === app.tipView.rows[app.tipView.near].ln); })()"""


def hover_mutations(page: Page, at: At) -> list[str]:
    """Move the pointer over the plot `at`, its tooltip already shown, and return what the moves changed of the document
    outside the header and the sidebar (which the data that comes meanwhile may change): the mutated nodes, each as its
    tag, id and class."""
    page.evaluate("""() => { window.__muts = [];
      window.__mo = new MutationObserver((ms) => { for (const m of ms) { const t = m.target.nodeType === 1 ? m.target : m.target.parentElement;
        if (!t?.closest("header, aside")) window.__muts.push(`${m.type} ${t?.tagName}#${t?.id}.${t?.className}`); } });
      window.__mo.observe(document.documentElement, { subtree: true, childList: true, attributes: true, characterData: true }); }""")
    for k in range(8):
        page.mouse.move(*at(0.3 + 0.05 * k, 0.5))
        page.evaluate("new Promise((ok) => requestAnimationFrame(() => requestAnimationFrame(ok)))")
    return list(page.evaluate("(() => { window.__mo.disconnect(); return window.__muts; })()"))


def interactions_smoke(page: Page, url: str) -> bool:
    """Whether charts carry no legend and the chart controls do what they say: hover and Shift-pinned tooltips
    (which the wheel scrolls, and whose rows reveal and open runs), x and box zooms and their reset, the chart
    settings (smoothing, axes, outliers, density, reset), the sort box (type, pick, Enter, Escape), sidebar hiding,
    group-by search, keyboard scrolling, the media slider, and back/forward (back to the grouping the folder had)."""
    key = "train/loss"
    checks: dict[str, object] = {}
    sel = f".panel:has(.pname:text-is('{key}'))"
    chart = f"app.charts.get({key!r})"

    def plot() -> At:
        page.locator(f"{sel} canvas").nth(1).scroll_into_view_if_needed()
        b = page.locator(f"{sel} canvas").nth(1).bounding_box()
        assert b is not None
        return lambda fx, fy: (b["x"] + b["width"] * fx, b["y"] + b["height"] * fy)

    def soon(js: str, timeout: float = 3000) -> bool:
        """Whether `js` becomes true within `timeout` ms."""
        try:
            page.wait_for_function(js, timeout=timeout)
            return True
        except Exception:
            return False

    def hover(at: At) -> None:
        page.mouse.move(*at(0.5, 0.5))
        page.mouse.move(*at(0.55, 0.45))
        page.wait_for_function("!!app.tipAt", timeout=5000)

    page.goto(f"{url}/?ix#path=sweep&group=")
    page.wait_for_function(READY, timeout=30000)
    checks["charts carry no legend (the hover and the sidebar name the lines)"] = page.locator(".panel .legend").count() == 0
    at = plot()
    page.mouse.move(*at(0.2, 0.5))
    page.mouse.down()
    page.mouse.move(*at(0.6, 0.52), steps=6)
    page.mouse.up()
    checks["a drag zooms x"] = page.evaluate("!!app.xrange")
    page.mouse.move(*at(0.3, 0.2))
    page.mouse.down()
    page.mouse.move(*at(0.5, 0.8), steps=6)
    page.mouse.up()
    checks["a box zooms y"] = page.evaluate(f"!!{chart}.yzoom")
    page.click("#resetZoom")
    checks["reset zoom clears both"] = page.evaluate(f"!app.xrange && !{chart}.yzoom")

    page.click(f"{sel} .gear")

    def row(label: str) -> str:
        return f"#menu .srow:has(> label:text-is('{label}'))"

    page.check(f"{row('smoothing')} input[type=checkbox]")
    page.locator(f"{row('smoothing')} input[type=range]").fill("0.9")
    page.select_option(f"{row('y scale')} select", "true")
    page.select_option(f"{row('x axis')} select >> nth=0", '"runtime"')
    page.select_option(f"{row('ignore outliers')} select", "0.01")
    page.fill(f"{row('x range')} input >> nth=0", "0")
    page.locator(f"{row('x range')} input >> nth=0").evaluate("(e) => e.dispatchEvent(new Event('change', { bubbles: true }))")
    page.wait_for_timeout(600)
    cfg = page.evaluate(f"app.panelCfg[{key!r}] || {{}}")
    checks["settings are saved for the chart"] = {"smooth", "logy", "x", "outliers", "xmin"} <= set(cfg)
    checks["settings change the view"] = soon(f"(() => {{ const v = {chart}.view; return !!v && v.logy && v.xmode === 1 && v.alpha > 0; }})()")
    page.click("#menu button:text-is('reset chart')")
    checks["reset chart drops its settings"] = page.evaluate(f"!app.panelCfg[{key!r}]")
    page.select_option(f"{row('lines')} select", '"density"')
    page.click("#menu button:text-is('close')")
    page.wait_for_function(f"{chart}.view?.density", timeout=10000)
    hover(plot())
    checks["a density chart lists the nearest runs"] = "nearest" in page.evaluate("app.tipView.heading")
    page.mouse.move(5, 5)
    page.click(f"{sel} .gear")
    page.click("#menu button:text-is('reset chart')")
    page.click("#menu button:text-is('close')")

    order = lambda: page.evaluate("app.runList.map((r) => r.id).join()")
    before = order()
    page.click("#sortBy")
    checks["the sort box lists every field when focused"] = page.locator("#menu .mitem").count() == page.evaluate("app.sortFields.length")
    page.keyboard.type("los")
    checks["typing narrows the sort fields to matches"] = soon(
        "(() => { const b = [...document.querySelectorAll('#menu .mitem')]; return b.length > 0 && b.length < app.sortFields.length"
        " && b.every((x) => x.textContent.includes('los')); })()")
    page.click("#menu .mitem:has-text('train/loss')")
    checks["a picked field sorts the runs"] = soon("app.opts.sort === 'metric:train/loss' && document.querySelector('#sortBy').value === 'train/loss'")
    page.click("#sortDir")
    checks["sorting reorders the runs"] = order() != before
    page.click("#sortBy")
    page.keyboard.type("zzz")
    checks["a field that matches nothing says so"] = soon("!!document.querySelector('#menu .mhint')")
    page.keyboard.press("Escape")
    checks["escape keeps the current sort"] = soon("app.opts.sort === 'metric:train/loss' && document.querySelector('#sortBy').value === 'train/loss'")
    page.click("#sortBy")
    page.keyboard.type("creat")
    page.keyboard.press("Enter")
    checks["enter picks the first match"] = soon("app.opts.sort === 'created'")
    page.click("#sortDir")
    page.click("#hideAll")
    checks["hide all hides every run"] = soon("app.runList.every((r) => !r.shown)")
    page.click("#hideAll")
    soon("[...document.querySelectorAll('#runTable tr:has(td.name a) input[type=checkbox]')].every((c) => c.checked)")
    page.locator("#runTable tr:has(td.name a) input[type=checkbox]").last.uncheck()
    checks["a run's checkbox hides it"] = soon("app.runList.filter((r) => !r.shown).length === 1")
    page.locator("#runTable tr:has(td.name a) input[type=checkbox]").last.check()
    page.click("#groupBy")
    page.keyboard.press("Control+a")
    page.keyboard.type("see")
    checks["the group-by box completes fields"] = soon(
        "JSON.stringify([...document.querySelectorAll('#menu .mitem .ml')].map((e) => e.textContent)) === '[\"seed\"]'")
    page.keyboard.press("Escape")
    page.keyboard.press("Escape")
    checks["leaving the group-by box applies it"] = soon("app.opts.group === 'see' && document.activeElement.id !== 'groupBy'")
    page.evaluate("app.setGroup('')")
    page.locator("#panels").click(position={"x": 5, "y": 5})
    for k in ("End", "Home", "PageDown", "ArrowUp"):
        page.keyboard.press(k)
    slider = page.locator(".panel.media input[type=range]").first
    slider.scroll_into_view_if_needed()
    soon("+document.querySelector('.panel.media input[type=range]').max > 0")
    slider.fill("0")
    checks["the media slider steps back"] = page.evaluate("[...app.mediaPanels.values()].some((m) => !m.follow)")

    hover(plot())
    checks["the tooltip shows the line nearest the cursor"] = page.evaluate("(() => { const v = app.tipView, a = app.tipAt.a; return v.near >= a && v.near < a + 14; })()")
    checks["the tooltip is drawn beside the pointer, on its canvas"] = page.evaluate(TIP_DRAWN)
    checks["a hover changes no element of the page"] = hover_mutations(page, plot()) == []
    page.keyboard.down("Shift")
    checks["shift pins the tooltip, as elements in its place"] = page.evaluate(TIP_PINNED)
    page.keyboard.up("Shift")
    checks["letting shift go draws the tooltip again"] = page.evaluate("document.querySelector('#tip').hidden && " + TIP_DRAWN)
    page.keyboard.down("Shift")
    checks["shift pins the tooltip"] = page.evaluate("document.querySelector('#tip').classList.contains('pinned')")
    checks["a pinned tooltip lists every line"] = page.evaluate("app.tipView.rows.length === app.runList.filter((r) => r.shown).length")
    scroll = "document.querySelector('#tip .tlist').scrollTop"
    top, room = page.evaluate(f"[{scroll}, document.querySelector('#tip .tlist').scrollHeight - document.querySelector('#tip .tlist').clientHeight]")
    page.mouse.wheel(0, 100 if top < room else -100)
    checks["the wheel scrolls a pinned tooltip smoothly"] = soon(f"Math.abs({scroll} - {top}) >= 18")
    tip_row = page.locator("#tip .trow").first
    tip_row.hover()
    checks["a pinned row marks its run in the sidebar"] = page.evaluate("!!app.sideMark")
    tip_row.click()
    page.keyboard.up("Shift")
    page.wait_for_function("app.scopeIsRun && app.runList.length === 1 && !!document.querySelector('#infoPanel .st')", timeout=10000)
    checks["a pinned row opens its run"] = True
    page.go_back()
    page.wait_for_function(READY + " && !app.scopeIsRun", timeout=30000)
    checks["back from a run restores the folder ungrouped, as it was"] = page.evaluate("!app.opts.group.length")
    page.go_forward()
    page.wait_for_function("app.scopeIsRun", timeout=30000)

    width = "document.querySelector('aside').offsetWidth"
    w0, g = page.evaluate(width), page.locator("#sideGrip").bounding_box()
    assert g is not None
    page.mouse.move(g["x"] + g["width"] / 2, g["y"] + 300)
    page.mouse.down()
    page.mouse.move(g["x"] + g["width"] / 2 + 120, g["y"] + 300, steps=8)
    page.mouse.up()
    checks["dragging the grip widens the sidebar"] = page.evaluate(width) >= w0 + 100
    page.reload()
    page.wait_for_function(READY, timeout=30000)
    checks["the sidebar keeps its width after a reload"] = page.evaluate(width) >= w0 + 100
    page.dblclick("#sideGrip")
    checks["double-clicking the grip resets the width"] = soon(f"{width} === {w0}")
    page.goto(f"{url}/?alone#path=sweep")
    page.wait_for_function(READY, timeout=30000)
    top = page.evaluate("(() => { const p = document.getElementById('panels'); p.scrollTop = Math.min(400, p.scrollHeight - p.clientHeight); return p.scrollTop; })()")
    page.evaluate("""() => { const p = document.getElementById('panels'), box = p.getBoundingClientRect();
        [...p.querySelectorAll('.panel button.full')].find((b) => { const r = b.getBoundingClientRect(); return r.top > box.top && r.bottom < box.bottom; }).click(); }""")
    page.wait_for_function("document.getElementById('panels').classList.contains('alone')", timeout=10000)
    page.keyboard.press("Escape")
    checks["leaving a chart shown alone returns to where the charts were scrolled"] = top > 0 and soon(
        f"!document.getElementById('panels').classList.contains('alone') && Math.abs(document.getElementById('panels').scrollTop - {top}) <= 2")
    page.goto(f"{url}/?iqm#path=sweep&group=lr&center=iqm")
    page.wait_for_function(READY, timeout=30000)
    plot()
    checks["IQM draws each group's interquartile mean with a CI band"] = soon(
        f"(() => {{ const v = {chart}.view, k = v?.lines[0]; return document.querySelector('#center').value === 'iqm' && !!k"
        f" && {BINS}({chart}, k).some(([c, lo, hi]) => c > lo && c < hi); }})()", timeout=10000)
    span = page.evaluate(f"""(() => {{ const v = {chart}.view, c = v.lines.flatMap((l) => {BINS}({chart}, l).map((b) => b[0]).filter(Number.isFinite));
        return [Math.min(...c), Math.max(...c), v.y0, v.y1]; }})()""")
    reach = (span[1] - span[0]) * 0.25
    checks["the y axis follows the group lines; a band widens it by at most a quarter"] = (
        span[2] <= span[0] and span[3] >= span[1] and span[2] >= span[0] - reach - 0.05 * (span[1] - span[0] + 2 * reach)
        and span[3] <= span[1] + reach + 0.05 * (span[1] - span[0] + 2 * reach))
    failed = [k for k, v in checks.items() if not v]
    print(f"interactions: {len(checks) - len(failed)}/{len(checks)} as intended" + (f"; not: {failed}" if failed else ""))
    return not failed


def dir_pages(url: str) -> dict[str, str]:
    """{name: page path} of every directory the trex at `url` serves."""
    with urllib.request.urlopen(f"{url}/api/node") as r:
        return {x["name"]: x["url"] for x in json.loads(r.read())["dirs"]}


def node_smoke(page: Page, runs: Path, tmp: Path, env: dict[str, str], out: Path, errors: list[str]) -> bool:
    """Whether this machine's trex's root shows every tracked directory (a folder in it as `/ name`), a workspace made in the panel (opened from the trex
    brand) merges them and opens though its URL was visited before it existed, and the panel removes, adds and
    forgets directories. The refused add's 400 is taken out
    of `errors`."""
    port = free_port()
    url = f"http://127.0.0.1:{port}"
    node = subprocess.Popen([sys.executable, "-m", "trex", "serve", str(runs / "sweep"), "--port", str(port),
                             "--cache", str(tmp / "cache")], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    row = "#menu .mrow:has(.ml:text-is('{}'))"
    try:
        first_line(node)
        subprocess.run([sys.executable, "-m", "trex", "serve", str(runs / "live"), "-y"], check=True, env=env)
        page_of = dir_pages(url)
        with urllib.request.urlopen(f"{url}{page_of['sweep']}api/runs") as r1, urllib.request.urlopen(f"{url}{page_of['live']}api/runs") as r2:
            total = len(json.loads(r1.read())["runs"]) + len(json.loads(r2.read())["runs"])
        page.goto(f"{url}/w/both/")
        page.goto(f"{url}/")
        page.wait_for_function(READY, timeout=30000)
        root = page.evaluate("[app.data.runs.size, app.rootName, [...new Set([...app.data.runs.values()].map((r) => r.id.split('/')[0]))].sort()]")
        page.goto(f"{url}/#path=sweep")
        page.wait_for_function("document.querySelector('#crumbPath .seg.current')?.textContent.startsWith('sweep')", timeout=30000)
        crumbs = page.evaluate("[...document.querySelectorAll('#crumbPath > .seg > .crumb, #crumbPath > .sep')].map((e) => e.textContent)")
        page.click(".brand")
        page.click("#menu button:text-is('+ new workspace')")
        page.fill("#menu .madd input", "both")
        for box in page.query_selector_all("#menu .mcheck input"):
            box.check()
        page.click("#menu button:text-is('save')")
        page.wait_for_url(f"{url}/w/both/")
        page.wait_for_function(READY, timeout=30000)
        merged = page.evaluate("[app.data.runs.size, app.groupFieldItems().some((f) => f.label === 'dir')]")
        page.screenshot(path=str(out / "workspace.png"))
        page.click("#crumbPath .crumb:text-is('/')")
        page.wait_for_url(f"{url}/")
        page.wait_for_function(READY, timeout=30000)
        page.click(".brand")
        page.click(row.format("live") + " .mitem")
        page.wait_for_url(f"{url}{page_of['live']}")
        page.wait_for_function(READY, timeout=30000)
        page.once("dialog", lambda d: d.accept())
        page.click(".brand")
        page.click(row.format("live") + " .chev")
        page.wait_for_url(f"{url}/")
        page.wait_for_function(READY, timeout=30000)
        page.goto(f"{url}{page_of['sweep']}")
        page.wait_for_function(READY, timeout=30000)
        page.click(".brand")
        page.fill("#menu .madd input", str(tmp / "no-such-dir"))
        before = len(errors)
        page.click("#menu .madd button")
        page.wait_for_function("document.querySelector('#menu .merr').textContent.includes('not a directory')")
        refused = errors[before:]
        del errors[before:]
        if len(refused) != 1 or "400" not in refused[0]:
            errors += refused
        page.click("#menu .mrecent .mitem")
        page.wait_for_url(f"{url}{page_of['live']}")
        page.wait_for_function(READY, timeout=30000)
        page.click(".brand")
        page.wait_for_selector("#menu .mrow")
        page.screenshot(path=str(out / "node_menu.png"))
        recent_left = page.query_selector("#menu .mrecent") is not None
        panel_text = page.inner_text("#menu")
        tidy = "null" not in panel_text.split() and "unknown" not in panel_text
        with urllib.request.urlopen(f"{url}/api/node") as r:
            d = json.loads(r.read())
        left, workspaces = [x["name"] for x in d["dirs"]], [(w["name"], w["members"]) for w in d["workspaces"]]
        print(f"node: root shows {root[0]}/{total} runs in {root[2]} as {root[1]!r}, a folder in it as {crumbs}; workspace of both shows "
              f"{merged[0]}/{total}, dir field {merged[1]}; removed live, re-added it from history; tracking {left}, "
              f"workspaces {workspaces}, history {d['history']}, panel tidy {tidy}")
        return (root == [total, "/", ["live", "sweep"]] and crumbs == ["/", "sweep"] and merged == [total, True] and sorted(left) == ["live", "sweep"]
                and workspaces == [("both", ["sweep"])] and not d["history"] and not recent_left and tidy)
    finally:
        node.terminate()
        node.wait()


def link_smoke(page: Page, tmp: Path, env: dict[str, str], out: Path, upstream: str) -> bool:
    """Whether this machine's trex tracking nothing, given another trex's http://host:port in its panel, pulls and shows every run
    that trex holds, lists the directory under that trex, below this one, each with its × (the directory's removes it
    there), and lets go of that directory with the link, removed from that trex's row."""
    port = free_port()
    url = f"http://127.0.0.1:{port}"
    node = subprocess.Popen([sys.executable, "-m", "trex", "serve", "--port", str(port), "--cache", str(tmp / "cache-links"),
                             "--name", "laptop"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            env={**env, "TREX_DAEMON_DIR": str(tmp / "daemon-links")})
    try:
        first_line(node)
        with urllib.request.urlopen(f"{upstream}/api/runs") as r:
            want = len(json.loads(r.read())["runs"])
        page.goto(f"{url}/")
        page.wait_for_selector(".dhome .madd input")
        page.fill(".dhome .madd input", upstream)
        page.click(".dhome .madd button")
        page.wait_for_function(f"window.app && app.data.runs.size === {want} && app.charts.size > 0", timeout=60000)
        page.click(".brand")
        page.wait_for_selector("#menu .mrow")
        rows = page.evaluate("[...document.querySelectorAll('#menu .mrow')].map((r) => [r.querySelector('.ml').textContent, !!r.querySelector('.chev')])")
        page.screenshot(path=str(out / "links_menu.png"))
        page.once("dialog", lambda d: d.accept())
        page.click("#menu .mrow:has(.ms:text-is('%s')) .chev" % upstream)
        page.wait_for_selector(".dhome .madd input", timeout=30000)
        with urllib.request.urlopen(f"{url}/api/node") as r:
            d = json.loads(r.read())
        print(f"links: pulled {want} runs from {upstream}; panel rows {rows}; after removing the link: directories {d['dirs']}, links {d['links']}")
        return (want > 0 and len(rows) == 3 and rows[0] == ["laptop", False] and rows[1][1] and rows[2] == ["runs", True]
                and d["dirs"] == [] == d["links"])
    finally:
        node.terminate()
        node.wait()


NESTED_WRITER = """
import sys, numpy as np, trex
r = trex.init(f"{sys.argv[1]}/nested/r0")
for step in range(50):
    r.log({"eval/return/mean": step, "eval/return/std": 1.0, "eval/len": 10 + step, "loss": 1 / (step + 1)}, step=step)
r.log_image("eval/video/frame", np.zeros((8, 8, 3), np.uint8), step=49)
r.finish()
"""


VISIBLE = ("[...app.charts.values()].filter((c) => { const r = c.el.getBoundingClientRect(); "
           "return r.bottom > 0 && r.top < innerHeight && r.width; })")


def failing_blocks_smoke(page: Page, url: str) -> bool:
    """Whether, while the server refuses block requests, the charts and the status say why and the requests back off,
    and the charts draw once it answers again."""
    error, asked, t0 = "internal error: OSError(24, 'Too many open files')", list[float](), time.monotonic()

    def refuse(route: Route) -> None:
        asked.append(time.monotonic() - t0)
        route.fulfill(status=500, content_type="application/json", body=json.dumps({"error": error}))

    page.route("**/api/buckets", refuse)
    try:
        page.goto(f"{url}/?failing#path=sweep")
        page.wait_for_function(f"window.app && {VISIBLE}.length && {VISIBLE}.every((c) => app.data.failure(c.key))", timeout=30000)
        page.wait_for_timeout(5000)
        said: list[list[Any]] = page.evaluate(f"{VISIBLE}.map((c) => [!!c.view, c.emptyText()])")
        status = page.inner_text("#status")
    finally:
        page.unroute("**/api/buckets")
    page.wait_for_function(f"{VISIBLE}.every((c) => c.view) && !app.data.failed.size", timeout=40000)
    recovered = page.inner_text("#status")
    print(f"failing blocks: {len(asked)} requests in {asked[-1]:.1f} s ({', '.join(f'{t:.1f}' for t in asked)}); charts said "
          f"{sorted({t for _, t in said})}; status {status!r}; after the server answered again {recovered!r}")
    return (bool(said) and all(s == [False, f"failed to load: {error}"] for s in said) and status.endswith(f"failing: {error}")
            and len(asked) <= 30 and "failing" not in recovered)


def filter_smoke(page: Page, url: str) -> bool:
    """Whether the filter box keeps the runs a WHERE clause or a name search selects, marks a clause that does not
    parse while filtering nothing, and completes fields, a field's values (with counts) and and / or from the keyboard."""
    page.goto(f"{url}/?filter#path=sweep")
    page.wait_for_function(READY, timeout=30000)
    shown = "app.runList.filter((r) => r.shown).map((r) => r.id).sort()"
    every = page.evaluate(shown)
    want = page.evaluate("app.runList.filter((r) => r.meta.config.lr === 0.001 && r.meta.config.seed === 1).map((r) => r.id).sort()")
    results: dict[str, list[Any]] = {}
    for text in ["lr = 0.001 and seed = 1", "lr = 0.00", "seed1", "lr ="]:
        page.fill("#runFilter", text)
        page.wait_for_timeout(600)
        results[text] = [page.evaluate(shown), page.evaluate("document.querySelector('#runFilter').classList.contains('bad')")]
    page.fill("#runFilter", "")
    painted = "new Promise((ok) => requestAnimationFrame(() => setTimeout(ok, 0)))"  # completions of input follow the frame
    items = lambda: page.evaluate(f"{painted}.then(() => [...document.querySelectorAll('#menu .mitem .ml')].map((e) => e.textContent))")
    page.click("#runFilter")
    page.keyboard.type("l")
    field_items = items()
    page.keyboard.press("Tab")
    after_field = page.input_value("#runFilter")
    page.keyboard.type("= ")
    value_items = page.evaluate(f"{painted}.then(() => [...document.querySelectorAll('#menu .mitem')].map((e) => e.textContent))")
    page.keyboard.press("ArrowDown")
    page.keyboard.press("Enter")
    picked = page.input_value("#runFilter")
    page.wait_for_timeout(600)
    kept = page.evaluate(shown)
    page.keyboard.type("a")
    joiners = items()
    page.keyboard.press("Escape")
    page.fill("#runFilter", "")
    page.wait_for_timeout(400)
    lrs = sorted({str(x) for x in page.evaluate("app.runList.map((r) => r.meta.config.lr)")})
    complete = ("lr" in field_items and after_field == "lr " and len(value_items) == len(lrs)
                and all("runs" in t for t in value_items) and picked.startswith("lr = ") and picked.endswith(" ")
                and 0 < len(kept) < len(every) and joiners == ["and"])
    print(f"filter completion: 'l' offers {field_items[:4]}…, Tab gives {after_field!r}; 'lr = ' offers {value_items}; "
          f"↓ Enter gives {picked!r}, keeping {len(kept)}; then 'a' offers {joiners}")
    return (complete and want and results["lr = 0.001 and seed = 1"] == [want, False] and results["lr = 0.00"] == [[], False]
            and results["seed1"][0] == [r for r in every if "seed1" in r] and results["lr ="] == [every, True])


def kept_runs_smoke(page: Page, url: str, runs: Path) -> bool:
    """Whether a view shown again (the same path, filter, grouping and sorting) shows its runs as computing them anew
    would: restored as they were kept while nothing they depend on changed, also through a change of a run's metadata
    that leaves them as they were, and computed anew once a run's state changed what a filter passes, a run came, or a
    run was hidden; and whether a filter on a run's rows, which grow without its metadata changing, is not kept."""
    shown = "app.runList.filter((r) => r.shown).map((r) => r.id).sort()"
    view = "JSON.stringify([app.runList.map((r) => [r.id, r.shown, r.match, r.color, r.part]), [...app.groups.keys()], app.grouped])"
    box = "#runTable tr:has(td.name[title='kept/a0']) input[type=checkbox]"

    def ids(*names: str) -> list[str]:
        return [f"kept/{n}" for n in names]

    def show(text: str, want: list[str]) -> tuple[list[str], str]:
        """The shown runs and the view once the filter is `text` and shows `want` (or what it shows instead)."""
        page.fill("#runFilter", text)
        try:
            page.wait_for_function(f"app.opts.filter === {json.dumps(text)} && JSON.stringify({shown}) === {json.dumps(json.dumps(want))}", timeout=5000)
        except Exception:
            pass
        page.keyboard.press("Escape")  # the box's completions, which lie over the sidebar
        return page.evaluate(shown), page.evaluate(view)

    def keep(name: str) -> bool:
        """Note the runs kept for the view shown, as `name`; whether there are any."""
        return page.evaluate(f"(window.keptOf ||= {{}})[{json.dumps(name)}] = app.runsNow, !!app.runsNow")

    def same(name: str) -> bool:
        """Whether the view shown is of the kept runs noted as `name`."""
        return page.evaluate(f"app.runsNow === window.keptOf[{json.dumps(name)}]")

    writer = subprocess.Popen([sys.executable, "-c", KEPT_WRITER, str(runs)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    assert writer.stdin is not None
    checks: dict[str, bool] = {}
    try:
        first_line(writer)
        page.goto(f"{url}/?kept#path=kept&group=lr")
        page.wait_for_function(f"window.app && app.data.runs.size === 4 && {LISTED} && app.data.runs.get('kept/live').meta.state === 'running'", timeout=30000)
        every, every_view = show("", ids("a0", "a1", "b0", "live"))
        checks["every run is shown, in two groups, and the view is kept"] = (every == ids("a0", "a1", "b0", "live") and page.evaluate("app.groups.size") == 2
                                                                           and keep("every"))
        running, running_view = show("state = 'running'", ids("live"))
        checks["the running run passes a filter on its state"] = running == ids("live") and keep("running")
        slow, slow_view = show("lr = 0.1", ids("a0", "a1"))
        checks["two runs pass a filter on their config"] = slow == ids("a0", "a1") and keep("slow")
        checks["a filter shown again is restored as it was kept"] = show("state = 'running'", ids("live")) == (running, running_view) and same("running")
        checks["the unfiltered view shown again is restored as it was kept"] = show("", every) == (every, every_view) and same("every")
        writer.stdin.write("\n")
        writer.stdin.flush()
        first_line(writer)
        page.wait_for_function("app.data.runs.get('kept/live').meta.state === 'finished'", timeout=20000)
        checks["a view kept through a change of metadata it does not depend on is restored"] = show("lr = 0.1", slow) == (slow, slow_view) and same("slow")
        checks["a filter on a state that changed is computed anew"] = show("state = 'running'", [])[0] == [] and not same("running")
        writer.stdin.write("\n")
        writer.stdin.flush()
        first_line(writer)
        page.wait_for_function(f"app.data.runs.size === 5 && {LISTED}", timeout=20000)
        checks["a view kept before a run came shows it"] = show("lr = 0.1", ids("a0", "a1", "late"))[0] == ids("a0", "a1", "late")
        show("", ids("a0", "a1", "b0", "late", "live"))
        page.locator(box).uncheck()
        checks["a view kept before a run was hidden shows it hidden"] = show("lr = 0.1", ids("a1", "late"))[0] == ids("a1", "late")
        show("", ids("a1", "b0", "late", "live"))
        page.locator(box).check()
        checks["and shown again once it is shown"] = show("lr = 0.1", ids("a0", "a1", "late"))[0] == ids("a0", "a1", "late")
        rows = show("rows >= 40", ids("a0", "a1", "b0", "late", "live"))[0]
        checks["a filter on rows shows the runs with them and is not kept"] = rows == ids("a0", "a1", "b0", "late", "live") and page.evaluate("app.runsNow === null")
        page.fill("#runFilter", "")
    finally:
        writer.stdin.close()
        writer.wait()
    failed = [name for name, passed in checks.items() if not passed]
    print(f"kept runs: {len(checks) - len(failed)}/{len(checks)} as intended" + (f"; not: {failed}" if failed else ""))
    return not failed


def listing_order_smoke(page: Page, url: str) -> bool:
    """Whether a page asks for its runs only once their stream has answered: a run added between a listing and the
    stream's start would be in neither."""
    order: list[str] = []

    def on_request(r: Request) -> None:
        if "/api/runs" in r.url:
            order.append("runs asked")

    def on_response(r: Response) -> None:
        if "/api/stream" in r.url:
            order.append("stream answered")

    page.on("request", on_request)
    page.on("response", on_response)
    try:
        page.goto(f"{url}/?order#path=sweep")
        page.wait_for_function(READY, timeout=30000)
    finally:
        page.remove_listener("request", on_request)
        page.remove_listener("response", on_response)
    print(f"listing order: {order[:4]}")
    return order[:2] == ["stream answered", "runs asked"]


def column_drag_smoke(page: Page, url: str) -> bool:
    """Whether a dragged zoom of a grouped chart drawn from its runs' columns bins nothing while it rests that its
    release bins anew: a release that changes the layers shown rebuilds the columns."""
    page.goto(f"{url}/?coldrag#path=sweep&group=lr")
    page.wait_for_function(READY, timeout=30000)
    page.wait_for_function(SETTLED, timeout=30000)
    page.evaluate(READS_HOOK)
    rested, released = rested_drag(page, 0.45, 0.55, "train/loss")
    binned = page.evaluate("!!app.charts.get('train/loss').view?.gpu && !app.charts.get('train/loss').binned")
    print(f"column drag: reads while it rested {rested}, at its release {released}; grouped from columns on the GPU {binned}")
    return binned and rested == 0 and released >= 1


def hidden_live_smoke(page: Page, url: str, runs: Path) -> bool:
    """Whether running runs that a filter hides leave the binned charts as they are while they stream: their columns
    are not rebuilt until a chart draws them again, so the GPU bins nothing anew meanwhile."""
    writer = subprocess.Popen([sys.executable, "-c", HIDDEN_LIVE_WRITER, str(runs)], stdout=subprocess.PIPE, text=True)
    state = "[app.data.runs.get('hiddenlive/r0')?.seq ?? 0, app.charts.get('loss')?.stats?.sig ?? null, window.reads ?? null, !!app.charts.get('loss')?.view?.gpu]"
    try:
        first_line(writer)
        page.goto(f"{url}/?hiddenlive#path=&group=run~1")
        page.wait_for_function("window.app && app.data.runs.has('hiddenlive/r1') && app.charts.has('loss')", timeout=30000)
        page.fill("#runFilter", "state = 'finished'")
        page.keyboard.press("Escape")
        page.evaluate("document.activeElement.blur()")
        page.wait_for_function(SETTLED, timeout=30000)
        page.wait_for_timeout(1000)
        page.evaluate(READS_HOOK)
        before = page.evaluate(state)
        page.wait_for_timeout(2500)
        after = page.evaluate(state)
        page.fill("#runFilter", "")
    finally:
        writer.wait()
    print(f"hidden live runs: rows {before[0]} to {after[0]} while hidden; the loss chart binned on the GPU {after[3]}, "
          f"binned anew {before[1] != after[1]}, reads from the GPU {after[2]}")
    return after[0] > before[0] and after[3] and before[1] == after[1] and after[2] == 0


def sections_smoke(page: Page, url: str) -> bool:
    """Whether chart sections nest by key path and fold one at a time, and a pinned chart shows in the pinned section
    while staying in its own."""
    tree = """() => { const walk = (d) => [d.querySelector(':scope > summary .stitle').textContent, d.open,
        [...d.querySelectorAll(':scope > .grid > .panel > .ptitle > span:first-child')].map((e) => e.textContent),
        [...d.querySelectorAll(':scope > details.section')].map(walk)];
        return [...document.querySelectorAll('#panels > details.section')].map(walk); }"""
    page.goto(f"{url}/?sections#path=nested")
    page.wait_for_function(READY, timeout=30000)
    page.wait_for_selector("#panels details.section details.section")
    nested = page.evaluate(tree)
    page.click("#panels details.section details.section > summary:has(.stitle:text-is('return (2)'))")
    folded = page.evaluate(tree)
    page.click("#panels details.section details.section > summary:has(.stitle:text-is('return (2)'))")
    head = "#panels details.section:has(> summary .stitle:text-is('eval (4)')) > summary"
    links = page.locator(f"{head} .sublinks button").all_inner_texts()
    sticky = page.locator(head).evaluate("(e) => getComputedStyle(e).position")
    page.click(f"{head} .subfold")
    inside = page.evaluate(tree)[1]
    page.click(f"{head} .subfold")
    reopened = page.evaluate(tree)[1]
    page.hover(".panel:has(.pname:text-is('eval/return/mean')) .ptitle")
    page.click(".panel:has(.pname:text-is('eval/return/mean')) .pin")
    page.wait_for_function("document.querySelector('#panels > details.section > summary')?.textContent.startsWith('pinned')")
    pinned = page.evaluate(tree)
    copies = page.evaluate("app.chartsOf('eval/return/mean').filter((c) => c.el.isConnected).length")
    page.click("#panels > details.section:first-of-type .pin")
    page.wait_for_function("!document.querySelector('#panels > details.section > summary')?.textContent.startsWith('pinned')")
    unpinned = [page.evaluate(tree)[0][0], page.evaluate("app.chartsOf('eval/return/mean').length")]
    print(f"sections: {nested}; return folded {folded[1][3][0][1]}, eval open {folded[1][1]}; eval links {links}, header {sticky}, "
          f"fold inside leaves {[inside[1]] + [c[1] for c in inside[3]]}; pinned {pinned[0]}, copies {copies}; unpinned {unpinned}")
    eval_sec = ["eval (4)", True, ["eval/len"], [["return (2)", True, ["eval/return/mean", "eval/return/std"], []],
                                                 ["video (1)", True, ["eval/video/frame"], []]]]
    return (nested == [["charts (1)", True, ["loss"], []], eval_sec] and folded[1][1] and not folded[1][3][0][1] and folded[1][3][1][1]
            and links == ["return (2)", "video (1)"] and sticky == "sticky" and [inside[1]] + [c[1] for c in inside[3]] == [True, False, False]
            and reopened == eval_sec
            and pinned[0] == ["pinned (1)", True, ["eval/return/mean"], []] and pinned[2] == eval_sec and copies == 2
            and unpinned == ["charts (1)", 1])


def hidden_panels_smoke(page: Page, url: str) -> bool:
    """Whether a hidden panel leaves its section for a link in the section's header that shows it again, the section
    menu checks shown panels and open subsections and toggles them, and hidden panels stay hidden on reload."""
    head = "#panels details.section:has(> summary .stitle:text-is('{}')) > summary"
    shown = "[...document.querySelectorAll('#panels .panel .ptitle > span:first-child')].map((e) => e.textContent).sort()"

    def links(title: str) -> list[str]:
        return page.locator(f"{head.format(title)} .sublinks .hiddenlink").all_inner_texts()

    def menu_items() -> list[list[str]]:
        return page.evaluate("[...document.querySelectorAll('#menu .mitem')].map((b) => [b.querySelector('.micon').textContent, "
                             "b.querySelector('.ml').textContent])")

    page.goto(f"{url}/?hiddenpanels#path=nested")
    page.wait_for_function(READY, timeout=30000)
    page.hover(".panel:has(.pname:text-is('eval/return/mean')) .ptitle")
    page.click(".panel:has(.pname:text-is('eval/return/mean')) .hide")
    page.wait_for_function(f"!{shown}.includes('eval/return/mean')")
    after_hide = [page.evaluate(shown), links("return (2)")]
    page.click(f"{head.format('eval (4)')} .secmenu")
    page.wait_for_selector("#menu .mitem")
    menu_before = menu_items()
    page.click("#menu .mitem:has(.ml:text-is('len'))")
    page.wait_for_function("!document.querySelector('#panels .pname') || ![...document.querySelectorAll('#panels .panel .ptitle > span:first-child')].some((e) => e.textContent === 'eval/len')")
    page.click("#menu .mitem:has(.ml:text-is('return/'))")
    folded = page.locator(head.format("return (2)")).evaluate("(s) => !s.parentElement.open")
    menu_after = menu_items()
    page.keyboard.press("Escape")
    eval_links = links("eval (4)")
    page.click(f"{head.format('return (2)')} .hiddenlink:text-is('mean')")
    page.wait_for_function(f"{shown}.includes('eval/return/mean')")
    page.reload()
    page.wait_for_function(READY, timeout=30000)
    persisted = [page.evaluate(shown), links("eval (4)")]
    page.click(f"{head.format('eval (4)')} .hiddenlink:text-is('len')")
    page.wait_for_function(f"{shown}.includes('eval/len')")
    page.locator(head.format("return (2)")).evaluate("(s) => { s.parentElement.open = true; }")
    print(f"hidden panels: hiding mean leaves {after_hide[0]}, return links {after_hide[1]}; eval menu {menu_before} then "
          f"{menu_after}, return folded {folded}, eval links {eval_links}; after reload {persisted}")
    return (after_hide == [["eval/len", "eval/return/std", "eval/video/frame", "loss"], ["mean"]]
            and menu_before == [["✓", "len"], ["✓", "return/"], ["✓", "video/"]]
            and menu_after == [[" ", "len"], [" ", "return/"], ["✓", "video/"]] and folded and eval_links == ["len"]
            and persisted == [["eval/return/mean", "eval/return/std", "eval/video/frame", "loss"], ["len"]])


def grouping_modes_smoke(page: Page, url: str) -> bool:
    """Whether runs group by their directory by default, one line per directory of several runs with a state dot;
    `run~2 / run~1` nests directories, each named within its parent, and opening one opens that directory; `run`
    lists the runs flat; and `visible = true` leaves out unchecked runs."""
    heads = "[...document.querySelectorAll('#runTable tr.grp .gname')].map((e) => e.firstChild.textContent)"
    runs = "document.querySelectorAll('#runTable tr:has(td.name a)').length"

    def group_by(text: str) -> None:
        page.fill("#groupBy", text)
        page.keyboard.press("Escape")
        page.keyboard.press("Escape")
        page.wait_for_function(f"app.opts.group === {json.dumps(text)}")

    page.goto(f"{url}/?modes#path=")
    page.wait_for_function(READY, timeout=30000)
    page.wait_for_function("document.querySelectorAll('#runTable tr.grp').length > 0")
    dirs = [page.evaluate(heads), page.evaluate("app.grouped"), page.evaluate("[app.opts.group, location.hash.includes('group=')]")]
    dots = page.evaluate("[...document.querySelectorAll('#runTable tr.grp td.stc .dot')].map((d) => [d.className, d.title])")
    group_by("run~2 / run~1")
    page.wait_for_function("[...document.querySelectorAll('#runTable tr.grp .gname')].some((e) => e.firstChild.textContent.startsWith('lr'))")
    nested = [page.evaluate(heads), page.evaluate("app.grouped")]
    page.click("#runTable tr.grp:has(td.gname[title='sweep/width128']) button.gfocus")
    page.wait_for_function("app.opts.path === 'sweep/width128' && app.runList.length === 9 && document.querySelectorAll('#runTable tr.grp').length === 3")
    opened = [page.evaluate("app.opts.focus.length"), page.evaluate(heads)]
    page.goto(f"{url}/?modes#path=&group=run")
    page.wait_for_function(READY, timeout=30000)
    flat = [page.evaluate(heads), page.evaluate(runs), page.evaluate("app.grouped")]
    first = page.locator("#runTable tr:has(td.name a)").first
    name = first.locator("td.name").get_attribute("title")
    first.locator("input[type=checkbox]").uncheck()
    page.fill("#runFilter", "visible = true")
    page.wait_for_function(f"![...document.querySelectorAll('#runTable td.name')].some((t) => t.title === {name!r})")
    filtered = [page.evaluate(runs), page.evaluate("app.runList.filter((r) => r.match).length")]
    page.fill("#runFilter", "")
    page.keyboard.press("Escape")
    page.wait_for_function(f"[...document.querySelectorAll('#runTable td.name')].some((t) => t.title === {name!r})")
    page.locator(f"#runTable tr:has(td.name[title='{name}']) input[type=checkbox]").check()
    print(f"grouping modes: run~1 {dirs}, dots {dots[:3]}; run~2 / run~1 {nested}, opened {opened}; run {flat}; "
          f"visible = true leaves {filtered} of {flat[1]}")
    lrs = ["lr0.001", "lr0.003", "lr0.01"]
    return ("sweep/width128/lr0.001" in dirs[0] and dirs[1] is True and dirs[2] == ["run~1", False]
            and len(dots) == len(dirs[0]) and all(c.split()[1] in ("running", "finished", "crashed", "failed") and t for c, t in dots)
            and {"sweep/width128", "sweep/width512", *lrs} <= set(nested[0]) and nested[1] is True
            and opened[0] == 0 and sorted(opened[1]) == lrs
            and flat[0] == [] and flat[1] > 10 and flat[2] is False and filtered == [flat[1] - 1, flat[1] - 1])


def hidden_runs_smoke(page: Page, url: str) -> bool:
    """Whether panels follow the shown runs: hiding the only run that logs some keys removes their panels, showing it
    brings them back, and hiding every run leaves no panels, as in a folder without runs."""
    panels = "[...document.querySelectorAll('#panels .panel > .ptitle > span:first-child')].map((e) => e.textContent).sort()"
    only = ["eval/len", "eval/return/mean", "eval/return/std", "eval/video/frame", "loss"]
    box = "#runTable tr:has(td.name[title='nested/r0']) input[type=checkbox]"

    def settle(want: list[str]) -> list[str]:
        try:
            page.wait_for_function(f"JSON.stringify({panels}) === {json.dumps(json.dumps(want))}", timeout=5000)
        except Exception:
            pass
        return page.evaluate(panels)

    page.goto(f"{url}/?hidden#path=&group=")
    page.wait_for_function(READY, timeout=30000)
    page.wait_for_function(f"{panels}.includes('eval/video/frame')", timeout=10000)
    before = page.evaluate(panels)
    rest = [k for k in before if k not in only]
    page.locator(box).uncheck()
    hidden = settle(rest)
    page.locator(box).check()
    shown = settle(before)
    page.click("#hideAll")
    none = settle([])
    count = page.inner_text("#panels .panelbar .muted")
    page.click("#hideAll")
    back = settle(before)
    print(f"hidden runs: {len(before)} panels; nested/r0 hidden leaves {len(hidden)}, shown again {len(shown)}; "
          f"all hidden {len(none)} ({count!r}); all shown {len(back)}")
    return (set(only) <= set(before) and bool(rest) and hidden == rest and shown == before and none == []
            and count == "0 sections · 0 panels" and back == before)


def side_mark_smoke(page: Page, url: str) -> bool:
    """Whether hovering a sidebar row traces its lines on the charts in view: a group's line when grouped, a run's line
    when not, and leaving the run table clears every chart's overlay."""
    traced = """[...app.charts.values()].filter((c) => c.inView && c.overlay.width).map((c) => {
      const d = c.overlay.getContext('2d').getImageData(0, 0, c.overlay.width, c.overlay.height).data;
      let n = 0;
      for (let i = 3; i < d.length; i += 4) n += d[i] > 0;
      return n;
    })"""
    out: list[Any] = []
    for hash_, row in (("#path=", "#runTable tr.grp"), ("#path=&group=run", "#runTable tr:has(td.name a)")):
        page.goto(f"{url}/?mark{hash_}")
        page.wait_for_function(READY, timeout=30000)
        page.locator(row).first.hover()
        page.wait_for_function(f"app.sideMarked && {traced}.some((n) => n > 0)", timeout=10000)
        marked = page.evaluate(traced)
        page.mouse.move(600, 300)
        page.wait_for_function(f"!app.sideMarked && {traced}.every((n) => n === 0)", timeout=10000)
        out.append([sum(1 for n in marked if n > 0), len(marked), page.evaluate("app.grouped")])
    print(f"side mark: charts traced of those in view, grouped then by run: {out}")
    return out[0][0] > 0 and out[0][2] is True and out[1][0] > 0 and out[1][2] is False


def palette(n: int) -> dict[str, str]:
    """The page's colors by name as index.html declares them: the light ones (0) or the dark ones (1)."""
    return dict(re.findall(r"--(\w+):([^;]+);", re.findall(r":root \{([^}]*)\}", (STATIC / "index.html").read_text())[n]))


def rgb(css: str) -> list[int]:
    """[r, g, b] of a CSS hex color."""
    h = css.lstrip("#")
    return [int(c * 2, 16) for c in h] if len(h) == 3 else [int(h[i:i + 2], 16) for i in (0, 2, 4)]


# How many pixels of chart `key`'s canvas have each of `colors` ([r, g, b]): its grid lines are drawn in the theme's.
CANVAS_COLORS = """([key, colors]) => { const c = app.charts.get(key).canvas, d = c.width ? c.getContext("2d").getImageData(0, 0, c.width, c.height).data : [];
  return colors.map(([r, g, b]) => { let n = 0; for (let i = 0; i < d.length; i += 4) n += d[i] === r && d[i + 1] === g && d[i + 2] === b && d[i + 3] === 255; return n; }); }"""

# The tooltip's background as its canvas holds it, between its border and its text, and where in the window: [[r, g, b], x, y].
TIP_BACKGROUND = """(() => { const [x, y, , h] = app.tipCanvas.box, at = [Math.round(x + 3), Math.round(y + h / 2)], dpr = devicePixelRatio;
  return [[...document.getElementById("tipCanvas").getContext("2d").getImageData(at[0] * dpr, at[1] * dpr, 1, 1).data.slice(0, 3)], ...at]; })()"""

# The colors of a PNG's pixels at `points`, as the browser decodes it.
PNG_PIXELS = """async ([png, points]) => { const image = await createImageBitmap(new Blob([Uint8Array.from(atob(png), (c) => c.charCodeAt(0))], { type: "image/png" }));
  const c = new OffscreenCanvas(image.width, image.height).getContext("2d");
  c.drawImage(image, 0, 0);
  return points.map(([x, y]) => [...c.getImageData(x, y, 1, 1).data.slice(0, 3)]); }"""

# What the page knows of its colors: those its canvases draw with (plot.js `theme`), whether it is told that the browser
# prefers dark, and the background color its own elements have.
THEME = """async () => { const t = (await import("/static/plot.js")).theme();
  return { draws: { bg: t.bg, fg: t.fg, muted: t.muted, grid: t.grid, line: t.line }, prefers: matchMedia("(prefers-color-scheme: dark)").matches,
           own: getComputedStyle(document.documentElement).getPropertyValue("--bg").trim() }; }"""


def theme_smoke(page: Page, url: str, out: Path) -> bool:
    """Whether the charts and the tooltip, which are canvases, are drawn in the page's dark colors wherever the page is
    dark: where the browser prefers the dark scheme, and where it darkens pages by force (Chromium's forced dark mode,
    which DevTools' automatic dark mode switches and qutebrowser's `colors.webpage.darkmode.enabled` turns on: it
    inverts the page's elements, tells the page it prefers light and leaves canvases as drawn). In a page loaded so, and
    in an open page as either is switched: the charts in view, those below it, and a tooltip shown then."""
    light, dark = palette(0), palette(1)
    keys = ["lr", "train/loss"]  # the first chart, and one further down whose tooltip is shown, the first then out of view
    plot = page.locator(".panel:has(.pname:text-is('train/loss')) canvas").nth(1)
    cdp = cdp_session(page)
    checks: dict[str, object] = {}

    def named(colors: dict[str, str]) -> str:
        return "dark" if colors is dark else "light"

    def force(on: bool) -> None:
        """Have the browser darken the page by force, or stop, as the page is."""
        cdp.send("Emulation.setAutoDarkModeOverride", {"enabled": on})

    def soon(js: str) -> bool:
        try:
            page.wait_for_function(js, timeout=5000)
            return True
        except Exception:
            return False

    def drawn(colors: dict[str, str]) -> bool:
        """Whether both charts come to show grid lines in `colors`' and none in the other palette's."""
        asked = json.dumps([rgb(colors["grid"]), rgb((light if colors is dark else dark)["grid"])])
        return soon(f"{json.dumps(keys)}.every((key) => {{ const [n, other] = ({CANVAS_COLORS})([key, {asked}]); return n > 0 && other === 0; }})")

    def tip_on(colors: dict[str, str]) -> bool:
        """Whether the tooltip comes to be drawn on `colors`' background."""
        return soon(f"{TIP_DRAWN} && JSON.stringify({TIP_BACKGROUND}[0]) === '{json.dumps(rgb(colors['bg']), separators=(',', ':'))}'")

    def hover() -> bool:
        """Show the tooltip of the chart further down, scrolled to the top of the view; whether the first chart is out
        of view then."""
        plot.evaluate("(canvas) => canvas.scrollIntoView({ block: 'start' })")
        away = soon(f"!app.charts.get({keys[0]!r}).inView && app.charts.get({keys[1]!r}).inView")
        b = plot.bounding_box()
        assert b is not None
        page.mouse.move(b["x"] + b["width"] * 0.5, b["y"] + b["height"] * 0.5)
        page.mouse.move(b["x"] + b["width"] * 0.55, b["y"] + b["height"] * 0.45)
        page.wait_for_function(TIP_DRAWN, timeout=5000)
        return away

    def shows(state: str, colors: dict[str, str], prefers: bool) -> None:
        """Add the checks of the page in `state`, its tooltip shown: what the page knows (`THEME`), and how its tooltip
        and the page around it look on screen. The screenshot is kept under the state's name."""
        knows = page.evaluate(THEME)
        bg, x, y = page.evaluate(TIP_BACKGROUND)
        shot = page.screenshot()
        (out / f"theme_{state.replace(' ', '_')}.png").write_bytes(shot)
        tip, corner = page.evaluate(PNG_PIXELS, [base64.b64encode(shot).decode(), [[x, y], [2, 2]]])
        checks[f"{state}: the canvases draw with the page's {named(colors)} colors"] = knows["draws"] == {k: colors[k] for k in knows["draws"]}
        checks[f"{state}: the page is told the browser prefers {'dark' if prefers else 'light'}, and its elements have those colors"] = (
            knows["prefers"] == prefers and knows["own"] == (dark if prefers else light)["bg"])
        checks[f"{state}: the tooltip has that background, on its canvas and on screen"] = bg == tip == rgb(colors["bg"])
        checks[f"{state}: the page around it is {named(colors)} on screen"] = max(corner) < 64 if colors is dark else min(corner) > 192

    try:
        page.goto(f"{url}/?theme#path=sweep")
        page.wait_for_function(READY, timeout=30000)
        checks["light: the charts are drawn in the light colors"] = drawn(light)
        checks["the first chart lies out of view while the other is hovered"] = hover()
        shows("light", light, False)
        force(True)
        checks["darkened by force: a tooltip shown then is drawn anew, dark"] = tip_on(dark)
        checks["darkened by force: the charts are drawn anew in the dark colors, in view or not"] = drawn(dark)
        shows("darkened by force", dark, False)
        force(False)
        checks["no longer darkened: the tooltip and the charts are light again"] = tip_on(light) and drawn(light)
        page.emulate_media(color_scheme="dark")
        checks["dark preferred: the tooltip and the charts are drawn anew, dark"] = tip_on(dark) and drawn(dark)
        shows("dark preferred", dark, True)
        page.emulate_media(color_scheme="light")
        checks["light preferred: the tooltip and the charts are light again"] = tip_on(light) and drawn(light)
        force(True)
        page.goto(f"{url}/?darkened#path=sweep")
        page.wait_for_function(READY, timeout=30000)
        checks["loaded darkened by force: the charts are drawn in the dark colors"] = drawn(dark)
        hover()
        shows("loaded darkened by force", dark, False)
        force(False)
        checks["loaded darkened by force, then no longer: the tooltip and the charts are light"] = tip_on(light) and drawn(light)
    finally:
        cdp.send("Emulation.setAutoDarkModeOverride", {})
        page.emulate_media(color_scheme="light")
        page.mouse.move(5, 5)
    failed = [name for name, passed in checks.items() if not passed]
    print(f"theme: {len(checks) - len(failed)}/{len(checks)} as intended" + (f"; not: {failed}" if failed else ""))
    return not failed


# Leaves a page whose address ends in ?nowebgl without WebGL2, as a browser that lacks it does: no canvas gives a
# context of it.
NO_WEBGL = """if (location.search === "?nowebgl") { const get = HTMLCanvasElement.prototype.getContext;
  HTMLCanvasElement.prototype.getContext = function (kind, ...more) { return kind === "webgl2" ? null : get.call(this, kind, ...more); }; }"""

# Per chart: its metric, what it says while it shows nothing, whether anything is drawn on its canvas, whether it is in view.
CHART_NOTICES = """[...app.charts.values()].map((c) => { const d = c.canvas.width ? c.canvas.getContext("2d").getImageData(0, 0, c.canvas.width, c.canvas.height).data : [];
  let drawn = false;
  for (let i = 3; i < d.length && !drawn; i += 4) drawn = d[i] > 0;
  return [c.key, c.emptyText(), drawn, c.inView]; })"""


def no_webgl_smoke(page: Page, url: str) -> bool:
    """Whether a browser without WebGL2 gets the page without its charts: every chart says that charts need WebGL2,
    those below the view too, a notice beside the status says so and stays while the status changes, the runs are
    listed and filtered as ever, the media panels are there, and nothing throws."""
    thrown: list[str] = []
    checks: dict[str, object] = {}

    def on_error(e: Error) -> None:
        thrown.append(str(e))

    def soon(js: str) -> bool:
        try:
            page.wait_for_function(js, timeout=5000)
            return True
        except Exception:
            return False

    page.add_init_script(NO_WEBGL)
    page.on("pageerror", on_error)
    try:
        page.goto(f"{url}/?nowebgl#path=sweep")
        page.wait_for_function(f"window.app && app.data.runs.size > 0 && app.charts.size > 0 && !app.data.queue.length && !app.data.posts && {LISTED}", timeout=30000)
        checks["the page has no WebGL2 (what the check is of)"] = page.evaluate("!document.createElement('canvas').getContext('webgl2')")
        checks["every chart says that charts need WebGL2, drawn on its canvas"] = soon(
            f"(() => {{ const cs = {CHART_NOTICES}; return cs.length > 0 && cs.every((c) => c[1] === 'charts need WebGL2' && c[2]); }})()")
        notices = page.evaluate(CHART_NOTICES)
        checks["some of them lie below the view"] = any(not c[3] for c in notices)
        checks["a notice beside the status says so, and stays as the status changes"] = soon(
            "(() => { const n = document.querySelector('#needsGl'); return !n.hidden && n.textContent.includes('WebGL2')"
            " && /blocks fetched/.test(document.querySelector('#status').textContent); })()")
        checks["the runs are listed"] = page.evaluate("app.runList.length") == 18
        checks["the media panels are there"] = page.locator("#panels .panel.media").count() == 3
        page.fill("#runFilter", "lr = 0.01")
        checks["a filter narrows them"] = soon("app.runList.filter((r) => r.shown).length === 6")
        page.fill("#runFilter", "")
        checks["cleared, it shows them all"] = soon("app.runList.filter((r) => r.shown).length === 18")
        page.keyboard.press("Escape")
        page.wait_for_timeout(300)
        checks["nothing throws"] = not thrown
    finally:
        page.remove_listener("pageerror", on_error)
    failed = [name for name, passed in checks.items() if not passed]
    print(f"no WebGL2: {len(checks) - len(failed)}/{len(checks)} as intended; charts {[c[0] for c in notices]}"
          + (f"; not: {failed}; thrown: {thrown}" if failed else ""))
    return not failed


# Two runs of 100 steps and one that has logged a single row, at step 0; each logs `once` at step 0 alone.
DOTS_WRITER = """
import math, sys, trex
for k, (name, n, lr) in enumerate((("long0", 100, 0.01), ("long1", 100, 0.01), ("new", 1, 0.02))):
    r = trex.init(f"{sys.argv[1]}/dots/{name}", config={"lr": lr})
    r.log({"once": k + 1.0}, step=0)
    for s in range(n):
        r.log({"loss": 3.0 if n == 1 else 2 * math.exp(-s / 30) + 0.1 * k}, step=s)
    r.finish()
"""

# Of chart `key`'s canvas, the pixels of color `css` (#rrggbb, to within 2 a channel): how many lie within a dot's
# radius of data point (x, y), how many there are in all, and how far past the plot's left side the leftmost one lies
# (device px).
DOT_PIXELS = """([key, x, y, css]) => { const c = app.charts.get(key), cv = c.canvas, dpr = devicePixelRatio, d = cv.getContext("2d").getImageData(0, 0, cv.width, cv.height).data;
  const [r, g, b] = css.match(/[0-9a-f]{2}/gi).map((h) => parseInt(h, 16)), cx = c.px(x) * dpr, cy = c.py(y) * dpr;
  let near = 0, all = 0, left = cv.width;
  for (let j = 0; j < cv.height; j++) for (let i = 0; i < cv.width; i++) { const o = 4 * (j * cv.width + i);
    if (Math.abs(d[o] - r) > 2 || Math.abs(d[o + 1] - g) > 2 || Math.abs(d[o + 2] - b) > 2 || d[o + 3] < 250) continue;
    all++;
    left = Math.min(left, i);
    if (Math.hypot(i + 0.5 - cx, j + 0.5 - cy) <= 3.5 * dpr) near++; }
  return [near, all, Math.round(c.px(c.view.x0) * dpr) - left]; }"""

# How many colored pixels (as LINE_PIXELS counts them) of chart `key`'s canvas lie outside its plot: in its margins.
MARGIN_PIXELS = """(key) => { const c = app.charts.get(key), cv = c.canvas, v = c.view, dpr = devicePixelRatio, d = cv.getContext("2d").getImageData(0, 0, cv.width, cv.height).data;
  const x0 = Math.round(c.px(v.x0) * dpr), x1 = Math.round(c.px(v.x1) * dpr), y0 = Math.round(c.py(v.y1) * dpr), y1 = Math.round(c.py(v.y0) * dpr);
  let n = 0;
  for (let j = 0; j < cv.height; j++) for (let i = 0; i < cv.width; i++) { const o = 4 * (j * cv.width + i);
    if ((i < x0 || i >= x1 || j < y0 || j >= y1) && d[o + 3] && Math.max(d[o], d[o + 1], d[o + 2]) - Math.min(d[o], d[o + 1], d[o + 2]) > 60) n++; }
  return n; }"""


def dots_smoke(page: Page, url: str, runs: Path) -> bool:
    """Whether a point no segment of its line reaches is drawn as a dot, where a line of one point would show nothing: a
    run that has logged one row among longer ones (whole, though it lies on the plot's left side, where lines that go
    on past a side end), a metric logged at one step of each run, a group of one such run, and the same in a heatmap;
    and whether a run of a single row loads without a failing block (the levels finer than its charts' are not asked
    for where there are none)."""
    subprocess.run([sys.executable, "-c", DOTS_WRITER, str(runs)], check=True)
    checks: dict[str, object] = {}

    def dot(found: list[int]) -> bool:
        """Whether `found` (of DOT_PIXELS) is a dot's pixels and no other of its color."""
        return found[0] >= 12 and found[1] == found[0]

    def show(hash_: str, n: int) -> None:
        page.goto(f"{url}/?dots{n}#path={hash_}")  # a page of its own for the run opened alone: its runs are that run
        page.wait_for_function(f"window.app && app.data.runs.size === {n}", timeout=60000)
        page.wait_for_function(SETTLED + " && [...app.charts.values()].every((c) => c.view)", timeout=60000)
        page.wait_for_timeout(300)

    def color(name: str) -> str:
        return page.evaluate(f"app.data.runs.get('dots/{name}').color")

    def found(key: str, x: float, y: float, css: str) -> list[int]:
        return page.evaluate(DOT_PIXELS, [key, x, y, css])

    show("dots&group=", 3)
    new = found("loss", 0, 3.0, color("new"))
    checks["a run of one row is a dot among the others' lines"] = dot(new)
    checks["on the plot's left side, it is drawn whole"] = new[2] >= 2 and page.evaluate(MARGIN_PIXELS, "loss") >= 6
    page.evaluate("app.setXRange([20, 60, 0])")
    page.wait_for_function(SETTLED + " && app.charts.get('loss').view?.x0 === 20", timeout=60000)
    page.wait_for_timeout(300)
    crossing = [page.evaluate(LINE_PIXELS, "loss"), page.evaluate(MARGIN_PIXELS, "loss")]
    checks["lines that go on past the plot's sides end there"] = crossing[0] > 300 and crossing[1] == 0
    # the run of one row 2 px left of the plot, on a y axis that holds its value
    page.evaluate("(() => { app.panelCfg.loss = { ymin: 0, ymax: 4 }; app.savePanelCfg('loss'); app.setXRange([2 * 99 / app.charts.get('loss').pw, 99, 0]); })()")
    page.wait_for_function(SETTLED + " && app.charts.get('loss').view?.x0 > 0 && app.charts.get('loss').view.y1 === 4", timeout=60000)
    page.wait_for_timeout(300)
    past = found("loss", 0, 3.0, color("new"))
    checks["a point just past the plot's side shows no dot"] = past[1] == 0
    page.evaluate("(() => { delete app.panelCfg.loss; app.savePanelCfg('loss'); })()")
    page.evaluate("app.setXRange(null)")
    page.wait_for_function(SETTLED + " && [...app.charts.values()].every((c) => c.view)", timeout=60000)
    page.wait_for_timeout(300)
    once = [found("once", 0, k + 1.0, color(name)) for k, name in enumerate(("long0", "long1", "new"))]
    checks["a metric logged at one step is a dot a run"] = all(dot(f) for f in once)
    page.evaluate("(() => { app.panelCfg.once = { render: 'density' }; app.savePanelCfg('once'); })()")
    page.wait_for_function(SETTLED + " && app.charts.get('once').view?.density && app.charts.get('once').view.gpu", timeout=60000)
    page.wait_for_timeout(300)
    heat = page.evaluate(LINE_PIXELS, "once")
    checks["as a heatmap, its points are counted where they are"] = heat >= 3 * 12
    page.evaluate("(() => { delete app.panelCfg.once; app.savePanelCfg('once'); })()")
    show("dots&group=lr", 3)
    group = page.evaluate("app.charts.get('loss').view.lines.map((l) => [l.label, l.color]).find(([label]) => label.startsWith('0.02'))")
    lone = found("loss", 0, 3.0, group[1]) if group else [0, 0, 0]
    checks["a group of one such run is a dot"] = page.evaluate("!!app.charts.get('loss').view.gpu") and 12 <= lone[1] <= 60
    show("dots/new", 1)
    page.wait_for_timeout(2500)  # what is fetched ahead has been asked for
    alone = [found(key, 0, y, color("new")) for key, y in (("loss", 3.0), ("once", 3.0))]
    checks["a run of one row shows a dot a chart"] = all(dot(f) for f in alone)
    failing = page.evaluate("[app.data.failed.size, document.querySelector('#status').textContent]")
    checks["and no block of it fails"] = failing[0] == 0 and "failing" not in failing[1]
    failed = [name for name, passed in checks.items() if not passed]
    print(f"dots: {len(checks) - len(failed)}/{len(checks)} as intended; a run of one row {new}, lines zoomed into {crossing}, past the side {past}, a metric logged once {once}, "
          f"as a heatmap {heat} px, a group {lone}, a run alone {alone}, {failing}" + (f"; not: {failed}" if failed else ""))
    return not failed


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp(prefix="trex-shots-"))
    out.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="trex-smoke-"))
    runs = tmp / "runs"
    subprocess.run([sys.executable, str(REPO / "examples/demo.py"), str(runs / "sweep"), "--seeds", "3", "--steps", "1500"], check=True)
    subprocess.run([sys.executable, "-c", NESTED_WRITER, str(runs)], check=True)
    port = free_port()
    env = {**os.environ, "TREX_DAEMON_DIR": str(tmp / "daemon"), "TREX_REFRESH": "1"}
    server = subprocess.Popen([sys.executable, "-m", "trex", "serve", str(runs), "--temporary", "--port", str(port),
                               "--cache", str(tmp / "cache")], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    url = f"http://127.0.0.1:{port}"
    ok = True
    failed: list[str] = []

    def check(name: str, passed: object) -> bool:
        if not passed:
            failed.append(name)
        return bool(passed)

    try:
        first_line(server)
        with sync_playwright() as p:
            exe = next(Path.home().glob(".cache/ms-playwright/chromium-*/chrome-linux64/chrome"), None)
            browser = p.chromium.launch(executable_path=str(exe) if exe else None)
            page = browser.new_page(viewport={"width": 1500, "height": 950})
            coverage = JsCoverage(page)
            coverage.watch(page)
            errors: list[str] = []

            def console(m: ConsoleMessage) -> None:
                if m.type == "error":
                    errors.append(f"{m.type}: {m.text}")

            page.on("console", console)
            page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
            ready = READY
            for label, h in [("cold", "#path=sweep"), ("warm", "#path=sweep"), ("grouped", "#path=sweep&group=lr")]:
                page.goto(f"{url}/?{label}{h}")
                page.wait_for_function(ready, timeout=30000)
                page.wait_for_timeout(500)
                print(f"{label}: {page.inner_text('#status')}")
                page.screenshot(path=str(out / f"{label}.png"))

            ok &= check("listing_order_smoke", listing_order_smoke(page, url))
            ok &= check("group_levels_smoke", group_levels_smoke(page, url))
            ok &= check("sections_smoke", sections_smoke(page, url))
            ok &= check("hidden_runs_smoke", hidden_runs_smoke(page, url))
            ok &= check("grouping_modes_smoke", grouping_modes_smoke(page, url))
            ok &= check("hidden_panels_smoke", hidden_panels_smoke(page, url))
            ok &= check("filter_smoke", filter_smoke(page, url))
            ok &= check("interactions_smoke", interactions_smoke(page, url))
            ok &= check("theme_smoke", theme_smoke(page, url, out))
            ok &= check("no_webgl_smoke", no_webgl_smoke(page, url))
            ok &= check("dots_smoke", dots_smoke(page, url, runs))
            ok &= check("flicker_smoke", flicker_smoke(page, url, runs))
            ok &= check("kept_runs_smoke", kept_runs_smoke(page, url, runs))
            ok &= check("binned_smoke", binned_smoke(page, url, runs))
            ok &= check("hidden_live_smoke", hidden_live_smoke(page, url, runs))
            ok &= check("column_drag_smoke", column_drag_smoke(page, url))
            ok &= check("side_mark_smoke", side_mark_smoke(page, url))
            ok &= check("hidden_extent_smoke", hidden_extent_smoke(page, url, runs))
            ok &= check("line_pixels_smoke", line_pixels_smoke(page, url))
            ok &= check("failing_blocks_smoke", failing_blocks_smoke(page, url))
            writer = subprocess.Popen([sys.executable, "-c", LIVE_WRITER, str(runs)])
            deadline = time.time() + 20
            while not (runs / "live").exists() and time.time() < deadline:
                time.sleep(0.1)
            time.sleep(1.5)
            page.goto(f"{url}/?live#path=live")
            page.wait_for_function("window.app && app.data.runs.size === 3", timeout=30000)
            page.evaluate("""() => {
                const d = app.data, orig = d.onRows.bind(d);
                let n = 0;
                d.dropped = 0;
                d.onRows = (r, ev) => (++n % 5 === 0 ? d.dropped++ : orig(r, ev));
            }""")
            writer.wait()
            page.wait_for_timeout(4000)
            page.wait_for_function(ready, timeout=30000)
            page.screenshot(path=str(out / "live.png"))
            client = page.evaluate("""() => {
                const out = {};
                for (const r of app.data.runs.values()) {
                    const cols = {};
                    for (const [k, c] of r.cols) {
                        let n = 0;
                        for (let i = 0; i < c.n; i++) n += c.w ? c.w[i] : 1;
                        cols[k] = { n, seqs: (app.data.partsOf(r, k) || []).map((p) => p.v.seq[p.row]) };
                    }
                    out[r.id] = { seq: r.seq, media: r.mseq, cols, tail: r.tail.length };
                }
                return { runs: out, dropped: app.data.dropped };
            }""")
            for rid, c in sorted(client["runs"].items()):
                n, finite, mseq = file_columns(runs / rid)
                counts = {k: col["n"] for k, col in c["cols"].items()}
                match = (c["seq"] == n and c["media"] == mseq and c["tail"] == 0 and counts
                         and all(col["seqs"] and set(col["seqs"]) == {n} for col in c["cols"].values())
                         and all(counts[k] == finite.get(k, 0) for k in counts))
                ok &= check(f"live {rid}", match)
                print(f"{rid}: client {c['seq']} rows / {c['media']} media / column counts {counts}, "
                      f"file {n} / {mseq} / finite {finite}: {'match' if match else 'MISMATCH'}")
            print(f"dropped {client['dropped']} rows events; client {'converged' if ok else 'DIVERGED'}")
            ok &= check("link_smoke", link_smoke(page, tmp, env, out, url))
            server.terminate()
            ok &= check("node_smoke", node_smoke(page, runs, tmp, env, out, errors))
            coverage.take()
            coverage.add_node_tests()
            lines = coverage.report()
            covered, total = sum(c for c, _, _ in lines.values()), sum(n for _, n, _ in lines.values())
            (out / "ui_coverage.txt").write_text("".join(f"{k}: uncovered lines {miss}\n" for k, (_, _, miss) in lines.items()))
            print("UI line coverage: " + ", ".join(f"{k} {c / n:.0%}" for k, (c, n, _) in lines.items())
                  + f"; all {covered / total:.1%} (floor {UI_COVERAGE:.0%}; uncovered lines in ui_coverage.txt)")
            ok &= covered / total >= UI_COVERAGE
            print("\n".join(errors) or "no console errors")
            ok &= not errors
            browser.close()
    finally:
        server.terminate()
        server.wait()
    if failed:
        print(f"failed: {', '.join(failed)}")
    if ok:
        shutil.rmtree(tmp)
    print(f"screenshots in {out}" + ("" if ok else f"; scratch data kept in {tmp}"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
