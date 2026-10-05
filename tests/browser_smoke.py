"""Headless-browser smoke test against a throwaway trex server on a temporary runs directory.

Checks cold and warm loads, grouping, opening groups as path levels, nested chart sections and pinning, panels of
hidden runs, the x range of hidden runs, the filter box, console errors, UI line coverage (at least UI_COVERAGE of the modules' code lines run),
and that a client dropping every 5th stream event still converges to the run files: every row and media item, and columns whose
points' counts add up to each metric's finite values. Then, under a throwaway `trex daemon`: adding
a directory with `trex serve -y`, the root view of both, making a workspace of them in the panel the trex
brand opens, and from that panel removing a directory and re-adding it from the remembered ones.

    uv run --with playwright python tests/browser_smoke.py [screenshot_dir]
"""

import functools
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import json
import math
import statistics
import os
import urllib.request

import numpy as np
from playwright.sync_api import sync_playwright

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
    time.sleep(0.04)
for r in runs:
    r.finish()
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
  C.prototype.draw = function () {
    draw.call(this);
    const v = this.view;
    if (!v || !this.w) return;
    const lines = v.lines.map((ln) => {
      const c = ln.cols?.[0], n = ln.xy ? ln.xy.length / 2 : c ? c.n : 0;
      return [ln.run?.id ?? ln.group ?? ln.label, n, ln.xy ? ln.xy[ln.xy.length - 2] : c && c.n ? c.s[c.n - 1] : null];
    });
    draws.push({ key: this.key, paced: app.paced, y0: v.y0, y1: v.y1, lines });
  };
}"""


def flicker_faults(draws):
    """What a viewer would see flicker between consecutive draws of a chart: a line with fewer points or its end
    further back, a line gone for a draw, and a streamed redraw shrinking the y axis."""
    faults, by = [], {}
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


def flicker_smoke(page, url, runs):
    """Whether charts drawing runs that stream, line by line and grouped, never show a line shrink, its end move back
    or vanish for a draw, nor a streamed redraw shrink the y axis."""
    writer = subprocess.Popen([sys.executable, "-c", FLICKER_WRITER, str(runs)], stdout=subprocess.PIPE, text=True)
    try:
        writer.stdout.readline()
        page.goto(f"{url}/?flicker#path=flicker&group=run")
        page.wait_for_function("window.app && app.data.runs.size === 6 && [...app.charts.values()].some((c) => c.view)", timeout=30000)
        page.evaluate(RECORD_DRAWS)
        page.wait_for_timeout(3500)
        lines = page.evaluate("draws")
        page.evaluate("draws = []; app.setGroup('run~1')")
        page.wait_for_timeout(3500)
        groups = page.evaluate("draws")
    finally:
        writer.wait()
    faults = flicker_faults(lines) + flicker_faults(groups)
    print(f"flicker: {len(lines)} draws of lines, {len(groups)} of groups while 6 runs streamed; "
          + (f"faults {faults[:5]}" if faults else "no line shrank, moved back or vanished, no streamed redraw shrank an axis"))
    return len(lines) > 4 and len(groups) > 4 and not faults


def many_value(group, i, step):
    """What MANY_WRITER logs."""
    return math.exp(-step / 50) + (i % 7) * 0.01 + (0.1 if group == "b" else 0) + 0.02 * math.sin(step * (i + 1))


def group_medians(group, g0, dx, bins):
    """Per bin of a grid, the median over a MANY_WRITER group's runs of each run's mean of its rows in the bin."""
    out = []
    for k in range(bins):
        steps = [s for s in range(200) if g0 + k * dx <= s < g0 + (k + 1) * dx]
        out.append(statistics.median(statistics.fmean(many_value(group, i, s) for s in steps) for i in range(160)) if steps else None)
    return out


def hidden_extent_smoke(page, url, runs):
    """Whether hiding a run takes its steps out of a chart's x range, and showing it again puts them back."""
    subprocess.run([sys.executable, "-c", EXTENT_WRITER, str(runs)], check=True)
    page.goto(f"{url}/?extent#path=extent&group=run")
    page.wait_for_function("window.app && app.data.runs.size === 2", timeout=60000)
    page.wait_for_function(SETTLED + " && app.charts.get('loss')?.view", timeout=60000)
    end = "(() => { const v = app.charts.get('loss').view; return [v.lines.length, Math.round(v.ex1)]; })()"

    def toggle_long():
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


def line_pixels_smoke(page, url):
    """Whether charts put their lines on their canvases: grouped, one per run in a folder, and grouped again after
    going back, drawn as at first whatever the folder's lines drew before."""
    chart = "app.charts.get('train/loss')"

    def drawn():
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


def binned_smoke(page, url, runs):
    """Whether a zoom of more runs than a chart draws one by one draws them from bins of their buckets, as group
    statistics (each group's median per bin of its runs' means of their rows) from the chart's worker, and as a
    heatmap; and whether a server of another protocol is stated."""
    subprocess.run([sys.executable, "-c", MANY_WRITER, str(runs)], check=True)

    def zoomed(query, hash_):
        page.goto(f"{url}/?binned&{query}#path=many&{hash_}")
        page.wait_for_function("window.app && app.data.runs.size === 320", timeout=60000)
        page.wait_for_function(SETTLED, timeout=60000)
        page.evaluate("app.setXRange([60, 140, 0])")
        page.wait_for_timeout(300)
        page.wait_for_function(SETTLED, timeout=60000)

    zoomed("", "group=run~1")
    binned, lines, worker = page.evaluate("""(() => { const c = app.charts.get('loss');
        return [c.binned, c.view.lines.map((l) => [l.label.split(' ')[0], l.g0, l.dx, [...l.center]]), !!c.stats]; })()""")
    zoomed("", "group=run")
    heat = page.evaluate("(() => { const c = app.charts.get('loss'); return [c.binned, c.view.density, c.view.lines.length]; })()")
    page.evaluate("app.showProtocol(1)")
    stated = page.evaluate("[!document.querySelector('#mismatch').hidden, document.querySelector('#mismatch').title]")
    off = [abs(c - w) for g, g0, dx, center in lines for c, w in zip(center, group_medians(g, g0, dx, len(center))) if w is not None]
    groups = sorted(l[0] for l in lines)
    print(f"binned: a zoom of 320 runs drew bins of their buckets {binned} on its worker {worker}, groups {groups} of "
          f"{len(lines[0][3]) if lines else 0} bins, at most {max(off, default=1):.2g} off their exact medians; heatmap {heat}; "
          f"protocol 1 stated {stated}")
    return (binned is True and worker and groups == ["a", "b"] and off and max(off) < 1e-5 and heat == [True, True, 320]
            and stated[0] and "the server 1" in stated[1])


class JsCoverage:
    """Which lines of the UI's modules ran, from Chromium's V8 coverage and the node tests'. A page load discards the previous page's
    counts, so `take` runs before every navigation (`watch` makes the page do so) and takes are merged per file."""

    def __init__(self, page):
        self.cdp = page.context.new_cdp_session(page)
        self.cdp.send("Profiler.enable")
        self.cdp.send("Profiler.startPreciseCoverage", {"callCount": True, "detailed": True})
        self.ran = {}  # module -> per UTF-16 unit: whether it ran

    def watch(self, page):
        for name in ("goto", "go_back", "go_forward", "reload", "click"):
            step = getattr(page, name)
            setattr(page, name, lambda *a, step=step, **kw: (self.take(), step(*a, **kw))[1])

    def take(self):
        for script in self.cdp.send("Profiler.takePreciseCoverage")["result"]:
            self.merge(script)

    def add_node_tests(self):
        """Merge in the node tests' coverage of the same modules (V8's, through NODE_V8_COVERAGE)."""
        with tempfile.TemporaryDirectory() as d:
            subprocess.run(["node", "--test", *map(str, sorted((REPO / "tests").glob("*.test.mjs")))], cwd=REPO, check=True,
                           capture_output=True, env={**os.environ, "NODE_V8_COVERAGE": d})
            for f in Path(d).glob("*.json"):
                for script in json.loads(f.read_text())["result"]:
                    self.merge(script)

    def merge(self, script):
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

    def report(self):
        """{module: (covered lines, lines with code, uncovered line numbers)}; a line is covered when any
        non-blank character on it ran."""
        out = {}
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
def units(name):
    """(line, is blank) per UTF-16 code unit of a UI module, the unit V8 counts offsets in."""
    out = []
    for i, text in enumerate((STATIC / name).read_text().split("\n")):
        for ch in text + "\n":
            out.extend([(i, ch.isspace())] * (2 if ord(ch) > 0xFFFF else 1))
    return np.array(out[:-1] if out else [(0, True)], np.int64)


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def file_columns(run_dir):
    """(rows, {key: finite values}, media items) of a run file."""
    c = connect_ro(run_dir)
    rows = chunks.rows(c)
    mseq = c.execute("SELECT count(*) FROM media").fetchone()[0]
    c.close()
    finite = Counter(k for _, _, _, d in rows for k, v in d.items() if math.isfinite(v))
    return len(rows), dict(finite), mseq


READY = ("window.app && app.data.runs.size > 0 && app.charts.size > 0 && !app.data.queue.length && !app.data.posts"
         " && [...app.charts.values()].some((c) => c.view)")
SETTLED = READY + " && !app.data.busy && !app.round && !app.raf && !app.planTimer"


def group_levels_smoke(page, url):
    """Whether opening a group narrows the view to its runs, drawn as lines, a group nested inside it opens a deeper
    level, and the path bar and back button return to each level, the group-by staying as set."""
    state = """() => [app.opts.focus.length, app.opts.group, app.grouped, app.runList.filter((r) => r.shown).length,
        [...document.querySelectorAll('#crumbPath .crumb')].map((e) => e.textContent)]"""
    def ends(text):
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


def interactions_smoke(page, url):
    """Whether charts carry no legend and the chart controls do what they say: hover and Shift-pinned tooltips
    (which the wheel scrolls, and whose rows reveal and open runs), x and box zooms and their reset, the chart
    settings (smoothing, axes, outliers, density, reset), the sort box (type, pick, Enter, Escape), sidebar hiding,
    group-by search, keyboard scrolling, the media slider, and back/forward (back to the grouping the folder had)."""
    key, checks = "train/loss", {}
    sel = f".panel:has(.pname:text-is('{key}'))"
    chart = f"app.charts.get({key!r})"

    def plot():
        page.locator(f"{sel} canvas").nth(1).scroll_into_view_if_needed()
        b = page.locator(f"{sel} canvas").nth(1).bounding_box()
        return lambda fx, fy: (b["x"] + b["width"] * fx, b["y"] + b["height"] * fy)

    def soon(js, timeout=3000):
        """Whether `js` becomes true within `timeout` ms."""
        try:
            page.wait_for_function(js, timeout=timeout)
            return True
        except Exception:
            return False

    def hover(at):
        page.mouse.move(*at(0.5, 0.5))
        page.mouse.move(*at(0.55, 0.45))
        page.wait_for_function("!document.querySelector('#tip').hidden", timeout=5000)

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
    row = lambda label: f"#menu .srow:has(> label:text-is('{label}'))"
    page.check(f"{row('smoothing')} input[type=checkbox]")
    page.locator(f"{row('smoothing')} input[type=range]").fill("0.9")
    page.select_option(f"{row('y scale')} select", "true")
    page.select_option(f"{row('x axis')} select >> nth=0", '"runtime"')
    page.select_option(f"{row('ignore outliers')} select", "0.01")
    page.fill(f"{row('x range')} input >> nth=0", "0")
    page.dispatch_event(f"{row('x range')} input >> nth=0", "change")
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
    checks["a density chart lists the nearest runs"] = "nearest" in page.inner_text("#tip")
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
    checks["the tooltip opens at the line nearest the cursor"] = page.evaluate("!!document.querySelector('#tip .trow.near')")
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
    page.wait_for_function("app.scopeIsRun && app.data.runs.size === 1 && !!document.querySelector('#infoPanel .st')", timeout=10000)
    checks["a pinned row opens its run"] = True
    page.go_back()
    page.wait_for_function(READY + " && !app.scopeIsRun", timeout=30000)
    checks["back from a run restores the folder ungrouped, as it was"] = page.evaluate("!app.opts.group.length")
    page.go_forward()
    page.wait_for_function("app.scopeIsRun", timeout=30000)

    width = "document.querySelector('aside').offsetWidth"
    w0, g = page.evaluate(width), page.locator("#sideGrip").bounding_box()
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
    page.goto(f"{url}/?iqm#path=sweep&group=lr&center=iqm")
    page.wait_for_function(READY, timeout=30000)
    plot()
    checks["IQM draws each group's interquartile mean with a CI band"] = soon(
        f"(() => {{ const v = {chart}.view, k = v?.lines[0]; return document.querySelector('#center').value === 'iqm' && !!k"
        " && k.center.some((c, i) => c > k.lo[i] && c < k.hi[i]); })()", timeout=10000)
    span = page.evaluate(f"""(() => {{ const v = {chart}.view, c = v.lines.flatMap((l) => [...l.center].filter(Number.isFinite));
        return [Math.min(...c), Math.max(...c), v.y0, v.y1]; }})()""")
    reach = (span[1] - span[0]) * 0.25
    checks["the y axis follows the group lines; a band widens it by at most a quarter"] = (
        span[2] <= span[0] and span[3] >= span[1] and span[2] >= span[0] - reach - 0.05 * (span[1] - span[0] + 2 * reach)
        and span[3] <= span[1] + reach + 0.05 * (span[1] - span[0] + 2 * reach))
    failed = [k for k, v in checks.items() if not v]
    print(f"interactions: {len(checks) - len(failed)}/{len(checks)} as intended" + (f"; not: {failed}" if failed else ""))
    return not failed


def dir_pages(url):
    """{name: page path} of every directory the trex at `url` serves."""
    with urllib.request.urlopen(f"{url}/api/node") as r:
        return {x["name"]: x["url"] for x in json.loads(r.read())["dirs"]}


def daemon_smoke(page, runs, tmp, env, out, errors):
    """Whether the daemon's root shows every tracked directory (a folder in it as `/ name`), a workspace made in the panel (opened from the trex
    brand) merges them and opens though its URL was visited before it existed, and the panel removes, adds and
    forgets directories. The refused add's 400 is taken out
    of `errors`."""
    port = free_port()
    url = f"http://127.0.0.1:{port}"
    daemon = subprocess.Popen([sys.executable, "-m", "trex", "serve", str(runs / "sweep"), "--port", str(port),
                               "--cache", str(tmp / "cache")], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    row = "#menu .mrow:has(.ml:text-is('{}'))"
    try:
        daemon.stdout.readline()
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
        page.screenshot(path=str(out / "daemon_menu.png"))
        recent_left = page.query_selector("#menu .mrecent") is not None
        panel_text = page.inner_text("#menu")
        tidy = "null" not in panel_text.split() and "unknown" not in panel_text
        with urllib.request.urlopen(f"{url}/api/node") as r:
            d = json.loads(r.read())
        left, workspaces = [x["name"] for x in d["dirs"]], [(w["name"], w["members"]) for w in d["workspaces"]]
        print(f"daemon: root shows {root[0]}/{total} runs in {root[2]} as {root[1]!r}, a folder in it as {crumbs}; workspace of both shows "
              f"{merged[0]}/{total}, dir field {merged[1]}; removed live, re-added it from history; tracking {left}, "
              f"workspaces {workspaces}, history {d['history']}, panel tidy {tidy}")
        return (root == [total, "/", ["live", "sweep"]] and crumbs == ["/", "sweep"] and merged == [total, True] and sorted(left) == ["live", "sweep"]
                and workspaces == [("both", ["sweep"])] and not d["history"] and not recent_left and tidy)
    finally:
        daemon.terminate()
        daemon.wait()

def link_smoke(page, tmp, env, out, upstream):
    """Whether a daemon tracking nothing, given another trex's http://host:port in its panel, pulls and shows every run
    that trex holds, lists the link under "pulled from" with no way to remove the pulled directory alone, and lets go
    of that directory with the link."""
    port = free_port()
    url = f"http://127.0.0.1:{port}"
    daemon = subprocess.Popen([sys.executable, "-m", "trex", "serve", "--port", str(port), "--cache", str(tmp / "cache-links"),
                               "--name", "laptop"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                              env={**env, "TREX_DAEMON_DIR": str(tmp / "daemon-links")})
    try:
        daemon.stdout.readline()
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
        return want > 0 and len(rows) == 2 and rows[0] == ["runs", False] and rows[1][1] and d["dirs"] == [] == d["links"]
    finally:
        daemon.terminate()
        daemon.wait()


NESTED_WRITER = """
import sys, numpy as np, trex
r = trex.init(f"{sys.argv[1]}/nested/r0")
for step in range(50):
    r.log({"eval/return/mean": step, "eval/return/std": 1.0, "eval/len": 10 + step, "loss": 1 / (step + 1)}, step=step)
r.log_image("eval/video/frame", np.zeros((8, 8, 3), np.uint8), step=49)
r.finish()
"""


def filter_smoke(page, url):
    """Whether the filter box keeps the runs a WHERE clause or a name search selects, marks a clause that does not
    parse while filtering nothing, and completes fields, a field's values (with counts) and and / or from the keyboard."""
    page.goto(f"{url}/?filter#path=sweep")
    page.wait_for_function(READY, timeout=30000)
    shown = "app.runList.filter((r) => r.shown).map((r) => r.id).sort()"
    every = page.evaluate(shown)
    want = page.evaluate("app.runList.filter((r) => r.meta.config.lr === 0.001 && r.meta.config.seed === 1).map((r) => r.id).sort()")
    results = {}
    for text in ["lr = 0.001 and seed = 1", "lr = 0.00", "seed1", "lr ="]:
        page.fill("#runFilter", text)
        page.wait_for_timeout(600)
        results[text] = [page.evaluate(shown), page.evaluate("document.querySelector('#runFilter').classList.contains('bad')")]
    page.fill("#runFilter", "")
    items = lambda: page.evaluate("[...document.querySelectorAll('#menu .mitem .ml')].map((e) => e.textContent)")
    page.click("#runFilter")
    page.keyboard.type("l")
    field_items = items()
    page.keyboard.press("Tab")
    after_field = page.input_value("#runFilter")
    page.keyboard.type("= ")
    value_items = page.evaluate("[...document.querySelectorAll('#menu .mitem')].map((e) => e.textContent)")
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


def sections_smoke(page, url):
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


def hidden_panels_smoke(page, url):
    """Whether a hidden panel leaves its section for a link in the section's header that shows it again, the section
    menu checks shown panels and open subsections and toggles them, and hidden panels stay hidden on reload."""
    head = "#panels details.section:has(> summary .stitle:text-is('{}')) > summary"
    shown = "[...document.querySelectorAll('#panels .panel .ptitle > span:first-child')].map((e) => e.textContent).sort()"

    def links(title):
        return page.locator(f"{head.format(title)} .sublinks .hiddenlink").all_inner_texts()

    def menu_items():
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


def grouping_modes_smoke(page, url):
    """Whether runs group by their directory by default, one line per directory of several runs with a state dot;
    `run~2 / run~1` nests directories, each named within its parent, and opening one opens that directory; `run`
    lists the runs flat; and `visible = true` leaves out unchecked runs."""
    heads = "[...document.querySelectorAll('#runTable tr.grp .gname')].map((e) => e.firstChild.textContent)"
    runs = "document.querySelectorAll('#runTable tr:has(td.name a)').length"

    def group_by(text):
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
    page.wait_for_function("app.opts.path === 'sweep/width128' && app.data.runs.size === 9 && document.querySelectorAll('#runTable tr.grp').length === 3")
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


def hidden_runs_smoke(page, url):
    """Whether panels follow the shown runs: hiding the only run that logs some keys removes their panels, showing it
    brings them back, and hiding every run leaves no panels, as in a folder without runs."""
    panels = "[...document.querySelectorAll('#panels .panel > .ptitle > span:first-child')].map((e) => e.textContent).sort()"
    only = ["eval/len", "eval/return/mean", "eval/return/std", "eval/video/frame", "loss"]
    box = "#runTable tr:has(td.name[title='nested/r0']) input[type=checkbox]"

    def settle(want):
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


def main():
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
    ok, failed = True, []

    def check(name, passed):
        if not passed:
            failed.append(name)
        return bool(passed)

    try:
        server.stdout.readline()
        with sync_playwright() as p:
            exe = next(Path.home().glob(".cache/ms-playwright/chromium-*/chrome-linux64/chrome"), None)
            browser = p.chromium.launch(executable_path=str(exe) if exe else None)
            page = browser.new_page(viewport={"width": 1500, "height": 950})
            coverage = JsCoverage(page)
            coverage.watch(page)
            errors = []
            page.on("console", lambda m: m.type == "error" and errors.append(f"{m.type}: {m.text}"))
            page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
            ready = READY
            for label, h in [("cold", "#path=sweep"), ("warm", "#path=sweep"), ("grouped", "#path=sweep&group=lr")]:
                page.goto(f"{url}/?{label}{h}")
                page.wait_for_function(ready, timeout=30000)
                page.wait_for_timeout(500)
                print(f"{label}: {page.inner_text('#status')}")
                page.screenshot(path=str(out / f"{label}.png"))

            ok &= check("group_levels_smoke", group_levels_smoke(page, url))
            ok &= check("sections_smoke", sections_smoke(page, url))
            ok &= check("hidden_runs_smoke", hidden_runs_smoke(page, url))
            ok &= check("grouping_modes_smoke", grouping_modes_smoke(page, url))
            ok &= check("hidden_panels_smoke", hidden_panels_smoke(page, url))
            ok &= check("filter_smoke", filter_smoke(page, url))
            ok &= check("interactions_smoke", interactions_smoke(page, url))
            ok &= check("flicker_smoke", flicker_smoke(page, url, runs))
            ok &= check("binned_smoke", binned_smoke(page, url, runs))
            ok &= check("hidden_extent_smoke", hidden_extent_smoke(page, url, runs))
            ok &= check("line_pixels_smoke", line_pixels_smoke(page, url))
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
            ok &= check("daemon_smoke", daemon_smoke(page, runs, tmp, env, out, errors))
            coverage.take()
            coverage.add_node_tests()
            lines = coverage.report()
            covered, total = (sum(v[i] for v in lines.values()) for i in (0, 1))
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
