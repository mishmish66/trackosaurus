"""Headless-browser smoke test against a throwaway trex server on a temporary runs directory.

Checks cold and warm loads, grouping, opening groups as path levels, nested chart sections and pinning, the filter box, console errors,
UI line coverage (at least UI_COVERAGE of the modules' code lines run), and that a client dropping every 5th
stream event still converges to the run files: every row and media item, and top tiles whose
bucket counts add up to each metric's finite values. Then, under a throwaway `trex daemon`: adding
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


def group_levels_smoke(page, url):
    """Whether opening a group makes it a path level showing its runs ungrouped, a grouping inside it opens a
    deeper level, and the path bar and back button return to each level with the grouping it had."""
    state = """() => [app.opts.focus.length, app.opts.group.join(), app.grouped, app.runList.filter((r) => r.shown).length,
        [...document.querySelectorAll('#crumbPath .crumb')].map((e) => e.textContent)]"""
    def ends(text):
        page.wait_for_function(f"[...document.querySelectorAll('#crumbPath .crumb')].at(-1)?.textContent === {json.dumps(text)}")

    page.goto(f"{url}/?levels#path=sweep&group=config.lr")
    page.wait_for_function(READY, timeout=30000)
    sizes = page.evaluate("Object.fromEntries([...app.groups].map(([k, g]) => [k, g.runs.length]))")
    page.click("button.gfocus[title='open this group']")
    page.wait_for_function("app.opts.focus.length === 1")
    lr = page.evaluate("app.opts.focus[0][1]")
    ends(f"lr: {lr}")
    opened = page.evaluate(state)
    page.evaluate("app.setGroup(['config.seed'])")
    page.click("button.gfocus[title='open this group']")
    page.wait_for_function("app.opts.focus.length === 2")
    seed = page.evaluate("app.opts.focus[1][1]")
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
    return (opened[:4] == [1, "", False, sizes[lr]] and opened[4][-2:] == ["sweep", f"lr: {lr}"]
            and nested[:4] == [2, "", False, both] and nested[4][-3:] == ["sweep", f"lr: {lr}", f"seed: {seed}"]
            and up[:3] == [1, "config.seed", True] and back == [2, ""] and home == [0, "config.lr", True])


def interactions_smoke(page, url):
    """Whether charts carry no legend and the chart controls do what they say: hover and Shift-pinned tooltips
    (which the wheel scrolls, and whose rows reveal and open runs), x and box zooms and their reset, the chart
    settings (smoothing, axes, outliers, density, reset), the sort box (type, pick, Enter, Escape), sidebar hiding,
    group-by search, keyboard scrolling, the media slider, back/forward (back to the grouping the folder had), and
    Canvas 2D drawing."""
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
    page.click("#groupAdd")
    page.fill("#menu input[type=search]", "see")
    checks["group-by search narrows the fields"] = page.locator("#menu .mitem").count() == 1
    page.keyboard.press("Escape")
    page.locator("#panels").click(position={"x": 5, "y": 5})
    for k in ("End", "Home", "PageDown", "ArrowUp"):
        page.keyboard.press(k)
    slider = page.locator(".panel.media input[type=range]").first
    slider.scroll_into_view_if_needed()
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
    page.goto(f"{url}/?iqm#path=sweep&group=config.lr&center=iqm")
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
    page.goto(f"{url}/?gl=0#path=sweep&group=config.lr")
    page.wait_for_function(READY, timeout=30000)
    hover(plot())
    page.goto(f"{url}/?gl=0#path=sweep&group=")
    page.wait_for_function(READY, timeout=30000)
    checks["Canvas 2D draws without WebGL"] = page.evaluate(f"!!{chart}.view && !{chart}.view.gl")
    page.click("#clearCache")
    page.wait_for_timeout(500)
    failed = [k for k, v in checks.items() if not v]
    print(f"interactions: {len(checks) - len(failed)}/{len(checks)} as intended" + (f"; not: {failed}" if failed else ""))
    return not failed


def daemon_smoke(page, runs, tmp, env, out, errors):
    """Whether the daemon's root shows every tracked directory (a folder in it as `/ name`), a workspace made in the panel (opened from the trex
    brand) merges them and opens though its URL was visited before it existed, and the panel removes, adds and
    forgets directories. The refused add's 400 is taken out
    of `errors`."""
    port = free_port()
    url = f"http://127.0.0.1:{port}"
    daemon = subprocess.Popen([sys.executable, "-m", "trex", "daemon", str(runs / "sweep"), "--port", str(port),
                               "--cache", str(tmp / "cache")], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    row = "#menu .mrow:has(.ml:text-is('{}'))"
    try:
        daemon.stdout.readline()
        subprocess.run([sys.executable, "-m", "trex", "serve", str(runs / "live"), "-y"], check=True, env=env)
        with urllib.request.urlopen(f"{url}/r/sweep/api/runs") as r1, urllib.request.urlopen(f"{url}/r/live/api/runs") as r2:
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
        merged = page.evaluate("[app.data.runs.size, app.groupFields().some((f) => f.id === 'dir')]")
        page.screenshot(path=str(out / "workspace.png"))
        page.click("#crumbPath .crumb:text-is('/')")
        page.wait_for_url(f"{url}/")
        page.wait_for_function(READY, timeout=30000)
        page.click(".brand")
        page.click(row.format("live") + " .mitem")
        page.wait_for_url(f"{url}/r/live/")
        page.wait_for_function(READY, timeout=30000)
        page.once("dialog", lambda d: d.accept())
        page.click(".brand")
        page.click(row.format("live") + " .chev")
        page.wait_for_url(f"{url}/")
        page.wait_for_function(READY, timeout=30000)
        page.goto(f"{url}/r/sweep/")
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
        page.wait_for_url(f"{url}/r/live/")
        page.wait_for_function(READY, timeout=30000)
        page.click(".brand")
        page.wait_for_selector("#menu .mrow")
        page.screenshot(path=str(out / "daemon_menu.png"))
        recent_left = page.query_selector("#menu .mrecent") is not None
        panel_text = page.inner_text("#menu")
        tidy = "null" not in panel_text.split() and "unknown" not in panel_text
        with urllib.request.urlopen(f"{url}/api/daemon") as r:
            d = json.loads(r.read())
        left, workspaces = [x["name"] for x in d["roots"]], [(w["name"], w["members"]) for w in d["workspaces"]]
        print(f"daemon: root shows {root[0]}/{total} runs in {root[2]} as {root[1]!r}, a folder in it as {crumbs}; workspace of both shows "
              f"{merged[0]}/{total}, dir field {merged[1]}; removed live, re-added it from history; tracking {left}, "
              f"workspaces {workspaces}, history {d['history']}, panel tidy {tidy}")
        return (root == [total, "/", ["live", "sweep"]] and crumbs == ["/", "sweep"] and merged == [total, True] and sorted(left) == ["live", "sweep"]
                and workspaces == [("both", ["sweep"])] and not d["history"] and not recent_left and tidy)
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
    """Whether the filter box keeps the runs a WHERE clause or a name search selects, and marks a clause that does
    not parse while filtering nothing."""
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
    print(f"filter: of {len(every)} runs, " + "; ".join(f"{t!r} keeps {len(v[0])}{' (marked bad)' if v[1] else ''}" for t, v in results.items()))
    return (want and results["lr = 0.001 and seed = 1"] == [want, False] and results["lr = 0.00"] == [[], False]
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


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp(prefix="trex-shots-"))
    out.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="trex-smoke-"))
    runs = tmp / "runs"
    subprocess.run([sys.executable, str(REPO / "examples/demo.py"), str(runs / "sweep"), "--seeds", "3", "--steps", "1500"], check=True)
    subprocess.run([sys.executable, "-c", NESTED_WRITER, str(runs)], check=True)
    port = free_port()
    env = {**os.environ, "TREX_DAEMON_DIR": str(tmp / "daemon")}
    server = subprocess.Popen([sys.executable, "-m", "trex", "serve", str(runs), "--standalone", "--port", str(port),
                               "--cache", str(tmp / "cache")], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    url = f"http://127.0.0.1:{port}"
    ok = True
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
            for label, h in [("cold", "#path=sweep"), ("warm", "#path=sweep"), ("grouped", "#path=sweep&group=config.lr")]:
                page.goto(f"{url}/?{label}{h}")
                page.wait_for_function(ready, timeout=30000)
                page.wait_for_timeout(500)
                print(f"{label}: {page.inner_text('#status')}")
                page.screenshot(path=str(out / f"{label}.png"))

            ok &= group_levels_smoke(page, url)
            ok &= sections_smoke(page, url)
            ok &= filter_smoke(page, url)
            ok &= interactions_smoke(page, url)
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
                    const tiles = {};
                    for (const [k, e] of r.tiles) if (e.top) tiles[k] = { seq: e.topSeq, n: e.top.reduce((s, t) => { for (let i = 0; i < t.count; i++) s += t.u32[t.f + 5 * t.count + i]; return s; }, 0) };
                    out[r.id] = { seq: r.seq, media: r.mseq, tiles, tail: r.tail.length };
                }
                return { runs: out, dropped: app.data.dropped };
            }""")
            for rid, c in sorted(client["runs"].items()):
                n, finite, mseq = file_columns(runs / rid)
                tiles = {k: t["n"] for k, t in c["tiles"].items()}
                match = (c["seq"] == n and c["media"] == mseq and c["tail"] == 0 and tiles
                         and all(t["seq"] == n for t in c["tiles"].values()) and all(tiles[k] == finite.get(k, 0) for k in tiles))
                ok &= bool(match)
                print(f"{rid}: client {c['seq']} rows / {c['media']} media / tile points {tiles}, "
                      f"file {n} / {mseq} / finite {finite}: {'match' if match else 'MISMATCH'}")
            print(f"dropped {client['dropped']} rows events; client {'converged' if ok else 'DIVERGED'}")
            server.terminate()
            ok &= daemon_smoke(page, runs, tmp, env, out, errors)
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
    if ok:
        shutil.rmtree(tmp)
    print(f"screenshots in {out}" + ("" if ok else f"; scratch data kept in {tmp}"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
