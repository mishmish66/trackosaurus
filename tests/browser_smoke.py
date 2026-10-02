"""Headless-browser smoke test against a throwaway trex server on a temporary runs directory.

Checks cold and warm loads, grouping, opening groups as path levels, nested chart sections and pinning, console errors, and that a client dropping every 5th
stream event still converges to the run files: every row and media item, and top tiles whose
bucket counts add up to each metric's finite values. Then, under a throwaway `trex daemon`: adding
a directory with `trex serve -y`, the root view of both, making a workspace of them in the panel the trex
brand opens, and from that panel removing a directory and re-adding it from the remembered ones.

    uv run --with playwright python tests/browser_smoke.py [screenshot_dir]
"""

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


def sections_smoke(page, url):
    """Whether chart sections nest by key path and fold one at a time, and a pinned chart shows in the pinned section
    while staying in its own."""
    tree = """() => { const walk = (d) => [d.querySelector(':scope > summary').textContent, d.open,
        [...d.querySelectorAll(':scope > .grid > .panel > .ptitle > span:first-child')].map((e) => e.textContent),
        [...d.querySelectorAll(':scope > details.section')].map(walk)];
        return [...document.querySelectorAll('#panels > details.section')].map(walk); }"""
    page.goto(f"{url}/?sections#path=nested")
    page.wait_for_function(READY, timeout=30000)
    page.wait_for_selector("#panels details.section details.section")
    nested = page.evaluate(tree)
    page.click("#panels details.section details.section > summary:text-is('return (2)')")
    folded = page.evaluate(tree)
    page.click("#panels details.section details.section > summary:text-is('return (2)')")
    page.hover(".panel:has(.pname:text-is('eval/return/mean')) .ptitle")
    page.click(".panel:has(.pname:text-is('eval/return/mean')) .pin")
    page.wait_for_function("document.querySelector('#panels > details.section > summary')?.textContent.startsWith('📌')")
    pinned = page.evaluate(tree)
    copies = page.evaluate("app.chartsOf('eval/return/mean').filter((c) => c.el.isConnected).length")
    page.click("#panels > details.section:first-of-type .pin")
    page.wait_for_function("!document.querySelector('#panels > details.section > summary')?.textContent.startsWith('📌')")
    unpinned = [page.evaluate(tree)[0][0], page.evaluate("app.chartsOf('eval/return/mean').length")]
    print(f"sections: {nested}; return folded {folded[1][3][0][1]}, eval open {folded[1][1]}; pinned {pinned[0]}, copies {copies}; unpinned {unpinned}")
    eval_sec = ["eval (4)", True, ["eval/len"], [["return (2)", True, ["eval/return/mean", "eval/return/std"], []],
                                                 ["video (1)", True, ["eval/video/frame"], []]]]
    return (nested == [["charts (1)", True, ["loss"], []], eval_sec] and folded[1][1] and not folded[1][3][0][1] and folded[1][3][1][1]
            and pinned[0] == ["📌 pinned (1)", True, ["eval/return/mean"], []] and pinned[2] == eval_sec and copies == 2
            and unpinned == ["charts (1)", 1])


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp(prefix="trex-shots-"))
    out.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="trex-smoke-"))
    runs = tmp / "runs"
    subprocess.run([sys.executable, str(REPO / "examples/demo.py"), str(runs / "sweep"), "--seeds", "2", "--steps", "1500"], check=True)
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
