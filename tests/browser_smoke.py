"""Headless-browser smoke test against a throwaway trex server on a temporary runs directory.

Checks cold and warm loads, grouping, console errors, and that a client dropping every 5th
stream event still converges to the run files: every row and media item, and top tiles whose
bucket counts add up to each metric's finite values. Then, under a throwaway `trex daemon`: adding
a directory with `trex serve -y`, and from the path bar menu switching directories, removing one,
re-adding it from the remembered ones, and clearing them.

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


def daemon_smoke(page, runs, tmp, env, out, errors):
    """Whether the daemon serves both directories and its menu switches, removes, adds and forgets them.
    The refused add's 400 is taken out of `errors`."""
    port = free_port()
    url = f"http://127.0.0.1:{port}"
    daemon = subprocess.Popen([sys.executable, "-m", "trex", "daemon", str(runs / "sweep"), "--port", str(port),
                               "--cache", str(tmp / "cache")], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    try:
        daemon.stdout.readline()
        subprocess.run([sys.executable, "-m", "trex", "serve", str(runs / "live"), "-y"], check=True, env=env)
        page.goto(f"{url}/")
        page.wait_for_url(f"{url}/r/sweep/")
        page.wait_for_function(READY + " && app.daemon?.length === 2", timeout=30000)
        page.click("#crumbPath button[title='switch directory']")
        page.wait_for_selector("#menu .mrow")
        page.screenshot(path=str(out / "daemon_menu.png"))
        page.click("#menu .mrow:nth-child(2) .mitem")
        page.wait_for_url(f"{url}/r/live/")
        page.wait_for_function(READY, timeout=30000)
        page.once("dialog", lambda d: d.accept())
        page.click("#crumbPath button[title='switch directory']")
        page.click("#menu .mrow:nth-child(2) .chev")
        page.wait_for_url(f"{url}/r/sweep/")
        page.wait_for_function(READY, timeout=30000)
        menu = "#crumbPath button[title='switch directory']"
        page.click(menu)
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
        page.once("dialog", lambda d: d.accept())
        page.click(menu)
        page.click("#menu .mrow:nth-child(1) .chev")
        page.wait_for_selector("#menu .mrecent")
        page.screenshot(path=str(out / "daemon_recent.png"))
        page.click("#menu .mclear")
        page.wait_for_selector("#menu .mrecent", state="detached")
        with urllib.request.urlopen(f"{url}/api/daemon") as r:
            d = json.loads(r.read())
        left = [x["name"] for x in d["roots"]]
        print(f"daemon: switched to live, removed it, re-added it from history, removed sweep, cleared history; "
              f"serving {left}, history {d['history']}")
        return left == ["live"] and d["history"] == []
    finally:
        daemon.terminate()
        daemon.wait()


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp(prefix="trex-shots-"))
    out.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="trex-smoke-"))
    runs = tmp / "runs"
    subprocess.run([sys.executable, str(REPO / "examples/demo.py"), str(runs / "sweep"), "--seeds", "2", "--steps", "1500"], check=True)
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
