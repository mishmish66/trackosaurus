"""Interaction latency benchmark. The target: every interaction redraws every chart in view within TARGET_MS.

A headless Chromium on the GPU replays the interactions of a session against a throwaway trex (`trex serve
--temporary`) on generated runs shaped like a large project, or against a running trex (`--url`):

- wheel scrolls through the charts, to charts not drawn yet (their blocks fetched ahead) and back to drawn ones;
- a chart shown alone (its ⛶ button) and left again (Escape);
- hovering a chart (its tooltip);
- a drag zoom, a scroll while zoomed, and a click resetting the zoom;
- filtering the runs and clearing the filter, ungrouping and regrouping;
- entering a run and leaving it.

Inputs keep a user's timing where the page's work depends on it: a filter is typed KEY_GAP_MS after its box is clicked,
and a drag is released DRAG_REST_MS after its last movement (the page bins a zoom while its drag rests).

Each is timed from its input event to the end of the draw of the last chart in view showing its data as it now stands
(every block of its view in hand, no column left to rebuild; "no data" counts as drawn), a hover to the end of its
tooltip. Printed beside: the GPU time of the WebGL draws of that last chart's round (a timer query around them), which
the lines take on the GPU after the page has issued them; the frame showing them needs both. With `--verify N` it then
makes N random interactions and after each compares what the GPU binned for the grouped charts in view with what
kernel.js computes of the same runs: a check of this machine's GPU and driver, which the smoke test's software
rendering of a small folder cannot make. The generated runs (RUNS of them in sweeps of SEEDS, under DIRS, METRICS
metrics each; LIVE of them logging while the benchmark runs) and the trex index of them are kept in `--data`, so only
the first run of the benchmark writes and indexes them. Prints each interaction's median and worst over the rounds
against the target, and exits 1 when some interaction's worst exceeds it.

The browser's profile is kept too (`--profile`, by default in `--data`), as a used browser's is, for its cache of
compiled shaders. Without it the browser compiles the shaders of its own drawing at their first use (a focus ring,
what a 2D canvas draws of a selection): 10 to 30 ms on the GPU process's thread, which the page's own draws wait
for. So the first run with a new profile, or after a browser or driver update, shows such first uses slow.

    uv run playwright install chromium   # once
    uv run python tests/browser_bench.py [--url URL] [--rounds N] [--target MS] [--verify N] [--json FILE] [--profile DIR]
"""

import argparse
import json
import math
import os
import platform
import random
import shutil
import socket
import statistics
import subprocess
import sys
import time
import urllib.request
import uuid
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from playwright.sync_api import BrowserContext, Page, Playwright, sync_playwright

from trex import chunks, journal
from trex.format import FORMAT, connect_rw

REPO = Path(__file__).resolve().parents[1]
TARGET_MS = 10.0
KEY_GAP_MS = 100  # from a click into a text box to the first key typed there
DRAG_REST_MS = 80  # from the end of a drag's movement to the release of the button
GPU_FLAGS = ["--headless=new", "--use-gl=angle", "--use-angle=gl-egl", "--ignore-gpu-blocklist", "--enable-gpu",
             "--disable-smooth-scrolling"]  # a wheel event scrolls at once: the time is the page's, not an animation's
VIEWPORT = {"width": 1800, "height": 1100}
SCALE = 1.25  # device pixels per CSS pixel

GENERATION = 1  # bump whenever the generated runs change
DIRS = ("cheetah", "hopper", "walker", "ant")
SWEEPS = 64  # sweep folders per directory
SEEDS = 8  # runs per sweep folder
RUNS = len(DIRS) * SWEEPS * SEEDS
ROWS = 400  # rows per run
STRIDE = 1000  # steps between rows
BASES = ("loss", "q_loss", "pi_loss", "v_loss", "entropy", "alpha", "td_error", "q_value", "q_target", "grad_norm", "kl", "adv",
         "lr", "ratio", "clip", "explained_var", "reward", "done", "log_prob", "std", "mean", "value", "bootstrap", "delta")
STATS = ("max", "mean", "min", "q05", "q25", "q50", "q75", "q95", "var")
TRAIN = [f"train/{b}/{s}" for b in BASES for s in STATS]  # every row, every run
EVAL = [f"eval/{k}" for k in ("return", "len", "success", "max_progress")]  # every EVAL_EVERY rows, every run
EVAL_EVERY = 10
DEBUG = [f"debug/{k}" for k in ("grad_spike", "nan_count", "clip_frac", "var_ratio")]  # every row, two sweeps per directory
METRICS = len(TRAIN) + len(EVAL) + len(DEBUG)
LIVE = 32  # runs logging while the benchmark runs
LIVE_RATE = 10.0  # rows per second each
WRITERS = os.cpu_count() or 4

INSTRUMENT = """async () => {
  if (window.__bench) return;
  const B = (window.__bench = { draws: [], hovers: [], inputs: {} }), gl = (await import("/static/gl.js")).renderer()?.gl;
  const C = [...app.charts.values()][0].constructor.prototype, draw = C.draw, fit = C.fitCanvases, hover = C.hover;
  // the GPU time of a task's chart drawing: a timer query from its first renderGL to the end of the task's draws
  const timer = gl?.getExtension("EXT_disjoint_timer_query_webgl2"), renderGL = C.renderGL;
  let batch = null;
  const ended = () => (timer && batch.query && gl.endQuery(timer.TIME_ELAPSED_EXT), (batch = null));
  const inBatch = () => batch || ((batch = { query: null, draws: [] }), queueMicrotask(ended), batch);
  C.renderGL = function (...args) {
    const b = inBatch();
    if (timer && !b.query) (b.query = gl.createQuery()), gl.beginQuery(timer.TIME_ELAPSED_EXT, b.query);
    return renderGL.apply(this, args);
  };
  B.gpuOf = (d) => { // the draw's round's GPU ms once known; 0 without timer queries or draws on the GPU
    const q = d.batch.query;
    if (!q || d.gpu !== null) return d.gpu ?? (d.gpu = 0);
    if (!gl.getQueryParameter(q, gl.QUERY_RESULT_AVAILABLE)) return null;
    d.gpu = gl.getParameter(timer.GPU_DISJOINT_EXT) ? NaN : gl.getQueryParameter(q, gl.QUERY_RESULT) / 1e6;
    return d.gpu;
  };
  C.fitCanvases = function () { this.__drew = true; return fit.call(this); };
  C.draw = function (...args) {
    this.__drew = false;
    const out = draw.apply(this, args), t = performance.now();
    if (this.__drew) {
      // settled: it shows the layers its view wants now, with every block of them for every run it shows, no column left
      const d = app.data, ch = d.charts.get(this.key), shows = this.view ? 1 : this.emptyText() === "no data" ? 2 : 0;
      const runs = d.runsWith(app.shown, this.key), want = d.layersOf(app.demand(this, app.shown), runs);
      const settled = !!ch?.ready && JSON.stringify(ch.ready) === JSON.stringify(want) && d.complete(this.key, want, runs) && !d.pending(this.key);
      B.draws.push({ t, chart: this, shows, settled, batch: inBatch(), gpu: null });
    }
    return out;
  };
  C.hover = function (e) { hover.call(this, e); B.hovers.push(performance.now()); };
  for (const kind of ["wheel", "mouseup", "click", "keydown", "mousemove", "input"]) addEventListener(kind, () => (B.inputs[kind] = performance.now()), true);
}"""

# Whether every chart in view shows its data as it now stands, drawn after t0 (or, unless `every`, untouched and current
# since before it), the end of the last such draw after t0 (t0 itself when none was needed), and the GPU time of its
# round's draws.
DRAWN = """([t0, every]) => {
  const B = window.__bench, panels = document.getElementById("panels"), box = panels.getBoundingClientRect();
  const alone = panels.classList.contains("alone"); // the chart shown alone covers the grid
  const shown = [...app.charts.values()].filter((c) => { const r = c.el.getBoundingClientRect();
    return alone ? c.full : r.bottom > box.top && r.top < box.bottom && r.width > 0 && c.el.checkVisibility(); });
  let last = t0, lastDraw = null;
  for (const c of shown) {
    const mine = B.draws.filter((d) => d.chart === c), after = mine.find((d) => d.t >= t0 && d.shows && d.settled);
    if (after && (!lastDraw || after.t > lastDraw.t)) lastDraw = after;
    if (after) { last = Math.max(last, after.t); continue; }
    if (every) return { done: false, charts: shown.length };
    const prev = mine.at(-1);
    if (!(prev && prev.t < t0 && prev.shows && prev.settled && !c.dirty && !c.waiting && c.canvas.width > 0)) return { done: false, charts: shown.length };
  }
  const gpu = lastDraw ? B.gpuOf(lastDraw) : 0;
  if (gpu === null) return { done: false, charts: shown.length };
  return { done: shown.length > 0, ms: last - t0, gpu, charts: shown.length };
}"""

VISIBLE_PLOT = """() => { const box = document.getElementById("panels").getBoundingClientRect();
  const c = [...app.charts.values()].find((c) => { const r = c.el.getBoundingClientRect();
    return r.top > box.top + 40 && r.bottom < box.bottom - 10 && r.width > 0 && c.el.checkVisibility() && c.view; });
  if (!c) return null;
  const r = c.overlay.getBoundingClientRect(), b = c.el.querySelector("button.full").getBoundingClientRect();
  return { key: c.key, left: r.left + 52, top: r.top + 6, width: r.width - 62, height: r.height - 26,
           full: [b.left + b.width / 2, b.top + b.height / 2] }; }"""


@dataclass(frozen=True, slots=True)
class RunSpec:
    """One generated run: its directory, seed and config, which metrics it logs, and its final state."""

    path: str
    seed: int
    sweep: int
    config: dict[str, Any]
    debug: bool
    state: str


@dataclass(slots=True)
class Results:
    """Times of each interaction (ms; with the GPU time of the draws of its last chart's round, Drawn), and those that
    never finished within the timeout."""

    ms: dict[str, list[float]] = field(default_factory=dict[str, list[float]])
    gpu: dict[str, list[float]] = field(default_factory=dict[str, list[float]])
    timeouts: dict[str, int] = field(default_factory=dict[str, int])

    def add(self, name: str, ms: float | None) -> None:
        if ms is None:
            self.timeouts[name] = self.timeouts.get(name, 0) + 1
        else:
            self.ms.setdefault(name, []).append(ms)

    def add_drawn(self, name: str, d: "Drawn | None") -> None:
        self.add(name, d and d.ms)
        if d:
            self.gpu.setdefault(name, []).append(d.gpu)


@dataclass(frozen=True, slots=True)
class Drawn:
    """ms from an interaction's input until the last chart in view was drawn, and the GPU ms of the draws of that chart's
    round (NaN when the GPU's timing was disturbed)."""

    ms: float
    gpu: float


# ---- the generated runs ----

def run_specs(root: Path) -> list[RunSpec]:
    """Every generated run: DIRS / sweep / seed, sweeps varying the learning rate, batch size and width."""
    out: list[RunSpec] = []
    for d, name in enumerate(DIRS):
        for sweep in range(SWEEPS):
            for seed in range(SEEDS):
                n = (d * SWEEPS + sweep) * SEEDS + seed
                config = {"algo": name, "lr": (1e-4, 3e-4, 1e-3, 3e-3)[sweep % 4], "batch": (64, 256)[sweep // 4 % 2],
                          "seed": seed, "net": {"width": (128, 256, 512, 1024)[sweep // 8 % 4], "depth": 2 + sweep // 32}}
                state = "failed" if n % 23 == 0 else "running" if n % 11 == 0 else "finished"  # "running", long silent: crashed
                out.append(RunSpec(str(root / name / f"sweep{sweep:02d}" / str(seed)), n, sweep, config, sweep < 2, state))
    return out


def curve(rng: np.random.Generator, step: npt.NDArray[np.float64], spec: RunSpec, k: int) -> npt.NDArray[np.float64]:
    """A metric's values: a decay toward a floor set by the sweep, with noise and the occasional spike."""
    tau = 1e5 * (1 + spec.sweep % 5) * (1 + 0.3 * (k % 7))
    floor = 0.1 * (1 + spec.sweep % 3) + 0.01 * spec.seed
    v = (1 + k % 3) * np.exp(-step / tau) + floor + rng.normal(0, 0.03 * (1 + k % 4), step.size)
    spikes = rng.random(step.size) < 0.002
    v[spikes] *= 5
    return v


def write_run(spec: RunSpec) -> None:
    """Write one run's file directly, as one commit (far faster than logging its rows)."""
    d = Path(spec.path)
    (d / "media").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(spec.seed)
    step = np.arange(ROWS, dtype=np.float64) * STRIDE
    names = TRAIN + EVAL + (DEBUG if spec.debug else [])
    ops: list[tuple[str, tuple[Any, ...]]] = [("keys", (kid, name)) for kid, name in enumerate(names)]
    ops.append(("rowmeta", (0, ROWS, float(step[0]), float(step[-1]), step.tobytes() + (step / 2000).tobytes())))
    every = np.arange(0, ROWS, EVAL_EVERY, dtype=np.uint16)
    for kid, name in enumerate(names):
        sparse = name in EVAL
        v = curve(rng, step[every] if sparse else step, spec, kid)
        ops.append(("chunk", (kid, 0, chunks._blob(every.tobytes() if sparse else None, v.size, v.tobytes()))))
    created = time.time() - 86400
    meta: dict[str, Any] = {"id": uuid.uuid4().hex, "created": created, "format": FORMAT, "summary": {}, "name": d.name,
                            "state": spec.state, "heartbeat": created + ROWS * STRIDE / 2000, "tags": [], "config": spec.config,
                            "info": {}}
    c = connect_rw(d)
    try:
        c.execute("BEGIN")
        journal.replay(c, ops)
        c.executemany("INSERT OR REPLACE INTO meta VALUES (?, ?)", [(k, json.dumps(v)) for k, v in meta.items()])
        c.execute("COMMIT")
    finally:
        c.close()


def ensure_runs(data: Path) -> Path:
    """The generated runs under `data`, written when missing or of another generation."""
    runs, mark = data / "runs", data / "generation"
    if mark.exists() and mark.read_text() == str(GENERATION) and runs.exists():
        return runs
    shutil.rmtree(data, ignore_errors=True)
    runs.mkdir(parents=True)
    specs = run_specs(runs)
    t = time.monotonic()
    with ProcessPoolExecutor(WRITERS) as pool:
        for i, _ in enumerate(pool.map(write_run, specs, chunksize=16)):
            if (i + 1) % 512 == 0:
                print(f"  wrote {i + 1}/{len(specs)} runs", flush=True)
    print(f"wrote {len(specs)} runs of {METRICS} metrics in {time.monotonic() - t:.0f} s", flush=True)
    mark.write_text(str(GENERATION))
    return runs


LIVE_WRITER = """
import math, sys, time, trex
root, n, rate = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
names = sys.argv[4].split(",")
runs = [trex.init(f"{root}/live{i // 8:02d}/{i % 8}", config={"seed": i % 8, "live": True}, commit_interval=0.2) for i in range(n)]
print("ready", flush=True)
step = 0
while True:
    for i, r in enumerate(runs):
        r.log({k: math.exp(-step / 3000) + 0.01 * (j % 9) + 0.002 * i for j, k in enumerate(names)}, step=step * 1000)
    step += 1
    time.sleep(1 / rate)
"""


# ---- the server and the page ----

def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def get_json(url: str) -> Any:
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def start_server(runs: Path, data: Path) -> tuple[subprocess.Popen[str], str]:
    """A temporary trex serving `runs`, its index kept in `data`, once it holds every generated run."""
    port = free_port()
    env = {**os.environ, "TREX_DAEMON_DIR": str(data / "daemon")}
    server = subprocess.Popen([sys.executable, "-m", "trex", "serve", str(runs), "--temporary", "--port", str(port),
                               "--cache", str(data / "cache")], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    url = f"http://127.0.0.1:{port}"
    t, seen = time.monotonic(), -1
    while True:
        if server.poll() is not None:
            raise RuntimeError(f"trex exited: {server.stdout.read() if server.stdout else ''}")
        try:
            n = len(get_json(f"{url}/api/runs?path=")["runs"])
        except OSError:
            n = -1
        if n >= RUNS:
            break
        if n != seen and n >= 0:
            print(f"  trex indexing: {n}/{RUNS} runs", flush=True)
            seen = n
        time.sleep(1.0)
    print(f"trex serving {url} ({time.monotonic() - t:.0f} s to index)", flush=True)
    return server, url


def browser_path() -> str | None:
    found = next(Path.home().glob(".cache/ms-playwright/chromium-*/chrome-linux64/chrome"), None)
    return str(found) if found else shutil.which("chromium")


def launch(p: Playwright, profile: Path) -> BrowserContext:
    """Chromium on the GPU, its profile in `profile`: what a browser keeps between sessions, its compiled shaders among
    it, is there for the next run."""
    return p.chromium.launch_persistent_context(str(profile), executable_path=browser_path(), headless=True, args=GPU_FLAGS,
                                                viewport={"width": VIEWPORT["width"], "height": VIEWPORT["height"]}, device_scale_factor=SCALE)


def open_page(ctx: BrowserContext, url: str, local: dict[str, str]) -> Page:
    """The page at `url` once its charts exist, its localStorage holding `local` alone (as a user's would hold it),
    its draws recorded from then on (INSTRUMENT): charts drawn ahead of their being scrolled to count as drawn."""
    page = ctx.new_page()
    page.add_init_script("(() => { if (!sessionStorage.getItem('seeded')) { const ls = %s; localStorage.clear(); "
                         "for (const [k, v] of Object.entries(ls)) localStorage.setItem(k, v); sessionStorage.setItem('seeded', '1'); } })();"
                         % json.dumps(local))
    page.goto(url)
    page.wait_for_function("window.app && app.runList && app.charts.size > 0", timeout=180000)
    page.evaluate(INSTRUMENT)
    return page


def settle(page: Page, limit_s: float) -> None:
    """Wait while the page fetches blocks (its plans and fetching ahead), at most `limit_s` seconds."""
    end, quiet = time.monotonic() + limit_s, 0
    while time.monotonic() < end and quiet < 10:
        busy = page.evaluate("(() => { const d = app.data; return d.posts + d.aheadPosts + d.queue.length + d.rebuildQ.size; })()")
        quiet = quiet + 1 if not busy else 0
        page.wait_for_timeout(200)


def drawn(page: Page, t0: float, timeout_ms: float, every: bool = False) -> Drawn | None:
    """The times from t0 until every chart in view shows its data as it now stands (each drawn after t0 when `every`);
    None when not within `timeout_ms`."""
    end = time.monotonic() + timeout_ms / 1000
    while time.monotonic() < end:
        r = page.evaluate(DRAWN, [t0, every])
        if r["done"]:
            return Drawn(ms=float(r["ms"]), gpu=float(r["gpu"]) if r["gpu"] is not None else math.nan)
        page.wait_for_timeout(10)
    return None


def drawn_ms(page: Page, t0: float, timeout_ms: float, every: bool = False) -> float | None:
    """The first of `drawn`'s times."""
    d = drawn(page, t0, timeout_ms, every)
    return d and d.ms


def quiet(page: Page, timeout_ms: float) -> bool:
    """Wait until every chart in view shows its data as it now stands (drawn since this call, or untouched since its last
    draw); whether it did within `timeout_ms`."""
    return drawn(page, now(page), timeout_ms) is not None


# ---- the interactions ----

def now(page: Page) -> float:
    return float(page.evaluate("performance.now()"))


def last_input(page: Page, kind: str) -> float:
    return float(page.evaluate(f"window.__bench.inputs[{json.dumps(kind)}]"))


def wheel(page: Page, screens: float) -> float:
    """A fast wheel scroll over the charts by `screens` screen heights; the time of its last wheel event."""
    page.mouse.move(VIEWPORT["width"] * 0.6, VIEWPORT["height"] * 0.6)
    ticks = max(1, round(abs(screens) * VIEWPORT["height"] / 240))
    for _ in range(ticks):
        page.mouse.wheel(0, 240 if screens > 0 else -240)
        page.wait_for_timeout(8)
    page.wait_for_timeout(30)
    return last_input(page, "wheel")


def api(page: Page, js: str) -> float:
    """Run `js` in the page; the time just before it ran."""
    return float(page.evaluate(f"(() => {{ const t0 = performance.now(); {js}; return t0; }})()"))


def type_filter(page: Page, text: str) -> float:
    """Replace the run filter's text with `text` as typing or pasting it does, a moment after the box is clicked, as a
    user's first key follows the click (the frame the click draws, with the box's focus ring and selection, is then not
    being drawn while the filter is applied); the time of its input event."""
    page.click("#runFilter")
    page.keyboard.press("Control+a")
    page.wait_for_timeout(KEY_GAP_MS)
    if text:
        page.keyboard.insert_text(text)
    else:
        page.keyboard.press("Backspace")
    return last_input(page, "input")


def drag_zoom(page: Page) -> float | None:
    """Drag across the middle of a chart in view, releasing where the pointer came to rest a moment before, as a
    user's release follows the end of the movement; the time of the mouse release."""
    plot = page.evaluate(VISIBLE_PLOT)
    if not plot:
        return None
    y = plot["top"] + plot["height"] / 2
    page.mouse.move(plot["left"] + 0.3 * plot["width"], y)
    page.mouse.down()
    for f in (0.4, 0.5, 0.6):
        page.mouse.move(plot["left"] + f * plot["width"], y)
        page.wait_for_timeout(16)
    page.wait_for_timeout(DRAG_REST_MS)
    page.mouse.up()
    return last_input(page, "mouseup")


def click_plot(page: Page) -> float | None:
    """Click inside a chart in view (a click resets the zoom); the time of the release."""
    plot = page.evaluate(VISIBLE_PLOT)
    if not plot:
        return None
    page.mouse.click(plot["left"] + plot["width"] / 2, plot["top"] + plot["height"] / 2)
    return last_input(page, "mouseup")


def show_alone(page: Page) -> float | None:
    """Press the ⛶ button of a chart in view; the time of the click."""
    plot = page.evaluate(VISIBLE_PLOT)
    if not plot:
        return None
    page.mouse.move(*plot["full"])
    page.wait_for_timeout(50)
    page.mouse.click(*plot["full"])
    return last_input(page, "click")


def hover_ms(page: Page) -> float | None:
    """Move the pointer onto a chart in view: ms from the move to the end of its tooltip."""
    plot = page.evaluate(VISIBLE_PLOT)
    if not plot:
        return None
    page.mouse.move(plot["left"] + plot["width"] * 0.7, plot["top"] + plot["height"] * 0.5)
    t0 = last_input(page, "mousemove")
    end = time.monotonic() + 2
    while time.monotonic() < end:
        done = page.evaluate("(t0) => window.__bench.hovers.find((t) => t >= t0) ?? null", t0)
        if done is not None:
            page.mouse.move(5, 5)
            return float(done) - t0
        page.wait_for_timeout(5)
    return None


def a_run(page: Page) -> str | None:
    """A finished run of the charts in view."""
    return page.evaluate("(() => app.shown.find((r) => r.meta.state === 'finished')?.id ?? null)()")


def timer(page: Page, res: Results, timeout_ms: float) -> Callable[..., None]:
    """timed(name, start, every=False): once the charts in view are drawn, start an interaction (start() returns the
    time of its input, or None when it cannot be made) and add the ms until every chart in view shows its data as it
    now stands, each drawn anew when `every` (an interaction changing what every chart shows)."""
    def timed(name: str, start: Callable[[], float | None], every: bool = False) -> None:
        if not quiet(page, timeout_ms):
            print(f"  {name:14} (charts in view not drawn before it)", flush=True)
        t0 = start()
        if t0 is not None:
            d = drawn(page, t0, timeout_ms, every)
            res.add_drawn(name, d)
            print(f"  {name:14} {fmt(d and d.ms)}" + (f"   GPU {d.gpu:5.1f}" if d else ""), flush=True)
        page.wait_for_timeout(300)
    return timed


def round_of(page: Page, res: Results, r: int, timeout_ms: float) -> None:
    """One round of the interactions within a view, starting r * 8 screens further down each round (the scrolls go to
    charts not drawn yet)."""
    H = VIEWPORT["height"]
    page.evaluate(f"(() => {{ const p = document.getElementById('panels'); p.scrollTop = Math.min({2 * H + r * 8 * H}, p.scrollHeight - 8 * {H}); }})()")
    page.wait_for_timeout(1500)
    timed = timer(page, res, timeout_ms)
    timed("scroll-new", lambda: wheel(page, 4))
    timed("scroll-back", lambda: wheel(page, -4))
    timed("show-alone", lambda: show_alone(page), True)
    timed("leave-alone", lambda: (page.keyboard.press("Escape"), last_input(page, "keydown"))[1])
    quiet(page, timeout_ms)
    for _ in range(3):
        res.add("hover", hover_ms(page))
    timed("zoom", lambda: drag_zoom(page), True)
    timed("scroll-zoomed", lambda: wheel(page, 3))
    timed("reset-zoom", lambda: click_plot(page), True)
    timed("filter", lambda: type_filter(page, "seed < 4"), True)
    timed("unfilter", lambda: type_filter(page, ""), True)
    page.keyboard.press("Escape")  # its completions
    page.evaluate("document.activeElement.blur()")
    timed("ungroup", lambda: api(page, "app.setGroup('run')"), True)
    timed("regroup", lambda: api(page, "app.setGroup(app.defaultGroup(app.opts.path))"), True)


def navigate(page: Page, res: Results, timeout_ms: float, settle_s: float) -> None:
    """Enter a run, leave it for its folder, and go back to the view the page opened with."""
    timed = timer(page, res, timeout_ms)
    run = a_run(page)
    if not run:
        return
    parent = run.rsplit("/", 1)[0] if "/" in run else ""
    timed("enter-run", lambda: api(page, f"app.setPath({json.dumps(run)})"), True)
    timed("leave-run", lambda: api(page, f"app.setPath({json.dumps(parent)})"), True)
    timed("open-root", lambda: api(page, "app.setPath('')"), True)
    settle(page, settle_s)


# Of every chart in view whose groups the GPU binned: each group's center, band and count in each bin as the GPU holds
# them against kernel.js aggGroups and plot.js bandOf of the same runs. {charts, compared, wrong, eg (the first few)}.
VERIFY = """async () => {
  const k = await import("/static/kernel.js"), { bandOf } = await import("/static/plot.js");
  const NO_TAIL = { s: [], v: [], t: [], q: [], n: 0 }, ROW = Object.fromEntries(k.STATS.map((s, i) => [s, i]));
  const box = document.getElementById("panels").getBoundingClientRect(), keys = [...app.groups.keys()];
  const out = { charts: 0, compared: 0, wrong: 0, eg: [] };
  for (const c of app.charts.values()) {
    const r = c.el.getBoundingClientRect(), v = c.view, g = v?.gpu;
    if (!(r.bottom > box.top && r.top < box.bottom && r.width > 0 && c.el.checkVisibility()) || !g?.agg) continue;
    const groups = app.linesFor(c.key), got = g.out.read(g.out.x, 0, g.bins, g.n);
    if (!got) continue;
    const cols = c.sources(groups, c.binned).map((gr) => gr.map((s) => (!s.parts ? s
      : s.parts.length === 1 ? k.runColumn(s.parts[0].v, s.parts[0].row) : k.buildColumn(s.parts, NO_TAIL, s.level, false))));
    const flags = (v.logx ? k.LOGX : 0) | (v.o.center === "iqm" ? k.IQM : 0);
    const st = k.aggGroups(cols, v.xmode, g.g0, g.g0 + g.bins * g.dx, g.bins, flags, 0, 1);
    out.charts++;
    groups.forEach((ln, li) => {
      const at = { a: st, o: li * k.NSTAT * g.bins }, [lo, hi] = bandOf(at, v.o.center, v.o.band, g.bins), gi = keys.indexOf(ln.group);
      for (let b = 0; b < g.bins; b++) {
        const want = [st[at.o + ROW[v.o.center] * g.bins + b], lo[b], hi[b], st[at.o + ROW.n * g.bins + b]];
        const scale = Math.max(1e-30, ...want.slice(0, 3).map((x) => (Number.isFinite(x) ? Math.abs(x) : 0)));
        want.forEach((x, q) => {
          const y = got[4 * (gi * g.bins + b) + q];
          out.compared++;
          if ((Number.isNaN(x) && Number.isNaN(y)) || x === y || (q < 3 && Math.abs(x - y) <= 2e-4 * scale)) return;
          out.wrong++;
          if (out.eg.length < 5) out.eg.push([c.key, ln.label, b, ["center", "low", "high", "runs"][q], x, y]);
        });
      }
    });
  }
  return out;
}"""

EVENT = "(() => {{ const e = document.getElementById({id}); e.value = {value}; e.dispatchEvent(new Event({kind})); }})()"


def verify(page: Page, steps: int, generated: bool, timeout_ms: float) -> bool:
    """`steps` random interactions (regroupings, the center and the band, scrolls; of the generated runs also filters,
    zooms, set and dragged (a dragged one is binned while its drag rests), and their large groups), each followed,
    while the view is grouped, by VERIFY; whether the GPU's statistics agreed with kernel.js every time."""
    rnd = random.Random(1)
    acts: list[str | Callable[[], object]] = [
        "app.setGroup('run')", "app.setGroup(app.defaultGroup(app.opts.path))",
        "document.getElementById('panels').scrollTop += 900", "document.getElementById('panels').scrollTop -= 900",
        *(EVENT.format(id="'center'", value=json.dumps(c), kind="'change'") for c in ("mean", "median", "iqm")),
        *(EVENT.format(id="'band'", value=json.dumps(b), kind="'change'") for b in ("ci", "iqr", "minmax", "std"))]
    if generated:
        acts += [*(EVENT.format(id="'runFilter'", value=json.dumps(f), kind="'input'") for f in ("seed < 4", "seed >= 6", "")),
                 "app.setGroup('seed')", "app.setGroup('run~2')", "app.setGroup('seed')",
                 f"app.setXRange([{100 * STRIDE}, {180 * STRIDE}, 0])", f"app.setXRange([{50 * STRIDE}, {90 * STRIDE}, 0])", "app.setXRange(null)",
                 lambda: drag_zoom(page), lambda: drag_zoom(page)]
    checks = compared = wrong = 0
    for _ in range(steps):
        act = rnd.choice(acts)
        if isinstance(act, str):
            page.evaluate(act)
        else:
            act()
        page.wait_for_timeout(rnd.choice([150, 400, 900]))
        if not page.evaluate("app.grouped") or not quiet(page, timeout_ms):
            continue
        page.wait_for_timeout(250)
        r = page.evaluate(VERIFY)
        checks, compared, wrong = checks + 1, compared + int(r["compared"]), wrong + int(r["wrong"])
        if r["wrong"]:
            state = page.evaluate("[app.opts.group, app.opts.filter, app.xrange, document.getElementById('center').value]")
            print(f"  {r['wrong']} values differ with group, filter, zoom, center {state}: {r['eg']}", flush=True)
    print(f"verified: {compared} values of {checks} views against kernel.js, {wrong} differ", flush=True)
    return wrong == 0 and checks > 0


def fmt(ms: float | None) -> str:
    return "timeout" if ms is None else f"{ms:7.1f} ms"


def report(res: Results, target: float) -> bool:
    """Print each interaction's median and worst against the target, with the median and worst GPU time of its draws;
    whether every worst meets the target."""
    ok = True
    print(f"\n{'interaction':14} {'median':>9} {'worst':>9}  n   target {target:g} ms   GPU draws (median, worst)")
    def spread(v: list[float]) -> str:
        v = [x for x in v if not math.isnan(x)]
        return f"{statistics.median(v):5.1f} {max(v):5.1f}" if v else "    -     -"
    for name in [*res.ms, *(k for k in res.timeouts if k not in res.ms)]:
        ms, out = sorted(res.ms.get(name, [])), res.timeouts.get(name, 0)
        worst = float("inf") if out else ms[-1]
        good = worst <= target
        ok &= good
        med = f"{statistics.median(ms):7.1f} ms" if ms else "    -    "
        print(f"{name:14} {med:>9} {fmt(None if out else worst):>9} {len(ms) + out:2d}  {'ok  ' if good else 'SLOW'}"
              f"          {spread(res.gpu.get(name, []))}" + (f" ({out} timed out)" if out else ""))
    return ok


def gpu_name(page: Page) -> str:
    return page.evaluate("""(() => { const gl = document.createElement('canvas').getContext('webgl2');
      const e = gl && gl.getExtension('WEBGL_debug_renderer_info');
      return gl ? (e ? gl.getParameter(e.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER)) : 'no WebGL2'; })()""")


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--url", help="measure this trex instead of a throwaway one on generated runs")
    ap.add_argument("--data", default=str(Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "trex" / "bench"),
                    help="where the generated runs and their index are kept")
    ap.add_argument("--profile", help="the browser's profile directory, kept between runs (default: profile in --data)")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--target", type=float, default=TARGET_MS)
    ap.add_argument("--timeout", type=float, default=10000, help="ms an interaction may take before it counts as timed out")
    ap.add_argument("--settle", type=float, default=60, help="seconds to let the page fetch ahead before measuring")
    ap.add_argument("--local", help="JSON file of localStorage entries to seed (a user's closed sections, say)")
    ap.add_argument("--verify", type=int, default=0, metavar="N",
                    help="after the rounds, make N random interactions, checking the GPU's statistics against kernel.js")
    ap.add_argument("--json", help="write the times to this file")
    a = ap.parse_args()
    server: subprocess.Popen[str] | None = None
    writer: subprocess.Popen[str] | None = None
    url, ok, agree = a.url, False, True
    try:
        if not url:
            data = Path(a.data)
            runs = ensure_runs(data)
            shutil.rmtree(runs / "live", ignore_errors=True)
            server, url = start_server(runs, data)
            writer = subprocess.Popen([sys.executable, "-c", LIVE_WRITER, str(runs / "live"), str(LIVE), str(LIVE_RATE), ",".join(TRAIN[:18])],
                                      stdout=subprocess.PIPE, text=True)
            assert writer.stdout is not None
            writer.stdout.readline()
        local: dict[str, str] = json.loads(Path(a.local).read_text()) if a.local else {}
        res = Results()
        with sync_playwright() as p:
            ctx = launch(p, Path(a.profile) if a.profile else Path(a.data) / "profile")
            page = open_page(ctx, url, local)
            print(f"{platform.processor() or platform.machine()} · {gpu_name(page)} · {page.evaluate('app.data.runs.size')} runs · "
                  f"{page.evaluate('app.charts.size')} charts made", flush=True)
            settle(page, a.settle)
            for r in range(a.rounds):
                print(f"round {r + 1}", flush=True)
                round_of(page, res, r, a.timeout)
            print("navigation", flush=True)
            for _ in range(a.rounds):
                navigate(page, res, a.timeout, a.settle)
            if a.verify:
                agree = verify(page, a.verify, not a.url, a.timeout)
            ctx.close()
        ok = report(res, a.target) and agree
        if a.json:
            Path(a.json).write_text(json.dumps({"ms": res.ms, "gpu": res.gpu, "timeouts": res.timeouts, "target": a.target}))
    finally:
        for proc in (writer, server):
            if proc is not None:
                proc.terminate()
                proc.wait()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
