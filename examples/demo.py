"""Synthetic sweep written as trex runs: DEST/<sweep>/lr<lr>/seed<k>, with images, videos and HTML.

    uv run python examples/demo.py demo_runs/synthetic                      # finished sweep
    uv run python examples/demo.py demo_runs/live --live --hours 12         # slow runs that keep streaming
"""

import argparse
import math
import random
import shutil
import threading
import time
from pathlib import Path

import numpy as np

import trex


def fake_run(dest, lr, seed, steps, delay, media_every, width):
    rng = random.Random(seed * 1000 + int(lr * 1e5))
    run = trex.init(dest, config={"lr": lr, "seed": seed, "model": {"width": width, "depth": 4}}, tags=["demo"],
                    info={"git": {"sha": "3f9c2e1", "branch": "main", "dirty": seed == 1},
                          "host": {"name": "gpu-box", "gpu": seed % 2}, "notes": f"lr {lr}, width {width}, seed {seed}"})
    loss, plateau = 2.5, rng.uniform(0.05, 0.3) / (width / 128)
    for step in range(steps):
        loss = plateau + (loss - plateau) * (1 - lr * 0.8) + rng.gauss(0, 0.01 + 0.03 * lr)
        run.log({"train": {"loss": loss, "acc": 1 - math.exp(-step * lr * 0.3) + rng.gauss(0, 0.01)},
                 "lr": lr * 0.5 * (1 + math.cos(math.pi * step / steps))}, step=step)
        if step % 50 == 0:
            run.log({"eval/loss": loss * 1.1 + rng.gauss(0, 0.02), "eval/success": min(1, max(0, 1 - loss / 2 + rng.gauss(0, 0.05)))},
                    step=step)
        if step % media_every == 0:
            y, x = np.mgrid[0:64, 0:64]
            img = np.stack([np.sin(x / 8 + step / 200 + seed), np.cos(y / 8 * lr * 10), np.sin((x + y) / 16)], -1)
            run.log_image("samples/heatmap", (img + 1) / 2, step=step)
            run.log_html("reports/table", f"<h3>lr {lr} seed {seed} @ {step}</h3><table border=1>" + "".join(
                f"<tr><td>{i}</td><td>{rng.random():.4f}</td></tr>" for i in range(50)) + "</table>", step=step)
            if shutil.which("ffmpeg") and step % (media_every * 2) == 0:
                t = np.arange(24)[:, None, None]
                frames = (np.sin(x[None] / 6 + t / 3 + seed + step / 500) * 127 + 128).astype(np.uint8)
                run.log_video("rollouts/video", np.repeat(frames[..., None], 3, -1), step=step, fps=12)
        time.sleep(delay)
    run.summary(final_loss=loss)
    run.finish()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dest", type=Path)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--live", action="store_true", help="a few slow runs that keep streaming")
    ap.add_argument("--hours", type=float, default=12.0)
    args = ap.parse_args()
    jobs = []
    if args.live:
        steps = int(args.hours * 3600 / 1.0)
        for lr in (0.001, 0.003):
            for seed in range(2):
                jobs.append((args.dest / "sweep" / f"lr{lr}" / f"seed{seed}", lr, seed, steps, 1.0, 600, 128))
    else:
        for width in (128, 512):
            for lr in (0.001, 0.003, 0.01):
                for seed in range(args.seeds):
                    jobs.append((args.dest / f"width{width}" / f"lr{lr}" / f"seed{seed}", lr, seed, args.steps, 0.0, 1000, width))
    if args.live:
        trex.folder_info(args.dest, question="does a slow live run stream smoothly?", started=time.strftime("%Y-%m-%d %H:%M"))
        trex.folder_info(args.dest / "sweep", trex={"group_by": ["subfolder"]})
    else:
        trex.folder_info(args.dest, question="how do width and learning rate interact?",
                         design={"widths": [128, 512], "lrs": [0.001, 0.003, 0.01], "seeds": args.seeds},
                         trex={"group_by": ["subfolder"]})
        for width in (128, 512):
            trex.folder_info(args.dest / f"width{width}", width=width, params_m=round(width * width * 4 / 1e6, 2))
    threads = [threading.Thread(target=fake_run, args=j) for j in jobs]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


if __name__ == "__main__":
    main()
