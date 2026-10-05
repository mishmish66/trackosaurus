"""trex (trackosaurus exp): log runs to directories; explore them with `trex serve` or the trex CLI.

<video src="media/tour.mp4" poster="media/screenshot.png" controls width="100%"></video>

Install the trex command, or the logging API (`trex.writer`) in a training project:

    uv tool install git+https://github.com/mishmish66/trackosaurus
    uv add git+https://github.com/mishmish66/trackosaurus

Log:

    run = trex.init("runs/sweep/lr3e-4/seed0", config={"lr": 3e-4})
    run.log({"train/loss": loss}, step=step)
    run.finish()

Explore in a browser or a terminal (`trex --help`); `trex.node` runs trex as a service, crawls other machines over
ssh and pulls what other trex hold:

    trex serve runs      # http://127.0.0.1:13898
    trex tree runs

The UI's filter box and `trex ls -w` take a name search or a SQL WHERE clause (`trex.where`):

    trex ls runs -w "lr = 0.001 and seed in (0, 1)"

Its group-by box takes fields, `,` within a level and ` / ` between levels (`algo, env / lr`). `run~n` is the
directory n levels above a run, `run~1` (the default) its own; `run` puts each run alone. A group of several runs is
one line with a band; a folder's `trex_info.json` sets its default (`trex.folder_info(path, trex={"group_by": ...})`).
"""

from .writer import FinalState, ImageInput, MetricValue, Metrics, Run, VideoInput, folder_info, init

__all__ = ["FinalState", "ImageInput", "MetricValue", "Metrics", "Run", "VideoInput", "folder_info", "init"]
