"""trex (trackosaurus exp): log runs to directories; explore them with `trex serve` or the trex CLI.

<video src="media/tour.mp4" poster="media/screenshot.png" controls width="100%"></video>

Install the trex command, or the logging API (`trex.writer`) in a training project:

    uv tool install git+https://github.com/mishmish66/trackosaurus
    uv add git+https://github.com/mishmish66/trackosaurus

Log:

    run = trex.init("runs/sweep/lr3e-4/seed0", config={"lr": 3e-4})
    run.log({"train/loss": loss}, step=step)
    run.finish()

Explore in a browser or a terminal (`trex --help`); `trex.daemon` serves many directories under systemd or launchd:

    trex serve runs      # http://127.0.0.1:13898
    trex tree runs
    trex daemon

The UI's filter box and `trex ls -w` take a name search or a SQL WHERE clause (`trex.where`):

    trex ls runs -w "lr = 0.001 and seed in (0, 1)"
"""

from .writer import FinalState, ImageInput, MetricValue, Metrics, Run, VideoInput, folder_info, init

__all__ = ["FinalState", "ImageInput", "MetricValue", "Metrics", "Run", "VideoInput", "folder_info", "init"]
