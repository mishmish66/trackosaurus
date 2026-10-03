# trex — trackosaurus exp

A fast, file-based experiment tracker: runs are directories, `trex serve` shows them live. [Docs](https://mishmish66.github.io/trackosaurus/)

https://github.com/user-attachments/assets/f8d20d02-22fd-407d-8275-eee554718186

## Install

```bash
uv tool install git+https://github.com/mishmish66/trackosaurus   # the trex command
uv add git+https://github.com/mishmish66/trackosaurus            # the logging API, in a training project
```

## Log

```python
import trex

run = trex.init("runs/sweep/lr3e-4/seed0", config={"lr": 3e-4})
run.log({"train/loss": loss}, step=step)        # also log_image, log_video, log_html, summary
run.finish()                                    # also at exit; an uncaught exception marks the run failed
```

- The directory is the run: reopening it resumes, and a copy of it is a readable run.
- Rows commit every second; a run silent for 5 minutes shows as crashed.
- `trex.folder_info(path, trex={"group_by": ["subfolder"]})` sets a folder's notes and default grouping.

## Explore

- `trex serve runs` opens the UI at http://127.0.0.1:13898; it stays fast with 10,000 runs.
- Group by folder or config key: median, mean or IQM lines with 95% CI, IQR or min/max bands.
- Drag to zoom, hover for values, Shift to pin them and scroll; ⚙ sets smoothing and axes per metric.
- Filter by name or a SQL WHERE clause (`lr = 0.001 and seed in (0, 1)`), in the UI or `trex ls -w`; `trex --help` lists the rest.

## Daemon

- `trex daemon` tracks many directories on one port, all shown at `/`; `trex serve DIR` adds one.
- Click `trex` (top left) to switch to or manage tracked directories and workspaces.
- A workspace merges chosen directories, local or remote, into one tree; group by `dir` to compare them.
- Add `host:path` for another machine: trex runs there over ssh with `uvx`, so it needs only uv.
- On a cluster, use a data-transfer node (`xfer:/scratch/me/runs`); jobs on any node stream live.
- `trex systemd-unit` or `trex launchd-plist` (macOS) gives a service with an update button in the `trex` panel.
- `trex compact DIR` shrinks runs written before compaction: their commits merged, the rows unchanged.

## Limits

- No auth: listen on localhost or a private network. Other websites cannot reach or change it.
- A run is its path: moving a run directory makes it a new run.
- Development notes are in [AGENTS.md](AGENTS.md); docs build with `uv run python docs/build.py`.

## License

[VibeCoded AI-Slop License v1.0](LICENSE).
