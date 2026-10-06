import getpass
import grp
import io
import json
import math
import os
import plistlib
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from _pytest.capture import CaptureResult

import trex
from trex import query as Q
from trex import update
from trex.cli import RUNIT_AS_USER, field_columns, main, where_test
from trex.format import connect_ro

from helpers import committed_rows, wait_for


@pytest.fixture
def runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """sweep/lr{0.001,0.01}/seed{0,1,2}: loss falls faster with larger lr; seed2 has info notes."""
    root = tmp_path / "runs"
    for lr in (0.001, 0.01):
        for seed in range(3):
            r = trex.init(root / "sweep" / f"lr{lr}" / f"seed{seed}", config={"lr": lr, "seed": seed, "opt": {"name": "adam"}},
                          info={"notes": "flaky"} if seed == 2 else None, commit_interval=0.05)
            for step in range(100):
                r.log({"loss": 1.0 / (1 + lr * step * (seed + 1)), "acc": step / 100}, step=step)
            r.log_image("img", b"\x89PNG\r\n\x1a\n" + bytes(8), step=50)
            r.log_image("img", b"\x89PNG\r\n\x1a\n" + bytes(9), step=99)
            r.finish()
    trex.folder_info(root / "sweep", question="does lr matter?")
    monkeypatch.setenv("TREX_CACHE", str(tmp_path / "cache"))
    return root


def run_json(capsys: pytest.CaptureFixture[str], *argv: str | Path) -> Any:
    main([*map(str, argv), "--json"])
    return json.loads(capsys.readouterr().out)


def test_every_run_is_visible_outside_the_ui_and_runs_filter_by_state(runs: Path, capsys: pytest.CaptureFixture[str]) -> None:
    every = run_json(capsys, "ls", runs)
    assert len(run_json(capsys, "ls", runs, "-w", "visible = true")) == len(every) > 0
    assert run_json(capsys, "ls", runs, "-w", "visible = false") == []
    states = {r["state"] for r in every}
    assert [len(run_json(capsys, "ls", runs, "-w", f"state = {s}")) for s in states] == [sum(r["state"] == s for r in every) for s in states]


def test_ls_filters_sorts_and_limits(runs: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = run_json(capsys, "ls", runs, "-w", "config.lr = 0.01", "-w", "summary.loss < 0.4", "--sort", "summary.loss:desc")
    assert [r["path"] for r in out] == ["sweep/lr0.01/seed1", "sweep/lr0.01/seed2"]
    assert all(r["config"]["lr"] == 0.01 and r["summary"]["loss"] < 0.6 for r in out)
    out = run_json(capsys, "ls", runs, "--sort=-summary.loss", "-n", "1")
    assert out[0]["path"] == "sweep/lr0.001/seed0"
    assert [r["path"] for r in run_json(capsys, "ls", runs, "-w", "info.notes is not null")] == ["sweep/lr0.001/seed2", "sweep/lr0.01/seed2"]
    assert len(run_json(capsys, "ls", runs, "-w", "path ~ 'seed[01]$' and config.opt/name = adam")) == 4
    assert len(run_json(capsys, "ls", runs, "-w", "seed0")) == 2
    assert run_json(capsys, "ls", runs / "sweep" / "lr0.01")[0]["path"] == "seed0"


def test_ls_paths_pipe_into_series_and_diff(runs: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    main(["ls", str(runs), "-w", "config.seed=0", "--paths"])
    paths = capsys.readouterr().out
    assert len(paths.split()) == 2
    monkeypatch.setattr("sys.stdin", io.StringIO(paths))
    rows = run_json(capsys, "series", "-", "-k", "loss", "--last", "1")
    assert [(r["run"], r["step"]) for r in rows] == [("seed0", 99.0), ("seed0", 99.0)]
    monkeypatch.setattr("sys.stdin", io.StringIO(paths))
    assert run_json(capsys, "diff", "-") == [{"key": "lr", "lr0.001/seed0": 0.001, "lr0.01/seed0": 0.01}]


def test_groups_report_center_and_order_statistic_ci(runs: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = run_json(capsys, "groups", runs / "sweep", "-g", "run~1", "-m", "loss")
    assert [g["run~1"] for g in out] == ["lr0.001", "lr0.01"]
    for g in out:
        lr = float(g["run~1"][2:])
        vals = sorted(1.0 / (1 + lr * 99 * (s + 1)) for s in range(3))
        st = g["loss:stats"]
        assert g["runs"] == 3 and st["median"] == pytest.approx(statistics.median(vals))
        assert (st["ci_lo"], st["ci_hi"]) == pytest.approx((vals[0], vals[-1]))
        assert st["ci_coverage"] == pytest.approx(0.75)
    at = run_json(capsys, "groups", runs, "-g", "config.lr", "-m", "loss", "--at", "10", "--center", "mean")
    g = next(g for g in at if g["config.lr"] == 0.01)
    vals = [1.0 / (1 + 0.01 * 10 * (s + 1)) for s in range(3)]
    assert g["loss"] == pytest.approx(statistics.mean(vals))
    assert g["loss:stats"]["ci_hi"] - g["loss"] == pytest.approx(4.303 * statistics.stdev(vals) / math.sqrt(3))


def test_keys_lists_metrics_and_media(runs: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = run_json(capsys, "keys", runs)
    assert {(k["key"], k["kind"], k["runs"]) for k in out} == {("loss", "metric", 6), ("acc", "metric", 6), ("img", "image", 6)}


def test_show_tail_media_and_tree(runs: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run = runs / "sweep" / "lr0.01" / "seed2"
    s = run_json(capsys, "show", run)
    assert s["info"] == {"notes": "flaky"} and s["keys"]["loss"]["points"] == 100 and s["rows"] == 100
    assert s["folders"][-1]["info"] == {"question": "does lr matter?"}
    main(["tail", str(run), "-n", "2", "-k", "acc", "--format", "jsonl"])
    assert [json.loads(l)["seq"] for l in capsys.readouterr().out.split()] == [98, 99]
    media = run_json(capsys, "media", run, "--latest")
    assert [(m["key"], m["step"]) for m in media] == [("img", 99.0)] and media[0]["file"].endswith(".png")
    tree = run_json(capsys, "tree", runs)
    (sweep,) = tree["dirs"]
    assert (sweep["runs"], sweep["states"], sweep["info"]) == (6, {"finished": 6}, {"question": "does lr matter?"})


def test_series_downsampling_and_smoothing_match_query_helpers(runs: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run = runs / "sweep" / "lr0.01" / "seed0"
    rows = run_json(capsys, "series", run, "-k", "loss", "--points", "5", "--smooth", "0.9")
    assert [r["step"] for r in rows] == [0, 25, 50, 74, 99]
    xs = list(range(100))
    ys = [1.0 / (1 + 0.01 * s) for s in xs]
    sm = Q.twema(xs, ys, 0.9, Q.smooth_scale(99))
    assert [r["smoothed"] for r in rows] == pytest.approx([sm[i] for i in (0, 25, 50, 74, 99)])


def test_table_output_is_aligned_text(runs: Path, capsys: pytest.CaptureFixture[str]) -> None:
    main(["ls", str(runs), "-c", "config.lr"])
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].split() == ["path", "state", "step", "runtime", "config.lr"] and len(lines) == 7


def test_refuses_to_index_home(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["ls", "~"])


BLANK = Q.Record(path="", name="", parent="", state="finished", step=None, runtime=None, rows=0, media=0, created=None,
                 updated=None, tags=[], dir="", visible=True, config={}, info={}, summary={})


@pytest.mark.parametrize("clause,rec,want", [
    ("config.lr >= 0.01", replace(BLANK, config={"lr": 0.01}), True),
    ("config.lr > 0.01", replace(BLANK, config={"lr": 0.01}), False),
    ("config.lr != 0.01", BLANK, True),
    ("config.lr = 0.01", BLANK, False),
    ("config.flag = true", replace(BLANK, config={"flag": True}), True),
    ("config.name ~ '^ad'", replace(BLANK, config={"name": "adam"}), True),
    ("config.name !~ '^ad'", replace(BLANK, config={"name": "adam"}), False),
    ("summary.loss < 1", replace(BLANK, summary={"loss": "nan"}), False),
    ("info.notes is null", BLANK, True),
    ("state = running", replace(BLANK, state="running"), True),
])
def test_where_clauses_read_run_fields(clause: str, rec: Q.Record, want: bool) -> None:
    assert where_test(clause)(lambda f: Q.get(rec, f)) is want


def test_median_ci_rank_matches_exact_binomial_coverage() -> None:
    for n in range(1, 60):
        k = Q.median_ci_rank(n)
        cov = 1 - 2 * sum(math.comb(n, j) for j in range(k)) / 2**n
        better = 1 - 2 * sum(math.comb(n, j) for j in range(k + 1)) / 2**n
        assert (cov >= 0.95 or k == 1) and not (better >= 0.95 and k + 1 <= n // 2)


type Mixed = tuple[Path, list[tuple[float, dict[str, float]]]]


@pytest.fixture
def mixed(tmp_path: Path) -> Mixed:
    """One run written in three sessions (one commit each): every step logs a train row {loss, acc};
    even steps 0..8 also log an eval row {loss, eval/score} at the same step. Runtime equals step."""
    path = tmp_path / "mixed"
    expected: list[tuple[float, dict[str, float]]] = []
    for steps in (range(0, 5), range(5, 10), range(10, 11)):
        r = trex.init(path, commit_interval=60)
        for s in steps:
            r.log({"loss": float(s), "acc": s / 10}, step=s, timestamp=r.created + s)
            expected.append((float(s), {"loss": float(s), "acc": s / 10}))
            if s % 2 == 0 and s < 10:
                r.log({"loss": -float(s), "eval/score": 100.0 + s}, step=s, timestamp=r.created + s)
                expected.append((float(s), {"loss": -float(s), "eval/score": 100.0 + s}))
        r.finish()
    c = connect_ro(path)
    assert c.execute("SELECT count(*) FROM rowmeta").fetchone()[0] == 3
    c.close()
    return path, expected


def test_series_returns_each_keys_points_in_row_order_across_commits_of_mixed_rows(mixed: Mixed, capsys: pytest.CaptureFixture[str]) -> None:
    path, expected = mixed
    rows = run_json(capsys, "series", path, "-k", "loss", "-k", "eval/score", "-k", "absent")
    for key in ("loss", "eval/score"):
        want = [(s, d[key]) for s, d in expected if key in d]
        assert [(r["step"], r["value"]) for r in rows if r["key"] == key] == want
    assert not [r for r in rows if r["key"] == "absent"]
    xs, ys = Q.series(path, {"loss"}.__contains__, x="runtime")["loss"]
    assert list(xs) == pytest.approx([s for s, _ in expected])
    assert list(ys) == [d["loss"] for _, d in expected]


def test_tail_last_rows_span_a_commit_boundary(mixed: Mixed, capsys: pytest.CaptureFixture[str]) -> None:
    path, expected = mixed
    main(["tail", str(path), "-n", "3", "--format", "jsonl"])
    got = [json.loads(l) for l in capsys.readouterr().out.split()]
    n = len(expected)
    assert [r["seq"] for r in got] == [n - 3, n - 2, n - 1]
    for r, (step, d) in zip(got, expected[-3:]):
        assert r["step"] == step and r["runtime"] == pytest.approx(step)
        assert {k: v for k, v in r.items() if k not in ("seq", "step", "runtime")} == d
    main(["tail", str(path), "-n", "3", "-k", "acc", "--format", "jsonl"])
    got = [json.loads(l) for l in capsys.readouterr().out.split()]
    assert [r["seq"] for r in got] == [n - 3, n - 2, n - 1]
    assert [r.get("acc") for r in got] == [d.get("acc") for _, d in expected[-3:]]


def test_read_rows_from_mid_commit_start_matches_logged_rows(mixed: Mixed) -> None:
    path, expected = mixed
    rows = Q.read_rows(path, start=3)
    assert [r.seq for r in rows] == list(range(3, len(expected)))
    assert [(r.step, r.values) for r in rows] == expected[3:]


def test_show_reports_point_counts_and_last_values_of_mixed_rows(mixed: Mixed, capsys: pytest.CaptureFixture[str]) -> None:
    path, expected = mixed
    s = run_json(capsys, "show", path)
    assert (s["rows"], s["step"]) == (len(expected), 10.0)
    assert s["runtime"] == pytest.approx(10.0)
    assert s["keys"] == {
        "acc": {"points": 11, "first_step": 0.0, "last_step": 10.0, "last": 1.0},
        "eval/score": {"points": 5, "first_step": 0.0, "last_step": 8.0, "last": 108.0},
        "loss": {"points": 16, "first_step": 0.0, "last_step": 10.0, "last": 10.0},
    }


def text_of(capsys: pytest.CaptureFixture[str], *argv: str | Path) -> CaptureResult[str]:
    main([*map(str, argv)])
    return capsys.readouterr()


@pytest.mark.parametrize("argv", [
    ("ls", "{runs}", "-c", "config.lr"), ("keys", "{runs}"), ("tree", "{runs}", "--runs"), ("show", "{run}"),
    ("groups", "{runs}", "-g", "config.lr", "-m", "loss"), ("series", "{run}", "-k", "loss", "--last", "3"),
    ("tail", "{run}", "-n", "3"), ("media", "{run}"), ("diff", "{run}", "{other}"), ("index", "{runs}"),
])
def test_piped_text_output_has_no_color_codes_or_trailing_spaces(runs: Path, capsys: pytest.CaptureFixture[str],
                                                                 argv: tuple[str, ...]) -> None:
    run, other = runs / "sweep" / "lr0.01" / "seed0", runs / "sweep" / "lr0.001" / "seed0"
    out = text_of(capsys, *(a.format(runs=runs, run=run, other=other) for a in argv)).out
    assert out and "\x1b[" not in out
    assert not [line for line in out.splitlines() if line != line.rstrip()]


def test_groups_table_goes_to_stdout_and_the_ci_note_to_stderr(runs: Path, capsys: pytest.CaptureFixture[str]) -> None:
    res = text_of(capsys, "groups", runs, "-g", "config.lr", "-m", "loss", "--reduce", "max")
    header, *rows = res.out.splitlines()
    assert header.split() == ["config.lr", "runs", "loss", "loss:ci", "loss:range", "loss:n"] and len(rows) == 2
    assert "max over the run" in res.err and "n=3: 75.0%" in res.err
    assert [r.split()[2] for r in rows] == ["1", "1"]


def test_groups_without_metrics_count_runs_per_group(runs: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = run_json(capsys, "groups", runs, "-g", "run~2")
    assert [(g["run~2"], g["runs"]) for g in out] == [("sweep", 6)]


def test_groups_take_the_ui_group_by_expression(runs: Path, capsys: pytest.CaptureFixture[str]) -> None:
    nested = run_json(capsys, "groups", runs, "-g", "run~2 / run~1")
    assert [(g["run~2"], g["run~1"], g["runs"]) for g in nested] == [("sweep", "sweep/lr0.001", 3), ("sweep", "sweep/lr0.01", 3)]
    alone = run_json(capsys, "groups", runs / "sweep", "-g", "run, config.lr")
    assert sorted(g["run"] for g in alone) == sorted(f"lr{lr}/seed{s}" for lr in ("0.001", "0.01") for s in range(3))
    assert all(g["runs"] == 1 and g["config.lr"] == float(g["run"][2:].split("/")[0]) for g in alone)


def test_groups_default_to_the_declared_group_by_else_the_run_directory(runs: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert [(g["run~1"], g["runs"]) for g in run_json(capsys, "groups", runs / "sweep")] == [("lr0.001", 3), ("lr0.01", 3)]
    trex.folder_info(runs / "sweep", trex={"group_by": "config.lr"})
    assert [(g["config.lr"], g["runs"]) for g in run_json(capsys, "groups", runs / "sweep" / "lr0.01")] == [(0.01, 3)]


def test_show_lists_media_and_folder_notes(runs: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = text_of(capsys, "show", runs / "sweep" / "lr0.01" / "seed2").out
    assert "img [image] 2 items, steps 50..99" in out
    assert "folder info" in out and "question: does lr matter?" in out


def test_diff_compares_info_and_lists_equal_keys_with_all(runs: Path, capsys: pytest.CaptureFixture[str]) -> None:
    a, b = runs / "sweep" / "lr0.01" / "seed2", runs / "sweep" / "lr0.01" / "seed0"
    assert run_json(capsys, "diff", a, b, "--info") == [{"key": "notes", "seed2": "flaky", "seed0": None}]
    assert {r["key"] for r in run_json(capsys, "diff", a, b, "--all")} == {"lr", "seed", "opt/name"}


def test_tail_follow_streams_new_rows_until_the_run_ends(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run = trex.init(tmp_path / "live", commit_interval=0.02)
    run.log({"loss": 0.0}, step=0)
    assert wait_for(lambda: committed_rows(tmp_path / "live") == 1)

    def write() -> None:
        for s in range(1, 4):
            time.sleep(0.1)
            run.log({"loss": float(s)}, step=s)
        time.sleep(0.1)
        run.finish()

    t = threading.Thread(target=write)
    t.start()
    res = text_of(capsys, "tail", tmp_path / "live", "-n", "10", "-f", "--interval", "0.05", "--timeout", "20")
    t.join()
    header, *lines = res.out.splitlines()
    table = [l.split()[1] for l in lines if "=" not in l]
    followed = [l.split()[1].removeprefix("step=") for l in lines if "=" in l]
    assert header.split() == ["seq", "step", "runtime", "loss"] and table[0] == "0" and followed
    assert table + followed == ["0", "1", "2", "3"]
    assert "is finished" in res.err


def test_tail_follow_of_a_finished_run_returns(runs: Path, capsys: pytest.CaptureFixture[str]) -> None:
    res = text_of(capsys, "tail", runs / "sweep" / "lr0.01" / "seed0", "-n", "0", "-f", "--interval", "0.01")
    assert res.out == "" and "is finished" in res.err


def test_index_reports_run_counts_by_state(runs: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = run_json(capsys, "index", runs)
    assert (out[0]["runs"], out[0]["states"]) == (6, {"finished": 6})


def test_short_sort_flag_takes_a_descending_field(runs: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = run_json(capsys, "ls", runs, "-s", "-summary.loss", "-n", "1")
    assert out[0]["path"] == "sweep/lr0.001/seed0"


@pytest.mark.parametrize("argv,message", [
    (("ls", "{runs}", "-w", "config.lr ="), "trex ls: bad filter"),
    (("ls", "{runs}", "-w", "name ~ '('"), "trex ls: bad filter"),
    (("series", "{run}"), "give --key"),
    (("series", "{runs}", "-k", "loss"), "not a run directory"),
    (("diff", "{run}"), "at least two runs"),
    (("ls", "{runs}/missing"), "does not exist"),
    (("ls", "{run}", "--root", "{other}"), "is not under --root"),
])
def test_bad_input_exits_with_a_message_naming_the_command(runs: Path, capsys: pytest.CaptureFixture[str], argv: tuple[str, ...],
                                                           message: str) -> None:
    run, other = runs / "sweep" / "lr0.01" / "seed0", runs / "sweep" / "lr0.001"
    with pytest.raises(SystemExit) as e:
        main([a.format(runs=runs, run=run, other=other) for a in argv])
    assert e.value.code != 0
    err = capsys.readouterr().err + str(e.value.code)
    assert message in err


def test_unknown_options_are_usage_errors(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as e:
        main(["ls", "--bogus"])
    assert e.value.code == 2 and "--bogus" in re.sub(r"\x1b\[[0-9;]*m", "", capsys.readouterr().err)


RECORD = replace(BLANK, path="a/r", name="r", state="finished", config={"lr": 0.1, "both": "config", "opt/name": "adam"},
                 summary={"loss": "nan", "both": 2.0, "acc": 0.5}, info={"git": {"sha": "abc"}, "a.b": 1})


@pytest.mark.parametrize("field,want", [
    ("lr", 0.1), ("both", "config"), ("acc", 0.5), ("missing", None), ("opt/name", "adam"),
    ("config.lr", 0.1), ("c.lr", 0.1), ("summary.acc", 0.5), ("s.both", 2.0), ("m.missing", None),
    ("info.git.sha", "abc"), ("info.a.b", 1), ("info.git.nope", None), ("state", "finished"),
])
def test_fields_resolve_config_first_then_summary_with_prefixes_and_nested_info(field: str, want: float | str | None) -> None:
    assert Q.get(RECORD, field) == want


def test_summary_markers_sort_as_numbers_and_nan_as_missing() -> None:
    recs = [replace(RECORD, summary={"loss": v}) for v in ("inf", "nan", 1.0, "-inf")]
    assert [r.summary["loss"] for r in Q.sort_records(recs, "summary.loss")] == ["-inf", 1.0, "inf", "nan"]


@pytest.mark.parametrize("clause", ["summary.note = best", "note = best", "done = true", "summary.done = true"])
def test_text_and_boolean_summaries_filter_as_in_the_ui(clause: str) -> None:
    rec = replace(RECORD, summary={"note": "best", "done": True})
    assert where_test(clause)(lambda f: Q.get(rec, f))


@pytest.mark.parametrize("clause,want", [("name > q", True), ("name < q", False), ("name >= r", True), ("config.both < d", True)])
def test_text_values_compare_lexically(clause: str, want: bool) -> None:
    assert where_test(clause)(lambda f: Q.get(RECORD, f)) is want


def test_statistics_and_reductions_of_series_without_finite_values() -> None:
    assert Q.stats([None, math.nan]) == Q.Stats(n=0)
    assert Q.reduce([0.0, 1.0], [math.nan, math.inf], "mean") is None
    assert Q.reduce([5.0, 6.0], [1.0, 2.0], "last", at=4.0) is None
    assert Q.twema([0.0, 1.0, 2.0], [1.0, math.nan, None], 0.9, 1.0)[1:] == pytest.approx([math.nan, None], nan_ok=True)


def test_folder_notes_run_from_the_root_down_and_skip_unreadable_files(tmp_path: Path) -> None:
    run = tmp_path / "a" / "b" / "r"
    run.mkdir(parents=True)
    (tmp_path / "trex_info.json").write_text('{"level": 0}')
    (tmp_path / "a" / "trex_info.json").write_text("{broken")
    (tmp_path / "a" / "b" / "trex_info.json").write_text('{"level": 2}')
    assert [i for _, i in Q.folder_infos(run, tmp_path)] == [{"level": 0}, {"level": 2}]
    assert [i for _, i in Q.folder_infos(run, tmp_path / "a" / "b")] == [{"level": 2}]


def test_systemd_unit_runs_this_trex_and_restarts_it_after_an_update(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    main(["systemd-unit", "--host", "127.0.0.1", "--port", "9000", "--source", "git+https://example.org/trex"])
    unit = capsys.readouterr().out
    lines = set(unit.splitlines())
    assert f"ExecStart={shlex.join([sys.executable, '-m', 'trex', 'serve', '--host', '127.0.0.1', '--port', '9000'])}" in lines
    assert {"Environment=TREX_SOURCE=git+https://example.org/trex", "SuccessExitStatus=75", "RestartForceExitStatus=75",
            "Restart=on-failure", "WantedBy=default.target"} <= lines
    if shutil.which("systemd-analyze"):
        (tmp_path / "trex.service").write_text(unit)
        res = subprocess.run(["systemd-analyze", "verify", str(tmp_path / "trex.service")], capture_output=True, text=True)
        assert res.returncode == 0, res.stderr


def test_systemd_unit_updates_from_the_trex_repository_by_default(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TREX_SOURCE", raising=False)
    main(["systemd-unit"])
    assert f"Environment=TREX_SOURCE={update.DEFAULT_SOURCE}" in capsys.readouterr().out.split("[Unit]")[1].splitlines()


def test_systemd_unit_with_an_empty_source_has_no_update_source(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TREX_SOURCE", "git+https://example.org/trex")
    main(["systemd-unit", "--source", ""])
    assert "TREX_SOURCE" not in capsys.readouterr().out.split("[Unit]")[1]


def test_launchd_plist_runs_this_trex_and_restarts_it_after_an_update(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    main(["launchd-plist", "--host", "127.0.0.1", "--port", "9000", "--allow-host", "box.tailnet.ts.net",
          "--cache", str(tmp_path / "cache"), "--source", "git+https://example.org/trex"])
    agent = plistlib.loads(capsys.readouterr().out.encode())
    assert agent["ProgramArguments"] == [sys.executable, "-m", "trex", "serve", "--host", "127.0.0.1",
                                         "--allow-host", "box.tailnet.ts.net", "--port", "9000"]
    env = agent["EnvironmentVariables"]
    assert {"TREX_SOURCE": "git+https://example.org/trex", "TREX_CACHE": str((tmp_path / "cache").resolve())}.items() <= env.items()
    assert update.service(env) and str(Path(sys.executable).parent) in env["PATH"].split(":")
    assert agent["RunAtLoad"] and agent["KeepAlive"] == {"SuccessfulExit": False}
    assert Path(agent["StandardOutPath"]).is_absolute() and agent["StandardErrorPath"] == agent["StandardOutPath"]
    if shutil.which("plutil"):
        (tmp_path / "trex.plist").write_bytes(plistlib.dumps(agent))
        assert subprocess.run(["plutil", "-lint", str(tmp_path / "trex.plist")], capture_output=True).returncode == 0


def test_launchd_plist_updates_from_the_trex_repository_by_default(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TREX_SOURCE", raising=False)
    main(["launchd-plist"])
    assert plistlib.loads(capsys.readouterr().out.encode())["EnvironmentVariables"]["TREX_SOURCE"] == update.DEFAULT_SOURCE


def test_launchd_plist_with_an_empty_source_has_no_update_source(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TREX_SOURCE", "git+https://example.org/trex")
    main(["launchd-plist", "--source", ""])
    assert "TREX_SOURCE" not in plistlib.loads(capsys.readouterr().out.encode())["EnvironmentVariables"]


def runit_env(script: str, tmp_path: Path) -> dict[str, str]:
    """The environment runit run script `script` gives the trex it runs: the script run with its last line, the exec,
    replaced by env."""
    probe = tmp_path / "probe"
    probe.write_text("".join(f"{line}\n" for line in [*script.splitlines()[:-1], "exec env"]))
    out = subprocess.run(["sh", str(probe)], env={}, capture_output=True, text=True, check=True).stdout
    return dict(line.split("=", 1) for line in out.splitlines() if re.match(r"[A-Za-z_]\w*=", line))


def test_runit_service_runs_this_trex_as_you_and_runsv_starts_it_again_after_an_update(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    main(["runit-service", "--host", "127.0.0.1", "--port", "9000", "--allow-host", "box.tailnet.ts.net",
          "--cache", str(tmp_path / "my cache"), "--source", "git+https://example.org/trex"])
    script = capsys.readouterr().out
    lines = script.splitlines()
    assert lines[0] == "#!/bin/sh" and "exec 2>&1" in lines
    assert lines[-1] == f"exec {RUNIT_AS_USER} " + shlex.join([sys.executable, "-m", "trex", "serve", "--host", "127.0.0.1",
                                                              "--allow-host", "box.tailnet.ts.net", "--port", "9000"])
    (tmp_path / "run").write_text(script)
    assert subprocess.run(["sh", "-n", str(tmp_path / "run")]).returncode == 0
    env = runit_env(script, tmp_path)
    assert {"TREX_SOURCE": "git+https://example.org/trex", "TREX_CACHE": str((tmp_path / "my cache").resolve()),
            "HOME": str(Path.home()), "USER": getpass.getuser()}.items() <= env.items()
    assert update.service(env) and str(Path(sys.executable).parent) in env["PATH"].split(":")
    as_user = subprocess.run(["sh", "-c", f"USER={getpass.getuser()}; echo {RUNIT_AS_USER.removeprefix('chpst -u ')}"],
                             capture_output=True, text=True, check=True).stdout.strip()
    assert as_user.split(":")[:2] == [getpass.getuser(), grp.getgrgid(os.getgid()).gr_name]


def test_runit_service_updates_from_the_trex_repository_by_default(capsys: pytest.CaptureFixture[str], tmp_path: Path,
                                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TREX_SOURCE", raising=False)
    main(["runit-service"])
    assert runit_env(capsys.readouterr().out, tmp_path)["TREX_SOURCE"] == update.DEFAULT_SOURCE


def test_runit_service_with_an_empty_source_has_no_update_source(capsys: pytest.CaptureFixture[str], tmp_path: Path,
                                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TREX_SOURCE", "git+https://example.org/trex")
    main(["runit-service", "--source", ""])
    assert "TREX_SOURCE" not in runit_env(capsys.readouterr().out, tmp_path)


def test_version_names_the_installed_trex(capsys: pytest.CaptureFixture[str]) -> None:
    main(["--version"])
    out = capsys.readouterr().out
    assert re.fullmatch(r"trex \d+\.\d+\.\d+\S*( \([0-9a-f]{12}\))?\n", out)


def test_systemd_unit_passes_allowed_host_names_to_trex(capsys: pytest.CaptureFixture[str]) -> None:
    main(["systemd-unit", "--allow-host", "box.tailnet.ts.net"])
    assert "--allow-host box.tailnet.ts.net" in capsys.readouterr().out.split("ExecStart=")[1].splitlines()[0]


def test_table_columns_include_dotted_fields_a_where_clause_or_sort_reads() -> None:
    assert field_columns(["config.lr = 1 and state = running or summary.loss < 2"], "-summary.acc,path") == [
        "config.lr", "summary.loss", "summary.acc"]


def test_iqm_is_the_mean_of_the_middle_half_with_yuens_ci() -> None:
    st = Q.stats([1, 2, 3, 4, 5, 6, 7, 8, 100, None], center="iqm")
    half = 2.776 * math.sqrt(26 / (5 * 4))  # winsorized deviations 26, 5 kept, t(0.975, 4)
    assert st.iqm == 5 and st.ci_lo == pytest.approx(5 - half) and st.ci_hi == pytest.approx(5 + half)
    assert Q.stats([3.0], center="iqm").iqm == 3.0 and Q.stats([3.0], center="iqm").ci_lo is None
    assert Q.stats([1, 1, 1, 9], center="iqm").iqm == 1
