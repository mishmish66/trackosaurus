import json
import sqlite3
import subprocess
import sys
import textwrap
import time

import pytest

import trex
from trex import chunks, compact
from trex.cli import main
from trex.format import DB, connect_ro, connect_rw


def readback(d):
    """Everything readers see of a run: meta, metric names, media, rows and every metric's points (NaN-safe)."""
    c = connect_ro(d)
    try:
        tables = {t: sorted(c.execute(f"SELECT * FROM {t}").fetchall()) for t in ("meta", "keys", "media")}
        rows = [(r.seq, r.step, r.t, sorted((k, repr(v)) for k, v in r.values.items())) for r in chunks.rows(c)]
        points = {name: [a.tobytes() for a in chunks.metric(c, kid)] for kid, name in chunks.key_names(c).items()}
        return tables, rows, points
    finally:
        c.close()


def commits(d):
    c = connect_ro(d)
    try:
        return c.execute("SELECT count(*) FROM rowmeta").fetchone()[0]
    finally:
        c.close()


@pytest.fixture
def uncompacted(monkeypatch):
    monkeypatch.setattr(trex.writer, "merge_plan", lambda tail: None)


def small_commit_run(d, n=200):
    run = trex.init(d, config={"lr": 0.1}, commit_interval=0.001)
    for i in range(n):
        run.log({"loss": i + 0.5, "nan": float("nan"), **({"eval": -float(i)} if i % 7 == 0 else {})}, step=i)
        if i % 50 == 0:
            run.log_html("report", f"<b>{i}</b>", step=i)
        time.sleep(0.002)
    run.summary(best=0.5)
    run.finish()


def test_compacting_a_finished_run_keeps_everything_readers_see_in_few_commits(tmp_path, uncompacted):
    d = tmp_path / "r"
    small_commit_run(d)
    before, many = readback(d), commits(d)
    r = compact.compact(d)
    assert readback(d) == before and commits(d) == r.commits_after == 1 and r.commits_before == many > 50
    assert r.bytes_after < r.bytes_before and sorted(p.name for p in d.iterdir()) == ["media", DB]


def test_a_run_another_process_has_open_is_refused_and_left_as_it_was(tmp_path, uncompacted):
    d = tmp_path / "r"
    small_commit_run(d, 30)
    before, many = readback(d), commits(d)
    holder = subprocess.Popen([sys.executable, "-c", textwrap.dedent(f"""
        import time
        from trex.format import connect_rw
        c = connect_rw({str(d)!r})
        c.execute("SELECT 1 FROM meta").fetchall()
        print("open", flush=True)
        time.sleep(30)
    """)], stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout is not None and holder.stdout.readline().strip() == "open"
        with pytest.raises(compact.InUse):
            compact.compact(d)
    finally:
        holder.kill()
        holder.wait()
    assert readback(d) == before and commits(d) == many and not (d / compact.TMP).exists()


def test_a_killed_writers_run_is_compacted_with_the_rows_in_its_wal(tmp_path):
    d = tmp_path / "r"
    code = textwrap.dedent(f"""
        import os, signal, time, trex
        trex.writer.merge_plan = lambda tail: None
        run = trex.init({str(d)!r}, commit_interval=0.001)
        for i in range(300):
            run.log({{"x": float(i)}}, step=i)
            time.sleep(0.002)
        time.sleep(0.05)
        os.kill(os.getpid(), signal.SIGKILL)
    """)
    assert subprocess.run([sys.executable, "-c", code]).returncode == -9
    assert (d / (DB + "-wal")).exists()
    before = readback(d)
    compact.compact(d)
    assert readback(d) == before and [r[3] for r in before[1]] == [[("x", repr(float(i)))] for i in range(300)]
    assert commits(d) == 1 and not (d / (DB + "-wal")).exists()


def test_a_compaction_that_fails_partway_leaves_the_run_as_it_was(tmp_path, uncompacted, monkeypatch):
    d = tmp_path / "r"
    small_commit_run(d, 60)
    before, many, size = readback(d), commits(d), (d / DB).stat().st_size
    calls = [0]
    merge = chunks.merge_chunks

    def fail_on_the_second_metric(*args):
        calls[0] += 1
        if calls[0] == 2:
            raise OSError("disk full")
        return merge(*args)

    monkeypatch.setattr(chunks, "merge_chunks", fail_on_the_second_metric)
    with pytest.raises(OSError, match="disk full"):
        compact.compact(d)
    assert readback(d) == before and commits(d) == many and (d / DB).stat().st_size == size
    assert not (d / compact.TMP).exists()


def test_a_compacted_run_takes_new_rows_and_merges_them(tmp_path, uncompacted, monkeypatch):
    d = tmp_path / "r"
    small_commit_run(d, 40)
    compact.compact(d)
    monkeypatch.undo()
    run = trex.init(d, commit_interval=0.001)
    for i in range(40, 140):
        run.log({"loss": i + 0.5}, step=i)
        time.sleep(0.002)
    run.finish()
    rows = readback(d)[1]
    assert [r[0] for r in rows] == list(range(140)) and commits(d) < 20


def test_compact_command_reports_each_run_and_skips_open_ones(tmp_path, uncompacted, capsys):
    for name in ("a", "b/c"):
        small_commit_run(tmp_path / "runs" / name, 30)
    held = connect_rw(tmp_path / "runs" / "b" / "c")
    held.execute("SELECT 1 FROM meta").fetchall()
    try:
        main(["compact", str(tmp_path / "runs")])
    finally:
        held.close()
    out = capsys.readouterr().out
    assert "[1/2]" in out and "-> 1 commits" in out and "open in another process" in out
    assert "compacted 1 runs" in out and "1 skipped, 0 failed" in out
    assert json.loads(dict(sqlite3.connect(tmp_path / "runs" / "a" / DB).execute("SELECT key, value FROM meta").fetchall())["state"]) == "finished"
