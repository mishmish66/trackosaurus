import os
import sqlite3
import time

import numpy as np
import pytest

import trex
from trex import chunks, journal
from trex.format import DB, SCHEMA, connect_ro
from trex.index import Explorer


@pytest.fixture(autouse=True)
def journaled(tmp_path, monkeypatch):
    monkeypatch.setenv("TREX_JOURNAL", "1")
    monkeypatch.setenv("TREX_REPLICAS", str(tmp_path / "replicas"))


def tables(c):
    """Every table's rows, as stored."""
    return {t: sorted(c.execute(f"SELECT {cols[cols.index('(') + 1:-1]} FROM {t}").fetchall())
            for t, cols in journal.TABLES.items()}


def replica_of(d):
    c = sqlite3.connect(journal.sync(d, SCHEMA))
    try:
        return tables(c)
    finally:
        c.close()


def stored(d):
    c = sqlite3.connect(d / DB)
    try:
        return tables(c)
    finally:
        c.close()


def log_some(run, n, start=0):
    for i in range(start, start + n):
        run.log({"loss": 1.0 / (i + 1), "odd": float("nan")} if i % 2 else {"loss": 1.0 / (i + 1)}, step=i)


def test_replaying_the_journal_rebuilds_the_run_exactly(tmp_path):
    d = tmp_path / "r"
    run = trex.init(d, config={"lr": 0.1}, commit_interval=0.02)
    log_some(run, 70_000)
    run.log_image("img", b"\x89PNG\r\n\x1a\n" + bytes(32), step=5)
    run.summary(best=0.5)
    run.finish()
    assert (d / journal.JOURNAL).exists() and replica_of(d) == stored(d)


def test_a_live_journaled_run_is_read_from_its_replica(tmp_path):
    d = tmp_path / "r"
    run = trex.init(d, commit_interval=0.02)
    log_some(run, 10)
    time.sleep(0.2)
    assert (d / (DB + "-wal")).exists()
    c = connect_ro(d)
    try:
        path = c.execute("PRAGMA database_list").fetchone()[2]
        assert str(tmp_path / "replicas") in path and chunks.row_count(c) == 10
    finally:
        c.close()
    log_some(run, 5, start=10)
    time.sleep(0.2)
    c = connect_ro(d)
    try:
        assert chunks.row_count(c) == 15
    finally:
        c.close()
    run.finish()
    c = connect_ro(d)
    try:
        assert not (d / (DB + "-wal")).exists() and c.execute("PRAGMA database_list").fetchone()[2] == str(d / DB)
    finally:
        c.close()


def test_a_partly_written_record_is_not_read_and_the_writer_drops_it_on_reopening(tmp_path):
    d = tmp_path / "r"
    run = trex.init(d)
    log_some(run, 3)
    run.finish()
    whole = (d / journal.JOURNAL).read_bytes()
    with open(d / journal.JOURNAL, "ab") as f:
        f.write(whole[-20:-3])
    assert list(journal.records((d / journal.JOURNAL).read_bytes()))[-1].end == len(whole)
    run = trex.init(d)
    log_some(run, 2, start=3)
    run.finish()
    recs = list(journal.records(data := (d / journal.JOURNAL).read_bytes()))
    assert recs[-1].end == len(data) and replica_of(d) == stored(d)
    assert [r.payload.get("session") for r in recs if "session" in r.payload] == [journal.host()]


def test_a_run_reopened_without_its_journal_is_journaled_from_a_snapshot(tmp_path, monkeypatch):
    d = tmp_path / "r"
    monkeypatch.setenv("TREX_JOURNAL", "0")
    run = trex.init(d)
    log_some(run, 4)
    run.finish()
    assert not (d / journal.JOURNAL).exists()
    monkeypatch.setenv("TREX_JOURNAL", "1")
    run = trex.init(d)
    log_some(run, 3, start=4)
    run.finish()
    assert replica_of(d) == stored(d) and stored(d)["rowmeta"][-1][0] == 4


def test_a_rewritten_journal_rebuilds_the_replica(tmp_path):
    d = tmp_path / "r"
    run = trex.init(d)
    log_some(run, 4)
    run.finish()
    replica_of(d)
    (d / DB).unlink()
    run = trex.init(d)
    log_some(run, 2)
    run.finish()
    assert replica_of(d) == stored(d) and len(stored(d)["rowmeta"]) == 1


def test_a_failing_journal_stops_journaling_but_not_the_run(tmp_path, monkeypatch, capsys):
    d = tmp_path / "r"
    run = trex.init(d, commit_interval=0.02)

    def full(*a):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(run._journal, "append", full)
    log_some(run, 5)
    run.finish()
    c = sqlite3.connect(d / DB)
    try:
        assert chunks.row_count(c) == 5
    finally:
        c.close()
    assert "journal" in capsys.readouterr().err


def test_runs_off_network_filesystems_are_not_journaled_unless_asked(tmp_path, monkeypatch):
    monkeypatch.delenv("TREX_JOURNAL")
    trex.init(tmp_path / "local").finish()
    assert not (tmp_path / "local" / journal.JOURNAL).exists()


MOUNTINFO = """\
22 1 8:1 / / rw,relatime - ext4 /dev/sda1 rw
40 22 0:50 / /scratch rw - nfs4 vast:/scratch rw
41 40 0:51 / /scratch/local\\040disk rw - xfs /dev/sdb rw
42 22 0:52 / /home rw - nfs vast:/home rw
"""


@pytest.mark.parametrize("path,want", [("/scratch/me/runs", True), ("/scratch/local disk/runs", False), ("/home/me", True),
                                       ("/tmp/runs", False), ("/scratchy", False)])
def test_network_filesystems_are_told_by_their_mount(path, want):
    assert journal.on_network_fs(path, MOUNTINFO) is want


def test_the_explorer_follows_a_live_journaled_run(tmp_path):
    root = tmp_path / "runs"
    run = trex.init(root / "r", commit_interval=0.02)
    log_some(run, 50)
    time.sleep(0.2)
    ex = Explorer(root, tmp_path / "cache")
    ex.rewalk()
    ex.poll()
    assert ex.run_meta("r")["seq"] == 50
    log_some(run, 30, start=50)
    run.log_image("img", np.zeros((4, 4, 3), np.uint8), step=60)
    time.sleep(0.2)
    ex.poll()
    meta = ex.run_meta("r")
    assert (meta["seq"], meta["mseq"]) == (80, 1)
    run.finish()
    ex.poll()
    assert ex.run_meta("r")["state"] == "finished" and ex.tiles([["r", "loss", "top"]])[0]
    assert sorted(os.listdir(root / "r")) == sorted(["media", DB, journal.JOURNAL])
