import json
import math
import os
import subprocess
import sys
import textwrap
import time
from typing import Any, cast

import pytest

import trex
from trex import chunks
from trex.format import connect_ro, key_names, row_count


def rows_of(d):
    c = connect_ro(d)
    out = [(seq, step, vals) for seq, step, _, vals in chunks.rows(c)]
    c.close()
    return out


def meta_of(d):
    c = connect_ro(d)
    m = {k: json.loads(v) for k, v in c.execute("SELECT key, value FROM meta")}
    c.close()
    return m


def test_logs_contiguous_rows_with_flattened_scalars_and_nonfinite(tmp_path):
    run = trex.init(tmp_path / "r", name="n", config={"opt": {"lr": 1e-3}}, tags=["a"])
    run.log({"a": {"b": 1.5}, "nan": float("nan"), "flag": True, "text": "skip"})  # pyright: ignore[reportArgumentType]  # non-numbers are skipped
    run.log({"x": 2}, step=10)
    run.log({"x": 3})
    run.finish()
    rows = rows_of(tmp_path / "r")
    assert [(s, st) for s, st, _ in rows] == [(0, 0.0), (1, 10.0), (2, 11.0)]
    first = rows[0][2]
    assert first["a/b"] == 1.5 and first["flag"] == 1 and "text" not in first and math.isnan(first["nan"])
    m = meta_of(tmp_path / "r")
    assert (m["name"], m["state"], m["config"], m["tags"]) == ("n", "finished", {"opt/lr": 1e-3}, ["a"])


def test_rows_become_visible_to_readers_within_the_commit_interval(tmp_path):
    run = trex.init(tmp_path / "r", commit_interval=0.2)
    run.log({"x": 1})
    deadline = time.time() + 2
    while not rows_of(tmp_path / "r") and time.time() < deadline:
        time.sleep(0.05)
    assert len(rows_of(tmp_path / "r")) == 1
    assert meta_of(tmp_path / "r")["state"] == "running"
    run.finish()


def test_reopening_a_run_resumes_its_sequence_and_step(tmp_path):
    a = trex.init(tmp_path / "r")
    a.log({"x": 1}, step=5)
    a.finish()
    uid = meta_of(tmp_path / "r")["id"]
    b = trex.init(tmp_path / "r")
    b.log({"x": 2})
    b.finish()
    assert [(s, st) for s, st, _ in rows_of(tmp_path / "r")] == [(0, 5.0), (1, 6.0)]
    assert meta_of(tmp_path / "r")["id"] == uid


def test_killed_writer_leaves_a_readable_consistent_run(tmp_path):
    code = textwrap.dedent(f"""
        import os, signal, time, trex
        run = trex.init({str(tmp_path / "r")!r}, commit_interval=0.05)
        for i in range(20000):
            run.log({{"x": i}})
            if i % 1000 == 0:
                time.sleep(0.06)
        os.kill(os.getpid(), signal.SIGKILL)
    """)
    subprocess.run([sys.executable, "-c", code])
    c = connect_ro(tmp_path / "r")
    assert c.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    n = row_count(c)
    total = c.execute("SELECT coalesce(sum(n), 0) FROM rowmeta").fetchone()[0]
    c.close()
    assert n > 0 and n == total
    assert [r[2]["x"] for r in rows_of(tmp_path / "r")] == list(range(n))
    assert meta_of(tmp_path / "r")["state"] == "running"


def test_merge_plan_merges_fan_in_commits_of_a_value_tier_newest_first():
    F, plan = trex.writer.FAN_IN, trex.writer.merge_plan
    ones = [(1, 1)] * F
    assert plan(ones[:-1]) is None
    assert plan(ones) == 0
    assert plan([(50, 50), (F, F)] + ones) == 2
    assert plan([(F, F)] + ones[:-1]) == 0
    assert plan([(1, 100)] * F) == 0
    big = trex.writer.SEALED - 1
    assert plan([(1, big)] * F) is None
    assert plan([(30_000, 30_000)] * F) is None
    assert plan([(1, trex.writer.SEALED)] + ones[:-1]) is None
    assert trex.writer.sealed(1, trex.writer.SEALED) and trex.writer.sealed(40_000, 40_000) and not trex.writer.sealed(10, 10)


def log_one_row_per_commit(run, n, start=0):
    for i in range(start, start + n):
        run.log({"loss": i + 0.5, **({"eval": -float(i)} if i % 7 == 0 else {})}, step=i)
        time.sleep(0.003)


def expected_rows(n):
    return [(i, float(i), {"loss": i + 0.5, **({"eval": -float(i)} if i % 7 == 0 else {})}) for i in range(n)]


def commit_count(d):
    c = connect_ro(d)
    try:
        return c.execute("SELECT count(*) FROM rowmeta").fetchone()[0]
    finally:
        c.close()


def test_small_commits_are_merged_as_the_run_goes(tmp_path, monkeypatch):
    merges = []
    apply = chunks.apply_merge
    monkeypatch.setattr(chunks, "apply_merge", lambda c, m: merges.append(m.commits) or apply(c, m))
    run = trex.init(tmp_path / "r", commit_interval=0.001)
    log_one_row_per_commit(run, 300)
    run.finish()
    assert rows_of(tmp_path / "r") == expected_rows(300)
    assert len(merges) >= 10 and commit_count(tmp_path / "r") < 3 * trex.writer.FAN_IN


def test_a_failing_merge_stops_compaction_and_leaves_every_row(tmp_path, monkeypatch, capsys):
    def broken(c, m):
        c.execute("DELETE FROM rowmeta WHERE seq0 >= ? AND seq0 < ?", (m.seq0, m.stop))
        raise OSError("disk full")

    monkeypatch.setattr(chunks, "apply_merge", broken)
    run = trex.init(tmp_path / "r", commit_interval=0.001)
    log_one_row_per_commit(run, 40)
    run.finish()
    assert rows_of(tmp_path / "r") == expected_rows(40)
    assert "compaction of" in capsys.readouterr().err


def check_killed_run(d):
    """Every committed row of a run whose writer was killed is there, in commits without gaps or overlaps, and the
    run resumes."""
    c = connect_ro(d)
    assert c.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    n = row_count(c)
    commits = c.execute("SELECT seq0, n FROM rowmeta ORDER BY seq0").fetchall()
    c.close()
    assert n > 0 and [s for s, _ in commits] == [sum(k for _, k in commits[:i]) for i in range(len(commits))]
    assert sum(k for _, k in commits) == n and rows_of(d) == expected_rows(n)
    run = trex.init(d, commit_interval=0.001)
    log_one_row_per_commit(run, 20, start=n)
    run.finish()
    assert rows_of(d) == expected_rows(n + 20)


LOGGER = """
    run = trex.init({d!r}, commit_interval=0.001)
    for i in range(5000):
        run.log({{"loss": i + 0.5, **({{"eval": -float(i)}} if i % 7 == 0 else {{}})}}, step=i)
        time.sleep(0.002)
"""


@pytest.mark.parametrize("where", ["halfway through a swap", "before a swap commits"])
@pytest.mark.parametrize("nth", [2, 9])
def test_a_writer_killed_while_merging_keeps_every_committed_row(tmp_path, where, nth):
    d = tmp_path / "r"
    kill = ("c.execute('DELETE FROM chunk WHERE id >= ? AND seq0 >= ? AND seq0 < ?', (m.first_id, m.seq0, m.stop))"
            if where == "halfway through a swap" else "real(c, m)")
    code = textwrap.dedent(f"""
        import os, signal, time, trex
        from trex import chunks
        real, calls = chunks.apply_merge, [0]
        def explode(c, m):
            calls[0] += 1
            if calls[0] == {nth}:
                {kill}
                os.kill(os.getpid(), signal.SIGKILL)
            return real(c, m)
        chunks.apply_merge = explode
    """) + textwrap.dedent(LOGGER.format(d=str(d)))
    assert subprocess.run([sys.executable, "-c", code]).returncode == -9
    check_killed_run(d)


@pytest.mark.parametrize("after", [0.15, 0.4, 0.75, 1.3])
def test_a_writer_killed_at_any_moment_keeps_every_committed_row(tmp_path, after):
    d = tmp_path / "r"
    code = textwrap.dedent(f"""
        import os, signal, threading, time, trex
        threading.Timer({after}, lambda: os.kill(os.getpid(), signal.SIGKILL)).start()
    """) + textwrap.dedent(LOGGER.format(d=str(d)))
    assert subprocess.run([sys.executable, "-c", code]).returncode == -9
    check_killed_run(d)


def test_media_files_are_written_before_their_rows(tmp_path):
    run = trex.init(tmp_path / "r")
    png = b"\x89PNG\r\n\x1a\n" + bytes(100)
    run.log_image("img", png, step=3)
    run.log_html("page", "<b>hi</b>", step=4)
    run.finish()
    c = connect_ro(tmp_path / "r")
    media = c.execute("SELECT seq, step, key, kind, file, size FROM media ORDER BY seq").fetchall()
    c.close()
    assert [(m[0], m[1], m[2], m[3]) for m in media] == [(0, 3.0, "img", "image"), (1, 4.0, "page", "html")]
    assert (tmp_path / "r" / media[0][4]).read_bytes() == png
    assert (tmp_path / "r" / media[1][4]).stat().st_size == media[1][5]


def test_exception_inside_context_marks_run_failed(tmp_path):
    with pytest.raises(ValueError):
        with trex.init(tmp_path / "r") as run:
            run.log({"x": 1})
            raise ValueError
    assert meta_of(tmp_path / "r")["state"] == "failed"


def test_info_dict_keeps_nesting_and_merges_updates(tmp_path):
    run = trex.init(tmp_path / "r", info={"git": {"sha": "abc", "dirty": False}, "notes": "first try"})
    run.info(notes="second try", host={"name": "gpu-box", "gpus": [0, 1]}, bad=float("nan"))
    run.finish()
    assert meta_of(tmp_path / "r")["info"] == {"git": {"sha": "abc", "dirty": False}, "notes": "second try",
                                               "host": {"name": "gpu-box", "gpus": [0, 1]}, "bad": "nan"}


def test_folder_info_merges_into_a_json_file(tmp_path):
    trex.folder_info(tmp_path / "sweep", {"question": "does width help?"})
    trex.folder_info(tmp_path / "sweep", owner="me")
    assert json.loads((tmp_path / "sweep" / "trex_info.json").read_text()) == {"question": "does width help?", "owner": "me"}


def test_rows_are_stored_as_one_commit_with_one_chunk_per_metric_and_names_once(tmp_path):
    run = trex.init(tmp_path / "r", commit_interval=60)
    for i in range(10):
        run.log({"train/loss": 1 / (i + 1), "lr": 0.1})
        if i % 5 == 4:
            run.log({"eval/score": i}, step=i)
    run.finish()
    c = connect_ro(tmp_path / "r")
    assert c.execute("SELECT seq0, n FROM rowmeta").fetchall() == [(0, 12)]
    assert c.execute("SELECT count(*) FROM chunk").fetchone()[0] == 3
    assert sorted(key_names(c).values()) == ["eval/score", "lr", "train/loss"]
    c.close()
    rows = rows_of(tmp_path / "r")
    assert [r[0] for r in rows] == list(range(12))
    assert rows[5] == (5, 4.0, {"eval/score": 4})


def test_row_times_are_seconds_since_creation(tmp_path):
    run = trex.init(tmp_path / "r")
    run.log({"x": 1}, timestamp=run.created + 12.5)
    run.finish()
    c = connect_ro(tmp_path / "r")
    assert chunks.rows(c)[0][2] == 12.5
    c.close()


def media_of(d):
    c = connect_ro(d)
    out = c.execute("SELECT key, kind, file FROM media ORDER BY seq").fetchall()
    c.close()
    return [(k, kind, (d / f).read_bytes()) for k, kind, f in out]


class FakePIL:
    def save(self, fp, format):
        assert format == "PNG"
        fp.write(b"\x89PNG\r\n\x1a\nfake")


def test_images_are_logged_from_arrays_paths_bytes_and_pil_images(tmp_path):
    np = pytest.importorskip("numpy")
    jpg = tmp_path / "photo.JPEG"
    jpg.write_bytes(b"\xff\xd8\xff\xe0jpeg")
    run = trex.init(tmp_path / "r")
    run.log_image("array", np.zeros((4, 6, 3), np.uint8), step=0)
    run.log_image("path", jpg, step=0)
    run.log_image("bytes", bytearray(b"GIF89a..."), step=0)
    run.log_image("pil", FakePIL(), step=0)
    run.finish()
    got = {k: (kind, data) for k, kind, data in media_of(tmp_path / "r")}
    assert got["array"][1][:8] == b"\x89PNG\r\n\x1a\n" and got["path"][1] == jpg.read_bytes()
    assert got["bytes"][1] == b"GIF89a..." and got["pil"][1] == b"\x89PNG\r\n\x1a\nfake"
    c = connect_ro(tmp_path / "r")
    assert dict(c.execute("SELECT key, file FROM media"))["path"].endswith(".jpg")
    c.close()


def test_videos_and_html_are_logged_from_bytes_and_paths(tmp_path):
    mp4, page = tmp_path / "clip.mp4", tmp_path / "page.html"
    mp4.write_bytes(b"\0\0\0\x18ftypisom")
    page.write_text("<i>report</i>")
    run = trex.init(tmp_path / "r")
    run.log_video("file", mp4, step=1)
    run.log_video("bytes", b"\0\0\0\x18ftypmp42", step=1)
    run.log_html("page", page, step=1)
    run.finish()
    assert media_of(tmp_path / "r") == [("file", "video", mp4.read_bytes()), ("bytes", "video", b"\0\0\0\x18ftypmp42"),
                                        ("page", "html", b"<i>report</i>")]


def test_metric_values_unwrap_scalars_count_bools_and_drop_non_numbers(tmp_path):
    np = pytest.importorskip("numpy")
    run = trex.init(tmp_path / "r")
    untyped: Any = {"vec": np.array([1, 2]), "name": "x", "none": None}
    run.log({"f32": np.float32(1.5), "i64": np.int64(3), "flag": True, **untyped})
    run.log({"zero_d": np.array(2.0)})
    run.log(cast(Any, {"only": "text"}))
    run.finish()
    assert rows_of(tmp_path / "r") == [(0, 0.0, {"f32": 1.5, "i64": 3.0, "flag": 1.0}), (1, 1.0, {"zero_d": 2.0})]


def test_writer_errors_are_raised_by_finish_and_never_by_log(tmp_path, monkeypatch, capsys):
    def broken(self, c, final_state=None):
        raise OSError("disk full")

    monkeypatch.setattr(trex.writer.Run, "_commit", broken)
    run = trex.init(tmp_path / "r", commit_interval=0.01)
    run.log({"loss": 1.0})
    time.sleep(0.1)
    run.log({"loss": 2.0})
    with pytest.raises(RuntimeError, match="disk full"):
        run.finish()
    assert "disk full" in capsys.readouterr().err


def test_uncaught_exception_marks_the_run_failed_and_still_reports_the_error(tmp_path):
    code = textwrap.dedent(f"""
        import trex
        run = trex.init({str(tmp_path / "r")!r})
        run.log({{"x": 1.0}})
        raise SystemError("boom")
    """)
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env={**os.environ, "PYTHON_COLORS": "0"})
    assert res.returncode != 0 and "SystemError: boom" in res.stderr
    assert meta_of(tmp_path / "r")["state"] == "failed" and rows_of(tmp_path / "r") == [(0, 0.0, {"x": 1.0})]


def test_summary_and_info_values_that_are_not_json_become_text(tmp_path):
    run = trex.init(tmp_path / "r")
    run.summary(path=tmp_path / "ckpt", best=float("inf"), pair=(1, 2))
    run.finish()
    s = meta_of(tmp_path / "r")["summary"]
    assert (s["path"], s["best"], s["pair"]) == (str(tmp_path / "ckpt"), "inf", [1, 2])
