import json
from pathlib import Path

import pytest

from trex.where import as_number, compile_where, text_of

CASES = json.loads((Path(__file__).parent / "where_cases.json").read_text())


@pytest.mark.parametrize("expr,want", CASES["cases"], ids=[c[0] or "empty" for c in CASES["cases"]])
def test_where_selects_the_runs_the_shared_cases_name(expr, want):
    test = compile_where(expr).test
    assert [k for k, run in CASES["runs"].items() if test(run.get)] == want


@pytest.mark.parametrize("expr", CASES["errors"])
def test_where_refuses_clauses_that_do_not_parse(expr):
    with pytest.raises(ValueError):
        compile_where(expr)


def test_where_reports_the_fields_it_reads():
    assert compile_where("lr = 1 and (summary.loss < 2 or tags is null)").fields == ["lr", "summary.loss", "tags"]
    assert compile_where("ppo").fields == ["name", "path"]


@pytest.mark.parametrize("v,want", [(1.0, "1"), (0.001, "0.001"), (True, "true"), ("x", "x"), ([1, "a"], '[1,"a"]'), (float("nan"), "NaN")])
def test_values_are_matched_as_their_text(v, want):
    assert text_of(v) == want


@pytest.mark.parametrize("v,want", [("1e-3", 0.001), (" 2 ", 2.0), ("Infinity", float("inf")), ("inf", None), (True, None), ("1_000", None)])
def test_only_numbers_and_their_text_count_as_numbers(v, want):
    assert as_number(v) == want
