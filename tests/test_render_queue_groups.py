"""Regression coverage for rendering queue groups in source appearance order."""

import re
from types import SimpleNamespace

import pandas
import pytest

from kfbatch import render


@pytest.mark.parametrize("scheduler", ["slurm", "uge"])
@pytest.mark.parametrize("categorical", [False, True])
def test_compact_queues_preserve_order_missing_rows_and_attrs(capsys, scheduler, categorical):
    queues = ["z", "a", "z", "login01", "a", None, float("nan"), pandas.NA]
    frame = pandas.DataFrame(
        {
            "queue_name": queues,
            "node_name": [f"node{i:02}" for i in range(len(queues))],
            "status": ["", "d", "", "", "", "", "", ""],
            "ncore_total": [8] * len(queues),
            "ncore_used": [2] * len(queues),
            "ncore_available": [6] * len(queues),
            "hl:mem_total": [16.0] * len(queues),
            "hc:mem_req": [12.0] * len(queues),
        },
        index=[11, 11, 3, 3, 19, 19, 4, 4],
    )
    frame["queue_name"] = pandas.Series(queues, dtype=object, index=frame.index)
    if categorical:
        frame["queue_name"] = pandas.Categorical(queues, categories=["unused", "login01", "a", "z"])
    frame.attrs = {"synthetic_metadata": {"preserved": [1, 2, 3]}}
    original = frame.copy(deep=True)
    args = SimpleNamespace(exclude_abnormal_node=True, all_tiers=True, ntop=1)
    extra = (
        pandas.DataFrame({"queue_name": ["qfreeonly", "a"], "self_slots": [3, 4]})
        if scheduler == "uge"
        else None
    )
    getattr(render, f"print_{scheduler}_compact_summary")(frame, extra, args)
    table = capsys.readouterr().out.split("\n\n", 1)[0].splitlines()[1:]
    cells = [re.split(r" {2,}", row.rstrip()) for row in table]
    if scheduler == "uge":
        assert [row[0] for row in cells] == ["z", "a", "login01", "qfreeonly"]
        assert cells[-1][1:5] == ["0/0/0", "0/0/0", "?/?", "-"]
    else:
        assert [row[0] for row in cells] == (
            ["z", "a", "nan"] if categorical else ["z", "a", "None", "nan", "<NA>"]
        )
        assert all(row[1:4] == ["0/0/0", "0/0/0", "?/?"] for row in cells[2:])
    assert cells[0][1:6] == [
        "2/0/2",
        "12/4/16",
        "24/32",
        "node00 6c/12GiB, node02 6c/12GiB",
        "same",
    ]
    assert cells[1][1:4] == ["1/1/2", "6/2/16", "12/32"]
    pandas.testing.assert_frame_equal(frame, original, check_exact=True)
    assert frame.attrs == original.attrs


def test_uge_mixed_queue_keys_keep_original_string_matching(capsys):
    frame = pandas.DataFrame(
        {
            "queue_name": [1, "1", 1],
            "node_name": ["numeric", "text", "numeric2"],
            "status": ["", "", ""],
            "ncore_total": [8, 8, 8],
            "ncore_used": [1, 2, 1],
            "ncore_available": [7, 6, 7],
            "hl:mem_total": [16.0, 16.0, 16.0],
            "hc:mem_req": [12.0, 12.0, 12.0],
        }
    )
    render.print_uge_compact_summary(frame, None, SimpleNamespace(exclude_abnormal_node=True))
    out = capsys.readouterr().out
    assert "1/0/1" in out
    assert "text 6c/12GiB" in out
    assert "numeric" not in out
