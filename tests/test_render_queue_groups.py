"""Regression coverage for rendering queue groups in source appearance order."""

import re
from types import SimpleNamespace

import pandas
import pytest

from kfbatch import render


@pytest.mark.parametrize("scheduler", ["slurm", "uge"])
def test_compact_queues_preserve_order_and_attrs(capsys, scheduler):
    queues = ["z", "a", "z", "login01", "a"]
    frame = pandas.DataFrame(
        {
            "queue_name": queues,
            "node_name": [f"node{i:02}" for i in range(len(queues))],
            "status": ["", "d", "", "", ""],
            "ncore_total": [8] * len(queues),
            "ncore_used": [2] * len(queues),
            "ncore_available": [6] * len(queues),
            "hl:mem_total": [16.0] * len(queues),
            "hc:mem_req": [12.0] * len(queues),
        },
    )
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
        assert [row[0] for row in cells] == ["z", "a"]
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
