"""Preserve compact table layout, including padding and empty summaries."""

from types import SimpleNamespace

import pandas
import pytest

from kfbatch.stat import print_slurm_compact_summary, print_uge_compact_summary


@pytest.mark.parametrize(
    "render,header,row_suffix,legend_suffix",
    [
        (
            print_slurm_compact_summary,
            "part             nodes  cpu(a/u/t)  ram(a/t)GiB  topCPU          topRAM  launch",
            "-     ",
            ", launch=res=CPU/RAM-only ceiling",
        ),
        (
            print_uge_compact_summary,
            "queue            nodes  cpu(a/u/t)  ram(a/t)GiB  topCPU          topRAM  quota(s/g/l)  launch2G",
            "-             -       ",
            "",
        ),
    ],
)
@pytest.mark.parametrize("empty", [False, True])
def test_compact_summary_exact_layout(render, header, row_suffix, legend_suffix, empty, capsys):
    df = pandas.DataFrame(
        {
            "queue_name": ["short", "long_queue_name"],
            "node_name": ["node01", "node02"],
            "status": ["", "d"],
            "ncore_available": [2, 0],
            "ncore_used": [1, 0],
            "ncore_total": [3, 4],
            "hc:mem_req": [1.9, float("nan")],
            "hl:mem_total": [3.9, float("nan")],
        }
    )
    if empty:
        df = df.iloc[:0]
    render(df, None, SimpleNamespace(exclude_abnormal_node=True))
    expected = (
        ""
        if empty
        else (
            f"{header}\n"
            f"short            1/0/1  2/1/3       1/3          node01 2c/1GiB  same    {row_suffix}\n"
            f"long_queue_name  0/1/1  0/0/4       ?/?          -               -       {row_suffix}\n"
            "\nlegend: nodes=working/abnormal/total, cpu=available/used/total, "
            f"ram=available/total{legend_suffix}\n\n"
        )
    )
    assert capsys.readouterr().out == expected
