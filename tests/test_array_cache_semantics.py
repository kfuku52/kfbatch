"""Repeated array ranges retain exact and estimated task counts."""

from kfbatch.uge_parser import get_user_df


def test_repeated_uge_arrays_keep_invalid_ranges_and_estimates():
    cases = [
        ("", 1, False),
        ("1-5:2", 3, False),
        ("1,2-4", 4, False),
        ("2-1", 1, True),
        ("1-4:0", 1, True),
        ("3,invalid", 1, True),
        ("1-5,2-6", 10, False),
        ("1-9:3,99", 4, False),
        ("2,,4", 2, True),
        ("bad", 1, True),
    ]
    cases = cases + [(f"1-{i}", i, False) for i in range(1, 301)] + cases
    lines = [
        f"{i + 1} 0.5 job current_user qw 10/23/2026 00:00:00 2 {expression}"
        for i, (expression, _, _) in enumerate(cases)
    ]

    frame = get_user_df(iter(lines))

    assert frame["ja_task_id"].tolist() == [expression for expression, _, _ in cases]
    assert frame["total_slots"].tolist() == [2 * count for _, count, _ in cases]
    assert frame["task_count_estimated"].tolist() == [estimated for _, _, estimated in cases]
    assert frame.attrs["candidate_rows"] == len(cases)
    assert frame.attrs["rejected_rows"] == 0
