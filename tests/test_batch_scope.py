import pandas
import pytest

from kfbatch.batch_scope import (
    JobTotals,
    _print_user_breakdown,
    aggregate_jobs,
    print_group_job_summary,
)


def test_aggregate_jobs_counts_slurm_slots_in_all_buckets():
    frame = pandas.DataFrame(
        {
            "state": ["R", "PD", "FAILED", "SUSPENDED"],
            "total_slots": [4, 3, 2, 1],
        }
    )
    totals = aggregate_jobs(frame, "slurm")
    assert (totals.running, totals.queued, totals.failed, totals.other) == (4, 3, 2, 1)


@pytest.mark.parametrize("scheduler,state", [("slurm", "RUNNING"), ("uge", "r")])
def test_aggregate_jobs_preserves_slot_validation_and_missing_columns(scheduler, state):
    frame = pandas.DataFrame(
        {"state": state, "total_slots": [3, "4", None, float("nan"), "bad", -7]},
        index=[8, 8, 2, 3, 6, 1],
    )
    assert aggregate_jobs(frame, scheduler) == JobTotals(running=7)
    assert aggregate_jobs(frame.drop(columns="total_slots"), scheduler) == JobTotals()
    assert aggregate_jobs(frame.drop(columns="state"), scheduler) == JobTotals(other=7)


def test_aggregate_uge_jobs_retains_state_precedence_over_queue_assignment():
    frame = pandas.DataFrame(
        {
            "state": ["Rq", "hRq", "Eqw", "dr", "h", "", ""],
            "queue_name": ["main.q"] * 6 + [""],
            "total_slots": [1, 2, 3, 4, 5, 6, 7],
        }
    )
    assert aggregate_jobs(frame, "uge") == JobTotals(running=6, queued=8, failed=7, other=7)


@pytest.mark.parametrize(
    "scheduler,states,expected",
    [
        (
            "slurm",
            ["R", "PD", "FAILED", "SUSPENDED", "R", "R"],
            "      absent:R/Q/X/O=0/0/0/0\n"
            "      user_a:R/Q/X/O=0/3/0/0\n"
            "      user_b:R/Q/X/O=4/0/0/2\n",
        ),
        (
            "uge",
            ["r", "qw", "Eqw", "", "r", "r"],
            "      absent:R/Q/F=0/0/0\n      user_a:R/Q/F=0/3/0\n      user_b:R/Q/F=4/0/0\n",
        ),
    ],
)
def test_user_breakdown_retains_members_order_and_slot_validation(
    scheduler, states, expected, capsys
):
    frame = pandas.DataFrame(
        {
            "user": ["user_b", "user_a", "user_a", "user_b", None, ""],
            "state": states,
            "queue_name": [""] * 6,
            "total_slots": [4, "3", -7, 2, float("inf"), float("inf")],
        },
        index=[9, 9, 1, 2, 2, 4],
    )
    _print_user_breakdown(frame, scheduler, ["absent", "user_b", ""])
    assert capsys.readouterr().out == expected


@pytest.mark.parametrize("missing", ["state", "queue_name", "total_slots"])
def test_user_breakdown_keeps_users_when_job_columns_are_missing(missing, capsys):
    frame = pandas.DataFrame(
        {"user": ["user_b", "user_a"], "state": ["r", "qw"], "queue_name": ["", ""]}
    )
    frame["total_slots"] = [4, 3]
    _print_user_breakdown(frame.drop(columns=missing), "uge", ["absent"])
    expected = (
        "      absent:R/Q/F=0/0/0\n"
        f"      user_a:R/Q/F=0/{0 if missing in {'state', 'total_slots'} else 3}/0\n"
        f"      user_b:R/Q/F={0 if missing in {'state', 'total_slots'} else 4}/0/0\n"
    )
    assert capsys.readouterr().out == expected


def test_slurm_group_summary_discovers_account_and_breaks_down_users(capsys):
    jobs = pandas.DataFrame(
        {
            "user": ["current_user", "user_a", "user_c"],
            "account": ["account_a", "account_a", "account_b"],
            "state": ["PD", "R", "R"],
            "total_slots": [2, 4, 8],
        }
    )
    shares = pandas.DataFrame(
        {
            "account": ["account_a", "account_a", "account_b"],
            "user": ["current_user", "user_a", "user_c"],
        }
    )
    assert print_group_job_summary(
        jobs,
        scheduler="slurm",
        current_user="current_user",
        by_user=True,
        share_frame=shares,
    )
    out = capsys.readouterr().out
    assert "group[account_a]:R/Q/X/O=4/2/0/0" in out
    assert "current_user:R/Q/X/O=0/2/0/0" in out
    assert "user_a:R/Q/X/O=4/0/0/0" in out
    assert "user_c" not in out


def test_slurm_group_summary_does_not_guess_without_association(capsys):
    jobs = pandas.DataFrame(
        {
            "user": ["current_user"],
            "account": ["account_a"],
            "state": ["R"],
            "total_slots": [1],
        }
    )
    assert not print_group_job_summary(
        jobs,
        scheduler="slurm",
        current_user="current_user",
    )
    assert "unavailable" in capsys.readouterr().out
