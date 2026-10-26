import pandas
import pytest

from kfbatch.batch_scope import (
    JobTotals,
    _print_user_breakdown,
    aggregate_jobs,
    print_group_job_summary,
)


@pytest.mark.parametrize("repetitions", [1, 32])
def test_aggregate_jobs_counts_slurm_slots_in_all_buckets(repetitions):
    frame = pandas.DataFrame(
        {
            "state": ["R", "PD", "FAILED", "SUSPENDED"],
            "total_slots": [4, 3, 2, 1],
        }
    )
    frame = pandas.concat([frame] * repetitions, ignore_index=True)
    totals = aggregate_jobs(frame, "slurm")
    assert totals == JobTotals(
        running=4 * repetitions,
        queued=3 * repetitions,
        failed=2 * repetitions,
        other=repetitions,
    )


@pytest.mark.parametrize("repetitions", [1, 32])
def test_aggregate_uge_jobs_retains_state_precedence_over_queue_assignment(repetitions):
    frame = pandas.DataFrame(
        {
            "state": ["Rq", "hRq", "Eqw", "dr", "h", "", ""],
            "queue_name": ["main.q"] * 6 + [""],
            "total_slots": [1, 2, 3, 4, 5, 6, 7],
        }
    )
    frame = pandas.concat([frame] * repetitions, ignore_index=True)
    assert aggregate_jobs(frame, "uge") == JobTotals(
        running=6 * repetitions,
        queued=8 * repetitions,
        failed=7 * repetitions,
        other=7 * repetitions,
    )


@pytest.mark.parametrize(
    "scheduler,states,expected",
    [
        (
            "slurm",
            ["R", "PD", "FAILED", "SUSPENDED"],
            "      absent:R/Q/X/O=0/0/0/0\n"
            "      user_a:R/Q/X/O=0/3/7/0\n"
            "      user_b:R/Q/X/O=4/0/0/2\n",
        ),
        (
            "uge",
            ["r", "qw", "Eqw", ""],
            "      absent:R/Q/F=0/0/0\n      user_a:R/Q/F=0/3/7\n      user_b:R/Q/F=4/0/0\n",
        ),
    ],
)
def test_user_breakdown_lists_members_in_name_order(scheduler, states, expected, capsys):
    frame = pandas.DataFrame(
        {
            "user": ["user_b", "user_a", "user_a", "user_b"],
            "state": states,
            "queue_name": [""] * 4,
            "total_slots": [4, 3, 7, 2],
        },
        index=[9, 3, 1, 2],
    )
    _print_user_breakdown(frame, scheduler, ["absent", "user_b", ""])
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
