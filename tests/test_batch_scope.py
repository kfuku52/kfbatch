import pandas
import pytest

from kfbatch.batch_scope import JobTotals, aggregate_jobs, print_group_job_summary


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
