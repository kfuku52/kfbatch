import pandas

from kfbatch.batch_scope import aggregate_jobs, print_group_job_summary


def test_aggregate_jobs_counts_slurm_slots_in_all_buckets():
    frame = pandas.DataFrame(
        {
            "state": ["R", "PD", "FAILED", "SUSPENDED"],
            "total_slots": [4, 3, 2, 1],
        }
    )
    totals = aggregate_jobs(frame, "slurm")
    assert (totals.running, totals.queued, totals.failed, totals.other) == (4, 3, 2, 1)


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


def test_uge_group_summary_uses_qfree_members(capsys):
    jobs = pandas.DataFrame(
        {
            "user": ["user_a", "user_b", "user_c"],
            "state": ["r", "Rq", "Eqw"],
            "queue_name": ["mjobs.q", "", ""],
            "total_slots": [4, 20, 5],
        }
    )
    jobs.attrs["all_users"] = True
    qfree = pandas.DataFrame({"group_slots": [4]})
    qfree.attrs["group_name"] = "group_a"
    qfree.attrs["group_users"] = ["user_a", "user_b"]
    assert print_group_job_summary(
        jobs,
        scheduler="uge",
        current_user="user_a",
        qfree_frame=qfree,
        by_user=True,
    )
    out = capsys.readouterr().out
    assert "group[group_a]:R/Q/F=4/20/0" in out
    assert "user_c" not in out
