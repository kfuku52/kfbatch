import pandas
import pytest

from kfbatch.batch_scope import JobTotals, _print_user_breakdown, aggregate_jobs


@pytest.mark.parametrize("field", ["state", "queue_name", "total_slots"])
def test_user_breakdown_preserves_first_seen_user_conversion_order(field, capsys):
    class AlternatingValue:
        calls = 0

        def __str__(self):
            self.calls += 1
            choices = ("R", "PD") if field == "state" else ("", "main.q")
            return choices[(self.calls - 1) % 2]

        def __int__(self):
            self.calls += 1
            return 1 if self.calls % 2 else 2

    value = AlternatingValue()
    frame = pandas.DataFrame(
        {"user": ["user_b", "user_a"] * 64, "state": "R", "queue_name": "", "total_slots": 1}
    )
    frame[field] = [value] * len(frame)
    scheduler = "uge" if field == "queue_name" else "slurm"
    if scheduler == "uge":
        frame["state"] = ""
    _print_user_breakdown(frame, scheduler)
    counts = {
        "state": "R/Q/X/O=32/32/0/0",
        "queue_name": "R/Q/F=32/0/0",
        "total_slots": "R/Q/X/O=96/0/0/0",
    }
    assert capsys.readouterr().out == "".join(
        f"      {user}:{counts[field]}\n" for user in ["user_a", "user_b"]
    )
    assert value.calls == len(frame)


@pytest.mark.parametrize("scheduler,states", [("slurm", ["R", "PD"]), ("uge", ["r", "qw"])])
def test_aggregate_jobs_classifies_non_string_states_for_every_row(scheduler, states):
    class ChangingState:
        def __init__(self):
            self.calls = 0

        def __str__(self):
            state = states[self.calls % len(states)]
            self.calls += 1
            return state

    state = ChangingState()
    frame = pandas.DataFrame({"state": [state] * 260, "total_slots": [1] * 260})
    assert aggregate_jobs(frame, scheduler) == JobTotals(running=130, queued=130)
    assert state.calls == 260


def test_aggregate_uge_jobs_distinguishes_empty_and_assigned_queues_with_repeated_states():
    frame = pandas.DataFrame(
        {
            "state": [""] * 7,
            "queue_name": ["", " ", "main.q@node01", "other.q@node02", "", None, []],
            "total_slots": [1, 2, 4, 8, 16, 32, 64],
        }
    )
    frame = pandas.concat([frame] * 20, ignore_index=True)
    assert aggregate_jobs(frame, "uge") == JobTotals(running=2160, other=380)


@pytest.mark.parametrize("scheduler", ["slurm", "uge"])
def test_aggregate_jobs_preserves_nullable_state_and_queue_values(scheduler):
    frame = pandas.DataFrame(
        {
            "state": pandas.Series(["R", pandas.NA, "PD", ""], dtype="string"),
            "queue_name": pandas.Series(["", pandas.NA, "", "main.q"], dtype="string"),
            "total_slots": [1, 2, 4, 8],
        }
    )
    frame = pandas.concat([frame] * 40, ignore_index=True)
    expected = (
        JobTotals(running=40, queued=160, other=400)
        if scheduler == "slurm"
        else JobTotals(running=440, failed=160)
    )
    assert aggregate_jobs(frame, scheduler) == expected


def test_uge_bucket_cache_preserves_queue_only_running_and_state_precedence():
    states = ["", "qw", "Eqw", "dr", "h", "w", "r", "s"] * 32
    queues = ["main.q", "main.q", "main.q", "main.q", "", "", "", ""] * 32
    frame = pandas.DataFrame({"state": states, "queue_name": queues, "total_slots": [1] * 256})
    assert aggregate_jobs(frame, "uge") == JobTotals(running=96, queued=96, failed=64)


def test_uge_bucket_cache_bypasses_str_subclasses_with_custom_conversion():
    class QueuedText(str):
        def __str__(self):
            return "qw"

    class AssignedQueue(str):
        def __str__(self):
            return "main.q"

    frame = pandas.DataFrame(
        {
            "state": pandas.Series(["r", QueuedText("r"), ""] * 64, dtype=object),
            "queue_name": pandas.Series(["", "", AssignedQueue("")] * 64, dtype=object),
            "total_slots": [1] * 192,
        }
    )
    assert aggregate_jobs(frame, "uge") == JobTotals(running=128, queued=64)


def test_uge_bucket_cache_retains_assigned_queue_after_vocabulary_cutoff():
    frame = pandas.DataFrame(
        {
            "state": [f"v{index:03d}" for index in range(140)] + ["", "qw", "Eqw"] * 64,
            "queue_name": ["main.q"] * 332,
            "total_slots": [1] * 332,
        }
    )
    assert aggregate_jobs(frame, "uge") == JobTotals(running=204, queued=64, failed=64)
