"""Canonical, mutually exclusive scheduler job-state classification."""

import re

SLURM_RUNNING_STATES = {"R", "CG", "ST", "SO"}
SLURM_PENDING_STATES = {"PD", "CF", "RD", "RF", "RH", "RQ"}
SLURM_ERROR_STATES = {
    "BF",  # BOOT_FAIL
    "CA",  # CANCELLED
    "DL",  # DEADLINE
    "F",  # FAILED
    "LF",  # LAUNCH_FAILED
    "NF",  # NODE_FAIL
    "OOM",  # OUT_OF_MEMORY
    "PR",  # PREEMPTED
    "RV",  # REVOKED
    "SE",  # SPECIAL_EXIT
    "TO",  # TIMEOUT
}
SLURM_STATE_NAME_TO_CODE = {
    "RUNNING": "R",
    "COMPLETING": "CG",
    "PENDING": "PD",
    "CONFIGURING": "CF",
    "COMPLETED": "CD",
    "BOOT_FAIL": "BF",
    "CANCELLED": "CA",
    "DEADLINE": "DL",
    "FAILED": "F",
    "LAUNCH_FAILED": "LF",
    "NODE_FAIL": "NF",
    "OUT_OF_MEMORY": "OOM",
    "PREEMPTED": "PR",
    "REQUEUE_FED": "RF",
    "REQUEUE_HOLD": "RH",
    "REQUEUED": "RQ",
    "RESIZING": "RS",
    "RESV_DEL_HOLD": "RD",
    "REVOKED": "RV",
    "SIGNALING": "SI",
    "SPECIAL_EXIT": "SE",
    "STAGE_OUT": "SO",
    "STOPPED": "ST",
    "SUSPENDED": "S",
    "TIMEOUT": "TO",
}


def _normalize_slurm_job_state(state_raw):
    if state_raw is None:
        return ""
    state = str(state_raw).strip().upper()
    if state == "":
        return ""
    m = re.match(r"^([A-Z_]+)", state)
    if m is not None:
        state = m.group(1)
    return SLURM_STATE_NAME_TO_CODE.get(state, state)


def _normalize_uge_job_state(state_raw, queue_name=""):
    state = str(state_raw).strip()
    state_lower = state.lower()
    if ("e" in state_lower) or ("d" in state_lower):
        return "F"
    if ("q" in state_lower) or (state_lower in {"h", "w"}):
        return "Q"
    if (str(queue_name).strip() != "") or any(marker in state_lower for marker in ["r", "s", "t"]):
        return "R"
    return ""


def slurm_bucket(state):
    state = _normalize_slurm_job_state(state)
    if state in SLURM_RUNNING_STATES:
        return "running"
    if state in SLURM_PENDING_STATES:
        return "queued"
    if state in SLURM_ERROR_STATES:
        return "failed"
    return "other"


def uge_bucket(state, queue_name=""):
    return {"R": "running", "Q": "queued", "F": "failed"}.get(
        _normalize_uge_job_state(state, queue_name), "other"
    )
