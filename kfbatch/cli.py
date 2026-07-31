import argparse
import math
import shutil
import sys

from kfbatch import __version__

MAX_NITER = 100


def _default_stat_command():
    if shutil.which("squeue") is not None:
        return "squeue"
    if shutil.which("qstat") is not None:
        return "qstat -F"
    return "squeue"


def parse_bool(value):
    if isinstance(value, bool):
        return value
    txt = str(value).strip().lower()
    if txt in {"y", "yes", "t", "true", "on", "1"}:
        return True
    if txt in {"n", "no", "f", "false", "off", "0"}:
        return False
    raise argparse.ArgumentTypeError("Expected one of yes/no, true/false, on/off, or 1/0.")


def parse_nonnegative_float(value):
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("Expected a finite non-negative number.") from error
    if (not math.isfinite(number)) or number < 0:
        raise argparse.ArgumentTypeError("Expected a finite non-negative number.")
    return number


def parse_positive_int(value):
    try:
        number = int(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("Expected a positive integer.") from error
    if number < 1:
        raise argparse.ArgumentTypeError("Expected a positive integer.")
    return number


def _build_parser(*, prog="kfbatch", add_help=True):
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Compact resource summaries for Slurm and AGE/UGE/SGE batch clusters.",
        allow_abbrev=False,
        add_help=add_help,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "--stat_command",
        metavar="command",
        default=_default_stat_command(),
        help=(
            "default=%(default)s (auto-detected): Command that shows cluster-wide batch job status."
        ),
    )
    parser.add_argument(
        "--scheduler",
        choices=["auto", "slurm", "uge"],
        default="auto",
        help="default=%(default)s: Scheduler override for wrappers or remote commands.",
    )
    parser.add_argument(
        "--current_user",
        metavar="NAME",
        default="",
        help=(
            "default=effective local user: Scheduler user override for remote commands "
            "and synthetic fixtures."
        ),
    )
    parser.add_argument(
        "--example_file",
        metavar="PATH",
        default="",
        help="default=%(default)s: File containing --stat_command stdout for demos/tests.",
    )
    parser.add_argument(
        "--slurm_node_command",
        metavar="command",
        default="scontrol show node -o",
        help="default=%(default)s: Command for Slurm node status/capacity details.",
    )
    parser.add_argument(
        "--slurm_node_example_file",
        metavar="PATH",
        default="",
        help="default=%(default)s: File containing --slurm_node_command stdout.",
    )
    parser.add_argument(
        "--slurm_partition_command",
        metavar="command",
        default="scontrol show partition -o",
        help="default=%(default)s: Command for Slurm partition metadata.",
    )
    parser.add_argument(
        "--slurm_partition_example_file",
        metavar="PATH",
        default="",
        help="default=%(default)s: File containing --slurm_partition_command stdout.",
    )
    parser.add_argument(
        "--slurm_reservation_command",
        metavar="command",
        default="scontrol show reservation",
        help="default=%(default)s: Command for active Slurm reservations.",
    )
    parser.add_argument(
        "--slurm_reservation_example_file",
        metavar="PATH",
        default="",
        help="default=%(default)s: File containing --slurm_reservation_command stdout.",
    )
    parser.add_argument(
        "--slurm_prio_command",
        metavar="command",
        default="sprio -h -o '%i|%r|%Y|%S|%A|%F|%J|%P'",
        help="default=%(default)s: Command for a stable Slurm priority-factor format.",
    )
    parser.add_argument(
        "--slurm_prio_example_file",
        metavar="PATH",
        default="",
        help="default=%(default)s: File containing --slurm_prio_command stdout.",
    )
    parser.add_argument(
        "--slurm_share_command",
        metavar="command",
        default="sshare -a -P",
        help="default=%(default)s: Command for Slurm FairShare association data.",
    )
    parser.add_argument(
        "--slurm_share_example_file",
        metavar="PATH",
        default="",
        help="default=%(default)s: File containing --slurm_share_command stdout.",
    )
    parser.add_argument(
        "--ntop",
        metavar="INT",
        default=1,
        type=parse_positive_int,
        help="default=%(default)s: Maximum top nodes shown per resource and partition.",
    )
    parser.add_argument(
        "--all_tiers",
        metavar="[yes,no]",
        default="no",
        type=parse_bool,
        help='default=%(default)s: Include nodes tied with the "ntop" resource tier.',
    )
    parser.add_argument(
        "--niter",
        metavar="INT",
        default=1,
        type=parse_positive_int,
        help=f"default=%(default)s: Number of qstat snapshots merged conservatively (max {MAX_NITER}).",
    )
    parser.add_argument(
        "--uge_job_command",
        metavar="command",
        default="qstat -u '*'",
        help="default=%(default)s: AGE/UGE/SGE all-user job-status command.",
    )
    parser.add_argument(
        "--uge_job_example_file",
        metavar="PATH",
        default="",
        help="default=%(default)s: File containing --uge_job_command stdout.",
    )
    parser.add_argument(
        "--uge_qfree_command",
        metavar="command",
        default="qfree -c",
        help="default=%(default)s: Optional site quota and 2G launch-slot command.",
    )
    parser.add_argument(
        "--uge_qfree_example_file",
        metavar="PATH",
        default="",
        help="default=%(default)s: File containing --uge_qfree_command stdout.",
    )
    parser.add_argument(
        "--out",
        metavar="PATH",
        default="",
        help="default=%(default)s: Save the node/resource table (legacy alias).",
    )
    parser.add_argument(
        "--out_nodes",
        metavar="PATH",
        default="",
        help="default=%(default)s: Save the parsed node/resource table.",
    )
    parser.add_argument(
        "--out_jobs",
        metavar="PATH",
        default="",
        help="default=%(default)s: Save the parsed job table.",
    )
    parser.add_argument(
        "--command_timeout",
        metavar="SECONDS",
        default=60.0,
        type=parse_nonnegative_float,
        help="default=%(default)s: Scheduler command timeout; 0 disables it.",
    )
    parser.add_argument(
        "--exclude_abnormal_node",
        metavar="[yes,no]",
        default="yes",
        type=parse_bool,
        help="default=%(default)s: Exclude abnormal nodes from top-node displays.",
    )
    parser.add_argument(
        "--show_launch_heuristic",
        metavar="[yes,no]",
        default="yes",
        type=parse_bool,
        help="default=%(default)s: Show reservation-adjusted CPU/RAM resource ceilings.",
    )
    parser.add_argument(
        "--show_fairshare_rank",
        metavar="[yes,no]",
        default="yes",
        type=parse_bool,
        help="default=%(default)s: Show the current Slurm association FairShare rank.",
    )
    parser.add_argument(
        "--scope",
        choices=["overview", "self", "group", "all"],
        default="overview",
        help=(
            "default=%(default)s: Job summary scope. overview shows self, group, "
            "and cluster totals when group data is available."
        ),
    )
    parser.add_argument(
        "--group-id",
        "--group_id",
        dest="group_id",
        metavar="NAME",
        default="",
        help=(
            "default=auto: Slurm account or AGE/UGE group to summarize. "
            "Use this when automatic group discovery is unavailable."
        ),
    )
    parser.add_argument(
        "--by-user",
        "--by_user",
        dest="by_user",
        action="store_true",
        help="Break down group job totals by user.",
    )
    return parser


def _build_root_parser():
    parser = argparse.ArgumentParser(
        prog="kfbatch",
        description="Inspect batch-cluster jobs/resources and filesystem quotas.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="{batch,quota}")
    subparsers.add_parser(
        "batch",
        parents=[_build_parser(prog="kfbatch batch", add_help=False)],
        add_help=True,
        allow_abbrev=False,
        help="Show batch jobs and scheduler resource availability.",
        description="Compact resource summaries for Slurm and AGE/UGE/SGE batch clusters.",
    )

    from kfbatch.quota import add_quota_arguments

    quota_parser = subparsers.add_parser(
        "quota",
        add_help=True,
        allow_abbrev=False,
        help="Show personal and group filesystem quotas.",
        description="Compact personal and group filesystem quota summaries.",
    )
    add_quota_arguments(quota_parser)
    return parser


def _parse_command(argv):
    if not argv:
        return "batch", _build_parser().parse_args([])
    if argv[0] in {"-h", "--help", "--version"}:
        args = _build_root_parser().parse_args(argv)
        return getattr(args, "command", None), args
    if argv[0].startswith("-"):
        return "batch", _build_parser().parse_args(argv)
    args = _build_root_parser().parse_args(argv)
    return args.command, args


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    command, args = _parse_command(list(argv))

    from kfbatch.errors import KFBatchCommandError, KFBatchUsageError

    try:
        if command == "quota":
            from kfbatch.quota import quota_main

            quota_main(args)
        else:
            from kfbatch.stat import stat_main

            if args.niter > MAX_NITER:
                raise KFBatchUsageError(f"--niter must be <= {MAX_NITER}.")
            stat_main(args)
    except KFBatchUsageError as error:
        print(str(error), file=sys.stderr)
        return 2
    except KFBatchCommandError as error:
        print(str(error), file=sys.stderr)
        return 1
    return 0
