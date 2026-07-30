import argparse
import sys


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
        raise argparse.ArgumentTypeError("Expected a non-negative number.") from error
    if number < 0:
        raise argparse.ArgumentTypeError("Expected a non-negative number.")
    return number


def _build_parser():
    parser = argparse.ArgumentParser(description="A toolkit for the batch job management.")
    parser.add_argument(
        "--stat_command",
        metavar="command",
        default="squeue",
        type=str,
        required=False,
        action="store",
        help="default=%(default)s: PATH to the command that shows cluster-wide batch job status.",
    )
    parser.add_argument(
        "--example_file",
        metavar="PATH",
        default="",
        type=str,
        required=False,
        action="store",
        help="default=%(default)s: PATH to a file with --stat_command stdout. "
        "Only for demo and debugging.",
    )
    parser.add_argument(
        "--slurm_node_command",
        metavar="command",
        default="scontrol show node -o",
        type=str,
        required=False,
        action="store",
        help="default=%(default)s: Command for SLURM node status/capacity details.",
    )
    parser.add_argument(
        "--slurm_node_example_file",
        metavar="PATH",
        default="",
        type=str,
        required=False,
        action="store",
        help="default=%(default)s: PATH to a file with --slurm_node_command stdout.",
    )
    parser.add_argument(
        "--slurm_partition_command",
        metavar="command",
        default="scontrol show partition -o",
        type=str,
        required=False,
        action="store",
        help="default=%(default)s: Command for SLURM partition metadata.",
    )
    parser.add_argument(
        "--slurm_partition_example_file",
        metavar="PATH",
        default="",
        type=str,
        required=False,
        action="store",
        help="default=%(default)s: PATH to a file with --slurm_partition_command stdout.",
    )
    parser.add_argument(
        "--slurm_reservation_command",
        metavar="command",
        default="scontrol show reservation",
        type=str,
        required=False,
        action="store",
        help="default=%(default)s: Command for active SLURM reservations.",
    )
    parser.add_argument(
        "--slurm_reservation_example_file",
        metavar="PATH",
        default="",
        type=str,
        required=False,
        action="store",
        help="default=%(default)s: PATH to a file with --slurm_reservation_command stdout.",
    )
    parser.add_argument(
        "--slurm_prio_command",
        metavar="command",
        default="sprio",
        type=str,
        required=False,
        action="store",
        help="default=%(default)s: Command for SLURM pending-job priority breakdown.",
    )
    parser.add_argument(
        "--slurm_prio_example_file",
        metavar="PATH",
        default="",
        type=str,
        required=False,
        action="store",
        help="default=%(default)s: PATH to a file with --slurm_prio_command stdout.",
    )
    parser.add_argument(
        "--ntop",
        metavar="INT",
        default=3,
        type=int,
        required=False,
        action="store",
        help="default=%(default)s: Number of top available nodes to print.",
    )
    parser.add_argument(
        "--all_tiers",
        metavar="[yes,no]",
        default="no",
        type=parse_bool,
        required=False,
        action="store",
        help='default=%(default)s: Whether to show all nodes tied to the "ntop" resources.',
    )
    parser.add_argument(
        "--niter",
        metavar="INT",
        default=1,
        type=int,
        required=False,
        action="store",
        help="default=%(default)s: Number of qstat resource snapshots to merge using minimum availability.",
    )
    parser.add_argument(
        "--uge_job_command",
        metavar="command",
        default="qstat -u '*'",
        type=str,
        required=False,
        action="store",
        help="default=%(default)s: AGE/UGE/SGE command for all-user job status. "
        "Text output preserves pending job-array ranges.",
    )
    parser.add_argument(
        "--uge_job_example_file",
        metavar="PATH",
        default="",
        type=str,
        required=False,
        action="store",
        help="default=%(default)s: PATH to captured --uge_job_command stdout.",
    )
    parser.add_argument(
        "--uge_qfree_command",
        metavar="command",
        default="qfree -c",
        type=str,
        required=False,
        action="store",
        help="default=%(default)s: Optional site command for user/group queue quotas and 2G launch slots.",
    )
    parser.add_argument(
        "--uge_qfree_example_file",
        metavar="PATH",
        default="",
        type=str,
        required=False,
        action="store",
        help="default=%(default)s: PATH to captured --uge_qfree_command stdout.",
    )
    parser.add_argument(
        "--out",
        metavar="PATH",
        default="",
        type=str,
        required=False,
        action="store",
        help="default=%(default)s: Save the parsed node/resource table (legacy alias).",
    )
    parser.add_argument(
        "--out_nodes",
        metavar="PATH",
        default="",
        type=str,
        required=False,
        action="store",
        help="default=%(default)s: Save the parsed node/resource table.",
    )
    parser.add_argument(
        "--out_jobs",
        metavar="PATH",
        default="",
        type=str,
        required=False,
        action="store",
        help="default=%(default)s: Save the parsed job table.",
    )
    parser.add_argument(
        "--command_timeout",
        metavar="SECONDS",
        default=60.0,
        type=parse_nonnegative_float,
        required=False,
        action="store",
        help="default=%(default)s: Scheduler command timeout; 0 disables the timeout.",
    )
    parser.add_argument(
        "--exclude_abnormal_node",
        metavar="[yes,no]",
        default="yes",
        type=parse_bool,
        required=False,
        action="store",
        help="default=%(default)s: Whether to report nodes with abnormal status, such as a(larm) and d(isabled).",
    )
    parser.add_argument(
        "--show_launch_heuristic",
        metavar="[yes,no]",
        default="yes",
        type=parse_bool,
        required=False,
        action="store",
        help="default=%(default)s: Whether to show reservation-adjusted, priority-aware SLURM launch ceilings.",
    )
    return parser


def main(argv=None):
    if argv is None:
        argv = sys.argv
    argv = list(argv)
    parser = _build_parser()
    args = parser.parse_args(argv[1:])
    from kfbatch.errors import KFBatchError
    from kfbatch.stat import stat_main

    try:
        stat_main(args)
    except KFBatchError as e:
        print(str(e), file=sys.stderr)
        return 1
    return 0
