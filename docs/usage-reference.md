# Usage reference

See the [README](../README.md) for installation, live commands, and accuracy
limits. The examples below run from the repository root after installation and
use only synthetic fixtures. No cluster access is needed.

## Offline examples

Grid Engine, including the separate all-user jobs and site quota inputs:

```bash
kfbatch batch --scheduler uge --stat_command "qstat -F" \
  --example_file tests/fixtures/age/qstat_f_1.txt \
  --uge_job_example_file tests/fixtures/age/qstat_all_users.txt \
  --uge_qfree_example_file tests/fixtures/age/qfree.txt \
  --current_user user_a
```

The cluster slot totals are `all:R/Q/F=4/248/5`. For Slurm, use the complete
[README fixture example](../README.md#useful-options).

```bash
kfbatch quota \
  --quota-example-file tests/fixtures/quota/normalized.txt \
  --current-user user_a
```

This prints personal and shared group records, including group usage `71.2TiB`.
Quota has no file-output option; it prints the table and notes to stdout.

## Commands, defaults, and precedence

Use `kfbatch -h`, `kfbatch batch -h`, and `kfbatch quota -h` for option names and
accepted values. Options are optional unless a selected provider requires a
command: `quota --provider lustre` and `--provider custom` require
`--quota-command` for live queries. A quota fixture bypasses that requirement.
Batch options generally use underscores; do not assume hyphenated aliases exist.

There is no kfbatch configuration file or dedicated environment-variable layer.
Explicit CLI values replace defaults. A nonempty `*_example_file` replaces only
its corresponding command, even if that command is empty. Batch fixtures do not
replace the other scheduler queries. The scheduler is still chosen by
`--scheduler` or the executable name in `--stat_command`, not by fixture contents.
`--scheduler` alone does not change command defaults.

The effective local OS user supplies the default identity, not `USER`/`LOGNAME`.
Use `--current_user` for remote/synthetic batch identities or `--current-user`
for quota. These select report identities, not the credentials used to run tools.
Quota identity and filesystem arguments filter returned records; they do not
rewrite the underlying query to fetch another user's quota.

Commands are split into arguments and executed without a shell. Pipes,
redirection, shell variable expansion, and `~` expansion inside command strings
are not performed. A wrapper must supply the expected scheduler output and any
remote auxiliary queries: overriding `--stat_command` alone leaves auxiliary
commands local. Direct `squeue` calls replace format/header flags with kfbatch's
full tab-separated format and remove `SQUEUE_*` environment variables; wrappers
must arrange their own output format. Direct `quota`, `lfs`, and `lfsq` calls set
`LC_ALL=C`; other environment variables are inherited.

Grid Engine resource snapshots are read consecutively without a polling delay.
With a fixture, `--niter` rereads that same file. There is no persistent cache or
resume state. A new invocation performs new reads/queries.

## Quota input and interpretation

The normalized wrapper format requires a header with `scope owner filesystem
bytes_used`; see [the synthetic example](../tests/fixtures/quota/normalized.txt).
Use `self` or `group` for scope and exact owner names. Optional columns are listed
in the [README](../README.md#disk-quota). Keep every row aligned with its header:
whitespace and tab separators collapse empty fields, so use `-` for missing
optional values, or pipe separators to retain empty fields.

Space values without suffixes are KiB, including in `bytes_*` columns. File counts
are counts; suffixes k/M/G/T use powers of 1000. Standard tables use their declared
space/file header scale (`Gbytes` uses 1024 cubed and `kfiles` uses 1000).
Zero limits mean unlimited; unavailable or unlimited limits both display as `-`.
Unknown used-space values cannot form a record. Invalid or misaligned rows may
be skipped; a successful parse does not validate every input row. No matching
records is an error, not a zero-usage report.

`--filesystem home` matches `home`, `/home`, and descendants of `/home/`, not
names such as `home_user_a`. Other filters match the full filesystem name/path or
its final path component. `--provider auto` tries available `lfsq`, then Linux
`quota`, until one yields records; it does not merge providers. An explicit
`--quota-command` replaces that search. Use `--provider posix` with a custom
Linux quota command to retain valid over-limit output on exit status 1.

## TSV columns and units

The TSVs contain headers, no DataFrame index, and no console summary/legend rows.
The schemas differ between schedulers. They are parsed tables, not a serialization
of the compact display. `--scope`, `--group-id`, and `--by-user` do not filter them.

| Node column | Meaning |
| --- | --- |
| `queue_name`, `node_name` | Queue/partition instance and host identity; one host may appear in several partitions/queues, so do not sum aliases as distinct physical capacity. |
| `ncore_resv`, `ncore_used`, `ncore_total`, `ncore_available` | Reserved, used, total, and available CPU/slot counts. Read these with `status`; abnormal Grid Engine rows can retain positive raw availability. |
| `status` | Empty for normal instances; otherwise scheduler/metadata status. |
| `hc:mem_req`, `hl:mem_total` | Available request headroom and total memory, converted to GiB (despite the historical attribute names). Slurm headroom uses allocated memory, not OS free memory, and reservation adjustments are applied before export. |
| `hc:mem_req_known`, `hl:mem_total_known` | Whether the corresponding value is known; false must not be interpreted as free capacity. |
| `hc:mem_req_unit`, `hl:mem_total_unit` | `GiB` for known values, empty otherwise. Unknown memory is an empty TSV cell. |

Other node fields include `qtype`, `np_load`, and `arch`; Slurm also includes
`slurm_state` and `reservation_name`. Reservation processing can add
`reservation_cores`, `reservation_mem_mb` (MiB), and `reservation_accessible`.
These describe reservation adjustments/access, not job requests. Grid Engine qfree quota/launch values are
console-only and do not appear in the node TSV. The console can use qfree RAM
totals, while the TSV retains queue-instance memory from qstat.

| Job column | Meaning |
| --- | --- |
| `job_id`, `user`, `name`, `state` | Scheduler identifiers and state; preserve IDs as strings, including array expressions. Rows are not expanded into one row per array task. |
| `total_slots` | Slurm: number of array tasks (not requested CPUs). Grid Engine: `slots` per task multiplied by array-task count. |
| `task_count_estimated` | True when complete task counts could not be established. |
| Slurm `partition`, `account`, `num_nodes`, `req_cpus`, `req_mem`, `time_limit`, `elapsed_time`, `node_or_reason`, `pending_reason`, `resource_fields_complete` | Parsed job fields. `req_mem` retains scheduler text/suffixes, not normalized GiB; a unitless request remains ambiguous for launch calculations. Legacy input may have missing account/resource data. |
| Grid Engine `queue_name`, `prior`, `slots`, `ja_task_id`, `submit_or_start_date`, `submit_or_start_time` | Parsed job fields; pending jobs may have no queue instance. |

Diagnostic notes and DataFrame metadata (including whether all-user collection
succeeded) are not exported. Retain the console output and exit status alongside
TSVs when completeness matters.

Writes replace existing files, retain their permission bits, and use temporary
files in the destination directory. New files use private temporary-file
permissions. Job output is written before node output, so a later failure can
leave a new job file alongside an older node file. Normal completion cleans up
temporary files; interrupted runs are not a transaction. `--out` and `--out_nodes`
can be used together only with the same argument string. Node and job paths must
resolve to different paths. See the [README](../README.md#tsv-output) for output
path requirements.
