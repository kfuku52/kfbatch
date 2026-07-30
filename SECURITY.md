# Security policy

## Supported versions

Security fixes are applied to the latest release and the `main` branch.

## Reporting a vulnerability

Please use GitHub's private vulnerability reporting for this repository:

<https://github.com/kfuku52/kfbatch/security/advisories/new>

Do not include credentials, private scheduler captures, user/account lists, internal
host names, or network addresses in a public issue. Include only the minimum
synthetic input needed to demonstrate the problem.

## Data handled by kfbatch

`kfbatch` invokes scheduler commands locally and prints or writes their parsed
results. It does not intentionally send scheduler data over the network. TSV output
can contain job, user, account, queue, and node identifiers; protect it according to
your site's policy.
