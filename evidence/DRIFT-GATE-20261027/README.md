# Drift diagnostic and setup-error regression receipt

Date: 2026-10-27 UTC. Host-only shell tooling, no device access.

- Source commit: `54fac045b90f441d627240de8fb3156cf9997ea7`.
- Base: `67f3135ef88d8f520dfc15a2b2bc3264f53f0d6a`.
- Related change: CUDApp #122.
- Machine class: Linux x86_64, GCC 14.2.0. This is not a GB10 or whole-host-suite qualification.

## What was reproduced

The old classifier pipes `diff` through `grep` under `pipefail`. Normal `diff` status 1 makes the condition false even when code changed. An early-exiting `grep -q` can also produce broken pipes. The diagnostic then reports code differences as licence-header-only. The blob-based drift decision is separate.

Direct mode also ignored failures while building or comparing the map. A nonexistent Omega ref, or a directory used as BASELINE, could print PASS after an error. The existing workflow self-test already rejected an invalid ref at its initial map call; no workflow bypass is claimed.

The repair checks blob reads, preserves normal diff status 1, consumes the whole diff, rejects invalid baseline input, and propagates setup failures as NOT_RUN with exit 2. The blob predicate and `map.sh` are unchanged. The retained header-word classifier is a textual heuristic, not a legal licensing or semantic-harmlessness determination.

## MEASURED results

- `regression.log`: `make test-drift`, exit 0, final `DRIFT REGRESSION TESTS PASS`.
- `original-regression.log`: the same test script with the base commit's unmodified `drift.sh` and `map.sh` in an isolated fixture, exit 1. It exposes the classification and error-handling failures before the repair.
- `host-attempt.log`: `make test`, exit 2 at unchanged AArch64 assembly in `src/omega_blackwell_engine.c:558` and `:393` on x86_64, after the new drift tests pass. No instruction substitution, skipped gate, or full-suite PASS claim.
- `bash -n tools/extract/drift.sh tools/extract/test-drift.sh` and `git diff --check`: exit 0.

Fixtures cover code-only, header-like-only, mixed, identical, large and binary differences; unchanged blob gating; built-in negative controls; invalid refs; baseline directory/read failures; missing blobs; diff errors; and one-time setup errors in each self-test control. All repositories are temporary local fixtures. No remote, account, driver, service, or GPU is touched by these tests.

An independent local review found no remaining must-fix issue in the scoped repair. The Makefile carry onto the named base was separately reviewed to retain its existing host targets and add only `test-drift`.

## Re-run

```sh
make test-drift
bash -n tools/extract/drift.sh tools/extract/test-drift.sh
git diff --check
make test
```

The supported ARM host CI result must be evaluated separately on the PR's exact tested tree. Existing historical receipts were not modified.
