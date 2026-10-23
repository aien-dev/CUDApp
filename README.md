# Recovery archive, 2026-10-23

This branch is a preservation index, not a product implementation or merge proposal.
Original committed work and unreviewed working/staged/conflict snapshots are reachable
through this commit's parent history. The index tree deliberately contains only recovery
records. No active branch, original staging area, service or working file was replaced.

## Contents

- 5 local branch references absent from the published heads/PR histories inspected.
- 7 working-folder snapshots, 0 separate staged snapshots, 0 unresolved conflict alternatives.
- 6963 excluded untracked generated/model/nested-repository entries; see working-snapshots.json.
- 20 previously unreferenced commits, including the new recovery snapshots.
- Published main at capture: 80f3a2492eef315106feebfdffcb32de39be7d95.

The capture covered discovered workspace clones and their registered worktrees for
Sovereign Core, Omega and CUDApp, plus their existing omega-bridge migration repository.
Ignored files, arbitrary reflog-only objects and unrelated repositories were not swept.
Six missing registered worktree directories were recorded in the cross-repository summary.
Known compiler/device owners retain implementation ownership. No test, quality, hardware,
license-transition or release approval is implied by publication. Historical Rust/C/.osc
is retained here as recovery history, not adopted as the final first-party implementation.

## Recover saved work

Use the commit links below, or fetch this branch and create a new local branch at the
desired commit. Do not reset an active working folder. Working snapshots capture eligible
on-disk files; staged and conflict snapshots preserve additional index alternatives.
Generated/model artifacts and nested repositories are excluded explicitly.

### Previously unpublished branch tips

| Original local branch | Saved commit |
|---|---|
| backup/pre-selftest-rewrite | [f28b2ec5dfdc](https://github.com/aien-dev/CUDApp/commit/f28b2ec5dfdca0e947b714b98ffe1ee5d468329f) |
| feat/native-amd-agx-lab | [e9ff18accda8](https://github.com/aien-dev/CUDApp/commit/e9ff18accda8372584d20cd14bd9db3747b7e733) |
| runtime/r1-api | [1ee104fda3fb](https://github.com/aien-dev/CUDApp/commit/1ee104fda3fbc9e018e99e3ea37cce185eddf43b) |
| runtime/r2-hello | [ac6460b9146a](https://github.com/aien-dev/CUDApp/commit/ac6460b9146a5b9b0ca73b01889f8841e4ea99f2) |
| runtime/r3-portability | [e8501a59321a](https://github.com/aien-dev/CUDApp/commit/e8501a59321a8e3e6a4eeff86bc938b4d31753ee) |

### Unfinished working-folder snapshots

| Source folder | Saved commit |
|---|---|
| cudapp-omega | [8c58ad133f1e](https://github.com/aien-dev/CUDApp/commit/8c58ad133f1e1d983b4c87edaa5c5085fd52ddc1) |
| cudapp-quantum-weights | [e2bb05ea2165](https://github.com/aien-dev/CUDApp/commit/e2bb05ea216539bba001c3c50a26b99aa4fdc478) |
| cudapp-roadmap-memory24 | [6b9025edd577](https://github.com/aien-dev/CUDApp/commit/6b9025edd5777e41e0c35a3270a6551285c071b6) |
| cudapp-roadmap-numerics | [e41af6004b00](https://github.com/aien-dev/CUDApp/commit/e41af6004b00c3fafe795d09b005b112ccfaa3dc) |
| cudapp-swap | [afcf3dc302e8](https://github.com/aien-dev/CUDApp/commit/afcf3dc302e879b70e18716edc50c36ee80ee76b) |
| cudapp-verification-l478 | [346acaba9eb1](https://github.com/aien-dev/CUDApp/commit/346acaba9eb19504b69d909d1240b0c16b14ae40) |
| cudapp-roadmap-host15-parent-x86 | [06bba1d4836f](https://github.com/aien-dev/CUDApp/commit/06bba1d4836f9456395ce1bd0552603794fed23a) |

## Verification and limits

Unpublished blobs were read and checked for credential filenames, common token/private-key
patterns and oversized objects. Two CUDApp audit-selftest blobs contain intentional fake
secret negative controls; those exact blobs were reviewed. Pattern screening is not a
formal security audit. Original source tests were not rerun for recovery publication.
Working files selected for capture were reread and compared before creating snapshots.
Root publication and reachability are verified separately after upload.
