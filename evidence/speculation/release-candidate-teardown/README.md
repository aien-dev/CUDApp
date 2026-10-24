# Release candidate: native cleanup ownership repair

Refs #65. This candidate is CUDApp `main` 22e9d9b plus the two commits of
PR #71, cherry-picked unchanged. PR #71 merged into
`experiment/27b-adaptive-sizing` (52db648), not `main`. The four files it
changes are blob-identical to 52db648. No other experiment code is included.
This candidate does not finish #65.

Candidate code commit: `2e2421694f6110bd84885ae0b2d3caf249809d34`.
Exact hashes and raw results are in `receipt.json`.

## What was run, on Linux AArch64 (GB10, driver 580.173.02)

| Check | Result |
| --- | --- |
| Pinned compiler rebuilt from seed, Omega d7390d88, fresh clone | omc `9749f388…` matches the pin; stage1 = stage2 = stage3 |
| `run-teardown-host.sh`, original f9cd649 cleanup | exit 1, `NATIVE_TEARDOWN_HOST_FAIL` (test assertion, no compile error) |
| `run-teardown-host.sh`, repaired cleanup | exit 0, `NATIVE_TEARDOWN_HOST_PASS` |
| Same, second run | identical unit and ELF hashes |
| Ten mutants of the repaired cleanup (`mutants.sh`) | 7 killed by assertion, 1 detected by trap 5, 2 survived (see below) |
| Vector host regression | `VECTOR_CHECKS_HOST_PASS`, byte-identical to the 0cf8cd7 receipt |
| GB10 vector test, wrong-math control, corrected recovery | exit 0, `NATIVE_SM121_VECTOR_PASS`, 0 failed syscalls |

The fixture covers retained mapping ownership after a refused unmap (BAR,
channel, buffer and the unpublished mapping), repeated refusal across two
shutdown attempts, eventual release on a later retry, one close attempt per
descriptor with no repeated driver calls, and a sticky fault that keeps
refusing `nvstart` after cleanup. Mutants M1 to M6 and M8 break one of these
each and are killed by the fixture's own assertion.

The GPU run used a shared spark-window (`--gb10 shared --quiet none`) for
11 seconds of test run (17 seconds of window) beside the live model server. A model-list request to that server succeeded just before and just after; those two checks were not saved as files.
It is a correctness result. No timing or performance is claimed. Older
receipts were used only to check the build recipe; the same recipe applied to
0cf8cd7 reproduces that receipt's unit `b3a36ac6…` and ELF `e3da26b0…`.

## Known gap

The only faulted-mapping case has the context closed (state 2) and faulted
together. Either half of the new `nvcpumap` guard alone satisfies it, so
mutants M7b and M7c survive. Removing the whole guard is caught only by a
runtime trap (exit 105). A faulted context that is still live (state 1) is
not exercised. The repair is unchanged here; a fixture case for that state
is a follow-up.

## Reproduce from a clean checkout

Requires Linux AArch64. Steps 1 to 3 use only the CPU. Step 4 needs a
GB10 and a chip window.

```sh
# 1. Pinned compiler, built from seed (no C toolchain involved)
git clone https://github.com/aien-dev/omega-mv.git omega && cd omega
git checkout --detach d7390d88d07dbf48cfbb0583070599c350f814d3
sh seed/omega-from-seed.sh            # last line RC=0
sha256sum seed/out/omc                # 9749f38841a55917d41924a68bce31463ec2339910bddd5e6cbf6ea4420551db
OMC=$PWD/seed/out/omc; cd ..

# 2. Candidate and teardown red/green
git clone https://github.com/aien-dev/CUDApp.git cudapp && cd cudapp
git checkout --detach <PR head>   # adds only evidence and docs on top of 2e24216
git diff --stat 2e2421694f6110bd84885ae0b2d3caf249809d34 HEAD -- native tests   # empty
sh tests/speculation/run-teardown-host.sh "$OMC"   # PASS line, exit 0

# 3. Mutants (outputs in ../mutants)
sh evidence/speculation/release-candidate-teardown/mutants.sh "$PWD" "$OMC" "$PWD/../mutants"

# 4. GPU test unit: files joined with one blank line after each of the first three
T=tests/speculation
k=$(grep -n '^fn trial ' $T/native-kernel-test.om | cut -d: -f1)
{ cat native/nvrm/context.om; echo; cat native/blackwell/vector.om; echo
  cat native/blackwell/launch.om; echo; head -n $((k-1)) $T/native-kernel-test.om
  cat $T/vector-checks.om; tail -n +$k $T/native-kernel-test.om; } > kernel-unit.om
"$OMC" < kernel-unit.om > kernel.elf && chmod 700 kernel.elf
sha256sum kernel-unit.om kernel.elf   # ff29f45a… and b0754a4e…
# In an admitted chip window only (Spark: spark-window open NAME --gb10 shared --quiet none):
./kernel.elf                          # NATIVE_SM121_VECTOR_PASS, exit 0
```

The vector host regression row is rebuilt with the recipe in
`docs/speculation/NATIVE-KERNEL.md` (declarations before `fn supply` in
`vector-checks-test.om`, then `vector-checks.om`, then the remainder) and runs
on the CPU. `mutants.sh` merges the two scripts that produced `mutants.log`.

Exit 77 or `NOT_RUN` from the teardown script means an unsupported host and
is not a pass.

## Remaining dependencies

- An independent rerun of step 4 on the PR head by someone other than the
  author of this receipt.
- The live-state faulted `nvcpumap` fixture case above.
- Everything else #65 asks for: FP8 arithmetic, the 27B model, speculative
  state rollback, AIENOS attachment and the four-arm comparison.
- Review follow-ups (not defects in this candidate): add `maphold`,
  `mapbind`, `mapaddr`, `mapget`, `mapset`, `mapwrite`, `sleepns` and `monons`
  to the composer's escape guard; pin the before arm to 22e9d9b (same
  `context.om` blob as f9cd649, which lives only on side branches); the early
  `nvcpumap` failure paths ignore a failed descriptor close (pre-existing).
