# C6 receipt: Omega's own disassembler

Row C6 of `docs/OMEGA.md`: our own sm_121 disassembler agrees with nvdisasm on every library kernel, and every check
that called nvdisasm calls ours. Produced by one script run on the Spark (2026-10-21), physics reference pinned at
`~/workspace/physics-l386`. Each log ends with the exit code of the command it records.

Re-run 2026-10-24 on the L386 train tip (this branch rebased over c5/isolation, c3/gemm, c3/reduce-scan and main 0d72560a; physics pinned at fb2ac4c, the commit physics.lock names): the `*-20261024.log` files beside the originals, one per row, same commands, PHYSICS_DIR=~/workspace/l386-physics-pr. New that day: rows 11 and 12, the ZERO-NVDISASM gate and its red check, and the judge gate `tests/gates/omega-sm121dis.gate` (step word `sm121dis` in tests/gates/run.sh) that runs rows 01 and 11 with their red checks on the Spark judge.

| Log | What it checks | Needed | Result |
|---|---|---|---|
| 01-test-sm121dis | all 331 library kernels (59,152 words) word for word against nvdisasm, spacing included | exit 0 | exit 0, 0 kernels differ; 2026-10-24 on the L386 train tip c8d7d5294: exit 0, 331 kernels, 59,152 words, 0 differ, oracle nvdisasm release 13.0 V13.0.85; red decoder fails all 331 (01-test-sm121dis-20261024.log). Two corpus programs did not build on that tip, gpu_cmdlist_test and gpu_gpr_boundary_test, both main regressions (missing initializers under -Werror, fixer L825); the corpus still reached 331 kernels |
| 02-reduce-on-sm121dis | the E1 reduction listing check, now on `build/sm121dis` | exit 0 | exit 0; 2026-10-24 on the L386 train tip c8d7d5294: exit 0 (02-...-20261024.log) |
| 03-divsqrt-on-sm121dis | the divide and square root listing check, on ours | exit 0 | exit 0; 2026-10-24 on the L386 train tip c8d7d5294: exit 0, VERDICT PASS (03-...-20261024.log) |
| 04-ldst-on-sm121dis | the load and store listing check, on ours | exit 0 | exit 0; 2026-10-24 on the L386 train tip c8d7d5294: exit 0, VERDICT PASS (04-...-20261024.log) |
| 05, 06, 07 (red) | the same three checks with a decoder carrying one planted fault | must fail | all fail; 2026-10-24 on the L386 train tip c8d7d5294: all fail, exits 1, 1, 2 (05/06/07-...-20261024.log) |
| 08-gpu_* | the GPU tests whose listing checks now link our decoder | exit 0 | exit 0; 2026-10-24 on the L386 train tip c8d7d5294: exit 0 on all four: attention 107 checks (184 of 184 and 216 of 216 instruction lines decoded), elementwise 24, fp8 4,997, reconv 119 (08-...-20261024.log) |
| 08-gpu_attention_test-hd128 | head size 128, recorded for completeness | | exit 1, 14 failures, none from the disassembler: a known kernel-size bug, fixed separately (ledger L407); 2026-10-24 on the L386 train tip c8d7d5294: exit 0, 107 checks, 624 of 624 and 512 of 512 instruction lines decoded: the kernel-size bug is gone on the train tip (08-gpu_attention_test-hd128-20261024.log) |
| 09-prime-race-host | the sieve benchmark's host test, its listing check on ours | exit 0 | exit 0; 2026-10-24 on the L386 train tip c8d7d5294: exit 0, VERDICT PASS (09-...-20261024.log) |
| 10 (red) | the previous decoder, which printed `NOP ;` where nvdisasm prints `NOP;` after the last EXIT | must fail | fails on all 331 |
| 11-zero-nvdisasm-20261024 | ZERO-NVDISASM (bar C6, second half): no file outside the oracle harness names the CUDA toolkit; the five GPU host tiers and the three provenance checks run with a decoy nvdisasm first on PATH, the CUDA directories removed from PATH and strace on execve (tools/sm121dis/zero_nvdisasm.sh, `make test-zero-nvdisasm`, gate OMEGA-SM121DIS) | exit 0 | exit 0: 8 commands, decoy called 0 times, no execve of nvdisasm, static clean (2026-10-24, tip c8d7d5294) |
| 12-red-zero-nvdisasm-planted-calls-20261024 | the same gate with two planted shell-outs, `nvdisasm` by name and by the toolkit path | must fail | exit 1: the decoy caught the call by name, strace caught both (2026-10-24, tip c8d7d5294) |
| 13-corpus-producer-controls-20261025 | a missing or failing corpus producer must invalidate the corpus, permanently (tools/sm121dis/corpus_selftest.sh, run by `make test-sm121dis`) | before: exit 0 with 31 producers missing; after: exit 127 and a verdict, producer diagnostics kept; a producer that exits 42 stops the corpus with 42 | PASS: "missing and failing corpus producers are refused"; before/after captured in the log (2026-10-25) |
| 14-test-sm121dis-20261025 | the whole C6 host tier again after that repair, on the re-stacked train tip | exit 0 | exit 0: 402 kernels, 74,096 words, 0 differ; red decoder fails all 402. The four GPU families the corpus used to skip (scan, GEMM, streams, isolation) are in it, and the two programs that did not build on 2026-10-24 (gpu_cmdlist_test, gpu_gpr_boundary_test) build again after L825 landed |

nvdisasm stays only as the reference in 01 and the red checks. No kernel and no other check depends on it.

2026-10-25, L386 resumed (Codex session pickup): the corpus script reported success while 31 of its producer programs
were missing (the two main regressions above, plus programs no longer built), so a hidden or superseded producer could
silently shrink the corpus. `tools/sm121dis/corpus.sh` now refuses the first failed producer, keeps that producer's own
output, and exits with its status; `tools/sm121dis/corpus_selftest.sh` holds that behaviour permanently, and the corpus
now includes the scan, GEMM, streams and isolation GPU programs. Row 13 records the before/after, row 14 the full tier
on the repaired corpus (402 kernels).
