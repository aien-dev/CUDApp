# CUDApp Native Roadmap Implementation Plan

> For agentic workers: execute the recorded task brief using test-driven development and isolated worktrees; parent performs independent review and verification before integration.

Goal: implement CUDApp issues15-25 under roadmap26 without replacing canonical motor priority5 or the selectable baseline.
Architecture: additive, bounded native experiments beside the existing canonical GPU backend. Portable host code and test-only fixtures do not claim chip qualification. One shared hardware/serving train is owned by the existing motor coordinator. No production default changes without repeated paired complete-model evidence.
Tech stack: existing native C/SASS/Omega sources and simulator; no deployed CUDA userspace or new inference engine. New Omega code uses the existing pinned compiler; don't silently use a C-built substitute for seed-born code where its contract requires seed provenance.
Spec: exact issue bodies/comments frozen in `~/handoffs/cudapp-roadmap-20261022/tracker-snapshot.json` on Spark; canonical https://github.com/aien-dev/CUDApp/issues/26 and linked15-25.
Base: CUDApp `349913f17c083996680c72b32f767b225005155d` `integration/canonical-motor`, NOT default main23233242. This candidate already carries existing C3/C4/C5/C6 and live motor changes. It remains unqualified for production and its L458 review fixes belong to L457, not these workers.
Parent ledger: L465.

## Global constraints
- #5 stays first. Preserve current open Omega614-618; snapshot records their current exact heads. Do not merge the old extraction over carries, rebase existing PRs, or duplicate L457.
- No live motor18091 request/load/restart, no kernel module/driver change, no GPU work before a coordinated explicit slot. Host/simulator gates may run independently. No sensitive timings in parallel.
- No experiment enabled by default. Baseline selectable, fail-closed unsupported shape/contract behavior, current arithmetic and cache guarantees preserved unless an explicit contract labels a comparison.
- Exact source/tool/model/config/prompt pins, commands, hashes, exit status and raw results under evidence/. Fixture values are fixture values, never hardware or whole-model results. Preserve unfavorable results and unknown counters.
- Every gate uses negative controls; no threshold relaxation, skip-as-PASS, or retry-until-green. Full host suite plus targeted tests; no test duplication across shared base unless executable input changed.
- New branch per reviewable issue. No PR auto-closing keywords. Renda owns PR writes; push+READY packet, then parent routes creation and verifies links. Do not read desk keys or bypass guard.
- Never edit other owners' trees, main, coordinator OWNER/reservations/cron, or LEDGER.tsv. Notes only via ledger CLI; parent alone closes work.

## Interface and file ownership
| Producer | Consumer | Interface / collision ruling |
|---|---|---|
| #15 host foundation | all later host tasks | preserves `-include mk/*.mk`; #15 owns root Makefile, engine architecture barriers, tests/omega/run.sh. Other tasks add issue-local mk files and tests, not root edits. |
| #16 contracts | #17 and #18-23 | publish versioned numerical contract ID and validation API/receipt schema first; metrics uses that exact ID. Same worker owns16 then17, reviewed as separate commits/branches. |
| #17 benchmark | #18-23/#25 | measured/estimated/unknown fields, paired raw samples, explicit NOT_RUN, instrumentation mode and complete-model correctness checks. No invented baseline. |
| #24 reservations | #25 | bounded reserve/acquire/commit/release, owner/lifetime generation, cancel/crash policy; #25 cannot claim mmap is GPU import. Same worker sequential implementation with separate reviewable heads. |
| #19 dependency ranges | #18/#20/#21/#23 execution | conservative alias handling; no production WFI reduction until chip hazard/mutant qualification. |
| #20 stable states | #21 ragged plan | per-query output plus LSE; merge partitions with max rescaling; never reuse another query's output. |
| #22 scheduler | existing own disassembler | independently decode candidate bytes; IR simulation alone is insufficient; baseline retained. |

## Review focus
Untracked/assume-unchanged dependency bytes; exceptional floating-point/tie behavior and arithmetic reordering; allocation overflow and owner death/PID reuse; stale plans/page generations and overlapping aliases; fixtures mistaken for hardware or kernel gains mistaken for complete-model gains. Each owning issue must include the relevant negative control.

## Work sequence
1. Foundations: #15 portable fresh checkout plus explicit pinned eight-slice harness; #16 operation and whole-model contracts; #17 benchmark collection/comparison CLI. #24 cooperative reservation host work is independent preparation, not priority inversion or deployment.
2. After contracts+benchmark interfaces have passed host review, implement opt-in #18 narrow gate/up-SwiGLU fusion and #20 stable shared-prefix states; #22 bounded schedule search only for a named hot kernel justified by baseline evidence, otherwise prepare validator and mark selection/timing NOT_RUN.
3. #19 hazard-derived synchronization and #21 bounded ragged plan/run; no silent weaker waits, hidden sealed allocations or undocumented reduction-order changes.
4. #23 traversal variants and #25 read-only file-page mapping probe after their dependencies. Unsupported actual native import is an explicit refusal/no-go with API evidence, not a staged-copy success.
5. Parent merges compatible candidate branches into one test-only integration head, reviews full diff, runs host suite once on final executable inputs, requests one shared hardware train. Existing owner alone loads/switches. Keep precise chip/whole-model gates NOT_RUN while unavailable; do not close issues from host-only work.

## Per-issue task contract
For each issue: read frozen spec+live updates; reproduce current gap RED; preserve/reuse existing implementations; define exact callable limits; implement minimum complete behavior; target tests GREEN plus behavioral mutants; run full host suite (use shared #15 baseline where unchanged); publish source/tool/binary/raw-output receipt; push focused branch and append READY with explicit NOT_RUN rows. Fresh reviewer checks spec and quality; fix Important findings before acceptance. Parent verifies remote head/tests/PR readback and only closes GitHub acceptance when all required dependencies and hardware/integration gates actually pass.

| Issue | Ledger | Dependency gates before performance/integration |
|---|---|---|
| #15 | L466 | host work now; reuse5/6 |
| #16 | L467 | 5, existing9/13 |
| #17 | L468 | 5/16; reuse14 |
| #18 | L469 | 16/17 |
| #19 | L470 | 6/16/17 |
| #20 | L471 | 16/17 |
| #21 | L472 | 16/17 |
| #22 | L473 | 10/16/17 |
| #23 | L474 | 16/17 |
| #24 | L475 | 6/8; reuse14 |
| #25 | L476 | 6/24/16/17 |

No full issue is marked complete until its own acceptance is evidenced. Missing chip windows do not stop safe implementation/host work, but do block default enabling, performance claims and closure.
