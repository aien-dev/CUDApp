# L777 native-host evidence, source11ad6a0

**Implemented and exercised: a public task-contract/reference-output slice of CUDApp #53. Full #53 and #52 remain OPEN.** This is not GPU, whole-model, protected-evaluation or performance qualification.

## Immutable identities

- CUDApp source/test commit: `11ad6a0a7783f5d221de79111d6abba4ee1ae52f`.
- Public main/base: `80f3a2492eef315106feebfdffcb32de39be7d95`.
- Compiler source package: omega-bridge `0fc7d8662cb799e7c206b17c18ea4c3ec2b8fea9`; imported seed origin omega-mv `29b12760f4c962606411d6bf25032aedef063301`.
- Rebuilt compiler SHA256: `191b776ef98821597c3e30655da6d680a9b66edbc9ceaca3c39c61ea6df098e9`. `bootstrap.log` records actual source-only reconstruction and stage1/stage2 equality. Fixpoint is lineage evidence, not a general semantic proof.
- Target: actual Spark Linux AArch64 host, exact unsigned32 modular profile, schemaCUL1/reference revision1. No model, tokenizer, policy checkpoint, GPU binary/launch or driver is consumed; these are NOT_APPLICABLE to this slice. GB10/actual-consumer qualification is NOT_RUN.
- Complete consumed product/test files: `source.sha256`; executable identities: `binaries.sha256` and `replay-results.json`.

## Executed results

| Run | Actual result | Raw evidence |
|---|---|---|
| Compilable unconditional-success RED scaffold | rc1;58 of71 assertions fail | red.log, red ELF hash |
| Core final suite | rc0;77 of77 assertions pass | unit-final.log/status |
| Data-frame RED scaffold | rc1;15 of17 assertions fail | stream-red.log, ELF hash |
| Data-frame final suite | rc0;17 of17 assertions pass | stream-unit-final.log/status |
| Valid CLI against empty-output stub | independent native stdout checker rc1 | cli-red-status.txt, cli-red.out |
| Real data-only CLI, valid frame | rc0,12 output bytes; native checker rc0 | cli-0.out/status, artifact-probe.txt |
| CLI wrong output | rc1,zero stdout | cli-1.out/status |
| CLI truncated / trailing / forged length | rc65 each,zero stdout | cli-2..4.out/status |
| CLI final-pool access attempt | rc77,zero stdout | cli-5.out/status |
| Empty stdin | rc65 | commands/receipt statement; no PASS emitted |
| Fresh Git-source export reconstruction | all5 binaries identical;77/77 core and17/17 frame replay pass | replay-results.json, replay-unit.log, replay-stream-unit.log |
| Unchanged repository legacy host suite | rc0,33 targets | baseline-host.log; source80f3a24 |

The legacy suite includes expected simulator-refusal and mutation diagnostics. Its overall recorded exit is0 and it prints `CUDA++ HOST TESTS PASS (33 targets)`. It is historical-path regression evidence, not a seeded migration or proof that the root Makefile consumes this new module. New native .om tests are separate.

Four source mutants compiled successfully to nonempty AArch64 ELF before execution. Each exits1 because assertions fail, not because construction or startup failed:

| Mutation | Failed assertions out of77 | Evidence |
|---|---:|---|
| Ignore absolute tolerance change | 2 | mutant-tolerance.* |
| Omit last physical output word | 6 | mutant-tail.* |
| Ignore A's element stride | 5 | mutant-stride.* |
| Substitute constant zero for add result | 7 | mutant-wrong-op.* |

Original failures and all mutation patches are preserved. No cases, tolerances or acceptance policies were weakened. The independent arithmetic inputs/results are in `independent-arithmetic.json`; Wolfram is an external qualification oracle, not linked or shipped as runtime logic. Fixtures are public developer data only.

## Closure and bounds

The native data-only executable is9108 bytes. ELF headers show static AArch64 EXEC, no interpreter/dynamic section. Its traced process only reads stdin and writes the fixed diagnostic for the tested valid frame; no file, GPU, ioctl, network or model operation occurs. The trace also includes the separately identified host `timeout` utility's startup. This is a tested syscall observation, not exhaustive proof of all executions. `replay-static-elf.txt`, `replay-stream-syscalls.log` and `artifact-probe.txt` preserve the actual output.

Manifest capacity4096 bytes; each input/output physical footprint <=256 u32 words; candidate buffer1032 bytes; stream reader5137 bytes including an extra-byte refusal margin. Supported shape families are explicit in the contract. This is not measured system-wide memory admission. The compiler reserves an8MiB stack as part of its host ABI; file size is not RSS.

No optimization candidate generated, no GPU timing, no parameter training or learned/controller change. Complete baseline/search/teacher/compiler/judge lifecycle costs and uncertainty are NOT_RUN. Host test durations printed by old tools are not performance evidence. Current native tests validate values, refusal and bounds; they do not qualify scheduling, sealed custody, wall-clock enforcement or production activation.

## Reproduction and upstream boundaries

See [CONTRACT](../../docs/learn53/CONTRACT.md), [dependency matrix](../../docs/learn53/DEPENDENCIES.md), and `COMMANDS.md` for source concatenation and exact executed commands. Existing external Linux tools orchestrate construction, source export, ELF checks, timeout and hashes. No new foreign-language executable build/test wrapper was added; the inherited host automation is still an outstanding .om replacement owned by Omega635/636 and CUDA++15. The public CLI is not an authenticated judge: its caller must bind task and evaluator identities; synthetic candidate PASS bytes are refused, but recorder/executable substitution protection belongs to Omega653.

No replaced production implementation/dependency was deleted: this is a new bounded domain module and none of the legacy owners' consumers were cut over. Existing native weightpack and host/numeric/collector/reservation source and receipts are preserved, not requalified by this work. All #54-60 experiments retain their explicit prerequisite gates.

Independent review is tracked as L778 on exact source11ad6a0; its report and disposition must be added before any review-approved statement. No issue closure, merge, default change or release is authorized by this receipt.
