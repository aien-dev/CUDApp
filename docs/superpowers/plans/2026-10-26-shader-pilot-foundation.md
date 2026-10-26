# Offline compute pilot implementation plan

> **For agentic workers:** Use superpowers:executing-plans. Follow the checked steps in order.

**Goal:** Make a reproducible, non-launching seeded `.om` specimen for CUDApp #86.

**Architecture:** Reuse the unchanged `gbvec` emitter and #53 CUL1 catalog/evaluator. A small selector and complete-byte matcher admit only the two existing dense unsigned32 add cases. An offline exporter writes either the canonical task or the code, never a runnable GPU submission or an authenticated cache entry.

**Tech Stack:** Seeded Omega, existing public CUL1, existing Linux development tools.

**Spec:** https://github.com/aien-dev/CUDApp/issues/86, bounded stage-one increment only.

## Global constraints

- No graphics stage, general shader compiler, new cache/evaluator framework or new kernel.
- No GPU access, native execution on x86_64, merge, deployment or qualification claim.
- Reuse catalog IDs 2 and 3 only; unsupported shapes and operations refuse.
- The existing launch uses 64-thread CTAs; the 17-element task needs its bound exit. Export is not proof that this control flow ran.
- New product and behavioral tests are `.om`; foreign tools remain explicitly external development evidence.
- Preserve existing source, owners and historical receipts. Do not repair unrelated stale crumbs.

## Review focus

- Stale task for a different catalog ID must refuse even when code bytes are identical.
- Truncated, oversized or single-byte-corrupted task/code must refuse.
- Invalid request framing must emit no payload, including after a valid request prefix.
- Catalog drift away from dense unsigned32 add must not silently gain support.
- Matching bytes are not source authentication, independent machine decoding or execution evidence.

## Task 1: offline selector and matcher

Files: `native/shader-pilot/core.om`, `tests/shader-pilot/core-test.om`.
Interfaces: `spload(id:ulong)->ulong` prepares canonical `kt`, `gbcode`, `gblen` and returns task length, or zero. `spmatch(id:ulong,tn:ulong,cn:ulong)->int` compares separate caller-owned `sptask[232]` and `spcode[512]` against freshly prepared bytes. Status 0 matches, 65 bad lengths, 77 unsupported, 1 byte mismatch. Neither function launches anything.

- [x] Write tests for valid IDs, lengths, refusal preservation, every-byte corruption, stale ID, dirty trusted scratch and independent CUL1 output fixtures.
- [x] Run with an explicit temporary stub and observe failing assertions; separately show missing implementation cannot compile.
- [x] Implement the smallest selector/matcher using existing functions, then rerun unchanged tests.
- [x] Compile original `.om` with the pinned compiler source and challenge with compiled mutations.

## Task 2: bounded exporter and repeatable evidence

Files: `native/shader-pilot/export.om`, `docs/shader-pilot/README.md`, `tests/shader-pilot/stream-check.om`, new evidence folder.
Interfaces: exporter stdin exactly two bytes `[catalog_id, selector]`, selector 1 emits existing CUL1, selector 2 emits unchanged native bytes. Invalid length 65, unsupported ID/mode 77, I/O failure 74. No payload on validation refusal; I/O failure may leave a partial stream and the nonzero status invalidates it.

- [x] Define stream checker and input/refusal matrix; prove stub exporter fails expected byte/status checks.
- [x] Implement read/validate/export with bounded partial-I/O loops, then replay matrix.
- [x] Record deterministic exports, source/compiler/ELF/artifact hashes, corruption and stale controls.
- [x] Document native-host compile/run commands and separate first-observed/repeated compilation, actual cache absence, and NOT_RUN load/pipeline/GPU stages.
- [x] Run full `make test`, focused checks, independent review and fresh ownership check.
- [x] Paused before public draft PR for parent review; publication is not claimed.

## Execution record

- Design choice: reuse two existing catalog tasks rather than inventing a second arithmetic specification.
- Environment: Linux x86_64. AArch64 compiler output may be compiled and hashed but must not be executed here.
- Baseline crumbs report two stale existing directories. They are outside this slice.

- Source/test commit: `5d95e4b00a95770a6f73f299ec33827d78b49802`. Receipt and independent review: `evidence/shader-pilot/host-foundation-20261026/`.
- Ruling: external referee evidence is permitted development evidence, never native execution. Compiler outputs are checked and hashed without execution.
- Ruling: GNU time was absent; measured reference-process samples use Bash time and explicitly omit RSS. No cold-cache/native-performance claim.
- Ruling: crumb compile scans the whole repository even with a subdirectory argument. Its unrelated trace-crumb rewrites were restored to HEAD; only new crumbs are included. The existing global verify failure remains disclosed.
