# Independent read-only review: shader pilot foundation

## Outcome

No blocking or nonblocking correctness finding in the bounded host-only foundation at `5d95e4b00a95770a6f73f299ec33827d78b49802` (base `b5e0da48ee7b8a64ae0b51068769fa0659b2e33a`). No severity-ranked defect to report. This is not approval of native execution, emitter semantics, GPU admission, caching, performance, merge or publication.

Reviewed the repository instructions, supplied public RULES.md, current plan, product documentation, all added `.om` files and unchanged task/catalog/emitter dependencies. The working tree remained unmodified by this review. Scratch artifacts are separate from the author's receipts.

## Exact independently executed commands

Observed commands with local scratch paths replaced by `$REVIEW` and `$AUDIT`. The review scripts were external development orchestration, not shipped product code. The committed `.om` tests and REPRODUCE.md document the reproducible program boundaries and controls:

```sh
python3 $REVIEW/review.py
python3 $REVIEW/extra.py
sha256sum $AUDIT/omega_ref.c \
  $AUDIT/omega_ref \
  $AUDIT/omc.om
sha256sum $REVIEW/{core,export,checker}.elf
git diff b5e0da48ee7b8a64ae0b51068769fa0659b2e33a -- \
  native/learn53/task.om native/learn53/catalog.om native/blackwell/vector.om
git status --short
uname -sm
```

The scripts are external development orchestration, not a product implementation or newly invented interpreter. They compose original sources directly from the reviewed tree. Every source run uses the supplied existing `omega_ref run SOURCE`; compilation uses `omega_ref run omc.om` with original composed source on stdin. ELF files are only saved and hashed, never executed.

## Measured results

- `review.py`: exit 0, `review_cases: 2937`, `status: PASS`, `compiled_negative_controls: 6`, reviewed commit matched. Results are retained in `results.json`.
- All three unmodified product/test units passed reference checking and pinned compiler-source compilation. Empty compile stderr, ELF header observed.
- Core source execution: exit 0, stdout exactly `SHADER_PILOT_CORE_PASS\n`, empty stderr. This exercises the existing per-byte candidate corruption, ULONG_MAX lengths, poisoned canonical scratch, result/canary controls, profile checks and stale task tests.
- Deterministic exports for both IDs and selectors. Task lengths matched the documented 160 and 232 bytes; each code stream was 512 bytes. Repeated exports were byte-identical. IDs 2 and 3 emitted identical code.
- Both task streams and both code streams: every payload-prefix truncation rejected with 65, each single-byte bit flip rejected with 1, trailing byte rejected with 65. Valid frames accepted with 0 and no output.
- Malformed request framing, including valid prefixes followed by extra input, rejected with 65 and no payload. Unsupported representative IDs and selectors rejected with 77 and no payload.
- Task2 under ID3: original task length rejected with 65; zero-padding it to task3 length rejected with 1. Identical code did not rescue stale task identity.
- Real pipe input delivered as separate bytes: process remained pending after the valid two-byte prefix; EOF then produced the canonical payload. A later third byte produced 65 and no bytes. This independently exercises actual partial reads and validation-before-output.
- Six changed implementations compiled successfully and were detected: omitted task scan, omitted final code byte, permitted short task, omitted stride guard, omitted shape guard, and accepted extra request byte. The first five made the unchanged core test exit 1 with `SHADER_PILOT_CORE_FAIL\n`; the framing mutant unexpectedly succeeded and emitted a payload for invalid input, which the review assertion identified.
- `extra.py`: exit 0. Reading stdin from a directory gave 74 and no payload; writing a valid export to `/dev/full` gave 74.
- Integrated upstream catalog drift control: injected a stride2 task for ID2, confirmed it remained valid CUL1 (kvalid exit 0), compiled the altered exporter successfully, then observed both selectors refuse it with 77 and no payload. This is not merely a direct helper test.
- Dependency diff and the initial `git status --short` were empty. Final status showed the implementation agent's new untracked `evidence/shader-pilot/` folder; reviewed source commit was unchanged. This reviewer did not modify the repository. Platform: `Linux x86_64`.

## Source and compiler identities

Measured SHA256:

| Artifact | Digest |
| --- | --- |
| Existing referee source | `52e747096c1d978103432460a2978b21a604cc5d3e961bbe4b253fcc7636290e` |
| Existing referee executable | `5923db3cfc6edaafb610b0eed4031b5cd349cafb0b4ccfca1c4852cbb4a35250` |
| Compiler source | `c792b401193d8477d38965d6ddb0fd0cb8fd5ac24fcb92fae9af52037a765f1c` |
| Compiled core test ELF | `35d7d67bb4b6a35a7832e08081ef7c722a3c5462c880160dc133388fef8e9299` |
| Compiled exporter ELF | `2211c87833e185f4f9a7636ec709c02b258dc11512efe7d9204f167fb83b8a2c` |
| Compiled checker ELF | `707ba3059e91e93e53efb520e0668a7d579a82fd07b6dc2cbdc0e3fddb185fa8` |

Referee provenance pin `190130b71d0a59b93f01c515218a5509b149a145` and Omega compiler-source pin `7505b3a3e3cf5ee46933e3a89a021b4289037983` were supplied by the implementation agent; this review checked the local source digests, not a fresh remote rebuild or native compiler self-reconstruction.

## Code and documentation assessment

- `native/shader-pilot/core.om:6-27`: supported-ID/profile checks are bounded and fail closed for invalid or unsupported shape/operation/stride data. `kvalid` supplies the remaining CUL1 validity constraints.
- `native/shader-pilot/core.om:29-60`: unsupported IDs return before reload, valid comparisons reload canonical bytes, exact lengths are checked before candidate reads, then all task and code bytes are compared. Candidate buffers remain separate from trusted scratch.
- `native/shader-pilot/export.om:8-29`: validation consumes EOF or detects the third byte before any write. Positive reads advance toward the bounded capacity, zero terminates, negative reads refuse. `:37-47` handles short positive writes and rejects nonpositive writes.
- `tests/shader-pilot/core-test.om:54-80,111-129`: byte corruption, extreme lengths, stale selections, buffer preservation and dirty scratch controls are relevant, rather than happy-path-only tests.
- `docs/shader-pilot/README.md:42-58,114-128` accurately limits equality checking: it does not authenticate source, compiler, driver or launch identity and does not implement a persistent cache key. Upstream source changes require new evidence, not silent reuse.
- `docs/shader-pilot/README.md:139-176` accurately distinguishes fresh reference-process timing from native compiler timing, cache effects and GPU/application latency. It requires uncontrolled first samples to be called first-observed, not cold-cache. The author's timing script labels its first retained timed sample as after validation, so it does not establish a cold start. This review did not benchmark performance.

## Limits and next gate

Native AArch64 execution, independent machine decoding, launch/control-flow effects, source authentication, real driver identity, GPU results, cache behavior, pipeline creation, graphics integration and timing distributions are NOT_RUN or absent as documented. The checker shares the emitter and cannot validate the emitter's meaning. Reference execution is not native-host qualification.

Full `make test` and figure checks were not rerun by this independent review. The implementation agent reports baseline/final exit 2 from existing AArch64 assembly on x86 and unchanged stale `stack.svg`; those are reported evidence, not independent passes from this review. No GPU, external publication, credential, environment or repository changes were performed.

No upstream guide change is indicated: this is a consumer-level review of existing language/compiler behavior and introduces no language, ABI, Bridge or qualification contract change. Preserve the existing limitations in any PR. Native/device gates remain a separate owner-authorized step.
