# Personal review of the L600 rotation qualification continuation

This is author review, not independent approval. The user required personal implementation and review. No merge or deployment is authorized by this document.

## Reviewed boundaries

- CUDApp native source `6605ee7f006fc32cd59a13ee8895ec3fe232411a`: v1 compatibility, v2 zero-aware selection, high-bit maximum tagging, rational inverse normalization, bounded frame parsing, and the exact transaction-capacity boundary.
- External AIEN reference `e1608d17ce0839886f85f8486f900d50c7924200`: feature fence, actual matrix attachment, native binary identity, byte framing, source-scale alignment, cache identity/metadata/payload validation, and explicit GPU attachment refusal.
- Diagnostic continuation `7ab43aed49e2ec8791ebd8f8f4c6cb62d8fe9fe7`: retained activation values, whole-vector error/refusal semantics, always-reached shared-expert probes and zero-reference output accounting. These changes are instrumentation, not model arithmetic or quality-rule tuning. The running full-corpus executable was preserved.
- Frozen corpus and thresholds, four-arm runner, full expert payload census, source closure, and the distinction between expanded reference storage, encoded payload, model quality and GPU/runtime claims.

## Findings resolved with recorded tests

1. The first v2 request was refused by the old codec. Implemented v2 and zero-aware mode; version-1 regression and independent explicit-matrix/Fraction checks pass.
2. The stream produced the correct 999,999 outputs but returned failure at exact capacity. Added the final EOF check and tested both exact-capacity acceptance and an additional-transaction refusal.
3. The reference originally did not apply native reconstruction to a real matmul. The direct matrix test verifies the specific reconstructed 4/7 result, and subsequent complete model forwards consume native-derived weights.
4. Cache integrity originally protected payload but not receipt metadata. Corrupting the receipt was a meaningful red test; metadata now has a separate checksum. Exact bits, wrong extents and linked-file preservation are tested.
5. The tensor-span stub did not read source data. Exact-span, reversed-range, overflow and out-of-bounds checks now pass.
6. Activation hashes alone do not measure error. Added all-layer residual-input and final-logit numerical diagnostics, including undefined-relative-error handling for an all-zero reference.
7. None of the five selected routed experts was reached in the short diagnostic, giving zero actual linear probes. Added layer-0 shared-expert gate/up/down probes, whose execution does not depend on router selection. Preserved the earlier zero-coverage result rather than relabeling synthetic probes as real activations.

## Invocation failures retained, not counted as semantic kills

- A long identifier failed the pinned compiler's minimal name probe; the bounded stream uses the accepted identifier. No compiler implementation was modified.
- The initial quality-rule mutation harness inspected the wrong Python `__main__` namespace and ran no tests. The corrected harness executes all five tests and rejects the two weakened rules.
- The first fresh-reference native-test invocation omitted required environment fixtures. Its five failures are setup failures. A corrected invocation supplied the actual pinned native stream, independently built native no-op and owned cache-test root. The final fresh-source package suite returned zero with 406 passed and 21 pre-existing ignored tests.
- Earlier build/foreground-timeout logs are retained. A timeout is never labeled a successful tool return; receipts distinguish observed exit status from an original runner's post-assert artifact evidence.

## Scope limits and rulings

- The reference's existing 21 ignored tests do not qualify hardware or model paths they did not execute. They were not removed or newly waived.
- The native product remains integer transform/packing/reconstruction. Rust/Python are named external reference and orchestration tools, not a native inference runtime or a fallback serving engine.
- The f32 cache is expanded test storage and protects accidental corruption in an owned, single-writer-per-key workspace. It is not an adversarial-filesystem sandbox or a crash-durable database. Process shutdown relies on the pinned codec's tested EOF contract.
- The short diagnostic covers one frozen prompt and 32 scored positions. It cannot replace the full held-out corpus or its negative control. The completed full-corpus run and separately implemented post-run aggregation are now published: 44 cases, 22,076 candidate-scored positions, unchanged baselines/source closure, negative control rejected, and all three candidates rejected by the frozen quality limits. This completes the bounded experiment, not GPU/runtime qualification or independent author review.
- Whole-expert payload bytes include frames, selector/max metadata and original scales. They exclude unchanged model regions and container/runtime overhead not represented by this experiment. No measured traffic, resident-memory, prefill, decode or energy benefit follows from them.
- The checked native GPU candidate is explicitly launch-prohibited. Existing L408 owns that prerequisite; creating another compiler worker or using a legacy product workaround would not satisfy this task. GPU/full-cost acceptance remains open.
- A minor source comment names only the v1 magic strings; the actual v2 contract and tests are authoritative. This wording does not alter execution and was not used to justify another native source change.
