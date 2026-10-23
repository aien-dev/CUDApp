# Rotation-aware INT4, zero-support preservation and model qualification

**Final verdict for these three frozen INT4 candidates: NO-GO. The complete model-quality experiment finished and rejected identity, Hadamard and zero-aware INT4 under the predeclared limits. The negative control was rejected too. Host representation remains qualified; no GPU qualification, speedup, deployment or adoption is claimed. CUDApp #49 remains open for its broader unmet integration/full-cost criteria.**

This continues the [first native rotation experiment](../ROTATION-INT4-L600/README.md). The new zero-aware mode fixes the measured destruction of structural zeros without adding a selector byte. Implementation and review are personal, as requested; this is not an independent-review certificate.

## What is now verified

- Seeded Omega owns the transform, quantizer, packing and rational inverse. Version 1 remains a regression contract. Version 2 adds a zero-aware selector and a bounded repeated-frame entry point.
- All **30,840 routed/shared expert gate/up/down matrices in all 40 layers** have been read and reconstructed for each of the three candidate arms. This is a payload census, not an estimate from safetensors headers or a five-matrix extrapolation.
- Those expert matrices contain **32,338,083,840 weights** and **363,374,628 numerical zeros**. The unchanged FP8 payload plus source scales occupies **32,342,031,360 bytes**. Each candidate's actual encoded frames plus unchanged scales totals **17,195,397,120 bytes**, a **46.832662% reduction for this expert region**. This excludes the unchanged trunk, router, embedding and head, and is not measured GPU residency or memory traffic.

| Candidate | Packed bytes, including frames and original scales | Original zero weights made nonzero |
|---|---:|---:|
| Identity INT4 | 17,195,397,120 | 0 |
| Hadamard INT4 | 17,195,397,120 | 106,453,476 |
| Zero-aware INT4 | 17,195,397,120 | 0 |

Sources: [payload census](payload-census.json), per-layer native receipts under `precompute/`, and the [frozen contract](../../native/rotquant/CONTRACT.md). The zero-aware arm rotated 246,850,026 of 252,641,280 blocks. Numerical zero preservation is not signed-zero byte preservation; the separate lossless codec retains that stronger contract. The nine-level negative control also completed the same 30,840-matrix census; all four arms consumed identical source payloads/scales and each executed 1,973,760 native transactions. Its full-corpus quality rejection is now measured, not inferred from its intentionally coarser grid.

## How the new mode works

A 128-weight block containing any numerical zero uses identity quantization. A block without zeros uses the existing Hadamard quantizer. Bit 31 of the existing maximum stores the choice. The inverse uses the original source-scale tile, with the normalization represented exactly as an integer numerator and fixed denominator. There is no extra selector payload and no activation-based fitting.

The native stream accepts up to 999,999 transactions. Exactly that many followed by EOF succeeds; an additional transaction fails with status 65 after the preceding complete outputs. A malformed current frame produces no semantic result for that frame. Transport write failures can leave partial bytes and invalidate the transaction.

## Five fixed matrices: local mechanism check

These are relative weight Frobenius errors, **not model quality or real activation errors**. Every arm uses 557,568 bytes per matrix including frames and scales.

| Layer / expert / projection | Identity | Hadamard | Zero-aware |
|---|---:|---:|---:|
| 0 / 0 / gate | 0.138351 | 0.116814 | 0.116831 |
| 0 / 0 / up | 0.121633 | 0.117072 | 0.117086 |
| 0 / 0 / down | 0.108895 | 0.115519 | 0.108895 |
| 20 / 128 / up | 0.119921 | 0.117235 | 0.117231 |
| 39 / 255 / down | 0.129733 | 0.117058 | 0.117083 |

Zero-aware reconstruction preserves all 280 zero columns of the sparse down matrix. Unconditional Hadamard does not. The selected lossless zero-channel representation remains smaller at 547,476 bytes, so these experiments do not establish that INT4 supersedes lossless packing. Full raw comparisons, source hashes, synthetic coordinate probes and independently decoded native outputs are in [matrix-comparison.json](checkpoint/matrix-comparison.json).

## Verification and provenance

Native source pin: `6605ee7f006fc32cd59a13ee8895ec3fe232411a` in CUDApp.

- Compiler: omega-bridge `d2010ab927b608931800c20d73a4c2b25f00a9fa`, SHA256 `191b776ef98821597c3e30655da6d680a9b66edbc9ceaca3c39c61ea6df098e9`.
- Single-frame binary: SHA256 `109d8ff2b033ce35302331cd32b62ecea227be589f92078a089902698119a0d1`.
- Stream binary: SHA256 `2343c13036b002d65b8c7cbd26fbbb6f4c966de21ad3b8e80c77ff2c0b451ce3`.
- 609 version-1 cases and 141 version-2/stream cases passed; both maximum-transaction cases passed; eight I/O controls and eight builder/filesystem safety controls passed. Six compiled semantic mutants were rejected.
- Five quality-rule unit tests passed, and both weakened-rule mutants were rejected. The initial mutation fixture inspected the wrong Python `__main__` namespace and ran no tests; that setup failure is retained and is not counted as a mutant kill.
- The repository host target passed all 33 targets. Its own mutation report is retained separately from the six rotation mutants.
- A fresh committed-source export reproduced the two native binaries, source closure and five-matrix comparison bytes exactly through the public Make entry and the pinned oracle environment. See [fresh-native-export.json](fresh-native-export.json).

The external AIEN host-reference source is `e1608d17ce0839886f85f8486f900d50c7924200`, based on `f0230a59eec985c3290ac78c0cc04322bacf9f0a`. Its package suite passed 403 tests with 21 existing ignored hardware/model tests. The original foreground tool timed out while its runner continued; [the status receipt](reference-full-suite-status.json) records the explicit post-assert build evidence rather than pretending that timeout was a directly observed successful return.

The reference is test-only Rust transport, unchanged-scale widening and instrumentation around AIEN's existing CPU forward pass. It is not a native GPU inference implementation. GPU graph attachment explicitly refuses reconstructed test matrices. CPU-stub GPU libraries are intentional for this oracle, not a production acceptance result.

The final diagnostic reference source is [`7ab43aed49e2ec8791ebd8f8f4c6cb62d8fe9fe7`](https://github.com/aien-dev/aien-sovereign-core-mv/tree/7ab43aed49e2ec8791ebd8f8f4c6cb62d8fe9fe7). A fresh export of that entire tree passed **406 package tests, 0 failures, with the same 21 existing ignored tests**, using the qualified native stream and actual native no-op control. Its earlier replay invocation omitted required fixture environment variables; the resulting five setup failures are preserved, separately from the corrected run. The completed full-corpus batch used the immutable earlier `e1608d17` / `f66d514c...` reference executable, not a binary replaced during a run.

## Frozen model-quality experiment

The [plan](../../docs/plans/2026-10-23-rotation-qualification.md) and [corpus manifest](corpus/manifest.json) were frozen before the corresponding full-corpus candidate runs. All routed and shared expert gate/up/down tensors are eligible; other weights and f32 recurrent state are unchanged.

For each of identity, Hadamard, zero-aware and a compiled nine-level-grid negative control:

- Five public and five held-out goldens, 512 scored positions each.
- A 400-token AGPL passage, 399 next-token prediction positions.
- Each five-golden group separately requires candidate/reference mean NLL <= 1.07; top-1 agreement >= 0.98 where the reference margin is at least 0.1 and >= 0.998 where it is at least 1.0.
- The AGPL mean-NLL increase must be <= 0.069 nats.
- An exact FP8 control must first match stored token IDs and margins within the frozen 0.001 bound. Every position's NLL, cosine, KL, route changes and near-tie flips is retained.
- The nine-level mutant must fail the same quality rule. Otherwise the gate is inconclusive, even if candidate arms pass.

The expanded f32 reference cache avoids repeating native reconstruction. It is **four bytes per weight per arm**, not the packed product format or a memory optimization. Cache identity binds source/scales, shape, mode and native binary; separate checksums protect metadata and payload. It refuses linked files, corruption, wrong extents and changed identities. Precompute partitions are disjoint; model jobs are bounded CPU-only references, not agents or additional GPU models. Timings in these diagnostic logs are not inference benchmarks.

The earlier 32-position diagnostic is retained under `smoke32/`, but it is not the declared quality gate. None of the five selected routed matrices was reached in that short prompt, so its real-linear-probe count is zero. The separately pinned diagnostic is now complete for all three candidate arms: three actual layer-0 shared-expert linear-map probes per arm, all 40 residual-input tensors and final logits. All scored outputs and full-logit hashes exactly match the earlier diagnostic. The primary full-corpus executable is unchanged. See [replay status](reference-final-export-status.json), [numerical summary](activation32-summary.json), and raw data under `activation32/`.

| Arm | Final-logit relative L2 | Final-logit cosine | Shared gate relative error | Shared up relative error | Shared down relative error |
|---|---:|---:|---:|---:|---:|
| Identity | 0.123198 | 0.992395 | 0.066061 | 0.094563 | 0.114841 |
| Hadamard | 0.103860 | 0.994655 | 0.052581 | 0.081216 | 0.091134 |
| Zero-aware | 0.103372 | 0.994700 | 0.052639 | 0.081268 | 0.091246 |

This is one frozen prompt with 36 input tokens and 32 scored positions. Final-logit metrics include all input positions. Each linear probe uses the actual candidate's activation input against the original matrix on that same input; the down-projection input can differ between arms. These particular real probes contain no exactly-zero reference outputs, so the published synthetic/unit zero-output tests remain separate. These diagnostics do not establish full-corpus quality or tune its thresholds.

## Complete quality verdict and acceptance disposition

The batch executed **44 arm/prompt cases**, with **5,519 scored positions per arm** and **22,076 candidate-scored positions** total. Every case's FP8 control passed, and the FP8 logits/positions match across all four arms. The pipeline's exit 0 means the experiment completed and its negative control was rejected; it does **not** mean any candidate passed.

The frozen criterion permits at most a 7% mean-NLL increase in each group. All three candidates exceed it in both groups, and all three fail both held-out agreement floors. Hadamard and zero-aware pass the separate AGPL passage criterion, but that cannot compensate for a failed group. Full aggregate results are in [qualification.json](full-quality/qualification.json); every per-position output, tensor identity, command status and log is retained losslessly under `full-quality/`.

| Arm | Golden mean-NLL increase | Held-out mean-NLL increase | Held-out agreement, margin >= 0.1 | Held-out agreement, margin >= 1 | AGPL mean-NLL change | Verdict |
|---|---:|---:|---:|---:|---:|---|
| Acceptance | <= 7% | <= 7% | >= 98% | >= 99.8% | <= +0.069 nats | All required |
| Identity | 26.32% | 18.00% | 97.0250% | 99.6030% | +0.079031 | REJECT |
| Hadamard | 17.73% | 18.53% | 97.4217% | 99.4707% | -0.051946 | REJECT |
| Zero-aware | 16.64% | 16.49% | 97.5407% | 99.3383% | -0.062920 | REJECT |
| Nine-level negative control | 142.03% | 141.66% | 89.4090% | 94.1332% | +0.204534 | REJECT |

The held-out agreement denominators are 2,521 positions at margin >= 0.1 and 2,267 at margin >= 1. The golden denominators are 2,538 and 2,351 respectively. Loss percentages are relative changes in mean negative log-likelihood, **not percentages of answers wrong**.

Teacher-forced perplexity (`exp(mean NLL)`) on the held-out reference sequences is 1.1335130509603708 for FP8, 1.1593765549307917 for identity, 1.1601455260338407 for Hadamard, and 1.1571834659811564 for zero-aware. These goldens score frozen reference-model tokens, not a general natural-text perplexity benchmark or a broad task suite. The separate AGPL text passage and all group-specific means/perplexities are retained in the audit. The acceptance rule is the predeclared NLL ratio, not a post-result replacement with a more favorable perplexity ratio.

The [independent audit](full-quality/independent-audit.json) recounts every case/position, recomputes sums and acceptance without calling the original `quality()` function, compares all actual source/native receipts against the full precompute census, and verifies identical baselines across arms. [Post-run checkpoint hashing](whole-model-postrun-closure.json) re-reads the complete recorded model closure and agrees with the earlier file hashes. [Wolfram checks](wolfram-final-check.md) independently confirm the derived percentages, perplexities, counts and held-out rejection.

The model-quality acceptance was measured, but failed. **Do not promote these candidates, weaken the limits, select only the favorable passage, or treat the storage saving as a speedup.** Zero preservation is a useful representation property, not proof of preserved model quality. This rules out these specific calibration-free block-128 INT4 candidates for adoption; it does not prove that every rotation, precision or calibration scheme must fail. A changed candidate requires a newly frozen experiment, not another attempt to pass the same run.

Native GPU attachment, setup, temporary/resident memory, actual traffic, prefill, decode and paired energy/latency remain unqualified. No hardware experiment was launched for a quality-rejected candidate. The broader issue remains open; this bounded experiment is complete with a reproducible no-go.

The checked GPU candidate `0e0b486c300c78ed335b26e86b1fed29d858adad` explicitly labels its output **LAB ONLY, LAUNCH PROHIBITED**, refuses production `sm_121`, and provides no qualified launcher/runtime/checker attachment. See [gpu-dependency.json](gpu-dependency.json). The dependency remains with existing owner L408; no duplicate compiler worker or legacy product workaround was introduced. No driver, live serving configuration or default model path was changed.

## Replay

From a fresh export of the native source pin on the qualified AArch64 host:

```sh
make -C "$SOURCE" test-native-rotquant \
  OUT_DIR="$NEW_OUTPUT" OMEGA_NATIVE_COMPILER="$PINNED_OMC"

"$ORACLE_VENV/bin/python" "$SOURCE/native/rotquant/qualify_rotation.py" \
  --compiler "$PINNED_OMC" --build "$NEW_MODEL_BUILD" \
  --evidence "$NEW_MODEL_EVIDENCE" --model "$CHECKPOINT"

python3 "$SOURCE/native/rotquant/test_quality_rule.py"
```

The oracle environment pins `numpy==2.5.4` and `ml-dtypes==0.6.0`; use `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`. Source reconstruction, errors, raw refusal/negative-control output and corpus hashes are published. Cached model weights and checkpoint tensors are not published. The full-model runner is `native/rotquant/run_model_qualification.py`; use a new output root and the exact binaries/source/corpus identities recorded in [its inputs receipt](full-batch-inputs.json). The batch used at most two disjoint CPU arm workers, with memory checks and no timing claim. The [whole-checkpoint closure](whole-model-input-closure.json) records actual file hashes and capture times; individual expert hashes are independently bound in every precompute receipt. [final_audit.py](final_audit.py) records the post-run recount/hash/archive procedure, separately from the model execution.

Reference reproduction uses the exact external-reference commit and `cargo test --offline --release -p aien-inference-abi --features rotation-qualification -- --test-threads=1`, followed by the matching release binary build. Set `AIEN_FORCE_CPU_STUB=1`, an isolated `CARGO_TARGET_DIR`, the qualified `AIEN_ROTQUANT_BIN` and `AIEN_ROTQUANT_SHA256`, an owned `AIEN_ROTQUANT_CACHE_TESTROOT`, and `AIEN_ROTQUANT_NOOP` built from [native-noop.om](native-noop.om). The actual commands and exit codes are in the replay status; [replay_and_diagnose.py](replay_and_diagnose.py) records the qualified Spark layout. For a different layout, copy the frozen corpus files byte-for-byte and relocate each manifest entry's `path` while retaining its original path and hashes; the script's recorded absolute paths are provenance, not portable installation defaults.

[Personal review and limitations](PERSONAL-REVIEW.md). No independent review, merge approval or production qualification is implied.
