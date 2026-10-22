# Native rotation-aware INT4: first offline qualification

CUDApp [#49](https://github.com/aien-dev/CUDApp/issues/49), L600. Personal implementation
and review; no delegated workers. Stacked on #48 commit
`566dc53502321166f4a898a29f8e4f828a209239`. Branch `feat/rotation-int4-native`.
This is an offline representation experiment, NOT a merged or deployed inference feature.

## Decision

**Reject blanket rotation. Continue only conditional model-level qualification.**
At identical encoded weight-frame/scale bytes, rotation improves reconstruction and
synthetic linear-output error on four frozen matrices but regresses on the sparse
down projection. That projection loses all 280 exact zero columns after rotated
quantization. Three diagnostic inputs whose original outputs are exactly zero
produce nonzero outputs with rotation; unrotated INT4 preserves those zeros.

For the sampled sparse expert, the existing lossless zero-channel codec uses fewer
bytes than either fixed-budget INT4 mode, with no quantization error. These
measurements do not justify replacing that exact representation with INT4.
No automatic policy, inference default, compiler worker or live service changed.

## Implemented

[`rotquant.om`](../../native/rotquant/rotquant.om) is seeded native Omega, compiled
through the verified machine-code-seed lineage. It performs exact integer FP8
format decoding, optional 128-point Hadamard rotation, nearest-even symmetric
INT4 encoding and exact rational reconstruction. The unnormalized transform
avoids approximating sqrt(128) while producing the normalized quantizer's codes.
The independent host oracle uses explicit sign-matrix multiplication, not the
native butterfly. Python/NumPy/ml-dtypes are outside product execution.
This is symmetric INT4, not NVFP4 or an implementation of a paper's runtime.

[Contract and reproduction](../../native/rotquant/CONTRACT.md).
[Personal source review](PERSONAL-REVIEW.md), not independent review.

## Frozen real-weight results

Five selections are inherited unchanged from the first study, not chosen after
seeing these results. Relative weight error is Frobenius norm of error divided by
reference norm, not MSE. Positive change means lower error; negative is regression.
Synthetic output probes are 16 unit Rademacher, 16 unit Gaussian and 8 unit
coordinate vectors, RNG49001, identical between candidates. They are **not real
model activations, logits, perplexity, task accuracy or a quality gate**.

| Layer/expert/projection | Unrotated weight error | Rotated weight error | Weight-error reduction | Synthetic-output-error reduction |
|---|---:|---:|---:|---:|
| 0/0/gate_proj | 0.13835131 | 0.11681424 | +15.5669% | +14.3535% |
| 0/0/up_proj | 0.12163337 | 0.11707152 | +3.7505% | +4.8714% |
| 0/0/down_proj | 0.10889453 | 0.11551923 | -6.0836% | -2.3429% |
| 20/128/up_proj | 0.11992097 | 0.11723491 | +2.2399% | +1.8378% |
| 39/255/down_proj | 0.12973331 | 0.11705831 | +9.7700% | +9.5689% |

Raw records, source/scale/vector hashes, zero-support and per-profile outputs:
[`matrix-comparison.json`](matrix-comparison.json). Every native packed code and
reconstructed integer numerator matches the independent oracle over all five
matrices. All 254 finite FP8 codes also match the independent ml-dtypes format.
No checkpoint weight data is included in this repository.

## Storage and work accounting

Each selected matrix has 1,048,576 weight bytes and 128 unchanged BF16-scale bytes.
Identity and rotation each use **557,568 bytes**, including 32 packet headers,
per-128-value maxima, packed nibbles and unchanged BF16 scales. That is **46.832662%**
less than the original weight/scale bytes. The common payload costs **4.25 bits per
weight**, before framing and original scales; it is not a pure four-bit total.
There is no frame padding in this serialized experiment. Final checkpoint indexes,
GPU alignment, upload conversions and resident memory are not measured here.

The sparse expert's three matrices each use **547,476 bytes** with the separate
lossless zero codec (47.794993% saved). The other two lossless examples use dense
fallback and cost 1,048,724 bytes. This exact comparator is explicitly **not** an
equal-bit approximate candidate, and does not complete #49's future zero-aware
quantization-policy comparison.

A 128-value Hadamard costs **896 add/sub operations**. A future packed rotated
linear kernel needs activation rotation and its normalization (or equivalent
folded scaling); preprocessing is offline. This codec's inverse returns int64
numerators for validation, not compact resident inference weights. No setup,
prefill/decode latency, throughput, traffic or energy measurement was performed.
Wolfram verifies normalization, ranges and these counts in [checks.wl](checks.wl)
and [wolfram-results.json](wolfram-results.json). Its initial unreduced symbolic
expression is retained separately as a diagnostic, not counted as proof.

## Verification

- Initial compiled empty-program baseline: 0/605 passing, retained in red-empty.json.
- Final native suite: **609/609 passing**, including all finite bytes,
  signed/tie cases, maximum zero and nonzero frames, malformed/truncated/trailing
  frames and bound refusals. Four additional native I/O controls pass.
- Three successfully compiled behavioral mutants are caught:
  lose-fp8-sign (50 cases), wrong-inverse-normalization (31 cases), ties-away-instead-of-even (22 cases).
- Eight reused-output/compiler/safety controls: 0/8 red, 8/8 green. Prior verdicts
  retire before preflight; unsafe outputs and Python optimization are refused.
- Existing complete host target `quiet-build make -j4 test`: exit0, **33 prerequisite
  targets**. Lossless baseline: 464/464 with its I/O/mutation controls.
- Initial missing NumPy environment is retained as a prerequisite failure; the
  actual experiment ran in an isolated oracle venv with NumPy2.5.4/ml-dtypes0.6.0.
- Immutable committed-export replay is recorded separately when completed.

`qualification.json` is the final current verdict. `qualification.previous-*`
are historical receipts, not new passes. The suite expanded from605 to609 solely
to add maximum-capacity nonzero reconstruction; the native source did not change.
Raw host logs and initial Wolfram output are losslessly gzipped to preserve their
bytes. MANIFEST.json binds publication files; source_closure binds runtime inputs.

## Remaining acceptance (issue stays open)

Real activation calibration/capture; fixed held-out model-quality data and
thresholds; logits/perplexity/task metrics; a matched-budget zero-aware alternative;
loader/native GPU integration; full setup/prefill/decode cost and energy receipts
coordinated with #16/#17. No winner is selected for adoption from weight error alone.
No third-party inference engine, model restart or default precision change.

Mathematical precedents, not product code: [QuaRot](https://arxiv.org/abs/2404.00456)
and [QuIP](https://arxiv.org/abs/2307.13304). Prior research:
[immutable research report](https://github.com/aien-dev/aien-architecture-mv/blob/e5dad756d5d4fc747393d55947a9594066fd9f6f/evidence/QUANTUM_EFFICIENCY/2026-10-22/README.md).
