<!-- SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception -->
# FP8, NVFP4 and MX block dot product units

Three Omega units sit beside `native/models/mxfp4.om`. They use integer bit operations only (`nvfp4scale` alone does one binary32 division) and touch no device.

## `native/models/fp8.om`: OCP OFP8 E4M3 and E5M2

The units follow the OCP 8-bit Floating Point Specification (OFP8) rev 1.0, 2023-12-01.

- `e4m3bits code` and `e5m2bits code` decode a code (0..255) to exact binary32 bits.
  - E4M3 `S.1111.111` and E5M2 `S.11111.{01,10,11}` become quiet NaN `0x7FC00000` with the sign kept.
  - E5M2 `S.11111.00` becomes ±Inf.
- `e4m3enc bits sat` and `e5m2enc bits sat` encode binary32 bits, following spec §5.2.1 and Table 3.
  - The value is first rounded to nearest, ties to even.
  - With `sat 1`, overflow and ±Inf clamp to ±max normal: 448 (code `0x7E`) for E4M3, 57344 (`0x7B`) for E5M2.
  - With `sat 0`, overflow gives NaN (`0x7F`) for E4M3 and ±Inf (`0x7C`) for E5M2.
  - Values below half the minimum subnormal become signed zero.
  - A NaN source keeps its sign: E4M3 gives `0x7F`, E5M2 gives `0x7E`.
- Every encoder rounds through the one `rnesh` function.
- Refusals trap 5 (exit 105): a code above 255, or `sat` above 1.

## `native/models/nvfp4.om`: NVFP4

NVFP4 elements are E2M1 (0, 0.5, 1, 1.5, 2, 3, 4, 6), with one unsigned E4M3 (UE4M3) scale per 16 elements.

- `nvfp4bits nib scale` gives the exact binary32 bits of element × scale. The product always fits binary32 exactly.
- `nvfp4enc bits scale` gives the nearest-even E2M1 nibble for x/scale.
  - It compares |x| against scale × midpoints exactly.
  - Values above 6 and ±Inf saturate to ±6.
  - A NaN input traps 5. Scale 0 gives signed zero.
- `nvfp4scale amax` computes `e4m3enc(amax / 6.0f32, sat 1)`. This is the per-block step of NVIDIA's FP32 two-level recipe.
- `nvfp4nib x q` encodes a 16-element block into 8 packed bytes (low nibble first) and returns the scale.
- Scale codes accepted are 0..126.
  - Code 127 (NaN) traps 5.
  - Codes 128..255 trap 5. cuBLAS ignores the sign bit of a UE4M3 scale; these units refuse it instead.
- Out of scope: the per-tensor FP32 second-level scale. Callers apply it.

## `native/models/mxdot.om`: block-scaled dot product and C = A·Bᵀ

`mxmm a sa b sb c d` computes C = A·Bᵀ in binary64. A is m×k and B is n×k, both K-contiguous (the TN layout of cuBLASLt block-scaled GEMM). `d` = (fmt, m, n, k). A dot product is the case m = n = 1.

| fmt | elements | block | scale |
|---|---|---|---|
| 0 | NVFP4 (E2M1) | 16 | UE4M3 |
| 1 | MXFP4 (E2M1) | 32 | E8M0 |
| 2 | MXFP8 E4M3 | 32 | E8M0 |

How each entry of C is computed:
- Each block term is exact. The element products are integers summed in a `long`, then scaled once by an exact power of two.
  - Wolfram: the maximum inner sums are {16·144·225, 32·144, 32·229376²} → {518400, 4608, 1683627180032}, all below 2^53.
- Block terms are added left to right in binary64.
- Error bound: |C − exact| ≤ γ(nb−1)·Σ|term|, with γ(j) = j·u/(1 − j·u) and u = 2^-53 (Higham, sequential summation).
  - Wolfram: γ(255) → 2.8311×10^-14.
- Refusals trap 5:
  - a NaN element or scale;
  - fmt above 2;
  - k = 0, or k not a multiple of the block;
  - m·n > 256, more than 256 scales per operand, or more than 4096 element bytes per operand.

## Check

```
python3 tests/models/mxfmt-check.py <omc>
```

- `<omc>` must have sha256 `2c82480596bbdc50cda9626866209ef662d5834f278f839ff4ce0250c495029f` (omega-mv `369dbc96`).
- Needs numpy and ml_dtypes, which are verification-only oracles.

References:
- **spec**: OFP8 §5.1 formulas, Table 2 rows, and §5.2.1/Table 3 rules, evaluated in exact rationals.
- **ml_dtypes**: `float8_e4m3fn`, `float8_e5m2`, `float4_e2m1fn`.
- **exact**: rational sums for the dot products.

Coverage:
- **FP8 decode**: all 256 codes per format against both spec and ml_dtypes, including the Table 2 rows.
- **FP8 encode**: run per format and per saturation mode.
  - All 65536 bfloat16 patterns are checked against the exact spec oracle and against ml_dtypes.
  - A stated binary32 sweep is checked against ml_dtypes. The sweep covers every sign, every exponent field and all 4096 top-12-mantissa patterns, with the low 11 bits taking each of {0, 1, 0x3FF, 0x400, 0x401, 0x7FF}.
  - Every code is round-tripped.
- **NVFP4 decode**: all 16×127 element×scale pairs, compared with the ml_dtypes product and the exact rational product.
- **NVFP4 encode**:
  - every non-NaN bfloat16 × 127 scales, against a binary64 quotient oracle;
  - ±1 ulp around every threshold, against an exact rational oracle;
  - round trip of every element × every nonzero scale.
- **NVFP4 block scale and packing**: 232641 amax values and 20000 random blocks.
- **mxdot**: 600 seeded cases over the three formats, including k at scale capacity. Every entry must lie within the γ bound and be bit-equal to the left-to-right binary64 sum of the exact terms.
- **Refusals**: 275 checked, including every invalid UE4M3 scale code for decode and encode.
- **Negative controls**: each must be rejected, or the check fails.
  - a flipped reference table entry (E4M3 `0x38` changed from 1.0 to 1.125);
  - round-half-away-from-zero FP8 and NVFP4 encoders;
  - a flipped NVFP4 midpoint table entry;
  - an E4M3 subnormal exponent mutant;
  - an NVFP4 dot-product scale mutant.

Observed on the DGX Spark host (aarch64), 2026-10-27. Load average was about 17 during the run (nice 19, cores 8-9), which does not affect exactness.

```
compiler_sha256=2c82480596bbdc50cda9626866209ef662d5834f278f839ff4ce0250c495029f
fp8.om_sha256=1baf86cbf966faf948913a30e10612ffcfeeed08f382f46c59fcd7539808bfad
nvfp4.om_sha256=5782686136803d3e19fa485bbe40630a0ff9bf22245c806744ed9095497abfed
mxdot.om_sha256=005f6e13490b33206f333bc04251e0d4d78cff200646cfa85afa85140af550e1
numpy=2.5.4 ml_dtypes=0.5.1
FP8 decode exhaustive: 512 codes (E4M3 256, E5M2 256) == ml_dtypes == OFP8 5.1 formula; Table 2 rows checked
FP8 encode e4m3/e5m2 sat=0/1: 12648448 inputs each (65536 bfloat16 vs exact spec + ml_dtypes; 12582912 structured binary32 vs ml_dtypes); round trip 256 codes
NVFP4 decode exhaustive: 2032 element x scale pairs (16 x 127) == ml_dtypes product == exact
NVFP4 encode: 8290814 (non-NaN bfloat16 x 127 scales); threshold +-1ulp Fraction check; round trip 2016
NVFP4 block scale: 232641 amax values; block encode 20000 blocks
MX dot/matmul: 600 cases, 33600 dot products, error <= gamma(nb-1) sum|t| for all; 33600 bit-equal to left-to-right binary64; worst err/sum|t| = 3.346e-16
refusals: 275
NEG (6 controls) rejected
GPU: NOT_RUN
```

The full receipt, including the candidate executable hashes, is in the check output.

Not run:
- GPU: no GB10, NVIDIA, Intel or AMD comparison against cuBLASLt block-scaled GEMM.
- The IR and SPIR-V back ends: `nvfp4scale` uses a float word, which IR mode refuses.
- Exhaustive 2^32 FP8 encode.
- Model logits.
