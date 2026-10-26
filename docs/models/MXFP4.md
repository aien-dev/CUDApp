<!-- SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception -->
# MXFP4 dequantisation unit

`native/models/mxfp4.om` turns one MXFP4 element into exact IEEE binary32 bits. An MXFP4 element is an E2M1 nibble together with the E8M0 scale of its 32-element block. This is the format gpt-oss stores its MoE expert weights in.

- `mxfp4bits nib scale` takes a nibble (0..15) and a biased scale (0..254) and returns the float bits.
- `mxfp4byte packed lane scale` takes a packed byte and selects the low nibble (lane 0) or the high nibble (lane 1). The low nibble comes first, as in gpt-oss.
- The functions return integer bits, not floats, so the same integer code can be lowered by every Omega back end.
- Refusals trap 5, which gives process exit 105. They cover a nibble above 15, a lane above 1, a packed value above 255 and the reserved E8M0 scale 255 (a NaN block).
- Results the format cannot represent finitely become signed infinity. Signed zero is preserved. Results below the normal range become subnormals.

It is a scalar primitive, not a GEMM and not an inference engine, and it does not touch any device.

## Check

```
python3 tests/models/mxfp4-check.py <omc>
```

`<omc>` must be the seed-built compiler with sha256 `189ada82e00bb440cb41351bd252cc68c54a1b1050051cc42a1df700081c3b54` (omega-mv `91b3c932`); the script refuses any other.

What the script does:
- Compiles the unit with `tests/models/mxfp4-driver.om` to a native AArch64 program.
- Compares every input (255 scales × 256 bytes × 2 lanes = 130560 records) byte for byte against an oracle. The oracle uses `FP4_VALUES` from `openai/gpt-oss` `gpt_oss/torch/weights.py`, `math.ldexp` and `struct.pack('<f')`.
- Checks six refusals.
- Requires a sign-dropping mutant to run to completion with a mismatching result.

The Python oracle is used only for verification. It is never part of the runtime.

Observed on the DGX Spark host (aarch64), 2026-10-26:

```
compiler_sha256=189ada82e00bb440cb41351bd252cc68c54a1b1050051cc42a1df700081c3b54
source_sha256=4bae7783aa357689f27d02ee20223bfc968b6b1ff8d2eaff12eef8a872651ef0
candidate_sha256=df526721822313cf9bda2367afe3ba054ae3ed510b3af8d62141c686a07da712
oracle_sha256=665ce3b50c419456c447dc8eabf300d79f20d38dfdbd6e5b3fd6b97308f0bf04
MXFP4 native CPU exact matches=130560; refusals=6; sign mutant rejected
NVIDIA/Intel/AMD GPU: NOT_RUN; model logits: NOT_RUN
```

## Not run

- SPIR-V through `irspirv` on PoCL. That gate pins a different compiler (`befb8199…`), and this host had neither PoCL nor `spirv-val`.
- Any GPU.
- gpt-oss weights or logits.
