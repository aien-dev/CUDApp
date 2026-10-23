# Native rotation-aware INT4 experiment (CUDApp #49)

Owner: this chat personally; L600. Approximate offline representation only.
No production inference changes, no worker/reviewer agents, no extra model load.

## Frozen first-slice contract
Input RQU1: magic + little-endian u32 block_count + u32 mode, followed by
block_count consecutive 128-byte F8_E4M3 blocks. Count 1..256. Mode 0 identity;
mode 1 unnormalized Sylvester Hadamard, natural index order. Both NaN codes are
refused. Signed zero becomes numerical zero: this is approximate, not lossless.
FP8 finite values are decoded exactly into integer units of 1/512. For exponent
e and fraction f, units=f if e=0, otherwise (8+f)*2^(e-1), with sign applied.

Output RQP1: same count/mode header, then 68 bytes per block: u32 maximum absolute
transformed integer M, followed by 128 four-bit codes packed low nibble first.
Each code represents q+7, q in [-7,7], nibble15 invalid. q is round-to-nearest,
ties-to-even of 7*v/M. M=0 produces 128 zero q values. No learned clipping,
random rotations, stochastic rounding, zero-point or unfrozen seed.

RQP1 can also be supplied to the native decoder. Output RQR1 contains the same
count/mode, a u32 common denominator, and one signed little-endian int64 numerator
per value. Identity numerator=q*M, denominator3584. Rotation numerator=H(q)*M,
denominator458752. This exact rational reconstruction avoids approximating
sqrt(128) inside the integer-only seed. Quantizer ratios are identical to a
normalized H/sqrt(128) transform. Original BF16 block scales are applied outside
this byte frame and retained unchanged. Alignment of 128-wide rotations with
the source's uniform 128-column scale tiles is mandatory; no mixing scales.

M bounds: 229376 identity, 29360128 rotated. Decoder requires M=0 iff all q=0;
positive M requires a saturated |q|=7. Reconstruction magnitude is bounded by
26306674688. All input lengths/modes/codes/bounds checked before stdout. Exit65
syntax/domain refusal, exit74 I/O error, OS signal behavior unchanged. No
content authentication; externally bind frame and scale hashes. One frame/EOF
per process. Caps: input32780, packed17420, decoded262160 bytes.

## Comparison fixed before results
Five inherited matrices: (0,0,gate), (0,0,up), (0,0,down), (20,128,up),
(39,255,down), from qwen3.6-35b-a3b-fp8-95a723d0. No selection changes after results.
Each matrix split into 256-block frames. Identity and rotation use identical
headers, M words, codes, unchanged scales and frame count: 4.25 payload bits per
weight, not a claimed pure 4-bit total. No learned calibration in this slice.
Reference: independent explicit sign-matrix multiplication, independently
decoded FP8 and exact rational rounding; NumPy/ml-dtypes only as host oracles.

Measure weight Frobenius error and linear-output errors on 40 fixed synthetic
vectors: 16 Rademacher, 16 Gaussian, 8 single-coordinate probes, RNG seed49001.
Rademacher and Gaussian vectors are unit-normalized.
Vectors are identical for both candidates. These are diagnostic activations,
NOT real model activations, logits, perplexity or task-quality evaluation.
Also report the existing lossless zero-channel representation's actual bytes
as a separate exact-cost comparator, not falsely equal-bit quantization.
A rotation benefit is conditional, not required for the test suite to pass.
Preserve regressions; no winner selected from reconstruction error alone.

Full-cost accounting includes frame/scale bytes, setup transforms, potential
online activation transforms (896 add/sub operations per 128 values plus
normalization), and inverse reconstruction. No timing/energy experiment or
speedup claim in this first slice. Entire-model quality and runtime adoption
remain gated by #16/#17, with held-out dataset and thresholds frozen before
any future model-level selection. Identity path and live inference unchanged.

## Reproduction and safety

Requires Linux/AArch64 and the #48 source-built seed (SHA256
191b776ef98821597c3e30655da6d680a9b66edbc9ceaca3c39c61ea6df098e9),
from omega-bridge d2010ab927b608931800c20d73a4c2b25f00a9fa. Compiler rebuilding
is documented in ../weightpack/CONTRACT.md. No implicit compiler download.

```sh
make test-native-rotquant OMEGA_NATIVE_COMPILER=/absolute/path/to/omc
python3 native/rotquant/test_safety.py --compiler /absolute/path/to/omc --out build/rotquant/safety.json
python3 -m venv build/rotation-oracle-env
build/rotation-oracle-env/bin/pip install -r native/rotquant/oracle-requirements.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 build/rotation-oracle-env/bin/python native/rotquant/qualify_rotation.py --compiler /absolute/path/to/omc --build build/rotquant --evidence build/rotation-comparison --model /absolute/path/to/checkpoint
```

The host qualification requires ordinary Python (not -O/PYTHONOPTIMIZE).
It archives a previous qualification.json before preflight; failed runs publish
no new qualification.json. Historical previous receipts are not current PASS.
It reuses #48's named-parent/symlink/hardlink/special-file refusal checks; unsafe
evidence trees are refused without following them. This is single-user preflight,
not protection against concurrent malicious filesystem replacement.
The test/oracle/runtime dependency hashes and versions are recorded; NumPy and
ml-dtypes are independent host references, never linked to the native product.

## Version2 and streaming extension

Version1 behavior above is retained byte-for-byte. Version2 changes only the magic
version byte (RQU2/RQP2/RQR2) and adds mode2. This zero-aware mode selects identity
whenever a source block contains any numerical zero, otherwise Hadamard. Bit31 of
the existing packed maximum indicates Hadamard; the remaining31 bits are M.
Thus metadata and complete encoded bytes equal modes0/1. Decoder mode2 always
uses denominator458752 and multiplies identity-block numerators by128. Original
scalar zeros, rows and columns are preserved numerically, not by signed-zero bytes.

`stream.om` is concatenated after this codec with the original main renamed.
It accepts up to999999 consecutive complete v1/v2 frames; clean EOF between frames
succeeds. Exactly the limit succeeds only on EOF; any following byte refuses.
A refused frame publishes no bytes for that frame; preceding successful frames
remain. I/O failure can leave a partial transport write and invalidates the host
transaction. The single-frame command retains its original EOF requirement.

The public qualification command also builds this entry, compares against the
independent Fraction/sign-matrix oracle, exercises both transaction-limit edges,
and catches selector, inverse-factor and zero-selection mutants. None of this
qualifies direct packed GPU execution; the model-quality adapter is test-only.
