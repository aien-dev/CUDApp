# Sparse core in Omega

This extends the existing [CSR transpose](CSR_TRANSPOSE.md) with stable COO↔CSR conversion, CSR sparse matrix multiplication, and triangular solves. The implementations are Omega programs built by the seed-born compiler, not Python algorithms wrapped as a library.

**Current scope: host-only Linux AArch64, not a cuSPARSE ABI replacement or GPU implementation.** No performance claim is made. Each matrix dimension is at most 64; each input or result has at most 256 stored entries. Unsupported sizes are refused, not silently truncated. All arrays are fixed global pools, so calls are non-reentrant.

## What runs

| Function | Contract |
| --- | --- |
| `cootocs rows cols nnz` | Stably group COO triples by row. Retain duplicates, stored zeros, and opaque value bits. Do not sort columns or sum duplicates. |
| `csrtoco rows cols nnz` | Expand CSR row pointers into COO row indices, preserving entry order and value bits. |
| `spgemm rows inner cols anz bnz` | Multiply CSR A by CSR B with separately rounded IEEE binary32 products and sums. Emit sorted, unique output columns per row. Accumulate paths in A stored order, then B stored order. Retain every structurally reached output entry, even if its value cancels to zero. |
| `trisolv n nnz upper unit` | Solve lower (`upper=0`) or upper (`upper=1`) triangular systems, with explicit (`unit=0`) or implicit unit (`unit=1`) diagonal. Off-diagonal duplicates are allowed and processed in stored order. |

Every function returns 0 on success and 1 on refusal. Inputs are separate pools from outputs. Every caller-visible output is unchanged on failure, including failures after partial internal computation. Scratch pools may change. Only the documented output prefixes are defined after success; unused tails may retain prior data.

### Shared pools

`omega/libraries/sparse_core.om` declares these pools and common CSR validation:

- A: `arp` (65 row pointers), `aci` (256 column indices), `aval` (256 value words).
- B: `brp`, `bci`, `bval`, with the same capacities.
- COO: `sr`, `sc`, `sv` (256 entries each). Input to `cootocs`; output from `csrtoco`.
- CSR result: `crp`, `cci`, `cval`; `onnz[0]` contains the result entry count. `csrtoco` also sets `onnz[0]`.
- Solve: input `rhs[0..n)` and output `sol[0..n)`.

`ulong` cells hold indices and 32-bit value words; callers must supply value words within 32 bits. CSR pointers must start at zero, end at the supplied entry count, be nondecreasing and bounded by that count. Column indices must be within the matrix shape. No input/output aliasing, one-based indices, implicit conversion, or allocation interface is offered.

### Multiplication details

Only stored paths multiply. An absent A entry does not multiply a stored infinity in B; an explicitly stored zero does. Canonical NaNs and signed-zero arithmetic follow the reused `binary32.om` implementation. The product does not use FMA, reassociate terms, or drop structural zeros. A row-local accumulator and marker pool feed staged CSR arrays. If the structural result would exceed 256 entries, no result is published.

### Solve details

All stored matrix values and RHS values must be finite. Each explicit-diagonal row requires exactly one nonzero diagonal. Unit mode permits a missing diagonal or one stored finite diagonal; its stored value is ignored. Duplicate diagonals, entries on the wrong side of the triangle, and nonfinite inputs are refused.

Each off-diagonal product is rounded to binary32, then subtracted from the row accumulator with another rounding, in stored order. Explicit mode divides by the diagonal once after those subtractions. A nonfinite product, intermediate sum or result is refused, even if a later mathematical cancellation might have made it finite. Underflow to a finite zero/subnormal is allowed. The complete solution is staged before `sol` is updated.

## Build and test

Use the SHA-pinned seed compiler documented in [CSR_TRANSPOSE.md](CSR_TRANSPOSE.md#run-it). Its SHA-256 is `10d7565239ca6bd3cca344b65a03c0f3498acd95e82ab3d420bd01408b01398e`.

```sh
make test-sparse-core OMEGA_SEED_COMPILER=/absolute/path/to/omc
make test-csr-transpose OMEGA_SEED_COMPILER=/absolute/path/to/omc
make test
```

The Omega targets are separate from the ordinary C host suite because this compiler emits Linux AArch64 programs. A missing or wrong compiler is a hard error, not a skip. Python's standard library builds and tests native Omega executables. Run the harness without Python's `-O` flag, since assertions express test failures.

Arithmetic sources `binary32.om` and `fdiv.om` are reused byte-for-byte from the existing `feat/fft-rng` implementation at `f564d78bb356f257e308319cb489b9b3369fb970`. They are not a new competing implementation. The new tests also reuse the existing transpose harness's compiler-pin and compilation helper.

## Evidence and test independence

- Conversion oracle: stable sorting/grouping of triples, independently of the count/scatter implementation; literal fixtures include negative zero and NaN payloads.
- Product oracle: enumerate all contributing stored paths separately for each output column, independently of the implementation's marked dense row accumulator. Round each operation to binary32.
- Solve oracle: literal systems and randomly constructed systems with known exact solutions, plus a separately rounded substitution oracle for an order-sensitive case.
- Boundary tests: empty shapes, maximum dimensions, 256 entries, duplicate/unsorted coordinates, structural overflow, invalid pointers/indices/flags, and every truncation of a valid request.
- Direct core probe: bypass wire validation, test unchanged output on refusal including late arithmetic/capacity failures, and repeat calls in one process.
- Compiled mutants: corrupt converted values/rows, replace product multiply with add, drop structural zero entries, publish partial capacity-failed output, replace solve division with multiplication, reverse solve order, or omit a triangle check. Both the normal adapter and direct core probe variants must compile before a mutant can count as caught.

Commands, compiler/source hashes, red runs, green runs and review are recorded in `evidence/sparse-core/`. Inherited transpose evidence remains in `evidence/csr-transpose/`.

## Binary test protocol

The adapter is a test transport, not the public library API. All fields are little-endian unsigned 32-bit words. The fixed header is:

```
mode rows inner cols anz bnz flags reserved
```

`reserved` is zero. Modes are 1=COO→CSR, 2=CSR→COO, 3=SpGEMM, 4=triangular solve. Except for multiplication, `inner` and `bnz` must be zero. Only solve accepts nonzero flags: bit 0 means upper, bit 1 means unit; solve also requires `rows == cols`. Other flag bits are invalid.

- Mode 1 input: `anz` row indices, `anz` column indices, `anz` value words.
- Mode 2 input: `rows+1` A pointers, `anz` A columns, `anz` A values.
- Mode 3 input: A arrays as above, then `inner+1` B pointers, `bnz` B columns, `bnz` B values.
- Mode 4 input: A arrays as above, then `rows` RHS words.

Modes 1 and 3 output: result entry count, `rows+1` result pointers, result columns, result values. Mode 2 output: entry count, row indices, column indices, value words. Mode 4 output: solution words only. Input length must match exactly. Invalid requests exit 2 with empty stdout; write failure exits 3. A broken pipe can still produce an OS signal rather than exit 3.

## Still outside this milestone

GPU kernels and submission, general/dynamic sizes, CUDA stream/handle/descriptor ABI compatibility, symbolic-analysis reuse, multi-RHS solve, transposed solve modes, cuSPARSE oracle comparisons, performance measurements, and integration into the running model engine remain separate work. These host primitives are not evidence that those stages are finished.
