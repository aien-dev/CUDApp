# CSR transpose in Omega

This is a host implementation of a sparse storage conversion, not the cuSPARSE ABI and not a GPU kernel. It adds a structural primitive alongside the separate sparse matrix-vector work on `feat/fft-rng`; it does not replace that branch or take over its implementation.

`omega/libraries/csr_transpose.om` turns a zero-based compressed sparse row (CSR) matrix into compressed sparse column (CSC) storage. The same output arrays describe the transposed matrix in CSR. It copies value words without doing arithmetic. Negative zero, infinities, and NaN payload bits survive unchanged. Duplicates and explicitly stored zeros are retained. Within each output column, rows appear in ascending order; duplicates in the same input row keep their original order. No NVIDIA code or library is used.

## Contract

The current fixed capacities are 64 rows, 64 columns, and 256 stored entries. These are bounds, not a general cuSPARSE size claim.

Concatenate the source with an Omega caller. Fill globals `trp` (row pointers), `tci` (column indices), and `tav` (value words); call `transpose rows cols nnz`. On success it returns 0 and writes `tcp` (column pointers), `tri` (row indices), and `tva` (value words). The representation uses `ulong` cells; callers supply 32-bit value words. Array prefixes of length `cols + 1`, `nnz`, and `nnz` are the defined output. Unused tails are unspecified.

It returns 1 for a dimension beyond capacity, an initial pointer other than zero, a final pointer other than `nnz`, descending/out-of-range pointers, or a column outside `[0, cols)`. Validation completes before any output or scratch cell is modified. Empty dimensions and zero stored entries are supported. Repeated calls reset counts and cursors. Globals make this implementation non-reentrant; concurrent calls require separate program instances. No dynamic allocation, aliasing input/output arrays, index-base conversion, or duplicate summation is offered.

The algorithm counts entries per column, forms column offsets, and stably scatters entries in input row order. Work is linear in rows + columns + stored entries. No speed claim has been measured.

## Run it

The pinned compiler emits Linux AArch64 executables. Python is only the test/oracle/build harness; the implementation is Omega. Set `OMEGA_SEED_COMPILER` to a seed-born `omc` whose SHA-256 is:

```
10d7565239ca6bd3cca344b65a03c0f3498acd95e82ab3d420bd01408b01398e
```

This is the compiler from Omega commit `f623467e21f9ea8a862a7c92965227fb67263e8e`. The tests refuse a different compiler. Reuse that qualified compiler if already available. To build the same compiler in a fresh checkout on Linux AArch64, without a C compiler:

```sh
git clone https://github.com/aien-dev/omega-mv.git omega-csr-seed
git -C omega-csr-seed checkout --detach f623467e21f9ea8a862a7c92965227fb67263e8e
cd omega-csr-seed
mkdir -p build/csr-seed
chmod +x seed/rung4/om4.bin
seed/rung4/om4.bin < seed/omc/omc.om > build/csr-seed/stage1
chmod +x build/csr-seed/stage1
build/csr-seed/stage1 < seed/omc/omc.om > build/csr-seed/omc
chmod +x build/csr-seed/omc
sha256sum build/csr-seed/omc
```

`seed/README.md` in that checkout documents rebuilding the checked-in `om4.bin` from the earlier seed rungs. The library suite reuses the existing qualified compiler rather than redoing that common work.

From the CUDApp checkout:

```sh
make test-csr-transpose OMEGA_SEED_COMPILER=/absolute/path/to/omega-csr-seed/build/csr-seed/omc
make test
```

`test-csr-transpose` is separate from the C host suite because the pinned compiler has a specific host target. It is never silently skipped: missing/wrong compiler or failed execution is an error.

## Binary test adapter

`tests/libraries/csr_transpose_driver.om` reads little-endian unsigned 32-bit words from stdin: `rows cols nnz reserved`, then `rows+1` row pointers, `nnz` column indices, and `nnz` value words. `reserved` must be zero; input length must be exact. On success stdout contains `cols+1` column pointers, `nnz` row indices, and `nnz` unchanged value words. Malformed input exits 2 with empty stdout. This is a test protocol, not a public CUDA API.

The Python oracle uses stable sorting of `(column,row,value)` tuples, independently of the count/prefix/scatter algorithm. Tests cover a literal rectangular matrix, all empty-dimension combinations, capacity edges, 120 deterministic random matrices with arbitrary 32-bit payloads, malformed offsets/indices/lengths, every truncation of a valid request, direct output/scratch preservation on refusal, and repeated calls. Compiled mutations corrupt a value, corrupt a row index, or delete the initial-pointer check; each must be caught by executing the changed program.

Receipts are under `evidence/csr-transpose/`. Host-only status remains explicit until a separately gated GPU implementation exists.
