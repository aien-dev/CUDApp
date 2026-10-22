# Sparse core review record

Scope: the additive sparse-core branch above inherited CSR transpose commit
57a4d45cbc3eecafffb4e425bec9fa53d5642c4a. Compiler pin and binary32/fdiv are reused,
not competing implementations.

## Independent-review attempt

A fresh read-only Codex review was launched on the Spark for ledger L455 with
`codex exec --sandbox read-only --model gpt-6.1-sol -c model_reasoning_effort=high`.
Its brief is `~/handoffs/briefs/L455-sparse-review.txt`. It exited 1 before returning
findings with: `You’ve hit your usage limit.` The CLI reported a retry time of
10:20 AM. No independent review occurred. No second implementation worker was
started and the run was not retried. L455 stays open for the desk's review.

## Author review

This section is an author review, not an independent approval.

- All dimension/entry bounds precede dimension-indexed loads in public functions.
  Validated CSR pointers are zero-based, bounded, monotone and end at nnz; hence
  every visited entry index lies in its declared prefix.
- COO coordinates are fully validated before conversion writes output. Stable
  row scatter does not sort columns or drop duplicates/payload bits.
- CSR expansion reads validated A arrays, writes separate COO arrays, and never
  modifies input CSR arrays.
- Both multiplication operands are validated before symbolic/numeric work.
  Column indices are bounded before accessing row-local accumulators. Each
  row resets all markers/accumulators for the declared output width. Ascending
  column emission is independent of input ordering.
- SpGEMM stages row pointers, indices and values; a capacity failure never calls
  publish. Direct state tests and an early-publish mutant verify this property;
  wire-only refusal tests would miss the corrupted core buffers.
- Structural zeros survive. Explicit stored zero times infinity generates a NaN;
  absent entries do not perform that multiplication. Separate multiply and add
  order is tested, including subnormals and cancellation.
- Solve validates triangle, finite inputs and diagonal counts for every row
  before substitution. Lower reads only earlier staged rows; upper reads only
  later staged rows. Matrix indices cannot access outside the vector shape.
  Every product/accumulator/division result is checked for finiteness. Results
  copy into sol only after all rows succeed; the direct probe exercises a late
  overflow after an earlier row was computed.
- Wire lengths are calculated only after bounded header validation. Maximum
  accepted packets and responses fit their fixed buffers. Malformed input
  cannot publish output because encode/emit occur only after core success.
- Oracle implementations differ structurally from the Omega algorithms.
  Constructed triangular systems have independently known exact solutions;
  extra floating-point cases use separately rounded Python operations.
- A mutation is counted only after both adapter and direct-state variants
  compile outside the mismatch catcher. Nonempty AArch64 ELF headers are
  required before execution, avoiding an empty executable falsely passing.
- No gate was removed, skipped or weakened; no GPU state, vendor driver,
  running server or other worker's checkout was changed.

No unresolved correctness issue found within the documented fixed-size host
scope. Independent desk review and merge are still pending. GPU execution,
CUDA ABI compatibility and performance are not qualified by these receipts.
