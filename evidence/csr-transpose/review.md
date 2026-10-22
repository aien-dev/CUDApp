# L452 implementation review

Scope: `omega/libraries/csr_transpose.om`, binary adapter, direct state probe,
independent Python oracle, optional Make target, and user documentation.

A separate reviewer was requested through Hermes delegation, but the configured
Anthropic provider refused before spawning because of its usage limit. No second
worker ran. The implementation author performed the following review; this is
not an independent-review claim. The desk still reviews before merging.

- Dimension and nnz bounds precede all dimension-indexed reads.
- `trp[0] == 0`, `trp[rows] == nnz`, monotonicity and per-row upper bounds imply
  every visited entry index is within `[0,nnz)`.
- All column indices are checked before count/scatter. No output or scratch
  mutation occurs on an invalid core call, including an invalid final entry.
- Counts use `column+1` within `[1,cols]`. The prefix pass leaves each next-column
  cursor at its starting offset. Total count is nnz, so scatter destinations stay
  in `[0,nnz)`. Processing rows and entries in input order preserves duplicates.
- No arithmetic is performed on value payloads. The wire adapter copies low
  32 bits and accepts only complete u32 words.
- Wire dimensions are bounded before computing expected byte length; exact
  length is checked before loading arrays. Output is emitted only after success.
- Empty dimensions, 256 repeated entries, the last allowed row/column, arbitrary
  payload bits, UINT_MAX indices, malformed offsets and every truncated prefix
  have executable coverage. The direct probe additionally checks output AND
  scratch preservation on error and repeated calls in one process.
- Three separately compiled behavioral mutants must fail. There are no mocks,
  tolerance relaxations, skipped checks, CUDA library calls or GPU claims.
- The seed compiler is SHA-pinned. The documented two-stage build was exercised
  from immutable source files and a third stage confirmed the fixed point.
- The direct state probe initially failed compilation because `iteration` is a
  nine-character variable, while the pinned seed accepts at most eight. It was
  renamed `iter`, with the failed log retained, and the full new suite passed.

No unresolved correctness issue found within the documented bounded host scope.
Non-reentrancy, fixed capacities and lack of CUDA ABI/GPU support are explicit
limitations, not silently waived acceptance criteria.
