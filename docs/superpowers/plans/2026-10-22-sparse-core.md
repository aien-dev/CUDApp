# Sparse Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Add three tested sparse-library primitives in Omega.
**Architecture:** Separate structural conversion, binary32 sparse multiplication, and triangular solve files share fixed CSR input/output pools and validation. A binary wire adapter exercises actual native executables. Publish once after all rungs pass.
**Tech Stack:** Seed-born Omega compiler, Linux AArch64, Python standard-library oracle.
**Spec:** docs/superpowers/specs/2026-10-22-sparse-core.md

## Global Constraints
Use the spec capacities (64 dimensions, 256 entries), transactional visible outputs on failure, no vendor code or GPU/API claim; only the SHA-pinned seed compiler. Reuse arithmetic byte-identically from f564d78. Own worktree feat/sparse-core based on L452.

## Review Focus
- Unsorted/duplicate entries: stable conversion and deterministic separately rounded products.
- Empty/zero-width and maximum-capacity matrices: no out-of-bounds access.
- Structural result exceeding 256: refuse before any visible output mutation.
- Triangular structural and arithmetic faults: no fabricated success or partial solution.
- Binary length/flags/indices: reject before publishing bytes; direct core checks bypass adapter validation.

### Task 1: Conversion and shared validation
Files: omega/libraries/sparse_core.om, sparse_convert.om; tests/libraries/sparse_core_driver.om and test_sparse_core.py; mk/sparse-core.mk.
Interfaces: cootocs(rows,cols,nnz), csrtoco(rows,cols,nnz) -> int; pools defined in spec. Validator pvalid(rows,cols,nnz,which), which 0=A, 1=B.
- [ ] Write conversion oracle/adapter tests: stable row grouping, literal rectangular result, opaque NaN/zero bits, malformed CSR/COO, all empty and capacity boundaries.
- [ ] Run against refusal-only functions; expect the first valid conversion to fail.
- [ ] Implement shared pools/validation and stable count/scatter conversion.
- [ ] Run conversion tests and compiled index/value mutants; expect PASS and mutants caught.
- [ ] Commit task with evidence.

### Task 2: CSR sparse multiplication
Files: omega/libraries/sparse_gemm.om; extend same adapter/tests.
Interfaces: consumes pvalid and shared A/B/output pools; spgemm(rows,inner,cols,anz,bnz) -> int.
- [ ] Add literal and per-column path-enumeration binary32 oracle tests, random duplicate/unsorted inputs, cancellation, output-capacity failure and zero shapes.
- [ ] Run refusal stub; expect valid product failure.
- [ ] Implement one-row dense accumulator plus structural markers, staged result arrays and atomic final copy.
- [ ] Run tests and compiled product/ordering mutants; expect bit-exact outputs and mutants caught.
- [ ] Commit task with evidence.

### Task 3: Triangular solve
Files: omega/libraries/sparse_solve.om; reuse arithmetic binary32.om/fdiv.om; extend adapter/tests, add sparse_core_state.om.
Interfaces: consumes validated A/rhs and finite binary32 arithmetic; trisolv(n,nnz,upper,unit) -> int, sol output.
- [ ] Add upper/lower explicit/unit diagonal oracle cases, constructed systems, duplicate off-diagonals, invalid triangles/diagonals/nonfinite/overflow cases; direct output-preservation and repeated calls.
- [ ] Run refusal stub; expect valid solve failure.
- [ ] Implement validate-first row substitution into staged solution; copy only on success.
- [ ] Run all sparse tests plus compiled divide/direction/validation mutants; expect PASS and mutants caught.
- [ ] Commit implementation and source provenance.

### Task 4: Whole-branch verification and publication
Files: docs/SPARSE_CORE.md, evidence/sparse-core/.
- [ ] Run make test, make test-csr-transpose and make test-sparse-core with pinned omc; expect exit 0, no skipped failures.
- [ ] Review whole diff, request fresh reviewer; record provider refusal if unavailable rather than claim independent review.
- [ ] Publish compiler/source hashes, commands, failed red runs, green runs, mutants and explicit host-only scope.
- [ ] Push feat/sparse-core and verify remote SHA; write/read READY naming inherited L452 branch and its supersession; close L454 with evidence.
