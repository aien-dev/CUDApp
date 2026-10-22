# Rotation-aware quantizer implementation plan

Execution: personal, inline, no delegated review. Spec: native/rotquant/CONTRACT.md.
Goal: a native deterministic encoder/decoder plus a reproducible equal-budget
comparison on frozen real weights, not a deployable model conversion.
Architecture: integer FP8 decoder -> optional 128-point butterfly -> nearest-even
symmetric INT4 -> packed frame; inverse reconstruction returns exact rationals.
Tech: seeded Omega; Python/NumPy/ml-dtypes independent reference and orchestration.
Base: stack on #48 at 566dc53502321166f4a898a29f8e4f828a209239; reuse its safety and
compiler helpers, without modifying its product files. Existing compiler SHA
191b776ef98821597c3e30655da6d680a9b66edbc9ceaca3c39c61ea6df098e9.

Review focus: sign/rounding ties; inverse normalization; scale-tile alignment;
overflow/domain refusals; stale PASS and unsafe reused output directories.

1. Contract and baseline
- [x] Create isolated feat/rotation-int4-native worktree and run inherited codec gate.
- [x] Save contract and Wolfram algebra/range/accounting checks before results.
2. Native codec (native/rotquant/rotquant.om, test_rotation.py)
- [x] Write exact independent fixtures (explicit Hadamard matrix, not butterfly).
- [x] Run them against a compiled empty program; retain meaningful RED.
- [x] Implement bounded native encode and rational reconstruction.
- [x] Require all-byte, signed/tie, boundary, malformed and native I/O controls.
- [x] Compile mutants for sign, inverse normalization and ties; require each caught.
3. Comparison and qualification (qualify_rotation.py, experiment.py)
- [x] Reuse pinned compiler and #48 output safety/receipt retirement helpers.
- [x] Read fixed source and scale bytes, assert header/shape/scale-grid contracts.
- [x] Compare native codes/numerators to an independent NumPy explicit-matrix oracle.
- [x] Compute paired weight/output errors and equal storage; retain every row.
- [x] Run inherited gates and complete existing host suite in this owned worktree.
4. Publication
- [ ] Personal source review, immutable export replay, push branch and verify GitHub bytes.
- [ ] Update #49 and L600 with precise first-slice results and remaining quality/runtime gates.

Not in this slice: real activation capture, calibrated clipping, GPU INT4 execution,
logits/perplexity/task quality, whole-model compression, default-path replacement.
These stay OPEN under #49; no synthetic probe is labeled model qualification.
