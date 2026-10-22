# D11: open libraries with an easy default path

Labels: library, D11

**The pain.** Closed libraries nobody outside NVIDIA can read or fix; on new chips they fall back or fail ("SM121 falls through to cutlass/cudnn only.", FlashInfer #3170). `docs/PAIN_POINTS.md` section 2.10.

**What exists.** Matrix multiply, matrix-vector, attention, elementwise and routing kernels in source, shaped for the models AIEN runs.

**What is left.** Any-shape GEMM in BF16 and FP8, reductions, prefix sums and sorts, each with a host reference. This is Omega bar C3, owned by ledger L386; contributions go through it.

**Done when.** Omega bar C3's receipts: 200+ GEMM shapes from 1 to 8192 against a host reference, speed as a fraction of the measured tensor-core ceiling; reductions and scans 1 to 2^28 elements bit-exact.
