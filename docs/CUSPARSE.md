# cuSPARSE, the slice that runs

I am replacing cuSPARSE one operation at a time. This page is the first slice. It is not a claim that CUDA++ replaces cuSPARSE.

## What runs

A host CSR matrix-vector product, `y = A x`, in binary32, built by the same pinned seed-born Omega compiler as the FFT work. No C compiler and no NVIDIA library is in the build. The binary is statically linked and does not load `libcusparse`.

The walk is left to right along each row. Each product and each sum rounds to binary32. Checked against an independent Python oracle that rounds the same way, including an empty row, a rectangular matrix, a zero-size product, and requests that must be refused before any output: too many rows or columns, a column past the end, a row pointer that goes backwards, a trailing byte, a non-zero reserved word, and a product into a zero-width vector. A copy that forgets to scale by `x` compiles and fails the value check.

Receipt: `evidence/fft-rng/sparse.log`.

## What this is not

- Not the cuSPARSE API. No `cusparseHandle_t`, no `cusparseSpMV`, no alpha or beta, no transpose, no COO, CSC, BSR, or blocked formats.
- Not on the GPU. The live server holds the chip, so I have not compared this to the installed `libcusparse.so.12.6.3.3`. That comparison is an oracle run, still open.
- Not a sparse triangular solve, SpMM, or SpGEMM.

## Source

The contract I matched is the documented CSR product, one row at a time, from the public cuSPARSE page (https://docs.nvidia.com/cuda/cusparse/index.html). I used the page for the operation, not for code. The installed library was not linked, copied, or disassembled.
