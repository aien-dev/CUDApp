# More of the CUDA toolkit, in Omega

These are host programs, built by the pinned seed-born Omega compiler. They are not the CUDA APIs, they do not link a NVIDIA library, and they have not run on the GPU. The live server holds the chip.

Expected bits that are not exact integers were computed in Wolfram and copied into the Omega self-checks. The programs do not ask Python for the answer.

| Piece | What runs | What it is not |
| --- | --- | --- |
| division and square root | binary32, round to nearest, ties even | a libm call |
| cuSOLVER | 3 by 3 Cholesky and its solve, 3 by 3 LU with partial pivoting and its solve, a 2 by 2 Householder QR, a 2 by 2 symmetric eigendecomposition, the singular values of a 2 by 2, a 4 by 4 Cholesky, and the singular vectors of a 2 by 2 (axis and one off-diagonal). A zero pivot refuses. A negative pivot refuses. A copy that skips the first row swap fails the LU check. A copy that skips the rank-1 subtraction fails the 4 by 4 factor | the cuSOLVER API, singular vectors above 2 by 2, the GPU |
| NPP | add, multiply, subtract, and a greater-than threshold on eight binary32 pixels | the NPP image API |
| nvJPEG | one 8 by 8 inverse DCT, JPEG scale, cosine table from Wolfram | a JPEG file decoder |
| level-1 products | dot, axpy, scal, and the 2-norm | cuBLAS GEMM, which is a separate bar |
| level-2 product | y = alpha A x + beta y for n at most 4 | the cuBLAS API, GEMM, the GPU |
| CUB scan | inclusive and exclusive prefix sums of at most 16 binary32 values | a scan of 2^28, the device API, the GPU |
| cuRAND XORWOW | the step in the public header, and the seed-0 mix. A copy that adds 362438 instead of 362437 fails the first output | curand_init skip-ahead, a live libcurand call, the GPU |

The XORWOW step is the listing on page 5 of Marsaglia, Journal of Statistical Software 8(14), 2003. CUDA 13.0.3 `curand()` is that listing. The published initial state is 123456789, 362436069, 521288629, 88675123, 5783321, 6615241. A 32-bit run of that listing and a Wolfram evaluation of it agree on the first eight outputs, and the Omega self-check requires those outputs. The header comment says seed 0 sets that published state. The function it calls does not: it always adds the salts. MRG32k3a in the same header does skip the mix when the seed is 0. XORWOW does not. The seed-0 state is 1478573778, 1163863128, 4171507460, 3705206908, 1360900310, 716983765. libcurand was not called: the live server holds the device.

Run `make test-libraries-native OMEGA_SEED_SOURCE=<omega clone>`. The new programs must exit 0. Receipts: `evidence/fft-rng/omega-libs.log`, `evidence/fft-rng/solvers.log`, and `evidence/fft-rng/more.log`.
