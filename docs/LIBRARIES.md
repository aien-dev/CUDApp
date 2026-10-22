# More of the CUDA toolkit, in Omega

These are host programs, built by the pinned seed-born Omega compiler. They are not the CUDA APIs, they do not link a NVIDIA library, and they have not run on the GPU. The live server holds the chip.

Expected bits that are not exact integers were computed in Wolfram and copied into the Omega self-checks. The programs do not ask Python for the answer.

| Piece | What runs | What it is not |
| --- | --- | --- |
| division and square root | binary32, round to nearest, ties even | a libm call |
| cuSOLVER | lower Cholesky of a 3 by 3 matrix, and a refusal when a pivot is not positive | `cusolverDn` , pivoting, QR, the GPU |
| NPP | add, multiply, subtract, and a greater-than threshold on eight binary32 pixels | the NPP image API |
| nvJPEG | one 8 by 8 inverse DCT, JPEG scale, cosine table from Wolfram | a JPEG file decoder |
| level-1 products | dot, axpy, scal, and the 2-norm | cuBLAS GEMM, which is a separate bar |

Run `make test-libraries-native OMEGA_SEED_SOURCE=<omega clone>`. The five new programs must exit 0. A square-root copy whose rounding shift is wrong must exit non-zero. Receipt: `evidence/fft-rng/omega-libs.log`.
