# cuFFT and cuRAND, what is actually replaced

I am replacing these two CUDA toolkit libraries with my own code. This page says what runs today. It is not a claim that CUDA++ replaces cuFFT or cuRAND.

## What runs

Host programs, built by the pinned seed-born Omega compiler (`f623467e`, omc sha256 `10d7565239ca6bd3cca344b65a03c0f3498acd95e82ab3d420bd01408b01398e`). No C compiler, no CUDA tool, and no NVIDIA library is in that build.

- **Philox4x32-10.** The algorithm is the one in Salmon, Moraes, Dror, and Shaw, "Parallel Random Numbers: As Easy as 1, 2, 3" (SC11). The implementation is mine. It is checked against the three published known-answer vectors and against an independent Python transcription of that paper, including counter carry, offset, split calls, and malformed requests. Receipt: `evidence/fft-rng/native.log` (`PHILOX PASS 405 checks`).
- **binary32 add and multiply.** Integer code for IEEE binary32, round to nearest, ties to even. Checked against an independent Python oracle, including subnormals, overflow, signed zero, and NaN. Receipt: same log (`BINARY32 PASS 41000 arithmetic cases`).
- **Radix-2 complex FFT.** Forward and inverse, inverse unnormalized, power-of-two lengths 1 through 1024, binary32 rounding at each butterfly. Checked against an independent discrete Fourier transform, not against this FFT. A sign mutant fails. Receipt: same log (`FFT PASS 207 checks`).

Run it with `make test-libraries-native OMEGA_SEED_SOURCE=<omega clone that contains f623467e>`.

## What this is not

- Not a `libcufft` or `libcurand` stand-in. There is no cuFFT plan, no cuRAND generator handle, and no CUDA stream argument.
- Not on the GPU. The live server holds the chip (`aien-serve-http` on port 18091). I have not run these algorithms through the CUDA++ encoder, and I have not compared them to the installed cuFFT 12.0.0.61 or cuRAND 10.4.0.35. Those libraries are oracles only, and that comparison is still open because it needs a GPU window.
- Not cuRAND compatibility. Philox4x32-10 equality is not the same as cuRAND's seed, subsequence, and offset ordering. I have not probed `curandGenerate` yet, so I do not claim that ordering.
- Not the rest of either API. Missing, and refused rather than guessed: real and half-precision transforms, batches, strides, ranks above 1, in-place workspace, every RNG other than Philox, and every distribution other than the raw Philox words.

## Sources I used

- Salmon et al., SC11, for Philox. The known-answer vectors are the ones that paper publishes.
- IEEE 754-2019 for binary32 rounding. I did not take the arithmetic from a NVIDIA header.
- cuFFT's documented forward sign and unnormalized inverse, from the public cuFFT library page (https://docs.nvidia.com/cuda/cufft/index.html). I used that page for the contract, not for code.

NVIDIA's installed libraries were not linked, copied, or disassembled.
