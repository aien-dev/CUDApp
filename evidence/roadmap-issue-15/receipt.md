# Issue 15 host receipt

## Source and compiler pins

- CUDApp executable source: `1f281708c0c50d97ee175ba64f78c1b1fba90647` on `roadmap/15-reproducible-host`.
- Base: `349913f17c083996680c72b32f767b225005155d`.
- Omega OSC compiler source: `1accb93164ce68a2ac2e2c680acf54782f260d62`; tracked and untracked source tree clean.
- Omega compiler binary SHA-256: `a007283780347dd4a4b024844dc0c81fc09e0926f27f83e50c07d42508886f23` (AArch64 ELF).
- Host: DGX Spark Linux `aarch64`; compiler `cc` from the host toolchain. x86-64 cross compiler: `x86_64-linux-gnu-gcc 13.3.0`.
- `nvcc` and `nvdisasm` were installed on this Spark, so this was not an environment with those tools removed. The host suite and slice runner logs contain no invocation of either tool, and the host checks passed without a tool dependency.
- Baseline RED: the x86-64 cross compile of the original engine source failed on the two `dsb sy` statements at source lines 393 and 558 (exit 1); see `x86-red-baseline.log`.

## Host results

- Full host suite: PASS, exit 0, 35 targets. Command: `quiet-build timeout 1200s make OUT_DIR=build-roadmap-host15-r4 test`. Raw output: `host-suite.log`.
- Backend fake-driver simulator: 334 passed, 0 failed.
- Backend mutations: 120 total; 107 killed, 13 documented redundant, 0 test gaps.
- Omega runner negative controls: missing compiler (exit 2), missing `launch.osc` slice (exit 1), wrong compiler source pin (exit 2). Raw diagnostics are in the three `negative-*.log` files.
- Eight-slice OSC qualification: PASS. Command: `quiet-build timeout 180s env OMEGA_OSC_COMPILER_DIR="$HOME/workspace/omega-osc-cubin" tests/omega/run.sh 120`. Raw output: `omega-eight-slices.log`. It ran 94/94 `slice-run` lines and 143/143 `expect-run` lines; compiler summary: 47/47 golden programs, 138/138 negatives, 60,600 checks, 0 failures.
- x86-64 architecture-guard binary cross-compiled: PASS, exit 0. Command: `TARGET_ARCH=x86_64 CROSS_COMPILE_ONLY=1 CC=x86_64-linux-gnu-gcc OUT_DIR=build-roadmap-host15-r4/x86-cross sh tests/omega/test-arch-guard.sh`. Binary SHA-256: `1f815073efa303d06c8a52ca2e31cca8e14aaf2750cea232ce02379026deebc8`.
- x86-64 Linux runtime: NOT_RUN. This worker runs AArch64; `qemu-x86_64` is unavailable. `qemu-system-x86_64` exists, but no x86 guest runtime was configured. Cross compilation is not runtime proof.

## Artifact and hardware matrix

- `build-roadmap-host15-r4/libomega_gpu.a`: SHA-256 `c66c211d755700063c3924a0adad58dc09802f293e8a370fa31d1cee8f342421`.
- `build-roadmap-host15-r4/gpu_matmul_api_test`: SHA-256 `03f8de92a01ea3d30e8b6370eb22362f94281268c5888119860ae3a0b4e12baa`.
- Final host suite log SHA-256: `6702b7a7a089a784fc890778815ed3af5c5afd782dd84c88295faf385e3d3568`.
- Eight-slice log SHA-256: `131aa90836c6fcb8ae56d7dc197f43e0e8f5efc5db7393c69cd170a41e07a9dc`.
- x86-64 cross-compile log SHA-256: `f6524c752318cb637863f80be1c07aa49fe722323b0aed5846b7049aadbabd6e`.
- Baseline RED log SHA-256: `b3ffa29737eb08b60f3e9070fc94ba9884d28de0af131b48eb31a4bc09d326ea`.
- Host C and software simulator tests: PASS as above.
- x86-64 compile: PASS; x86-64 runtime: NOT_RUN.
- GB10 device, driver/device identity, hardware gates, hardware simulator, whole-model run, and integrated final artifact: NOT_RUN. No GPU device was opened and no server/model/driver state was changed.

Host suite output includes simulation durations and `MEASURE`-prefixed bookkeeping printed by existing tests. They are test output only and are not benchmark claims.
