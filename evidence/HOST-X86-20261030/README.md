# x86-64 portable host suite

Measured 2026-10-30 on Linux 6.12.94+ x86_64, gcc 13.3.0 (Ubuntu 13.3.0-6ubuntu2~24.04.1), 4 cores.
`MemTotal` was 16387280 kB. `MemAvailable` during `make doctor` was 4.8 GiB.
`/dev/nvidiactl`, `/dev/nvidia0`, `/dev/nvidia-uvm` and `/dev/aien-gpu` were absent.
No GPU device was opened. `make hello-gpu` was not run.

`make test` and the commands in the sections below ran on the working tree recorded by
`0d8084b26d6183ebac8f12861801aae733a4b67c`. This directory was added afterwards.
It is not an input to `make`. The qemu command was then repeated with the probe
source stored here. `cmp` showed that copy identical to the source of the first run.

## Portable suite

`CUDAPP_MEM_RESERVE_MIB=0 make CC=gcc -j4 test` exited 0.
Last line: `CUDA++ HOST TESTS PASS (55 targets)`.
Selected lines are in [make-test-pass-lines.txt](make-test-pass-lines.txt).
The log also contains intentional `FAIL` lines from mutation tests and negative controls.
The exit code and the last line are the result.

An earlier `make CC=gcc -j4 test` with the default reserve (4096 MiB) exited 2.
The serving allocation test refused a 741.9 MiB upload because 4.6 GiB was free and the runtime keeps 4.0 GiB free.
That line is in [default-reserve.txt](default-reserve.txt). The default was not changed.
The passing suite sets the reserve to 0 because this machine does not have the free memory the default requires.

Inside the passing suite:

- `omega-runner-test: PASS (missing harness, missing slice, wrong source pin)`
- `Blackwell backend refusal: ENOTSUP, requires AArch64 Linux, driver not entered`
- `direct Blackwell entry points: ENOTSUP`
- `barrier codegen: PASS (x86_64 seq_cst fence, no dsb)`
- `barrier codegen: PASS (aarch64-linux-gnu-gcc cross compile emits dsb sy)`
- `barrier-contract: PASS`
- `binding-portable: PASS`
- `hello --sim tests: 12 passed, 0 failed`
- engine simulator `SUMMARY passed=334 failed=0`
- matmul bounds `SUMMARY passed=6 failed=0`
- `test_gpu_artifact_export: 47 checks, 0 failed`
- `PASS: freestanding codegen contract (14 fixtures, 0 failing checks)`
- `mutants: 120 total, 107 killed, 13 survived as redundant, 0 survived as test gaps`
- `allowlist control: a malloc caller is refused by the nm -u check (as required)`

`nm -u build/gpu_fs/omega_gpu_freestanding.o` on the object that suite used lists `memcmp memcpy memset strcmp strlen`.
The recipe's own line, from the build of that object in this tree, was `freestanding GPU link (15 files): only memcmp memcpy memset strcmp strlen undefined`.

`env -u CC make -n test-omega-binding-portable` prints `CC=gcc`. GNU make's built-in `CC=cc` is replaced when `CC` was not set.

## Separate commands

`make zero-cuda-gate` exited 0. [zero-cuda.txt](zero-cuda.txt).
The self-test rejects a CUDA probe. `build/hello_gpu` and `build/gpu_matmul_api_test` link `libm` and `libc` only.

`./build/hello_gpu --sim` exited 0 and printed `PASS`. [hello-sim.txt](hello-sim.txt).
That is the host warp simulator. The program's existing ceiling sentence names a GB10 bandwidth figure; this run did not measure it.

`make doctor` exited 1 and printed `NOT READY: 5 problems above`. [doctor.txt](doctor.txt).
The kernel module is absent, the three device nodes are absent, and `/sys` shows no NVIDIA GPU.
The doctor checks device nodes with `access`. It did not open one.
The seeded target table row is `NOT_RUN` because the pinned `omc` is not on this machine.

## Refusals

[refusals.txt](refusals.txt):

- `make test-om-encoder` exited 2 and printed `test-om-encoder: NOT_RUN (requires AArch64 Linux; src/cudapp_om_units.c is AArch64 machine code and was not executed)`.
- `make test-omega-slices` with no `OMEGA_OSC_COMPILER_DIR` exited 2. It named `compiler harness not configured` and source commit `1accb93164ce68a2ac2e2c680acf54782f260d62`.
- `make test-omega-target-only` exited 0 after printing the seven refusal lines. That exit is the report. It is not a pass of those checks.
- `gh api repos/aien-dev/omega-osc-cubin` returned HTTP 404.

## Host cross checks, not GB10 results

`aarch64-linux-gnu-gcc` 13.3.0 compiled `nvrm/cpu_barrier.h` during `make test`.
`aarch64-linux-gnu-objdump` showed `dsb sy`. That is a cross compile on this x86-64 host.

`src/cudapp_freest_units.c` was linked by:

```sh
aarch64-linux-gnu-gcc -std=gnu11 -O2 -Wall -Wextra -Werror -Isrc -o /tmp/rev_epoch_probe evidence/HOST-X86-20261030/rev_epoch_probe.c src/cudapp_freest_units.c
qemu-aarch64 -L /usr/aarch64-linux-gnu /tmp/rev_epoch_probe
```

The source file in this directory is the file that was compiled. `qemu-aarch64` is 8.2.2.
The run exited 0 and printed:

```
rev0 st=0 r=1
rev1 st=0 r=2
revmax-1 st=0 r=18446744073709551615
revmax st=0 r=0
revmis st=32
revnull st=32
ep0 st=0 r=1
ep3 st=0 r=4
epmax-1 st=0 r=18446744073709551615
epmax st=0 r=18446744073709551615
epnull st=32
```

Status 0 is `OMEGA_OK`. Status 32 is `OMEGA_BAD_ARG`. This is qemu-user on x86-64, not a GB10 run.

`omega_fp32_to_fp16_rne` was compared with gcc's `_Float16` cast for every `uint32_t` bit pattern.
The program included `src/omega_blackwell_matmul.h`, compiled with `gcc -O2 -Wall -Wextra -Werror`.
A fresh rebuild of that source was byte-identical to the binary that ran.
The run exited 0 and printed `compared 4294967296 mismatches 0 first u=00000000 got=0000 want=0000`.
That comparison is not a GB10 result. AArch64 still uses the hardware cast.

## NOT RUN

- Native AArch64 `make test`. This machine is x86-64. The AArch64 barrier object above is a cross compile.
- GB10 device behavior, `make hello-gpu`, and `./build/gpu_*_test` on a chip.
- The eight-slice gate. `aien-dev/omega-osc-cubin` is not available (HTTP 404). Source pin `1accb93164ce68a2ac2e2c680acf54782f260d62` was not checked out, and the historical binary SHA-256 `a007283780347dd4a4b024844dc0c81fc09e0926f27f83e50c07d42508886f23` was not re-measured.
- `make test-om-encoder`, `make test-om-cubin`, `make test-gpu-native-runtime`, `make test-om-spec-encoder`, `make test-portable-receipts` and `make test-model-capacity` as real runs. On this host each exits 2. The lines inside `make test` are the report from `make test-omega-target-only`.
