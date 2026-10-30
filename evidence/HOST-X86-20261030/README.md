# x86-64 portable host suite

On this VM, `make test` passes only with `CUDAPP_MEM_RESERVE_MIB=0` and exits 2 at the default 4096 MiB. The default was not changed. The refusing line is in [default-reserve.txt](default-reserve.txt).

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

## Drift gate, same failure on main

`tools/extract/drift.sh /tmp/omega-mv HEAD` against omega-mv main `089e5b05350ee61cc1439f17d3d3052a18a3e0a5` ("Warp simulators refuse IMAD.HI.U32 with a register Rc, which the chip reads as a 64-bit pair (#514)", committer date 2026-10-30T12:10:11Z) exits 1 on this branch and on CUDA++ main `78fea51d6f8859ec8cc5e5561295fd5583b92c19`. Both name the same seven paths and the same omega blobs. The lines are in [drift-main-and-pr.txt](drift-main-and-pr.txt).

That omega commit is the one that moved the seven paths. The previous omega main, `4c60d6baf5f1`, was the pin in the green main run [41050251136](https://github.com/aien-dev/CUDApp/actions/runs/41050251136) at 2026-10-30T11:45:16Z: `DRIFT-GATE: PASS (729 GPU paths, omega 4c60d6baf5f1, none moved away from CUDA++ since the baseline)`. At `4c60d6baf5f1` and at `b3dca0f1e28d`, the omega `Makefile` blob is the baseline blob `4e068c1de5026d2ac4f6c2463fbb4292e6c26ec0`. At `089e5b05350e` it is `a42b2b1f4cd7ffb683d1f54cadbceebde45d5e15`.

CUDA++ main `Makefile` blob is `c1c714d34dc57cb251bc5e2f948b4430cb73b51e`. This branch's `Makefile` blob is `ee85035b395b6cdef53e51661fb43b5bf6bd146c`. Neither equals omega's new blob, so the Makefile row fails on main as well. This branch did not introduce it.

The other six paths are untouched here. Their CUDA++ blobs match main, and `tests/gpu_imadhi_rc_test.c` is absent on both:

| Path | CUDA++ main and this branch | omega `089e5b05350e` |
|---|---|---|
| `src/omega_blackwell_codegen.h` | `9dd4f62116ed66ff12c60c06e9070f5aa8ead679` | `1f0b70eecd3860da68bbe013860f27ca01da0d2f` |
| `tests/bw_warp_sim.h` | `8652710eeaed8de86375a1c9eb46b1a0d5946463` | `6d93828f3979b3a7223ebe0f1952b53e1d414764` |
| `tests/bw_warp_sim_k.h` | `2b040ed7756ddcffe716e198b266efa3c276dd54` | `993875647985cc99ce27a78003df26df91fd18e1` |
| `tests/bw_warp_sim_pa.h` | `cee1061ba9d8cf88bd80ae80295863105e2d0810` | `157e396cde9d942cc0bfff02fd057ed393c52a0a` |
| `tests/bw_warp_sim_tc.h` | `16699a135bc1c7bf3a3faffdb0116c223ad83390` | `a08e38b6a05d815448fe8b4546d204aff6f3e166` |
| `tests/gpu_imadhi_rc_test.c` | absent | `179596a27cd1c24e02ec8459a45f2e7216ca2424` |

The path count is 739 here and 730 on main because this branch adds CUDA++ files to the map. The drift count stays 7. CI on this pull request stays red on `drift` until CUDA++ holds omega `089e5b05350e`'s bytes for those seven paths. That import is not part of this change.

## AArch64 objects match main

GB10 runs stay **NOT RUN**. There is no GB10 and no `/dev/aien-gpu` here.

Method: `aarch64-linux-gnu-gcc` 13.3.0 (`Ubuntu 13.3.0-6ubuntu2~24.04.1`) compiled each shared C file from this tree and from main `78fea51d6f8859ec8cc5e5561295fd5583b92c19` (`/tmp/cudapp-main`). Production flags were `-std=gnu11 -Wall -Wextra -Werror -D_GNU_SOURCE -O2` plus the `Makefile` include path, without `CUDAPP_PORTABLE_DEFS` and without `OMEGA_BLACKWELL_HOST_SIMULATOR`. `src/omega_blackwell_matmul.c` was also compiled freestanding (`-ffreestanding -fno-builtin -fno-stack-protector -DOMEGA_GPU_NO_POSIX`). `cmp` was byte-identical for every object. SHA-256 values are in [aarch64-objects.tsv](aarch64-objects.tsv). One hash is enough per row because the two files matched.

With `uname -m` forced to `aarch64`, `make` on this tree and on main both leave `CC` as `cc`. The production recipe adds an empty `$(CUDAPP_PORTABLE_DEFS)`. The shell collapses that extra space. The compiler arguments otherwise match main, and the objects above are the outputs.

On x86-64, `make` with no `CC` in the environment sets `CC := gcc`.

## Filed extension proposals

Omega filed the two drafts:

- Host CPU publication barrier: https://github.com/aien-dev/omega-mv/issues/939 (`nvrm/cpu_barrier.h`, [proposal-cpu-barrier.md](proposal-cpu-barrier.md))
- Portable native binding: https://github.com/aien-dev/omega-mv/issues/940 (`src/cudapp_binding_portable.c`, [proposal-binding-portable.md](proposal-binding-portable.md))

After those comment links replaced the drafted-not-filed sentences, the same `aarch64-linux-gnu-gcc` 13.3.0 command was run again. All eleven objects in [aarch64-objects.tsv](aarch64-objects.tsv) kept the same SHA-256. `src/cudapp_binding_portable.c` is not in that list. The AArch64 library does not compile it.

## NOT RUN

- Native AArch64 `make test`. This machine is x86-64. The AArch64 barrier object above is a cross compile.
- GB10 device behavior, `make hello-gpu`, and `./build/gpu_*_test` on a chip.
- The eight-slice gate. `aien-dev/omega-osc-cubin` is not available (HTTP 404). Source pin `1accb93164ce68a2ac2e2c680acf54782f260d62` was not checked out, and the historical binary SHA-256 `a007283780347dd4a4b024844dc0c81fc09e0926f27f83e50c07d42508886f23` was not re-measured.
- `make test-om-encoder`, `make test-om-cubin`, `make test-gpu-native-runtime`, `make test-om-spec-encoder`, `make test-portable-receipts` and `make test-model-capacity` as real runs. On this host each exits 2. The lines inside `make test` are the report from `make test-omega-target-only`.
