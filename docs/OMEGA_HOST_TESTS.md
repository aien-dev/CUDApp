# Portable host checks and target-only commands

`make test` is the portable host suite. It does not need a CUDA toolkit, NVIDIA
userspace tools, a GPU device node, or a sibling Omega checkout. It builds and
runs on x86-64 Linux and on AArch64 Linux. It does not open `/dev/nvidia*` or
`/dev/aien-gpu`.

On x86-64, when you did not set `CC`, `make` uses gcc. GNU make's built-in
`CC=cc` is left alone on AArch64, so a GB10 `make` still invokes `cc`.
`make CC=clang test` still uses the compiler you named. Engine simulator
recipes that name `gcc` directly keep that compiler.

The default `CUDAPP_MEM_RESERVE_MIB` is 4096. It is not changed. On the x86-64
VM in `evidence/HOST-X86-20261030`, `make CC=gcc -j4 test` exits 2 at that
default and passes only with `CUDAPP_MEM_RESERVE_MIB=0`. The failing check
refuses a 741.9 MiB serving upload because 4.6 GiB is free and the runtime
keeps 4.0 GiB free. A host that cannot keep 4096 MiB free fails the same way.
GitHub Actions runs `CUDAPP_MEM_RESERVE_MIB=0 make CC=gcc -j4 test` and
`make CC=gcc gpu-engine-freestanding` on `ubuntu-24.04`
(`.github/workflows/host-x86.yml`). The AArch64 `host-tests` job is unchanged.
GB10 stays NOT_RUN.

On x86-64, `make test` does not count checks that execute the seeded AArch64
binding or the pinned `omc`. Those targets still exist. Invoking one directly
exits 2 and names the requirement. `make test-omega-target-only` prints the
same lines and exits 0; that exit is the report, not a pass of the named
checks. On AArch64 those checks stay inside `make test`.

`test-gpu-imadhi-rc` and `test-dagsim` stay in `make test` on x86-64. They are
host programs and do not open a device. On the VM in
`evidence/HOST-X86-20261030`, after main `b4f0480` was merged,
`CUDAPP_MEM_RESERVE_MIB=0 make CC=gcc -j4 test` exited 0 and ended
`CUDA++ HOST TESTS PASS (57 targets)`. Inside that run `test-gpu-imadhi-rc`
printed PASS for `bw_warp_sim.h`, `bw_warp_sim_tc.h`, `bw_warp_sim_k.h` and
`bw_warp_sim_pa.h`. `test-dagsim` printed `dagsim_unit: 171 checks, 0 failed`.
The same `make CC=gcc test-gpu-imadhi-rc` command by itself also exited 0.
GB10 stays NOT RUN.

Three targets in that suite are the fresh-checkout contract:

```sh
make test-omega-slice-runner      # missing compiler, missing slice, wrong pin
make test-omega-arch-guard        # x86-64 production refusal
make test-omega-barrier-contract  # dsb sy stays on AArch64; x86-64 keeps a real fence
make test-omega-binding-portable  # revision and epoch vectors
make test-omega-target-only       # names the checks this host does not run
```

## Memory ordering

`nvrm/cpu_barrier.h` is the publication barrier.

- On AArch64 the instruction is `dsb sy`. That is the GB10 host barrier. The
  production backend is not rebuilt with a no-op, an empty `asm`, or a dropped
  fence.
- On x86-64 the instruction is a sequentially consistent fence. The object
  contains that fence. It is how the portable host suite compiles the same
  sources.
- `OMEGA_BLACKWELL_HOST_SIMULATOR` is a compile definition for the fake-driver
  engine tests (`test-gpu-engine-sim` and `test-gpu-engine-backend-mutants`).
  It lets those tests call the backend on a non-AArch64 host. It does not
  enable a real device. The production library is built without it. On x86-64
  that library's Blackwell open and the direct matmul and vector entries return
  `ENOTSUP` and the text `requires AArch64 Linux`, and they do not call the
  driver.

`make test-omega-arch-guard` runs that refusal on x86-64. On AArch64 the same
target prints `NOT_RUN` and does not build the binary, because a cross
compile is not an x86-64 runtime result and the production path must not be
asked to open a device from this check. `CROSS_COMPILE_ONLY=1` records the
same `NOT_RUN` after a cross compile.

`make test-omega-barrier-contract` rejects a raw `dsb sy` outside
`#if defined(__aarch64__)`, checks the header split, and disassembles a tiny
caller. A one-line unguarded fixture must fail. If `aarch64-linux-gnu-gcc` is
absent on an x86-64 host, the AArch64 object row prints `NOT_RUN`. The source
contract still runs.

## Seeded binding on a non-AArch64 host

`src/cudapp_om_units.c` and `src/cudapp_freest_units.c` are generated AArch64
machine code (`om/units.sh`). They are not edited. A host that is not AArch64
does not assemble them.

That host links `src/cudapp_binding_portable.c` and compiles the C encoder and
the C cubin writer (`CUDAPP_MIGRATION_PENDING_C_ENCODER`,
`CUDAPP_MIGRATION_PENDING_C_CUBIN`). `make` says so. This is how the portable
suite produces instruction words and cubin images. It is not a run of
`bw_enc_sm121.om` or `cubin_elf.om`, and it is not a pass of the `.om` versus C
differential.

`omega_native_revision` and `omega_native_epoch` in the portable file follow
the committed freestanding units. The vectors were taken by executing
`src/cudapp_freest_units.c` under `qemu-aarch64` 8.2.2 on an x86-64 host
(cross compile with `aarch64-linux-gnu-gcc`). That qemu run is not a GB10
result. `make test-omega-binding-portable` checks the same vectors against
whichever file this host links. Every other seeded entry returns status 34
(`OMEGA_BAD_RT`) and does not write its result.

These commands exit 2 on x86-64. They run for real on AArch64 when their other
tools are present:

```sh
make test-om-encoder
make test-om-cubin
make test-gpu-native-runtime
make test-om-spec-encoder
make test-portable-receipts
make test-model-capacity
```

`make test-codegen-freestanding` on x86-64 still runs the freestanding C
encoder checks. The hosted `.om` comparison prints `NOT_RUN` and is not
performed. `CUDAPP_ENCODER=diff` on x86-64 is an error: that mode executes the
seeded binding.

## Omega slice gate

The eight slices stay in `omega/slices/`: `doorbell.osc`, `sequencer.osc`,
`nvrm.osc`, `va.osc`, `open_plan.osc`, `kmod.osc`, `mesa.osc`, `launch.osc`.
A missing name fails before any compiler is required.

The compiler checkout is explicit. There is no default directory.

```sh
OMEGA_OSC_COMPILER_DIR=/path/to/omega-osc-cubin tests/omega/run.sh 120
# or: tests/omega/run.sh 120 /path/to/omega-osc-cubin
# or: make test-omega-slices OMEGA_OSC_COMPILER_DIR=/path/to/omega-osc-cubin
```

`make test-omega-slices` is not part of `make test`. A fresh checkout has no
sibling compiler, so the portable suite only runs the refusal checks.

The harness must be a clean Git checkout of source commit
`1accb93164ce68a2ac2e2c680acf54782f260d62` (the pin recorded on
`roadmap/15-reproducible-host`). The runner then uses
`build/compiler/test_osc_compiler` and builds that target when the binary is
absent. Failures:

| Condition | Exit | Diagnostic |
|---|---|---|
| `OMEGA_OSC_COMPILER_DIR` unset and no directory argument | 2 | `compiler harness not configured` plus the required commit |
| Directory is not a Git checkout | 2 | `harness directory is not a Git checkout` |
| `HEAD` is a different commit | 2 | `wrong harness source` with expected and found revisions |
| Tracked or untracked changes | 2 | `harness checkout is modified` |
| A named slice file is absent | 1 | `missing required slice` and the path |
| Binary still absent after the build | 2 | `compiler binary still missing` |

The 2026-10-22 receipt on that branch recorded an AArch64 binary SHA-256 of
`a007283780347dd4a4b024844dc0c81fc09e0926f27f83e50c07d42508886f23` and a
passing eight-slice run. This tree does not re-measure that binary. A run
prints the source commit and the binary SHA-256 of whatever it actually
executed.

## What is portable, and what is not

Portable, included in `make test`:

```sh
make
make test
make test-omega-slice-runner
make test-omega-arch-guard
make test-omega-barrier-contract
make test-omega-binding-portable
build/hello_gpu --sim
make doctor
```

`make doctor` reads `/proc`, `/sys` and `/dev`. It does not open a GPU. On a
machine that is not a DGX Spark it reports the missing pieces. That report is
the result.

Target-only, not a silent pass inside `make test`:

```sh
make test-omega-slices          # needs the pinned OSC compiler above
make test-om-encoder            # AArch64; seeded encoder binding
make test-om-cubin              # AArch64; seeded cubin packer
make test-gpu-native-runtime    # AArch64; seeded native units
make test-om-spec-encoder       # pinned omc and nvdisasm
make test-portable-receipts     # pinned omc
make test-model-capacity        # pinned omc
make hello-gpu                  # GB10, opens the NVIDIA device nodes
./build/gpu_<name>_test --out receipt.json
```

`make test-gpu-engine-gb10-compile` only compiles the Phase B chip source. It
does not run it. `make test-gpu-scan` compiles `src/omega_numeric.c`, whose
host reference uses AArch64 FP instructions (`mrs fpcr`, `fmadd`). On x86-64
that compile fails in the assembler. It is not in `make test`. `make test-gpu-gemm`
is the separate GEMM host and simulator target and is also outside `make test`.

An AArch64 host build of this tree keeps `dsb sy` in `nvrm_cpu_barrier`. A
GB10 run of the device path is a separate receipt. A host or simulator pass
does not qualify it.
