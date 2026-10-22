# Reproducible host checks

`make test` runs the portable host suite. It does not need a CUDA toolkit,
NVIDIA userspace tools, a GPU device node, or an Omega sibling checkout. The
Blackwell backend's real Linux driver open is supported only on AArch64. A
non-AArch64 build can compile the library, but its backend refuses to open with
`ENOTSUP`; the fake-driver backend test opts into
`OMEGA_BLACKWELL_HOST_SIMULATOR` and uses a sequentially consistent host fence.
That simulator setting does not enable the real backend. The AArch64 production
barriers remain `dsb sy`.

`test-omega-arch-guard` runs the production refusal behavior on a non-AArch64
Linux runtime. On AArch64 it prints `NOT_RUN` for the x86-64 runtime row; a
cross-compile alone does not change that row.

The portable host suite includes the fresh-checkout runner refusal checks:

```sh
make test-omega-slice-runner
```

Omega slice qualification is separate because it consumes a compiler source
checkout. `tests/omega/run.sh` requires the full pinned source commit
`1accb93164ce68a2ac2e2c680acf54782f260d62`, checks that tracked source files are
clean, and prints the compiled harness SHA-256. Point it at an existing checkout
with an argument or environment variable:

```sh
OMEGA_OSC_COMPILER_DIR="$HOME/workspace/omega-osc-cubin" tests/omega/run.sh 120
# or: tests/omega/run.sh 120 /path/to/omega-osc-cubin
```

If the binary is absent, the runner builds the pinned checkout's
`build/compiler/test_osc_compiler` target and then runs it against the eight
required slices: `doorbell.osc`, `sequencer.osc`, `nvrm.osc`, `va.osc`,
`open_plan.osc`, `kmod.osc`, `mesa.osc`, and `launch.osc`. The harness source and
binary digests, command, exit status, and raw output belong in the issue's
evidence receipt. No bundled compiler binary or disassembler is used.

On Linux x86-64, `make test` is the runtime host gate. A cross compile checks
source portability but is not an x86-64 runtime result. If no x86-64 Linux
runtime (native, QEMU user mode, or a VM) is available, record that runtime row
as `NOT_RUN`. Simulator and GB10 gates remain separate; this document does not
qualify either one.
