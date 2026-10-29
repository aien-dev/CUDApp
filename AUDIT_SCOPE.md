# Audit scope

Written for an outside, independent security audit. Revision described:
`780a0af1f08c1786ea53a4e8702ca8bb1396fdbb` on `main`, 2026-10-26. Everything below is taken from
the repository's own files at that revision; items marked NOT_RUN were not executed while writing
this document.

## What this repository is

CUDA++ (repository name `CUDApp`) is the GPU half of the Omega runtime: a C library that writes
NVIDIA sm_121 machine code (SASS) directly, launches it through its own user-space client for
NVIDIA's open kernel module ioctls, and ships matrix multiply, attention, elementwise and
mixture-of-experts libraries. It uses no CUDA toolkit, `nvcc`, PTX, `libcuda` or cuBLAS. It builds
with `make` and `gcc` on Linux; it runs kernels only on the NVIDIA GB10 (DGX Spark) with open kernel
module 580.173.02. What ships is `build/libomega_gpu.a` (copied to `build/libcudapp.a`). Host
simulators, references and Python scripts under `tests/` and `tools/` are test oracles. Status:
research stage; current native qualification of this checkout is NOT_RUN.

## What to review, in priority order

1. `nvrm/nvrm.c`, `nvrm/nvrm.h`. The driver client: opens `/dev/nvidiactl`, `/dev/nvidia0` and
   `/dev/nvidia-uvm`, allocates resource-manager objects, maps buffers at the same address for CPU
   and GPU, builds the command ring and rings the doorbell. Check ioctl argument structures and
   sizes against `third_party/nvidia-open-580.173.02/`, error handling on every call, and that
   mappings cannot alias or outlive their objects.
2. `m16/m16_gpu_wait.c` and `m16/m16_native.c`. The completion wait (hard deadline never extended,
   stall deadline restarted only on progress) and native path. Check that a hung GPU cannot hang
   the process forever and that an unknown completion state stops further GPU work.
3. `src/omega_gpu_platform.h`, `src/omega_gpu_platform*.c`. The platform contract: typed handles,
   explicit rights per buffer, no silent fallback, the `CUDAPP_MEM_RESERVE_MIB` reserve check
   before allocation, refusal of CPU-polled buffers that are not GPU-uncached. AGENTS.md requires
   approval before changing this contract.
4. `src/omega_gpu_session.*`, `src/omega_gpu_cmdlist.*`, `src/omega_gpu_engine.*`. Session
   lifetime, command lists, shader-cache invalidation before new code runs, wait-for-idle between
   dependent launches, and the job engine returning exactly one named failure.
5. `src/omega_blackwell_encoder.c`, `src/omega_blackwell_codegen*.c`, `src/omega_blackwell_qmd.c`,
   and the libraries `src/omega_gpu_*_api.*`. Generated code and launch descriptors: check that
   grid, block, register and shared-memory sizes and buffer bounds derived from caller shapes
   cannot address memory outside the caller's buffers.
6. `src/omega_gpu_chip.c`. The qualified chip table consulted by `make doctor`.
7. `tools/doctor/cudapp_doctor.c` and `tools/zero-cuda-gate.sh`. Read-only diagnostics; check that
   they cannot report READY or PASS for a wrong module version or a binary that imports CUDA.
8. `tools/interop/cubin_probe.c`. Links NVIDIA's `libcuda` and refuses to run unless
   `SPARK_WINDOW` is set; check it is not reachable from any host target.
9. `tools/extract/audit-scan.sh` with `rules-secrets.txt` and `rules-private.txt`, and
   `tools/extract/audit-selftest.sh`. The publication secret scan and its self-test.
10. `.github/`. Issue templates, a pull request template and, added by the audit-readiness
    change, `workflows/ci.yml`: `make test` (host-only targets, no GPU) on a GitHub-hosted
    arm64 runner plus an SPDX SBOM artifact; actions pinned by commit sha, read-only token.
    AGENTS.md states that CodeQL (C/C++ and Python) also runs on pull requests; how CodeQL is
    configured is unknown from the repository.

## Trust boundaries

- Untrusted input: tensor shapes, sizes and buffers passed by callers of the library API
  (`docs/runtime/API.md`, `src/omega_runtime_api.h`); environment variables `CUDAPP_MEM_RESERVE_MIB`,
  `CUDAPP_DOCTOR_ROOT`, `AIEN_OMEGA_DGRAPH_BARRIER`, `AIEN_OMEGA_DGRAPH_PREFETCH`;
  `/proc` and `/sys` contents read by the doctor.
- Trusted: NVIDIA's open kernel module 580.173.02, its headers in `third_party/`, and the
  signed GSP firmware. The kernel module enforces the privilege boundary between processes; this
  library runs entirely in user space with the calling user's device-file permissions.
- Shared memory: on the GB10 CPU and GPU share one memory, and every buffer is mapped at the same
  address for both. A GPU kernel bug therefore corrupts host-visible process memory.
- Isolated: host tests use simulators and a fake driver; they open no device.

## Threat model summary

- Assets: process memory shared with the GPU, the integrity of computed results, availability of
  the machine (memory exhaustion freezes a DGX Spark per README), and other processes' GPU work.
- Attackers considered: a caller passing hostile shapes or sizes; a misbehaving or hung GPU; a
  wrong kernel module version; secrets leaking through committed receipts.
- Defended: memory reserve refusal before the driver; bounded completion waits; refusal of work
  after an unknown completion; doctor refusal of other module versions; ELF check that built
  programs import no CUDA library; secret scanning before publication.
- Not defended / explicitly out of scope: no driver-version or PCI-ID admission guard on direct
  execution or library linking (README); host results are not hardware qualification; the
  diagnostic is not a sandbox; one device and one process only; NVIDIA kernel module and firmware
  bugs.

## Build from source at a pin

Linux with `git`, GNU `make` and `gcc` (`CC ?= gcc`, `-Wall -Wextra -Werror`). No CUDA toolkit.
There is no compiler pin file; the doctor records the compiler version it was built with.

```sh
git clone https://github.com/aien-dev/CUDApp.git
cd CUDApp
make -j8
```

Build variables: `CC`, `OUT_DIR` (default `build`), `PHYSICS_DIR` (default `.`). The host build
needs no GPU. Fresh x86-64 portability is tracked in issue #15 (`docs/START_HERE.md`).

## Run every test

```sh
make test
make test-native-artifact
make zero-cuda-gate
build/hello_gpu --sim
python3 tools/figures/make_figures.py --check
make doctor
make hello-gpu
./build/gpu_<name>_test --out receipt.json
```

- `make test`: every host-only target in `HOST_TESTS` (`Makefile`). Judge by exit code and the
  last line `CUDA++ HOST TESTS PASS (<count> targets)`. The log contains `FAIL` and `ERROR` lines
  by design (mutation tests and negative controls). About 7 minutes. NOT_RUN here.
- `make test-native-artifact`: offline native artifact export test. NOT_RUN here.
- `make zero-cuda-gate`: ELF metadata check with a self-test negative control. NOT_RUN here.
- `build/hello_gpu --sim`: first kernel on the host warp simulator. NOT_RUN here.
- `make_figures.py --check`: AGENTS.md records that it fails on current `main`
  (`stale figures ...: stack.svg`). NOT_RUN here.
- `make doctor`: reads `/proc`, `/sys`, `/dev` only; a non-GB10 host is expected to report NOT
  READY.
- `make hello-gpu` and the chip gates `build/gpu_*_test`: need a GB10 and an authorized GPU window
  on the DGX Spark. NOT_RUN on a plain host.

## Known limitations and NOT_RUN

- Current native qualification of this checkout is NOT_RUN. The imported
  `evidence/m16-blackwell-native-path-requalification-receipt.json` is an unauditable historical
  summary whose raw logs are absent (README).
- One device, one process; no streams or events; a timeout ends GPU use for the process.
- No any-shape GEMM; several libraries are shaped for specific models (README table).
- `tools/figures/make_figures.py --check` is known stale on `main`.
- Some host tests call NVIDIA `nvdisasm` as a cross-check when it is installed.
- `evidence/security-host-20261026/` records a host-only run with GPU NOT_RUN and full
  `make test` NOT_RUN.

## Dependencies and provenance

- No lockfile at the root; C with no external libraries beyond libc and libm for the library.
  `native/learn53/omega653.lock` is a sub-component lock.
- Vendored: `third_party/nvidia-open-580.173.02/` (67 NVIDIA headers, MIT, unchanged,
  `SHA256SUMS` in that folder). Mesa NVK/NAK is cited in comments for hardware facts, no code
  copied. Notices in `THIRD_PARTY.md` and `NOTICE`. History and licence audit in `docs/AUDIT.md`,
  `docs/LICENSE_PROVENANCE.md`, `docs/LICENSING_TRANSITION.md`.
- AGENTS.md: no NVIDIA binary, cubin, PTX, firmware or NVIDIA tool listing goes in the repository.
- Omega builds the same sources at a pinned commit (`cudapp.lock` in omega-mv).
- No network access at build or test time is documented. At the described revision the repository
  had no GitHub Actions workflow files; the audit-readiness change adds `.github/workflows/ci.yml`
  (item 10 above), whose only network use is fetching the pinned actions on the runner.

## Evidence

Receipts live in `evidence/<topic>/`, each recording what ran on a named commit. AGENTS.md: never
edit or regenerate an existing receipt; new receipts go in a new folder. Status words in README and
`ROADMAP.md` are PASS only with a receipt on a named commit. Cross-repository receipts, where filed,
belong under `aien-dev/aien-receipts` `CUDApp/<sha>/`; whether any exist is unknown from the
repository. Receipts are append-only and never edited.
