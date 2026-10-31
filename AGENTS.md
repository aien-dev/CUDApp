# CUDA++ (GPU machine-code toolchain, runtime and driver client for NVIDIA GPUs without the CUDA toolkit)

> CUDA++ is the GPU half of the Omega runtime. It writes sm_121 machine code (SASS) directly, launches it through its own client for NVIDIA's open kernel module ioctls, and ships matrix multiply, attention, elementwise and mixture-of-experts libraries on top. Its build and runtime use no CUDA toolkit, `nvcc`, PTX, `libcuda` or cuBLAS. Host verification uses NVIDIA's offline `nvdisasm` as an independent instruction oracle, never as an implementation dependency. It builds with `make` and a C compiler; it runs only on the NVIDIA GB10 (DGX Spark, sm_121) with the 580.173.02 open kernel module. The repository on GitHub is `CUDApp`; the project name is written CUDA++.

Shared rules: read https://www.aienos.com/agents/RULES.md first. This file adds what is specific to this repository; where the two differ, this file wins here.

## Scope

- Applies to the whole repository.
- `src/`: encoder, kernel builder and register allocator (`omega_blackwell_*`), runtime (`omega_gpu_session`, `omega_gpu_cmdlist`, `omega_gpu_engine`, `omega_gpu_platform*`), libraries (`omega_gpu_*_api.*`). The C names keep the `omega_` prefix on purpose (`docs/NAMING.md`); do not rename them.
- `nvrm/`, `m16/`: the driver client and the completion wait.
- `third_party/nvidia-open-580.173.02/`: NVIDIA's open kernel module headers, MIT, copied unchanged. Never edit them. A new driver version goes in a new `third_party/nvidia-open-<version>/` folder (`CONTRIBUTING.md`, "Port a driver version").
- `tests/`: host tests and the chip tests (`tests/gpu_*_test.c`). `mk/*.mk`: build fragments included by the `Makefile`.
- `evidence/`: dated receipts. They record what ran on a named commit; never edit or regenerate an existing receipt. New receipts go in a new folder.
- `build/` is generated (`make`, `OUT_DIR`) and ignored by git.
- Omega is meant to build these sources at a pinned commit (`cudapp.lock` in Omega). That pin does not exist yet: Omega main still builds its own diverged copy and takes `nvrm` from physics-mv (#106, #107). Changes to public headers or the runtime API will affect Omega once it does. Ownership rule: ARCH-0042, Accepted 2026-10-30 in aien-architecture-mv#342 (Omega owns computation, CUDA++ owns GPU execution, Omega Bridge owns compatibility). The cubin writer is this repository's: `docs/CUBIN_WRITERS.md` (#108). New cubin-writer work stays here. Do not import omega-mv `seed/sm80/cubin.om`.

## Installation

```bash
git clone https://github.com/aien-dev/CUDApp.git
cd CUDApp
make -j8          # build/libomega_gpu.a, copied to build/libcudapp.a
```

Needs Linux, `git`, `make` and `gcc` (the `Makefile` uses `CC ?= gcc` and builds with `-Wall -Wextra -Werror`). Nothing from the CUDA toolkit is needed to build or run. The complete host test suite additionally requires offline `nvdisasm` for independent encoder operand checks and the pinned Omega verification tools; missing required tools are NOT_RUN, not PASS. The host build and host tests need no GPU. Running a kernel on the chip needs a DGX Spark (GB10) with NVIDIA's open kernel module 580.173.02 loaded and `/dev/nvidia0` and `/dev/nvidia-uvm` present.

## Configuration

- Build variables: `CC` (default `gcc`), `OUT_DIR` (default `build`), `PHYSICS_DIR` (default `.`; Omega overrides it).
- `CUDAPP_MEM_RESERVE_MIB`: memory the runtime keeps free before any allocation reaches the driver (default 4096; `0` turns the check off). Tests point the check at a fake `/proc/meminfo` with `omega_gpu_platform_linux_test_set_meminfo` (no environment variable chooses that file).
- `CUDAPP_DOCTOR_ROOT` points `make doctor` at a fake machine root; `CUDAPP_DOCTOR_SHOW=1 sh tests/doctor/run.sh build/cudapp-doctor` prints the doctor's output on the nine fake machines.
- `AIEN_OMEGA_DGRAPH_BARRIER` and `AIEN_OMEGA_DGRAPH_PREFETCH` tune the decode graph (`src/omega_gpu_decode_graph.c`).

## Usage

Host commands (no GPU opened):

```bash
make test                     # portable host suite, x86-64 and AArch64: about 7 minutes
make test-native-artifact     # offline native artifact export test
make zero-cuda-gate           # the built programs need no CUDA library (self-test first)
make doctor                   # is this machine ready? reads /proc, /sys and /dev only
build/hello_gpu --sim         # the first kernel on the host warp simulator
make test-omega-slices        # eight OSC slices; needs OMEGA_OSC_COMPILER_DIR; not part of make test
python3 tools/figures/make_figures.py --check   # README figures match the code
python3 tools/figures/test_stack_legend.py -v   # rendered legend check; needs Inkscape
```

- `docs/OMEGA_HOST_TESTS.md` separates portable commands from target-only ones. `make test` needs no NVIDIA tool and no sibling checkout. On x86-64 the production Blackwell open returns `ENOTSUP` and names AArch64, and the seeded AArch64 binding is not assembled: the C encoder and C cubin writer are the portable bodies, and `make test-om-encoder`, `make test-om-cubin`, `make test-gpu-native-runtime`, `make test-om-spec-encoder`, `make test-portable-receipts` and `make test-model-capacity` exit 2. AArch64 keeps `dsb sy` and keeps those rows in `make test`. `make test-omega-arch-guard` prints `NOT_RUN` on AArch64 and does not build the binary.

- Judge `make test` by its exit code and its last line, `CUDA++ HOST TESTS PASS (<count> targets)`, where `<count>` is computed from the current `HOST_TESTS` manifest in `Makefile`. The log contains many `FAIL` and `ERROR` lines by design (mutation tests and negative controls that must fail). Do not grep the log for them.
- `make test` already runs the host-only `test-gpu-*` targets listed in `HOST_TESTS` in the `Makefile`. Run those through `make test`, not by hand.
- `make zero-cuda-gate` reads ELF metadata only (`tools/zero-cuda-gate.sh`).
- `test-portable-receipts` is included in `make test` on AArch64. It needs full Git history plus the pinned `OMEGA_OMC` and `OMEGA_BRIDGE` used by `test-om-units`. On x86-64 it is outside `make test` and exits 2: the pinned compiler is not part of the portable suite. The `.om` checker verifies original and scrubbed-payload SHA-256, exact HOME-substitution reversal, original Git-blob identity, and independent refusal controls. The TSV separately pins each complete derived file. These are portable derivatives, never rewritten original receipts or qualification authority.
- `python3 tools/figures/make_figures.py --check` must print `figures current: <n>`. Regenerate figures only when you change the data they are drawn from, and commit the regenerated SVGs with that change.

GPU commands (need a GB10; do not run from an agent session unless the maintainer has given you a GPU window on the DGX Spark):

```bash
make hello-gpu                              # runs make doctor, then examples/hello_gpu on the GPU
./build/gpu_<name>_test --out receipt.json  # chip gates; each test's comment in the Makefile says how it is run
```

`make cubin-probe` (`tools/interop/cubin_probe.c`) links NVIDIA's `libcuda` and refuses to run unless `SPARK_WINDOW` is set. It is an interop check, not part of CUDA++; never add it to a host target.

## Conventions

- Nothing counts until it runs and leaves a receipt: the command, the commit, the machine and what it printed, saved under `evidence/<topic>/` (`CONTRIBUTING.md`).
- Write the test first and watch it fail for the expected reason; include a negative control. A library change also needs a host reference, a simulator run where the simulator covers it, a mutation test and a chip receipt.
- A failing test means the work is wrong. Never skip, loosen or rerun a test until it passes once.
- Every instruction encoding has golden words in the tests and a chip run behind it. Do not change an encoder field without both.
- No NVIDIA binary, cubin, PTX, firmware or listing printed by NVIDIA's tools goes in this repository (`docs/AUDIT.md`). The former trace-emitter regression pin was removed in #93; its history is recorded in `PROVENANCE.md` section 6. The rules for allowed sources are in `PROVENANCE.md`. If you used NVIDIA's compiler or disassembler on your own small kernels to learn an encoding, say so in a code comment.
- The runtime API is versioned. `docs/runtime/omega_runtime_api.manifest` is checked against the C headers it names and against `docs/runtime/API.md` by `make test` (`test-runtime-api`, with negative controls). Change a public header, the manifest and `API.md` together.
- Status words in `README.md` and `ROADMAP.md` are PASS (receipt on a named commit), PARTIAL, MISSING, BUILDING. Do not upgrade a status without the receipt.
- New source files carry `SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception`. Keep third-party notices (`THIRD_PARTY.md`, `NOTICE`).
- Plain English, short sentences, every number with the command that produced it.
- Pull request CI runs the host-only `make test` suite, pinned Omega binding regeneration and mutants, SBOM generation and GitHub CodeQL. Its disposable arm64 host-test runner installs only the checksum-verified NVIDIA `nvdisasm` 13.0.85 executable and the bridge source pinned by `om/omega-bridge.lock`. The archive URL and SHA-256 are fixed in `.github/workflows/ci.yml`, from NVIDIA's `redistrib_13.0.2.json` manifest. The oracle is verification-only: no CUDA driver, runtime or compiler is installed, no NVIDIA binary is committed, and no GPU is opened. Provisioning errors and missing-oracle checks remain failures. Put local results in the pull request with their exact source revision; all-green host CI is not physical qualification. `.github/workflows/host-gates.yml` runs `make -j4`, `make test-native-artifact`, `make zero-cuda-gate`, `build/hello_gpu --sim`, read-only `make doctor`, and `python3 tools/figures/make_figures.py --check`. `make test` stays in the `host-tests` job. The rules are `docs/CI_ACCEPTANCE.md`.

## Safety and approvals

- Never run a chip test, `make hello-gpu` or anything that opens `/dev/nvidia*` on a shared DGX Spark without an authorized GPU window from the maintainer. A live model may be serving on the same GPU.
- Never commit secrets, private paths or host addresses. `tools/extract/audit-scan.sh` and its rule files show what the publication audit checks for.
- Ask before changing the platform contract (`src/omega_gpu_platform.h`), the driver client's ioctl usage, the qualified chip table (`src/omega_gpu_chip.c`) or the licence files.
- Work on a branch named for what it does. Never force-push `main`; the `main` ruleset blocks force pushes and deletion.

## Further reading

- `README.md`: what CUDA++ is, the driver part by part, how to run the first kernel.
- `CONTRIBUTING.md`: the receipt rule and how a change gets in.
- `ROADMAP.md`: every NVIDIA piece on the Spark, its replacement and receipt.
- `docs/DRIVER.md`, `docs/HOW_I_DID_IT.md`, `docs/runtime/API.md`, `docs/runtime/PORTABILITY.md`.
- `docs/AUDIT.md`, `docs/LICENSING_TRANSITION.md`, `THIRD_PARTY.md`: provenance and licences.
- `docs/CUBIN_WRITERS.md`: the two cubin writers, who owns new work, and what was not run (#108).
- Aienux, the Aien kernel, which holds Aien's GB10 driver research harness (L332, aienux#441; not a product, NVIDIA's open kernel module stays and GB10 support goes upstream to Nova and nouveau): https://github.com/aien-dev/aienux

## Cursor Cloud specific instructions

Cloud Agent VMs are x86_64. The sources use the ARM `dsb sy` barrier, so `/usr/bin/gcc` cannot build them (CUDApp#15). `/usr/local/bin/cc` and `/usr/local/bin/gcc` are the aarch64 cross compiler, which is what `make` runs (`CC` defaults to `cc`). The built binaries are aarch64. qemu-user is registered against `/usr/aarch64-linux-gnu` at boot, and again by `/usr/local/bin/make` if that boot hook did not run. If a binary exits with `Exec format error`, run any `make` target once, then retry it.

`make test` is the host suite. Judge the exit code and the last line, `CUDA++ HOST TESTS PASS (<count> targets)`. The log contains intentional `FAIL` lines. Under qemu it is slower than the arm64 CI runner. The attention simulator is one slow part. `test-codegen-freestanding` sweeps every 2^32 bf16 input; a 20 million iteration cast loop took 18 seconds here, so that sweep is on the order of an hour. `build/hello_gpu --sim` is the simulator hello world. `python3 tools/figures/make_figures.py --check` checks the README figures.

Pinned tools sit where the Makefile looks, next to the checkout:

- `../omega-mv` is omega-mv `e5b32567`, `seed/out/omc`, sha256 `2c82480596bbdc50cda9626866209ef662d5834f278f839ff4ce0250c495029f`
- `../omega-bridge` is omega-bridge `0c4cb9ff980af278890b231b994eaede4dc7621c` (`om/omega-bridge.lock`)
- `nvdisasm` 13.0.85 is the x86_64 redistributable from NVIDIA `redistrib_13.0.2.json` (CI uses the linux-sbsa archive of the same version)

`/usr/local/bin/make` exports `CUDAPP_MEM_RESERVE_MIB=0` when the variable is unset. MemAvailable on this VM under-reports because of the hypervisor balloon, and the default 4 GiB reserve then refuses the Qwen-shaped upload in `test-gpu-serving-alloc`. The platform test unsets the variable and still checks the 4096 default against a fake meminfo. Do not copy that `0` onto a Spark.

`make doctor` prints NOT READY here: no NVIDIA module and no `/dev/nvidia*`. That is the expected diagnostic. The doctor binary is aarch64, so its system line says aarch64; `uname -m` in the shell is x86_64. Do not run `make hello-gpu` or open `/dev/nvidia*`.
