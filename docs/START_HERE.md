# Start here: run and verify CUDA++

This is the shortest route for someone who has not worked on AIEN. The repository is named `CUDApp`; the project is called **CUDA++**. It is a research-stage GPU machine-code toolchain and runtime, not yet a drop-in replacement for all of CUDA. Read [the roadmap](../ROADMAP.md) for the tested, partial, and missing components.

## 1. Know what is supported

| Path | What it establishes | Current boundary |
| --- | --- | --- |
| Build and host tests | Compiles the library and checks host references, simulators, negative controls and API contracts. | Linux with gcc. `make test` is the portable suite on x86-64 and on AArch64. It needs no NVIDIA tool and no sibling checkout. Target-only checks name their requirement (`docs/OMEGA_HOST_TESTS.md`, [#15](https://github.com/aien-dev/CUDApp/issues/15)). |
| `build/hello_gpu --sim` | Runs the example through the **host simulator** and checks its answers. | No GPU is opened; this is **not** hardware qualification or a performance result. |
| `make doctor` | Reports whether the current Linux host matches the supported native GPU path. | Read-only diagnostic; a non-GB10 host is expected to report NOT READY. |
| `make hello-gpu` | Runs the vector-add example on the actual GPU and checks output against the CPU. | Qualified target: NVIDIA GB10 (DGX Spark, sm_121), Linux, kernel driver 580.173.02. Requires safe, authorized GPU access. |

The current runtime bypasses NVIDIA's **user-space** CUDA toolkit and `libcuda` API. It still relies on NVIDIA's **kernel module** and signed GPU firmware. This is not a CUDA-compatibility guarantee for unmodified CUDA applications or for arbitrary NVIDIA/AMD/Intel GPUs. See [the compatibility roadmap](../ROADMAP.md) and [portability contract](runtime/PORTABILITY.md).

## 2. Clone and run host-only checks

On a Linux host with `git`, GNU `make`, `gcc`, and `python3` for the optional figure check:

```sh
git clone https://github.com/aien-dev/CUDApp.git
cd CUDApp
git rev-parse HEAD
uname -s -m

make -j4
make test
make test-native-artifact
make zero-cuda-gate
build/hello_gpu --sim
```

`make test` is the host-only suite. Its success marker is `CUDA++ HOST TESTS PASS (<count> targets)`; `<count>` is computed from the current `HOST_TESTS` manifest in `Makefile`, not a fixed historical count. The suite contains intentional failure/mutation cases: judge the **exit code and final result**, not the presence of the words FAIL/ERROR in intermediate logs. It can take several minutes. `make zero-cuda-gate` inspects ELF metadata and runs its own negative control; it is **not** a proof that kernel drivers or firmware have disappeared.

The code-allocation source scan inside `make test` expands only existing source files and refuses an empty readset, a missing/unreadable input, or any scanner error. Its negative controls also require the intended refusal diagnostic, so a setup failure cannot count as detecting forbidden allocation sizing. A clean scan is only a check of the selected source idioms, not GPU execution evidence.

On a host without a seeded binding (neither AArch64 nor x86-64), `make test-gpu-native-runtime` is outside `make test` and exits 2.
On AArch64 and x86-64 it extends the existing session-platform fixture
with generated native Quant, Composer, encoder, residency and selector calls,
then the actual FP8 cache and command-list integration. Platform handles are
synthetic host fixtures, not live GPU handles. Recorded/inflight references
remain pinned on timeout; only observed completion or confirmed session
teardown permits their retirement. This does not qualify or activate native
candidates: production graph state remains absent until the existing
independent receipt/admission path accepts exact artifacts.

The same host fixture executes the actual graph recording and incumbent
nonempty-submit adapters. It checks that successful command-list ordinals are
not mistaken for error status. Paired submission has independent per-buffer
QMD-publication controls: omission for either queue must refuse. No synthetic
completion or handle is hardware or full-model evidence.

`make test-portable-receipts` is part of `make test` on AArch64 and needs full Git
history, `OMEGA_OMC`, and `OMEGA_BRIDGE`. Public derived receipts substitute
`$AIEN_RECEIPT_HOME` for the producer's HOME prefix. Their metadata states the
original Git blob, original SHA-256, exact substitution rule, and scrubbed
payload SHA-256. The scrubbed hash domain is the document with exactly its
inserted metadata line removed; `tests/portable_receipts.tsv` also pins the
complete derived-file SHA-256. The `.om` checker reconstructs original bytes
from each derivative and compares both hashes and the immutable original Git
blob, with refusal controls. Originals and historical commits are unchanged.
On x86-64 the same target is outside `make test` and exits 2, because the
pinned compiler is not part of the portable suite.

Canonical consumer integration uses the existing single-commit `cudapp.lock`,
not a second artifact or qualification registry. The isolated Omega consumer
cutover builds an authenticated committed source archive and links its actual
callers to that canonical library:

```sh
OMEGA_OMC=/absolute/path/to/pinned/omc \
OMEGA_BRIDGE=/absolute/path/to/pinned/omega-bridge \
make -C /absolute/path/to/Omega check-cudapp \
  CUDAPP_DIR=/absolute/path/to/committed/CUDApp
```

`tools/cudapp-stage.sh` verifies the exact lock, exports committed source,
requires native-unit regeneration agreement, and builds the canonical archive.
Fresh compiled Omega guards authenticate the staged source and emit its
source/archive/library provenance. Working checkout edits are not archived.
Consumer headers resolve from `stage/source`; the library is
`stage/build/libcudapp.a`, linked after the Omega CPU/evidence adapter companion.
Wrong/missing pins and changed staged source refuse; `LOCK_CHECK` is not a
bypass. No merge, deployment, live serving consumption, chip qualification or
default native activation follows from this development cutover.

The acceptance boundaries remain [#5](https://github.com/aien-dev/CUDApp/issues/5)
and [#106](https://github.com/aien-dev/CUDApp/issues/106). Historical extraction
maps and receipts remain unchanged. The obsolete
[consumer PR #618](https://github.com/aien-dev/omega-mv/pull/618) used an older
pin and different source-check contract; it is not the canonical cutover base.

On x86-64, `make test` is the portable host suite. Production `dsb sy` stays on AArch64 (`nvrm/cpu_barrier.h`). The x86-64 publication barrier is a real sequentially consistent fence. A target-only check that names its requirement and exits 2 is not a pass of that check. The two lists are in [docs/OMEGA_HOST_TESTS.md](OMEGA_HOST_TESTS.md) ([#15](https://github.com/aien-dev/CUDApp/issues/15)).

Optional: `python3 tools/figures/make_figures.py --check` verifies that README diagrams match the generator. Check it on your exact commit.

## 3. Diagnose a native GPU machine

```sh
make doctor
```

The doctor inspects the host and reports compatibility. It does not open a GPU device. **NOT READY is an expected result** on unsupported chips, absent driver/device nodes, or mismatched driver releases. Do not bypass its refusal to try hardware.

The doctor is a wrapper-only preflight, also run by `make hello-gpu`. Direct
binaries and applications linking the library have no driver-version or PCI-ID
admission guard. READY means the diagnostic checks matched, not that this commit
is chip-qualified. The imported M16 summary lacks its referenced raw logs; see
the README evidence limitation. Current native qualification is NOT_RUN.

Only when the machine is a qualified GB10 and you are authorized to use it without interrupting a serving workload:

```sh
make hello-gpu
```

The example adds one million floats and checks its answer against the CPU. Exit 0 and PASS describe that run, on that machine and commit. Do not run chip tests on a shared DGX Spark without a coordinated quiet window.

## 4. How to evaluate the evidence

- The primary implementation is under [`src/`](../src/), with the driver client in [`nvrm/`](../nvrm/) and [`m16/`](../m16/). The hello-world implementation is [`examples/hello_gpu/hello_gpu.c`](../examples/hello_gpu/hello_gpu.c).
- Read [the actual runtime API](runtime/API.md) and [driver explanation](DRIVER.md), then inspect the code and [provenance audit](AUDIT.md). The third-party licenses are described in [THIRD_PARTY.md](../THIRD_PARTY.md).
- Published results and numbers apply to the exact documented workload, commit, hardware and driver. A host simulator PASS does not prove machine-byte correctness on another GPU; a GB10 receipt does not establish general GPU support or performance superiority.
- `evidence/` contains experimental receipts. A dated or historical receipt does not prove an unrelated later commit passed the same gate. Preserve existing receipts; create a new folder for new evidence.
- Current runtime implementation is primarily C. Migration toward an exclusively first-party `.om` implementation is **planned, not complete** ([#79](https://github.com/aien-dev/CUDApp/issues/79)). Do not present a future compiler or driver as already shipped.

To help, follow [CONTRIBUTING.md](../CONTRIBUTING.md) and use an existing issue template. If you find a vulnerability, follow [SECURITY.md](../SECURITY.md) instead of posting exploit details publicly.

## 5. What to attach to an independent review

Report the full source commit (`git rev-parse HEAD`), OS and architecture (`uname -s -m`), compiler (`gcc --version`), commands with exit codes, and the relevant output. For a permitted GB10 test, include chip/driver identity and the immutable evidence path. Clearly mark each gate **PASS**, **FAIL** or **NOT RUN**; a skipped hardware gate is never PASS.

For the first public release bar and feature gaps, use [ROADMAP.md](../ROADMAP.md), [host reproducibility #15](https://github.com/aien-dev/CUDApp/issues/15), and [native `.om` migration #79](https://github.com/aien-dev/CUDApp/issues/79).
