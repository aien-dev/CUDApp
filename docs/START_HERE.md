# Start here: run and verify CUDA++

This is the shortest route for someone who has not worked on AIEN. The repository is named `CUDApp`; the project is called **CUDA++**. It is a research-stage GPU machine-code toolchain and runtime, not yet a drop-in replacement for all of CUDA. Read [the roadmap](../ROADMAP.md) for the tested, partial, and missing components.

## 1. Know what is supported

| Path | What it establishes | Current boundary |
| --- | --- | --- |
| Build and host tests | Compiles the library and checks host references, simulators, negative controls and API contracts. | Linux is the documented build environment. The full suite has previously passed on the DGX Spark host; fresh x86-64 portability remains tracked in [#15](https://github.com/aien-dev/CUDApp/issues/15). |
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

`make test` is the host-only suite. On a qualified build environment its final success marker is `CUDA++ HOST TESTS PASS`, followed by the target count computed from that revision's `HOST_TESTS` list. Record the exact commit, exit code and full marker in your receipt. The suite contains intentional failure/mutation cases: judge the **exit code and final result**, not the presence of the words FAIL/ERROR in intermediate logs. It can take several minutes. `make zero-cuda-gate` inspects ELF metadata and runs its own negative control; it is **not** a proof that kernel drivers or firmware have disappeared.

If an x86-64 checkout fails on an architecture-specific instruction, record the full error and compare with [the existing reproducibility issue #15](https://github.com/aien-dev/CUDApp/issues/15). Do not suppress the memory-ordering instruction or call a partial host run a passing full suite.

Optional: `python3 tools/figures/make_figures.py --check` verifies that README diagrams match the generator. A previously observed `stack.svg` mismatch is documented in [AGENTS.md](../AGENTS.md); check it again on your exact commit rather than assuming it is resolved.

## 3. Diagnose a native GPU machine

```sh
make doctor
```

The doctor inspects the host and reports compatibility. It does not open a GPU device. **NOT READY is an expected result** on unsupported chips, absent driver/device nodes, or mismatched driver releases. Do not bypass its refusal to try hardware.

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
