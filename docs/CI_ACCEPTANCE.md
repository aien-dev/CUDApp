# CI acceptance

A pull-request check is a result for the commit GitHub built. A green host check is not a GB10 qualification. Commands that did not run are `NOT_RUN`.

The workflow is `.github/workflows/host-gates.yml`. It composes with the jobs that already exist:

- `make test` is the `host-tests` job in `.github/workflows/ci.yml`. That job judges the process exit code and the last line, `CUDA++ HOST TESTS PASS (<count> targets)`. The log keeps intentional `FAIL` and `ERROR` lines from mutants and negative controls. This policy does not start a second copy of that suite.
- `make test-runtime-api` is `.github/workflows/runtime-api.yml`. Issue #189 owns any further change to that job.
- x86-64 `make test` is the `x86-host-tests` job in `.github/workflows/host-x86.yml`. This policy does not add that job.

## What each gate is

| Command | Where it runs | What a pass is |
| --- | --- | --- |
| `make -j4` | `host-gates` / `fresh-checkout`, `ubuntu-24.04-arm`, 30 minute limit | Exit 0. `build/libomega_gpu.a` and `build/libcudapp.a` exist. Their sha256 is uploaded. |
| `make test` | `ci.yml` / `host-tests`, same arm runner | Exit 0 and the final `CUDA++ HOST TESTS PASS` line. Needs the pinned `nvdisasm` 13.0.85, full git history, and the pinned Omega tools named in that workflow. |
| `make test-native-artifact` | `host-gates` / `fresh-checkout` | Exit 0, `ARTIFACT EXPORT SHAPES PASS` from `tools/artifact_export_shapes.sh`, and a last summary `test_gpu_artifact_export: <n> checks, 0 failed`. |
| `make zero-cuda-gate` | `host-gates` / `fresh-checkout` | Exit 0. The log still shows `ZERO_CUDA FAIL` for the stub `libcudart` probe, then `ZERO_CUDA SELF-TEST PASS`, then `ZERO_CUDA PASS` for `build/hello_gpu` and `build/gpu_matmul_api_test`. |
| `build/hello_gpu --sim` | `host-gates` / `fresh-checkout` | Exit 0, the host-simulator sentence, the 1048576-element check, the work line, the pinned output sha256, and the same hashes on a second run. This is the warp simulator. |
| `make doctor` | `host-gates` / `fresh-checkout` | The runner has no NVIDIA module and no GPU node, so `make` exits 2 and the log contains `NOT READY:` plus the five refusal lines and `NOT_RUN` for the seeded target table. A line `READY. Next: make hello-gpu` fails the job. Chip qualification stays `NOT_RUN`. |
| `python3 tools/figures/make_figures.py --check` | `host-gates` / `figures`, `ubuntu-latest`, 5 minute limit | Exit 0 and `figures current: <n>`. `docs/img/stack.svg` is one of those files. |

`build/hello_gpu --sim --bug=past-end` must exit 1, print the `FAULT` line, and name `0 bytes past the end of buffer a`. That path does not print `PASS`. `build/hello_gpu --bug=past-end` must exit 2 and refuse to run off the simulator. Those two are negative controls. `make zero-cuda-gate` also prints `ZERO_CUDA FAIL` for its stub probe. That text stays in the log. The job is judged by its exit code and its last line, `CUDA++ HOST GATES PASS`, the same way `make test` is judged.

The checker itself has a negative control: `tools/ci/host-gates.sh --self-test`. A log that contains `ZERO_CUDA FAIL` and still exits 0 must be accepted. A doctor log that says `NOT READY` must be accepted. A doctor log that also says `READY. Next: make hello-gpu` must be rejected. A figure log that says `stale figures` must be rejected.

## Classes

- **Host-only.** `make -j4`, `make test`, `make test-native-artifact`, `make zero-cuda-gate`. No device node is opened.
- **Simulator-only.** `build/hello_gpu --sim`. A simulator pass is not hardware qualification and not a performance result.
- **Target-specific compile.** AArch64 Linux is the native host. Production ordering there stays `dsb sy` (`nvrm/cpu_barrier.h`). On x86-64 the portable suite is the one in `docs/OMEGA_HOST_TESTS.md`: `make test` with gcc, no NVIDIA tool and no sibling checkout. The x86-64 device open returns `ENOTSUP` and the text `requires AArch64 Linux`. That is `NOT SUPPORTED` for a GPU run. The receipt `evidence/HOST-X86-20261030/` is for its own commit. On that VM the default `CUDAPP_MEM_RESERVE_MIB` of 4096 made `make test` exit 2, and the suite passed only with `CUDAPP_MEM_RESERVE_MIB=0`. The default is unchanged. This workflow does not set the reserve and does not run the x86-64 suite.
- **GPU-only.** `make hello-gpu` and `./build/gpu_*_test` on a GB10. GitHub runners have no GB10 and no `/dev/aien-gpu`. Those commands are `NOT_RUN`. A missing GPU run is never recorded as a pass.

## Figures

`tools/figures/make_figures.py` is the source of the README pictures. `--check` compares its bytes with `docs/img/`. On `522ee0d` it printed `figures current: 10`, so `docs/img/stack.svg` already matched the generator. A later change that moves the generator must commit the regenerated SVG in the same change. A hand edit that makes the SVG disagree with the generator fails the check. The rendered legend test (`tools/figures/test_stack_legend.py`) needs Inkscape. The hosted figure job does not install Inkscape, so that test is `NOT_RUN` there. When Inkscape is on `PATH`, the same job runs it.

## Toolchain

`fresh-checkout` uses the `ubuntu-24.04-arm` image's `gcc`, `make` and `python3`. It installs no packages and no CUDA toolkit. `make -j4` is the compile budget. The job records `git rev-parse HEAD`, `uname`, the compiler line and the archive sha256 in the `host-gates-<sha>` artifact. The figure job uploads `host-gates-figures-<sha>`. No vendor binary, model weight or home-directory path is committed.

`make test` on the arm runner still installs the checksum-verified `nvdisasm` named in `ci.yml`. That oracle is verification-only. The host-gates job does not need it.

## What a green pull request does not prove

The active ruleset `main-verified-promotion` blocks deletion and non-fast-forward pushes. It does not list required status checks. The workflows start on every pull request. Marking a check required is an administrator change and is `NOT_RUN` here.

An outside engineer following `docs/START_HERE.md` on a machine this repository does not control is `NOT_RUN`.

`make hello-gpu`, every chip test, and any open of `/dev/nvidia*` or `/dev/aien-gpu` are `NOT_RUN` on these runners.
