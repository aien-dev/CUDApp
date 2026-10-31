# Gate 1 — MAX pin and file-level licence inventory

Written 2026-10-30. Status: EXERCISED (host inventory run). This is not a legal opinion.

## Pins

| item | value |
|---|---|
| release pin | `max/v26.7.0` = modular/modular `527d069ee11a41ac5320af1f9402db0a66b2b868` (committed 2026-10-22T17:37:22Z). This is the newest `max/v*` tag (`git ls-remote --tags`, 2026-10-30). |
| experimental reference | modular/modular `main` = `8ed65fc6ac15950ce62da1f8e8d50c2c7c528065` (2026-10-29T19:28:33Z). It is in no tagged release. |
| wheels | max, max-core, max-mojo-libs, modular 26.7.0; mojo, mojo-compiler, mojo-compiler-mojo-libs, mojo-lldb-libs 1.2.0. Installed in venv `~/workspace/maxinterop/venv`. |
| pin licence | `Licenses/LICENSE` "Modular MAX Community License", last modified August 18, 2026, sha256 `f35ac2dfeb4a3e9482468874a50ce4dfb16fc11c3aefd2a1b4d51c44515ed02d` (blob bytes at the pin) |
| main licence | `Licenses/MAX_LICENSE` sha256 `5ee0ef8838bb7eb212daad69e88f0230f6af822c27d25570ef1eb331d3742ac4` |

## Method

`license_inventory.py REPO REV OUT` (in this directory) reads every blob of `REV` from git objects, not from the working tree. For each file it records the SPDX tag, the licence header, or "no header"; for files without a header it also records the nearest governing `LICENSE` file. `wheel_inventory.py` does the same for every file in the installed wheels' RECORD.

`~/workspace/maxinterop/max-source` is a `blob:none` partial clone, so every missing blob is fetched lazily one at a time. The first run against it (bg_21) timed out after 900 s. The counts below come from a shallow bare clone, `/dev/shm/modular-v26.7.0.git` (depth 1, tag `max/v26.7.0`, plus `main` at depth 1). That run took 1.5 s.

Raw files (all rows, one file per row; gzip -n, `zcat` to read):
- `max-source-527d069e.tsv.gz`: release pin, 11906 files
- `max-source-main-8ed65fc6.tsv.gz`: main, 14866 files
- `wheels-max-26.7.0.tsv.gz`: installed wheels

## Released set (pin 527d069e, source)

| count | kind | licence | governing file |
|---:|---|---|---|
| 9388 | header | Apache-2.0 WITH LLVM-exception | – |
| 2168 | no header | → root `LICENSE` (Apache-2.0 WITH LLVM-exception) | `LICENSE` |
| 297 | binary | → root `LICENSE` | `LICENSE` |
| 25 | no header | MIT | `Mojo/tools/mblack/LICENSE` |
| 11 | SPDX | Apache-2.0 WITH LLVM-exception | – |
| 4 | no header | → `mblib2to3` LICENSE | `Mojo/tools/mblack/src/mblib2to3/LICENSE` |
| 4 | SPDX | Apache-2.0 | – |
| 2 | no header | MAX licence directory | `Licenses/LICENSE` |
| 2+2 | header/SPDX | MIT | – |
| 1 | SPDX | BSD-2-Clause (`AsyncRT/include/AsyncRT/Support/ConcurrentMPMCQueue.h`) | – |
| 1 | header | MAX License (the licence text itself) | – |
| 1 | header | Apache-2.0 | – |

At the pin, no source file outside `Licenses/` carries a MAX License header.

Pin files without a header, by extension: bazel 675, md 413, mlir 257, json 171, mdx 165, rst 149, bzl 79, patch 36, toml 32, txt 27, lock 21, test 20, other.

## Released set (wheels 26.7.0)

Every distribution's metadata declares `LicenseRef-MAX-Platform-Software-License`. Only `max` ships `LICENSE` and `Third-Party-Notices`. File headers inside the wheels:

| dist | Apache-2.0 WITH LLVM-exception | Apache-2.0 | MIT | no header |
|---|---:|---:|---:|---:|
| max | 1958 | 3 | 1 | 1876 (1852 .pyc, 3 .so) |
| max-core | 9 | – | – | 10 (6 .so) |
| max-mojo-libs | – | – | – | 36 (32 .mojoc) |
| mojo-compiler | 6 | – | – | 20 |

Shared objects and `.mojoc` files without a header fall under the MAX Platform licence of their wheel. The wheel's Mojo sources keep their own Apache-2.0 WITH LLVM-exception headers (MAX licence §1.4).

## Experimental set (main 8ed65fc6, not released)

- 3578 paths exist only on main and 618 only at the pin.
- Main-only paths by top-level directory: GraphCompiler 1429, max 1292, Mojo 384, MLRT 283, Target 112, bazel 35, Support 15, SDK 15, Kernels 10, Licenses 2, AsyncRT 1.
- On main, 988 files carry a **MAX License** header. Their top-level directories:

| files | path |
|---:|---|
| 206 | GraphCompiler/lib |
| 185 | GraphCompiler/include |
| 98 | Target/lib (includes 75 files in `Target/lib/CUDA/`) |
| 85 | max/mojo |
| 78 | MLRT/lib |
| 61 | max/python |
| 61 | MLRT/include |
| 53 | GraphCompiler/test |
| 51 | MLRT/unittests |
| others | rest |

- Main-only paths matching HAL or driver: 282 have a MAX License header, 23 have no header, 7 are Apache-2.0 WITH LLVM-exception, 2 are binary. This includes the whole Driver SDK, `SDK/include/HAL/M_driver_*.h` (13 headers, all with a MAX License header). Its README calls it the ABI-safe C interface for hardware vendors' driver plugins, "developed against (SDK versions corresponding to) one of our (stable) releases". No stable release ships it yet.

## Files on our execution path

| component | pin | main |
|---|---|---|
| `max/mojo/max/gpu/host/device_context.mojo` (`DeviceExternalFunction`, `load_function`) | Apache-2.0 WITH LLVM-exception | same; plus `_legacy/device_context.mojo`, same licence |
| `max/python/max/graph/ops/custom.py` (`ops.custom`) | Apache-2.0 WITH LLVM-exception | same |
| `max/tests/integration/Inputs/kernel_verification_ops/__init__.mojo` (`op_with_external_cubin` pattern) | Apache-2.0 WITH LLVM-exception | same |
| `AsyncRT_DeviceContext_loadFunction` implementation | no source at the pin; binary only: `libAsyncRTMojoBindings.so` sha256 `ebefd2e8…830c` in the max-core wheel, MAX Platform licence | `MLRT/lib/Driver/DeviceContext/MojoBindings.cpp`, MAX License header |
| CUDA backend | binary only at the pin | `Target/lib/CUDA/*` (75 files), MAX License header |

Gates 2 and 3 call the released Apache-licensed Mojo/Python API and the released MAX-licensed binary. They do not modify or redistribute MAX.

## AI-use clause that applies to this work

- Pin: `Licenses/LICENSE` §1.3(d), lines 81–86.
- Main: `Licenses/MAX_LICENSE` §1.4(d), lines 65–71. Main adds a new §1.3 "Contributions", so every later section number moves up by one: Additional Requirements is §1.4, Separately Licensed Components §1.5, Patent License §1.6. The clause text is the same at both revisions, except that the pin contains a stray "is prohibited,".

Clause text: "You shall not use MAX, or any portion of it, as training data, fine-tuning data, or input to any AI system in order to produce software that reimplements or substitutes for MAX … This does not restrict using AI tools to read, analyze, improve, or explain MAX, or to develop Your own software that runs on or interoperates with MAX."

Consequences for this work stream:

| item | status |
|---|---|
| Gates 2–4: an Omega kernel loaded and run by MAX | interoperation; allowed |
| Gate 5: an Omega-backed device plugin written against the MAX-licensed HAL Driver SDK | interoperation by its stated purpose; experimental, main only |
| Using MAX-licensed GraphCompiler, MLRT or Target sources as AI input to build an Omega graph compiler, runtime or scheduler that replaces MAX's | forbidden; keep MAX's model, graph and serving layers |

Section headings checked against both blobs: pin lines 65/88 are §1.3/§1.4; main lines 45/50/73 are §1.3/§1.4/§1.5.

Upstream contributions also fall under modular `AI_TOOL_POLICY.md`: a human reviews and submits, and only bug fixes are accepted.
