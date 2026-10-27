# CUDA++: why Apache-2.0 WITH LLVM-exception, and how to use the MIT version

**First-party CUDA++ material in this revision is offered under Apache-2.0 WITH LLVM-exception**, except where a file carries a different explicit notice. The license applies to this revision whether obtained from a branch or a release. Earlier MIT grants remain valid; this change does not revoke them.

## Why I built it, and why the license changed

I built CUDA++ to give people an independently implemented path to their NVIDIA GPU without depending on NVIDIA's proprietary user-space compiler, runtime, driver library or math libraries. The goal was to remove that software dependency, not to distribute NVIDIA's proprietary implementation under my name.

The foundation came from my own implementation and interoperability research. I wrote probe kernels, inspected their outputs using vendor tools offline, compared one-variable changes, consulted public technical references and permissively licensed projects, and tested my own encodings and runtime on hardware. I acknowledge that reverse engineering openly. The method and source references are documented in [HOW_I_DID_IT.md](HOW_I_DID_IT.md); the scoped provenance review is in [LICENSE_PROVENANCE.md](LICENSE_PROVENANCE.md).

Understanding hardware behavior and writing my own implementation is different from taking proprietary source code. The documented provenance identifies my implementation work separately from the third-party material I actually include. I do not claim NVIDIA's open-source headers, Mesa's contributions, NVIDIA's firmware, or anyone else's work as mine. The audit records evidence and its limits, not a universal legal guarantee.

The MIT foundation remains available to everyone at the exact revision below. Going forward, my first-party improvements build on that foundation under Apache-2.0 WITH LLVM-exception. The LLVM exception lets a developer compile proprietary models or GPU kernels with CUDA++ and ship the resulting object code under terms of their choosing; it also resolves the GPLv2 patent/indemnity conflict at the kernel boundary. It does not license NVIDIA's firmware, the vendor driver, or third-party notices.

This is not a claim that every NVIDIA component has been replaced: the documented working path still uses NVIDIA's open kernel modules and signed firmware. Omega is relicensed under the same terms in its own repository.

## What remains under its original terms

- NVIDIA's open-kernel-module interface headers retain their MIT notices in `third_party/nvidia-open-580.173.02/`.
- Mesa/NVK/NAK hardware facts and references retain their attribution in [THIRD_PARTY.md](../THIRD_PARTY.md). The historical audit reports no copied Mesa source code.
- Attributed quotations remain third-party text, not my newly authored work.
- 34 first-party files outside `evidence/` still carry `SPDX-License-Identifier: AGPL-3.0-only` and are not covered by the Apache-2.0 WITH LLVM-exception grant until their headers change: `tests/speculation` 10, `tests/learn53` 9, `native/speculation` 5, `native/learn53` 5, `native/blackwell` 3, `tests/device-trace` 1, `native/nvrm` 1 (all added by aien-spark on 2026-10-24 and 2026-10-25). List them with `grep -rl 'SPDX-License-Identifier: AGPL' --exclude-dir=evidence --exclude-dir=.git .`. Relicensing them is the author's decision; until then their AGPL-3.0-only terms apply.
- Receipts under `evidence/` keep the terms they were recorded with and are not edited. `evidence/LEARN53-L777-CONTINUATION/{author,review}/omega-core.om` carry no licence header.
- The original first-party MIT license can be read in [LICENSE at the pinned MIT revision](https://github.com/aien-dev/CUDApp/blob/232332422bfdc31c8918085b50f13c1e18ab702e/LICENSE). That prior grant remains effective for material covered by it; it is not a blanket MIT option for later Apache-2.0 WITH LLVM-exception contributions.

## Get the MIT foundation

Use this exact historical commit:

`232332422bfdc31c8918085b50f13c1e18ab702e`

These checkout commands were tested against the public repository:

```sh
git clone --no-checkout https://github.com/aien-dev/CUDApp.git CUDApp-mit-snapshot
cd CUDApp-mit-snapshot
git fetch origin main
git checkout --detach 232332422bfdc31c8918085b50f13c1e18ab702e
git rev-parse HEAD
head -1 LICENSE
```

The final commands show that exact commit and `MIT License`. To develop your own fork from this foundation, create a branch there:

```sh
git switch -c my-mit-development 232332422bfdc31c8918085b50f13c1e18ab702e
```

You can push that branch to a repository you control. GitHub's fork button alone does not select the historical version: explicitly check out the pinned commit. This is an existing MIT snapshot, not a claim that I created a separate hosted fork or that it was the final MIT commit.

Keep the MIT license and third-party notices when redistributing that version. Later Apache-2.0 WITH LLVM-exception contributions do not become MIT merely because you copy them onto this branch. The terms attached to the material you actually use continue to apply.
