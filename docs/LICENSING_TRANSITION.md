# CUDA++: why AGPL, and how to use the MIT version

**First-party CUDA++ material in this revision is offered under AGPL-3.0-only**, except where a file carries a different explicit notice. The license applies to this revision whether obtained from a branch or a release. Earlier MIT grants remain valid; this change does not revoke them.

## Why I built it, and why the license changed

I built CUDA++ to give people an independently implemented path to their NVIDIA GPU without depending on NVIDIA's proprietary user-space compiler, runtime, driver library or math libraries. The goal was to remove that software dependency, not to distribute NVIDIA's proprietary implementation under my name.

The foundation came from my own implementation and interoperability research. I wrote probe kernels, inspected their outputs using vendor tools offline, compared one-variable changes, consulted public technical references and permissively licensed projects, and tested my own encodings and runtime on hardware. I acknowledge that reverse engineering openly. The method and source references are documented in [HOW_I_DID_IT.md](HOW_I_DID_IT.md); the scoped provenance review is in [LICENSE_PROVENANCE.md](LICENSE_PROVENANCE.md).

Understanding hardware behavior and writing my own implementation is different from taking proprietary source code. The documented provenance identifies my implementation work separately from the third-party material I actually include. I do not claim NVIDIA's open-source headers, Mesa's contributions, NVIDIA's firmware, or anyone else's work as mine. The audit records evidence and its limits, not a universal legal guarantee.

The MIT foundation remains available to everyone at the exact revision below. Going forward, my first-party improvements build on that foundation under AGPL-3.0-only. I chose reciprocity so that improvements to covered CUDA++ software remain available to its users under the license's source-sharing conditions, including applicable network-use provisions. This does not automatically license every program using a GPU, every service client, or generated output under AGPL.

This is not a claim that every NVIDIA component has been replaced: the documented working path still uses NVIDIA's open kernel modules and signed firmware. Omega's separate standalone Apache-2.0 language/toolchain extraction is a distinct, pending effort.

## What remains under its original terms

- NVIDIA's open-kernel-module interface headers retain their MIT notices in `third_party/nvidia-open-580.173.02/`.
- Mesa/NVK/NAK hardware facts and references retain their attribution in [THIRD_PARTY.md](../THIRD_PARTY.md). The historical audit reports no copied Mesa source code.
- Attributed quotations remain third-party text, not my newly authored work.
- The original first-party MIT license is preserved verbatim in [LICENSE-MIT](../LICENSE-MIT). It documents prior grants, not a blanket MIT option for future AGPL contributions.

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

Keep the MIT license and third-party notices when redistributing that version. Later AGPL-only contributions do not become MIT merely because you copy them onto this branch. The terms attached to the material you actually use continue to apply.
