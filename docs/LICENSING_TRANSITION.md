# CUDA++: licence history, and how to use the MIT or AGPL revision

**First-party CUDA++ material in this revision is offered under Apache-2.0** (`LICENSE`), except where a file carries a different explicit notice. The licence applies to this revision whether obtained from a branch or a release. Earlier grants are not revoked: the MIT grant covers the revisions published under MIT, and the AGPL-3.0-only grant covers the revisions published under AGPL. The terms attached to the material you actually use continue to apply.

## Licence history

| Revision | First-party licence | Notes |
| --- | --- | --- |
| Up to `232332422bfdc31c8918085b50f13c1e18ab702e` | MIT | The historical MIT foundation. |
| 2026-10-22 to the Apache-2.0 change | AGPL-3.0-only | Reciprocity, including the network-use condition. |
| This revision onward | Apache-2.0 | Ecosystem-wide permissive transition with an explicit patent grant. |

Each change applies from the revision that carries it. A downstream user who takes a revision gets the terms of that revision; a later relicensing does not withdraw rights already granted for an earlier revision.

## Why the licence changed

I built CUDA++ to give people an independently implemented path to their NVIDIA GPU without depending on NVIDIA's proprietary user-space compiler, runtime, driver library or math libraries. The goal was to remove that software dependency, not to distribute NVIDIA's proprietary implementation under my name.

The foundation came from my own implementation and interoperability research. I wrote probe kernels, inspected their outputs using vendor tools offline, compared one-variable changes, consulted public technical references and permissively licensed projects, and tested my own encodings and runtime on hardware. I acknowledge that reverse engineering openly. The method and source references are documented in [HOW_I_DID_IT.md](HOW_I_DID_IT.md); the scoped provenance review is in [LICENSE_PROVENANCE.md](LICENSE_PROVENANCE.md).

Understanding hardware behavior and writing my own implementation is different from taking proprietary source code. The documented provenance identifies my implementation work separately from the third-party material I actually include. I do not claim NVIDIA's open-source headers, Mesa's contributions, NVIDIA's firmware, or anyone else's work as mine. The audit records evidence and its limits, not a universal legal guarantee.

The ecosystem now standardizes on Apache-2.0 for first-party software: it is permissive, includes an express patent grant, and is accepted by corporate licence scanners and downstream redistributors without the AGPL's source-sharing conditions. This does not automatically license every program using a GPU, every service client, or generated output under Apache-2.0; the terms of `LICENSE` decide what a downstream user may do.

This is not a claim that every NVIDIA component has been replaced: the documented working path still uses NVIDIA's open kernel modules and signed firmware.

## What remains under its original terms

- NVIDIA's open-kernel-module interface headers retain their MIT notices in `third_party/nvidia-open-580.173.02/`.
- Mesa/NVK/NAK hardware facts and references retain their attribution in [THIRD_PARTY.md](../THIRD_PARTY.md). The historical audit reports no copied Mesa source code.
- Attributed quotations remain third-party text, not my newly authored work.
- The original first-party MIT grant remains valid for the revisions it covered; it is not a blanket MIT option for later revisions.
- The AGPL-3.0-only grant remains valid for the revisions published under it.

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

Keep the MIT licence and third-party notices when redistributing that version. Later Apache-2.0 or AGPL contributions do not become MIT merely because you copy them onto that branch.
