# CUDA++ licensing: AGPL-3.0-only candidate and MIT snapshot

**This branch is the authorized AGPL transition candidate.** At this ref, first-party CUDA++ material without a more specific notice is licensed under AGPL-3.0-only (`LICENSE`); first-party source SPDX identifiers are updated to match. This does not change canonical `main`: it remains at `232332422bfdc31c8918085b50f13c1e18ab702e`, whose `LICENSE` is MIT. The transition applies to the candidate tree only when and if the authorized publisher integrates it. This branch is not a claim that the default branch or any release has changed.

## License scope and retained grants

The root `LICENSE` is the AGPL-3.0-only license for first-party CUDA++ software and project material unless a file carries a different explicit license notice. The existing first-party MIT license is preserved verbatim as [`LICENSE-MIT`](../LICENSE-MIT) for users reproducing the historical MIT checkout below. That earlier MIT permission remains available for the earlier snapshot; this candidate does not retract it or purport to change the terms of copies already received under it.

Third-party material keeps its own terms and notices. In particular, the NVIDIA open-kernel-module headers remain under their per-file MIT notices in `third_party/nvidia-open-580.173.02/`; `THIRD_PARTY.md` records the source and scope. Mesa/NVK/NAK is cited for hardware facts and values, with MIT attribution; the audit found no Mesa source code copied. `docs/PAIN_POINTS.md` contains attributed quotations from its cited public sources. Those quotations remain third-party text, not newly authored AGPL text. The license transition does not change any third-party grant.

The rationale is straightforward: CUDA++ is first-party software built through independent implementation and hardware research. AGPL-3.0-only is the project's chosen reciprocal license for eligible first-party work, including its applicable source-sharing conditions. This is a licensing decision, not a conclusion about every possible use or every line's legal status.

## Provenance

Drake Stapleton describes the work as independently authored, with AI coding agents used as tools. The method includes original probe kernels, offline use of vendor compiler/disassembler tools on those probes, comparisons between one-variable probes, public technical references, and execution on the hardware. Historical reverse engineering is acknowledged and documented in [`docs/HOW_I_DID_IT.md`](HOW_I_DID_IT.md). These facts describe the process; they do not prove that every line in every revision is free of third-party influence.

The bounded current-history evidence, identity rows, source categories, and limits are in [`docs/LICENSE_PROVENANCE.md`](LICENSE_PROVENANCE.md). The earlier [`docs/AUDIT.md`](AUDIT.md) remains a historical record and is not rewritten as a current audit. The current audit found no outside human author in the candidate's commit metadata. It is not a legal opinion or a guarantee against infringement.

## Reproduce the historical MIT snapshot

The exact canonical snapshot verified for this transition is commit `232332422bfdc31c8918085b50f13c1e18ab702e`. It is a historical checkout, not a tag, fork, or assertion that it was the final MIT commit. The commands were exercised against the canonical repository and returned this commit with `LICENSE` beginning `MIT License`:

```sh
git clone --no-checkout https://github.com/aien-dev/CUDApp.git CUDApp-mit-snapshot
cd CUDApp-mit-snapshot
git fetch origin main
git checkout --detach 232332422bfdc31c8918085b50f13c1e18ab702e
git rev-parse HEAD
head -1 LICENSE
```

The documented commands identify that exact tree; they do not infer a license for other commits. Omega's separate planned Apache-2.0 extraction (L752) remains a distinct, pending effort and is not included in this transition.
