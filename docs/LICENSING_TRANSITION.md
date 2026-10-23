# CUDA++ licensing transition: MIT snapshot and planned AGPL

**Status: transition planned, not yet effective.** This document records the exact MIT-licensed snapshot and explains the rationale for a possible future AGPL-3.0 release. It does not change the repository's `LICENSE`, make a future license effective, establish that every historical line is eligible for relicensing, or give legal advice. The complete authorship and third-party audit required for a license change is still pending.

## An exact MIT-licensed checkout

The canonical `aien-dev/CUDApp` `main` fetched for this documentation work was at commit `232332422bfdc31c8918085b50f13c1e18ab702e`. Its tracked tree includes the `LICENSE` file headed `MIT License`. This SHA identifies a historical snapshot, not a fork, tag, promise that it is the final MIT commit, or claim that later commits have the same license.

The following commands were exercised from a fresh directory against the canonical repository; the detached checkout resolved to the SHA above and `LICENSE` began with `MIT License`:

```sh
git clone --no-checkout https://github.com/aien-dev/CUDApp.git CUDApp-mit-snapshot
cd CUDApp-mit-snapshot
git fetch origin main
git checkout --detach 232332422bfdc31c8918085b50f13c1e18ab702e
git rev-parse HEAD
head -1 LICENSE
```

To start an independent branch from the same snapshot:

```sh
git switch -c my-cudapp-work 232332422bfdc31c8918085b50f13c1e18ab702e
```

To work from your own GitHub fork, create the fork using GitHub, clone that fork, add the canonical repository as `upstream` if it is not already configured, fetch it, then create a branch at the pinned commit:

```sh
git clone https://github.com/YOUR-ACCOUNT/CUDApp.git
cd CUDApp
git remote add upstream https://github.com/aien-dev/CUDApp.git
git fetch upstream main
git switch -c my-cudapp-work 232332422bfdc31c8918085b50f13c1e18ab702e
```

GitHub's fork button follows the repository as it exists when the fork is made; the branch command above is what selects the exact historical commit. Do not infer a license for a later commit from this snapshot. If a future release changes CUDA++ to AGPL-3.0, its new commits and files must be evaluated under that license and their actual notices. Existing MIT permissions for the MIT snapshot were already granted under its `LICENSE`; a later license change does not retroactively erase permissions already granted for that snapshot. Preserve the MIT license and third-party notices when redistributing that version. This is a description of the stated grants, not a legal opinion about any particular use.

## Why a future AGPL-3.0 release is being considered

The project is intended to replace dependence on NVIDIA's proprietary user-space CUDA stack with an independently implemented user-space runtime and compiler path, using interoperability research to communicate with the GPU and the open kernel-module interface. Drake describes his implementation work as independently authored, with AI coding agents used as tools. The stated method is to write probe kernels, inspect the outputs of those probes using vendor tools offline, compare one-variable changes, consult public documentation and permissively licensed open-source implementations, and validate behavior on the hardware. That is a description of the research approach, not proof about every line in every revision and not a blanket legal conclusion.

In particular, the repository does retain 67 NVIDIA open-kernel-module interface headers under their MIT notices. `THIRD_PARTY.md` also attributes hardware facts to Mesa's MIT-licensed NVK/NAK sources; the audit says no Mesa source code was copied, while identifying values and facts taken from those sources. `docs/HOW_I_DID_IT.md` describes the use of the project's own probe kernels and their observed outputs. These source categories and attributions must remain visible. Do not summarize the repository as “all code ours,” as containing no third-party material, or as having no legal questions.

The `docs/AUDIT.md` file is a useful historical record of the extraction and scans it reports, including an author table, the NVIDIA-header and Mesa disclosures, and stated limitations. Its scope is bounded by the commits, refs, tools, and dates named in that file. It is not an independent legal review and does not prove provenance for every line or every commit subsequently added to the current tree. In particular, the audit itself says carried branches require rescanning before push. A complete history/authorship and third-party review is still required before any relicensing action. No absolute legal assurance is made here.

The intended AGPL rationale is reciprocity: for eligible CUDA++ code, preserve users' ability to obtain source and share modifications when the license's applicable conditions require it, including relevant network-use provisions. That does **not** mean AGPL automatically covers every remote GPU user, all output produced by the software, separately licensed components, or uses that do not trigger the license's conditions. The exact scope depends on the code, notices, and circumstances; this page is not a legal determination.

Omega's planned standalone Apache-2.0 separation is a separate effort (L752). It is intended, not complete, and does not by itself change CUDA++'s license. The two projects' boundaries and notices must be established by their own audits and releases.

## Before any license change

- Complete and record the full history, author/co-author, and third-party review for the exact tree to be relicensed. Confirm each relevant rights holder's permission; do not treat a prior audit's statement as a substitute for current evidence.
- Preserve applicable third-party materials under their existing terms and keep their notices, including the NVIDIA MIT headers and Mesa attribution.
- Make any license change explicit in the repository's license files and commit history only after that review. Until then, the repository `LICENSE` remains the operative license for its current tree; no AGPL transition is asserted by this document.
- Keep historical MIT checkout instructions and receipts available so users can identify and reproduce the exact snapshot they choose.
