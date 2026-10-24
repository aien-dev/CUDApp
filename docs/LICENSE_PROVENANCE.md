# CUDA++ license and provenance audit

**Result: eligible first-party transition candidate, with bounded audit scope.** This record covers the exact candidate base `b0d9c533636c7bc93539e83a3668aa3651713e74`, the authorized licensing changes, and the complete reachable history. The earlier `docs/AUDIT.md` is preserved unchanged as the historical extraction audit. This is evidence for the candidate, not a legal opinion or a universal statement about every line.

## Commit and identity evidence

The transition candidate at `d478c56856dacedc2b93c1d08ec70ef8c270e436` contains 189 commits. `git log` across that exact history returned these author identities:

| Commit author identity | Commits | Attribution used for this candidate |
| --- | ---: | --- |
| Drake Stapleton `<drake@aienos.com>` | 118 | Drake Stapleton |
| Drake Stapleton `<aien@aienos.com>` | 8 | Drake Stapleton |
| `aien-dev <drake@aienos.com>` | 57 | Drake's organization account, as recorded in the historical audit |
| `aien-spark <aien.atlas@proton.me>` | 6 | Drake's earlier organization account, as recorded in the historical audit |

Committer identities were Drake Stapleton `<drake@aienos.com>` (107), Drake Stapleton `<aien@aienos.com>` (12), and GitHub `<noreply@github.com>` (70). GitHub's identity appears as the committer on merged commits; it is not an additional source author.

Co-author trailers name Drake Stapleton and AI coding tools: Claude Opus 5.5, Claude Fable 5.1, Claude Sonnet 5.5, GPT-6 Astra / Codex, Gemini 3.8, and Grok 4.7. No other human name appears in the reachable author, committer, or co-author metadata. The candidate relies on Drake's explicit authorization for his own eligible work. Git metadata records identities and attribution; by itself it cannot prove who controlled an account or establish copyright ownership. The organization-account attribution is consistent with the historical audit and the user authorization, not an independent identity investigation.

## Third-party material and notices

- **NVIDIA open GPU kernel module.** The repository retains NVIDIA's public interface headers with their own NVIDIA MIT notices and `COPYING`; those files are excluded from first-party relicensing. `THIRD_PARTY.md` records the distinction between the user-space interface definitions and NVIDIA's separately licensed kernel-module combination. Vendor notices and files are preserved unchanged.
- **Mesa NVK/NAK.** `THIRD_PARTY.md` attributes hardware facts and values cited in comments to MIT-licensed Mesa sources (Collabora and Red Hat). The historical audit says no Mesa source code was copied. This is a statement about the reviewed references, not a claim that the project contains no third-party material.
- **Public research quotations.** `docs/PAIN_POINTS.md` contains attributed quotations and citations to public project, forum, and documentation pages. Those passages remain third-party text; the repository's license notice does not relicense them.
- **Probe and vendor tools.** The documented reverse-engineering process uses Drake's own probe kernels, vendor tools to inspect those probes, public references, and hardware observations. No NVIDIA compiler output, disassembler listing, firmware, CUDA library, cubin, fatbin, or PTX is asserted to be included based solely on the historical audit. The scan below found no binary blobs in the candidate-base history; content provenance still has the limits stated here.
- **Separate Omega work.** The standalone Omega Apache-2.0 extraction (L752) is pending and outside this audit and license change.

## Reproducible checks and limits

On an isolated clone at the transition candidate commit `d478c56856dacedc2b93c1d08ec70ef8c270e436`, `tools/extract/audit-selftest.sh` passed: it planted 15 test credentials (including a deleted-history blob and commit message) and the scanner found all 15. `tools/extract/audit-scan.sh` scanned all objects and messages reachable from that candidate history: 1,325 blobs, zero binary blobs, zero credential-pattern hits, and 505 private-data-pattern matches in 16 distinct categories. The private matches include documented commit identities, build-home paths, PTX manual section numbers, example addresses, self-test values, and public clone/URL text; the summary was inspected. These results were independently reproduced on that exact commit during review L759. Subsequent documentation corrections do not expand the scope of these historical scan results.

A source-and-notice search of the candidate tree excluding NVIDIA's vendor directory confirmed first-party source files carried MIT SPDX tags before transition, and found MIT notices in the retained third-party material and historical audit/evidence. In the 2026-10-22 transition candidate, first-party SPDX tags were changed to AGPL-3.0-only; `docs/AUDIT.md`, the captured historical MIT receipt, and vendor files were intentionally left unchanged. In the current revision, those first-party SPDX tags are Apache-2.0. The provenance findings and third-party exclusions above apply unchanged. Review of tracked `THIRD_PARTY.md`, `docs/HOW_I_DID_IT.md`, `docs/DRIVER.md`, and source citations preserves the NVIDIA and Mesa disclosures.

This review does not prove that no line was ever derived from another source, authenticate the identity behind every historical account, decide the copyright status of public quotations, or provide legal advice. It checks the candidate history's commit metadata, explicit license markers/notices, known provenance disclosures, and reproducible secret/binary scan. New commits after this audited base need the same author and third-party review before adding first-party material to the licensed line.
