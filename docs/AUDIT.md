# Licence and safety audit (before the first public push)

CUDA++ was split out of Omega (AGPL-3.0) and physics (AGPL-3.0) on 2026-10-21 to be published under MIT. Relicensing is only possible for lines whose authors agree, nothing secret may be in any version of any file, and no NVIDIA binary may be in the history. This file records what was checked, how, and the result. Every check below can be re-run with the scripts in `tools/extract/`.

Status: **CLEAN** for the history listed in section 1 (see section 6 for what this audit does not cover).

## 1. What was cut, from where

| Source | Commit | Paths | Commits kept |
| --- | --- | --- | --- |
| aien-spark/omega `main` | a8ffd2b912c84197d08f963b05f9c7d8b30a3f8a | 167 paths: 100 GPU source files, 9 shared-base files, 47 tests, 3 build fragments, 2 mutation scripts, 6 receipt folders (`tools/extract/out/`, `graph.sh`, `extra-paths.txt`) | 141 |
| aien-spark/physics `main` | 14273c5ce7d13fbc6e410c4fed9aa36837918177 | 17 paths: `nvrm/` + `m16/` sources, m16 tests and gates, one receipt, `third_party/nvidia-open-580.173.02/` (`physics-paths.txt`) | 18 |

- The file list comes from the include graph plus the header-to-source link rule (`tools/extract/graph.sh`): it starts from `src/omega_blackwell_*` and `src/omega_gpu_*` and needed no language file (src/compiler, src/checker2, tools/omegese).
- History was cut with git's own `fast-export` (path-limited) and `fast-import` (`tools/extract/extract.sh`). Every earlier name of every kept file was looked up with `git log --follow`; none of the kept files had one (they were created at their current paths).
- Tree check: at the two source commits, each of the 167 + 17 kept paths is byte-identical in the cut, and the cut holds nothing else. Old-to-new commit ids: `tools/extract/out/commit-map-{omega,physics}.tsv`.
- The runtime branches `runtime/r1-api`, `runtime/r2-hello`, `runtime/r3-portability` (each 1 to 2 commits on the same Omega commit) were carried over and merged.

## 2. Authors

Every author, committer and co-author line on CUDA++ `main` (`git log main --format='%an <%ae>'`, `%cn <%ce>`, `Co-authored-by` trailers):

| Who | Role | Commits |
| --- | --- | --- |
| Drake Stapleton <drake@aienos.com>, <aien@aienos.com> | author | 112 |
| aien-dev <drake@aienos.com> | Drake's GitHub organisation account (bots and merges) | 57 |
| aien-spark <aien.atlas@proton.me> | Drake's earlier organisation account | 6 |
| GitHub <noreply@github.com> | committer of merged pull requests | 65 |
| Claude (Fable 5.1, Opus 5.5, Sonnet 5.5), GPT-6 Astra / Codex, Grok 4.7, Gemini 3.8 | AI co-authors working for Drake | trailers only |

**No person other than Drake authored any line.** No outside contributor's pull request is in this history. Drake, as the sole author, relicenses these files under MIT. The AI co-authors are tools Drake ran; they hold no copyright.

## 3. Third-party material

- **NVIDIA open GPU kernel module headers** (`third_party/nvidia-open-580.173.02/`, 67 files, copied unchanged): every file carries NVIDIA's MIT notice (60 with `SPDX-License-Identifier: MIT`, 7 older files with the MIT permission text). NVIDIA's `COPYING` says the files are MIT and only a linked Linux kernel module is dual MIT/GPLv2. Fine for MIT; notices kept, listed in `THIRD_PARTY.md`.
- **Mesa (NVK and NAK)**: 20 comments in `src/` cite Mesa's open NVIDIA driver for hardware facts (opcodes, bit positions, a few latency values, QMD layout, dispatch and cache-invalidate order). The cited files were checked at gitlab.freedesktop.org/mesa/mesa on 2026-10-21: all `SPDX-License-Identifier: MIT` (Collabora, Red Hat). Fine for MIT; notice in `THIRD_PARTY.md`.
- Search over every kept file (not the NVIDIA headers) for `copyright`, `SPDX`, URLs and the names Mesa, NVK, NAK, nouveau, envytools, CUTLASS, cuobjdump, nvdisasm, maxas, turingas, CuAssembler, tinygrad, llama.cpp, ggml, FlashAttention, Triton, GPGPU-Sim, Linux kernel, GPL: no copyright header or licence text from anyone else, no URL, and no code from any of these. The only hits are the Mesa citations above and mentions of NVIDIA's `nvdisasm` and `cuobjdump` as cross-check tools.
- `tools/zero-cuda-gate.sh` is copied from aien-sovereign-core 14126c62, written by aien-dev (Drake).
- NVIDIA documentation: comments cite NVIDIA's public PTX ISA manual by section number (9.7.3.13, 9.7.3.21) and quote a few words of it (for example "flushes subnormal inputs and results to"), as references for chip behaviour. No table, listing or code is copied from NVIDIA documentation.
- No Linux kernel code, no GPL code and no AGPL code from anyone but Drake.

## 4. Secrets and private data, whole history

Scanned: **every blob reachable from any ref** (deleted files and old versions included) and **every commit and tag message** of both cuts, not just the final files.

| Check | Tool | Result |
| --- | --- | --- |
| Credentials (private keys, AWS, GitHub, GitLab, Slack, Anthropic, OpenAI, Google, xAI, npm, Tailscale/Headscale keys, JWTs, SSH public keys, `api_key=`-style assignments, Cortex token) | `tools/extract/audit-scan.sh` (rules `rules-secrets.txt`) | 0 hits in 624 + 151 blobs and all messages |
| Scanner works | `tools/extract/audit-selftest.sh`: plants 15 fake secrets, one in a deleted file's history and one in a commit message | PASS: 15 of 15 found |
| Second scanner | gitleaks 8.30.1 (linux arm64 release, sha256 e4a487ee7ccd7d3a7f7ec08657610aa3606637dab924210b3aee62570fb4b080 matched its published checksum), `gitleaks git --log-opts=--all` on clones of both cuts | no leaks found |
| Whole published history (`git clone --mirror` of this repository, every ref) | both scanners above | first run 2026-10-22 00:10Z: 10 hits (own) and 3 (gitleaks), all the fake keys `audit-selftest.sh` plants on purpose, written as plain strings in that script. Fixed: the script now builds them at run time from pieces, and the old versions were rewritten out of every commit (`git filter-branch --index-filter`, main only, from commit e1cc4f4 on; the Omega and physics cut commits are untouched, so `carry.sh` still matches). Re-run after the fix: see the line below |
| Same, after the fix | both scanners | own: 1018 blobs, 0 binaries, 0 secret hits; gitleaks: 162 commits, no leaks found (2026-10-22 01:34Z, GitHub mirror: main 72ec187 and docs/how-we-did-it 62ea5a2). Private-data hits are the same kinds as in the row below (plus `t@example.invalid`, the self-test identity, and `git@github.com` in clone commands) |
| IP addresses, host names, e-mail addresses, home paths | `rules-private.txt`, reviewed by hand | e-mail: only commit identities (drake@aienos.com, noreply addresses). IP-like: `9.7.3.13` / `9.7.3.21` are PTX ISA manual section numbers, `12.34.56.78` is an example in an NVIDIA header. Home paths: build folder names in receipt logs (`/home/drakestapleton/workspace/...`), no private content; kept |

Kept out of the history (removed from every commit, listed in `tools/extract/omega-exclude.txt` and `physics-exclude.txt`):

- six machine process listings (`evidence/R4-MATMUL-CTA-ffc3176a/*/ps-{before,after}.txt`): full command lines of other programs running on Drake's machine. Private, not needed by CUDA++. They stay in Omega with the rest of that receipt.
- `evidence/OWN30B-T3-TC-PEAK-BW-5faa0f70-20261013T2236Z/fa_ops.sass`: a listing of machine code produced by NVIDIA's compiler (see section 5). That receipt's `SHA256SUMS` still names it; the file is in Omega.
- `nvrm/lifecycle_gates`: a compiled program (74 KB ELF) built from `nvrm/lifecycle_gates.c`.

## 5. NVIDIA binaries

- Every blob in both histories was checked for binary content (any zero byte): **0 binary files** after the exclusions above.
- No firmware, no `libcuda`, no cubin, fatbin or PTX, no `.so`, `.o` or `.a`.
- No listing produced by NVIDIA tools (`nvdisasm`, `cuobjdump`) and nothing copied from NVIDIA's documentation. Some host tests call NVIDIA's `nvdisasm` at run time when it is installed, to cross-check our encoder; nothing it prints is stored.
- One CUDA C++ source remains in a receipt: `evidence/OWN30B-T3-TC-PEAK-BW-5faa0f70-20261013T2236Z/fa_ops.cu`, our own probe kernel, used once with NVIDIA's compiler to measure the tensor-core ceiling we compare against. It is our code, not NVIDIA's.

## 6. Limits of this audit

- It covers the CUDA++ `main` history named in section 1 plus the commits added in this repository. Branches carried in later (`tools/extract/carry.sh`) need `audit-scan.sh` and gitleaks again before they are pushed.
- "Only Drake authored it" rests on the commit identities and the organisation's membership (Drake and his agents). It does not prove that no line was ever pasted from elsewhere; section 3's search is the check for that, and it found only the MIT sources named there.
- The ten shared-base files (`tools/extract/shared.txt`: nine sources plus `tools/mutation_runner.sh`) are also in Omega under AGPL-3.0. Both copies are Drake's; CUDA++'s copies are MIT.

## Re-run

```bash
tools/extract/audit-selftest.sh
tools/extract/audit-scan.sh <repo> <out-dir>      # per repository or cut
```
