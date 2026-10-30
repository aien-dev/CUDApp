# Cubin writers (CUDApp #108)

Source comparison, 2026-10-30. No device was opened. No cubin was packed. Nothing here was run on GB10 or any other GPU. Device results quoted from comments or from omega-mv#864 are marked hardware-unverified.

This note records the duplication and the owner. It does not move, merge, or rewrite either writer. ARCH-0042 decision 7 (Drake, 2026-10-27, accepted 2026-10-30) keeps existing copies where they are. Consolidating them waits for a decision by Drake, recorded on [#108](https://github.com/aien-dev/CUDApp/issues/108). New work on the cubin container writer goes to this repository, or to a new aien-dev repository that pins this one. It does not go into the omega-mv copy.

[#106](https://github.com/aien-dev/CUDApp/issues/106) and [#107](https://github.com/aien-dev/CUDApp/issues/107) are other copies (GPU C sources, driver client). They are not resolved here.

## Owner

[ARCH-0042](https://github.com/aien-dev/aien-architecture-mv/blob/main/docs/adr/0042-omega-cudapp-bridge-ownership.md) is Accepted (Main, delegated by Drake, 2026-10-30, merged as aien-architecture-mv#342). The consequences table names one owner for the cubin writer: CUDApp. Omega owns computation. Omega Bridge owns compatibility and does not write cubins.

The container writer is the ELF a loader would read: sections, `.nv.info` attributes, the register count, the ABI version. That is GPU execution, so it is this repository's.

## What was read

CUDApp files below were read from main `d79519e0ecbb734781bdfe652de6524f45f6cfc2` before this note.

omega-mv `seed/sm80/cubin.om` is not on omega-mv's default branch. `gh api repos/aien-dev/omega-mv/contents/seed/sm80/cubin.om` returned 404. The same path on `ck/cubin-om` returned blob `b659fafe847a99e6ff3e6b3c7bc9b0840d7a783a`, 14129 bytes, 786 lines (`wc -l` after a base64 decode). That blob is commit `76639738acdd67030ae1ceb22ef957d264d836ef`. [omega-mv#864](https://github.com/aien-dev/omega-mv/pull/864) is a closed draft: `mergedAt` is null, `closedAt` is 2026-10-27T11:48:36Z, head is that commit. The close comment says the branch is kept and that the file is not to be merged, labelled, or built on. The reason given there is a licence question about earlier extracts, with no counsel answer on the PR. This note cites the file. It copies no section payload, note body, or instruction mask out of it.

The issue's 2026-10-26 table says the omega-mv writer targets sm_80, sm_86 and sm_89 and is a draft with a hardware retest pending. The file at `76639738` accepts more targets, and the pull request is closed rather than open. Those are the two states. The issue body was not edited to match the later file.

`evidence/CUDAPP-OM-ENCODER/README.md` still says #864 is on HOLD. That receipt is left as it was written. The public pull request was closed later the same day.

## The CUDApp writer is one writer with two bodies

Both bodies target sm_121 only and ABI version 8.

| Body | Path | When it is compiled |
|---|---|---|
| C | `src/omega_cubin.c` `c_cubin_pack`, called as `omega_cubin_pack` | Freestanding (`OMEGA_GPU_NO_POSIX`) or `CUDAPP_MIGRATION_PENDING_C_CUBIN` (`src/omega_cubin.c:41-42`, `386-395`) |
| `.om` | `om/cubin/cubin_elf.om` `cubin_elf`, called through `omega_cubin_elf` | The hosted build (`src/omega_cubin.c:6-9`, `510-518`; `mk/om-cubin.mk:1-4`) |

`om/cubin/cubin_elf.om:1-3` says it replaces the body of `omega_cubin_pack`. `evidence/CUDAPP-OM-ENCODER/README.md` says that unit is a translation of `src/omega_cubin.c` only, and that omega-mv#864 was not read for it. `make test-om-cubin` (`mk/om-cubin.mk:36-47`) packs each call both ways and fails if the code, bytes, or message differ. That target was not re-run for this note. The receipt's host counts stay in the receipt.

Constants the two bodies share, from `src/omega_cubin.h:46-52`: ABI version 8, `e_flags` `0x6007902`, SM 121, parameter base `0x380`, at most 32 parameters, names of at most 64 bytes. `om/cubin/cubin_elf.om:24` spells the parameter base as decimal 896. `evidence/CUDAPP-OM-CUBIN/wolfram-constants.txt` (Wolfram, 2026-10-27) maps the hex literal `0x380` to 896. No new numeric conversion was done for this note. Wolfram is not mounted in this session.

## The other writer

`seed/sm80/cubin.om` at `76639738` is a program, not a library function. `main` (`cubin.om:772-785`) reads arguments, the instruction bytes on stdin, checks registers, and writes a cubin to stdout. Its note (`cubin.om:17-23`) accepts SM 80, 86, 87, 89, 90, 100, 103, or 120. The check is `cubin.om:165-178`. SM 121 is not in that list. ABI 7 and ABI 8 are accepted (`cubin.om:180-182`). ABI 7 with SM >= 100 is refused (`cubin.om:183-185`).

The file's own note (`cubin.om:5-6`) says the layout was read from the author's nvcc 13.0 probes and from CUDApp `src/omega_cubin.c`. That is the file's claim about its sources. It is not a claim this session re-checked against those probes.

## Differences checked

Each row is a difference read off the files. Absence from this table is not a claim that the bytes match.

| | CUDApp (`omega_cubin.c` / `cubin_elf.om`) | omega-mv `cubin.om` at `76639738` |
|---|---|---|
| Targets | SM 121 only (`omega_cubin.h:48`, note word at `omega_cubin.c:237`) | 80, 86, 87, 89, 90, 100, 103, 120, else exit 2 (`cubin.om:165-178`). 121 is refused |
| ABI | Version 8 only (`omega_cubin.h:46`, `omega_cubin.c:335`). OS/ABI byte `0x41` (`omega_cubin.c:334`) | 7 or 8 (`cubin.om:689-701`). ABI 8 writes decimal 65 at byte 7 and 8 at byte 8. ABI 7 writes decimal 51 and 7. Whether 65 is the same byte as `0x41` was not re-evaluated |
| `e_flags` | The constant `0x6007902` (`omega_cubin.h:47`, stored at `omega_cubin.c:342`) | Computed from SM (`cubin.om:694-697`). `args` rejects SM 121 before `ehdr` runs, so this file never stores an sm_121 flags word. The formula was not evaluated against `0x6007902` |
| Register count stored | The caller's `gpr_count`, unchanged, in `.nv.info` attribute `0x2f` (`omega_cubin.c:222`) and attribute `0x1b` (`omega_cubin.c:254`). `cubin_elf.om:310` and `:367` store the same field | Input REGS is checked, then `VREG` is replaced with REGS+2 (`cubin.om:317`) and that sum is what `ginfo` stores (`cubin.om:386`). Attribute 27 (decimal) is the constant 255 on both the pre-sm_90 path (`cubin.om:539`) and the sm_90+ path (`cubin.om:545`), not `VREG` |
| Register refusal | `gpr_count > 255` (`omega_cubin.c:170`, `cubin_elf.om:164`). 0 is accepted. The text is not scanned | REGS outside 1..253 is exit 2 (`cubin.om:189`). `regchk` (`cubin.om:295-316`) exits 6 when REGS is below a register number taken from byte 2 of an instruction, with the exceptions named in the notes at `cubin.om:36-39` and `:291-294`. Then it adds 2 |
| Exit numbers | 0 ok, 1 bad argument, 2 text, 3 parameters, 4 constant bank, 5 GPR above 255, 6 shared size not a multiple of 4, 7 allocation (`omega_cubin.h:55-63`) | 0 written, 1 usage, 2 bad number, 3 bad name or signature, 4 bad input, 5 write failed, 6 register coverage (`cubin.om:35-37`). Exit 6 does not mean the same thing |
| `.note.nv.tkinfo` | Not in the section list (`omega_cubin.c:263-264`). `cubin_elf.om` writes `.note.nv.cuinfo` (`:375-385`) and has no tkinfo writer | Appended only for ABI 8 (`cubin.om:218-221`). Writer at `cubin.om:429` |
| `.nv.callgraph` | Not in the section list | Section writer `cgraph` at `cubin.om:394` |
| Other sections | Always: `.shstrtab`, `.strtab`, `.symtab`, `.text.<fn>`, `.nv.constant0.<fn>`, `.nv.info`, `.nv.compat`, `.nv.info.<fn>`, `.nv.shared.reserved.0`, `.note.nv.cuinfo` (`omega_cubin.c:263-264`) | Name table at `cubin.om:41`. `.nv.compat` and `.nv.shared.reserved.0` are added from sm_90 (`cubin.om:208-212`). Order is not the CUDApp order |
| `.text` `sh_info` | 2 (`omega_cubin.c:300`, `cubin_elf.om:422`) | 3 from sm_90, otherwise a word that includes `VREG` shifted (`cubin.om:640-644`) |
| `.note.nv.cuinfo` link | `sh_link` 0, `sh_info` 2, flags literal `0x1000040` (`omega_cubin.c:306`) | `sh_link` is `VTK`: 0 on ABI 7, and the tkinfo section index on ABI 8 (`cubin.om:217-221`, `:638`). `sh_info` is 0 below sm_90 and 10 from sm_90 (`cubin.om:634-637`). Flags literal 16777216 below sm_90 and 16777280 from sm_90 (`cubin.om:630-632`). The 2026-10-27 Wolfram receipt maps `0x1000040` to 16777280. The 16777216 literal was not given a hex spelling in that receipt |
| EXIT offsets | `EIATTR_EXIT_INSTR_OFFSETS` is omitted (`omega_cubin.c:257-258`, `omega_cubin.h:34-36`) | Input with no EXIT is exit 4 (`cubin.om:503-505`). Offsets are written (`cubin.om:506-513`) |
| Parameter base | `0x380` for the only target (`omega_cubin.h:49`) | Decimal 352 below sm_90, 528 from sm_90 and below sm_100, 896 from sm_100 (`cubin.om:196-202`). 896 is the receipt's reading of `0x380`. cubin.om never selects it for sm_121, because sm_121 is refused |
| Parameters | Caller supplies offset, size 1, 2, 4, or 8, and a pointer bit. At most 32 (`omega_cubin.h:51`, `omega_cubin.c:186`). Records are written in caller order (`omega_cubin.c:123`) | Signature letters `p` (8) and `u` (4) only, 1 to 16 (`cubin.om:32`, `:252-266`). `kparams` writes them last to first (`cubin.om:488`). The KPARAM type bytes (`cubin.om:475-480`) were not byte-compared with CUDApp's type words `0x0021f500` and `0x0011f000` (`omega_cubin.c:128`) |
| Name | 1 to 64 bytes, first character a letter or `_` (`omega_cubin.c:86-96`) | 1 to 32 bytes (`cubin.om:31`, buffer at `:49`). A leading digit is accepted (`cubin.om:233`) |
| Text | Nonzero multiple of 16, at most `1 << 24` bytes (`omega_cubin.c:162`) | Multiple of 16, below 65536 bytes (`cubin.om:33`, buffer at `:47`). The same note requires one EXIT (`cubin.om:34`) |
| Shared | Caller `shared_reserved`, multiple of 4, section always present (`omega_cubin.c:174`, `:305`) | From sm_100 the reservation literal is 64, otherwise 0 (`cubin.om:205-215`). The section exists from sm_90 |
| Program headers | Five headers are written (`omega_cubin.c:366-375`, `cubin_elf.om:467-479`) | Written (`cubin.om:746-758`). Count is 4 below sm_90 and 5 from sm_90 (`cubin.om:206-211`) |

`tools/artifact_export.c` passes `kern.gpr_count` into the cubin (`:102`, `:136`, `:175`). That field is the allocator peak (`src/omega_blackwell_codegen.c:1405`). The same tool sets `device-load=NOT_RUN` (`tools/artifact_export.c:183`). `omega_native_artifact.c:300-301` forwards that same `gpr_count` and a shared size of 64. Neither call adds 2.

## A register reserve that is not the cubin writer

`omega_bw_gpr_declared` (`src/omega_blackwell_codegen.c:2712`) returns `((peak + 2u + 7u) / 8u) * 8u`. The comment at `src/omega_blackwell_codegen.h:293-298` says this is the count to declare in the QMD: peak + 2, rounded up to a multiple of 8. Launchers call it. The cubin packer does not. The header cites chip runs from 2026-10-13 and says the exact minimum is UNKNOWN and the rule is INFERRED. Those runs were not repeated here: hardware-unverified.

That helper is one launch declaration. It is not the rule `cubin.om` implements. `cubin.om` adds 2 to the count it stores in `.nv.info` and does not round to 8 (`cubin.om:317`). The two `+ 2` operations are different code, on different outputs.

`src/omega_blackwell_submit.c:265` puts a third expression into a QMD config: `(kernel->gpr_count > 16) ? ((kernel->gpr_count + 15) & ~15U) : 32`. That value is not passed to `omega_cubin_pack`. This note did not classify every other launch site.

## Hardware-unverified

- `src/omega_cubin.c:71-73` says a cubin without program headers is rejected by `cuModuleLoadData` with error 200, and that this was confirmed on device. No receipt under `evidence/` mentions that sentence. `src/omega_cubin.h:31-33` still says `e_phoff` and `e_phnum` are 0 and that whether the loader needs program headers is what a NOT_RUN probe must decide. The C body and `cubin_elf.om` both write the headers. The header comment, the `.c` comment, and the code disagree. This note does not edit those sources.
- omega-mv#864's body and `cubin.om:30` describe `CUDA_ERROR_ILLEGAL_INSTRUCTION` (715) when the recorded count did not cover the registers the code wrote, and the PR body describes `CUDA_ERROR_INVALID_IMAGE` (200) for an ABI 8 image without `.note.nv.tkinfo`, plus later PASS lines on named GPUs through NVIDIA `libcuda`. None of that was run here. None of it names sm_121: `cubin.om` does not accept 121. Hardware-unverified, and not a GB10 result.
- `tools/artifact_export.c:183` already marks device load and device exec of the exported cubin NOT_RUN. That line was not changed, and no load was attempted.

## Open questions

1. Whether the two writers are ever consolidated is Drake's decision, on #108. This note does not choose a date or a design for that.
2. omega-mv#864's close comment says a later cubin writer is not a port of `cubin.om`. ARCH-0042 puts new cubin-writer work in CUDApp. Nobody has decided that a second, clean writer for the sm_80 family will be written.
3. The CUDApp cubin stores the integer it is given. The QMD helper stores peak + 2 rounded up to 8. The artifact exporter stores the codegen peak. Which of those belongs in `.nv.info` attribute `0x1b` and attribute `0x2f` for sm_121 is not decided, and was not loaded on a device here.
4. Whether an sm_121 ABI 8 image needs `.note.nv.tkinfo` is open. The CUDApp writer does not emit that section. No sm_121 load of either writer was run here.
5. `src/omega_cubin.h:31-33` describes a writer that omits program headers. `src/omega_cubin.c:340-375` writes them. Correcting the header is a change to a compiled file and was left for a separate change.
6. KPARAM payload bytes, `.nv.compat` words, and the tkinfo note layout were not byte-compared. The table is only the rows that were read.
