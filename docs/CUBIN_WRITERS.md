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
| Targets | SM 121 only (`omega_cubin.h:60`, note word at `omega_cubin.c:239`) | 80, 86, 87, 89, 90, 100, 103, 120, else exit 2 (`cubin.om:165-178`). 121 is refused |
| ABI | Version 8 only (`omega_cubin.h:58`, `omega_cubin.c:354`). OS/ABI byte `0x41` (`omega_cubin.c:353`) | 7 or 8 (`cubin.om:689-701`). ABI 8 writes decimal 65 at byte 7 and 8 at byte 8. ABI 7 writes decimal 51 and 7. Whether 65 is the same byte as `0x41` was not re-evaluated |
| `e_flags` | The constant `0x6007902` (`omega_cubin.h:59`, stored at `omega_cubin.c:361`) | Computed from SM (`cubin.om:694-697`). `args` rejects SM 121 before `ehdr` runs, so this file never stores an sm_121 flags word. The formula was not evaluated against `0x6007902` |
| Register count stored | The caller's `gpr_count`, unchanged, in `.nv.info` attribute `0x2f` (`omega_cubin.c:224-225`) and attribute `0x1b` (`omega_cubin.c:271`). `cubin_elf.om:316` and `:372-373` store the same field. Callers pass the declared count (`omega_bw_gpr_declared`, see below) | Input REGS is checked, then `VREG` is replaced with REGS+2 (`cubin.om:317`) and that sum is what `ginfo` stores (`cubin.om:386`). Attribute 27 (decimal) is the constant 255 on both the pre-sm_90 path (`cubin.om:539`) and the sm_90+ path (`cubin.om:545`), not `VREG` |
| Register refusal | `gpr_count > 255` (`omega_cubin.c:172`, `cubin_elf.om:164`). 0 is accepted. The text is not scanned | REGS outside 1..253 is exit 2 (`cubin.om:189`). `regchk` (`cubin.om:295-316`) exits 6 when REGS is below a register number taken from byte 2 of an instruction, with the exceptions named in the notes at `cubin.om:36-39` and `:291-294`. Then it adds 2 |
| Exit numbers | 0 ok, 1 bad argument, 2 text, 3 parameters, 4 constant bank, 5 GPR above 255, 6 shared size not a multiple of 4, 7 allocation (`omega_cubin.h:67-76`) | 0 written, 1 usage, 2 bad number, 3 bad name or signature, 4 bad input, 5 write failed, 6 register coverage (`cubin.om:35-37`). Exit 6 does not mean the same thing |
| `.note.nv.tkinfo` | Section 11, always written: type 2000, owner `NVIDIA Corp`, descriptor 2, 0 and the offsets 1, 8, 34, 52 of this writer's own strings (`omega_cubin.c:242-258`, `:325`; `cubin_elf.om:393-410`, `:484-486`). The GB10 driver refuses an sm_121 image without it (G2 windows, below) | Appended only for ABI 8 (`cubin.om:218-221`). Writer at `cubin.om:429` |
| `.nv.callgraph` | Not in the section list | Section writer `cgraph` at `cubin.om:394` |
| Other sections | Always: `.shstrtab`, `.strtab`, `.symtab`, `.text.<fn>`, `.nv.constant0.<fn>`, `.nv.info`, `.nv.compat`, `.nv.info.<fn>`, `.nv.shared.reserved.0`, `.note.nv.cuinfo`, `.note.nv.tkinfo` (`omega_cubin.c:280-282`) | Name table at `cubin.om:41`. `.nv.compat` and `.nv.shared.reserved.0` are added from sm_90 (`cubin.om:208-212`). Order is not the CUDApp order |
| `.text` `sh_info` | 2 (`omega_cubin.c:318`, `cubin_elf.om:452`) | 3 from sm_90, otherwise a word that includes `VREG` shifted (`cubin.om:640-644`) |
| `.note.nv.cuinfo` link | `sh_link` 11 (the tkinfo section), `sh_info` 2, flags literal `0x1000040` (`omega_cubin.c:324`, `cubin_elf.om:480-481`) | `sh_link` is `VTK`: 0 on ABI 7, and the tkinfo section index on ABI 8 (`cubin.om:217-221`, `:638`). `sh_info` is 0 below sm_90 and 10 from sm_90 (`cubin.om:634-637`). Flags literal 16777216 below sm_90 and 16777280 from sm_90 (`cubin.om:630-632`). The 2026-10-27 Wolfram receipt maps `0x1000040` to 16777280. The 16777216 literal was not given a hex spelling in that receipt |
| EXIT offsets | `EIATTR_EXIT_INSTR_OFFSETS` is omitted (`omega_cubin.c:274-275`, `omega_cubin.h:45-48`). The GB10 driver ran an image without it (G2 window `20261030T235858Z-10fa8c`, H6) | Input with no EXIT is exit 4 (`cubin.om:503-505`). Offsets are written (`cubin.om:506-513`) |
| Parameter base | `0x380` for the only target (`omega_cubin.h:61`) | Decimal 352 below sm_90, 528 from sm_90 and below sm_100, 896 from sm_100 (`cubin.om:196-202`). 896 is the receipt's reading of `0x380`. cubin.om never selects it for sm_121, because sm_121 is refused |
| Parameters | Caller supplies offset, size 1, 2, 4, or 8, and a pointer bit. At most 32 (`omega_cubin.h:63`, `omega_cubin.c:168`). Records are written in caller order (`omega_cubin.c:124`) | Signature letters `p` (8) and `u` (4) only, 1 to 16 (`cubin.om:32`, `:252-266`). `kparams` writes them last to first (`cubin.om:488`). The KPARAM type bytes (`cubin.om:475-480`) were not byte-compared with CUDApp's type words `0x0021f500` and `0x0011f000` (`omega_cubin.c:130`) |
| Name | 1 to 64 bytes, first character a letter or `_` (`omega_cubin.c:88-98`) | 1 to 32 bytes (`cubin.om:31`, buffer at `:49`). A leading digit is accepted (`cubin.om:233`) |
| Text | Nonzero multiple of 16, at most `1 << 24` bytes (`omega_cubin.c:164`) | Multiple of 16, below 65536 bytes (`cubin.om:33`, buffer at `:47`). The same note requires one EXIT (`cubin.om:34`) |
| Shared | Caller `shared_reserved`, multiple of 4, section always present (`omega_cubin.c:176`, `:323`) | From sm_100 the reservation literal is 64, otherwise 0 (`cubin.om:205-215`). The section exists from sm_90 |
| Program headers | Five headers are written (`omega_cubin.c:390-394`, `cubin_elf.om:501-512`) | Written (`cubin.om:746-758`). Count is 4 below sm_90 and 5 from sm_90 (`cubin.om:206-211`) |

`tools/artifact_export.c` passes `omega_bw_gpr_declared(kern.gpr_count)` into the QMD and the cubin, and refuses the export when `omega_bw_gpr_declared_ok` fails. `kern.gpr_count` is the allocator peak (`src/omega_blackwell_codegen.c:1405`). Before the G2 windows it passed the peak itself; that image faulted on GB10 (below). The same tool still sets `device-load=NOT_RUN` (`tools/artifact_export.c:210`) in each exported manifest: the export cannot know a later device result. `omega_native_artifact.c:300-301` forwards the manifest `gpr_count` and a shared size of 64.

For `matmul-bf16-v1` the tool picks one of two kernels. K = 16 uses the single-tile kernel (`omega_blackwell_codegen_matmul_tensor`), whose CTA (x, y) computes only the 16 x 8 tile at row 16y, column 8x, so the export grid is (n/8, m/16). K > 16 uses the looped kernel (`omega_blackwell_codegen_matmul_tensor_loop`, omega-mv `5e29b826`), whose CTA x walks column tiles x, x + grid_x, …; grid_x follows the matmul API's CTA budget. Shapes with m % 16, n % 8 or k % 16 nonzero are refused (exit 3). Before 2026-10-31 the tool wrote grid_x = 1 for every export and called only the single-tile kernel: every export with K > 16 was refused, and a K = 16 export with n > 8 would have left columns 8..n-1 unwritten. `tools/artifact_export_shapes.sh` (run by `make test-native-artifact`) checks the grids and refusals; against the old tool it reports four failures. The 16x16x8 export is unchanged (`1dac563e…c007b`).

## A register reserve that is not the cubin writer

`omega_bw_gpr_declared` (`src/omega_blackwell_codegen.c:2712`) returns `((peak + 2u + 7u) / 8u) * 8u`. The comment at `src/omega_blackwell_codegen.h:293-298` says this is the count to declare in the QMD: peak + 2, rounded up to a multiple of 8. Launchers call it, and so does `tools/artifact_export.c` before packing. The header calls the exact minimum UNKNOWN and the rule INFERRED; the G2 window below ran the matmul kernel (peak 32) bit-exact at REGCOUNT 34 (peak + 2) and 40 (the rule), and faulted at 32 (the peak). One kernel, one chip.

That helper is one launch declaration. It is not the rule `cubin.om` implements. `cubin.om` adds 2 to the count it stores in `.nv.info` and does not round to 8 (`cubin.om:317`). The two `+ 2` operations are different code, on different outputs.

`src/omega_blackwell_submit.c:265` puts a third expression into a QMD config: `(kernel->gpr_count > 16) ? ((kernel->gpr_count + 15) & ~15U) : 32`. That value is not passed to `omega_cubin_pack`. This note did not classify every other launch site.

## Device evidence (GB10, G2 windows)

Three windows ran on the DGX Spark GB10 under ledger L881 with NVIDIA `libcuda` 580.173.02, through a C probe (`cuModuleLoadDataEx`) and through MAX `v26.7.0` (`DeviceExternalFunction`). Raw logs and result tables are under `evidence/MAX-INTEROP/G2/<window>/`; images derived from ptxas output are named by sha256 only.

- `20261030T235211Z-0352f1`: the CUDApp#68 image (this writer, no tkinfo) is refused with 200 `INVALID_IMAGE` by both callers; the same image with this writer's own tkinfo note loads, then faults with 715 `ILLEGAL_INSTRUCTION` at REGCOUNT 32. The ptxas control loads and runs.
- `20261030T235858Z-10fa8c`: with tkinfo, REGCOUNT and MAXREG 40 (H1), REGCOUNT 40 alone (H2), 34 (H4) and tkinfo + 40 (H6) run bit-exact in both callers; MAXREG 40 alone (H3) and ISA_CLASS 1 alone (H5) still fault with 715; REGCOUNT 40 without tkinfo (H7) is refused with 200. So the two defects were the missing note and REGCOUNT equal to the peak; ISA_CLASS and the program headers are not causes.
- `20261031T003009Z-0b0fa9`: this writer's own `artifact-export` output, unedited. Matmul 16x16x8 (REGCOUNT 40, sha256 `1dac563e…c007b`) loads and runs with worst_abs 0 in both callers; attention hq=8 hkv=1 hd=128 (REGCOUNT 80, `90b5c752…6d2169`) loads and resolves (not launched). The main writer's matmul and attention images are still refused with 200; the four negative controls are refused.
- omega-mv#864's statements about other GPUs (715 for an uncovered register count, 200 for an ABI 8 image without tkinfo) were not rerun here; the GB10 results above agree with them on sm_121.

## Open questions

1. Whether the two writers are ever consolidated is Drake's decision, on #108. This note does not choose a date or a design for that.
2. omega-mv#864's close comment says a later cubin writer is not a port of `cubin.om`. ARCH-0042 puts new cubin-writer work in CUDApp. Nobody has decided that a second, clean writer for the sm_80 family will be written.
3. Answered for the CUDApp writer on GB10: attributes `0x1b` and `0x2f` both carry the declared count (peak + 2 rounded up to 8); the peak alone faults (G2 window `20261030T235858Z-10fa8c`). The exact minimum across kernels is still the INFERRED rule, measured on one kernel.
4. Answered: an sm_121 ABI 8 image needs `.note.nv.tkinfo`; without it GB10 refuses the image with 200. The CUDApp writer emits it (section 11).
5. Whether the program headers are necessary is open; the 200 previously blamed on them was the missing tkinfo note (`src/omega_cubin.h:40-44`, `src/omega_cubin.c:71-75`).
6. KPARAM payload bytes and `.nv.compat` words were not byte-compared. The tkinfo note was compared only with the E3 variant that ran on GB10 (same header words; descsz 0x60 against 0x58 because of the tool string), not with a ptxas note.
