# CUDAPP-OM-ENCODER: the sm_121 encoder and the cubin packer in seeded Omega

Status: IMPLEMENTED and EXERCISED on a Linux AArch64 host and on the GB10 (DGX Spark, sm_121) beside the
live model server. Draft, not merged.

## What changed

| Piece | `.om` unit (default hosted build) | C body, now out of the default build |
|---|---|---|
| Instruction encoder + program NOP padding (`encode_single_insn`, `omega_bw_encode_program`) | `om/encoder/bw_enc_sm121.om` (515 lines) | `bw_c_encode_insn` in `src/omega_blackwell_codegen.c` (568 lines), built only for `OMEGA_GPU_NO_POSIX` and `CUDAPP_ENCODER=c\|diff` |
| Cubin ELF writer (`omega_cubin_pack`) | `om/cubin/cubin_elf.om` (523 lines) | `c_cubin_pack` and helpers in `src/omega_cubin.c` (337 lines), built only for `OMEGA_GPU_NO_POSIX`, `CUDAPP_MIGRATION_PENDING_C_CUBIN` and `CUDAPP_MIGRATION_PENDING_CUBIN_DIFF` |

Both units are compiled by the pinned Omega compiler and linked through the omega-bridge native ABI v1
binding that `om/units.sh` generates (`src/omega_om_units.c`, `src/omega_native.h`,
`src/omega_om_units.ident`; generated, checked byte for byte by `make test-om-units`).

- omc: aien-dev/omega-mv `e5b32567a7aff3ce594e3254a622a846a6cd190b`, `seed/out/omc` sha256
  `2c82480596bbdc50cda9626866209ef662d5834f278f839ff4ce0250c495029f`.
- omega-bridge: `431fff0a` (`om/omega-bridge.lock`), `src/native.om` sha256
  `c77a5922f91d53d4a0bbd4e7b1c4703f7098263abc843be2918db1ece7496782`.
- Constants: every literal in both units is the decimal form of the C hex word, converted with
  Wolfram (`wolfram-constants.txt`, `../CUDAPP-OM-CUBIN/wolfram-constants.txt`).
- `cubin_elf.om` is a translation of `src/omega_cubin.c` only; nothing from omega-mv #864 (`cubin.om`,
  on HOLD) was read (docs/ARCHITECTURE.md §6).

Why the freestanding build keeps C: the bridge binding needs pthread and mmap, which the freestanding
allowlist refuses. Successor: a freestanding omega-bridge binding with a caller-supplied stack.

## Host evidence (Linux AArch64)

| Command | Observed |
|---|---|
| `make test-om-encoder` (in `make test`) | `PASS: hosted golden-word fixtures through the .om encoder`; `encoder diff: .om and C byte-identical on 68 golden instructions and 14 fixture programs`; red check: a build that flips one bit after encoding aborts |
| `make test-om-encoder-corpus` | `encoder corpus: 243 distinct kernels identical .om vs C; diff build 559 program encodes and 32 single-instruction encodes, all SAME` (`host/corpus.log`, `host/corpus-diff.log`; `host/corpus-k-om.sha256` and `host/corpus-k-c.sha256` are equal, 243 lines) |
| `make test-om-encoder-mutants` | 3 of 3 mutant units red (dst shift 16→17, HMMA bf16 flag 262144→0, NOP control word +1): `golden words FAIL, dump vs C DIFFERS` |
| `make test-om-cubin` (in `make test`) | artifact tests through the .om packer (47 and 48 checks, 0 failed); `cubin diff: .om and C packers identical (code, bytes, message) on 869 calls`; red check aborts |
| `make test-om-cubin-mutants` | 3 of 3 mutant units red |
| `make test-om-units` | `PASS src/{omega_om_units.c,omega_native.h,omega_om_units.ident} regenerate byte for byte` |
| `make test-codegen-freestanding` | PASS; `codegen dumps byte-identical: 14 fixtures, 24448 code bytes` (hosted side .om, freestanding side C) |

`test-om-units` and both mutant targets need the omc and omega-bridge checkouts (`OMEGA_MV=`,
`OMEGA_BRIDGE_DIR=`); without them they print NOT_RUN and exit 3. CI job `om-units` runs them on the
arm64 runner with pinned clones.

Corpus note: at main `c30c0be6` the corpus target `sm121dis-corpus-bins` cannot build
`build/gpu_scan_test` (its files exist only on PR #112, branch `c3/gemm-scan` `8822da4e`). The corpus run
overlaid those files from `8822da4e` (blobs: `mk/c3.mk` 3d440dbd, `src/omega_numeric.c` 271f0027,
`src/omega_numeric.h` b3548189, `src/omega_numeric_reduce.c` 3dffadfe, `src/omega_numeric_reduce.h`
ac9cf9c9, plus that branch's `tests/gpu_{gemm,scan}_test.c`) and removed them afterwards. None of those
files are in this change.

## GB10 evidence (`chip-20261027T080240Z/`)

Tree: `010abb1549c48a4be6d63da47101d393f76cb2dd` (+ documentation edits only). One hold of
`/tmp/aien-gb10.lock`, 45 s of GPU work, beside the live model server (compute apps before and after:
only the server; health ok both times, `health.txt`). The same programs ran from `build/` (the .om
encoder; `nm` shows `omega_blackwell_codegen.o` calls `omega_bw_enc_sm121`) and from `build-enc-c/`
(`CUDAPP_ENCODER=c`, `bw_c_encode_insn`):

| Program | .om encoder | C encoder |
|---|---|---|
| `cudapp-doctor` | rc 0 | rc 0 |
| `hello_gpu` | PASS, 1048576 of 1048576 elements, 0 wrong | same |
| `hello_gpu_control --bug=short` (negative control) | rc 1, FAIL, 1 wrong 1 never written | same |
| `gpu_elementwise_test` | PASS 35 checks | same |
| `gpu_reconv_test` | PASS 235 checks | same |
| `gpu_decode_attn_test` | PASS 29 checks | same |
| `gpu_bf16_gemv_test` | PASS 12 checks | same |
| `gpu_router_test` | PASS 17 checks | same |
| `gpu_matmul_api_test` | PASS 97 checks | same |

The 70 kernels the chip runs dumped (`OMEGA_KERNEL_DUMP_DIR`) hash identically between the two
builds (`kernels-om.sha256` = `kernels-c.sha256`). The timings in the logs are not an A/B measurement.
The cubin packer is host packaging; no chip program above loads a cubin.

## Not run / limits

- Freestanding builds still use the C encoder and C packer (see above).
- Out of scope, still C: register allocation and the kernel builders, the QMMA byte patch, TC marks,
  the inline emitters in `omega_gpu_attention_api.c` and `omega_gpu_elementwise_api.c`.
- The .om path refuses with MEM before validation on allocation failure and refuses images above
  128 MiB (omc arrays stop at 16777216 elements, error 23); the C path allocates them.
- Not QUALIFIED for serving: the live model server was not rebuilt or restarted.
