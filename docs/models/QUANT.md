<!-- SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception -->
# Canonical quantization metadata and lossless repacking

`om/quant/quant.om` is a bounded host-native preparation unit for #156 and #26. It does not encode a GPU instruction, authorize a device, change the serving default, or authenticate tensor content. Tensor/dtype/lifetime semantics remain Omega #663's concern.

The schema was reviewed before freezing. Its semantic prefix does not change when a physical-only scale fold or source offset changes. `quant_hash` is an FNV indexing aid, not content identity or a replacement for SHA-256. A loader must independently bind the source weight and scale bytes to its immutable content identity. The complete metadata record is a **preparation request identity**, not an artifact-content hash: offsets and available byte extents intentionally distinguish requests.

## API and representation

All arrays are fixed-size seeded Omega views/edits. Caller arrays must be disjoint.

- `quant_sem(d view ulong 48) -> uint`: semantic declaration validation, including non-executable integer declarations.
- `quant_check(d view ulong 48) -> uint`: executable host-repack capability, layout and extent validation.
- `quant_write(d view ulong 48, b edit uchar 384) -> uint` and `quant_read(b view uchar 384, d edit ulong 48) -> uint`: fieldwise little-endian u64 serialization. These validate semantic and structural metadata, **not executable capability**. Check the returned status; malformed parsing may modify `d`.
- `quant_hash(b view uchar 384) -> ulong`: complete request index; `quant_shash(b view uchar 384) -> ulong`: semantic-prefix index over the first 192 bytes. Compare canonical bytes after hash matching.
- `quant_repack(d view ulong 48, src view uchar 65536, sc view uchar 1024, dst edit uchar 65536, ds edit uchar 1024) -> uint`: all metadata and source validation precedes any output writes. Data and scale output padding is initialized to positive zero. Repacking never expands weights into a floating-point whole-weight array.

Status: 0 success; 1 malformed schema/tag/policy/reserved; 2 unsupported combination; 3 extent/overflow/capacity; 4 stride/alignment/tail; 5 invalid stored weight/nonfinite; 6 invalid scale or fold overflow; 7 zero-point.

The scalar ABI cannot inspect the destination address. Word 27 constrains **data-row stride** and records an allocation obligation; the eventual allocator must check actual base-address alignment. Host metadata success is not GPU admission or qualification.

### Semantic prefix, words 0 through 23

| Word | Meaning |
|---|---|
| 0 | Schema, exactly 1 |
| 1 | Format: 1 integer, 2 E4M3, 3 E5M2, 4 MXFP4, 5 NVFP4, 6 MXFP8-E4M3 |
| 2 | Element bits; integer 1..8, FP4 4, FP8 8 |
| 3 | Signedness: 0 floating, 1 unsigned integer, 2 two's complement |
| 4 | Input scale dtype: 0 absent, 1 BF16, 2 E8M0, 3 UE4M3, 4 F32 |
| 5, 6 | Scale-block rows and columns; both zero when scales absent |
| 7, 8 | Zero-point mode and code: 0 absent requires code zero; 1 unsigned affine requires unsigned integer and code representable in word 2 bits |
| 9, 10 | Historical quantization rounding and saturation: 0 unspecified, 1 source RNE / finite saturation. No quantization operation is requested by repacking. |
| 11 | Subnormal: 0 inapplicable integer, 1 preserve floating |
| 12 | Stored nonfinite: 0 refuse, 1 preserve FP8 codes; never overrides reserved block-scale refusals |
| 13 | Signed zero: 0 inapplicable integer, 1 preserve floating |
| 14 | Effective storage precision, exactly word 2; no caller-selected discriminator |
| 15 | Reference numeric contract version, exactly 1; word 1 chooses the existing model semantics |
| 16 | Storage transformation classification, exactly 0 lossless |
| 17, 18 | Nonzero logical rows and columns, product checked without overflow |
| 19 | Rank, exactly 2 |
| 20..23 | Reserved zero; no unverified caller-supplied content digest |

### Physical request, words 24 through 47

| Word | Meaning |
|---|---|
| 24 | Layout: 1 FP8 row-major without scales; 2 existing mxdot block-row storage; 3 existing E4M3 GPU weight/scale storage |
| 25, 26 | Input data / scale row stride in bytes |
| 27 | Required data output alignment: power of two, at most 4096 |
| 28, 29 | Source data / scale offset in bytes |
| 30, 31 | Available source data / scale bytes, at most 65536 / 1024 |
| 32 | Tail: 0 require whole scale blocks; 1 pad final partial block, layout 2 only |
| 33, 34 | Output data / scale row stride in bytes |
| 35 | Output scale dtype, same registry as word 4 |
| 36, 37 | Exact output data / scale allocation bytes, including initialized stride padding |
| 38, 39 | Byte order / bit order, both zero: little endian and low nibble first |
| 40 | Physical arithmetic contract: 0 reference storage; 1 existing FP8 folded-scale/kernel-bit workaround, layout 3 only |
| 41 | Padded output columns; logical columns remain word 18 |
| 42..47 | Reserved zero |

Integer declarations distinguish signed and unsigned affine values but do not invent an external W1-W8 packing convention. All integer execution currently refuses status 2. This includes integer W4; the supported four-bit block-scale adapter is MXFP4/NVFP4, not a guessed integer quantizer.

## Source-backed capabilities

Source baseline: CUDApp `b759afc3f5e46f152ffb06b39882ffa28cacaf96`.

- Layout 1 accepts E4M3/E5M2 byte storage without scales. Every preserved signed-zero, subnormal, NaN and infinity code remains byte-identical. Refuse mode rejects E4M3 NaNs or E5M2 NaNs/infinities.
- Layout 2 uses `native/models/mxdot.om`: K-contiguous rows, low nibble first for FP4, one scale per live block. MXFP4 uses E8M0 per 32 columns; NVFP4 uses UE4M3 per 16; MXFP8 uses E4M3 and E8M0 per 32. E8M0 255 and UE4M3 above 126 always refuse. NVFP4 here has no second-level per-tensor scale, matching the existing model.
- Ragged layout 2 requires a caller-provided scale for the block containing the last live element. It never invents or recomputes that scale. Only the last partial block is padded; there are no wholly padded blocks. Nonzero unused input high nibbles refuse. Output padding and stride gaps are zeroed. The existing `mxmm` consumer receives word 41 as K, while the tensor retains logical word 18. A future device consumer must independently qualify this tail contract.
- Layout 3 reproduces `src/omega_gpu_fp8_pack.c` and `omega_blackwell_codegen_moe.h`: E4M3 weight bytes unchanged; BF16 grid scales over 128-by-128 blocks become F32 times the filed scale exponent 56. Both source `fabsf(s) <= 3.0e38f` checks are retained, not weakened to `isfinite`. Negative finite scales and signed zero remain accepted. No ragged rows or columns. This preparation layout does not assert that every prepared shape satisfies GPU code-generator or resource limits. In particular the kernel's special E4M3 bit arithmetic is a physical arithmetic contract, not a change to tensor semantics.

The exact FP8, MXFP4, NVFP4 and MX dot numerical implementations remain in `native/models`. No numeric rounding or saturation algorithm was duplicated here.

FNV constants were evaluated through mounted Wolfram tools: `WolframLanguageEvaluator: {FromDigits["cbf29ce484222325",16],FromDigits["100000001b3",16]} -> {14695981039346656037,1099511628211}`. They are non-cryptographic index constants, never a trust primitive.

## Canonical fixture

All words zero except:

```
0=1 1=4 2=4 4=2 5=1 6=32 11=1 13=1 14=4 15=1
17=1 18=3 19=2 24=2 25=2 26=1 27=1 30=2 31=1
32=1 33=16 34=1 35=2 36=16 37=1 41=32
```

Source bytes `33,3`, scale byte `127`. Output first two bytes `33,3`, remaining data bytes through offset 15 zero; output scale `127`. This fixture preserves preexisting numerical codes; it requests no new quantization. `quant-stream.om` exercises a native caller parsing and serializing the descriptor, hashing it, and repacking the actual byte payload.

## Host checks

Use compiler SHA256 `2c82480596bbdc50cda9626866209ef662d5834f278f839ff4ce0250c495029f` (the production `om/units.sh` pin). No GPU is opened.

```
cat om/quant/quant.om native/models/fp8.om native/models/mxfp4.om \
  native/models/nvfp4.om native/models/mxdot.om \
  tests/models/quant-consumer.om tests/models/quant-caller.om > /tmp/quant-caller.om
"$OMEGA_OMC" < /tmp/quant-caller.om > /tmp/quant-caller && chmod +x /tmp/quant-caller
/tmp/quant-caller
```

Expected exit 0. The native caller checks canonical serialization, semantic/full-request separation, W1-W8 declaration/refusal, all FP8 stored codes, MXFP4/NVFP4/MXFP8 columns 1..65, row strides and offsets, tail padding, scale boundaries, invalid tags/zero points, no writes on refusal, and an actual existing `mxmm` consumer comparison.

`quant-scales.om` emits 65536 records (one status byte and four little-endian F32 bits) for every BF16 input pattern. Compare admission and every accepted byte result to unchanged `omega_fp8_pack_scales` compiled from `src/omega_gpu_fp8_pack.c`, `src/omega_blackwell_matmul.c`, `src/omega_gpu_hostmath.c`; do not replace that source oracle with a new arithmetic script. Invalid result bits are diagnostic only.

Source mutants must fail the native caller: flip a copied weight bit; accept E8M0 255; change the scale-fold exponent; truncate full hashing to the semantic prefix; ignore reserved physical tags. The exercised exits, source/binary hashes and source-oracle result are in `tests/models/quant-host-receipt.json`.

## Integration and remaining gates

Main owns shared build registration and generated headers. Add `om/quant/quant.om` to the existing `OMEGA_UNITS` list and regenerate through the pinned Omega Bridge. Consumers concatenate this unit or use its generated native binding; do not copy its serializer. KernelCompose embeds the exact 384 bytes; cache identity separately includes immutable content SHA and realization provenance; selector/MoE must call `quant_check` and still require target qualification.

HOST_TESTED is not CHIP_EXERCISED. GPU consumer qualification, upload/traffic/residency timings, complete-workload TTFT/TPOT, logits/tokens/quality and canonical serving cutover remain NOT_RUN until an authorized hardware window and coordinator integration. Issue #156 remains open. No approximate representation, serving activation or whole-model performance claim is made.
