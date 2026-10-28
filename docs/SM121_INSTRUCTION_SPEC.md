# Authoritative SM121 Instruction Specification Format

This document specifies the authoritative single instruction-field table format implemented in `om/encoder/bw_spec_sm121.om`, following Drake's three-layer architecture (Meaning -> Realization -> Encoding) described in the Omega/encoder direction.

## Overview

In Blackwell (SM121 / GB10), instructions are represented as 128-bit words (four 32-bit little-endian words `w[0..3]`).

Instead of writing imperative bit-shifting procedures for encoding and separate pattern-matching tables for decoding, both operations are derived from **one authoritative specification table** in native `.om`.

The specification defines:
1. **Instruction Form Identification**:
   - `op`: Blackwell IR opcode (0..62).
   - `var`: Variant selector when multiple forms share an opcode (e.g., `IMAD_RRR` vs `IMAD_RRU` vs `IMAD_RRI`).
   - Fixed bits and masks for words `w[0..2]`.
   - Default control word `w[3]`.
   - Field count `nf` (number of parameterized fields in the instruction).
2. **Parameterized Fields**:
   - `role`: Field semantic role (dest register, source register, predicate, immediate).
   - `word`: Word index (0, 1, or 2) in the 128-bit instruction.
   - `shift`: Bit position within the 32-bit word.
   - `width`: Bit width of the field.
   - `is_s`: Signedness (1 for signed two's complement, 0 for unsigned).
   - `min_v`: Minimum valid value (bounds check).
   - `max_v`: Maximum valid value (bounds check).

## Field Roles

The following integer constants represent semantic field roles in `bw_spec_sm121.om`:

| Role ID | Constant | Meaning | Source in IR (`o[]`) | Bounds Check |
|---|---|---|---|---|
| 0 | `R_NONE` | Unused slot | N/A | None |
| 1 | `R_RD` | Destination GPR (0..255) | `o[1]` | `0 <= v <= 255` |
| 2 | `R_RA` | Source 1 GPR (0..255) | `o[2]` | `0 <= v <= 255` |
| 3 | `R_RB` | Source 2 GPR (0..255) | `o[3]` | `0 <= v <= 255` |
| 4 | `R_RC` | Source 3 GPR (0..255) | `o[4]` | `0 <= v <= 255` |
| 5 | `R_URD` | Destination Uniform Reg (0..31) | `o[1]` | `0 <= v <= 31` |
| 6 | `R_URA` | Source 1 Uniform Reg (0..31) | `o[2]` | `0 <= v <= 31` |
| 7 | `R_URB` | Source 2 Uniform Reg (0..31) | `o[5]` or `o[1]` | `0 <= v <= 31` |
| 8 | `R_PRED`| 4-bit Guard Predicate | `pred(o)` | `0 <= v <= 7` (or 8) |
| 9 | `R_IMM32`| 32-bit Immediate | `o[6]` | Full 32-bit range |
| 10 | `R_OFF24`| Signed 24-bit byte offset | `o[6]` | `-8388608 <= v <= 8388607` |
| 11 | `R_IMM16`| 16-bit Immediate | `o[6]` | `0 <= v <= 65535` |
| 12 | `R_BAR` | Barrier register | `o[11]` | Declared form bounds |
| 13 | `R_BSSY` | Relative instruction delta | `o[6]` | Declared bounds and exact representability of `(v - 1) * 16` |
| 14 | `R_BRA` | Branch delta | `o[6]` | Unsigned 32-bit IR representation |
| 15 | `R_LDC` | Constant-bank offset | `o[6]` | Declared bounds and exact alignment/field representation |
| 16 | `R_SHFL` | Shuffle immediate | `o[6]` | Raw value, no masking before validation |
| 17 | `R_LUT` | LOP3 LUT | `o[6]` | Raw value, no masking before validation |
| 18 | `R_S2R` | Special-register index | `o[6]` | Declared form bounds |
| 19 | `R_LDSW` | Shared-memory offset | `o[6]` | Declared form bounds and exact representation |
| 20 | `R_FSET` | FSETP comparison encoding | `o[6]` | Declared form bounds; zero selects the default encoding |
| 21 | `R_PRED3` | Predicate destination/source | `o[1]` or `o[4]` | Raw value, no masking before validation |

## Specification Table Representation

In native seeded Omega, the specification is laid out as a flat global table `sp_tbl ulong 3000`.

Each instruction form header occupies 10 slots:
- `+0`: `op` (Blackwell IR opcode)
- `+1`: `var` (variant selector)
- `+2`: `w0_fix` (fixed bits for word 0)
- `+3`: `w0_mask` (mask of fixed bits for word 0)
- `+4`: `w1_fix` (fixed bits for word 1)
- `+5`: `w1_mask` (mask of fixed bits for word 1)
- `+6`: `w2_fix` (fixed bits for word 2)
- `+7`: `w2_mask` (mask of fixed bits for word 2)
- `+8`: `w3_def` (default control word 3)
- `+9`: `nf` (number of parameterized fields, 0..5)

Following the header, each of the `nf` fields occupies 7 slots:
- `+0`: `role`
- `+1`: `word`
- `+2`: `shift`
- `+3`: `width`
- `+4`: `is_s`
- `+5`: `min_v`
- `+6`: `max_v`

Total slots per form is `10 + 5 * 7 = 45`. For 63 forms, the table uses `63 * 45 = 2835` slots.

## Error Refusal Codes

The encoder refuses invalid input with explicit error codes instead of truncating or wrapping:

- `E_OK = 0`: Successful encoding or decoding.
- `E_UNSUP = 1`: Opcode or variant is not supported.
- `E_RANGE = 2`: Operand exceeds declared bounds or cannot be represented without losing semantic bits.
- `E_NOMAT = 3`: Instruction machine words do not match any known form's fixed bits and masks.

## Dual Derivation

1. **`sp_enc o w -> err`**:
   - Matches `o` to an instruction form.
   - Sets fixed word bits `w0_fix..w2_fix` and control word `w3`.
   - Validates the raw semantic operand before inserting masked bits.
   - Checks special-layout immediates by decoding their inserted representation and requiring exact equality.
2. **`sp_dec w o -> err`**:
   - Initializes its shared table automatically, including standalone cold calls.
   - Searches `sp_tbl` for a form whose fixed bits match: `(w[k] & mask) == fix`.
   - Extracts each field: `val = (w[word] >> shift) & mask`.
   - Populates decoded IR array `o[]`.
3. **Host verification**:
   - Compares every active semantic role after decoding, not just the opcode. Inactive IR packing slots are not operands.
   - Canonicalizations are explicit: absent source operands for LOP3/SHF/IMAD.HI mean RZ; zero control selects the form default; zero FSETP immediate selects its default encoding.
   - IMAD RRU has a GPR destination and a uniform second source. Corpus generation verifies that each intended form actually dispatches.
   - Boundary fixtures select representable aligned immediates; separate audit regressions require refusal of raw lossy values.
   - `sh tools/test_bw_spec_encoder.sh` checks reference bytes, active operands, cold decode, malformed streams and independent offline nvdisasm operands. Producer failure, truncated records and read errors fail closed.
   - All checks are host-only. Physical qualification of the changed encoder remains NOT_RUN until an authorized chip run. Frozen historical receipts are unchanged.
