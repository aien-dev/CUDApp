/* omega_sm121_disasm.h -- Omega's own sm_121 (GB10) disassembler (C6 of docs/OMEGA.md).
 *
 * Turns one 128-bit instruction word into the text nvdisasm prints for it, so no NVIDIA tool is needed to read
 * machine code back. It knows the instruction forms Omega's own builders emit (src/omega_blackwell_codegen*.c,
 * src/omega_osc_gpu.c), each as an exact template: the fixed bits must match exactly and only the operand fields
 * vary. A word that matches no template is printed as "?? <hex>" and reported, never guessed, so a comparison
 * with nvdisasm fails on any form this file does not know. The control bits (stall, barriers, yield, reuse:
 * bits 105..127) are not part of the text, as in nvdisasm's default listing.
 *
 * tools/sm121dis/sm121dis.c is the command line (a listing in nvdisasm's layout); tools/sm121dis/check.sh compares
 * it with nvdisasm, the oracle, on every kernel the library builds (captured with OMEGA_KERNEL_DUMP_DIR). */
#ifndef OMEGA_SM121_DISASM_H
#define OMEGA_SM121_DISASM_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

/* Writes the text of the instruction (lo = bits 0..63, hi = bits 64..127) at byte address pc into out, without the
 * trailing " ;". Returns 0 when the word matched a known form, -1 when it did not (out then holds "?? 0x<hi><lo>"),
 * -2 for a NULL or too small buffer. */
int omega_sm121_disasm(uint64_t lo, uint64_t hi, uint64_t pc, char *out, size_t outn);

/* Writes a listing of a whole kernel (len bytes, a whole number of 16-byte words) to out in nvdisasm's layout
 * (one line per word: the address in a comment, then TEXT ;), the form the listing checks in tests/ read. Returns
 * the number of words that matched no known form (printed "?? ..."), or -1 for a bad argument or a length that is
 * not whole words. */
int omega_sm121_listing(const uint8_t *code, size_t len, FILE *out);

/* The byte offset where a kernel's padding starts: just after its last EXIT (0 when it has none). nvdisasm prints the
 * words from there on (the branch to itself and the NOPs) as "TEXT;" with no space before the semicolon, and every
 * word before as "TEXT ;" (so on every library kernel tools/sm121dis/check.sh compares). */
size_t omega_sm121_padding_start(const uint8_t *code, size_t len);

#endif /* OMEGA_SM121_DISASM_H */
