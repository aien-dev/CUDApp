/* sm121dis -- Omega's sm_121 disassembler on the command line (C6 of docs/OMEGA.md; the decoder is
 * src/omega_sm121_disasm.c).
 *
 *   sm121dis <code.bin>            a listing in nvdisasm's layout: the address in a comment, then TEXT ;
 *   sm121dis -b SM121 <code.bin>   the same, called the way the checks used to call nvdisasm
 *   sm121dis --pairs <code.bin>    one line per word: <addr> <lo hex> <hi hex> <text> ;  (what check.sh compares)
 *   sm121dis --version             names the tool, for the check logs
 *
 * Exit 0 when every word was a known form, 1 when some were not (printed "?? ..."), 2 for a usage or file error. */
#include "omega_sm121_disasm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int usage(void) {
    fprintf(stderr, "usage: sm121dis [--pairs | -b SM121] <code.bin> | --version\n");
    return 2;
}

int main(int argc, char **argv) {
    if (argc == 2 && strcmp(argv[1], "--version") == 0) {
        printf("sm121dis: Omega's own sm_121 disassembler (src/omega_sm121_disasm.c)\n");
        return 0;
    }
    int pairs = argc == 3 && strcmp(argv[1], "--pairs") == 0;
    int arch = argc == 4 && strcmp(argv[1], "-b") == 0;
    if (arch && strcmp(argv[2], "SM121") != 0) { fprintf(stderr, "sm121dis: only -b SM121\n"); return 2; }
    if (argc != 2 + pairs + 2 * arch) return usage();
    const char *path = argv[argc - 1];
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return 2; }
    size_t cap = 1 << 16, len = 0, n;
    unsigned char *code = malloc(cap);
    while (code && (n = fread(code + len, 1, cap - len, f)) > 0) {
        len += n;
        if (len == cap) { unsigned char *c = realloc(code, cap *= 2); if (!c) { free(code); code = NULL; } else code = c; }
    }
    fclose(f);
    if (!code) { fprintf(stderr, "sm121dis: out of memory\n"); return 2; }
    if (len % 16) { fprintf(stderr, "sm121dis: %s is not a whole number of 16-byte words\n", path); free(code); return 2; }
    int unknown = 0;
    size_t pad = omega_sm121_padding_start(code, len);
    if (!pairs) unknown = omega_sm121_listing(code, len, stdout);
    else for (size_t at = 0; at < len; at += 16) {
        uint64_t lo = 0, hi = 0;
        for (int i = 7; i >= 0; i--) { lo = lo << 8 | code[at + i]; hi = hi << 8 | code[at + 8 + i]; }
        char t[256];
        int rc = omega_sm121_disasm(lo, hi, at, t, sizeof t);
        if (rc == -2) { unknown = -1; break; }
        if (rc) unknown++;
        printf("%04llx %016llx %016llx %s%s;\n", (unsigned long long)at, (unsigned long long)lo, (unsigned long long)hi, t,
               pad && at >= pad ? "" : " ");
    }
    free(code);
    if (unknown < 0) { fprintf(stderr, "sm121dis: text too long\n"); return 2; }
    return unknown ? 1 : 0;
}
