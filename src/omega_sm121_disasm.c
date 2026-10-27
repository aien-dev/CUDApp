/* omega_sm121_disasm.c -- see omega_sm121_disasm.h.
 *
 * Layout of a word (lo = bits 0..63, hi = bits 64..127): opcode bits 0..11; guard predicate bits 12..14 with its
 * negation at 15 (7 and no negation = no guard); destination register 16..23, first source 24..31, second source
 * 32..39 or a 32-bit immediate at 32..63; third source at hi 0..7; modifiers in hi 8..40; control from hi 41 on.
 *
 * A row of the table is one instruction form: its opcode and fixed bits (lov, hiv) and a format. Each operand code
 * in the format names the bits it reads; every other bit of lo and of hi 0..40 (outside the guard) must equal the
 * row's fixed value, or the row does not match. The rows come from the words Omega's builders write
 * (src/omega_blackwell_codegen.c encode_single_insn, the em helpers, src/omega_osc_gpu.c); the text of each was
 * checked against nvdisasm 13.0 -b SM121 by tools/sm121dis/check.sh. */
#include "omega_sm121_disasm.h"
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#define HI_BITS 41
#define HI_MASK ((1ULL << HI_BITS) - 1)

typedef struct {
    uint64_t lov, hiv;   /* fixed bits (operand fields zero) */
    const char *fmt;     /* text; %x codes are operands, see field() */
} Row;

/* operand codes: which bits they read (lo or hi, first bit, width); several fields for the branch target */
typedef struct { char code; int hi; int at; int n; } Field;
static const Field FIELDS[] = {
    {'d', 0, 16, 8}, {'a', 0, 24, 8}, {'b', 0, 32, 8}, {'c', 1, 0, 8},   /* R registers */
    {'D', 0, 16, 8}, {'A', 0, 24, 8}, {'B', 0, 32, 8}, {'C', 1, 0, 8},   /* uniform registers */
    {'i', 0, 32, 32}, {'s', 0, 32, 32}, {'f', 0, 32, 32},               /* 32-bit immediate: unsigned, signed, float */
    {'K', 0, 38, 21},                                                   /* c[bank][offset]: offset 38..53, bank 54..58 */
    {'U', 0, 37, 22},                                                   /* uniform load c[bank][offset]: offset 37..53 */
    {'o', 0, 40, 24},                                                   /* signed 24-bit address offset */
    {'Y', 0, 32, 32},                                                   /* BSSY target: signed byte distance */
    {'y', 0, 16, 4},                                                    /* convergence barrier B0..B15 */
    {'P', 1, 17, 3}, {'Q', 1, 20, 3}, {'R', 1, 23, 4},                  /* predicates; R carries its negation (hi 26) */
    {'l', 1, 8, 8},                                                     /* LOP3 truth table */
    {'h', 0, 53, 5}, {'m', 0, 40, 13},                                  /* SHFL lane distance, clamp */
    {'S', 1, 8, 8},                                                     /* special register */
    {'G', 1, 12, 3}, {'g', 1, 12, 4},                                   /* integer and float compare */
    {'Z', 1, 9, 1}, {'z', 1, 9, 3},                                     /* ISETP signed flag; memory access size */
    {'N', 1, 8, 1}, {'M', 0, 63, 1},                                    /* IADD3 negations: first, second source */
    {'j', 0, 32, 32},                                                   /* ISETP immediate (printed signed) */
};

static const Field *field_of(char c) {
    for (size_t i = 0; i < sizeof FIELDS / sizeof FIELDS[0]; i++) if (FIELDS[i].code == c) return &FIELDS[i];
    return NULL;
}

static uint64_t bits(uint64_t v, int at, int n) { return n >= 64 ? v >> at : (v >> at) & ((1ULL << n) - 1); }
static uint64_t mask(int at, int n) { return (n >= 64 ? ~0ULL : ((1ULL << n) - 1)) << at; }

/* the BRA distance: 4-byte units, low 8 bits at 16..23, the rest (signed) at 34..81 */
#define BRA_LO_MASK (mask(16, 8) | mask(34, 30))
#define BRA_HI_MASK mask(0, 18)

typedef struct { char *o; size_t n, at; int ok; } Out;

static void put(Out *w, const char *fmt, ...) {
    if (w->at >= w->n) { w->ok = 0; return; }
    va_list ap; va_start(ap, fmt);
    int k = vsnprintf(w->o + w->at, w->n - w->at, fmt, ap);
    va_end(ap);
    if (k < 0 || (size_t)k >= w->n - w->at) { w->ok = 0; w->at = w->n; return; }
    w->at += (size_t)k;
}

static void reg(Out *w, const char *p, uint64_t r) { if (r == 255) put(w, "%sZ", p); else put(w, "%s%u", p, (unsigned)r); }
static void hex_s(Out *w, int64_t v) { if (v < 0) put(w, "-0x%llx", (unsigned long long)(-v)); else put(w, "0x%llx", (unsigned long long)v); }
static void pred(Out *w, uint64_t p, int neg) { put(w, "%s", neg ? "!" : ""); if (p == 7) put(w, "PT"); else put(w, "P%u", (unsigned)p); }

static const char *SR[256] = {
    [0x21] = "SR_TID.X", [0x22] = "SR_TID.Y", [0x23] = "SR_TID.Z",
    [0x25] = "SR_CTAID.X", [0x26] = "SR_CTAID.Y", [0x27] = "SR_CTAID.Z",
    [0x00] = "SR_LANEID", [0x50] = "SR_CLOCKLO", [0x51] = "SR_CLOCKHI",
};
static const char *ICMP[8] = { "F", "LT", "EQ", "LE", "GT", "NE", "GE", "T" };
static const char *FCMP[16] = { "F", "LT", "EQ", "LE", "GT", "NE", "GE", "NUM", "NAN", "LTU", "EQU", "LEU", "GTU", "NEU", "GEU", "T" };
static const char *MSIZE[8] = { ".U8" , ".S8", ".U16", ".S16", "", ".64", ".128", ".U.128" };

/* nvdisasm prints a float immediate the shortest way that reads back exactly, with INF and NAN by name */
static void fimm(Out *w, uint32_t u) {
    float f; memcpy(&f, &u, 4);
    if ((u & 0x7f800000u) == 0x7f800000u) {
        if (u & 0x007fffffu) put(w, "%sQNAN", (u >> 31) ? "-" : "+");
        else put(w, "%sINF", (u >> 31) ? "-" : "+");
        return;
    }
    char b[64];
    for (int p = 1; p <= 20; p++) {
        snprintf(b, sizeof b, "%.*g", p, (double)f);
        float g; if (sscanf(b, "%g", &g) == 1 && memcmp(&g, &f, 4) == 0) break;
    }
    put(w, "%s", b);
}

/* Prints row's format for (lo, hi); 0 when every operand could be printed */
static void render(Out *w, const char *fmt, uint64_t lo, uint64_t hi, uint64_t pc) {
    for (const char *p = fmt; *p; p++) {
        if (*p != '%') { put(w, "%c", *p); continue; }
        char c = *++p;
        const Field *f = field_of(c);
        uint64_t v = f ? bits(f->hi ? hi : lo, f->at, f->n) : 0;
        switch (c) {
#ifdef SM121DIS_MUTANT
        case 'a': reg(w, "R", v == 255 ? v : v ^ 1); break; /* the red check: first source register off by one bit */
        case 'd': case 'b': case 'c': reg(w, "R", v); break;
#else
        case 'd': case 'a': case 'b': case 'c': reg(w, "R", v); break;
#endif
        case 'D': case 'A': case 'B': case 'C': reg(w, "UR", v); break;
        case 'i': put(w, "0x%x", (unsigned)v); break;
        case 's': hex_s(w, (int32_t)(uint32_t)v); break;
        case 'f': fimm(w, (uint32_t)v); break;
        case 'K': put(w, "c[0x%x][0x%x]", (unsigned)bits(lo, 54, 5), (unsigned)bits(lo, 38, 16)); break;
        case 'U': put(w, "c[0x%x][0x%x]", (unsigned)bits(lo, 54, 5), (unsigned)bits(lo, 37, 17)); break;
        case 'o': { int32_t off = (int32_t)((uint32_t)v << 8) >> 8; if (off) { put(w, "+"); hex_s(w, off); } /* nvdisasm writes a negative offset as +-0x4 */ break; }
        case 'Y': put(w, "0x%llx", (unsigned long long)(pc + 16 + (int64_t)(int32_t)(uint32_t)v)); break;
        case 'y': put(w, "B%u", (unsigned)v); break;
        case 'P': case 'Q': pred(w, v, 0); break;
        case 'R': pred(w, v & 7, (int)((v >> 3) & 1)); break;
        case 'l': put(w, "0x%x", (unsigned)v); break;
        case 'h': case 'm': put(w, "0x%x", (unsigned)v); break;
        case 'S': if (SR[v]) put(w, "%s", SR[v]); else put(w, "SR%u", (unsigned)v); break;
        case 'G': put(w, "%s", ICMP[v]); break;
        case 'Z': if (!v) put(w, ".U32"); break;
        case 'z': put(w, "%s", MSIZE[v]); break;
        case 'N': case 'M': if (v) put(w, "-"); break;
        case 'j': hex_s(w, (int32_t)(uint32_t)v); break; /* signed even in a .U32 compare, as nvdisasm prints it */
        case 'g': put(w, "%s", FCMP[v]); break;
        case 'T': {
            uint64_t units = bits(lo, 16, 8) | (bits(lo, 34, 30) << 8) | (bits(hi, 0, 18) << 38);
            int64_t s = (int64_t)(units << 8) >> 8; /* 56-bit signed */
            put(w, "0x%llx", (unsigned long long)(pc + 16 + s * 4));
            break;
        }
        default: w->ok = 0; return;
        }
    }
}

/* the bits a format reads */
static void fields_mask(const char *fmt, uint64_t *lom, uint64_t *him) {
    *lom = 0; *him = 0;
    for (const char *p = fmt; *p; p++) {
        if (*p != '%') continue;
        char c = *++p;
        if (c == 'T') { *lom |= BRA_LO_MASK; *him |= BRA_HI_MASK; continue; }
        const Field *f = field_of(c);
        if (!f) continue;
        if (f->hi) *him |= mask(f->at, f->n); else *lom |= mask(f->at, f->n);
    }
}

/* The forms. lo values include the opcode (bits 0..11); guard bits 12..15 are never part of a row. */
static const Row ROWS[] = {
    /* moves, constants, special registers */
    { 0x0000000000007802ull, 0x00000f00ull, "MOV %d, %i" },
    { 0x00000000ff007202ull, 0x00000f00ull, "MOV %d, %b" },
    { 0x0000000000007202ull, 0x00000f00ull, "MOV %d, %b" },
    { 0x0000000000007919ull, 0x00000000ull, "S2R %d, %S" },
    { 0x00000000ff007b82ull, 0x00000800ull, "LDC %d, %K" },
    { 0x00000000ff007b82ull, 0x00000a00ull, "LDC.64 %d, %K" },
    { 0x00000000ff0077acull, 0x08000800ull, "LDCU %D, %U" },
    { 0x00000000ff0077acull, 0x08000a00ull, "LDCU.64 %D, %U" },
    /* integer */
    { 0x0000000000007224ull, 0x078e0200ull, "IMAD %d, %a, %b, %c" },
    { 0x0000000000007c24ull, 0x0f8e0200ull, "IMAD %d, %a, %B, %c" },
    { 0x0000000000007825ull, 0x078e0000ull, "IMAD.WIDE.U32 %d, %a, %i, %c" },
    { 0x0000000000007227ull, 0x078e0000ull, "IMAD.HI.U32 %d, %a, %b, %c" },
    { 0x0000000000007210ull, 0x07ffe000ull, "IADD3 %d, PT, PT, %N%a, %M%b, %c" },
    { 0x0000000000007810ull, 0x07ffe000ull, "IADD3 %d, PT, PT, %N%a, %s, %c" },
    { 0x0000000000007212ull, 0x000e0000ull, "LOP3.LUT %d, %a, %b, %c, %l, %R" },
    { 0x0000000000007812ull, 0x000e0000ull, "LOP3.LUT %d, %a, %i, %c, %l, %R" },
    { 0x0000000000007819ull, 0x00011600ull, "SHF.R.U32.HI %d, %a, %i, %c" },
    { 0x0000000000007219ull, 0x00011600ull, "SHF.R.U32.HI %d, %a, %b, %c" },
    { 0x0000000000007219ull, 0x00000600ull, "SHF.L.U32 %d, %a, %b, %c" },
    { 0x0000000000007819ull, 0x00000600ull, "SHF.L.U32 %d, %a, %i, %c" },
    { 0x000000000000720cull, 0x00000070ull, "ISETP.%G%Z.AND %P, %Q, %a, %b, %R" },
    { 0x000000000000780cull, 0x00000070ull, "ISETP.%G%Z.AND %P, %Q, %a, %j, %R" },
    { 0x0000000000007c0cull, 0x08000070ull, "ISETP.%G%Z.AND %P, %Q, %a, %B, %R" },
    { 0x0000000000007207ull, 0x00000000ull, "SEL %d, %a, %b, %R" },
    { 0x0000000000007807ull, 0x00000000ull, "SEL %d, %a, %i, %R" },
    /* float */
    { 0x0000000000007221ull, 0x00000000ull, "FADD %d, %a, %b" },
    { 0x8000000000007221ull, 0x00000000ull, "FADD %d, %a, -%b" },
    { 0x0000000000007220ull, 0x00400000ull, "FMUL %d, %a, %b" },
    { 0x0000000000007223ull, 0x00000000ull, "FFMA %d, %a, %b, %c" },
    { 0x0000000000007209ull, 0x00000000ull, "FMNMX %d, %a, %b, %R" },
    { 0x000000000000720bull, 0x00000000ull, "FSETP.%g.AND %P, %Q, %a, %b, %R" },
    { 0x0000000000007245ull, 0x00201000ull, "I2FP.F32.U32 %d, %b" },
    { 0x0000000000007245ull, 0x00201400ull, "I2FP.F32.S32 %d, %b" },
    { 0x0000000000007305ull, 0x0020f000ull, "F2I.U32.TRUNC.NTZ %d, %b" },
    { 0x000000000000723eull, 0x0004807000ull, "F2FP.SATFINITE.E4M3.F32.PACK_AB_MERGE_C %d, %a, %b, %c" },
    { 0x000000000000723eull, 0x00000010ffull, "F2FP.BF16.F32.PACK_AB %d, %a, %b" },
    /* tensor cores */
    { 0x000000000000723cull, 0x00041800ull, "HMMA.16816.F32.BF16 %d, %a, %b, %c" },
    { 0x000000000000727aull, 0x00002c00ull, "QMMA.16832.F32.E4M3.E4M3 %d, %a, %b, %c" },
    { 0x0000000000007308ull, 0x00000800ull, "MUFU.EX2 %d, %b" },
    { 0x0000000000007308ull, 0x00001000ull, "MUFU.RCP %d, %b" },
    { 0x0000000000007308ull, 0x00001400ull, "MUFU.RSQ %d, %b" },
    /* memory */
    { 0x0000000000007981ull, 0x0c1e1100ull, "LDG.E%z %d, desc[%B][%a.64%o]" },
    { 0x0000000000007986ull, 0x0c101100ull, "STG.E%z desc[%C][%a.64%o], %b" },
    { 0x0000000000007984ull, 0x08000800ull, "LDS %d, [%a+%B]" },
    { 0x0000000000007988ull, 0x08000800ull, "STS [%a+%C], %b" },
    { 0x0000000000007984ull, 0x00000000ull, "LDS%z %d, [%a%o]" },
    { 0x0000000000007992ull, 0x0000b000ull, "MEMBAR.ALL.SYS" },
    { 0x0800000000007f89ull, 0x000e0000ull, "SHFL.DOWN PT, %d, %a, %h, %m" },
    { 0x0c00000000007f89ull, 0x000e0000ull, "SHFL.BFLY PT, %d, %a, %h, %m" },
    /* control */
    { 0x0000000000007918ull, 0x00000000ull, "NOP" },
    { 0x000000000000794dull, 0x03800000ull, "EXIT" },
    { 0x0000000000007947ull, 0x03800000ull, "BRA %T" },
    { 0x0000000000007945ull, 0x03800200ull, "BSSY.RECONVERGENT %y, %Y" },
    { 0x0000000000007941ull, 0x03800200ull, "BSYNC.RECONVERGENT %y" },
    { 0x0000000000007b1dull, 0x00010000ull, "BAR.SYNC.DEFER_BLOCKING 0x0" },
};

/* IMAD with an immediate: nvdisasm names the special cases. Returns 0 for another instruction, 1 when printed,
 * -1 for a shift by a power of two whose name nvdisasm was never seen to print (so it is reported, not guessed). */
static int imad_imm(Out *w, uint64_t lo, uint64_t hi) {
    if (bits(lo, 0, 12) != 0x824 || (hi & HI_MASK & ~0xffull) != 0x078e0200ull) return 0;
    uint32_t imm = (uint32_t)bits(lo, 32, 32);
    uint64_t c = bits(hi, 0, 8);
    const char *name = "IMAD";
    if (imm == 0) name = "IMAD.MOV";                                  /* d = c */
    else if (imm == 1) name = c == 255 ? "IMAD.MOV" : "IMAD.IADD";    /* d = a, d = a + c */
    else if (c == 255 && (imm & (imm - 1)) == 0) {
        /* a shift. Seen in the corpus: .SHL for 2^1..2^6, 2^9, 2^10, 2^12, 2^20; plain IMAD for 2^16 and 2^31 */
        int n = __builtin_ctz(imm);
        if (n == 16 || n == 31) name = "IMAD";
        else if (n <= 6 || n == 9 || n == 10 || n == 12 || n == 20) name = "IMAD.SHL";
        else return -1;
    }
    put(w, "%s ", name);
    reg(w, "R", bits(lo, 16, 8)); put(w, ", "); reg(w, "R", bits(lo, 24, 8)); put(w, ", ");
    hex_s(w, (int32_t)imm); put(w, ", "); reg(w, "R", c);
    return 1;
}

int omega_sm121_disasm(uint64_t lo, uint64_t hi, uint64_t pc, char *out, size_t outn) {
    if (!out || outn < 40) return -2;
    Out w = { out, outn, 0, 1 };
    out[0] = 0;
    uint64_t g = bits(lo, 12, 3); int gneg = (int)bits(lo, 15, 1);
    if (!(g == 7 && !gneg)) { put(&w, "@"); pred(&w, g, gneg); put(&w, " "); }
    const uint64_t lo_body = lo & ~mask(12, 4), hi_body = hi & HI_MASK;
    size_t start = w.at;
    int im = imad_imm(&w, lo_body, hi_body);
    if (im > 0) return w.ok ? 0 : -2;
    for (size_t i = 0; im == 0 && i < sizeof ROWS / sizeof ROWS[0]; i++) {
        const Row *r = &ROWS[i];
        if ((lo_body & 0xfff) != (r->lov & 0xfff)) continue;
        uint64_t lom, him; fields_mask(r->fmt, &lom, &him);
        lom = ~(lom | mask(12, 4)); him = HI_MASK & ~him;
        if ((lo_body & lom) != (r->lov & lom) || (hi_body & him) != (r->hiv & him)) continue;
        w.at = start; render(&w, r->fmt, lo, hi, pc);
        return w.ok ? 0 : -2;
    }
    w.at = 0;
    put(&w, "?? 0x%016llx%016llx", (unsigned long long)hi, (unsigned long long)lo);
    return w.ok ? -1 : -2;
}

static uint64_t word_half(const uint8_t *w) { uint64_t v = 0; for (int i = 7; i >= 0; i--) v = v << 8 | w[i]; return v; }

size_t omega_sm121_padding_start(const uint8_t *code, size_t len) {
    size_t end = 0;
    for (size_t at = 0; code && at + 16 <= len; at += 16)
        if ((word_half(code + at) & 0xfff) == 0x94d) end = at + 16; /* EXIT, any guard */
    return end;
}

int omega_sm121_listing(const uint8_t *code, size_t len, FILE *out) {
    if (!code || !out || len % 16) return -1;
    size_t pad = omega_sm121_padding_start(code, len);
    int unknown = 0;
    for (size_t at = 0; at < len; at += 16) {
        char t[256];
        int rc = omega_sm121_disasm(word_half(code + at), word_half(code + at + 8), at, t, sizeof t);
        if (rc == -2) return -1;
        if (rc) unknown++;
        fprintf(out, "        /*%04llx*/ %s%s;\n", (unsigned long long)at, t, pad && at >= pad ? "" : " ");
    }
    return unknown;
}
