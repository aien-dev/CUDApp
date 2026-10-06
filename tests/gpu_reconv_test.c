/* gpu_reconv_test: structured warp reconvergence (omega #308) on Omega's own Blackwell codegen.
 *
 *   ./gpu_reconv_test --host-only            encoder golden words, region bookkeeping, codegen refusals,
 *                                            nvdisasm listing (when /usr/local/cuda/bin/nvdisasm exists),
 *                                            every probe kernel through the SIMT warp simulator against
 *                                            the host oracle under both fragment orders, the mutation
 *                                            controls, scoreboard and control-edge walks. No device.
 *   ./gpu_reconv_test --out receipt.json     chip gate: host part first (a kernel the simulator rejects
 *                                            never gets chip time), then every probe on the chip against
 *                                            the oracle, bit-exact, then each math mutant must be caught.
 *   --observe-dropped-join                   chip only, optional: also launch the dropped-BSYNC mutant of
 *                                            one probe and record what the chip did (an observation, not
 *                                            a gate; the simulator rejects that kernel first).
 *
 * Probes (src/omega_gpu_elementwise_api.c, gen_rc_*): R3 diamond, R4 loop break, R5 nested,
 * R6 lanes past the end with STS/BAR/LDS + SHFL, and the 16-deep barrier-register bound.
 * Oracle: rc_oracle below; every comparison is exact (u32). Launcher chunking (64 CTAs x 128
 * threads per launch, P0 = words in the chunk) is part of the oracle, so a clamp to the wrong
 * chunk end shows up. */
#include "omega_gpu_elementwise_api.h"
#include "omega_blackwell_codegen.h"
#include "omega_bw_reconv.h"
#include "omega_gpu_session.h"
#include "bw_warp_sim.h"
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int g_checks, g_failed;
#define CHECK(cond, ...) do { g_checks++; if (!(cond)) { g_failed++; printf("FAIL: "); printf(__VA_ARGS__); printf("\n"); } } while (0)

static uint32_t lcg(uint32_t *s) { *s = *s * 1664525u + 1013904223u; return *s; }

typedef struct { OmegaGpuEwOp op; const char *name; const char *reg; } Probe;
static const Probe PROBES[] = {
    { OMEGA_GPU_EW_RC_DIAMOND, "rc_diamond", "R3" },
    { OMEGA_GPU_EW_RC_LOOPBREAK, "rc_loopbreak", "R4" },
    { OMEGA_GPU_EW_RC_NESTED, "rc_nested", "R5" },
    { OMEGA_GPU_EW_RC_EXIT_BAR, "rc_exit_bar", "R6" },
    { OMEGA_GPU_EW_RC_DEPTH16, "rc_depth16", "B0..B15" },
};
#define NPROBES (sizeof PROBES / sizeof PROBES[0])
static const uint32_t NS[] = { 1, 5, 31, 32, 33, 63, 64, 65, 96, 127, 128, 129, 200, 1000, 4096, 8191, 8192, 8193, 20000 };
#define NNS (sizeof NS / sizeof NS[0])
#define CHUNK (OMEGA_GPU_EW_MAX_CTAS * OMEGA_GPU_EW_THREADS)

/* ----------------------------------------------------------------- oracle */
static uint32_t count_loop(uint32_t x, uint32_t y, int math, uint32_t *v_out) {
    uint32_t v = x & 255u, st = (y & 15u) + 1u, cnt = 0, bound = math ? 255u : 256u;
    while (v < bound) { v += st; cnt++; }
    *v_out = v; return cnt;
}
static uint32_t rc_r(OmegaGpuEwOp op, int math, uint32_t x, uint32_t y, uint32_t partner_x) {
    uint32_t v, cnt;
    switch (op) {
    case OMEGA_GPU_EW_RC_DIAMOND:
        if (x >= y) { uint32_t t = 3u * x + y, u = t ^ x, r = 5u * u + 1u; r += r >> 3; return math ? r + 1u : r; }
        return x + 1u;
    case OMEGA_GPU_EW_RC_LOOPBREAK: cnt = count_loop(x, y, math, &v); return cnt * 256u + v;
    case OMEGA_GPU_EW_RC_NESTED: if (x & 1u) return 7u; cnt = count_loop(x, y, math, &v); return cnt * 256u + v + 100u;
    case OMEGA_GPU_EW_RC_EXIT_BAR: return 3u * x + (partner_x | 1u);
    case OMEGA_GPU_EW_RC_DEPTH16: return x + (x < y ? (uint32_t)BW_RECONV_MAX_BAR : 0u);
    default: return 0xdeadbeefu;
    }
}
static void rc_oracle(OmegaGpuEwOp op, int math, uint32_t n, const uint32_t *a, const uint32_t *b, uint32_t *out) {
    for (uint32_t i0 = 0; i0 < n; i0 += CHUNK) {
        uint32_t ni = n - i0 < CHUNK ? n - i0 : CHUNK;
        uint32_t threads = ((ni + OMEGA_GPU_EW_THREADS - 1) / OMEGA_GPU_EW_THREADS) * OMEGA_GPU_EW_THREADS;
        uint32_t *r = malloc((size_t)threads * 4);
        for (uint32_t t = 0; t < threads; t++) {
            uint32_t idx = t < ni ? t : ni - 1, pt = t ^ 1u, pidx = pt < ni ? pt : ni - 1;
            r[t] = rc_r(op, math, a[i0 + idx], b[i0 + idx], a[i0 + pidx]);
        }
        for (uint32_t i = 0; i < ni; i++) { uint32_t nb = (i & 31u) == 31u ? i : i + 1u; out[i0 + i] = r[i] + 4099u * r[nb]; }
        free(r);
    }
}
/* An LCG's low bits cycle quickly (bit 0 has period 2), and the probes key on low bits
 * (x & 255, y & 15, x & 1), so each draw goes through a mixing step. */
static uint32_t mix(uint32_t r) { r ^= r >> 16; r *= 0x45d9f3bu; r ^= r >> 16; r *= 0x45d9f3bu; r ^= r >> 16; return r; }
static void fill(uint32_t *a, uint32_t *b, uint32_t n, uint32_t seed) {
    for (uint32_t i = 0; i < n; i++) { a[i] = mix(lcg(&seed)); b[i] = mix(lcg(&seed)); }
}
static size_t diff_count(const uint32_t *got, const uint32_t *want, uint32_t n, uint32_t *first) {
    size_t bad = 0; *first = n;
    for (uint32_t i = 0; i < n; i++) if (got[i] != want[i]) { if (!bad) *first = i; bad++; }
    return bad;
}

/* --------------------------------------------------------- simulator driver */
static uint8_t g_smem[1024];
/* Mirrors run_1d: 64 CTAs x 128 threads per launch, P0 = words in the chunk. Returns the
 * simulator's error code (0 clean); the output words are written to out. */
static int ws_probe(const BlackwellIRProgram *p, uint32_t n, const uint32_t *a, const uint32_t *b, uint32_t *out, int order, WsSim *stats) {
    for (uint32_t i = 0; i < n; i++) out[i] = 0xffbadbadu;
    uint64_t wi = 0, li = 0, sp = 0; int mf = 0, pb = 0;
    for (uint32_t i0 = 0; i0 < n; i0 += CHUNK) {
        uint32_t ni = n - i0 < CHUNK ? n - i0 : CHUNK;
        WsSim s; memset(&s, 0, sizeof s);
        s.p = p; s.threads = OMEGA_GPU_EW_THREADS; s.smem = g_smem; s.smem_bytes = sizeof g_smem; s.order = order;
        uint64_t aa = (uintptr_t)(a + i0), ba = (uintptr_t)(b + i0), ca = (uintptr_t)(out + i0);
        s.cbank[0x380 / 4] = (uint32_t)aa; s.cbank[0x380 / 4 + 1] = (uint32_t)(aa >> 32);
        s.cbank[0x388 / 4] = (uint32_t)ba; s.cbank[0x388 / 4 + 1] = (uint32_t)(ba >> 32);
        s.cbank[0x390 / 4] = (uint32_t)ca; s.cbank[0x390 / 4 + 1] = (uint32_t)(ca >> 32);
        s.cbank[0x398 / 4] = ni;
        s.ranges[0] = (WsRange){ (uintptr_t)aa, (uintptr_t)aa + (size_t)ni * 4 };
        s.ranges[1] = (WsRange){ (uintptr_t)ba, (uintptr_t)ba + (size_t)ni * 4 };
        s.ranges[2] = (WsRange){ (uintptr_t)ca, (uintptr_t)ca + (size_t)ni * 4 };
        s.nranges = 3;
        int err = ws_run(&s, (ni + OMEGA_GPU_EW_THREADS - 1) / OMEGA_GPU_EW_THREADS, 1);
        wi += s.warp_issues; li += s.lane_issues; sp += s.splits;
        if (s.max_frags > mf) mf = s.max_frags;
        if (s.peak_bars > pb) pb = s.peak_bars;
        if (err) { if (stats) *stats = s; return err; }
    }
    if (stats) { memset(stats, 0, sizeof *stats); stats->warp_issues = wi; stats->lane_issues = li; stats->splits = sp; stats->max_frags = mf; stats->peak_bars = pb; }
    return 0;
}
static int codegen(OmegaGpuEwOp op, int mutant, BlackwellIRProgram *p, OmegaBlackwellKernel *k) {
    memset(k, 0, sizeof *k);
    return omega_gpu_elementwise_codegen_ir(op, mutant, p, k);
}
static int is_reconv_err(int e) {
    return e == WS_E_SHFL_SPLIT || e == WS_E_BAR_SPLIT || e == WS_E_STUCK || e == WS_E_NEVER_JOINED || e == WS_E_JOIN_MISMATCH ||
           e == WS_E_BSYNC_UNARMED || e == WS_E_BSSY_REARM || e == WS_E_NONMEMBER || e == WS_E_SHFL_EXITED;
}

/* --------------------------------------------------------- static walks */
/* Scoreboard walk (as bench/prime_race/tests/gb10_sieve_host_test.c sb_hazards): a destination
 * written under write scoreboard 0 is in flight until a wait mask that includes it. */
static int sb_hazards(const BlackwellIRProgram *p) {
    int inflight[64], n = 0, bad = 0;
    for (size_t i = 0; i < p->count; i++) {
        const BlackwellIRInsn *in = &p->insns[i];
        uint32_t c = in->control;
        if ((c >> 20) & 1u) n = 0;
        const int src[3] = { in->src1_vreg, in->src2_vreg, in->src3_vreg };
        for (int s = 0; s < 3; s++) for (int k = 0; src[s] >= 0 && k < n; k++) if (inflight[k] == src[s]) bad++;
        if (c != 0 && ((c >> 14) & 7u) == 0 && in->dst_vreg >= 0 && n < 64) inflight[n++] = in->dst_vreg;
    }
    return bad;
}
/* Control-edge walk: the straight-line scoreboard reasoning above holds only if nothing is in
 * flight across a branch, barrier op or exit. Every BRA / EXIT / BSSY / BSYNC must wait on every
 * scoreboard set since the last wait. Returns the number of control ops that do not. */
static int edge_waits(const BlackwellIRProgram *p) {
    uint32_t pending = 0; int bad = 0;
    for (size_t i = 0; i < p->count; i++) {
        const BlackwellIRInsn *in = &p->insns[i];
        uint32_t c = in->control, wait = (c >> 20) & 63u, wb = (c >> 14) & 7u, rb = (c >> 17) & 7u;
        pending &= ~wait;
        if (in->op == BW_IR_BRA || in->op == BW_IR_EXIT || in->op == BW_IR_BSSY || in->op == BW_IR_BSYNC) { if (pending) bad++; }
        if (wb != 7u) pending |= 1u << wb;
        if (rb != 7u) pending |= 1u << rb;
    }
    return bad;
}
static int count_op(const BlackwellIRProgram *p, BlackwellIROpcode op, int bar) {
    int n = 0;
    for (size_t i = 0; i < p->count; i++) if (p->insns[i].op == op && (bar < 0 || p->insns[i].bar_reg == bar)) n++;
    return n;
}
static int find_op(const BlackwellIRProgram *p, BlackwellIROpcode op, int nth) {
    for (size_t i = 0; i < p->count; i++) if (p->insns[i].op == op && nth-- == 0) return (int)i;
    return -1;
}

/* --------------------------------------------------------- nvdisasm listing */
static int nvdisasm_bytes(const uint8_t *code, size_t bytes, const char *name, const char *const *must, int n_must) {
    const char *nv = "/usr/local/cuda/bin/nvdisasm";
    if (access(nv, X_OK) != 0) { printf("nvdisasm: not present, listing check skipped for %s\n", name); return 0; }
    char bin[96], lst[96], cmd[320];
    snprintf(bin, sizeof bin, "/tmp/omega_rc_%s_%d.bin", name, (int)getpid());
    snprintf(lst, sizeof lst, "/tmp/omega_rc_%s_%d.sass", name, (int)getpid());
    FILE *f = fopen(bin, "wb"); if (!f) return -1;
    fwrite(code, 1, bytes, f); fclose(f);
    snprintf(cmd, sizeof cmd, "%s -b SM121 %s > %s 2>&1", nv, bin, lst);
    int rc = system(cmd);
    FILE *l = fopen(lst, "r"); if (!l) return -1;
    char line[512]; int seen[16] = {0}, bad = 0, insns = 0;
    while (fgets(line, sizeof line, l)) {
        if (!strstr(line, "/*0")) continue;
        insns++;
        if (strstr(line, "?") || strstr(line, "error")) bad++;
        for (int i = 0; i < n_must && i < 16; i++) if (strstr(line, must[i])) seen[i] = 1;
    }
    fclose(l);
    int miss = 0;
    for (int i = 0; i < n_must && i < 16; i++) if (!seen[i]) { printf("  %s: nvdisasm listing lacks %s\n", name, must[i]); miss++; }
    printf("nvdisasm %s: rc=%d insns=%d undecodable=%d missing=%d (%s)\n", name, rc, insns, bad, miss, lst);
    unlink(bin);
    return (rc == 0 && bad == 0 && miss == 0) ? 0 : -1;
}
static int nvdisasm_kernel(OmegaGpuEwOp op, int mutant, const char *name, const char *const *must, int n_must) {
    BlackwellIRProgram *p = calloc(1, sizeof *p); OmegaBlackwellKernel k;
    if (!p || codegen(op, mutant, p, &k) != OMEGA_GPU_EW_OK) { free(p); return -1; }
    int rc = nvdisasm_bytes(k.code, k.code_size, name, must, n_must);
    free(k.code); free(p);
    return rc;
}

/* --------------------------------------------------------- host part */
typedef struct { const char *name; uint64_t warp_issues, lane_issues, splits; int max_frags, peak_bars; } Model;
static Model g_model[NPROBES * 2]; static int g_nmodel;

static int host_part(void) {
    int before = g_failed;
    /* 1. encoder golden words (nvcc 13.0.88 -arch=sm_121 oracle, docs/gpu-reconvergence-308.md) */
    int fx = omega_blackwell_verify_codegen_fixtures_reconv();
    CHECK(fx == 0, "encoder golden words BSSY / BSYNC and refusals (rc %d)", fx);

    /* 2. region bookkeeping on a hand-built program */
    {
        BlackwellIRProgram *p = calloc(1, sizeof *p); BwRegions r; memset(&r, 0, sizeof r);
        omega_bw_ir_init(p);
        BlackwellIRInsn in; memset(&in, 0, sizeof in); in.dst_vreg = in.src1_vreg = in.src2_vreg = in.src3_vreg = -1;
        in.op = BW_IR_BSSY; in.bar_reg = 0; in.imm = 2; omega_bw_ir_append(p, &in);          /* 0 */
        CHECK(bw_regions_next_bar(&r) == 0 && bw_regions_begin(&r, 0) == 0, "region begin returns B0");
        in.op = BW_IR_BRA; in.imm = 1; in.predicate_p0 = true; omega_bw_ir_append(p, &in);   /* 1 */
        CHECK(bw_regions_exit(&r, 1) == 0, "region exit recorded");
        in.op = BW_IR_NOP; in.predicate_p0 = false; omega_bw_ir_append(p, &in);              /* 2 */
        in.op = BW_IR_BSYNC; in.bar_reg = 0; omega_bw_ir_append(p, &in);                      /* 3 */
        CHECK(bw_regions_join(&r, p, 3) == 0, "region join returns B0");
        CHECK(p->insns[0].imm == 4 && p->insns[1].imm == 2, "join patched BSSY to the insn after BSYNC (%u) and the exit to the BSYNC (%u)", p->insns[0].imm, p->insns[1].imm);
        CHECK(bw_regions_join(&r, p, 3) == -1, "join with no open region refused");
        CHECK(bw_regions_exit(&r, 1) == -1, "exit with no open region refused");
        /* exits table full */
        memset(&r, 0, sizeof r); bw_regions_begin(&r, 0);
        int ok = 1; for (int i = 0; i < BW_RECONV_MAX_EXITS; i++) ok &= bw_regions_exit(&r, 1) == 0;
        CHECK(ok && bw_regions_exit(&r, 1) == -1, "exit table bound (%d) fails closed", BW_RECONV_MAX_EXITS);
        /* join index inconsistent: BSYNC before BSSY, wrong barrier */
        memset(&r, 0, sizeof r); bw_regions_begin(&r, 3);
        CHECK(bw_regions_join(&r, p, 0) == -1, "join with BSSY after BSYNC refused");
        memset(&r, 0, sizeof r); bw_regions_begin(&r, 0); p->insns[3].bar_reg = 1;
        CHECK(bw_regions_join(&r, p, 3) == -1, "join with the wrong barrier register refused");
        p->insns[3].bar_reg = 0;
        /* depth bound */
        memset(&r, 0, sizeof r); ok = 1;
        for (int d = 0; d < BW_RECONV_MAX_DEPTH; d++) ok &= bw_regions_next_bar(&r) == d && bw_regions_begin(&r, d) == d;
        CHECK(ok && bw_regions_next_bar(&r) == -1 && bw_regions_begin(&r, 99) == -1, "nesting bound %d: the next begin fails closed", BW_RECONV_MAX_DEPTH);
        free(p);
    }

    /* 3. codegen: production probes build, refusals refuse, mutants differ */
    static const struct { OmegaGpuEwOp op; const char *name; } refuse[] = {
        { OMEGA_GPU_EW_RC_REFUSE_DEPTH17, "17 nested regions" }, { OMEGA_GPU_EW_RC_REFUSE_EXIT_SHFL, "EXIT then BAR/SHFL" },
        { OMEGA_GPU_EW_RC_REFUSE_EXIT_IN_REGION, "EXIT inside a region" }, { OMEGA_GPU_EW_RC_REFUSE_BAR_IN_REGION, "BAR.SYNC inside a region" },
        { OMEGA_GPU_EW_RC_REFUSE_SHFL_IN_REGION, "SHFL inside a region" }, { OMEGA_GPU_EW_RC_REFUSE_UNCLOSED, "region never joined" },
        { OMEGA_GPU_EW_RC_REFUSE_JOIN_ONLY, "BSYNC with no BSSY" },
    };
    for (size_t i = 0; i < sizeof refuse / sizeof refuse[0]; i++) {
        BlackwellIRProgram *p = calloc(1, sizeof *p); OmegaBlackwellKernel k;
        int rc = codegen(refuse[i].op, 0, p, &k);
        CHECK(rc == OMEGA_GPU_EW_CODEGEN_FAIL, "%s refused at codegen (rc %s)", refuse[i].name, omega_gpu_elementwise_rc_name(rc));
        if (rc == OMEGA_GPU_EW_OK) free(k.code);
        free(p);
    }
    for (size_t i = 0; i < NPROBES; i++) {
        BlackwellIRProgram *p = calloc(1, sizeof *p); OmegaBlackwellKernel k;
        int rc = codegen(PROBES[i].op, 0, p, &k);
        CHECK(rc == OMEGA_GPU_EW_OK && k.gpr_count <= 64, "%s builds (rc %s, %zu insns, %u gprs)", PROBES[i].name, omega_gpu_elementwise_rc_name(rc), k.insn_count, k.gpr_count);
        if (rc != OMEGA_GPU_EW_OK) { free(p); continue; }
        int nb = count_op(p, BW_IR_BSSY, -1), ns = count_op(p, BW_IR_BSYNC, -1);
        int want = PROBES[i].op == OMEGA_GPU_EW_RC_NESTED ? 2 : PROBES[i].op == OMEGA_GPU_EW_RC_DEPTH16 ? BW_RECONV_MAX_BAR : 1;
        CHECK(nb == want && ns == want, "%s has %d BSSY and %d BSYNC (want %d)", PROBES[i].name, nb, ns, want);
        if (PROBES[i].op == OMEGA_GPU_EW_RC_DEPTH16) {
            int each = 1; for (int b = 0; b < BW_RECONV_MAX_BAR; b++) each &= count_op(p, BW_IR_BSSY, b) == 1 && count_op(p, BW_IR_BSYNC, b) == 1;
            CHECK(each, "depth16 uses every barrier register B0..B15 exactly once");
        }
        CHECK(sb_hazards(p) == 0, "%s scoreboard walk clean", PROBES[i].name);
        CHECK(edge_waits(p) == 0, "%s every control op waits on everything in flight", PROBES[i].name);
        printf("codegen %s: %zu insns, %u gprs, %d regions\n", PROBES[i].name, p->count, k.gpr_count, nb);
        uint8_t d0[32]; memcpy(d0, k.code_digest, 32); free(k.code);
        for (int m = 1; m <= 3; m++) {
            BlackwellIRProgram *mp = calloc(1, sizeof *mp); OmegaBlackwellKernel mk;
            int mrc = codegen(PROBES[i].op, m, mp, &mk);
            int same_ok = m == 1 && PROBES[i].op == OMEGA_GPU_EW_RC_DEPTH16; /* depth16 has no math mutant */
            CHECK(mrc == OMEGA_GPU_EW_OK && ((memcmp(d0, mk.code_digest, 32) != 0) != same_ok), "%s mutant %d builds and %s (rc %s)", PROBES[i].name, m, same_ok ? "equals production" : "differs", omega_gpu_elementwise_rc_name(mrc));
            if (mrc == OMEGA_GPU_EW_OK) {
                if (m == 2) CHECK(count_op(mp, BW_IR_BSYNC, -1) == 0 && count_op(mp, BW_IR_BSSY, -1) == nb, "%s mutant 2 has no BSYNC", PROBES[i].name);
                if (m == 3) CHECK(count_op(mp, BW_IR_BSYNC, -1) == ns, "%s mutant 3 keeps its BSYNCs", PROBES[i].name);
                free(mk.code);
            }
            free(mp);
        }
        free(p);
    }
    /* the four regioned sieves build too (their sweep runs in gb10_sieve_host_test) */
    for (int op = OMEGA_GPU_EW_PRIME_SIEVE_DIV1; op <= OMEGA_GPU_EW_PRIME_SIEVE_SHARED_DIV2; op++) {
        BlackwellIRProgram *p = calloc(1, sizeof *p); OmegaBlackwellKernel k;
        int rc = codegen((OmegaGpuEwOp)op, 0, p, &k);
        int staged = op == OMEGA_GPU_EW_PRIME_SIEVE_SHARED_DIV1 || op == OMEGA_GPU_EW_PRIME_SIEVE_SHARED_DIV2;
        int nb = rc == OMEGA_GPU_EW_OK ? count_op(p, BW_IR_BSSY, -1) : -1, want = (op >= OMEGA_GPU_EW_PRIME_SIEVE_DIV2 ? 2 : 1) + staged;
        CHECK(rc == OMEGA_GPU_EW_OK && k.gpr_count <= 64 && nb == want && edge_waits(p) == 0, "sieve op %d builds with %d regions (rc %s, %u gprs)", op, nb, omega_gpu_elementwise_rc_name(rc), k.gpr_count);
        if (rc == OMEGA_GPU_EW_OK) free(k.code);
        free(p);
    }

    /* 4. nvdisasm: the words decode as BSSY/BSYNC on the real barrier registers, and the predicated
     * and B15 forms the encoder can produce decode too (built by hand: encode_one) */
    {
        static const char *const m0[] = { "BSSY", "BSYNC", "B0", "SHFL.DOWN", "@P0 BRA" };
        CHECK(nvdisasm_kernel(OMEGA_GPU_EW_RC_DIAMOND, 0, "diamond", m0, 5) == 0, "diamond listing decodes with BSSY/BSYNC B0");
        static const char *const m1[] = { "BSSY", "BSYNC", "B15", "B7" };
        CHECK(nvdisasm_kernel(OMEGA_GPU_EW_RC_DEPTH16, 0, "depth16", m1, 4) == 0, "depth16 listing decodes with B15");
        static const char *const m2[] = { "BSSY", "B1", "BSYNC" };
        CHECK(nvdisasm_kernel(OMEGA_GPU_EW_PRIME_SIEVE_DIV2, 0, "sieve_div2", m2, 3) == 0, "nested sieve listing decodes with B1");
        uint8_t code[64]; uint32_t w[4]; BlackwellIRInsn in; memset(&in, 0, sizeof in); in.dst_vreg = in.src1_vreg = in.src2_vreg = in.src3_vreg = -1;
        in.op = BW_IR_BSSY; in.bar_reg = 3; in.imm = 2; in.predicate_p0 = true; omega_blackwell_encode_one(&in, w); memcpy(code, w, 16);
        in.op = BW_IR_BSYNC; in.bar_reg = 3; in.predicate_p0 = true; in.predicate_not = true; omega_blackwell_encode_one(&in, w); memcpy(code + 16, w, 16);
        in.op = BW_IR_BSYNC; in.bar_reg = 15; in.predicate_p0 = false; in.predicate_not = false; omega_blackwell_encode_one(&in, w); memcpy(code + 32, w, 16);
        in.op = BW_IR_EXIT; omega_blackwell_encode_one(&in, w); memcpy(code + 48, w, 16);
        static const char *const m3[] = { "@P0 BSSY", "B3", "@!P0 BSYNC", "BSYNC.RECONVERGENT B15" };
        CHECK(nvdisasm_bytes(code, 64, "forms", m3, 4) == 0, "predicated BSSY/BSYNC and B15 decode");
    }

    /* 5. simulator: every probe, every n, both fragment orders, exact against the oracle */
    uint32_t *a = malloc(20000 * 4), *b = malloc(20000 * 4), *got = malloc(20000 * 4), *want = malloc(20000 * 4);
    for (size_t i = 0; i < NPROBES; i++) {
        BlackwellIRProgram *p = calloc(1, sizeof *p); OmegaBlackwellKernel k;
        if (codegen(PROBES[i].op, 0, p, &k) != OMEGA_GPU_EW_OK) { free(p); continue; }
        int fails = 0; uint64_t cases = 0;
        for (size_t j = 0; j < NNS; j++) for (int order = 0; order < 2; order++) {
            uint32_t n = NS[j]; fill(a, b, n, 0x308u + n * 7u + (uint32_t)order);
            rc_oracle(PROBES[i].op, 0, n, a, b, want);
            WsSim st; int err = ws_probe(p, n, a, b, got, order, &st);
            uint32_t first; size_t bad = diff_count(got, want, n, &first);
            cases++;
            if (err || bad) { fails++; printf("  %s n=%u order=%d: sim %s, %zu words differ (first %u: got %08x want %08x)\n", PROBES[i].name, n, order, err ? st.msg : "clean", bad, first, first < n ? got[first] : 0, first < n ? want[first] : 0); }
            if (n == 4096) { Model *m = &g_model[g_nmodel++]; m->name = PROBES[i].name; m->warp_issues = st.warp_issues; m->lane_issues = st.lane_issues; m->splits = st.splits; m->max_frags = st.max_frags; m->peak_bars = st.peak_bars; }
        }
        CHECK(fails == 0, "%s simulator vs oracle: %d of %" PRIu64 " cases differ", PROBES[i].name, fails, cases);
        /* math mutant: simulator clean, result wrong (the oracle sees it) */
        BlackwellIRProgram *mp = calloc(1, sizeof *mp); OmegaBlackwellKernel mk;
        if (codegen(PROBES[i].op, 1, mp, &mk) == OMEGA_GPU_EW_OK) {
            uint32_t n = 1000; fill(a, b, n, 0x1001u); rc_oracle(PROBES[i].op, 0, n, a, b, want);
            WsSim st; int err = ws_probe(mp, n, a, b, got, 0, &st); uint32_t first; size_t bad = diff_count(got, want, n, &first);
            if (PROBES[i].op == OMEGA_GPU_EW_RC_EXIT_BAR) CHECK(err == WS_E_SHFL_EXITED || err == WS_E_BAR_SPLIT, "R6 early-exit kernel: simulator reports SHFL_EXITED (got %s)", st.msg);
            else if (PROBES[i].op == OMEGA_GPU_EW_RC_DEPTH16) CHECK(err == 0 && bad == 0, "depth16 'mutant 1' is the production kernel: clean and exact (err %d bad %zu)", err, bad);
            else CHECK(err == 0 && bad > 0, "%s math mutant: simulator clean, %zu words differ (err %d)", PROBES[i].name, bad, err);
            free(mk.code);
        }
        free(mp);
        /* dropped join and corrupted target: the simulator must reject both, in both orders */
        for (int m = 2; m <= 3; m++) for (int order = 0; order < 2; order++) {
            BlackwellIRProgram *xp = calloc(1, sizeof *xp); OmegaBlackwellKernel xk;
            if (codegen(PROBES[i].op, m, xp, &xk) == OMEGA_GPU_EW_OK) {
                uint32_t n = 1000; fill(a, b, n, 0x2002u + (uint32_t)m);
                WsSim st; int err = ws_probe(xp, n, a, b, got, order, &st);
                CHECK(is_reconv_err(err), "%s mutant %d (%s) order %d rejected by the simulator: %s", PROBES[i].name, m, m == 2 ? "BSYNC dropped" : "BSSY target corrupted", order, err ? st.msg : "NOT CAUGHT");
                if (m == 3) CHECK(err == WS_E_JOIN_MISMATCH || err == WS_E_STUCK, "%s corrupted target shows as JOIN_MISMATCH or STUCK (%s)", PROBES[i].name, ws_err_name(err));
                free(xk.code);
            }
            free(xp);
        }
        free(k.code); free(p);
    }
    /* 6. hand mutations of the production diamond (controls on the simulator itself, independent of
     * the emitter's mutant hooks): exit jumping past the BSYNC, BSYNC on the wrong barrier, a BSSY
     * on an armed barrier. And the unmodified program as the positive control. */
    {
        BlackwellIRProgram *p = calloc(1, sizeof *p), *q = calloc(1, sizeof *q); OmegaBlackwellKernel k;
        if (codegen(OMEGA_GPU_EW_RC_DIAMOND, 0, p, &k) == OMEGA_GPU_EW_OK) {
            uint32_t n = 500; fill(a, b, n, 0x3003u); WsSim st;
            int bssy = find_op(p, BW_IR_BSSY, 0), bsync = find_op(p, BW_IR_BSYNC, 0), ex = -1;
            for (int i = bssy + 1; i < bsync; i++) if (p->insns[i].op == BW_IR_BRA && !p->insns[i].predicate_p0) ex = i;
            CHECK(bssy >= 0 && bsync > bssy && ex > bssy, "diamond shape: BSSY %d, exit %d, BSYNC %d", bssy, ex, bsync);
            CHECK(ws_probe(p, n, a, b, got, 0, &st) == 0, "positive control: untouched diamond runs clean");
            *q = *p; q->insns[ex].imm += 1;                       /* exit lands after the BSYNC */
            int e1 = ws_probe(q, n, a, b, got, 0, &st); CHECK(is_reconv_err(e1), "exit past the BSYNC rejected: %s", st.msg);
            *q = *p; q->insns[bsync].bar_reg = 1;                 /* BSYNC on a barrier nobody armed */
            int e2 = ws_probe(q, n, a, b, got, 0, &st); CHECK(e2 == WS_E_BSYNC_UNARMED, "BSYNC on an unarmed barrier rejected: %s", st.msg);
            *q = *p; q->insns[bsync].op = BW_IR_NOP;              /* join dropped by hand */
            int e3 = ws_probe(q, n, a, b, got, 0, &st); CHECK(is_reconv_err(e3), "join replaced by NOP rejected: %s", st.msg);
            *q = *p; q->insns[bssy].imm += 1;                     /* names the wrong join */
            int e4 = ws_probe(q, n, a, b, got, 0, &st); CHECK(e4 == WS_E_JOIN_MISMATCH, "wrong join target rejected: %s", st.msg);
            *q = *p; q->insns[bssy + 1] = p->insns[bssy];         /* second BSSY on the armed barrier (overwrites the ISETP) */
            int e5 = ws_probe(q, n, a, b, got, 0, &st); CHECK(e5 == WS_E_BSSY_REARM, "BSSY on an armed barrier rejected: %s", st.msg);
            free(k.code);
        }
        free(p); free(q);
    }
    free(a); free(b); free(got); free(want);
    for (int i = 0; i < g_nmodel; i++) printf("model %s n=4096: warp_issues=%" PRIu64 " lane_util=%.3f splits=%" PRIu64 " max_frags=%d peak_bars=%d\n", g_model[i].name, g_model[i].warp_issues, g_model[i].warp_issues ? (double)g_model[i].lane_issues / (32.0 * (double)g_model[i].warp_issues) : 0.0, g_model[i].splits, g_model[i].max_frags, g_model[i].peak_bars);
    return g_failed == before ? 0 : -1;
}

/* --------------------------------------------------------- chip part */
static int chip(const char *out_path, int observe) {
    FILE *out = out_path ? fopen(out_path, "w") : NULL;
    if (out_path && !out) { printf("cannot open %s\n", out_path); return 2; }
    if (out) fprintf(out, "{\"schema\":\"OMEGA_GPU_RECONV_V1\",\"issue\":308,\"comparison\":\"exact u32\",\"cases\":[");
    int first = 1;
    uint32_t *a = malloc(20000 * 4), *b = malloc(20000 * 4), *got = malloc(20000 * 4), *want = malloc(20000 * 4);
    for (size_t i = 0; i < NPROBES; i++) {
        for (size_t j = 0; j < NNS; j++) {
            uint32_t n = NS[j]; fill(a, b, n, 0x5050u + n);
            rc_oracle(PROBES[i].op, 0, n, a, b, want);
            OmegaGpuEwInfo info; memset(&info, 0, sizeof info);
            int rc = omega_gpu_reconv_probe_u32(PROBES[i].op, n, a, b, got, &info);
            uint32_t fi; size_t bad = rc == OMEGA_GPU_EW_OK ? diff_count(got, want, n, &fi) : n;
            int pass = rc == OMEGA_GPU_EW_OK && bad == 0;
            CHECK(pass, "%s n=%u chip rc=%s violations=%zu err=\"%s\"", PROBES[i].name, n, omega_gpu_elementwise_rc_name(rc), bad, omega_gpu_elementwise_last_error());
            if (out) fprintf(out, "%s{\"op\":\"%s\",\"regression\":\"%s\",\"n\":%u,\"rc\":\"%s\",\"violations\":%zu,\"pass\":%s,\"chip_elapsed_ns\":%" PRIu64 ",\"chip_calls\":%u,\"unwritten_words\":%u,\"cache_hit\":%s}",
                             first ? "" : ",", PROBES[i].name, PROBES[i].reg, n, omega_gpu_elementwise_rc_name(rc), bad, pass ? "true" : "false", info.elapsed_ns, info.chip_calls, info.unwritten_words, info.kernel_cache_hit ? "true" : "false");
            first = 0;
            if (!pass) { printf("STOP: %s failed on the chip at n=%u; no further launches of this probe\n", PROBES[i].name, n); break; }
        }
    }
    if (out) fprintf(out, "],\"mutants\":[");
    first = 1;
    for (size_t i = 0; i < NPROBES; i++) {
        if (PROBES[i].op == OMEGA_GPU_EW_RC_DEPTH16) continue; /* no math mutant */
        uint32_t n = 1000; fill(a, b, n, 0x6060u); rc_oracle(PROBES[i].op, 0, n, a, b, want);
        omega_gpu_elementwise_test_set_mutant((int)PROBES[i].op);
        /* R6's math mutant is the early-exit kernel: the emitter refuses it on the launch path
         * (the bypass exists only for codegen_ir), so the catch is CODEGEN_FAIL: no undefined
         * SHFL ever reaches the chip. */
        OmegaGpuEwInfo info; memset(&info, 0, sizeof info);
        int rc = omega_gpu_reconv_probe_u32(PROBES[i].op, n, a, b, got, &info);
        uint32_t fi; size_t bad = rc == OMEGA_GPU_EW_OK ? diff_count(got, want, n, &fi) : n;
        int caught = rc != OMEGA_GPU_EW_OK || bad > 0;
        omega_gpu_elementwise_test_set_mutant(0);
        CHECK(caught, "%s math mutant caught (rc %s, violations %zu)", PROBES[i].name, omega_gpu_elementwise_rc_name(rc), bad);
        if (PROBES[i].op == OMEGA_GPU_EW_RC_EXIT_BAR) CHECK(rc == OMEGA_GPU_EW_CODEGEN_FAIL, "R6 early-exit form refused at codegen on the launch path (rc %s)", omega_gpu_elementwise_rc_name(rc));
        if (out) fprintf(out, "%s{\"op\":\"%s\",\"mutant\":1,\"rc\":\"%s\",\"violations\":%zu,\"caught\":%s}", first ? "" : ",", PROBES[i].name, omega_gpu_elementwise_rc_name(rc), bad, caught ? "true" : "false");
        first = 0;
    }
    if (out) fprintf(out, "],\"observations\":[");
    if (observe) {
        /* The dropped-join kernel (rejected by the simulator above) on the chip once, small n:
         * recorded as an observation of what the hardware does without the BSYNC. Not a gate. */
        uint32_t n = 1000; fill(a, b, n, 0x7070u); rc_oracle(OMEGA_GPU_EW_RC_DIAMOND, 0, n, a, b, want);
        omega_gpu_elementwise_test_set_mutant_level(OMEGA_GPU_EW_RC_DIAMOND, 2);
        OmegaGpuEwInfo info; memset(&info, 0, sizeof info);
        int rc = omega_gpu_reconv_probe_u32(OMEGA_GPU_EW_RC_DIAMOND, n, a, b, got, &info);
        uint32_t fi; size_t bad = rc == OMEGA_GPU_EW_OK ? diff_count(got, want, n, &fi) : n;
        omega_gpu_elementwise_test_set_mutant(0);
        printf("observation: diamond with the BSYNC dropped, n=%u: rc=%s, %zu of %u words differ from the oracle (first %u)\n", n, omega_gpu_elementwise_rc_name(rc), bad, n, fi);
        if (out) fprintf(out, "{\"op\":\"rc_diamond\",\"mutant\":2,\"what\":\"BSYNC dropped\",\"n\":%u,\"rc\":\"%s\",\"violations\":%zu,\"first_bad\":%u,\"gate\":false}", n, omega_gpu_elementwise_rc_name(rc), bad, fi);
    }
    CHECK(omega_gpu_session_open_count() == 1, "device opened once for the battery (opens=%u)", omega_gpu_session_open_count());
    if (out) {
        fprintf(out, "],\"sim_model_n4096\":[");
        for (int i = 0; i < g_nmodel; i++) fprintf(out, "%s{\"op\":\"%s\",\"order\":%d,\"warp_issues\":%" PRIu64 ",\"lane_issues\":%" PRIu64 ",\"splits\":%" PRIu64 ",\"max_frags\":%d,\"peak_bars\":%d}", i ? "," : "", g_model[i].name, i & 1, g_model[i].warp_issues, g_model[i].lane_issues, g_model[i].splits, g_model[i].max_frags, g_model[i].peak_bars);
        fprintf(out, "],\"device_opens\":%u,\"checks\":%d,\"failed\":%d,\"verdict\":\"%s\"}\n", omega_gpu_session_open_count(), g_checks, g_failed, g_failed == 0 ? "OMEGA_GPU_RECONV_PASS" : "OMEGA_GPU_RECONV_FAIL");
        fclose(out);
    }
    free(a); free(b); free(got); free(want);
    return 0;
}

int main(int argc, char **argv) {
    const char *out_path = NULL; int host = 0, observe = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--host-only") == 0) host = 1;
        else if (strcmp(argv[i], "--observe-dropped-join") == 0) observe = 1;
        else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) out_path = argv[++i];
    }
    int hrc = host_part();
    if (!host) {
        if (hrc != 0) { printf("host part failed: no chip launches\nFAIL: %d checks, %d failed\n", g_checks, g_failed); return 1; }
        if (chip(out_path, observe) != 0) return 2;
    }
    printf("%s: %d checks, %d failed\n", g_failed ? "FAIL" : "PASS", g_checks, g_failed);
    return g_failed ? 1 : 0;
}
