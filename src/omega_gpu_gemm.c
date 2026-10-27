/* omega_gpu_gemm.c -- matrix multiply of any shape on the GB10 tensor cores (C3). See the header. */
#include "omega_gpu_gemm.h"
#include "omega_bw_em.h"
#include "omega_bw_em_q.h"
#include "omega_gpu_session.h"

#include <stdlib.h>
#include <string.h>

#define POISON 0x7fa00badu /* a signalling NaN no tensor-core op writes: a word still holding it was never stored */
#define TIMEOUT_MS 60000ull
#define FRAG 512u          /* bytes of one warp fragment: 32 lanes x 16 */

static uint32_t esz(OmegaGpuGemmType t) { return t == OMEGA_GPU_GEMM_BF16 ? 2u : 1u; }
static uint32_t kstep(OmegaGpuGemmType t) { return t == OMEGA_GPU_GEMM_BF16 ? 16u : 32u; }
static int pow124(uint32_t v) { return v == 1u || v == 2u || v == 4u; }

void omega_gpu_gemm_default_cfg(OmegaGpuGemmType type, OmegaGemmCfg *cfg) {
    memset(cfg, 0, sizeof *cfg);
    cfg->type = type; cfg->rt = 2; cfg->ct = 4; cfg->wm = 2; cfg->wn = 2; cfg->stall = type == OMEGA_GPU_GEMM_BF16 ? 4u : 2u;
}

int omega_gpu_gemm_cfg_ok(const OmegaGemmCfg *c) {
    return c && (c->type == OMEGA_GPU_GEMM_BF16 || c->type == OMEGA_GPU_GEMM_E4M3) && pow124(c->rt) && pow124(c->ct) &&
           pow124(c->wm) && pow124(c->wn) && c->stall >= 1u && c->stall <= 15u && c->mutant >= 0 && c->mutant <= 3 &&
           4u * (4u * c->rt + 2u * c->ct + c->rt * c->ct) <= 100u; /* registers: two operand sets + accumulators, R2..R127 in all */
}

/* ------------------------------------------------------------------ IR helpers */

typedef struct { int bssy, bra; } Region;
static Region skip_if_p0(Em *e) { Region r; r.bssy = bssy_fwd(e); r.bra = bra_if_p0_fwd(e); return r; }
static void region_end(Em *e, Region r) { int at = here(e); bsync(e); patch_bra(e, r.bra, at); patch_bra(e, r.bssy, at + 1); }
/* d = a - b */
static void isub(Em *e, int d, int a, int b) { imadi(e, d, b, 0, 0xffffffffu, a); }
/* d = 1 when a < b (both below 2^31), else 0 */
static void lt_bit(Em *e, int d, int a, int b) { isub(e, d, a, b); shr(e, d, d, 0, 31); }
static void and_r(Em *e, int d, int a, int b) { lop3(e, d, a, b, NOREG, 0xc0u); }
/* tensor-core op d = a * b[bsub..bsub+1] + c */
static void mma(Em *e, int fp8, int d, int a, int b, unsigned bsub, int c, uint32_t stall) {
    BlackwellIRInsn i = I0(BW_IR_HMMA_BF16);
    i.dst_vreg = d; i.src1_vreg = a; i.src2_vreg = b; i.src2_subreg = (uint8_t)bsub; i.src3_vreg = c;
    if (fp8) i.imm = OMEGA_QMMA_MARK;
    i.control = CW(stall, SB_NONE, SB_NONE, e->pending); e->pending = 0;
    if (omega_bw_ir_append(e->p, &i) < 0) e->err = -1;
}

/* ------------------------------------------------------------------ PACK_A / PACK_B
 * CTA (pair cx, row group cy), 128 threads; thread tid writes word tid of each of the F fragments of the block:
 * lane L = tid / 4, register q = tid % 4, g = L / 4, t = L % 4, r = elements per register (2 bf16, 4 e4m3).
 *   A fragment f = u * rt + i: row 16 (rt cy + i) + g + 8 (q & 1), k = 2 step cx + step u + r t + 4 r (q >> 1)
 *   B fragment f = j:          row 8 (ct cy + j) + g,             k = 2 step cx + step (q >> 1) + r t + 4 r (q & 1)
 * Element (row, k) of the source is at (row * row_stride + k * k_stride) * esz; past rows or kdim it is 0. */
static int gen_pack(Em *e, const OmegaGemmCfg *c, int isb) {
    const int fp8 = c->type == OMEGA_GPU_GEMM_E4M3;
    const uint32_t ks = kstep(c->type), r = 4u / esz(c->type), F = isb ? c->ct : 2u * c->rt;
    int udesc = UV64(e), psrc = V64(e), pdst = V64(e), R = V(e), Kd = V(e), sr = V(e), sk = V(e), ks2 = V(e);
    int tid = V(e), cx = V(e), cy = V(e), one = V(e), g = V(e), t = V(e), qa = V(e), qb = V(e), row0 = V(e), k0 = V(e);
    int tmp = V(e), pd = V64(e), row = V(e), vrow = V(e), rowoff = V(e), kx = V(e), v = V(e), off = V(e), word = V(e);
    int addr[4], val[4], mask[4], odd[4];
    for (unsigned x = 0; x < 4; x++) { addr[x] = V64(e); val[x] = V(e); mask[x] = V(e); odd[x] = V(e); }
    if (e->err) return -1;
    ldcu64(e, udesc, 0x358);
    ldc64(e, psrc, 0x380); ldc64(e, pdst, 0x388);
    ldc32(e, R, 0x390); ldc32(e, Kd, 0x394); ldc32(e, sr, 0x398); ldc32(e, sk, 0x39c); ldc32(e, ks2, 0x3a0);
    s2r(e, tid, BW_SR_TID_X); s2r(e, cx, BW_SR_CTAID_X); s2r(e, cy, BW_SR_CTAID_Y);
    movi(e, one, 1);
    shr(e, g, tid, 0, 4);
    shr(e, t, tid, 0, 2); andi(e, t, t, 0, 3);
    andi(e, qa, tid, 0, 1);
    shr(e, qb, tid, 0, 1); andi(e, qb, qb, 0, 1);
    if (!isb) {
        imadi(e, row0, cy, 0, 16u * c->rt, g); imadi(e, row0, qa, 0, 8u, row0);
        imadi(e, k0, cx, 0, 2u * ks, NOREG); imadi(e, k0, t, 0, r, k0); imadi(e, k0, qb, 0, 4u * r, k0);
    } else {
        imadi(e, row0, cy, 0, 8u * c->ct, g);
        imadi(e, k0, cx, 0, 2u * ks, NOREG); imadi(e, k0, t, 0, r, k0); imadi(e, k0, qb, 0, ks, k0); imadi(e, k0, qa, 0, 4u * r, k0);
    }
    /* this thread's word of fragment 0: ((cy * KS2 + cx) * F * 128 + tid) words */
    imadr(e, tmp, cy, ks2, cx); imadi(e, tmp, tmp, 0, F * 128u, tid); imadw(e, pd, tmp, 4u, pdst);
    for (uint32_t f = 0; f < F; f++) {
        const uint32_t drow = isb ? 8u * f : 16u * (f % c->rt), dk = isb ? 0u : ks * (f / c->rt);
        imadi(e, row, one, 0, drow, row0);
        lt_bit(e, vrow, row, R);
        imadr(e, rowoff, row, sr, NOREG);
        for (uint32_t x = 0; x < r; x++) {
            imadi(e, kx, one, 0, dk + x, k0);
            lt_bit(e, v, kx, Kd);
            and_r(e, v, v, vrow);
            imadr(e, off, kx, sk, rowoff);
            imadr(e, off, off, v, NOREG);                     /* outside: element 0, always readable, then masked */
            if (c->mutant == 2) movi(e, mask[x], 0xffffffffu); /* the red check: element 0 leaks in past the edges */
            else imadi(e, mask[x], v, 0, 0xffffffffu, NOREG);
            if (fp8) { andi(e, odd[x], off, 0, 1u); andi(e, off, off, 0, 0xfffffffeu); imadw(e, addr[x], off, 1u, psrc); }
            else imadw(e, addr[x], off, 2u, psrc);
        }
        for (uint32_t x = 0; x < r; x++) ldg_wide(e, BW_IR_LDG_E_U16, val[x], addr[x], 0, udesc, x == 0);
        for (uint32_t x = 0; x < r; x++) {
            if (fp8) {                                        /* the byte at off: low or high half of the u16 */
                andi(e, kx, val[x], 0, 0xffu);
                shr(e, v, val[x], 0, 8); andi(e, v, v, 0, 0xffu);
                isub(e, v, v, kx);
                imadr(e, val[x], v, odd[x], kx);
            }
            and_r(e, val[x], val[x], mask[x]);
        }
        imadi(e, word, val[0], 0, 1u, NOREG);
        for (uint32_t x = 1; x < r; x++) imadi(e, word, val[x], 0, 1u << (x * (32u / r)), word);
        stg(e, pd, f * FRAG, word, udesc);
    }
    tail(e);
    return e->err;
}

/* ------------------------------------------------------------------ MAIN */
static int gen_main(Em *e, const OmegaGemmCfg *c) {
    const int fp8 = c->type == OMEGA_GPU_GEMM_E4M3;
    const uint32_t RT = c->rt, CT = c->ct, FA = 2u * RT, FB = CT;
    int udesc = UV64(e), pa = V64(e), pb = V64(e), pc = V64(e), M = V(e), N = V(e), ks2 = V(e);
    int tid = V(e), cx = V(e), cy = V(e), warp = V(e), lane = V(e), g = V(e), t = V(e), one = V(e), zr = V(e);
    int rga = V(e), rgb = V(e), tmp = V(e), cnt = V(e);
    int acc[16], as[2][8], bs[2][4];
    for (uint32_t x = 0; x < RT * CT; x++) acc[x] = V128(e);
    for (unsigned s = 0; s < 2; s++) { for (uint32_t f = 0; f < FA; f++) as[s][f] = V128(e); for (uint32_t j = 0; j < FB; j++) bs[s][j] = V128(e); }
    if (e->err) return -1;
    ldcu64(e, udesc, 0x358);
    ldc64(e, pa, 0x380); ldc64(e, pb, 0x388); ldc64(e, pc, 0x390);
    ldc32(e, M, 0x398); ldc32(e, N, 0x39c); ldc32(e, ks2, 0x3a0);
    s2r(e, tid, BW_SR_TID_X); s2r(e, cx, BW_SR_CTAID_X); s2r(e, cy, BW_SR_CTAID_Y);
    movi(e, one, 1); movrz(e, zr);
    shr(e, warp, tid, 0, 5); andi(e, lane, tid, 0, 31u);
    /* warp (wm, wn) = (warp / WN, warp % WN); row groups rga = cy WM + wm, rgb = cx WN + wn */
    shr(e, tmp, warp, 0, (uint32_t)log2u(c->wn)); imadi(e, rga, cy, 0, c->wm, tmp);
    andi(e, tmp, warp, 0, c->wn - 1u); imadi(e, rgb, cx, 0, c->wn, tmp);
    imadr(e, tmp, rga, ks2, NOREG); imadw(e, pa, tmp, FA * FRAG, pa);
    imadr(e, tmp, rgb, ks2, NOREG); imadw(e, pb, tmp, FB * FRAG, pb);
    imadi(e, tmp, lane, 0, 16u, NOREG); imadw(e, pa, tmp, 1u, pa); imadw(e, pb, tmp, 1u, pb);
    for (uint32_t x = 0; x < RT * CT; x++) for (unsigned q = 0; q < 4; q++) { BlackwellIRInsn mz = I0(BW_IR_MOV_RZ); mz.dst_vreg = acc[x]; mz.dst_subreg = (uint8_t)q; emit(e, mz, K_FIXED); }
    shr(e, cnt, ks2, 0, 1);                                   /* loop iterations: two k pairs each */

    #define ISSUE(s, sb) do { \
        for (uint32_t f = 0; f < FA; f++) ldw_pf(e, BW_IR_LDG_E_128, as[s][f], pa, f * FRAG, udesc, (sb)); \
        for (uint32_t j = 0; j < FB; j++) ldw_pf(e, BW_IR_LDG_E_128, bs[s][j], pb, j * FRAG, udesc, (sb)); \
        imadw(e, pa, one, FA * FRAG, pa); imadw(e, pb, one, FB * FRAG, pb); } while (0)
    #define COMPUTE(s) do { \
        for (uint32_t u = 0; u < 2; u++) { \
            if (c->mutant == 1 && u == 1) break; \
            for (uint32_t i = 0; i < RT; i++) for (uint32_t j = 0; j < CT; j++) \
                mma(e, fp8, acc[i * CT + j], as[s][u * RT + i], bs[s][j], 2u * u, acc[i * CT + j], c->stall); } } while (0)

    ISSUE(0, 3u);
    const int top = here(e);
    loop_begin(e);
    ISSUE(1, 4u);
    consume(e, 3);
    COMPUTE(0);
    nop(e, 15);                                               /* the tensor ops have read set 0 before it is reloaded */
    ISSUE(0, 3u);                                             /* the last iteration reads one pair past the end: slack */
    consume(e, 4);
    COMPUTE(1);
    nop(e, 15);
    imadi(e, cnt, one, 0, 0xffffffffu, cnt);
    isetp_ge_u32(e, cnt, one);
    bra_if_p0(e, top);
    loop_end(e);
    #undef ISSUE
    #undef COMPUTE
    consume(e, 3);
    nop(e, 15); nop(e, 15); nop(e, 15); nop(e, 15);           /* the last results settle before the stores read them */

    /* stores: C[m][n] for m = 16 (RT rga + i) + 8 h + g, n = 8 (CT rgb + j) + 2 t + cc, when m < M and n < N */
    int mb = V(e), nb = V(e), m = V(e), n = V(e), vm = V(e), vn = V(e), rowoff = V(e), ca = V64(e);
    if (e->err) return -1;
    shr(e, g, lane, 0, 2); andi(e, t, lane, 0, 3u);
    imadi(e, mb, rga, 0, 16u * RT, g);
    imadi(e, tmp, t, 0, 2u, NOREG); imadi(e, nb, rgb, 0, 8u * CT, tmp);
    for (uint32_t i = 0; i < RT; i++) for (uint32_t h = 0; h < 2; h++) {
        imadi(e, m, one, 0, 16u * i + 8u * h, mb);
        lt_bit(e, vm, m, M);
        imadr(e, rowoff, m, N, NOREG);
        for (uint32_t j = 0; j < CT; j++) for (uint32_t cc = 0; cc < 2; cc++) {
            imadi(e, n, one, 0, 8u * j + cc, nb);
            lt_bit(e, vn, n, N);
            and_r(e, vn, vn, vm);
            isetp_ge_u32(e, zr, vn);                          /* P0: 0 >= ok, i.e. outside: skip */
            Region rg = skip_if_p0(e);
            iadd(e, tmp, rowoff, n); imadw(e, ca, tmp, 4u, pc);
            BlackwellIRInsn st = I0(BW_IR_STG_E); st.src1_vreg = ca; st.src2_vreg = acc[i * CT + j];
            st.src2_subreg = (uint8_t)(2u * h + (c->mutant == 3 ? 1u - cc : cc)); st.ureg = udesc; emit(e, st, K_STORE);
            region_end(e, rg);
        }
    }
    tail(e);
    return e->err;
}

int omega_gpu_gemm_codegen(const OmegaGemmCfg *cfg, OmegaGemmKind kind, BlackwellIRProgram *prog) {
    if (!omega_gpu_gemm_cfg_ok(cfg) || !prog || (unsigned)kind >= OMEGA_GEMM_K_COUNT) return -1;
    omega_bw_ir_init(prog);
    Em em; memset(&em, 0, sizeof em); em.p = prog;
    int rc = kind == OMEGA_GEMM_K_MAIN ? gen_main(&em, cfg) : gen_pack(&em, cfg, kind == OMEGA_GEMM_K_PACK_B);
    if (rc != 0 || em.err || em.depth != 0) return -1;
    return omega_bw_regalloc_solve(prog);
}

/* ------------------------------------------------------------------ plan and driver */

static uint32_t roundup(uint32_t v, uint32_t m) { return (v + m - 1u) / m * m; }

int omega_gpu_gemm_plan(const OmegaGemmCfg *c, uint32_t M, uint32_t N, uint32_t K, OmegaGemmPlan *p) {
    if (!omega_gpu_gemm_cfg_ok(c) || !p || !M || !N || !K) return OMEGA_GPU_GEMM_BAD_ARGS;
    if (M > OMEGA_GPU_GEMM_MAX_DIM || N > OMEGA_GPU_GEMM_MAX_DIM || K > OMEGA_GPU_GEMM_MAX_DIM) return OMEGA_GPU_GEMM_TOO_LARGE;
    memset(p, 0, sizeof *p);
    const uint32_t ks = kstep(c->type), ga = 16u * c->rt, gb = 8u * c->ct, FA = 2u * c->rt, FB = c->ct;
    p->Mp = roundup(M, ga * c->wm); p->Np = roundup(N, gb * c->wn); p->Kp = roundup(K, 4u * ks);
    p->KS2 = p->Kp / (2u * ks);
    p->a_bytes = (uint64_t)M * K * esz(c->type); p->b_bytes = (uint64_t)K * N * esz(c->type);
    p->ap_bytes = (uint64_t)(p->Mp / ga) * p->KS2 * FA * FRAG + 2u * FA * FRAG;
    p->bp_bytes = (uint64_t)(p->Np / gb) * p->KS2 * FB * FRAG + 2u * FB * FRAG;
    p->grid[OMEGA_GEMM_K_PACK_A][0] = p->KS2; p->grid[OMEGA_GEMM_K_PACK_A][1] = p->Mp / ga;
    p->grid[OMEGA_GEMM_K_PACK_B][0] = p->KS2; p->grid[OMEGA_GEMM_K_PACK_B][1] = p->Np / gb;
    p->grid[OMEGA_GEMM_K_MAIN][0] = p->Np / (gb * c->wn); p->grid[OMEGA_GEMM_K_MAIN][1] = p->Mp / (ga * c->wm);
    p->threads[OMEGA_GEMM_K_PACK_A] = p->threads[OMEGA_GEMM_K_PACK_B] = 128u;
    p->threads[OMEGA_GEMM_K_MAIN] = 32u * c->wm * c->wn;
    return OMEGA_GPU_GEMM_OK;
}

static void put64(uint32_t *a, unsigned w, uint64_t v) { a[w] = (uint32_t)v; a[w + 1] = (uint32_t)(v >> 32); }

int omega_gpu_gemm_drive(const OmegaGemmCfg *c, unsigned layout, uint32_t M, uint32_t N, uint32_t K, uint64_t a_va,
                         uint64_t b_va, uint64_t ap_va, uint64_t bp_va, uint64_t c_va, OmegaGemmLaunchFn launch, void *ctx) {
    if (!launch || (layout & ~3u)) return OMEGA_GPU_GEMM_BAD_ARGS;
    OmegaGemmPlan p;
    int rc = omega_gpu_gemm_plan(c, M, N, K, &p);
    if (rc != OMEGA_GPU_GEMM_OK) return rc;
    uint32_t a[OMEGA_GPU_GEMM_ARGS_WORDS];
    /* A element (m, k): row-major m K + k, column-major k M + m */
    memset(a, 0, sizeof a);
    put64(a, 0, a_va); put64(a, 2, ap_va); a[4] = M; a[5] = K;
    a[6] = layout & OMEGA_GPU_GEMM_A_COL ? 1u : K; a[7] = layout & OMEGA_GPU_GEMM_A_COL ? M : 1u; a[8] = p.KS2;
    if (launch(ctx, OMEGA_GEMM_K_PACK_A, a, p.grid[0][0], p.grid[0][1], p.threads[0]) != 0) return OMEGA_GPU_GEMM_CHIP_FAIL;
    /* B element (k, n) as packed row n: row-major k N + n, column-major n K + k */
    memset(a, 0, sizeof a);
    put64(a, 0, b_va); put64(a, 2, bp_va); a[4] = N; a[5] = K;
    a[6] = layout & OMEGA_GPU_GEMM_B_COL ? K : 1u; a[7] = layout & OMEGA_GPU_GEMM_B_COL ? 1u : N; a[8] = p.KS2;
    if (launch(ctx, OMEGA_GEMM_K_PACK_B, a, p.grid[1][0], p.grid[1][1], p.threads[1]) != 0) return OMEGA_GPU_GEMM_CHIP_FAIL;
    memset(a, 0, sizeof a);
    put64(a, 0, ap_va); put64(a, 2, bp_va); put64(a, 4, c_va); a[6] = M; a[7] = N; a[8] = p.KS2;
    if (launch(ctx, OMEGA_GEMM_K_MAIN, a, p.grid[2][0], p.grid[2][1], p.threads[2]) != 0) return OMEGA_GPU_GEMM_CHIP_FAIL;
    return OMEGA_GPU_GEMM_OK;
}

/* ------------------------------------------------------------------ the chip */

static struct {
    int set[2];
    OmegaGemmCfg cfg[2];
    OmegaGpuMem code[2][OMEGA_GEMM_K_COUNT];
    uint32_t gpr[2][OMEGA_GEMM_K_COUNT];
    int hooked;
} g;

/* lock held; the session calls it right before it closes the device (also in omega_gpu_session_recover): the kernels go
 * with the device, the next call builds them again on the new one */
static void on_session_close(void) {
    for (unsigned t = 0; t < 2; t++)
        for (unsigned k = 0; k < OMEGA_GEMM_K_COUNT; k++) if (g.code[t][k].cpu) omega_gpu_session_free(&g.code[t][k]);
    memset(g.code, 0, sizeof g.code);
}

static const OmegaGemmCfg *cfg_of(OmegaGpuGemmType t) {
    if (!g.set[t]) { omega_gpu_gemm_default_cfg(t, &g.cfg[t]); g.set[t] = 1; }
    return &g.cfg[t];
}

int omega_gpu_gemm_test_set_cfg(OmegaGpuGemmType type, const OmegaGemmCfg *cfg) {
    if ((unsigned)type > OMEGA_GPU_GEMM_E4M3 || (cfg && (!omega_gpu_gemm_cfg_ok(cfg) || cfg->type != type))) return OMEGA_GPU_GEMM_BAD_ARGS;
    omega_gpu_session_lock();
    for (unsigned k = 0; k < OMEGA_GEMM_K_COUNT; k++) if (g.code[type][k].cpu) omega_gpu_session_free(&g.code[type][k]);
    memset(g.code[type], 0, sizeof g.code[type]);
    if (cfg) g.cfg[type] = *cfg; else omega_gpu_gemm_default_cfg(type, &g.cfg[type]);
    g.set[type] = 1;
    omega_gpu_session_unlock();
    return OMEGA_GPU_GEMM_OK;
}

const char *omega_gpu_gemm_rc_name(int rc) {
    switch (rc) {
    case OMEGA_GPU_GEMM_OK: return "OK";
    case OMEGA_GPU_GEMM_BAD_ARGS: return "BAD_ARGS";
    case OMEGA_GPU_GEMM_TOO_LARGE: return "TOO_LARGE";
    case OMEGA_GPU_GEMM_CODEGEN_FAIL: return "CODEGEN_FAIL";
    case OMEGA_GPU_GEMM_CHIP_FAIL: return "CHIP_FAIL";
    default: return "?";
    }
}

/* lock held */
static int kernel_locked(OmegaGpuGemmType t, OmegaGemmKind k) {
    if (g.code[t][k].cpu) return OMEGA_GPU_GEMM_OK;
    if (!g.hooked) { /* without the hook a kernel would outlive its device */
        if (omega_gpu_session_on_close(on_session_close) != 0) return OMEGA_GPU_GEMM_CHIP_FAIL;
        g.hooked = 1;
    }
    BlackwellIRProgram *prog = omega_gpu_host_calloc(1, sizeof *prog);
    if (!prog) return OMEGA_GPU_GEMM_CODEGEN_FAIL;
    int rc = OMEGA_GPU_GEMM_OK;
    size_t need = 0;
    if (omega_gpu_gemm_codegen(cfg_of(t), k, prog) != 0 || (need = omega_bw_program_code_bytes(prog)) == 0) rc = OMEGA_GPU_GEMM_CODEGEN_FAIL;
    else if (omega_gpu_session_alloc_code(need, &g.code[t][k]) != 0) rc = OMEGA_GPU_GEMM_CHIP_FAIL;
    else {
        OmegaBlackwellKernel kk;
        if (omega_blackwell_kernel_encode_into(prog, g.code[t][k].cpu, need, &kk) != 0 || kk.code_size != need ||
            omega_qmma_patch_code(prog, g.code[t][k].cpu, need) < 0) rc = OMEGA_GPU_GEMM_CODEGEN_FAIL;
        else if ((g.gpr[t][k] = omega_bw_gpr_declared(kk.gpr_count)) > 255u) rc = OMEGA_GPU_GEMM_CODEGEN_FAIL;
        else if (omega_gpu_session_sync_to_gpu(&g.code[t][k]) != 0) rc = OMEGA_GPU_GEMM_CHIP_FAIL;
        if (g.gpr[t][k] < 64u) g.gpr[t][k] = 64u;
        if (rc != OMEGA_GPU_GEMM_OK) { omega_gpu_session_free(&g.code[t][k]); memset(&g.code[t][k], 0, sizeof g.code[t][k]); }
    }
    if (rc == OMEGA_GPU_GEMM_CODEGEN_FAIL) omega_gpu_session_set_error("gemm: our codegen refused a kernel");
    omega_gpu_host_free(prog);
    return rc;
}

typedef struct { OmegaGpuGemmType t; uint64_t pack_ns, main_ns; } ChipCtx;

/* lock held */
static int chip_launch(void *ctx, OmegaGemmKind k, const uint32_t args[OMEGA_GPU_GEMM_ARGS_WORDS], uint32_t gx, uint32_t gy, uint32_t threads) {
    ChipCtx *c = ctx;
    if (kernel_locked(c->t, k) != OMEGA_GPU_GEMM_OK) return -1;
    OmegaGpuLaunch L = {
        .code_va = g.code[c->t][k].va, .gpr_count = g.gpr[c->t][k], .threads_x = threads, .threads_y = 1,
        .grid_x = gx, .grid_y = gy, .num_elements = (size_t)gx * gy * threads,
        .args = args, .n_args = OMEGA_GPU_GEMM_ARGS_WORDS, .timeout_ms = TIMEOUT_MS,
    };
    uint64_t ns = 0;
    if (omega_gpu_session_launch(&L, &ns, NULL) != 0) return -1;
    if (k == OMEGA_GEMM_K_MAIN) c->main_ns += ns; else c->pack_ns += ns;
    return 0;
}

static uint64_t up16(uint64_t v) { return (v + 255u) & ~(uint64_t)255u; }

int omega_gpu_gemm(OmegaGpuGemmType type, unsigned layout, uint32_t M, uint32_t N, uint32_t K, const void *A, const void *B,
                   float *C, OmegaGpuGemmInfo *info) {
    uint64_t t0 = omega_gpu_session_now_ns();
    if (info) memset(info, 0, sizeof *info);
    if ((unsigned)type > OMEGA_GPU_GEMM_E4M3 || (layout & ~3u) || !A || !B || !C) return OMEGA_GPU_GEMM_BAD_ARGS;
    omega_gpu_session_lock();
    const OmegaGemmCfg *cfg = cfg_of(type);
    OmegaGemmPlan p;
    int rc = omega_gpu_gemm_plan(cfg, M, N, K, &p);
    if (rc != OMEGA_GPU_GEMM_OK) { omega_gpu_session_unlock(); return rc; }
    /* one buffer: A | B | packed A | packed B | C (each 256-byte aligned; A and B carry 16 bytes of readable slack) */
    const uint64_t oa = 0, ob = up16(p.a_bytes + 16u), oap = ob + up16(p.b_bytes + 16u), obp = oap + up16(p.ap_bytes),
                   oc = obp + up16(p.bp_bytes), total = oc + (uint64_t)M * N * 4u;
    OmegaGpuMem m;
    if (!omega_gpu_session_open() || omega_gpu_session_alloc(total, &m) != 0) { omega_gpu_session_unlock(); return OMEGA_GPU_GEMM_CHIP_FAIL; }
    uint8_t *base = m.cpu;
    memcpy(base + oa, A, p.a_bytes); memcpy(base + ob, B, p.b_bytes);
    uint32_t *cw = (uint32_t *)(base + oc);
    for (uint64_t i = 0; i < (uint64_t)M * N; i++) cw[i] = POISON;
    ChipCtx cc = { type, 0, 0 };
    if (omega_gpu_session_sync_to_gpu(&m) != 0) rc = OMEGA_GPU_GEMM_CHIP_FAIL;
    else rc = omega_gpu_gemm_drive(cfg, layout, M, N, K, m.va + oa, m.va + ob, m.va + oap, m.va + obp, m.va + oc, chip_launch, &cc);
    if (rc == OMEGA_GPU_GEMM_OK && omega_gpu_session_sync_to_cpu(&m) != 0) rc = OMEGA_GPU_GEMM_CHIP_FAIL;
    if (rc == OMEGA_GPU_GEMM_OK) {
        uint64_t unwritten = 0;
        for (uint64_t i = 0; i < (uint64_t)M * N; i++) unwritten += cw[i] == POISON;
        if (unwritten) { omega_gpu_session_set_error("gemm: output words never written"); rc = OMEGA_GPU_GEMM_CHIP_FAIL; }
        else memcpy(C, cw, (size_t)M * N * 4u);
    }
    if (info) {
        info->pack_ns = cc.pack_ns; info->main_ns = cc.main_ns;
        for (unsigned k = 0; k < OMEGA_GEMM_K_COUNT; k++) info->gpr[k] = g.gpr[type][k];
    }
    omega_gpu_session_free(&m);
    omega_gpu_session_unlock();
    if (info) info->call_ns = omega_gpu_session_now_ns() - t0;
    return rc;
}
