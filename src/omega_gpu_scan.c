/* omega_gpu_scan.c -- see omega_gpu_scan.h. Kernels from Omega's IR (emitter helpers of omega_bw_em.h, the
 * chip-proven scheduling rules of the tensor-core generators), host references, the level driver, the chip calls. */
#include "omega_gpu_scan.h"
#include "omega_bw_em.h"
#include "omega_gpu_session.h"

#include <stdlib.h>
#include <string.h>

#define PAD_SUM 0x80000000u    /* -0.0: x + -0.0 == x for every x */
#define PAD_MINMAX 0x7fc00000u /* quiet NaN: max(NaN, x) == min(NaN, x) == x */
#define POISON 0x7fa00badu     /* a signalling NaN: no FADD or FMNMX writes it, so a word still holding it was never written */
#define TIMEOUT_MS 20000ull

/* ------------------------------------------------------------------ IR */

static void fmnmx(Em *e, BlackwellIROpcode op, int d, int a, int b) {
    BlackwellIRInsn i = I0(op); i.dst_vreg = d; i.src1_vreg = a; i.src2_vreg = b; emit(e, i, K_FIXED);
}
typedef struct { int bssy, bra; } Region;
/* the lanes with P0 set skip to the region's end; the warp joins again there */
static Region skip_if_p0(Em *e) { Region r; r.bssy = bssy_fwd(e); r.bra = bra_if_p0_fwd(e); return r; }
static void region_end(Em *e, Region r) { int at = here(e); bsync(e); patch_bra(e, r.bra, at); patch_bra(e, r.bssy, at + 1); }

static int gen(Em *e, const OmegaScanSpec *sp) {
    const OmegaScanKind k = sp->kind;
    const int red = k <= OMEGA_SCAN_K_RED_MIN;
    int udesc = UV64(e), pin = V64(e), pout = V64(e), ptot = V64(e), len = V(e), tiles = V(e), warps = V(e);
    int tid = V(e), cta = V(e), lane = V(e), gw = V(e), one = V(e), tile = V(e), el = V(e), v = V(e), t = V(e);
    int a64 = V64(e), b64 = V64(e), rl = V(e), nz = V(e), c31 = V(e), m1 = V(e), off = V(e), tm1 = V(e);
    int mask[5];
    for (unsigned s = 0; s < 5; s++) mask[s] = k == OMEGA_SCAN_K_TILE ? V(e) : NOREG;
    if (e->err) return -1;
    ldcu64(e, udesc, 0x358);
    ldc64(e, pin, OMEGA_GPU_SCAN_A_IN); ldc64(e, pout, OMEGA_GPU_SCAN_A_OUT);
    if (k == OMEGA_SCAN_K_TILE) ldc64(e, ptot, OMEGA_GPU_SCAN_A_TOT);
    ldc32(e, len, OMEGA_GPU_SCAN_A_LEN); ldc32(e, tiles, OMEGA_GPU_SCAN_A_TILES); ldc32(e, warps, OMEGA_GPU_SCAN_A_WARPS);
    s2r(e, tid, BW_SR_TID_X); s2r(e, cta, BW_SR_CTAID_X);
    movi(e, one, 1); movi(e, m1, 0xffffffffu); movi(e, nz, PAD_SUM);
    andi(e, lane, tid, 0, 31u);
    shr(e, t, tid, 0, 5);
    imadi(e, gw, cta, 0, OMEGA_GPU_SCAN_THREADS / 32u, t);      /* global warp = cta * 8 + warp */
    if (k == OMEGA_SCAN_K_ADD) imadi(e, tile, gw, 0, 1u, one);  /* the carry starts at tile 1 */
    else imadi(e, tile, gw, 0, 1u, NOREG);
    if (k == OMEGA_SCAN_K_TILE) {
        /* reversed lanes: lane L holds tile position 31 - L, so SHFL.DOWN by d reads position - d */
        movi(e, c31, 31u);
        lop3(e, rl, lane, c31, NOREG, 0x3cu);                    /* 31 - L == L ^ 31 for L < 32 */
        /* mask[s] = all ones when position - d exists (L + d <= 31), else 0: ((L + d) >> 5) - 1 */
        for (unsigned s = 0; s < 5; s++) {
            movi(e, t, 1u << s);
            iadd(e, mask[s], lane, t);
            shr(e, mask[s], mask[s], 0, 5);
            imadi(e, mask[s], mask[s], 0, 1u, m1);
        }
    }

    const int top = here(e);
    loop_begin(e);
    isetp_ge_u32(e, tile, tiles);
    const int done = bra_if_p0_fwd(e);
    if (red) {
        const BlackwellIROpcode op = k == OMEGA_SCAN_K_RED_MAX ? BW_IR_FMNMX_MAX : BW_IR_FMNMX_MIN;
        imadi(e, el, tile, 0, 32u, lane);
        movi(e, v, k == OMEGA_SCAN_K_RED_SUM ? PAD_SUM : PAD_MINMAX);
        isetp_ge_u32(e, el, len);
        { Region r = skip_if_p0(e); imadw(e, a64, el, 4u, pin); ldg32(e, v, a64, udesc, 1); region_end(e, r); }
        for (uint32_t d = 16; d >= 1; d /= 2) {
            if (sp->mutant == 1 && d == 1) break;
            shfl_down(e, t, v, d);
            if (k == OMEGA_SCAN_K_RED_SUM) fadd(e, v, v, t); else fmnmx(e, op, v, v, t);
        }
        isetp_ge_u32(e, lane, one);
        { Region r = skip_if_p0(e); imadw(e, a64, tile, 4u, pout); stg(e, a64, 0, v, udesc); region_end(e, r); }
    } else if (k == OMEGA_SCAN_K_TILE) {
        imadi(e, el, tile, 0, 32u, rl);
        movi(e, v, PAD_SUM);
        isetp_ge_u32(e, el, len);
        { Region r = skip_if_p0(e); imadw(e, a64, el, 4u, pin); ldg32(e, v, a64, udesc, 1); region_end(e, r); }
        for (unsigned s = 0; s < 5; s++) {
            if (sp->mutant == 2 && s == 4) break;
            shfl_down(e, t, v, 1u << s);
            lop3(e, t, t, mask[s], nz, 0xe2u);                   /* (t & mask) | (-0.0 & ~mask) */
            fadd(e, v, t, v);                                    /* lower position on the left */
        }
        isetp_ge_u32(e, el, len);
        { Region r = skip_if_p0(e); imadw(e, a64, el, 4u, pout); stg(e, a64, 0, v, udesc); region_end(e, r); }
        isetp_ge_u32(e, lane, one);                               /* lane 0 holds position 31: the tile total */
        { Region r = skip_if_p0(e); imadw(e, a64, tile, 4u, ptot); stg(e, a64, 0, v, udesc); region_end(e, r); }
    } else {
        imadi(e, el, tile, 0, 32u, lane);
        isetp_ge_u32(e, el, len);
        {
            Region r = skip_if_p0(e);
            if (sp->mutant == 3) imadi(e, tm1, tile, 0, 1u, NOREG); else imadi(e, tm1, tile, 0, 1u, m1);
            imadw(e, b64, tm1, 4u, pout); imadw(e, a64, el, 4u, pin);
            ldg32(e, off, b64, udesc, 1); ldg32(e, v, a64, udesc, 0);
            fadd(e, v, off, v);
            stg(e, a64, 0, v, udesc);
            region_end(e, r);
        }
    }
    iadd(e, tile, tile, warps);
    bra(e, top);
    loop_end(e);
    patch_bra(e, done, here(e));
    tail(e);
    return e->err;
}

int omega_gpu_scan_codegen(const OmegaScanSpec *spec, BlackwellIRProgram *prog) {
    if (!spec || !prog || (unsigned)spec->kind >= OMEGA_SCAN_K_COUNT) return -1;
    omega_bw_ir_init(prog);
    Em em; memset(&em, 0, sizeof em); em.p = prog;
    if (gen(&em, spec) != 0 || em.err || em.depth != 0) return -1;
    return omega_bw_regalloc_solve(prog);
}

/* ------------------------------------------------------------------ the level driver */

uint32_t omega_gpu_scan_grid(size_t len) {
    size_t tiles = (len + 31) / 32, ctas = (tiles + OMEGA_GPU_SCAN_THREADS / 32u - 1) / (OMEGA_GPU_SCAN_THREADS / 32u);
    return ctas > OMEGA_GPU_SCAN_MAX_CTAS ? OMEGA_GPU_SCAN_MAX_CTAS : (uint32_t)(ctas ? ctas : 1);
}

static void args_for(uint32_t a[OMEGA_GPU_SCAN_ARGS_WORDS], uint64_t in, uint64_t out, uint64_t tot, size_t len) {
    memset(a, 0, OMEGA_GPU_SCAN_ARGS_WORDS * 4u);
    a[0] = (uint32_t)in; a[1] = (uint32_t)(in >> 32); a[2] = (uint32_t)out; a[3] = (uint32_t)(out >> 32);
    a[4] = (uint32_t)tot; a[5] = (uint32_t)(tot >> 32);
    a[6] = (uint32_t)len; a[7] = (uint32_t)((len + 31) / 32); a[8] = omega_gpu_scan_grid(len) * (OMEGA_GPU_SCAN_THREADS / 32u);
}

size_t omega_gpu_scan_reduce_scratch_words(size_t n) {
    size_t w = 0;
    do { n = (n + 31) / 32; w += n; } while (n > 1);
    return w;
}

size_t omega_gpu_scan_prefix_scratch_words(size_t n) {
    size_t w = 0;
    while (n > 32) { n = (n + 31) / 32; w += n; }
    return w + 1; /* the last level's single total */
}

int omega_gpu_scan_drive_reduce(OmegaGpuRedOp op, size_t n, uint64_t in_va, uint64_t scratch_va, uint64_t *out_va,
                                OmegaScanLaunchFn launch, void *ctx, uint32_t *launches) {
    if ((unsigned)op > OMEGA_GPU_RED_MIN || n == 0 || !launch || !out_va) return OMEGA_GPU_SCAN_BAD_ARGS;
    if (n > OMEGA_GPU_SCAN_MAX_N) return OMEGA_GPU_SCAN_TOO_LARGE;
    const OmegaScanKind k = (OmegaScanKind)(OMEGA_SCAN_K_RED_SUM + (int)op);
    uint64_t in = in_va, out = scratch_va;
    size_t len = n;
    uint32_t a[OMEGA_GPU_SCAN_ARGS_WORDS];
    do {
        args_for(a, in, out, 0, len);
        if (launch(ctx, k, a, omega_gpu_scan_grid(len)) != 0) return OMEGA_GPU_SCAN_CHIP_FAIL;
        if (launches) (*launches)++;
        len = (len + 31) / 32;
        in = out; out += len * 4u;
    } while (len > 1);
    *out_va = in;
    return OMEGA_GPU_SCAN_OK;
}

int omega_gpu_scan_drive_prefix(size_t n, uint64_t in_va, uint64_t out_va, uint64_t scratch_va,
                                OmegaScanLaunchFn launch, void *ctx, uint32_t *launches) {
    if (n == 0 || !launch) return OMEGA_GPU_SCAN_BAD_ARGS;
    if (n > OMEGA_GPU_SCAN_MAX_N) return OMEGA_GPU_SCAN_TOO_LARGE;
    enum { MAXL = 8 }; /* 32^6 >= 2^28 */
    uint64_t buf[MAXL], tot[MAXL];
    size_t len[MAXL];
    unsigned L = 0;
    uint64_t in = in_va, sp = scratch_va;
    buf[0] = out_va; len[0] = n;
    uint32_t a[OMEGA_GPU_SCAN_ARGS_WORDS];
    for (;;) { /* forward: scan every level, its totals become the next level (scanned in place) */
        size_t nt = (len[L] + 31) / 32;
        tot[L] = sp; sp += nt * 4u;
        args_for(a, in, buf[L], tot[L], len[L]);
        if (launch(ctx, OMEGA_SCAN_K_TILE, a, omega_gpu_scan_grid(len[L])) != 0) return OMEGA_GPU_SCAN_CHIP_FAIL;
        if (launches) (*launches)++;
        if (nt == 1) break;
        if (L + 1 >= MAXL) return OMEGA_GPU_SCAN_TOO_LARGE;
        buf[L + 1] = tot[L]; len[L + 1] = nt; in = tot[L];
        L++;
    }
    while (L-- > 0) { /* back down: carry the scanned totals of level L + 1 into level L */
        args_for(a, buf[L], buf[L + 1], 0, len[L]);
        if (launch(ctx, OMEGA_SCAN_K_ADD, a, omega_gpu_scan_grid(len[L])) != 0) return OMEGA_GPU_SCAN_CHIP_FAIL;
        if (launches) (*launches)++;
    }
    return OMEGA_GPU_SCAN_OK;
}

/* ------------------------------------------------------------------ the chip */

static struct {
    int mutant;
    OmegaGpuMem code[OMEGA_SCAN_K_COUNT];
    uint32_t gpr[OMEGA_SCAN_K_COUNT];
} g;

void omega_gpu_scan_test_set_mutant(int mutant) {
    omega_gpu_session_lock();
    for (unsigned k = 0; k < OMEGA_SCAN_K_COUNT; k++) if (g.code[k].cpu) omega_gpu_session_free(&g.code[k]);
    memset(g.code, 0, sizeof g.code);
    g.mutant = mutant;
    omega_gpu_session_unlock();
}

const char *omega_gpu_scan_rc_name(int rc) {
    switch (rc) {
    case OMEGA_GPU_SCAN_OK: return "OK";
    case OMEGA_GPU_SCAN_BAD_ARGS: return "BAD_ARGS";
    case OMEGA_GPU_SCAN_TOO_LARGE: return "TOO_LARGE";
    case OMEGA_GPU_SCAN_CODEGEN_FAIL: return "CODEGEN_FAIL";
    case OMEGA_GPU_SCAN_CHIP_FAIL: return "CHIP_FAIL";
    default: return "?";
    }
}

/* lock held */
static int kernel_locked(OmegaScanKind k) {
    if (g.code[k].cpu) return OMEGA_GPU_SCAN_OK;
    BlackwellIRProgram *prog = omega_gpu_host_calloc(1, sizeof *prog);
    if (!prog) return OMEGA_GPU_SCAN_CODEGEN_FAIL;
    OmegaScanSpec s = { k, g.mutant };
    int rc = OMEGA_GPU_SCAN_OK;
    size_t need = 0;
    if (omega_gpu_scan_codegen(&s, prog) != 0 || (need = omega_bw_program_code_bytes(prog)) == 0) rc = OMEGA_GPU_SCAN_CODEGEN_FAIL;
    else if (omega_gpu_session_alloc_code(need, &g.code[k]) != 0) rc = OMEGA_GPU_SCAN_CHIP_FAIL;
    else {
        OmegaBlackwellKernel kk;
        if (omega_blackwell_kernel_encode_into(prog, g.code[k].cpu, need, &kk) != 0 || kk.code_size != need) rc = OMEGA_GPU_SCAN_CODEGEN_FAIL;
        else if (omega_gpu_session_sync_to_gpu(&g.code[k]) != 0) rc = OMEGA_GPU_SCAN_CHIP_FAIL;
        else { g.gpr[k] = omega_bw_gpr_declared(kk.gpr_count); if (g.gpr[k] < 64u) g.gpr[k] = 64u; }
        if (rc != OMEGA_GPU_SCAN_OK) { omega_gpu_session_free(&g.code[k]); memset(&g.code[k], 0, sizeof g.code[k]); }
    }
    omega_gpu_host_free(prog);
    return rc;
}

typedef struct { uint64_t elapsed_ns; } ChipCtx;

/* lock held */
static int chip_launch(void *ctx, OmegaScanKind k, const uint32_t args[OMEGA_GPU_SCAN_ARGS_WORDS], uint32_t grid_x) {
    if (kernel_locked(k) != OMEGA_GPU_SCAN_OK) return -1;
    OmegaGpuLaunch L = {
        .code_va = g.code[k].va, .gpr_count = g.gpr[k], .threads_x = OMEGA_GPU_SCAN_THREADS, .threads_y = 1,
        .grid_x = grid_x, .grid_y = 1, .num_elements = (size_t)grid_x * OMEGA_GPU_SCAN_THREADS,
        .args = args, .n_args = OMEGA_GPU_SCAN_ARGS_WORDS, .timeout_ms = TIMEOUT_MS,
    };
    uint64_t ns = 0;
    if (omega_gpu_session_launch(&L, &ns, NULL) != 0) return -1;
    ((ChipCtx *)ctx)->elapsed_ns += ns;
    return 0;
}

/* lock held: one data buffer [in n | out n | scratch], the input copied in, outputs poisoned */
static int stage_locked(const float *x, size_t n, size_t out_words, size_t scratch_words, OmegaGpuMem *m) {
    if (!omega_gpu_session_open()) return OMEGA_GPU_SCAN_CHIP_FAIL;
    if (omega_gpu_session_alloc((n + out_words + scratch_words) * 4u, m) != 0) return OMEGA_GPU_SCAN_CHIP_FAIL;
    uint32_t *w = m->cpu;
    memcpy(w, x, n * 4u);
    for (size_t i = n; i < n + out_words + scratch_words; i++) w[i] = POISON;
    if (omega_gpu_session_sync_to_gpu(m) != 0) { omega_gpu_session_free(m); return OMEGA_GPU_SCAN_CHIP_FAIL; }
    return OMEGA_GPU_SCAN_OK;
}

int omega_gpu_reduce(OmegaGpuRedOp op, const float *x, size_t n, float *out, OmegaGpuScanInfo *info) {
    uint64_t t0 = omega_gpu_session_now_ns();
    if (info) memset(info, 0, sizeof *info);
    if ((unsigned)op > OMEGA_GPU_RED_MIN || !x || !out || n == 0) return OMEGA_GPU_SCAN_BAD_ARGS;
    if (n > OMEGA_GPU_SCAN_MAX_N) return OMEGA_GPU_SCAN_TOO_LARGE;
    omega_gpu_session_lock();
    OmegaGpuMem m;
    const size_t sw = omega_gpu_scan_reduce_scratch_words(n);
    int rc = stage_locked(x, n, 0, sw, &m);
    if (rc == OMEGA_GPU_SCAN_OK) {
        ChipCtx c = { 0 }; uint64_t res = 0; uint32_t nl = 0;
        rc = omega_gpu_scan_drive_reduce(op, n, m.va, m.va + n * 4u, &res, chip_launch, &c, &nl);
        if (rc == OMEGA_GPU_SCAN_OK && omega_gpu_session_sync_to_cpu(&m) != 0) rc = OMEGA_GPU_SCAN_CHIP_FAIL;
        if (rc == OMEGA_GPU_SCAN_OK) {
            uint32_t r; memcpy(&r, (const uint8_t *)m.cpu + (res - m.va), 4);
            if (r == POISON) { omega_gpu_session_set_error("reduce: the result word was never written"); rc = OMEGA_GPU_SCAN_CHIP_FAIL; }
            else memcpy(out, &r, 4);
        }
        if (info) { info->launches = nl; info->elapsed_ns = c.elapsed_ns; }
        omega_gpu_session_free(&m);
    }
    omega_gpu_session_unlock();
    if (info) info->call_ns = omega_gpu_session_now_ns() - t0;
    return rc;
}

int omega_gpu_prefix_sum(const float *x, size_t n, float *y, OmegaGpuScanInfo *info) {
    uint64_t t0 = omega_gpu_session_now_ns();
    if (info) memset(info, 0, sizeof *info);
    if (!x || !y || n == 0) return OMEGA_GPU_SCAN_BAD_ARGS;
    if (n > OMEGA_GPU_SCAN_MAX_N) return OMEGA_GPU_SCAN_TOO_LARGE;
    omega_gpu_session_lock();
    OmegaGpuMem m;
    int rc = stage_locked(x, n, n, omega_gpu_scan_prefix_scratch_words(n), &m);
    if (rc == OMEGA_GPU_SCAN_OK) {
        ChipCtx c = { 0 }; uint32_t nl = 0;
        rc = omega_gpu_scan_drive_prefix(n, m.va, m.va + n * 4u, m.va + 2 * n * 4u, chip_launch, &c, &nl);
        if (rc == OMEGA_GPU_SCAN_OK && omega_gpu_session_sync_to_cpu(&m) != 0) rc = OMEGA_GPU_SCAN_CHIP_FAIL;
        if (rc == OMEGA_GPU_SCAN_OK) {
            const uint32_t *w = (const uint32_t *)m.cpu + n;
            size_t unwritten = 0;
            for (size_t i = 0; i < n; i++) unwritten += w[i] == POISON;
            if (unwritten) { omega_gpu_session_set_error("prefix sum: output words never written"); rc = OMEGA_GPU_SCAN_CHIP_FAIL; }
            else memcpy(y, w, n * 4u);
        }
        if (info) { info->launches = nl; info->elapsed_ns = c.elapsed_ns; }
        omega_gpu_session_free(&m);
    }
    omega_gpu_session_unlock();
    if (info) info->call_ns = omega_gpu_session_now_ns() - t0;
    return rc;
}
