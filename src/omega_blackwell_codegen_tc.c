/* omega_blackwell_codegen_tc.c: tensor-core kernels for the OWN-30B one-pass prefill (T3). See the header.
 * The emitter helpers below are copied from omega_blackwell_codegen_gemm.c (same chip-learned scheduling rules: every load
 * sets the read barrier SB1 and the next load waits on it; prefetch loads hold scoreboards 3/4 until consumed).
 * HMMA scheduling is taken from nvcc 13.0.88 -arch=sm_121 output (oracle sealed in
 * ~/workspace/investigations/2026-10-13-t3-mma-oracle/oracle.{cu,sass}):
 *   independent HMMA.16816.F32.BF16 back to back: control 0x000ff000 / 0x020ff000 (stall 8, no barriers);
 *   a consumer of an HMMA result (dependent HMMA, store): HMMA stall 15 then NOP stall 13..14 (about 29 cycles), as
 *   omega_blackwell_verify_codegen_fixtures_fragmm records. */
#include "omega_blackwell_codegen_tc.h"
#include "omega_bw_em.h"
#include "omega_gpu_fmt.h"

int omega_blackwell_codegen_tc_probe_prog(int mutant, BlackwellIRProgram *prog) {
    if (!prog || mutant < 0 || mutant > 1) return -1;
    omega_bw_ir_init(prog);
    Em em; memset(&em, 0, sizeof em); em.p = prog;
    Em *e = &em;
    int udesc = UV64(e), pin = V64(e), pout = V64(e), tid = V(e), cx = V(e), pa = V64(e), po = V64(e);
    int a = V128(e), b = V64(e), c = V128(e), d = V128(e);
    if (e->err) return -1;
    ldcu64(e, udesc, 0x358);
    ldc64(e, pin, 0x380); ldc64(e, pout, 0x388);
    s2r(e, tid, BW_SR_TID_X); s2r(e, cx, BW_SR_CTAID_X);
    int pcase = V64(e), pb = V64(e); if (e->err) return -1;
    imadw(e, pcase, cx, OMEGA_TC_PROBE_IN_BYTES, pin); imadw(e, pa, tid, 16, pcase); imadw(e, pb, tid, 8, pcase);
    imadw(e, po, cx, OMEGA_TC_PROBE_OUT_BYTES, pout); imadw(e, po, tid, 16, po);
    ldg_wide(e, BW_IR_LDG_E_128, a, pa, 0, udesc, 1);
    ldg_wide(e, BW_IR_LDG_E_128, c, pa, 768, udesc, 0);
    ldg_wide(e, BW_IR_LDG_E_64, b, pb, 512, udesc, 0);
    /* wait for every load, then one HMMA, then 29 cycles before the stores read d */
    hmma(e, d, a, b, mutant == 1 ? NOREG : c, 15);
    nop(e, 14);
    for (unsigned q = 0; q < 4; q++) { BlackwellIRInsn i = I0(BW_IR_STG_E); i.src1_vreg = po; i.src2_vreg = d; i.src2_subreg = (uint8_t)q; i.imm = 4u * q; i.ureg = udesc; emit(e, i, K_STORE); }
    tail(e);
    if (e->err) return -1;
    return omega_bw_regalloc_solve(prog);
}

/* ======================= grouped FP8 GEMM on tensor cores (see the header) ======================= */

int omega_tcgemm_spec_ok(const OmegaTcGemmSpec *s) {
    if (!s) return 0;
    if (s->k < 128u || s->k > OMEGA_TCGEMM_MAX_K || s->k % 128u) return 0;
    if (s->rt != 1 && s->rt != 2 && s->rt != 4) return 0;
    if (s->ct != 1 && s->ct != 2 && s->ct != 4) return 0;
    if (s->rt * s->ct < 4u) return 0; /* >= 4 independent HMMAs between dependent ones (8 cycles each, nvcc's 29-cycle rule) */
    if (s->warps != 1 && s->warps != 2 && s->warps != 4 && s->warps != 8) return 0;
    if ((16u * s->rt * 8u) % (32u * s->warps)) return 0; /* whole LDG.128 staging slots per thread */
    if (s->mutant < 0 || s->mutant > 4 || (s->mutant == 3 && s->rt < 2)) return 0;
    return 1;
}
uint32_t omega_tcgemm_shared_bytes(const OmegaTcGemmSpec *s) { return omega_tcgemm_spec_ok(s) ? 2u * 16u * s->rt * 128u : 0u; }
uint32_t omega_tcgemm_span(const OmegaTcGemmSpec *s) { return omega_tcgemm_spec_ok(s) ? s->warps * 8u * s->ct : 0u; }

static uint16_t bf16_rne(float f) {
    uint32_t u; memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x40u); /* NaN stays NaN */
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}
void omega_tcgemm_permute_x(const float *x, uint32_t k, uint16_t *out) {
    static const unsigned ord[4] = { 0, 2, 1, 3 };
    for (uint32_t q = 0; q + 4 <= k; q += 4)
        for (unsigned i = 0; i < 4; i++) out[q + i] = bf16_rne(x[q + ord[i]] * 18446744073709551616.0f);
}
uint16_t omega_tcgemm_w_bf16(uint8_t c) { return (uint16_t)(((c & 0x7Fu) << 4) | ((c & 0x80u) << 8)); }

/* one e4m3 word (bytes b0..b3) -> B register pair: lo = (bf(b0), bf(b2)), hi = (bf(b1), bf(b3)) */
static void decode_word(Em *e, int bpair, int wq, unsigned word, int m, int s, int t) {
    andi(e, m, wq, word, 0x007F007Fu); andi(e, s, wq, word, 0x00800080u);
    imadi(e, t, m, 0, 16u, NOREG);
    { BlackwellIRInsn i = I0(BW_IR_IMAD); i.dst_vreg = bpair; i.dst_subreg = 0; i.src1_vreg = s; i.imm = 256u; i.src3_vreg = t; emit(e, i, K_FIXED); }
    andi(e, m, wq, word, 0x7F007F00u); andi(e, s, wq, word, 0x80008000u);
    shr(e, t, m, 0, 4);
    { BlackwellIRInsn i = I0(BW_IR_IADD3); i.dst_vreg = bpair; i.dst_subreg = 1; i.src1_vreg = t; i.src2_vreg = s; emit(e, i, K_FIXED); }
}

int omega_blackwell_codegen_tcgemm_prog(const OmegaTcGemmSpec *spec, BlackwellIRProgram *prog) {
    if (!omega_tcgemm_spec_ok(spec) || !prog) return -1;
    const uint32_t K = spec->k, RT = spec->rt, CT = spec->ct, W = spec->warps, T = 32u * W;
    const uint32_t AQN = RT * CT >= 8u ? 1u : 2u; /* A fragments in flight: one for the 8-tile shapes keeps peak <= 126 */
    const uint32_t BM = 16u * RT, KB = K / 128u, SPAN = W * 8u * CT, NJ = BM * 8u / T, BUF = BM * 128u;
    omega_bw_ir_init(prog);
    Em em; memset(&em, 0, sizeof em); em.p = prog;
    Em *e = &em;
    (void)lop3;

    /* ---- prologue: arguments, ids, tile entry ---- */
    int udesc = UV64(e), ptab = V64(e), pidx = V64(e), px = V64(e), py = V64(e), nreg = V(e);
    int tid = V(e), cx = V(e), cy = V(e), warp = V(e), lane = V(e), g = V(e), tq = V(e), one = V(e), tent = V64(e);
    int wbase = V64(e), sbase = V64(e), rc = V64(e), row0 = V(e), cnt = V(e);
    int n0w = V(e), tmp = V(e), tmp2 = V(e), pw = V64(e), ps = V64(e), la = V(e), ysn = V(e), pyr = V64(e);
    if (e->err) return -1;
    ldcu64(e, udesc, 0x358);
    ldc64(e, ptab, OMEGA_TCGEMM_ARG_TILES); ldc64(e, pidx, OMEGA_TCGEMM_ARG_ROWIDX); ldc64(e, px, OMEGA_TCGEMM_ARG_X); ldc64(e, py, OMEGA_TCGEMM_ARG_Y);
    ldc32(e, nreg, OMEGA_TCGEMM_ARG_N);
    s2r(e, tid, BW_SR_TID_X); s2r(e, cx, BW_SR_CTAID_X); s2r(e, cy, BW_SR_CTAID_Y);
    movi(e, one, 1);
    shr(e, warp, tid, 0, 5); andi(e, lane, tid, 0, 31); shr(e, g, lane, 0, 2); andi(e, tq, lane, 0, 3);
    imadw(e, tent, cy, OMEGA_TCGEMM_TILE_BYTES, ptab);
    ldg_wide(e, BW_IR_LDG_E_64, wbase, tent, 0, udesc, 1);
    ldg_wide(e, BW_IR_LDG_E_64, sbase, tent, 8, udesc, 0);
    ldg_wide(e, BW_IR_LDG_E_64, rc, tent, 16, udesc, 0);
    imadi(e, row0, rc, 0, 1, NOREG); imadi(e, cnt, rc, 1, 1, NOREG);
    /* this warp's first weight row; weight pointer row (n0w + g), byte 16 t; scale row block n0w / 128 */
    imadi(e, n0w, cx, 0, SPAN, NOREG); imadi(e, tmp, warp, 0, 8u * CT, NOREG); iadd(e, n0w, n0w, tmp);
    iadd(e, tmp, n0w, g); imadw(e, pw, tmp, K, wbase); imadi(e, tmp2, tq, 0, 16u, NOREG); imadw(e, pw, tmp2, 1, pw);
    shr(e, tmp, n0w, 0, 7); imadw(e, ps, tmp, KB * 4u, sbase);
    imadi(e, la, lane, 0, 16u, NOREG);
    imadi(e, ysn, nreg, 0, 4u, NOREG);
    /* output pointer: y + ((row0 + g) * n + n0w + 2t) * 4 */
    iadd(e, tmp, row0, g); imadr(e, tmp, tmp, nreg, n0w); imadi(e, tmp2, tq, 0, 2u, NOREG); iadd(e, tmp, tmp, tmp2); imadw(e, pyr, tmp, 4u, py);

    /* ---- staging assignment: slot = tid + jj * T -> row r = slot / 8, words w0 = 4 * (tid % 8) of the 64-element chunk ---- */
    int gx[8], sa[8];
    for (uint32_t jj = 0; jj < NJ; jj++) { gx[jj] = V64(e); sa[jj] = V(e); }
    int r = V(e), sidx = V64(e), srow = V(e), w0b = V(e), ts = V(e), s0 = V(e);
    if (e->err) return -1;
    andi(e, tmp, tid, 0, 7);
    imadi(e, w0b, tmp, 0, 16u, NOREG);              /* byte offset of word w0 in the row chunk: 4 words * 4 bytes */
    shr(e, ts, tmp, 0, 1);                          /* t of these words = w0 / 8 */
    andi(e, s0, tid, 0, 1); imadi(e, s0, s0, 0, 2u, NOREG); /* s0 = (w0 % 8) / 2 */
    for (uint32_t jj = 0; jj < NJ; jj++) {
        shr(e, r, tid, 0, 3); imadi(e, r, one, 0, jj * T / 8u, r);
        iadd(e, tmp, row0, r); imadw(e, sidx, tmp, 4u, pidx);
        ldg32(e, srow, sidx, udesc, 1);
        imadw(e, gx[jj], srow, K * 2u, px); imadw(e, gx[jj], w0b, 1, gx[jj]);
        /* shared byte address ((i*4 + s0)*32 + g'*4 + t)*16 + h*4 with i = r/16, g' = r%8, h = (r/8)%2 */
        shr(e, tmp, r, 0, 4); imadi(e, tmp, tmp, 0, 4u, s0); imadi(e, tmp, tmp, 0, 32u, ts);
        andi(e, tmp2, r, 0, 7); imadi(e, tmp2, tmp2, 0, 4u, tmp); imadi(e, sa[jj], tmp2, 0, 16u, NOREG);
        shr(e, tmp2, r, 0, 3); andi(e, tmp2, tmp2, 0, 1); imadi(e, sa[jj], tmp2, 0, 4u, sa[jj]);
    }

    /* ---- loop-wide registers ---- */
    int acc[16], tm[16], wa[4], wb[4], bp[4], aq[2], stg[8], sc = V(e), scur = V(e), cntk = V(e), dm = V(e), ds = V(e), dt = V(e), sv = V(e), onef = V(e);
    for (uint32_t x = 0; x < RT * CT; x++) { acc[x] = V128(e); tm[x] = V128(e); }
    for (uint32_t j = 0; j < CT; j++) { wa[j] = V128(e); wb[j] = V128(e); bp[j] = V64(e); }
    for (uint32_t q = 0; q < AQN; q++) aq[q] = V128(e);
    for (uint32_t jj = 0; jj < NJ; jj++) stg[jj] = V128(e);
    if (e->err) return -1;

    /* issue the loads of one chunk: activations into stg (row chunk at gx + xoff), weights into wset (pw + woff) */
    #define ISSUE(wset, sb, xoff, woff, with_scale) do { \
        for (uint32_t jj = 0; jj < NJ; jj++) ldw_pf(e, BW_IR_LDG_E_128, stg[jj], gx[jj], (xoff), udesc, (sb)); \
        for (uint32_t j = 0; j < CT; j++) ldw_pf(e, BW_IR_LDG_E_128, (wset)[j], pw, (woff) + j * 8u * K, udesc, (sb)); \
        if (with_scale) ldw_pf(e, BW_IR_LDG_E, sc, ps, 0, udesc, (sb)); } while (0)
    /* stage stg into buffer `buf` (byte offset): words w0, w0+1 -> s0 pair 0/1, w0+2, w0+3 -> s0+1 */
    #define STAGE(buf) do { for (uint32_t jj = 0; jj < NJ; jj++) { static const uint32_t off[4] = { 0, 8, 512, 520 }; \
        for (unsigned q = 0; q < 4; q++) { BlackwellIRInsn st = I0(BW_IR_STS32); imadi(e, tmp, one, 0, (buf) + off[q], sa[jj]); \
            st.src1_vreg = tmp; st.src2_vreg = stg[jj]; st.src2_subreg = (uint8_t)q; emit(e, st, K_STORE); } } bar_sync(e); } while (0)

    /* compute one 64-element chunk from shared buffer `buf` with weight set wset; first: the first chunk of a block */
    #define COMPUTE(buf, wset, first, odd) do { \
        for (uint32_t s = 0; s < 4; s++) { \
            if (spec->mutant == 4 && (odd) && s == 3) break; \
            for (uint32_t j = 0; j < CT; j++) decode_word(e, bp[j], (wset)[j], s, dm, ds, dt); \
            for (uint32_t i0 = 0; i0 < RT; i0 += AQN) { \
                const uint32_t nb = RT - i0 < AQN ? RT - i0 : AQN; \
                for (uint32_t q = 0; q < nb; q++) { uint32_t i = i0 + q, ii = spec->mutant == 3 ? (i ^ 1u) : i; \
                    lds128(e, aq[q], la, (buf) + (ii * 4u + s) * 512u, q == 0); } \
                for (uint32_t q = 0; q < nb; q++) for (uint32_t j = 0; j < CT; j++) { uint32_t x = (i0 + q) * CT + j; \
                    hmma(e, tm[x], aq[q], bp[j], ((first) && s == 0) ? NOREG : tm[x], 8); } \
            } \
        } } while (0)

    /* chunk 0 of block 0: activations to buffer 0, weights to set A, scale of block 0 */
    for (uint32_t x = 0; x < RT * CT; x++) for (unsigned q = 0; q < 4; q++) { BlackwellIRInsn mz = I0(BW_IR_MOV_RZ); mz.dst_vreg = acc[x]; mz.dst_subreg = (uint8_t)q; emit(e, mz, K_FIXED); }
    ISSUE(wa, 3u, 0u, 0u, 1);
    consume(e, 3);
    STAGE(0u);
    movi(e, cntk, KB);
    if (spec->mutant == 1) movi(e, onef, 0x3f800000u);

    /* ---- K loop: one 128-wide block (two chunks) per iteration ---- */
    const int L_k = here(e);
    loop_begin(e);
    ISSUE(wb, 4u, 128u, 64u, 0);                   /* chunk 2kb+1 */
    consume(e, 3);
    imadi(e, scur, sc, 0, 1u, NOREG);              /* keep this block's scale: the next block's load reuses sc */
    COMPUTE(0u, wa, 1, 0);
    consume(e, 4);
    STAGE(BUF);
    {   /* chunk 2kb+2 (the next block's first chunk); the last iteration re-reads its own chunk (harmless, drained below) */
        int nf = V(e), inc = V(e); if (e->err) return -1;
        imadi(e, nf, one, 0, 0xfffffffeu, cntk); shr(e, nf, nf, 0, 31); imadi(e, nf, nf, 0, 0xffffffffu, one);
        for (uint32_t jj = 0; jj < NJ; jj++) imadw(e, gx[jj], nf, 256u, gx[jj]);
        imadw(e, pw, nf, 128u, pw);
        imadi(e, inc, nf, 0, spec->mutant == 2 ? 0u : 4u, NOREG); imadw(e, ps, inc, 1, ps);
    }
    ISSUE(wa, 3u, 0u, 0u, 1);
    COMPUTE(BUF, wb, 0, 1);
    nop(e, 15); nop(e, 14);                        /* the last HMMAs' results settle before the FFMAs read them */
    for (uint32_t x = 0; x < RT * CT; x++) for (unsigned q = 0; q < 4; q++) {
        BlackwellIRInsn f = I0(BW_IR_FFMA); f.dst_vreg = acc[x]; f.dst_subreg = (uint8_t)q; f.src1_vreg = tm[x]; f.src1_subreg = (uint8_t)q;
        f.src2_vreg = spec->mutant == 1 ? onef : scur; f.src3_vreg = acc[x]; f.src3_subreg = (uint8_t)q; emit(e, f, K_FIXED);
    }
    consume(e, 3);
    STAGE(0u);
    imadi(e, cntk, one, 0, 0xffffffffu, cntk);
    isetp_ge_u32(e, cntk, one);
    bra_if_p0(e, L_k);
    loop_end(e);
    #undef ISSUE
    #undef STAGE
    #undef COMPUTE

    /* ---- store: row tile i, half h (rows 16i + 8h + g) when < cnt; columns n0w + 8j + 2t (+1) ---- */
    int rr = V(e), pya = V64(e);
    if (e->err) return -1;
    for (uint32_t i = 0; i < RT; i++) for (uint32_t h = 0; h < 2; h++) {
        imadi(e, rr, one, 0, 16u * i + 8u * h, g);
        imadi(e, sv, rr, 0, 1u, NOREG);
        isetp_ge_u32(e, sv, cnt);
        const int at_bssy = bssy_fwd(e);
        const int bra_skip = bra_if_p0_fwd(e);
        imadi(e, tmp, ysn, 0, 16u * i + 8u * h, NOREG); imadw(e, pya, tmp, 1, pyr);
        for (uint32_t j = 0; j < CT; j++) for (unsigned q = 0; q < 2; q++) { BlackwellIRInsn st = I0(BW_IR_STG_E); st.src1_vreg = pya; st.src2_vreg = acc[i * CT + j]; st.src2_subreg = (uint8_t)(2u * h + q); st.imm = j * 32u + 4u * q; st.ureg = udesc; emit(e, st, K_STORE); }
        const int at_bsync = here(e);
        bsync(e);
        patch_bra(e, bra_skip, at_bsync);
        patch_bra(e, at_bssy, at_bsync + 1);
    }
    tail(e);
    if (e->err) return -1;
    return omega_bw_regalloc_solve(prog);
}

/* ---------------------------------------------------------------- placeholder opcodes (header: OMEGA_TC_MARK_*) */

static uint32_t rd32(const uint8_t *p) { return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24; }
static void wr32(uint8_t *p, uint32_t v) { p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); p[2] = (uint8_t)(v >> 16); p[3] = (uint8_t)(v >> 24); }
int omega_tc_patch_code(const BlackwellIRProgram *prog, uint8_t *code, size_t len) {
    if (!prog || !code || len < prog->count * 16u) return -1;
    int n = 0;
    for (size_t i = 0; i < prog->count; i++) {
        const BlackwellIRInsn *in = &prog->insns[i]; uint8_t *w = code + i * 16u;
        if (in->op == BW_IR_SHFL_DOWN && (in->imm & OMEGA_TC_MARK_BFLY)) {
            uint32_t w0 = rd32(w), w1 = rd32(w + 4);
            if ((w0 & 0xffffu) != 0x7f89u || (w1 & 0xfc0000ffu) != 0x08000000u || (w1 & 0x0000ff00u) != 0x1f00u) return -1;
            wr32(w + 4, w1 | 0x04000000u); n++;
        } else if (in->op == BW_IR_FMNMX_MAX && in->imm == OMEGA_TC_MARK_F2FP) {
            uint32_t w0 = rd32(w), w1 = rd32(w + 4), w2 = rd32(w + 8);
            if ((w0 & 0xffffu) != 0x7209u || (w1 & 0xffffff00u) != 0 || w2 != 0x07800000u) return -1;
            wr32(w, (w0 & 0xffff0000u) | 0x723eu); wr32(w + 8, 0x000010ffu); n++;
        }
    }
    return n;
}

int omega_tc_patch_selftest(char *msg, size_t msgn) {
    static BlackwellIRProgram p; static uint8_t buf[64 * 16]; size_t len = 0;
    omega_bw_ir_init(&p);
    Em em; memset(&em, 0, sizeof em); em.p = &p; Em *e = &em;
    int udesc = UV64(e), py = V64(e), tid = V(e), a = V(e), b = V(e), c = V(e), d = V(e);
    if (e->err) { omega_gpu_fmt(msg, msgn, "vregs"); return -1; }
    ldcu64(e, udesc, 0x358); ldc64(e, py, 0x380); s2r(e, tid, BW_SR_TID_X);
    shfl_bfly(e, a, tid, 1); shfl_bfly(e, b, a, 2); shfl_down(e, c, b, 1);
    f2fp_pack(e, d, a, c);
    imadw(e, py, tid, 4u, py); stg(e, py, 0, d, udesc); stg(e, py, 128, b, udesc);
    tail(e);
    if (e->err || omega_bw_regalloc_solve(&p) != 0 || omega_bw_encode_program(&p, buf, sizeof buf, &len) != 0) { omega_gpu_fmt(msg, msgn, "build/encode failed"); return -1; }
    int np = omega_tc_patch_code(&p, buf, len);
    if (np != 3) { omega_gpu_fmt(msg, msgn, "patched %d placeholders, expected 3", np); return -1; }
    int seen = 0;
    for (size_t i = 0; i < p.count; i++) {
        const uint8_t *w = buf + i * 16u; uint32_t w0 = rd32(w), w1 = rd32(w + 4), w2 = rd32(w + 8);
        const BlackwellIRInsn *in = &p.insns[i];
        if (in->op == BW_IR_SHFL_DOWN) {
            /* oracle words with the register bytes (16..31 of word 0) masked: BFLY lane 1 0x0c201f00, lane 2 0x0c401f00; DOWN 1 0x08201f00 */
            uint32_t lane = in->imm & 31u, want = ((in->imm & OMEGA_TC_MARK_BFLY) ? 0x0c001f00u : 0x08001f00u) | lane << 21;
            if ((w0 & 0xffffu) != 0x7f89u || w1 != want || w2 != 0x000e0000u) { omega_gpu_fmt(msg, msgn, "SHFL word %zu: %08x %08x %08x", i, w0, w1, w2); return -1; }
            seen++;
        } else if (in->op == BW_IR_FMNMX_MAX) {
            if ((w0 & 0xffffu) != 0x723eu || (w1 & 0xffffff00u) != 0 || w2 != 0x000010ffu) { omega_gpu_fmt(msg, msgn, "F2FP word %zu: %08x %08x %08x", i, w0, w1, w2); return -1; }
            seen++;
        }
    }
    if (seen != 4) { omega_gpu_fmt(msg, msgn, "saw %d of 4 checked words", seen); return -1; }
    omega_gpu_fmt(msg, msgn, "SHFL.BFLY x2, SHFL.DOWN x1, F2FP.BF16.F32.PACK_AB x1 match the oracle bits");
    return 0;
}

int omega_blackwell_codegen_tc_peak_prog(uint32_t iters, uint32_t chains, uint32_t stall, uint32_t threads, BlackwellIRProgram *prog) {
    if (!prog || iters == 0 || chains == 0 || chains > 16 || stall == 0 || stall > 15 || threads == 0 || threads % 32 || threads > 1024) return -1;
    omega_bw_ir_init(prog);
    Em em; memset(&em, 0, sizeof em); em.p = prog; Em *e = &em;
    int udesc = UV64(e), py = V64(e), tid = V(e), cta = V(e), one = V(e), cnt = V(e), a = V128(e), b = V64(e), acc[16];
    for (uint32_t c = 0; c < chains; c++) acc[c] = V128(e);
    if (e->err) return -1;
    ldcu64(e, udesc, 0x358); ldc64(e, py, 0x380); s2r(e, tid, BW_SR_TID_X); s2r(e, cta, BW_SR_CTAID_X);
    movi(e, one, 1);
    for (unsigned q = 0; q < 4; q++) { BlackwellIRInsn m = I0(BW_IR_MOV_IMM); m.dst_vreg = a; m.dst_subreg = (uint8_t)q; m.imm = 0x3f803f80u; emit(e, m, K_FIXED); }
    for (unsigned q = 0; q < 2; q++) { BlackwellIRInsn m = I0(BW_IR_MOV_IMM); m.dst_vreg = b; m.dst_subreg = (uint8_t)q; m.imm = 0x3f803f80u; emit(e, m, K_FIXED); }
    for (uint32_t c = 0; c < chains; c++) for (unsigned q = 0; q < 4; q++) { BlackwellIRInsn mz = I0(BW_IR_MOV_RZ); mz.dst_vreg = acc[c]; mz.dst_subreg = (uint8_t)q; emit(e, mz, K_FIXED); }
    movi(e, cnt, iters);
    const int L = here(e);
    loop_begin(e);
    for (uint32_t c = 0; c < chains; c++) hmma(e, acc[c], a, b, acc[c], stall);
    imadi(e, cnt, one, 0, 0xffffffffu, cnt);
    isetp_ge_u32(e, cnt, one);
    bra_if_p0(e, L);
    loop_end(e);
    nop(e, 15); nop(e, 15);
    {   int t = V(e); if (e->err) return -1;
        imadi(e, t, cta, 0, threads, tid); imadw(e, py, t, 16u * chains, py);
        for (uint32_t c = 0; c < chains; c++) for (unsigned q = 0; q < 4; q++) { BlackwellIRInsn st = I0(BW_IR_STG_E); st.src1_vreg = py; st.src2_vreg = acc[c]; st.src2_subreg = (uint8_t)q; st.imm = c * 16u + 4u * q; st.ureg = udesc; emit(e, st, K_STORE); }
    }
    tail(e);
    if (e->err) return -1;
    return omega_bw_regalloc_solve(prog);
}

int omega_blackwell_codegen_tc_bw_prog(uint32_t iters, uint32_t groups, uint32_t threads, BlackwellIRProgram *prog) {
    if (!prog || iters == 0 || groups == 0 || threads == 0 || threads % 32 || threads > 1024) return -1;
    omega_bw_ir_init(prog);
    Em em; memset(&em, 0, sizeof em); em.p = prog; Em *e = &em;
    int udesc = UV64(e), px = V64(e), py = V64(e), tid = V(e), cta = V(e), one = V(e), cnt = V(e), grp = V(e), t = V(e), acc = V(e), ld[8];
    for (int q = 0; q < 8; q++) ld[q] = V128(e);
    if (e->err) return -1;
    ldcu64(e, udesc, 0x358); ldc64(e, px, 0x380); ldc64(e, py, 0x388); s2r(e, tid, BW_SR_TID_X); s2r(e, cta, BW_SR_CTAID_X);
    movi(e, one, 1); movrz(e, acc);
    /* grp = cta % groups (cta - (cta / groups) * groups via the unsigned high multiply is not needed: groups is 1 or the grid) */
    if (groups == 1) movrz(e, grp); else imadi(e, grp, cta, 0, 1u, NOREG);
    imadw(e, px, grp, iters * threads * 128u, px);
    imadi(e, t, tid, 0, 16u, NOREG); imadw(e, px, t, 1, px);
    movi(e, cnt, iters);
    const int L = here(e);
    loop_begin(e);
    for (int q = 0; q < 8; q++) ldg_wide(e, BW_IR_LDG_E_128, ld[q], px, (uint32_t)q * threads * 16u, udesc, q == 0);
    for (int q = 0; q < 8; q++) for (unsigned s = 0; s < 4; s++) { BlackwellIRInsn x = I0(BW_IR_LOP3_XOR); x.dst_vreg = acc; x.src1_vreg = acc; x.src2_vreg = ld[q]; x.src2_subreg = (uint8_t)s; emit(e, x, K_FIXED); }
    imadw(e, px, one, threads * 128u, px);
    imadi(e, cnt, one, 0, 0xffffffffu, cnt);
    isetp_ge_u32(e, cnt, one);
    bra_if_p0(e, L);
    loop_end(e);
    imadi(e, t, cta, 0, threads, tid); imadw(e, py, t, 4u, py);
    stg(e, py, 0, acc, udesc);
    tail(e);
    if (e->err) return -1;
    return omega_bw_regalloc_solve(prog);
}
