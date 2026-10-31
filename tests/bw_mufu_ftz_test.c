/* bw_mufu_ftz_test.c: the MUFU flush rule (tests/bw_mufu_ftz.h, omega #462) and every warp simulator's MUFU cases.
 *
 * Built once per model (Makefile test-mufu-ftz):
 *   no WS_SIM_HDR                    the helper itself: subnormal -> sign-preserving zero; zeros, normals, inf, NaN pass
 *   -DWS_SIM_HDR='"bw_warp_sim.h"'   (and _k, _pa, _tc) one MUFU instruction through that simulator's ws_lane_exec
 * The simulator cases expect the chip rule: a subnormal input or result is a zero of the same sign. Before #462 the
 * first three simulators returned the IEEE subnormal (RCP of 1.5 * 2^127, EX2 of -130) and the IEEE finite value for a
 * subnormal input (RCP / RSQ of 2^-127, 2^-140), and fail here. Unreachable: an RSQ result can never be subnormal
 * (1 / sqrt(x) >= 2^-64 for every finite f32 x), so RSQ only exercises the input flush. */
#ifdef WS_SIM_HDR
#include WS_SIM_HDR /* the simulator under test; it must get its MUFU rule from bw_mufu_ftz.h */
#else
#include "bw_mufu_ftz.h"
#endif
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int g_checks, g_failed;
__attribute__((unused)) static uint32_t f2u(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static float u2f(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
/* expect bits exactly, or any NaN when want is a NaN */
static void expect(const char *what, uint32_t in, uint32_t got, uint32_t want) {
    int nan = (want & 0x7fffffffu) > 0x7f800000u, ok = nan ? (got & 0x7fffffffu) > 0x7f800000u : got == want;
    g_checks++;
    if (!ok) { g_failed++; printf("FAIL: %s(0x%08x) = 0x%08x (%g), want 0x%08x (%g)\n", what, in, got, u2f(got), want, u2f(want)); }
}

#ifndef WS_SIM_HDR
#define MODEL "helper"
static void run_model(void) {
    /* every subnormal of both signs, sampled across the whole mantissa range, plus its extremes */
    for (uint32_t m = 1; m < 0x800000u; m += (m < 64 || m > 0x7fffc0u) ? 1u : 4093u) {
        expect("ftz", m, f2u(ws_mufu_ftz(u2f(m))), 0x00000000u);
        expect("ftz", 0x80000000u | m, f2u(ws_mufu_ftz(u2f(0x80000000u | m))), 0x80000000u);
    }
    static const uint32_t pass[] = { 0x00000000u, 0x80000000u, 0x00800000u, 0x80800000u, 0x3f800000u, 0xbf800000u,
        0x7f7fffffu, 0xff7fffffu, 0x7f800000u, 0xff800000u, 0x7fc00000u, 0xffc00000u, 0x7f800001u, 0x00ffffffu };
    for (size_t i = 0; i < sizeof pass / sizeof pass[0]; i++) expect("ftz", pass[i], f2u(ws_mufu_ftz(u2f(pass[i]))), pass[i]);
    /* the three functions, same table the simulators use below */
    expect("ex2", f2u(-130.0f), f2u(ws_mufu_ex2(-130.0f)), 0x00000000u);
    expect("rcp", 0xff400000u, f2u(ws_mufu_rcp(u2f(0xff400000u))), 0x80000000u);
    expect("rsq", 0x00000200u, f2u(ws_mufu_rsq(u2f(0x00000200u))), 0x7f800000u);
}
#else
#define MODEL WS_SIM_HDR
static BlackwellIRProgram g_prog;
static uint32_t sim_mufu(BlackwellIROpcode op, uint32_t x) {
    static WsSim s; static WsLane l;
    memset(&s, 0, sizeof s); memset(&l, 0, sizeof l);
    BlackwellIRInsn in; memset(&in, 0, sizeof in);
    in.op = op; in.dst_vreg = 0; in.src1_vreg = 1; in.src2_vreg = -1; in.src3_vreg = -1; in.ureg = -1;
    s.p = &g_prog; l.r[1] = x; l.r[0] = 0xdeadbeefu;
    ws_lane_exec(&s, &l, &in, 0, 0, 0, 0);
    return l.r[0];
}
static void run_model(void) {
    (void)ws_run; /* the warp scheduler is not under test here */
    g_prog.regalloc.vreg_to_phys[0] = 0; g_prog.regalloc.vreg_to_phys[1] = 1;
    static const struct { BlackwellIROpcode op; const char *name; uint32_t in, want; } c[] = {
        /* subnormal result -> sign-preserving zero (IEEE: 0x00080000, 0x00155555 / 0x80155555) */
        { BW_IR_MUFU_EX2, "EX2", 0xc3020000u /* -130 */, 0x00000000u },
        { BW_IR_MUFU_EX2, "EX2", 0xc2fe0000u /* -127 */, 0x00000000u },
        { BW_IR_MUFU_RCP, "RCP", 0x7f400000u /* 1.5 * 2^127 */, 0x00000000u },
        { BW_IR_MUFU_RCP, "RCP", 0xff400000u /* -1.5 * 2^127 */, 0x80000000u },
        /* subnormal input -> treated as a zero of the same sign (IEEE: 2^127, -2^127, 2^70) */
        { BW_IR_MUFU_RCP, "RCP", 0x00400000u /* 2^-127 */, 0x7f800000u },
        { BW_IR_MUFU_RCP, "RCP", 0x80400000u /* -2^-127 */, 0xff800000u },
        { BW_IR_MUFU_RSQ, "RSQ", 0x00000200u /* 2^-140 */, 0x7f800000u },
        { BW_IR_MUFU_EX2, "EX2", 0x80000200u /* -2^-140 */, 0x3f800000u },
        /* controls: unchanged by the flush */
        { BW_IR_MUFU_EX2, "EX2", 0xc2fc0000u /* -126 -> smallest normal */, 0x00800000u },
        { BW_IR_MUFU_EX2, "EX2", 0x40400000u /* 3 */, 0x41000000u },
        { BW_IR_MUFU_EX2, "EX2", 0xff800000u /* -inf */, 0x00000000u },
        { BW_IR_MUFU_EX2, "EX2", 0x7f800000u /* +inf */, 0x7f800000u },
        { BW_IR_MUFU_EX2, "EX2", 0x7fc00000u /* NaN */, 0x7fc00000u },
        { BW_IR_MUFU_RCP, "RCP", 0x7e800000u /* 2^126 -> smallest normal */, 0x00800000u },
        { BW_IR_MUFU_RCP, "RCP", 0x40000000u /* 2 */, 0x3f000000u },
        { BW_IR_MUFU_RCP, "RCP", 0x80000000u /* -0 */, 0xff800000u },
        { BW_IR_MUFU_RCP, "RCP", 0xff800000u /* -inf */, 0x80000000u },
        { BW_IR_MUFU_RCP, "RCP", 0x7fc00000u /* NaN */, 0x7fc00000u },
        { BW_IR_MUFU_RSQ, "RSQ", 0x40800000u /* 4 */, 0x3f000000u },
        { BW_IR_MUFU_RSQ, "RSQ", 0x00800000u /* 2^-126 */, 0x5f000000u },
        { BW_IR_MUFU_RSQ, "RSQ", 0x7f800000u /* +inf */, 0x00000000u },
        { BW_IR_MUFU_RSQ, "RSQ", 0x80000000u /* -0 */, 0xff800000u },
    };
    for (size_t i = 0; i < sizeof c / sizeof c[0]; i++) expect(c[i].name, c[i].in, sim_mufu(c[i].op, c[i].in), c[i].want);
}
#endif

int main(void) {
    run_model();
    printf("%s: mufu-ftz %s: %d checks, %d failed\n", g_failed ? "FAIL" : "PASS", MODEL, g_checks, g_failed);
    return g_failed ? 1 : 0;
}
