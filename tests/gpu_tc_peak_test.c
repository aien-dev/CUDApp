/* gpu_tc_peak_test.c (OWN-30B T3, 2026-10-13): tensor-core throughput ceiling on GB10 (HMMA.16816.F32.BF16).
 *   --host-only  the generator in the warp simulator: every accumulator 16 * iters
 *   (default)    the chip (caller holds flock /tmp/aien-gb10.lock): a sweep of issue stall x accumulator chains x warps x CTAs;
 *                each point checks every stored value (16 * iters) and reports TFLOPS = 4096 flop per HMMA / chip time
 *                (median of 5). Halving the chains must roughly halve the time when the tensor pipe is the limit.
 * Verdict line last: "GPU_TC_PEAK: PASS|FAIL". */
#include "omega_blackwell_codegen_tc.h"
#include "omega_gpu_session.h"
#include "bw_warp_sim_tc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int cmp_u64(const void *a, const void *b) { uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b; return x < y ? -1 : x > y; }

static int host_only(void) {
    static BlackwellIRProgram p; int fail = 0;
    for (uint32_t chains = 8; chains <= 16; chains += 8) {
        if (omega_blackwell_codegen_tc_peak_prog(5, chains, 2, 64, &p) != 0) { printf("codegen refused chains %u\n", chains); return 1; }
        static float out[2 * 64 * 64]; for (size_t i = 0; i < sizeof out / 4; i++) out[i] = -1;
        static WsSim s; memset(&s, 0, sizeof s); s.p = &p; s.threads = 64;
        uint64_t va = (uintptr_t)out; s.cbank[0x380 / 4] = (uint32_t)va; s.cbank[0x384 / 4] = (uint32_t)(va >> 32);
        s.ranges[0].lo = va; s.ranges[0].hi = va + 2u * 64u * 16u * chains; s.nranges = 1;
        int err = ws_run(&s, 2, 1); size_t bad = 0;
        for (size_t i = 0; i < 2u * 64u * 4u * chains; i++) if (out[i] != 80.0f) bad++;
        printf("sim chains %u: %s, peak %u registers, %zu instructions, %zu wrong\n", chains, err ? s.msg : "ran", p.regalloc.peak_gpr_usage, p.count, bad);
        if (err || bad) fail++;
    }
    return fail;
}

int main(int argc, char **argv) {
    int fail = host_only();
    if (argc > 1 && !strcmp(argv[1], "--host-only")) { printf("GPU_TC_PEAK: %s\n", fail ? "FAIL" : "PASS"); return fail ? 1 : 0; }
    if (fail) { printf("GPU_TC_PEAK: FAIL\n"); return 1; }
    static const struct { uint32_t stall, chains, threads, ctas; } pts[] = {
        { 8, 16, 128, 192 }, { 4, 16, 128, 192 }, { 2, 16, 128, 192 }, { 2, 8, 128, 192 },
        { 2, 16, 256, 192 }, { 2, 16, 128, 384 }, { 2, 16, 512, 96 }, { 4, 8, 256, 384 }, { 1, 16, 256, 384 },
    };
    const uint32_t iters = 4096;
    omega_gpu_session_lock();
    if (!omega_gpu_session_open()) { printf("open failed: %s\n", omega_gpu_session_last_error()); omega_gpu_session_unlock(); return 1; }
    for (size_t i = 0; i < sizeof pts / sizeof pts[0]; i++) {
        static BlackwellIRProgram p; OmegaGpuMem code = { 0 }, out = { 0 }; OmegaBlackwellKernel k;
        if (omega_blackwell_codegen_tc_peak_prog(iters, pts[i].chains, pts[i].stall, pts[i].threads, &p) != 0) { printf("codegen refused\n"); fail++; continue; }
        size_t need = omega_bw_program_code_bytes(&p), ob = (size_t)pts[i].ctas * pts[i].threads * 16u * pts[i].chains;
        uint32_t decl = omega_bw_gpr_declared(p.regalloc.peak_gpr_usage);
        if (omega_gpu_session_alloc_code(need, &code) != 0 || omega_blackwell_kernel_encode_into(&p, code.cpu, need, &k) != 0 || omega_gpu_session_sync_to_gpu(&code) != 0 ||
            omega_gpu_session_alloc(ob, &out) != 0) { printf("upload failed: %s\n", omega_gpu_session_last_error()); fail++; break; }
        uint64_t ns[5]; int ok = 1; size_t bad = 0;
        for (int r = 0; r < 5 && ok; r++) {
            ns[r] = 0; /* omega_gpu_session_launch adds the elapsed time */
            float *o = out.cpu; for (size_t q = 0; q < ob / 4; q++) o[q] = -1.0f;
            uint32_t args[2] = { (uint32_t)out.va, (uint32_t)(out.va >> 32) };
            OmegaGpuLaunch L = { .code_va = code.va, .gpr_count = decl, .threads_x = pts[i].threads, .threads_y = 1, .grid_x = pts[i].ctas, .grid_y = 1,
                                 .num_elements = ob / 4, .args = args, .n_args = 2, .timeout_ms = 20000 };
            if (omega_gpu_session_sync_to_gpu(&out) != 0 || omega_gpu_session_launch(&L, &ns[r], NULL) != 0 || omega_gpu_session_sync_to_cpu(&out) != 0) { printf("launch failed: %s\n", omega_gpu_session_last_error()); ok = 0; break; }
            for (size_t q = 0; q < ob / 4; q++) if (o[q] != 16.0f * iters) bad++;
        }
        if (!ok) { fail++; omega_gpu_session_free(&code); omega_gpu_session_free(&out); break; }
        qsort(ns, 5, 8, cmp_u64);
        double hm = (double)pts[i].ctas * (pts[i].threads / 32u) * iters * pts[i].chains;
        printf("peak stall %u chains %u threads %u ctas %u (declared %u): median %.1f us, %.2f TFLOPS, %zu wrong values\n", pts[i].stall, pts[i].chains, pts[i].threads, pts[i].ctas, decl,
               ns[2] / 1e3, hm * 4096.0 / ns[2] / 1e3, bad);
        if (bad) fail++;
        omega_gpu_session_free(&code); omega_gpu_session_free(&out);
    }
    /* read bandwidth: every CTA distinct vs every CTA the same bytes (any cache reuse of these GPU-uncached buffers) */
    for (int mode = 0; mode < 4 && !fail; mode++) {
        const uint32_t T = 256, grid = 384, iters = mode < 2 ? 32u : 8u, groups = (mode & 1) ? 1u : grid;
        static BlackwellIRProgram p; OmegaGpuMem code = { 0 }, src = { 0 }, out = { 0 }; OmegaBlackwellKernel k;
        if (omega_blackwell_codegen_tc_bw_prog(iters, groups, T, &p) != 0) { printf("bw codegen refused\n"); fail++; break; }
        size_t need = omega_bw_program_code_bytes(&p), region = (size_t)iters * T * 128u, sb = region * groups, ob = (size_t)grid * T * 4u;
        if (omega_gpu_session_alloc_code(need, &code) != 0 || omega_blackwell_kernel_encode_into(&p, code.cpu, need, &k) != 0 || omega_gpu_session_sync_to_gpu(&code) != 0 ||
            omega_gpu_session_alloc(sb, &src) != 0 || omega_gpu_session_alloc(ob, &out) != 0) { printf("bw upload failed: %s\n", omega_gpu_session_last_error()); fail++; break; }
        uint32_t *sw = src.cpu; for (size_t q = 0; q < sb / 4; q++) sw[q] = (uint32_t)(q * 2654435761u);
        uint32_t *want = calloc((size_t)grid * T, 4);
        for (uint32_t c = 0; c < grid; c++) for (uint32_t t = 0; t < T; t++) { uint32_t a = 0; const uint32_t *r = sw + (size_t)(groups == 1 ? 0 : c) * region / 4;
            for (uint32_t it = 0; it < iters; it++) for (uint32_t q = 0; q < 8; q++) for (uint32_t s = 0; s < 4; s++) a ^= r[((size_t)it * T * 128u + (size_t)q * T * 16u + t * 16u) / 4 + s];
            want[(size_t)c * T + t] = a; }
        uint64_t ns[5]; size_t bad = 0; int ok = 1;
        if (omega_gpu_session_sync_to_gpu(&src) != 0) ok = 0;
        for (int r = 0; r < 5 && ok; r++) {
            ns[r] = 0; uint32_t *o = out.cpu; memset(o, 0xee, ob);
            uint32_t args[4] = { (uint32_t)src.va, (uint32_t)(src.va >> 32), (uint32_t)out.va, (uint32_t)(out.va >> 32) };
            OmegaGpuLaunch L = { .code_va = code.va, .gpr_count = omega_bw_gpr_declared(p.regalloc.peak_gpr_usage), .threads_x = T, .threads_y = 1, .grid_x = grid, .grid_y = 1,
                                 .num_elements = ob / 4, .args = args, .n_args = 4, .timeout_ms = 20000 };
            if (omega_gpu_session_sync_to_gpu(&out) != 0 || omega_gpu_session_launch(&L, &ns[r], NULL) != 0 || omega_gpu_session_sync_to_cpu(&out) != 0) { printf("bw launch failed: %s\n", omega_gpu_session_last_error()); ok = 0; break; }
            for (size_t q = 0; q < (size_t)grid * T; q++) if (o[q] != want[q]) bad++;
        }
        if (ok) { qsort(ns, 5, 8, cmp_u64); double bytes = (double)grid * region;
            printf("bw %s, %zu KB per CTA, %u CTAs: median %.1f us, %.1f GB/s delivered (%zu MB distinct), %zu wrong\n", groups == 1 ? "SAME bytes for every CTA" : "distinct bytes per CTA", region >> 10, grid, ns[2] / 1e3, bytes / ns[2], sb >> 20, bad);
            if (bad) fail++; } else fail++;
        free(want); omega_gpu_session_free(&code); omega_gpu_session_free(&src); omega_gpu_session_free(&out);
    }
    omega_gpu_session_unlock();
    printf("GPU_TC_PEAK: %s\n", fail ? "FAIL" : "PASS");
    return fail ? 1 : 0;
}
