/* gpu_pf_mem_test.c (OWN-30B T3b, 2026-10-13): device-resident prefill memory.
 *   --host-only  the copy kernel in the warp simulator (exact copy, nothing past `units` written)
 *   (default)    the chip (caller holds flock /tmp/aien-gb10.lock):
 *     1. upload/download round trip of 96 MB through a GPU-cached device buffer (bit-exact; the host buffer is poisoned
 *        first, so a download that did not run is caught), and a second upload over the same buffer read back exactly
 *        (a stale L2 line would show here: the GPU rewrote lines it had cached);
 *     2. read bandwidth, uncached (session buffer) vs cached (device buffer): every CTA distinct (DRAM) and every CTA the
 *        same bytes (cache reuse), 5 runs each, median; every XOR result checked.
 * Verdict line last: "GPU_PF_MEM: PASS|FAIL". */
#include "omega_gpu_prefill.h"
#include "omega_blackwell_codegen_pf.h"
#include "omega_blackwell_codegen_tc.h"
#include "bw_warp_sim_tc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int cmp_u64(const void *a, const void *b) { uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b; return x < y ? -1 : x > y; }

static int host_only(void) {
    static BlackwellIRProgram p;
    if (omega_blackwell_codegen_pf_copy_prog(&p) != 0) { printf("copy codegen refused\n"); return 1; }
    enum { U = 1000, CAP = 1100 };
    static uint32_t src[CAP * 4], dst[CAP * 4];
    for (int i = 0; i < CAP * 4; i++) { src[i] = 0x9e3779b9u * (uint32_t)i; dst[i] = 0xdeadbeefu; }
    static WsSim s; memset(&s, 0, sizeof s); s.p = &p; s.threads = OMEGA_PF_COPY_THREADS;
    uint64_t d = (uintptr_t)dst, sv = (uintptr_t)src;
    s.cbank[0x380 / 4] = (uint32_t)d; s.cbank[0x384 / 4] = (uint32_t)(d >> 32); s.cbank[0x388 / 4] = (uint32_t)sv; s.cbank[0x38c / 4] = (uint32_t)(sv >> 32); s.cbank[0x390 / 4] = U;
    s.ranges[0].lo = d; s.ranges[0].hi = d + sizeof dst; s.ranges[1].lo = sv; s.ranges[1].hi = sv + sizeof src; s.nranges = 2;
    int err = ws_run(&s, (U + OMEGA_PF_COPY_THREADS - 1) / OMEGA_PF_COPY_THREADS, 1);
    size_t bad = 0, past = 0;
    for (int i = 0; i < CAP * 4; i++) { if (i < U * 4 && dst[i] != src[i]) bad++; if (i >= U * 4 && dst[i] != 0xdeadbeefu) past++; }
    printf("sim copy: %s, peak %u registers, %zu wrong, %zu written past the end\n", err ? s.msg : "ran", p.regalloc.peak_gpr_usage, bad, past);
    return (err || bad || past) ? 1 : 0;
}

typedef struct { uint32_t iters, groups, threads; } BwSpec;
static int gen_bw(const OmegaPfKey *k, void *prog) { return omega_blackwell_codegen_tc_bw_prog(k->words[0], k->words[1], k->words[2], prog); }

int main(int argc, char **argv) {
    int fail = host_only();
    if (argc > 1 && !strcmp(argv[1], "--host-only")) { printf("GPU_PF_MEM: %s\n", fail ? "FAIL" : "PASS"); return fail ? 1 : 0; }
    if (fail) { printf("GPU_PF_MEM: FAIL\n"); return 1; }

    /* 1. round trip */
    const size_t RB = 96ull << 20;
    uint32_t *a = malloc(RB), *b = malloc(RB);
    if (!a || !b) return 1;
    for (size_t i = 0; i < RB / 4; i++) a[i] = (uint32_t)(i * 2654435761u) ^ 0x5bd1e995u;
    OmegaGpuMem dev;
    int rc = omega_gpu_pf_alloc_dev(RB, &dev);
    if (rc) { printf("alloc_dev: %s (%s)\n", omega_gpu_pf_rc_name(rc), omega_gpu_session_last_error()); printf("GPU_PF_MEM: FAIL\n"); return 1; }
    memset(b, 0xa5, RB);
    rc = omega_gpu_pf_upload(&dev, 0, a, RB);
    if (!rc) rc = omega_gpu_pf_download(&dev, 0, b, RB);
    size_t bad = 0; for (size_t i = 0; i < RB / 4; i++) if (a[i] != b[i]) bad++;
    printf("round trip 1 (%zu MB, cached device buffer): rc %s, %zu wrong words\n", RB >> 20, omega_gpu_pf_rc_name(rc), bad);
    if (rc || bad) fail++;
    for (size_t i = 0; i < RB / 4; i++) a[i] = ~a[i] + (uint32_t)i;
    memset(b, 0x5a, RB);
    rc = omega_gpu_pf_upload(&dev, 0, a, RB);
    if (!rc) rc = omega_gpu_pf_download(&dev, 0, b, RB);
    bad = 0; for (size_t i = 0; i < RB / 4; i++) if (a[i] != b[i]) bad++;
    printf("round trip 2 (rewritten by the GPU over cached lines): rc %s, %zu wrong words\n", omega_gpu_pf_rc_name(rc), bad);
    if (rc || bad) fail++;
    omega_gpu_pf_free(&dev);

    /* 2. bandwidth: uncached vs cached, distinct vs same */
    const uint32_t T = 256, grid = 384;
    for (int cached = 0; cached < 2 && !fail; cached++) for (int same = 0; same < 2 && !fail; same++) {
        const uint32_t iters = same ? 8u : 32u, groups = same ? 1u : grid;
        const size_t region = (size_t)iters * T * 128u, sb = region * groups, ob = (size_t)grid * T * 4u;
        OmegaGpuMem src = { 0 }, out = { 0 };
        uint32_t *pat = malloc(sb);
        for (size_t q = 0; q < sb / 4; q++) pat[q] = (uint32_t)(q * 2654435761u);
        rc = cached ? omega_gpu_pf_alloc_dev(sb, &src) : omega_gpu_pf_alloc_host(sb, &src);
        if (!rc) rc = omega_gpu_pf_alloc_host(ob, &out);
        if (!rc) { if (cached) rc = omega_gpu_pf_upload(&src, 0, pat, sb); else { memcpy(src.cpu, pat, sb); } }
        uint32_t *want = calloc((size_t)grid * T, 4);
        for (uint32_t c = 0; c < grid; c++) for (uint32_t t = 0; t < T; t++) { uint32_t x = 0; const uint32_t *r = pat + (size_t)(groups == 1 ? 0 : c) * region / 4;
            for (uint32_t it = 0; it < iters; it++) for (uint32_t q = 0; q < 8; q++) for (uint32_t s = 0; s < 4; s++) x ^= r[((size_t)it * T * 128u + (size_t)q * T * 16u + t * 16u) / 4 + s];
            want[(size_t)c * T + t] = x; }
        uint64_t ns[5] = { 0 }; size_t wrong = 0;
        OmegaPfKey key; memset(&key, 0, sizeof key); key.kind = 0x100u; key.words[0] = iters; key.words[1] = groups; key.words[2] = T;
        for (int r = 0; r < 5 && !rc; r++) {
            uint32_t *o = out.cpu; memset(o, 0, ob);
            uint32_t args[4] = { (uint32_t)src.va, (uint32_t)(src.va >> 32), (uint32_t)out.va, (uint32_t)(out.va >> 32) };
            omega_gpu_session_lock();
            OmegaPfKernel k; rc = omega_gpu_pf_open_locked();
            if (!rc) rc = omega_gpu_pf_kernel_locked(&key, gen_bw, &k);
            if (!rc && omega_gpu_session_sync_to_gpu(&out) != 0) rc = OMEGA_GPU_PF_CHIP_FAIL;
            if (!rc) rc = omega_gpu_pf_launch_locked(&k, T, grid, 1, 0, args, 4, &ns[r]);
            if (!rc && omega_gpu_session_sync_to_cpu(&out) != 0) rc = OMEGA_GPU_PF_CHIP_FAIL;
            omega_gpu_session_unlock();
            if (!rc) for (size_t q = 0; q < (size_t)grid * T; q++) if (o[q] != want[q]) wrong++;
        }
        qsort(ns, 5, 8, cmp_u64);
        const double bytes = (double)grid * iters * T * 128.0;
        printf("bw %s %s: rc %s, median %.1f us, %.1f GB/s, %zu wrong\n", cached ? "cached  " : "uncached", same ? "same-1MB" : "distinct", omega_gpu_pf_rc_name(rc), ns[2] / 1e3, bytes / ns[2], wrong);
        if (rc || wrong) { fail++; printf("  error: %s\n", omega_gpu_session_last_error()); }
        free(pat); free(want); omega_gpu_pf_free(&src); omega_gpu_pf_free(&out);
    }
    free(a); free(b);
    printf("GPU_PF_MEM: %s\n", fail ? "FAIL" : "PASS");
    return fail ? 1 : 0;
}
