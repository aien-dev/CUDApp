/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */
/* GB10 RNG and FFT launchers; see omega_gpu_mathlibs_api.h. The kernels come from om/mathlibs/mathlibs_ir.om.
 * MIGRATION-PENDING C: launch, buffers and poison scan only. Successor: om/mathlibs/mathlibs_launch.om over the
 * session launch path, once a seeded unit can call it through omega-bridge; then this file is deleted. */
#include "omega_gpu_mathlibs_api.h"

#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "omega_blackwell_codegen.h"
#include "omega_gpu_om_ir.h"
#include "omega_gpu_session.h"

#define ML_POISON32 0xffbadbadu
#define ML_POISON64 0xffbadbadffbadbadull
#define ML_WAIT_MS 600000ull
#define ML_GOLDEN 11400714819323198485ull /* SplitMix64 increment, 0x9E3779B97F4A7C15 */

/* One kernel per (kind, a, b, mutant), device copy uploaded once and resident for the process. */
typedef struct {
    int used;
    uint64_t kind, a, b, mutant;
    uint32_t threads, declared;
    uint8_t *buf;
    OmegaBlackwellKernel kernel;
    OmegaGpuMem code;
} Slot;
static Slot g_cache[16];
static int g_mutant_which, g_mutant_level;
static OmegaGpuScratch g_in, g_out, g_tw;
static bool g_hooked;

static uint64_t now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000000000ULL + (uint64_t)t.tv_nsec;
}

static void slot_clear(Slot *s) {
    free(s->buf);
    omega_gpu_session_free(&s->code);
    memset(s, 0, sizeof *s);
}

static void cache_clear_locked(void) {
    for (size_t i = 0; i < sizeof g_cache / sizeof g_cache[0]; i++)
        if (g_cache[i].used) slot_clear(&g_cache[i]);
}

void omega_gpu_mathlibs_cache_clear(void) {
    omega_gpu_session_lock();
    cache_clear_locked();
    omega_gpu_session_unlock();
}

static void on_session_close(void) {
    cache_clear_locked();
    omega_gpu_session_scratch_free(&g_in);
    omega_gpu_session_scratch_free(&g_out);
    omega_gpu_session_scratch_free(&g_tw);
}

void omega_gpu_mathlibs_test_set_mutant(int which, int level) {
    omega_gpu_session_lock();
    g_mutant_which = which;
    g_mutant_level = level < 0 ? 0 : level > 2 ? 2 : level;
    cache_clear_locked();
    omega_gpu_session_unlock();
}

const char *omega_gpu_mathlibs_last_error(void) { return omega_gpu_session_last_error(); }

/* Caller holds the lock. Builds the kernel on a miss; uploads it when upload is set. */
static int kernel_for(uint64_t kind, uint64_t a, uint64_t b, uint32_t threads, bool upload, Slot **out, bool *hit) {
    uint64_t mutant = (g_mutant_which == 1 && kind == 0) || (g_mutant_which == 2 && kind == 1) ? (uint64_t)g_mutant_level : 0;
    Slot *s = NULL;
    *hit = false;
    for (size_t i = 0; i < sizeof g_cache / sizeof g_cache[0]; i++)
        if (g_cache[i].used && g_cache[i].kind == kind && g_cache[i].a == a && g_cache[i].b == b &&
            g_cache[i].mutant == mutant) {
            s = &g_cache[i];
            *hit = true;
            break;
        }
    if (!s) {
        BlackwellIRProgram *prog = (BlackwellIRProgram *)malloc(sizeof *prog);
        uint8_t *buf = (uint8_t *)malloc(OMEGA_BW_CODE_CAP_MAX);
        OmegaBlackwellKernel k;
        uint32_t declared = 0;
        int rc = (prog && buf) ? omega_gpu_om_ir_build(kind, a, b, mutant, threads, prog, buf, OMEGA_BW_CODE_CAP_MAX,
                                                       &k, &declared, NULL)
                               : OMEGA_OM_IR_REPLAY;
        free(prog);
        if (rc != OMEGA_OM_IR_OK) {
            free(buf);
            omega_gpu_session_set_error(omega_gpu_om_ir_rc_name(rc));
            return OMEGA_GPU_ML_CODEGEN_FAIL;
        }
        size_t slot = 0;
        for (size_t i = 0; i < sizeof g_cache / sizeof g_cache[0]; i++)
            if (!g_cache[i].used) { slot = i; break; }
        s = &g_cache[slot];
        if (s->used) slot_clear(s); /* full: evict slot 0 */
        s->used = 1;
        s->kind = kind; s->a = a; s->b = b; s->mutant = mutant;
        s->threads = threads; s->declared = declared;
        s->buf = buf;
        s->kernel = k;
    }
    if (upload && !s->code.cpu) {
        if (omega_gpu_session_alloc_code(s->kernel.code_size, &s->code) != 0) return OMEGA_GPU_ML_CHIP_FAIL;
        memcpy(s->code.cpu, s->kernel.code, s->kernel.code_size);
        __asm__ volatile("dsb sy" ::: "memory");
    }
    *out = s;
    return OMEGA_GPU_ML_OK;
}

static void info_begin(OmegaGpuMathInfo *info, const Slot *s, bool hit) {
    if (!info) return;
    memset(info, 0, sizeof *info);
    info->kernel_cache_hit = hit;
    info->insn_count = (uint32_t)s->kernel.insn_count;
    info->gpr_peak = s->kernel.gpr_count;
    info->gpr_declared = s->declared;
    memcpy(info->code_digest, s->kernel.code_digest, 32);
    info->threads_per_cta = s->threads;
}

static int codegen_only(uint64_t kind, uint64_t a, uint64_t b, uint32_t threads, OmegaGpuMathInfo *info) {
    omega_gpu_session_lock();
    Slot *s;
    bool hit;
    int rc = kernel_for(kind, a, b, threads, false, &s, &hit);
    if (rc == OMEGA_GPU_ML_OK) info_begin(info, s, hit);
    omega_gpu_session_unlock();
    return rc;
}

/* Caller holds the lock: open the shared device, build and upload the kernel. */
static int begin(uint64_t kind, uint64_t a, uint64_t b, uint32_t threads, Slot **s, OmegaGpuMathInfo *info) {
    if (!g_hooked) { (void)omega_gpu_session_on_close(on_session_close); g_hooked = true; }
    if (!omega_gpu_session_open()) return OMEGA_GPU_ML_CHIP_FAIL;
    bool hit;
    int rc = kernel_for(kind, a, b, threads, true, s, &hit);
    if (rc == OMEGA_GPU_ML_OK) info_begin(info, *s, hit);
    return rc;
}

static int launch(const Slot *s, const uint32_t *args, uint32_t n_args, uint32_t grid, size_t elements,
                  uint32_t shared_bytes, OmegaGpuMathInfo *info) {
    OmegaGpuLaunch L = {
        .code_va = s->code.va, .gpr_count = s->declared, .threads_x = s->threads, .threads_y = 1,
        .grid_x = grid, .grid_y = 1, .num_elements = elements, .args = args, .n_args = n_args,
        .timeout_ms = ML_WAIT_MS, .shared_bytes = shared_bytes,
    };
    uint64_t elapsed = 0;
    uint32_t marker = 0;
    if (omega_gpu_session_launch(&L, &elapsed, &marker) != 0) return OMEGA_GPU_ML_CHIP_FAIL;
    if (info) {
        info->chip_calls++;
        info->elapsed_ns += elapsed;
        info->completion_marker = marker;
        info->ctas_last_launch = grid;
    }
    return OMEGA_GPU_ML_OK;
}

static void put64(uint32_t *w, uint64_t v) { w[0] = (uint32_t)v; w[1] = (uint32_t)(v >> 32); }

/* ------------------------------------------------------------------ RNG */

static int rng_args_ok(OmegaGpuRngKind kind, uint32_t nstreams, uint32_t per) {
    if ((unsigned)kind > OMEGA_GPU_RNG_NORMAL || nstreams == 0 || per == 0) return OMEGA_GPU_ML_BAD_ARGS;
    if (nstreams > OMEGA_GPU_RNG_MAX_STREAMS || (uint64_t)nstreams * per > OMEGA_GPU_RNG_MAX_OUTPUTS)
        return OMEGA_GPU_ML_TOO_LARGE;
    return OMEGA_GPU_ML_OK;
}

int omega_gpu_rng_codegen(OmegaGpuRngKind kind, OmegaGpuMathInfo *info) {
    if ((unsigned)kind > OMEGA_GPU_RNG_NORMAL) return OMEGA_GPU_ML_BAD_ARGS;
    return codegen_only(0, (uint64_t)kind, 0, OMEGA_GPU_RNG_THREADS, info);
}

int omega_gpu_rng_generate(OmegaGpuRngKind kind, uint64_t seed, uint64_t stream0, uint32_t nstreams,
                           uint32_t offset, uint32_t per_stream, void *out, OmegaGpuMathInfo *info) {
    if (!out) return OMEGA_GPU_ML_BAD_ARGS;
    int rc = rng_args_ok(kind, nstreams, per_stream);
    if (rc != OMEGA_GPU_ML_OK) return rc;
    uint64_t t0 = now_ns();
    size_t count = (size_t)nstreams * per_stream;
    size_t esize = kind == OMEGA_GPU_RNG_U64 ? 8u : 4u;
    size_t bytes = count * esize;
    omega_gpu_session_lock();
    Slot *s;
    rc = begin(0, (uint64_t)kind, 0, OMEGA_GPU_RNG_THREADS, &s, info);
    if (rc == OMEGA_GPU_ML_OK && omega_gpu_session_scratch(&g_out, bytes) != 0) rc = OMEGA_GPU_ML_CHIP_FAIL;
    if (rc == OMEGA_GPU_ML_OK) {
        uint32_t *dev = (uint32_t *)g_out.mem.cpu;
        for (size_t i = 0; i < bytes / 4; i++) dev[i] = ML_POISON32;
        __asm__ volatile("dsb sy" ::: "memory");
        /* 0x380 out, 0x388 + 8 i base_i, 0x3a8 skip, 0x3ac per-stream count, 0x3b0 streams */
        uint32_t args[13];
        put64(&args[0], g_out.mem.va);
        uint64_t x = seed + stream0;
        for (int i = 0; i < 4; i++) { x += ML_GOLDEN; put64(&args[2 + 2 * i], x); }
        args[10] = offset;
        args[11] = per_stream;
        args[12] = nstreams;
        uint32_t grid = (nstreams + OMEGA_GPU_RNG_THREADS - 1) / OMEGA_GPU_RNG_THREADS;
        rc = launch(s, args, 13, grid, count, 0, info);
        if (rc == OMEGA_GPU_ML_OK) {
            memcpy(out, g_out.mem.cpu, bytes);
            uint64_t unwritten = 0;
            if (esize == 8) {
                for (size_t i = 0; i < count; i++) unwritten += ((const uint64_t *)out)[i] == ML_POISON64;
            } else {
                for (size_t i = 0; i < count; i++) unwritten += ((const uint32_t *)out)[i] == ML_POISON32;
            }
            if (info) info->unwritten_words = unwritten;
            if (unwritten) rc = OMEGA_GPU_ML_UNWRITTEN;
        }
    }
    omega_gpu_session_unlock();
    if (info) info->call_ns = now_ns() - t0;
    return rc;
}

/* ------------------------------------------------------------------ FFT */

static int fft_log2(uint32_t n) {
    if (n < OMEGA_GPU_FFT_MIN_N || n > OMEGA_GPU_FFT_MAX_N || (n & (n - 1)) != 0) return -1;
    int k = 0;
    while ((1u << k) < n) k++;
    return k;
}

int omega_gpu_fft_codegen(uint32_t n, int inverse, OmegaGpuMathInfo *info) {
    int lg = fft_log2(n);
    if (lg < 0 || (inverse != 0 && inverse != 1)) return OMEGA_GPU_ML_BAD_ARGS;
    return codegen_only(1, (uint64_t)lg, (uint64_t)inverse, n / 4, info);
}

int omega_gpu_fft_twiddles(uint32_t n, int inverse, float *tw) {
    static uint64_t words[OMEGA_OM_IR_WORDS];
    int lg = fft_log2(n);
    if (lg < 0 || !tw || (inverse != 0 && inverse != 1)) return OMEGA_GPU_ML_BAD_ARGS;
    omega_gpu_session_lock();
    int rc = omega_gpu_om_ir_data(2, (uint64_t)lg, (uint64_t)inverse, 0, words);
    if (rc == OMEGA_OM_IR_OK)
        for (uint32_t m = 0; m < n; m++) {
            uint32_t re = (uint32_t)words[m], im = (uint32_t)(words[m] >> 32);
            memcpy(&tw[2 * m], &re, 4);
            memcpy(&tw[2 * m + 1], &im, 4);
        }
    omega_gpu_session_unlock();
    return rc == OMEGA_OM_IR_OK ? OMEGA_GPU_ML_OK : OMEGA_GPU_ML_CODEGEN_FAIL;
}

int omega_gpu_fft_c2c(const float *in, float *out, uint32_t n, uint32_t batch, int inverse, OmegaGpuMathInfo *info) {
    int lg = fft_log2(n);
    if (lg < 0 || !in || !out || batch == 0 || (inverse != 0 && inverse != 1)) return OMEGA_GPU_ML_BAD_ARGS;
    size_t floats = (size_t)2 * n * batch;
    if (floats > ((size_t)1 << 28)) return OMEGA_GPU_ML_TOO_LARGE;
    if ((out < in + floats) && (in < out + floats)) return OMEGA_GPU_ML_BAD_ARGS;
    float *tw = (float *)malloc((size_t)2 * n * sizeof *tw);
    if (!tw) return OMEGA_GPU_ML_CHIP_FAIL;
    int rc = omega_gpu_fft_twiddles(n, inverse, tw);
    if (rc != OMEGA_GPU_ML_OK) { free(tw); return rc; }
    uint64_t t0 = now_ns();
    size_t bytes = floats * 4;
    omega_gpu_session_lock();
    Slot *s;
    rc = begin(1, (uint64_t)lg, (uint64_t)inverse, n / 4, &s, info);
    if (rc == OMEGA_GPU_ML_OK && (omega_gpu_session_scratch(&g_in, bytes) != 0 ||
                                  omega_gpu_session_scratch(&g_out, bytes) != 0 ||
                                  omega_gpu_session_scratch(&g_tw, (size_t)8 * n) != 0))
        rc = OMEGA_GPU_ML_CHIP_FAIL;
    if (rc == OMEGA_GPU_ML_OK) {
        memcpy(g_in.mem.cpu, in, bytes);
        memcpy(g_tw.mem.cpu, tw, (size_t)8 * n);
        uint32_t *dev = (uint32_t *)g_out.mem.cpu;
        for (size_t i = 0; i < floats; i++) dev[i] = ML_POISON32;
        __asm__ volatile("dsb sy" ::: "memory");
        for (uint32_t b0 = 0; b0 < batch && rc == OMEGA_GPU_ML_OK; b0 += OMEGA_GPU_FFT_MAX_CTAS) {
            uint32_t grid = batch - b0 < OMEGA_GPU_FFT_MAX_CTAS ? batch - b0 : OMEGA_GPU_FFT_MAX_CTAS;
            uint64_t off = (uint64_t)b0 * n * 8u;
            uint32_t args[6]; /* 0x380 in, 0x388 out, 0x390 twiddles */
            put64(&args[0], g_in.mem.va + off);
            put64(&args[2], g_out.mem.va + off);
            put64(&args[4], g_tw.mem.va);
            rc = launch(s, args, 6, grid, (size_t)grid * n * 2u, 8u * n, info);
        }
        if (rc == OMEGA_GPU_ML_OK) {
            memcpy(out, g_out.mem.cpu, bytes);
            uint64_t unwritten = 0;
            for (size_t i = 0; i < floats; i++) {
                uint32_t u;
                memcpy(&u, &out[i], 4);
                unwritten += u == ML_POISON32;
            }
            if (info) info->unwritten_words = unwritten;
            if (unwritten) rc = OMEGA_GPU_ML_UNWRITTEN;
        }
    }
    omega_gpu_session_unlock();
    free(tw);
    if (info) info->call_ns = now_ns() - t0;
    return rc;
}
