/* SPDX-License-Identifier: Apache-2.0
 * Copyright 2026 Drake Stapleton (AIEN / aien-dev)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* omega_gpu_fp8_api.c: FP8 e4m3 GEMV entry points on the shared session (OWN-30B Cut 6 / Cut 5).
 * See the header. Submission path and buffer policy are the matmul API's: every buffer both sides
 * touch is GPU-uncached (nvos.h NVOS32_ATTR2_GPU_CACHEABLE: "For system memory this will not be
 * coherent with direct CPU mappings"), one launch = omega_gpu_session_launch (shader-cache
 * invalidate, constant bank, QMD0/QMD1, marker, L2_FLUSH_DIRTY, second marker on uncached memory). */
#include "omega_gpu_fp8_api.h"
#include "omega_gpu_mem_cached.h"
#include "omega_blackwell_codegen_moe.h"
#include "omega_blackwell_codegen_moe2.h" /* variant 2 (W-L2c) */
#include "omega_gpu_session.h"
#include "omega_gpu_fmt.h"
#include "omega_gpu_cmdlist.h"
#include "omega_blackwell_qmd.h" /* OMEGA_BW_QMD_MAX_SHARED_BYTES: the deep kernel's x stage must fit the launch descriptor */
#include <stdlib.h>
#include <string.h>

#define CACHE_SLOTS 16
#define MARKER_TIMEOUT_MS 20000ull
#define POISON_F32 (-999.0f)

struct OmegaGpuFp8Tensor {
    uint32_t experts, n, k;
    OmegaGpuMem w; /* experts * n * k e4m3 bytes, row-major */
    OmegaGpuMem s; /* experts * (n/128) * (k/128) f32 (scale * 2^56) */
};

typedef struct {
    bool valid, pinned;
    uint32_t variant; /* 1 or 2: which generator built the kernel (part of the cache key) */
    OmegaFp8GemvSpec spec;
    OmegaBlackwellKernel kernel;
    OmegaGpuMem code;
} Slot;

static struct {
    OmegaGpuScratch x_buf, y_buf;
    OmegaGpuMem tab; /* group table, OMEGA_GPU_FP8_MAX_GROUPS * 16 bytes */
    uint32_t cta_budget; int cta_budget_set;
    uint32_t lane_bytes, rows, warps; /* overrides, 0 = automatic */
    int depth_set; uint32_t depth;    /* depth override (depth_set: 0 = automatic) */
    uint32_t gpr_pad;                 /* registers declared beyond the kernel's peak (diagnosis knob) */
    int mutant, no_fold, sealed;
    int variant, variant_set; /* GEMV variant 1 or 2 (omega_gpu_fp8_set_variant / AIEN_OMEGA_FP8_GEMV), read once */
    int weights_cached; /* omega_gpu_fp8_set_weight_cache: uploads after it allocate GPU-cached (read-only weights) */
    int weights_cached_set; /* 0: never set, the first upload reads env AIEN_OMEGA_FP8_WEIGHT_CACHE (W-L2d: "1" = cached) */
    int sched, sched_set; /* CUT5: kernel schedule 0 (default) or 1 (pipelined decode + stall pass), variant 1 only; env AIEN_OMEGA_FP8_SCHED when never set */
    int coal, coal_set, wfirst, wfirst_set; /* CUT5 variants A (coalesced output) and B (weights before x), variant 1 only; env AIEN_OMEGA_FP8_COAL / AIEN_OMEGA_FP8_WFIRST when never set */
    int deep, deep_set; /* SPEED-1B: automatic deep kernel for single projections whenever it fits (1, default) or never (0); env AIEN_OMEGA_FP8_DEEP when never set */
    int bigk, bigk_set; /* qwen38/decode-speed: single projections with k > 4096 (Qwen3.8-27B) get 16 warps and a partial-depth deep kernel when x fits; default on, env AIEN_OMEGA_FP8_BIGK when never set */
    bool hooked;
    Slot cache[CACHE_SLOTS];
    unsigned next;
} g = { .cta_budget = OMEGA_GPU_FP8_MAX_CTAS };
static int g_rec; /* 1 while a record call runs (lock held): kernel_for must not evict */

const char *omega_gpu_fp8_last_error(void) { return omega_gpu_session_last_error(); }
const char *omega_gpu_fp8_rc_name(int rc) {
    switch (rc) {
    case OMEGA_GPU_FP8_OK: return "OK";
    case OMEGA_GPU_FP8_BAD_ARGS: return "BAD_ARGS";
    case OMEGA_GPU_FP8_TOO_LARGE: return "TOO_LARGE";
    case OMEGA_GPU_FP8_CODEGEN_FAIL: return "CODEGEN_FAIL";
    case OMEGA_GPU_FP8_CHIP_FAIL: return "CHIP_FAIL";
    case OMEGA_GPU_FP8_NAN_WEIGHT: return "NAN_WEIGHT";
    default: return "UNKNOWN";
    }
}

static void cache_clear_locked(void) {
    for (unsigned i = 0; i < CACHE_SLOTS; i++) {
        if (!g.cache[i].valid) continue;
        omega_gpu_session_free(&g.cache[i].code);
        memset(&g.cache[i], 0, sizeof g.cache[i]);
    }
    g.next = 0;
}
static void on_session_close(void) {
    cache_clear_locked();
    omega_gpu_session_scratch_free(&g.x_buf);
    omega_gpu_session_scratch_free(&g.y_buf);
    omega_gpu_session_free(&g.tab);
    g.sealed = 0;
}
static OmegaGpuSession *dev_open_locked(void) {
    if (!g.hooked) { (void)omega_gpu_session_on_close(on_session_close); g.hooked = true; }
    return omega_gpu_session_open();
}
void omega_gpu_fp8_cache_clear(void) { omega_gpu_session_lock(); cache_clear_locked(); omega_gpu_session_unlock(); }
void omega_gpu_fp8_set_shape(uint32_t lb, uint32_t rows, uint32_t warps) {
    omega_gpu_session_lock(); g.lane_bytes = lb; g.rows = rows; g.warps = warps; cache_clear_locked(); omega_gpu_session_unlock();
}
void omega_gpu_fp8_set_depth(int set, uint32_t depth) { omega_gpu_session_lock(); g.depth_set = set; g.depth = depth; cache_clear_locked(); omega_gpu_session_unlock(); }
void omega_gpu_fp8_set_gpr_pad(uint32_t pad) { omega_gpu_session_lock(); g.gpr_pad = pad; omega_gpu_session_unlock(); }
void omega_gpu_fp8_set_cta_budget(uint32_t ctas) { omega_gpu_session_lock(); g.cta_budget_set = ctas != 0; g.cta_budget = ctas ? ctas : OMEGA_GPU_FP8_MAX_CTAS; cache_clear_locked(); omega_gpu_session_unlock(); }
void omega_gpu_fp8_test_set_mutant(int m) { omega_gpu_session_lock(); g.mutant = m; cache_clear_locked(); omega_gpu_session_unlock(); }
void omega_gpu_fp8_test_set_no_fold(int on) { g.no_fold = on ? 1 : 0; }
void omega_gpu_fp8_seal(int on) { omega_gpu_session_lock(); g.sealed = on ? 1 : 0; omega_gpu_session_unlock(); }

/* ---- GEMV variant (W-L2c): 1 = omega_blackwell_codegen_moe.c (default), 2 = omega_blackwell_codegen_moe2.c (row-pipelined) ---- */
static int variant_now(void) {
    if (!g.variant_set) {
        const char *v = getenv("AIEN_OMEGA_FP8_GEMV");
        g.variant = (v && v[0] == '2' && v[1] == 0) ? 2 : 1;
        g.variant_set = 1;
    }
    return g.variant;
}
void omega_gpu_fp8_set_variant(int variant) {
    omega_gpu_session_lock();
    g.variant = variant == 2 ? 2 : 1; g.variant_set = 1;
    cache_clear_locked();
    omega_gpu_session_unlock();
}
int omega_gpu_fp8_variant(void) { return variant_now(); }

/* ---- kernel schedule (CUT5): 0 = emitted stalls (default), 1 = pipelined decode + dependency stall pass (variant 1 only) ---- */
static int sched_now(void) {
    if (!g.sched_set) {
        const char *v = getenv("AIEN_OMEGA_FP8_SCHED");
        g.sched = (v && v[0] == '1' && v[1] == 0) ? 1 : 0;
        g.sched_set = 1;
    }
    return g.sched;
}
void omega_gpu_fp8_set_sched(int sched) {
    omega_gpu_session_lock();
    g.sched = sched == 1 ? 1 : 0; g.sched_set = 1;
    cache_clear_locked();
    omega_gpu_session_unlock();
}
int omega_gpu_fp8_sched(void) { return sched_now(); }
/* ---- CUT5 variants A and B (variant 1 only): 0 = off, 1 = on. Never set: the env decides once; A (coal) defaults ON (env
 * value "0" turns it off; SPEED-A 2026-10-17: OWN-30B decode graph 24.49 -> 21.34 ms/token, tokens identical), B (wfirst)
 * defaults OFF (env value "1" turns it on; cold bench: not distinguishable from noise) ---- */
static int flag_now(int *v, int *set, const char *env, int dflt) {
    if (!*set) {
        const char *s = getenv(env);
        *v = (s && s[0] == '0' && s[1] == 0) ? 0 : (s && s[0] == '1' && s[1] == 0) ? 1 : dflt;
        *set = 1;
    }
    return *v;
}
static void flag_set(int *v, int *set, int on) { omega_gpu_session_lock(); *v = on == 1 ? 1 : 0; *set = 1; cache_clear_locked(); omega_gpu_session_unlock(); }
void omega_gpu_fp8_set_coal(int on) { flag_set(&g.coal, &g.coal_set, on); }
int omega_gpu_fp8_coal(void) { return flag_now(&g.coal, &g.coal_set, "AIEN_OMEGA_FP8_COAL", 1); }
void omega_gpu_fp8_set_wfirst(int on) { flag_set(&g.wfirst, &g.wfirst_set, on); }
int omega_gpu_fp8_wfirst(void) { return flag_now(&g.wfirst, &g.wfirst_set, "AIEN_OMEGA_FP8_WFIRST", 0); }
void omega_gpu_fp8_set_deep(int on) { flag_set(&g.deep, &g.deep_set, on); }
int omega_gpu_fp8_deep(void) { return flag_now(&g.deep, &g.deep_set, "AIEN_OMEGA_FP8_DEEP", 1); }
void omega_gpu_fp8_set_bigk(int on) { flag_set(&g.bigk, &g.bigk_set, on); }
int omega_gpu_fp8_bigk(void) { return flag_now(&g.bigk, &g.bigk_set, "AIEN_OMEGA_FP8_BIGK", 1); }
/* codegen by variant */
static int prog_for(int variant, const OmegaFp8GemvSpec *spec, BlackwellIRProgram *prog) {
    return variant == 2 ? omega_blackwell_codegen_fp8_gemv2_prog(spec, prog) : omega_blackwell_codegen_fp8_gemv_prog(spec, prog);
}
static int stats_for(int variant, const OmegaFp8GemvSpec *spec, uint32_t *insns, uint32_t *gprs) {
    return variant == 2 ? omega_fp8_gemv2_prog_stats(spec, insns, gprs) : omega_fp8_gemv_prog_stats(spec, insns, gprs);
}
static uint32_t shared_for(int variant, const OmegaFp8GemvSpec *spec) {
    return variant == 2 ? omega_fp8_gemv2_shared_bytes(spec) : omega_fp8_gemv_shared_bytes(spec);
}

/* ---- shape choice ---- */

/* Largest lane width dividing k (16 for k % 512 == 0: one LDG.E.128 per row per step); one row per warp:
 * the chip sweep (evidence/OWN30B-FP8-GEMV-7ce5dd4b-*, rows 1/2/4/8 one variable at a time) measured rows = 1
 * fastest at every shape (more warps and CTAs in flight on the same bytes beats fewer, wider warps). */
void omega_gpu_fp8_shape_for(uint32_t k, uint32_t n, uint32_t *lane_bytes, uint32_t *rows, uint32_t *warps) {
    uint32_t lb = k % 512u == 0 ? 16u : 8u; /* k % 256 == 0 is required (codegen refuses otherwise) */
    uint32_t r = 1; (void)n;
    if (lane_bytes) *lane_bytes = g.lane_bytes ? g.lane_bytes : lb;
    if (rows) *rows = g.rows ? g.rows : r;
    if (warps) *warps = g.warps ? g.warps : 8u;
}

/* ---- kernel cache (lock held, device open) ---- */

/* Registers declared in the QMD: the peak plus one, rounded up to a multiple of 8. Measured on the chip
 * (evidence OWN30B-FP8-GEMV-c4006391, kernel log Xid 13 "Out Of Range Register"): a kernel whose peak is
 * exactly 64 (top register R63) and declares 64 faults; declaring 65 runs. Every earlier kernel had a peak
 * that was not a multiple of 8, so the hardware's rounding hid the rule. */
static uint32_t gpr_declared(uint32_t peak) { return omega_bw_gpr_declared(peak); } /* peak + 2 rounded up to 8: the chip hung at peak + 1 (omega_blackwell_codegen.h) */

/* Register file: a CTA of 32 * warps threads must fit the SM's 65536 registers at the declared count. Measured
 * (chip, 2026-10-13, kernel log Xid 13): the deep k=2048 kernel (72 declared) at 32 warps faults, 73728 > 65536. */
static bool register_file_fits(uint32_t warps, uint32_t peak) {
    uint32_t decl = gpr_declared(peak);
    if ((uint64_t)32u * warps * decl <= 65536u) return true;
    char e[160]; omega_gpu_fmt(e, sizeof e, "fp8 kernel: %u warps x %u declared registers exceed the 65536-register file", warps, decl);
    omega_gpu_session_set_error(e);
    return false;
}

int omega_gpu_fp8_kernel_fits(const OmegaFp8GemvSpec *spec, uint32_t *declared) {
    uint32_t peak = 0;
    if (!spec || stats_for(variant_now(), spec, NULL, &peak) != 0) { omega_gpu_session_set_error("fp8 codegen refused the shape"); return OMEGA_GPU_FP8_CODEGEN_FAIL; }
    if (declared) *declared = gpr_declared(peak);
    return register_file_fits(spec->warps, peak) ? OMEGA_GPU_FP8_OK : OMEGA_GPU_FP8_TOO_LARGE;
}

static int kernel_for(const OmegaFp8GemvSpec *spec, Slot **out, bool *hit) {
    const uint32_t variant = (uint32_t)variant_now();
    for (unsigned i = 0; i < CACHE_SLOTS; i++) {
        Slot *s = &g.cache[i];
        if (s->valid && s->variant == variant && memcmp(&s->spec, spec, sizeof *spec) == 0) { *out = s; *hit = true; return OMEGA_GPU_FP8_OK; }
    }
    if (g.sealed) {
        char e[160];
        omega_gpu_fmt(e, sizeof e, "serving reservation exceeded: fp8 kernel k=%u lb=%u rows=%u warps=%u was not prepared",
                      spec->k, spec->lane_bytes, spec->rows, spec->warps);
        omega_gpu_session_set_error(e);
        return OMEGA_GPU_FP8_TOO_LARGE;
    }
    Slot *s = NULL;
    for (unsigned i = 0; i < CACHE_SLOTS && !s; i++) {
        Slot *c = &g.cache[(g.next + i) % CACHE_SLOTS];
        if (!c->valid || !c->pinned) { s = c; g.next = (unsigned)((c - g.cache) + 1) % CACHE_SLOTS; }
    }
    if (!s) { omega_gpu_session_set_error("fp8 kernel cache: every slot holds a prepared kernel"); return OMEGA_GPU_FP8_TOO_LARGE; }
    /* record path: a recorded launch may still point at the slot's code until the list is submitted */
    if (g_rec && s->valid) { omega_gpu_session_set_error("fp8 record: kernel cache miss would evict a resident kernel (warm up before recording)"); return OMEGA_GPU_FP8_TOO_LARGE; }
    BlackwellIRProgram *prog = omega_gpu_host_calloc(1, sizeof *prog);
    if (!prog) { omega_gpu_session_set_error("fp8 codegen: host memory"); return OMEGA_GPU_FP8_CODEGEN_FAIL; }
    int cg = prog_for((int)variant, spec, prog);
    size_t need = cg == 0 ? omega_bw_program_code_bytes(prog) : 0;
    if (cg != 0 || need == 0) { omega_gpu_host_free(prog); omega_gpu_session_set_error("fp8 codegen refused the shape"); return OMEGA_GPU_FP8_CODEGEN_FAIL; }
    if (!register_file_fits(spec->warps, prog->regalloc.peak_gpr_usage)) { omega_gpu_host_free(prog); return OMEGA_GPU_FP8_TOO_LARGE; }
    OmegaGpuMem code;
    if (omega_gpu_session_alloc_code(need, &code) != 0) { omega_gpu_host_free(prog); return OMEGA_GPU_FP8_CHIP_FAIL; }
    OmegaBlackwellKernel kernel;
    cg = omega_blackwell_kernel_encode_into(prog, code.cpu, need, &kernel);
    omega_gpu_host_free(prog);
    if (cg != 0 || !kernel.code || kernel.code_size != need) { omega_gpu_session_free(&code); return OMEGA_GPU_FP8_CODEGEN_FAIL; }
    if (omega_gpu_session_sync_to_gpu(&code) != 0) { omega_gpu_session_free(&code); return OMEGA_GPU_FP8_CHIP_FAIL; }
    if (s->valid) omega_gpu_session_free(&s->code);
    memset(s, 0, sizeof *s);
    s->valid = true; s->variant = variant; s->spec = *spec; s->kernel = kernel; s->code = code;
    *out = s; *hit = false;
    return OMEGA_GPU_FP8_OK;
}

/* Deep kernel (x' staged in shared memory, every load of the row issued at once) whenever the whole row fits
 * the budgets (rows * KS * lane_bytes <= 128, i.e. rows * k <= 4096) and the staging rounds are exact; else the step-by-step kernel. */
uint32_t omega_gpu_fp8_depth_for(const OmegaFp8GemvSpec *spec, uint32_t n, uint32_t groups) {
    if (g.depth_set) return g.depth;
    uint32_t ks = spec->k / (32u * spec->lane_bytes);
    /* big k (qwen38/decode-speed, opt-in AIEN_OMEGA_FP8_BIGK=1): the whole row cannot be in flight, so the largest step
     * count that divides KS within the 128-byte lane budget; x' still has to fit the launch descriptor's shared memory */
    const int partial = ks * spec->rows * spec->lane_bytes > 128u && spec->k > 4096u && groups == 1u && omega_gpu_fp8_bigk();
    if (partial) { do ks--; while (ks > 1u && (spec->k / (32u * spec->lane_bytes) % ks || ks * spec->rows * spec->lane_bytes > 128u)); if (ks < 2u) return 0; }
    if (ks * spec->rows * spec->lane_bytes > 128u || spec->k % (32u * spec->warps) || spec->k > 512u * spec->warps) return 0;
    /* Measured (evidence OWN30B-FP8-GEMV-fc05176a-*, main vs depth0 in one run, before variant A): the deep kernel won on
     * single projections up to a few MB (gate+up 1536x2048: 30 us vs 51; k_proj: 30 vs 44) and lost on the grouped
     * 8-expert launches (its per-CTA x staging re-reads 8 KB per CTA: 188 us vs 150) and on o_proj (79 vs 64), so the rule
     * was "classic above 4 MB". Re-measured under A (SPEED-1B, 2026-10-17, cold GPU-stamped, 5/5 interleaved,
     * ~/workspace/investigations/2026-10-17-SPEED-1B): deep wins on qkv 5120 x 2048 (10.5 MB) 51.8 us vs 60.4 (0.853 vs 0.732
     * of the 237.2 GB/s line, floor spread 0.078), so the size clause is gone. o_proj 2048 x 4096 is not a candidate: its deep
     * kernel's x stage exceeds the launch descriptor's shared-memory limit (qmd build (shared_bytes), 5/5), which is the bound
     * checked below. AIEN_OMEGA_FP8_DEEP=0 or set_deep(0) restores the classic kernel everywhere. */
    (void)n;
    if (groups != 1u || !omega_gpu_fp8_deep()) return 0;
    OmegaFp8GemvSpec deep = *spec; deep.depth = ks; deep.mutant = 0;
    if (omega_fp8_gemv_shared_bytes(&deep) > (partial ? OMEGA_BW_QMD_MAX_SHARED_BYTES : OMEGA_GPU_FP8_DEEP_MAX_SHARED_BYTES)) return 0;
    /* The register check builds the whole kernel (codegen + register allocation). spec_for runs on every launch and every
     * record, so the answer is remembered per spec: SPEED-1B graph A/B (2026-10-17) measured the unremembered check at
     * 853..1172 us of host record time per token (48 qkv records) against 45..113 us without the deep default. The answer is
     * a pure function of the spec (deterministic codegen), so the memo never needs clearing; spec_for zeroes the struct,
     * so memcmp over the padding is exact. */
    static struct { OmegaFp8GemvSpec spec; uint32_t fits; } memo[8];
    static unsigned memo_n, memo_next;
    for (unsigned i = 0; i < memo_n; i++)
        if (memcmp(&memo[i].spec, &deep, sizeof deep) == 0) return memo[i].fits ? ks : 0;
    uint32_t peak = 0;
    const uint32_t fits = omega_fp8_gemv_prog_stats(&deep, NULL, &peak) == 0 && (uint64_t)32u * spec->warps * gpr_declared(peak) <= 65536u;
    const unsigned at = memo_n < 8u ? memo_n++ : (memo_next++ % 8u);
    memo[at].spec = deep; memo[at].fits = fits;
    return fits ? ks : 0;
}

static void spec_for(uint32_t k, uint32_t n, uint32_t groups, OmegaFp8GemvSpec *spec) {
    memset(spec, 0, sizeof *spec);
    spec->k = k;
    omega_gpu_fp8_shape_for(k, n, &spec->lane_bytes, &spec->rows, &spec->warps);
    /* Grouped launches (8 experts): 32 warps per CTA and few CTAs per group (OMEGA_GPU_FP8_GROUPED_CTAS, one per SM
     * for 8 groups) measured fastest: down 2048x768 x 8 went from 181 us (256 x 8 CTAs of 8 warps) to 90 us
     * (4..6 x 8 CTAs of 32 warps); gate+up 146 -> 140 us (probe logs listed in ~/handoffs/own30b-K-report.md). */
    if (groups > 1u && !g.warps) spec->warps = 32u;
    /* big k (opt-in AIEN_OMEGA_FP8_BIGK=1): 16 warps per CTA (sweep 2026-10-19, ~/workspace/omega-q38-speed-work/runs/sweep1:
     * warps 16 beat 8 on every Qwen3.8-27B projection, 4 lost) */
    if (groups == 1u && k > 4096u && !g.warps && omega_gpu_fp8_bigk()) spec->warps = 16u;
    if (variant_now() == 2) {
        /* variant 2 (W-L2c): 16 warps per CTA (two CTAs per SM at its register count) for every launch unless overridden;
         * the whole row group is always in flight (depth = KS, informational: the generator ignores the field). */
        if (!g.warps) spec->warps = OMEGA_GPU_FP8_V2_WARPS;
        spec->depth = spec->k / (32u * spec->lane_bytes);
    } else {
        spec->sched = (uint32_t)sched_now();
        spec->coal = omega_gpu_fp8_coal() && spec->warps * spec->rows <= 32u; /* A: shapes whose round fits one store warp */
        spec->wfirst = (uint32_t)omega_gpu_fp8_wfirst();                       /* B: deep kernels ignore it */
        spec->depth = omega_gpu_fp8_depth_for(spec, n, groups);                /* after coal: the deep kernel's shared bytes include A's slots */
    }
    spec->mutant = g.mutant;
}
/* Host-only: the spec the automatic choice makes for a call (no chip, no cache; tests check the defaults with it). */
void omega_gpu_fp8_spec_for(uint32_t k, uint32_t n, uint32_t groups, OmegaFp8GemvSpec *spec) { spec_for(k, n, groups, spec); }

int omega_gpu_fp8_prepare(uint32_t k) {
    OmegaFp8GemvSpec spec; spec_for(k, 1u << 20, 1, &spec);
    omega_gpu_session_lock();
    int rc = OMEGA_GPU_FP8_OK;
    if (!dev_open_locked()) rc = OMEGA_GPU_FP8_CHIP_FAIL;
    else { Slot *s = NULL; bool hit = false; rc = kernel_for(&spec, &s, &hit); if (rc == OMEGA_GPU_FP8_OK) s->pinned = true; }
    omega_gpu_session_unlock();
    return rc;
}

/* ---- tensors ---- */

void omega_gpu_fp8_tensor_shape(const OmegaGpuFp8Tensor *t, uint32_t *experts, uint32_t *n, uint32_t *k) {
    if (experts) *experts = t ? t->experts : 0;
    if (n) *n = t ? t->n : 0;
    if (k) *k = t ? t->k : 0;
}

int omega_gpu_fp8_tensor_upload(uint32_t experts, uint32_t n, uint32_t k, const uint8_t *w, const uint16_t *scale, OmegaGpuFp8Tensor **out) {
    if (out) *out = NULL;
    if (!w || !scale || !out || experts == 0 || n == 0 || k == 0 || n % OMEGA_FP8_BLOCK || k % OMEGA_FP8_BLOCK) return OMEGA_GPU_FP8_BAD_ARGS;
    if (k > OMEGA_FP8_MAX_K) return OMEGA_GPU_FP8_TOO_LARGE;
    const size_t wbytes = (size_t)experts * n * k, scount = (size_t)experts * (n / OMEGA_FP8_BLOCK) * (k / OMEGA_FP8_BLOCK);
    size_t first = 0;
    if (omega_fp8_pack_scan_nan(w, wbytes, &first)) {
        char e[120]; omega_gpu_fmt(e, sizeof e, "fp8 upload: e4m3 NaN code at byte %zu", first);
        omega_gpu_session_set_error(e);
        return OMEGA_GPU_FP8_NAN_WEIGHT;
    }
    OmegaGpuFp8Tensor *t = omega_gpu_host_calloc(1, sizeof *t);
    if (!t) return OMEGA_GPU_FP8_BAD_ARGS;
    t->experts = experts; t->n = n; t->k = k;
    int rc = OMEGA_GPU_FP8_OK;
    omega_gpu_session_lock();
    if (!g.weights_cached_set) { const char *v = getenv("AIEN_OMEGA_FP8_WEIGHT_CACHE"); g.weights_cached = (v && v[0] == '1' && v[1] == 0) ? 1 : 0; g.weights_cached_set = 1; }
    int (*const alloc)(size_t, OmegaGpuMem *) = g.weights_cached ? omega_gpu_session_alloc_cached : omega_gpu_session_alloc;
    if (!dev_open_locked() || alloc(wbytes, &t->w) != 0 || alloc(scount * 4u, &t->s) != 0) rc = OMEGA_GPU_FP8_CHIP_FAIL;
    omega_gpu_session_unlock();
    if (rc == OMEGA_GPU_FP8_OK) {
        memcpy(t->w.cpu, w, wbytes);
        if (omega_fp8_pack_scales(scale, scount, t->s.cpu, g.no_fold) != 0) { omega_gpu_session_set_error("fp8 upload: NaN, inf or overflowing scale"); rc = OMEGA_GPU_FP8_BAD_ARGS; }
        else if (omega_gpu_session_sync_to_gpu(&t->w) != 0 || omega_gpu_session_sync_to_gpu(&t->s) != 0) rc = OMEGA_GPU_FP8_CHIP_FAIL;
    }
    if (rc != OMEGA_GPU_FP8_OK) { omega_gpu_fp8_tensor_free(t); return rc; }
    *out = t;
    return OMEGA_GPU_FP8_OK;
}

void omega_gpu_fp8_tensor_free(OmegaGpuFp8Tensor *t) {
    if (!t) return;
    omega_gpu_session_lock();
    if (t->w.cpu) omega_gpu_session_free(&t->w);
    if (t->s.cpu) omega_gpu_session_free(&t->s);
    omega_gpu_session_unlock();
    omega_gpu_host_free(t);
}

/* ---- the GEMV (lock held inside) ---- */

/* L310 (2026-10-24): CTAs per group of a grouped launch. A grouped CTA is 32 warps = 1024 threads; CC 12.x holds 1536
 * resident threads per SM (CUDA Programming Guide, Compute Capabilities, Table 30), so one such CTA fits per SM and the
 * GB10's 48 SMs (GB10 Superchip Whitepaper WP-12718-001_v02, Table 4) hold one wave of 48 CTAs. The compiled 6 was
 * measured on 8 groups (48 CTAs = one wave). The 35B launches 9 groups: 54 CTAs run as two waves, the second 6 CTAs wide;
 * 5 per group (45 = one wave) measured gate_up + down 6.962 -> 5.874 ms per 1-row step
 * (~/workspace/investigations/2026-10-24-l310-grouped-ctas). Env, read once like the other switches:
 *   AIEN_OMEGA_FP8_GROUPED_CTAS=1..256  exact CTAs per group (experiments; wins over the wave rule);
 *   AIEN_OMEGA_FP8_GROUPED_WAVE=1       wave rule: min(GROUPED_CTAS, SM_COUNT / groups), at least 1.
 * Neither set: the compiled GROUPED_CTAS, unchanged until the merge train has judged the wave arm. */
uint32_t omega_gpu_fp8_grouped_ctas_rule(uint32_t groups, uint32_t forced, int wave) {
    if (forced >= 1u && forced <= OMEGA_GPU_FP8_MAX_CTAS) return forced;
    if (!wave || groups == 0u) return OMEGA_GPU_FP8_GROUPED_CTAS;
    uint32_t one_wave = OMEGA_GPU_SM_COUNT / groups;
    if (one_wave == 0u) one_wave = 1u;
    return one_wave < OMEGA_GPU_FP8_GROUPED_CTAS ? one_wave : OMEGA_GPU_FP8_GROUPED_CTAS;
}
static uint32_t env_u32(const char *name) { /* 0 when unset, empty or not a plain number */
    const char *e = getenv(name); uint32_t n = 0;
    if (!e || !*e) return 0;
    for (; *e >= '0' && *e <= '9' && n < 1000u; e++) n = n * 10u + (uint32_t)(*e - '0');
    return *e ? 0 : n;
}
uint32_t omega_gpu_fp8_grouped_ctas_auto(uint32_t groups) {
    static uint32_t forced; static int wave, set;
    if (!set) { forced = env_u32("AIEN_OMEGA_FP8_GROUPED_CTAS"); const char *w = getenv("AIEN_OMEGA_FP8_GROUPED_WAVE"); wave = w && w[0] == '1' && !w[1]; set = 1; }
    return omega_gpu_fp8_grouped_ctas_rule(groups, forced, wave);
}

/* dev_tab / dev_x non-zero: the group table / the x' row already sit on the device (written by the router launch).
 * rec non-NULL (record path, caller holds the session lock): dev_tab, dev_x and dev_y are required, the launch is
 * appended to rec instead of submitted, and nothing is staged, poisoned or read back (the list's per-launch semaphore
 * and the caller's end-of-token checks replace the poison check). */
static int gemv_core(const OmegaGpuFp8Tensor *t, uint32_t groups, const uint32_t *ids, const float *x32, const uint16_t *x16,
                     uint32_t x_rows, float *y, OmegaGpuFp8Info *info, uint64_t dev_tab, uint64_t dev_x,
                     OmegaGpuCmdList *rec, uint64_t dev_y) {
    uint64_t t0 = omega_gpu_session_now_ns();
    if (info) memset(info, 0, sizeof *info);
    if (rec && (!dev_tab || !dev_x || !dev_y)) return OMEGA_GPU_FP8_BAD_ARGS;
    if (!t || !t->w.cpu || (!ids && !dev_tab) || (!x32 && !x16 && !dev_x) || (!y && !rec) || groups == 0 || (x_rows != 1 && x_rows != groups)) return OMEGA_GPU_FP8_BAD_ARGS;
    if (groups > OMEGA_GPU_FP8_MAX_GROUPS) return OMEGA_GPU_FP8_TOO_LARGE;
    if (!dev_tab) for (uint32_t gi = 0; gi < groups; gi++) if (ids[gi] >= t->experts) return OMEGA_GPU_FP8_BAD_ARGS;
    const uint32_t n = t->n, k = t->k;
    OmegaFp8GemvSpec spec; spec_for(k, n, groups, &spec);
    if (n % spec.rows) return OMEGA_GPU_FP8_CODEGEN_FAIL;
    const uint32_t RG = n / spec.rows;
    uint32_t grid_x = (RG + spec.warps - 1) / spec.warps;
    const int variant = variant_now();
    uint32_t budget = g.cta_budget_set ? g.cta_budget : (groups > 1u ? (variant == 2 ? OMEGA_GPU_FP8_V2_GROUPED_CTAS : omega_gpu_fp8_grouped_ctas_auto(groups)) : OMEGA_GPU_FP8_MAX_CTAS);
    /* big-k grid (qwen38/decode-speed): up to OMEGA_GPU_FP8_BIGK_CTAS CTAs, split so every warp gets the same number of row
     * groups, at least two (measured, runs/sweep5 and sweep7: budget 384 over 256 gave mix +3%, gate_up +2.6%; the n = 5120
     * shapes run best at 160 CTAs, two row groups per warp: o_proj 186 -> 198 GB/s) */
    const int bigk_grid = !g.cta_budget_set && groups == 1u && k > 4096u && omega_gpu_fp8_bigk();
    if (bigk_grid) { uint32_t waves = (grid_x + OMEGA_GPU_FP8_BIGK_CTAS - 1u) / OMEGA_GPU_FP8_BIGK_CTAS; if (waves < 2u) waves = 2u; grid_x = (grid_x + waves - 1u) / waves; budget = grid_x; }
    if (grid_x > budget) grid_x = budget;
    if (grid_x == 0) grid_x = 1;

    if (!rec) omega_gpu_session_lock();
#define UNLOCK() do { if (!rec) omega_gpu_session_unlock(); } while (0)
    int rc = OMEGA_GPU_FP8_OK;
    if (!dev_open_locked()) { UNLOCK(); return OMEGA_GPU_FP8_CHIP_FAIL; }
    if (!rec) { /* staging buffers: the record path reads and writes only the caller's device buffers */
        if (!g.tab.cpu && omega_gpu_session_alloc((size_t)OMEGA_GPU_FP8_MAX_GROUPS * 16u, &g.tab) != 0) { UNLOCK(); return OMEGA_GPU_FP8_CHIP_FAIL; }
        int src = omega_gpu_session_scratch(&g.x_buf, (size_t)x_rows * k * 4u);
        if (src == 0) src = omega_gpu_session_scratch(&g.y_buf, (size_t)groups * n * 4u);
        if (src != 0) { UNLOCK(); return src == OMEGA_GPU_SESSION_SCRATCH_RESERVED ? OMEGA_GPU_FP8_TOO_LARGE : OMEGA_GPU_FP8_CHIP_FAIL; }
    }
    Slot *ks = NULL; bool hit = false;
    g_rec = rec != NULL;
    rc = kernel_for(&spec, &ks, &hit);
    g_rec = 0;
    if (rc != OMEGA_GPU_FP8_OK) { UNLOCK(); return rc; }

    /* group table: {w_va, s_va} per group */
    uint64_t *tab = g.tab.cpu;
    const uint64_t wstride = (uint64_t)n * k, sstride = (uint64_t)(n / OMEGA_FP8_BLOCK) * (k / OMEGA_FP8_BLOCK) * 4u;
    if (!dev_tab) for (uint32_t gi = 0; gi < groups; gi++) { tab[2 * gi] = t->w.va + ids[gi] * wstride; tab[2 * gi + 1] = t->s.va + ids[gi] * sstride; }
    /* x' rows, y poisoned */
    float *xd = g.x_buf.mem.cpu, *yd = g.y_buf.mem.cpu;
    for (uint32_t r = 0; r < x_rows && !dev_x; r++) {
        if (x32) omega_fp8_pack_x_f32(x32 + (size_t)r * k, k, xd + (size_t)r * k);
        else omega_fp8_pack_x_bf16(x16 + (size_t)r * k, k, xd + (size_t)r * k);
    }
    if (!rec) {
        for (size_t i = 0; i < (size_t)groups * n; i++) yd[i] = POISON_F32;
        if (omega_gpu_session_sync_to_gpu(&g.tab) != 0 || omega_gpu_session_sync_to_gpu(&g.x_buf.mem) != 0 || omega_gpu_session_sync_to_gpu(&g.y_buf.mem) != 0) rc = OMEGA_GPU_FP8_CHIP_FAIL;
    }

    uint32_t args[OMEGA_FP8_ARGS_WORDS];
    const uint64_t tab_va = dev_tab ? dev_tab : g.tab.va, x_va = dev_x ? dev_x : g.x_buf.mem.va, y_va = rec ? dev_y : g.y_buf.mem.va;
    args[0] = (uint32_t)tab_va; args[1] = (uint32_t)(tab_va >> 32);
    args[2] = (uint32_t)x_va; args[3] = (uint32_t)(x_va >> 32);
    args[4] = (uint32_t)y_va; args[5] = (uint32_t)(y_va >> 32);
    args[6] = RG; args[7] = x_rows == 1 ? 0u : k * 4u; args[8] = n * 4u; args[9] = grid_x * spec.warps;
    if (info) {
        info->experts = t->experts; info->n = n; info->k = k; info->groups = groups;
        info->lane_bytes = spec.lane_bytes; info->rows_per_warp = spec.rows; info->warps_per_cta = spec.warps; info->depth = spec.depth;
        info->grid_x = grid_x; info->grid_y = groups; info->steps = omega_fp8_gemv_steps(&spec);
        info->weight_bytes = (uint64_t)groups * wstride; info->scale_bytes = (uint64_t)groups * sstride;
        info->kernel_cache_hit = hit;
        (void)omega_gpu_fmt(info->target_chip, sizeof info->target_chip, "%s", "NVIDIA DGX Spark (Grace Blackwell GB10)");
        info->sm_architecture = 121;
    }
    if (rc == OMEGA_GPU_FP8_OK) {
        OmegaGpuLaunch L = {
            .code_va = ks->code.va, .gpr_count = gpr_declared(ks->kernel.gpr_count) + g.gpr_pad,
            .threads_x = 32u * spec.warps, .threads_y = 1, .grid_x = grid_x, .grid_y = groups,
            .num_elements = (size_t)groups * n,
            .args = args, .n_args = OMEGA_FP8_ARGS_WORDS, .timeout_ms = MARKER_TIMEOUT_MS,
            .shared_bytes = shared_for(variant, &spec),
        };
        if (L.gpr_count < ks->kernel.gpr_count + 2u) { omega_gpu_session_set_error("declared registers below peak + 2: refusing to launch"); rc = OMEGA_GPU_FP8_CHIP_FAIL; }
        else if (rec) { if (omega_gpu_cmdlist_add(rec, &L) < 0) rc = OMEGA_GPU_FP8_CHIP_FAIL; }
        else if (omega_gpu_session_launch(&L, info ? &info->elapsed_ns : NULL, info ? &info->completion_marker : NULL) != 0) rc = OMEGA_GPU_FP8_CHIP_FAIL;
    }
    if (rc == OMEGA_GPU_FP8_OK && !rec && omega_gpu_session_sync_to_cpu(&g.y_buf.mem) != 0) rc = OMEGA_GPU_FP8_CHIP_FAIL;
    if (rc == OMEGA_GPU_FP8_OK && !rec) {
        size_t unchanged = 0;
        for (size_t i = 0; i < (size_t)groups * n; i++) if (yd[i] == POISON_F32) unchanged++;
        if (unchanged) { omega_gpu_session_set_error("output poison survived"); rc = OMEGA_GPU_FP8_CHIP_FAIL; }
        else memcpy(y, yd, (size_t)groups * n * sizeof *y);
    }
    UNLOCK();
    if (info) info->call_ns = omega_gpu_session_now_ns() - t0;
    return rc;
}
#undef UNLOCK

int omega_gpu_fp8_gemv_f32(const OmegaGpuFp8Tensor *t, uint32_t expert, const float *x, float *y, OmegaGpuFp8Info *info) {
    return gemv_core(t, 1, &expert, x, NULL, 1, y, info, 0, 0, NULL, 0);
}
int omega_gpu_fp8_gemv_bf16(const OmegaGpuFp8Tensor *t, uint32_t expert, const uint16_t *x, float *y, OmegaGpuFp8Info *info) {
    return gemv_core(t, 1, &expert, NULL, x, 1, y, info, 0, 0, NULL, 0);
}
int omega_gpu_fp8_gemv_grouped_f32(const OmegaGpuFp8Tensor *t, uint32_t groups, const uint32_t *ids, const float *x, uint32_t x_rows, float *y, OmegaGpuFp8Info *info) {
    return gemv_core(t, groups, ids, x, NULL, x_rows, y, info, 0, 0, NULL, 0);
}
int omega_gpu_fp8_gemv_grouped_dev(const OmegaGpuFp8Tensor *t, uint32_t groups, uint64_t tab_va, uint64_t x_va, float *y, OmegaGpuFp8Info *info) {
    if (!tab_va || !x_va) return OMEGA_GPU_FP8_BAD_ARGS;
    return gemv_core(t, groups, NULL, NULL, NULL, 1, y, info, tab_va, x_va, NULL, 0);
}
int omega_gpu_fp8_tensor_entry(const OmegaGpuFp8Tensor *t, uint32_t expert, uint64_t *w_va, uint64_t *s_va) {
    if (!t || !t->w.cpu || expert >= t->experts || !w_va || !s_va) return OMEGA_GPU_FP8_BAD_ARGS;
    *w_va = t->w.va + (uint64_t)expert * t->n * t->k;
    *s_va = t->s.va + (uint64_t)expert * (t->n / OMEGA_FP8_BLOCK) * (t->k / OMEGA_FP8_BLOCK) * 4u;
    return OMEGA_GPU_FP8_OK;
}

/* Record path (OWN-30B W-L2a decode graph): the launch of omega_gpu_fp8_gemv_grouped_dev appended to cl, every input
 * and the output in device memory. Caller holds omega_gpu_session_lock(). */
int omega_gpu_fp8_gemv_grouped_record(OmegaGpuCmdList *cl, const OmegaGpuFp8Tensor *t, uint32_t groups, uint64_t tab_va,
                                      uint64_t x_va, uint32_t x_rows, uint64_t y_va) {
    if (!cl) return OMEGA_GPU_FP8_BAD_ARGS;
    return gemv_core(t, groups, NULL, NULL, NULL, x_rows, NULL, NULL, tab_va, x_va, cl, y_va);
}

/* ---- weight cache policy (OWN-30B L2 kernel-time cut) ---- */
void omega_gpu_fp8_set_weight_cache(int on) { omega_gpu_session_lock(); g.weights_cached = on ? 1 : 0; g.weights_cached_set = 1; omega_gpu_session_unlock(); }
int omega_gpu_fp8_weight_cache(void) { return g.weights_cached; }

/* ---- host quantiser (qwen38/decode-speed: the FP8 lm_head) ---- */

/* f32 -> e4m3 (bias 7, max 448, no infinities), round to nearest, ties to even; |v| >= 448 (and NaN) saturate to 448:
 * 0x7F (NaN) is never produced. Pure integer/float bit work, no libm. */
uint8_t omega_gpu_fp8_e4m3_rne(float v) {
    uint32_t u; memcpy(&u, &v, 4);
    const uint8_t s = (uint8_t)((u >> 24) & 0x80u);
    u &= 0x7FFFFFFFu;
    float a; memcpy(&a, &u, 4);
    if (!(a < 448.0f)) return (uint8_t)(s | 0x7Eu);
    const int ex = (int)(u >> 23) - 127;
    if (ex < -6) { /* e4m3 subnormals: units of 2^-9 (code 8 is the smallest normal, 2^-6) */
        const float x = a * 512.0f; uint32_t xi = (uint32_t)x; const float fr = x - (float)xi;
        if (fr > 0.5f || (fr == 0.5f && (xi & 1u))) xi++;
        return (uint8_t)(s | xi);
    }
    const uint32_t m = u & 0x7FFFFFu, rb = (m >> 19) & 1u, st = m & 0x7FFFFu;
    uint32_t q = m >> 20, e = (uint32_t)(ex + 7);
    if (rb && (st || (q & 1u))) q++;
    if (q == 8u) { q = 0; e++; }
    uint32_t code = (e << 3) | q;
    if (code > 0x7Eu) code = 0x7Eu;
    return (uint8_t)(s | code);
}

int omega_gpu_fp8_quantize_bf16(uint32_t n, uint32_t k, const uint16_t *w, uint8_t *w8, uint16_t *scale_inv_bf16) {
    if (!w || !w8 || !scale_inv_bf16 || n == 0 || k == 0 || n % OMEGA_FP8_BLOCK || k % OMEGA_FP8_BLOCK) return OMEGA_GPU_FP8_BAD_ARGS;
    const uint32_t kb = k / OMEGA_FP8_BLOCK;
    for (uint32_t bn = 0; bn < n / OMEGA_FP8_BLOCK; bn++)
        for (uint32_t bk = 0; bk < kb; bk++) {
            float amax = 0.0f;
            for (uint32_t r = 0; r < OMEGA_FP8_BLOCK; r++) {
                const uint16_t *row = w + ((size_t)bn * OMEGA_FP8_BLOCK + r) * k + (size_t)bk * OMEGA_FP8_BLOCK;
                for (uint32_t c = 0; c < OMEGA_FP8_BLOCK; c++) {
                    uint32_t b = (uint32_t)(row[c] & 0x7FFFu) << 16; float f; memcpy(&f, &b, 4);
                    if (!(f <= 3.0e38f)) return OMEGA_GPU_FP8_BAD_ARGS; /* inf / NaN weight */
                    if (f > amax) amax = f;
                }
            }
            /* scale = amax / 448 rounded UP to bf16, so every |w / scale| <= 448 before rounding; an all-zero block gets 1 */
            uint16_t sb = 0x3F80u;
            if (amax > 0.0f) {
                const float q = amax / 448.0f; uint32_t b; memcpy(&b, &q, 4);
                sb = (uint16_t)((b >> 16) + ((b & 0xFFFFu) ? 1u : 0u));
            }
            scale_inv_bf16[(size_t)bn * kb + bk] = sb;
            uint32_t sbits = (uint32_t)sb << 16; float sc; memcpy(&sc, &sbits, 4);
            for (uint32_t r = 0; r < OMEGA_FP8_BLOCK; r++) {
                const size_t at = ((size_t)bn * OMEGA_FP8_BLOCK + r) * k + (size_t)bk * OMEGA_FP8_BLOCK;
                for (uint32_t c = 0; c < OMEGA_FP8_BLOCK; c++) {
                    uint32_t b = (uint32_t)w[at + c] << 16; float f; memcpy(&f, &b, 4);
                    w8[at + c] = omega_gpu_fp8_e4m3_rne(f / sc);
                }
            }
        }
    return OMEGA_GPU_FP8_OK;
}
