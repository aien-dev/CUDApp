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

/* test_omega_gpu_decode_attn_maxpos: the decode attention position bound (omega #535: OMEGA_GPU_DATTN_MAX_POS 131072,
 * the 128k context) on a FAKE OmegaGpuPlatform. Host program; opens no device, touches no GPU. The Makefile links a
 * sanitized copy of src/omega_gpu_decode_attn_api.c ahead of libomega_gpu.a, so every cache write, RoPE fill and
 * launch staging of the file under test is ASan/UBSan checked against the fake's exact-size buffers.
 *
 * The fake device completes every submission: completion-marker releases write their payload, and the launch "writes"
 * the attention output buffer (the one DATA buffer of 32 * 128 f32), so the API's poison check passes. It computes
 * nothing: this is a bound and sizing test, not a kernel result (the kernel's numbers are the chip battery's,
 * tools/run_gpu_decode_attn_chip.sh).
 *
 * Contract under test, at the cap C = OMEGA_GPU_DATTN_MAX_POS = 131072:
 *   - a cache of C positions allocates; C + 256 (the next legal multiple) is refused before any device work;
 *   - every buffer the cache owns is sized from max_pos: K|V planes 2 * (4 * C * 256 + slack), RoPE C * 512,
 *     split partials 4 * (C / 256) * 1040 * 4;
 *   - KV rows at 65535, 65536, 131071 round-trip; 131072 and 131073 are refused, as is a count crossing the cap;
 *   - RoPE rows 0, 65535, 65536, 99999, 131071 equal cos/sin of pos * theta^(-2i/128) from libm;
 *   - a decode step at pos 131071 launches with 512 splits (the whole partial buffer); 131072, 131073 are refused
 *     before any submission;
 *   - a cache of the old cap 65536 refuses pos 65536: the bound is the cache's own max_pos.
 * Each case runs in its own child process.
 */
#include "omega_gpu_decode_attn_api.h"
#include "omega_blackwell_codegen_attn.h"
#include "omega_gpu_session.h"
#include "omega_gpu_platform.h"
#include "omega_blackwell_submit.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

static int case_ok;
#define EXPECT(c) do { if (!(c)) { case_ok = 0; printf("  expect failed at line %d: %s\n", __LINE__, #c); } } while (0)

#define CAP OMEGA_GPU_DATTN_MAX_POS
#define THETA 1000000.0

/* ---- the fake platform ---- */
#define NB OMEGA_GPU_HANDLE_SLOTS
#define VA_BASE 0x100000000ull
#define VA_STRIDE (1ull << 31) /* one slot per 2 GiB of VA: the largest buffer here (256 MiB + slack) never overlaps the next */
#define OUT_BYTES ((uint64_t)OMEGA_GPU_DATTN_Q_HEADS * 128u * 4u)
static struct {
    OmegaGpuHandleTable t;
    void *mem[NB];
    OmegaGpuBufferDesc desc[NB];
    OmegaGpuDeviceHandle dev;
    OmegaGpuQueueHandle q;
    OmegaGpuBufferHandle ring;
    uint64_t ns;
    unsigned locks_held, double_locks, bad_unlocks, submits, allocs;
    uint64_t data_bytes[64]; /* sizes of the DATA buffers, in allocation order */
    unsigned n_data;
} fk;

static uint64_t f_now_ns(void *c) { (void)c; fk.ns += 1000u; return fk.ns; }
static uint64_t f_now_ms(void *c) { return f_now_ns(c) / 1000000u; }
static int f_wait(void *c, uint64_t ms) { (void)c; fk.ns += ms * 1000000u; return 0; }
static int f_lock(void *c, uint32_t id) {
    (void)c;
    if (id != OMEGA_GPU_SESSION_LOCK_ID) return OMEGA_GPU_PLAT_REFUSED_ARGS;
    if (fk.locks_held) { fk.double_locks++; return OMEGA_GPU_PLAT_REFUSED_BUSY; }
    fk.locks_held = 1;
    return 0;
}
static int f_unlock(void *c, uint32_t id) {
    (void)c;
    if (id != OMEGA_GPU_SESSION_LOCK_ID || !fk.locks_held) { fk.bad_unlocks++; return OMEGA_GPU_PLAT_REFUSED_ARGS; }
    fk.locks_held = 0;
    return 0;
}
static const char *f_diag(void *c) { (void)c; return ""; }
static int f_dev_open(void *c, uint32_t ord, OmegaGpuDeviceHandle *out) {
    (void)c;
    if (ord != 0 || fk.dev.generation) return OMEGA_GPU_PLAT_REFUSED_LIMIT;
    int rc = omega_gpu_handles_open(&fk.t, OMEGA_GPU_HANDLE_DEVICE, 0, 1, 0, &out->index, &out->generation);
    if (rc == 0) fk.dev = *out;
    return rc;
}
static int f_dev_close(void *c, OmegaGpuDeviceHandle d) {
    (void)c;
    int rc = omega_gpu_handles_check(&fk.t, d.index, d.generation, OMEGA_GPU_HANDLE_DEVICE);
    if (rc) return rc;
    (void)omega_gpu_handles_close(&fk.t, d.index, d.generation, OMEGA_GPU_HANDLE_DEVICE);
    memset(&fk.dev, 0, sizeof fk.dev);
    return 0;
}
static OmegaGpuPlatformCaps g_caps = {
    .abi_major = OMEGA_GPU_PLATFORM_ABI_MAJOR, .abi_minor = OMEGA_GPU_PLATFORM_ABI_MINOR,
    .services = OMEGA_GPU_SVC_REQUIRED_SESSION, .max_devices = 1, .max_buffers = 1000, .max_queues = 1, .max_inflight = 1,
    .max_buffer_bytes = 1ull << 30, .max_total_bytes = 1ull << 32,
};
static int f_alloc(void *c, OmegaGpuDeviceHandle d, const OmegaGpuBufferDesc *desc, OmegaGpuBufferHandle *o,
                   OmegaCpuView *cpu, OmegaGpuVa *g, OmegaGpuBusAddr *b) {
    (void)c;
    int rc = omega_gpu_handles_check(&fk.t, d.index, d.generation, OMEGA_GPU_HANDLE_DEVICE);
    if (!rc) rc = omega_gpu_platform_check_buffer(desc, &g_caps);
    if (!rc) rc = omega_gpu_handles_open(&fk.t, desc->kind, desc->bytes, 1000, 1ull << 32, &o->index, &o->generation);
    if (rc) return rc;
    void *m = calloc(1, desc->bytes); /* exact size: ASan sees any write past the end */
    if (!m) { (void)omega_gpu_handles_close(&fk.t, o->index, o->generation, desc->kind); return OMEGA_GPU_PLAT_REFUSED_LIMIT; }
    fk.mem[o->index] = m; fk.desc[o->index] = *desc;
    if (desc->kind == OMEGA_GPU_HANDLE_BUF_DATA && fk.n_data < 64) fk.data_bytes[fk.n_data++] = desc->bytes;
    fk.allocs++;
    cpu->ptr = m; cpu->len = desc->bytes; g->va = VA_BASE + (uint64_t)o->index * VA_STRIDE; b->iova = 0;
    return 0;
}
static int f_free(void *c, OmegaGpuBufferHandle h) {
    (void)c;
    if (h.index >= NB) return OMEGA_GPU_PLAT_REFUSED_BAD_HANDLE;
    int rc = omega_gpu_handles_close(&fk.t, h.index, h.generation, fk.t.kind[h.index]);
    if (rc) return rc;
    free(fk.mem[h.index]); fk.mem[h.index] = NULL;
    return 0;
}
static int buf_live(OmegaGpuBufferHandle h) { return h.index < NB && fk.t.live[h.index] && fk.t.generation[h.index] == h.generation; }
/* the host view of [va, va + len): NULL unless the whole range lies inside one live buffer */
static void *va_to_cpu(uint64_t va, uint64_t len) {
    for (uint32_t i = 0; i < NB; i++) {
        if (!fk.mem[i] || !fk.t.live[i]) continue;
        const uint64_t base = VA_BASE + (uint64_t)i * VA_STRIDE;
        if (va >= base && va - base <= fk.desc[i].bytes && len <= fk.desc[i].bytes - (va - base)) return (uint8_t *)fk.mem[i] + (va - base);
    }
    return NULL;
}
static int sync_check(OmegaGpuBufferHandle h, uint64_t off, uint64_t len) {
    if (!buf_live(h)) return OMEGA_GPU_PLAT_REFUSED_STALE_HANDLE;
    if (len == 0 || off > fk.desc[h.index].bytes || len > fk.desc[h.index].bytes - off) return OMEGA_GPU_PLAT_REFUSED_BAD_LENGTH;
    return 0;
}
static int f_c2g(void *c, OmegaGpuBufferHandle h, uint64_t off, uint64_t len) { (void)c; return sync_check(h, off, len); }
/* A read-back of the split-counter / intermediate-semaphore words finds them DONE, as the device leaves them. */
static int f_g2c(void *c, OmegaGpuBufferHandle h, uint64_t off, uint64_t len) {
    (void)c;
    int rc = sync_check(h, off, len);
    if (rc) return rc;
    uint32_t *w = (uint32_t *)((uint8_t *)fk.mem[h.index] + off);
    if (len == 4u) {                       /* the launch's QMD release semaphore: the kernel finished */
        w[0] = OMEGA_BW_SEMAPHORE_INTERMEDIATE_DONE;
    } else if (len % 16u == 0) {           /* a row of four-word semaphores */
        uint64_t i = 0, n = len / 16u;
        while (i < n && w[4 * i] == OMEGA_BW_SEMAPHORE_INTERMEDIATE_INIT) i++;
        if (i == n) for (i = 0; i < n; i++) w[4 * i] = OMEGA_BW_SEMAPHORE_INTERMEDIATE_DONE;
    }
    return 0;
}
static int f_poll(void *c, OmegaGpuBufferHandle h, uint64_t off, uint32_t want, uint64_t ms, uint32_t *seen) {
    (void)c;
    if (!buf_live(h)) return OMEGA_GPU_PLAT_REFUSED_STALE_HANDLE;
    if (!(fk.desc[h.index].rights & OMEGA_GPU_RIGHT_HOST_POLLED)) return OMEGA_GPU_PLAT_REFUSED_BAD_RIGHTS;
    const volatile uint32_t *w = (const volatile uint32_t *)((uint8_t *)fk.mem[h.index] + off);
    *seen = *w;
    if ((int32_t)(*w - want) >= 0) return 0;
    fk.ns += ms * 1000000u;
    return OMEGA_GPU_POLL_TIMED_OUT;
}
static int f_qopen(void *c, OmegaGpuDeviceHandle d, OmegaGpuBufferHandle ring, OmegaGpuQueueHandle *o) {
    (void)c;
    int rc = omega_gpu_handles_check(&fk.t, d.index, d.generation, OMEGA_GPU_HANDLE_DEVICE);
    if (!rc && (!buf_live(ring) || fk.t.kind[ring.index] != OMEGA_GPU_HANDLE_BUF_QUEUE)) rc = OMEGA_GPU_PLAT_REFUSED_BAD_KIND;
    if (!rc) rc = omega_gpu_handles_open(&fk.t, OMEGA_GPU_HANDLE_QUEUE, 0, 1, 0, &o->index, &o->generation);
    if (!rc) { fk.q = *o; fk.ring = ring; }
    return rc;
}
static int f_qclose(void *c, OmegaGpuQueueHandle q) {
    (void)c;
    int rc = omega_gpu_handles_close(&fk.t, q.index, q.generation, OMEGA_GPU_HANDLE_QUEUE);
    if (!rc) memset(&fk.q, 0, sizeof fk.q);
    return rc;
}
static uint32_t mthd(uint32_t subc, uint32_t m, uint32_t count) { return (2u << 28) | (count << 16) | (subc << 13) | (m >> 2); }
/* The fake device runs the stream: completion-marker releases (0x005c) write their payload, and the attention output
 * (the DATA buffer of 32 * 128 f32) is overwritten with zeros, the "kernel ran" the API's poison check looks for. */
static int f_submit(void *c, OmegaGpuQueueHandle q, const uint32_t *w, uint32_t n) {
    (void)c;
    int rc = omega_gpu_handles_check(&fk.t, q.index, q.generation, OMEGA_GPU_HANDLE_QUEUE);
    if (rc) return rc;
    if ((uint64_t)n * 4u > fk.desc[fk.ring.index].bytes) return OMEGA_GPU_PLAT_REFUSED_BAD_LENGTH;
    fk.submits++;
    for (uint32_t i = 0; i < NB; i++)
        if (fk.mem[i] && fk.t.live[i] && fk.desc[i].kind == OMEGA_GPU_HANDLE_BUF_DATA && fk.desc[i].bytes == OUT_BYTES) memset(fk.mem[i], 0, OUT_BYTES);
    for (uint32_t i = 0; i + 5 < n; i++)
        if (w[i] == mthd(0, 0x005c, 5)) {
            uint32_t *m = va_to_cpu((uint64_t)w[i + 1] | ((uint64_t)w[i + 2] << 32), 4);
            if (m) *m = w[i + 3];
            i += 5;
        }
    return 0;
}
static int f_retire(void *c, OmegaGpuQueueHandle q) { (void)c; (void)q; return 0; }
static void *f_halloc(void *c, size_t b) { (void)c; return malloc(b); }
static void f_hfree(void *c, void *p) { (void)c; free(p); }

static OmegaGpuPlatform g_plat = {
    .ctx = &fk, .time_now_ms = f_now_ms, .wait_ms = f_wait, .lock = f_lock, .unlock = f_unlock,
    .dma_alloc = f_alloc, .dma_free = f_free, .sync_cpu_to_gpu = f_c2g, .sync_gpu_to_cpu = f_g2c, .poll_word = f_poll,
    .device_open = f_dev_open, .device_close = f_dev_close, .diag_text = f_diag,
    .queue_open = f_qopen, .queue_close = f_qclose, .queue_submit = f_submit, .queue_retire = f_retire,
    .time_now_ns = f_now_ns, .host_alloc = f_halloc, .host_free = f_hfree,
};
static int install(void) {
    memset(&fk, 0, sizeof fk); omega_gpu_handles_init(&fk.t);
    return omega_gpu_session_set_platform(&g_plat, &g_caps);
}
#define EXPECT_LOCK_FREE() do { EXPECT(fk.locks_held == 0); EXPECT(fk.double_locks == 0); EXPECT(fk.bad_unlocks == 0); } while (0)

/* the session rounds every buffer up to a 4 KiB page */
static int has_data(uint64_t bytes) {
    bytes = (bytes + 0xfffu) & ~0xfffull;
    for (unsigned i = 0; i < fk.n_data; i++) if (fk.data_bytes[i] == bytes) return 1;
    return 0;
}
static OmegaGpuDecodeAttnCache *alloc_at(uint32_t max_pos) {
    OmegaGpuDecodeAttnCache *c = NULL;
    if (install() != OMEGA_GPU_PLAT_OK) { EXPECT(!"fake platform installs"); return NULL; } /* never fall back to the real driver */
    int rc = omega_gpu_decode_attn_cache_alloc(1, max_pos, THETA, &c);
    if (rc != OMEGA_GPU_DATTN_OK) printf("  cache_alloc(1, %u): %s (%s)\n", max_pos, omega_gpu_decode_attn_rc_name(rc), omega_gpu_decode_attn_last_error());
    EXPECT(rc == OMEGA_GPU_DATTN_OK && c != NULL);
    return rc == OMEGA_GPU_DATTN_OK ? c : NULL;
}

/* ---- cases ---- */

static void c_alloc_cap(void) {
    EXPECT(CAP == 131072u);
    OmegaGpuDecodeAttnCache *c = alloc_at(CAP);
    if (!c) return;
    const uint64_t plane = 4ull * CAP * OMEGA_DATTN_ROW_BYTES + OMEGA_DATTN_PLANE_SLACK;
    EXPECT(has_data(2 * plane));                                                         /* K | V planes */
    EXPECT(has_data((uint64_t)CAP * OMEGA_DATTN_ROPE_ROW_BYTES));                        /* RoPE table */
    EXPECT(has_data(4ull * (CAP / OMEGA_DATTN_CHUNK) * OMEGA_DATTN_PARTIAL_WORDS * 4u)); /* split partials, 512 splits */
    OmegaGpuDecodeAttnCacheView v;
    EXPECT(omega_gpu_decode_attn_cache_view(c, 0, &v) == OMEGA_GPU_DATTN_OK && v.max_pos == CAP && v.v_va - v.k_va == plane);
    omega_gpu_decode_attn_cache_free(c);
    EXPECT_LOCK_FREE();
    const unsigned allocs = fk.allocs;
    c = (OmegaGpuDecodeAttnCache *)1;
    EXPECT(omega_gpu_decode_attn_cache_alloc(1, CAP + OMEGA_DATTN_CHUNK, THETA, &c) == OMEGA_GPU_DATTN_BAD_ARGS && c == NULL);
    EXPECT(fk.allocs == allocs); /* refused before any device work */
    EXPECT_LOCK_FREE();
}

static void c_kv_rows(void) {
    OmegaGpuDecodeAttnCache *c = alloc_at(CAP);
    if (!c) return;
    static const uint32_t ok_pos[] = { 0, 65535, 65536, 131071 };
    uint16_t k[128], v[128], kr[128], vr[128];
    for (unsigned n = 0; n < sizeof ok_pos / sizeof ok_pos[0]; n++)
        for (uint32_t h = 0; h < OMEGA_GPU_DATTN_KV_HEADS; h++) {
            for (unsigned i = 0; i < 128; i++) { k[i] = (uint16_t)(ok_pos[n] * 7u + h * 131u + i); v[i] = (uint16_t)~k[i]; }
            EXPECT(omega_gpu_decode_attn_cache_write(c, 0, h, ok_pos[n], 1, k, v) == OMEGA_GPU_DATTN_OK);
            EXPECT(omega_gpu_decode_attn_cache_read(c, 0, h, ok_pos[n], kr, vr) == OMEGA_GPU_DATTN_OK);
            EXPECT(memcmp(k, kr, sizeof k) == 0 && memcmp(v, vr, sizeof v) == 0);
        }
    /* the rows really are where the view says: (kv_head * max_pos + pos) * 256 from the plane base */
    OmegaGpuDecodeAttnCacheView view;
    EXPECT(omega_gpu_decode_attn_cache_view(c, 0, &view) == OMEGA_GPU_DATTN_OK);
    const uint16_t *last = va_to_cpu(view.v_va + ((uint64_t)3 * CAP + 131071u) * 256u, 256);
    EXPECT(last && omega_gpu_decode_attn_cache_read(c, 0, 3, 131071, kr, vr) == OMEGA_GPU_DATTN_OK && memcmp(last, vr, 256) == 0);
    EXPECT(omega_gpu_decode_attn_cache_write(c, 0, 0, 131072, 1, k, v) == OMEGA_GPU_DATTN_BAD_ARGS);
    EXPECT(omega_gpu_decode_attn_cache_write(c, 0, 0, 131073, 1, k, v) == OMEGA_GPU_DATTN_BAD_ARGS);
    EXPECT(omega_gpu_decode_attn_cache_read(c, 0, 0, 131072, kr, vr) == OMEGA_GPU_DATTN_BAD_ARGS);
    EXPECT(omega_gpu_decode_attn_cache_read(c, 0, 0, 131073, kr, vr) == OMEGA_GPU_DATTN_BAD_ARGS);
    uint16_t k2[256] = {0}, v2[256] = {0};
    EXPECT(omega_gpu_decode_attn_cache_write(c, 0, 3, 131071, 2, k2, v2) == OMEGA_GPU_DATTN_BAD_ARGS); /* crosses the cap */
    omega_gpu_decode_attn_cache_free(c);
    EXPECT_LOCK_FREE();
}

static void c_rope_rows(void) {
    OmegaGpuDecodeAttnCache *c = alloc_at(CAP);
    if (!c) return;
    OmegaGpuDecodeAttnCacheView view;
    EXPECT(omega_gpu_decode_attn_cache_view(c, 0, &view) == OMEGA_GPU_DATTN_OK);
    EXPECT(va_to_cpu(view.rope_va, (uint64_t)CAP * OMEGA_DATTN_ROPE_ROW_BYTES) != NULL);         /* C rows ... */
    EXPECT(va_to_cpu(view.rope_va, (uint64_t)CAP * OMEGA_DATTN_ROPE_ROW_BYTES + 1u) == NULL);    /* ... and no more */
    static const uint32_t rows[] = { 0, 65535, 65536, 99999, 131071 };
    double worst = 0;
    for (unsigned n = 0; n < sizeof rows / sizeof rows[0]; n++) {
        const float *r = va_to_cpu(view.rope_va + (uint64_t)rows[n] * OMEGA_DATTN_ROPE_ROW_BYTES, OMEGA_DATTN_ROPE_ROW_BYTES);
        if (!r) { EXPECT(!"rope row inside the table"); continue; }
        for (unsigned i = 0; i < 64; i++) {
            const double a = (double)rows[n] * pow(THETA, -(double)(2 * i) / 128.0);
            const double dc = fabs((double)r[i] - cos(a)), ds = fabs((double)r[64 + i] - sin(a));
            if (dc > worst) worst = dc;
            if (ds > worst) worst = ds;
        }
    }
    printf("  rope rows 0..131071: worst |table - libm| %.3g\n", worst);
    EXPECT(worst <= 1e-6); /* f32 rounding of an exact value is <= 6e-8; 1e-6 leaves room for the pow ulps at pos 131071 */
    omega_gpu_decode_attn_cache_free(c);
    EXPECT_LOCK_FREE();
}

static float g_qkv[OMEGA_GPU_DATTN_QKV_WORDS], g_out[OMEGA_GPU_DATTN_Q_HEADS * 128];
static void c_decode_at_cap(void) {
    OmegaGpuDecodeAttnCache *c = alloc_at(CAP);
    if (!c) return;
    OmegaGpuDecodeAttnInfo info;
    int rc = omega_gpu_decode_attn(c, 0, 131071, g_qkv, 1e-6f, 1.0f, g_out, &info);
    if (rc) printf("  decode_attn(pos 131071): %s (%s)\n", omega_gpu_decode_attn_rc_name(rc), omega_gpu_decode_attn_last_error());
    EXPECT(rc == OMEGA_GPU_DATTN_OK);
    EXPECT(info.splits == CAP / OMEGA_DATTN_CHUNK && info.grid_x == 512u && info.grid_y == 4u);
    EXPECT(info.kv_bytes == 2ull * 4u * CAP * 256u);
    EXPECT_LOCK_FREE();
    const unsigned submits = fk.submits;
    EXPECT(omega_gpu_decode_attn(c, 0, 131072, g_qkv, 1e-6f, 1.0f, g_out, &info) == OMEGA_GPU_DATTN_BAD_ARGS);
    EXPECT(omega_gpu_decode_attn(c, 0, 131073, g_qkv, 1e-6f, 1.0f, g_out, &info) == OMEGA_GPU_DATTN_BAD_ARGS);
    EXPECT(fk.submits == submits); /* refused before any submission */
    EXPECT_LOCK_FREE();
    omega_gpu_decode_attn_cache_free(c);
}

static void c_old_cap_cache(void) {
    OmegaGpuDecodeAttnCache *c = alloc_at(65536u);
    if (!c) return;
    uint16_t k[128] = {0}, v[128] = {0};
    EXPECT(omega_gpu_decode_attn_cache_write(c, 0, 3, 65535, 1, k, v) == OMEGA_GPU_DATTN_OK);
    EXPECT(omega_gpu_decode_attn_cache_write(c, 0, 3, 65536, 1, k, v) == OMEGA_GPU_DATTN_BAD_ARGS);
    OmegaGpuDecodeAttnInfo info;
    EXPECT(omega_gpu_decode_attn(c, 0, 65535, g_qkv, 1e-6f, 1.0f, g_out, &info) == OMEGA_GPU_DATTN_OK && info.splits == 256u);
    EXPECT(omega_gpu_decode_attn(c, 0, 65536, g_qkv, 1e-6f, 1.0f, g_out, &info) == OMEGA_GPU_DATTN_BAD_ARGS);
    omega_gpu_decode_attn_cache_free(c);
    EXPECT_LOCK_FREE();
}

static const struct { const char *name; void (*fn)(void); } CASES[] = {
    { "cache of 131072 positions allocates, buffers sized for it; 131328 refused", c_alloc_cap },
    { "KV rows 65535/65536/131071 round-trip; 131072/131073 refused", c_kv_rows },
    { "RoPE rows up to 131071 match libm; table holds exactly 131072 rows", c_rope_rows },
    { "decode step at pos 131071 launches 512 splits; 131072/131073 refused", c_decode_at_cap },
    { "a 65536-position cache refuses pos 65536", c_old_cap_cache },
};

int main(int argc, char **argv) {
    const unsigned n = sizeof CASES / sizeof CASES[0];
    if (argc == 2) { /* one case in this process (debugging): its index */
        const unsigned i = (unsigned)atoi(argv[1]);
        if (i >= n) return 2;
        case_ok = 1; CASES[i].fn();
        printf("%s %s\n", case_ok ? "ok  " : "FAIL", CASES[i].name);
        return case_ok ? 0 : 1;
    }
    unsigned failed = 0;
    for (unsigned i = 0; i < n; i++) {
        fflush(stdout);
        pid_t pid = fork();
        if (pid == 0) { case_ok = 1; CASES[i].fn(); fflush(stdout); _exit(case_ok ? 0 : 1); }
        int status = 0;
        int ok = pid > 0 && waitpid(pid, &status, 0) == pid && WIFEXITED(status) && WEXITSTATUS(status) == 0;
        printf("%s %s\n", ok ? "ok  " : "FAIL", CASES[i].name);
        if (!ok) failed++;
    }
    printf("test_omega_gpu_decode_attn_maxpos: %u of %u cases passed: %s\n", n - failed, n, failed ? "FAIL" : "PASS");
    return failed ? 1 : 0;
}
