/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */
/*
 * gpu_q36_prefix_test: host test of src/omega_gpu_q36_prefix.{c,h} (DRA-86). No device: the copy / alloc / free
 * callbacks work on a host arena standing in for device memory, with the chip copy's rule (every length a multiple
 * of 4096 bytes) enforced.
 *   identical bytes after a save / scribble / restore round trip; rows past pos and other conversations untouched
 *   refusals: stale model tag, token digest, pos off by one turn (8 positions either way), blob length, KV and GDN
 *   shape; a refused restore copies nothing
 *   budget: OVER_BUDGET at save, least-recently-used eviction by make_room, AIEN_Q36_PREFIX_POOL_GB parsing
 *   the 35B layout: 739,901,440 bytes for 32k positions (Wolfram, in the DRA-86 PR)
 * AIEN_OMEGA_Q36_PF_MUTANT=10 (the hive's DRA-86 number: restore one 8-position block short at the boundary) must make
 * it FAIL. --negative-control runs the same checks with the refusal checks removed (the unmodified base had no
 * restore checks at all), which must FAIL too.
 * Verdict line last: "gpu_q36_prefix_test: N checks, F failed: PASS|FAIL".
 */
#include "omega_gpu_q36_prefix.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_checks, g_failed;
#define CHECK(cond, ...) do { g_checks++; if (!(cond)) { g_failed++; printf("FAIL: "); printf(__VA_ARGS__); printf("\n"); } } while (0)

/* ---- the fake device: one arena, VA = BASE + offset ---- */
#define BASE 0x100000ull
#define ARENA (8u << 20)
static uint8_t *g_mem;
static size_t g_top;           /* bump pointer for blobs */
static int g_allocs;           /* live blob allocations */
static int g_copies, g_unaligned, g_fail_copy_at = -1;

static uint8_t *at(uint64_t va, size_t n) {
    if (va < BASE || va - BASE > ARENA || n > ARENA - (va - BASE)) return NULL;
    return g_mem + (va - BASE);
}
static int f_alloc(void *ctx, size_t bytes, uint64_t *va) {
    (void)ctx;
    size_t n = (bytes + 4095u) & ~(size_t)4095u;
    if (n > ARENA - g_top) return -1;
    *va = BASE + g_top;
    g_top += n;
    g_allocs++;
    return 0;
}
static void f_free(void *ctx, uint64_t va, size_t bytes) { (void)ctx; (void)va; (void)bytes; g_allocs--; }
static int f_copy(void *ctx, uint64_t src, uint64_t dst, size_t bytes) {
    (void)ctx;
    if (g_fail_copy_at >= 0 && g_copies == g_fail_copy_at) return -1;
    g_copies++;
    if (bytes % OMEGA_Q36_PREFIX_COPY_ALIGN) g_unaligned++;
    uint8_t *s = at(src, bytes), *d = at(dst, bytes);
    if (!s || !d) return -1;
    memmove(d, s, bytes);
    return 0;
}
static const OmegaQ36PrefixOps OPS = {f_alloc, f_free, f_copy, NULL};

/* ---- the live state: 2 attention layers x 2 KV heads x 64 positions, 3 GDN layers of 4096 + 4096 bytes ---- */
#define ATTN 2u
#define HEADS 2u
#define MAXPOS 64u
#define GDN 3u
#define SB 4096u
#define TB 4096u
#define PLANE ((size_t)HEADS * MAXPOS * OMEGA_Q36_PREFIX_ROW_BYTES)
#define LIVE_BYTES ((size_t)GDN * (SB + TB) + (size_t)ATTN * 2u * PLANE)

static OmegaQ36PrefixLive live_at(uint64_t off) {
    OmegaQ36PrefixLive l;
    memset(&l, 0, sizeof l);
    l.attn_layers = ATTN; l.kv_heads = HEADS; l.max_pos = MAXPOS;
    l.gdn_layers = GDN; l.state_bytes = SB; l.tail_bytes = TB;
    l.state_va = BASE + off;
    l.tail_va = l.state_va + (size_t)GDN * SB;
    for (uint32_t i = 0; i < ATTN; i++) {
        l.k_va[i] = l.tail_va + (size_t)GDN * TB + (size_t)i * 2u * PLANE;
        l.v_va[i] = l.k_va[i] + PLANE;
    }
    return l;
}
static void fill(uint64_t off, uint32_t seed) {
    uint32_t x = seed * 2654435761u + 1u;
    for (size_t i = 0; i < LIVE_BYTES; i++) { x ^= x << 13; x ^= x >> 17; x ^= x << 5; g_mem[off + i] = (uint8_t)x; }
}
/* bytes of the live block at off that differ from ref, split into rows < pos and the rest */
static void diff(uint64_t off, const uint8_t *ref, uint32_t pos, size_t *gdn, size_t *below, size_t *above) {
    *gdn = *below = *above = 0;
    const uint8_t *m = g_mem + off;
    size_t g = (size_t)GDN * (SB + TB);
    for (size_t i = 0; i < g; i++) *gdn += m[i] != ref[i];
    for (size_t i = g; i < LIVE_BYTES; i++) {
        size_t row = ((i - g) % PLANE) / OMEGA_Q36_PREFIX_ROW_BYTES % MAXPOS;
        if (m[i] != ref[i]) { if (row < pos) (*below)++; else (*above)++; }
    }
}

static void digest_of(uint8_t d[32], uint32_t salt, uint32_t pos) {
    for (int i = 0; i < 32; i++) d[i] = (uint8_t)(salt * 31u + pos * 7u + (uint32_t)i);
}

int main(int argc, char **argv) {
    /* --host-only: the only tier */
    const char *mv = getenv("AIEN_OMEGA_Q36_PF_MUTANT");
    int mutant = mv && atoi(mv) == 10 ? 1 : 0;
    for (int i = 1; i < argc; i++)
        if (!strcmp(argv[i], "--negative-control")) mutant = 2;
    omega_q36_prefix_test_set_mutant(mutant);
    if (mutant) printf("gpu_q36_prefix_test: %s\n", mutant == 1 ? "mutant 10 (restore off by one block)" : "negative control (no refusal checks)");
    g_mem = calloc(1, ARENA);
    if (!g_mem) return 2;
    const uint64_t L1 = 0, L2 = LIVE_BYTES; /* two conversations' live states */
    g_top = 2u * LIVE_BYTES;
    uint8_t *h1 = malloc(LIVE_BYTES), *h2 = malloc(LIVE_BYTES), *hc = malloc(LIVE_BYTES);
    if (!h1 || !h2 || !hc) return 2;
    OmegaQ36PrefixLive live = live_at(L1);
    uint8_t tag[32], tag2[32], dg[32], dg_other[32];
    memset(tag, 0xa5, 32); memset(tag2, 0xa5, 32); tag2[31] ^= 1;
    digest_of(dg, 1, 48); digest_of(dg_other, 2, 48);
    size_t g, lo, hi;

    /* 1. layout */
    size_t blob = omega_q36_prefix_blob_bytes(&live, 48);
    CHECK(blob == (size_t)GDN * (SB + TB) + (size_t)ATTN * 2u * HEADS * 48u * 512u, "blob bytes %zu", blob);
    CHECK(omega_q36_prefix_blob_bytes(&live, 44) == 0, "pos 44 is not 8-aligned");
    CHECK(omega_q36_prefix_blob_bytes(&live, MAXPOS + 8) == 0, "pos past max_pos");
    OmegaQ36PrefixLive big;
    memset(&big, 0, sizeof big);
    big.attn_layers = 10; big.kv_heads = 2; big.max_pos = 33792; big.gdn_layers = 30;
    big.state_bytes = 32u * 128u * 128u * 4u; big.tail_bytes = 32u * 6144u;
    CHECK(omega_q36_prefix_blob_bytes(&big, 32768) == 739901440u, "35B 32k blob %zu", omega_q36_prefix_blob_bytes(&big, 32768));
    CHECK(omega_q36_prefix_blob_bytes(&big, 32767) == 0, "35B pos 32767 refused");
    big.state_bytes = 32u * 128u * 128u * 2u; /* bf16 state */
    CHECK(omega_q36_prefix_blob_bytes(&big, 32768) == 739901440u - 31457280u, "35B bf16 32k blob");

    /* 2. live from the real views */
    OmegaGpuDecodeAttn256CacheView v[ATTN];
    memset(v, 0, sizeof v);
    for (uint32_t i = 0; i < ATTN; i++) { v[i].kv_heads = HEADS; v[i].max_pos = MAXPOS; v[i].k_va = live.k_va[i]; v[i].v_va = live.v_va[i]; }
    OmegaGpuGdnStateView gv;
    memset(&gv, 0, sizeof gv);
    gv.layers = GDN; gv.state_bytes = SB; gv.tail_bytes = TB; gv.state_va = live.state_va; gv.tail_va = live.tail_va;
    OmegaQ36PrefixLive lv;
    CHECK(omega_q36_prefix_live_from_views(&lv, v, ATTN, &gv) == 0 && !memcmp(&lv, &live, sizeof lv), "live from views");
    v[1].kv_heads = 4;
    CHECK(omega_q36_prefix_live_from_views(&lv, v, ATTN, &gv) == OMEGA_Q36_PREFIX_SHAPE_MISMATCH, "mixed head counts refused");

    /* 3. round trip */
    OmegaQ36PrefixPool pool;
    omega_q36_prefix_pool_init(&pool, 1u << 20);
    fill(L1, 1); fill(L2, 9);
    memcpy(h1, g_mem + L1, LIVE_BYTES);
    memcpy(hc, g_mem + L2, LIVE_BYTES);
    OmegaQ36PrefixSnap s;
    memset(&s, 0, sizeof s);
    int rc = omega_q36_prefix_save(&pool, &OPS, &live, tag, dg, 48, &s);
    CHECK(rc == 0 && s.blob_bytes == blob && pool.resident == blob && pool.snaps == 1, "save: %s", omega_q36_prefix_rc_name(rc));
    CHECK(!memcmp(g_mem + L1, h1, LIVE_BYTES), "save leaves the live state alone");
    fill(L1, 2);
    memcpy(h2, g_mem + L1, LIVE_BYTES);
    diff(L1, h1, 48, &g, &lo, &hi);
    CHECK(g > 0 && lo > 0, "control: the scribbled state differs from the saved one (%zu, %zu)", g, lo);
    rc = omega_q36_prefix_restore(&pool, &OPS, &s, &live, tag, dg, 48);
    CHECK(rc == 0, "restore: %s", omega_q36_prefix_rc_name(rc));
    diff(L1, h1, 48, &g, &lo, &hi);
    CHECK(g == 0, "GDN state and tail identical after restore (%zu bytes differ)", g);
    CHECK(lo == 0, "K/V rows 0..47 identical after restore (%zu bytes differ)", lo);
    diff(L1, h2, 48, &g, &lo, &hi);
    CHECK(hi == 0, "K/V rows 48..63 untouched by the restore (%zu bytes differ)", hi);
    CHECK(!memcmp(g_mem + L2, hc, LIVE_BYTES), "the other conversation is untouched");
    CHECK(g_unaligned == 0, "every copy a multiple of 4096 bytes (%d not)", g_unaligned);
    CHECK(g_copies == 2 * (2 + (int)(ATTN * 2u * HEADS)), "copies %d", g_copies);
    /* restore into the other conversation's live state: same bytes there */
    OmegaQ36PrefixLive live2 = live_at(L2);
    CHECK(omega_q36_prefix_restore(&pool, &OPS, &s, &live2, tag, dg, 48) == 0, "restore into conversation 2");
    diff(L2, h1, 48, &g, &lo, &hi);
    CHECK(g == 0 && lo == 0, "conversation 2 holds the saved prefix");
    diff(L2, hc, 48, &g, &lo, &hi);
    CHECK(hi == 0, "conversation 2 rows past pos keep their own bytes");

    /* 4. refusals: each copies nothing and leaves the live state as it was */
    struct { const char *what; const uint8_t *tag, *dg; uint32_t pos; int want; } R[] = {
        {"stale model tag", tag2, dg, 48, OMEGA_Q36_PREFIX_STALE_MODEL},
        {"other tokens", tag, dg_other, 48, OMEGA_Q36_PREFIX_TOKEN_MISMATCH},
        {"one turn short (pos 40)", tag, dg, 40, OMEGA_Q36_PREFIX_LENGTH_MISMATCH},
        {"one turn long (pos 56)", tag, dg, 56, OMEGA_Q36_PREFIX_LENGTH_MISMATCH},
    };
    uint64_t refused0 = pool.refused;
    for (size_t i = 0; i < sizeof R / sizeof R[0]; i++) {
        fill(L1, 3u + (uint32_t)i);
        memcpy(h2, g_mem + L1, LIVE_BYTES);
        int c0 = g_copies;
        rc = omega_q36_prefix_restore(&pool, &OPS, &s, &live, R[i].tag, R[i].dg, R[i].pos);
        CHECK(rc == R[i].want, "%s: %s, want %s", R[i].what, omega_q36_prefix_rc_name(rc), omega_q36_prefix_rc_name(R[i].want));
        CHECK(g_copies == c0 && !memcmp(g_mem + L1, h2, LIVE_BYTES), "%s: the live state was touched", R[i].what);
    }
    OmegaQ36PrefixLive bad = live;
    bad.kv_heads = 1;
    rc = omega_q36_prefix_restore(&pool, &OPS, &s, &bad, tag, dg, 48);
    CHECK(rc == OMEGA_Q36_PREFIX_SHAPE_MISMATCH, "KV head count: %s", omega_q36_prefix_rc_name(rc));
    bad = live;
    bad.state_bytes = 2u * SB;
    rc = omega_q36_prefix_restore(&pool, &OPS, &s, &bad, tag, dg, 48);
    CHECK(rc == OMEGA_Q36_PREFIX_SHAPE_MISMATCH, "GDN state bytes: %s", omega_q36_prefix_rc_name(rc));
    bad = live;
    bad.max_pos = 40;
    rc = omega_q36_prefix_restore(&pool, &OPS, &s, &bad, tag, dg, 48);
    CHECK(rc == OMEGA_Q36_PREFIX_LENGTH_MISMATCH, "pos past the live max_pos: %s", omega_q36_prefix_rc_name(rc));
    OmegaQ36PrefixSnap torn = s;
    torn.blob_bytes -= 4096;
    rc = omega_q36_prefix_restore(&pool, &OPS, &torn, &live, tag, dg, 48);
    CHECK(rc == OMEGA_Q36_PREFIX_LENGTH_MISMATCH, "blob length: %s", omega_q36_prefix_rc_name(rc));
    CHECK(pool.refused == refused0 + 8, "refused counter %llu", (unsigned long long)(pool.refused - refused0));

    /* 5. save refusals and a failed copy */
    OmegaQ36PrefixSnap t;
    memset(&t, 0, sizeof t);
    CHECK(omega_q36_prefix_save(&pool, &OPS, &live, tag, dg, 44, &t) == OMEGA_Q36_PREFIX_BAD_ARGS, "save at pos 44");
    CHECK(omega_q36_prefix_save(&pool, &OPS, &live, tag, dg, 72, &t) == OMEGA_Q36_PREFIX_BAD_ARGS, "save past max_pos");
    CHECK(omega_q36_prefix_save(&pool, &OPS, &live, tag, dg, 48, &s) == OMEGA_Q36_PREFIX_BAD_ARGS, "save over a held snapshot");
    int a0 = g_allocs;
    g_fail_copy_at = g_copies + 3;
    rc = omega_q36_prefix_save(&pool, &OPS, &live, tag, dg, 48, &t);
    g_fail_copy_at = -1;
    CHECK(rc == OMEGA_Q36_PREFIX_COPY_FAILED && g_allocs == a0 && pool.resident == blob && !t.blob_va, "failed copy: %s", omega_q36_prefix_rc_name(rc));

    /* 6. budget and least-recently-used eviction */
    OmegaQ36PrefixSnap S[3];
    memset(S, 0, sizeof S);
    omega_q36_prefix_drop(&pool, &OPS, &s);
    CHECK(pool.resident == 0 && pool.snaps == 0 && g_allocs == 0, "drop frees the blob");
    omega_q36_prefix_pool_init(&pool, 2u * blob + 4095u);
    uint8_t d0[32], d1[32], d2[32];
    digest_of(d0, 10, 48); digest_of(d1, 11, 48); digest_of(d2, 12, 48);
    CHECK(omega_q36_prefix_save(&pool, &OPS, &live, tag, d0, 48, &S[0]) == 0, "save A");
    CHECK(omega_q36_prefix_save(&pool, &OPS, &live, tag, d1, 48, &S[1]) == 0, "save B");
    a0 = g_allocs;
    rc = omega_q36_prefix_save(&pool, &OPS, &live, tag, d2, 48, &S[2]);
    CHECK(rc == OMEGA_Q36_PREFIX_OVER_BUDGET && g_allocs == a0 && pool.resident == 2u * blob, "save C over budget: %s", omega_q36_prefix_rc_name(rc));
    CHECK(omega_q36_prefix_restore(&pool, &OPS, &S[0], &live, tag, d0, 48) == 0, "A used again");
    rc = omega_q36_prefix_make_room(&pool, &OPS, S, 3, blob);
    CHECK(rc == 1 && S[0].blob_va && !S[1].blob_va, "make_room drops B (least recently used): %d", rc);
    CHECK(omega_q36_prefix_save(&pool, &OPS, &live, tag, d2, 48, &S[2]) == 0, "save C after eviction");
    CHECK(pool.resident == 2u * blob && pool.resident <= pool.budget && pool.evicted == 1 && pool.snaps == 2, "resident %llu",
          (unsigned long long)pool.resident);
    CHECK(omega_q36_prefix_make_room(&pool, &OPS, S, 3, 3u * blob) == OMEGA_Q36_PREFIX_OVER_BUDGET && pool.snaps == 2,
          "a need over the whole budget drops nothing");
    CHECK(omega_q36_prefix_make_room(&pool, &OPS, S, 1, 2u * blob) == OMEGA_Q36_PREFIX_OVER_BUDGET && pool.snaps == 2,
          "listed snapshots too small to free enough drop nothing");

    /* 7. the budget knob */
    uint64_t b = 0;
    unsetenv(OMEGA_Q36_PREFIX_POOL_ENV);
    CHECK(omega_q36_prefix_budget_from_env(&b) == 0 && b == 17179869184ull, "default 16 GiB: %llu", (unsigned long long)b);
    setenv(OMEGA_Q36_PREFIX_POOL_ENV, "0.5", 1);
    CHECK(omega_q36_prefix_budget_from_env(&b) == 0 && b == 536870912ull, "0.5 GiB");
    setenv(OMEGA_Q36_PREFIX_POOL_ENV, "0", 1);
    CHECK(omega_q36_prefix_budget_from_env(&b) == 0 && b == 0, "0: nothing resident");
    const char *badv[] = {"-1", "x", "16G", "nan", "inf"};
    for (size_t i = 0; i < sizeof badv / sizeof badv[0]; i++) {
        setenv(OMEGA_Q36_PREFIX_POOL_ENV, badv[i], 1);
        CHECK(omega_q36_prefix_budget_from_env(&b) == OMEGA_Q36_PREFIX_BAD_ARGS, "%s refused", badv[i]);
    }
    unsetenv(OMEGA_Q36_PREFIX_POOL_ENV);

    printf("gpu_q36_prefix_test: %d checks, %d failed: %s\n", g_checks, g_failed, g_failed ? "FAIL" : "PASS");
    free(h1); free(h2); free(hc); free(g_mem);
    return g_failed ? 1 : 0;
}
