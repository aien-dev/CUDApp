/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */
/* test_omega_bw_qmd_shared: the shared-memory field of the Blackwell QMD (word 36), host only.
 *
 * Word 36 carries NVCEC0_QMDV05_00_SHARED_MEMORY_SIZE_SHIFTED7 (clcec0qmd.h, MW(1162:1152)): 11
 * bits in units of 128 bytes. Next to it sit MIN_/MAX_/TARGET_SM_CONFIG_SHARED_MEM_SIZE (6 bits
 * each), which this builder keeps at its fixed values 9 / 26 / 9 (32 KiB / 100 KiB / 32 KiB under
 * Mesa NVK's size_kb / 4 + 1 rule). Checks:
 *   - 0 keeps the long-standing 1024 (field value 8); the cap itself is accepted;
 *   - the encoded value is size / 128 and never changes the SM-config fields;
 *   - one byte over the cap is refused, so the cap is a real gate (negative control) and not
 *     just a comment;
 *   - the production tensor-core FP8 grouped GEMM shape (rt 8 -> 32768 bytes) is inside the cap,
 *     which is why the cap moved from 16384 (gpu_pfgemm_test --chip failed at "qmd build
 *     (shared_bytes)" on every shape, 2026-10-25).
 * No device is opened. */
#include "omega_blackwell_qmd.h"
#include <stdio.h>
#include <string.h>

static int g_checks, g_failed;
#define CHECK(cond, ...) do { g_checks++; if (!(cond)) { g_failed++; printf("FAIL: "); printf(__VA_ARGS__); printf("\n"); } } while (0)

#define FIELD_SHARED7(w)  ((w) & 0x7ffu)          /* MW(1162:1152) */
#define FIELD_MIN(w)      (((w) >> 11) & 0x3fu)   /* MW(1168:1163) */
#define FIELD_MAX(w)      (((w) >> 17) & 0x3fu)   /* MW(1174:1169) */
#define FIELD_TARGET(w)   (((w) >> 23) & 0x3fu)   /* MW(1180:1175) */

static OmegaBlackwellQmdConfig base(void) {
    OmegaBlackwellQmdConfig c;
    memset(&c, 0, sizeof c);
    c.code_va = 0x100000; c.cbank_va = 0x200000; c.scratch_va = 0x300000; c.sem_va = 0x400000;
    c.qmd0_va = 0x500000; c.qmd1_va = 0x600000;
    c.num_elements = 128; c.threads_x = 128; c.grid_x = 1; c.gpr_count = 64;
    return c;
}

/* Build one QMD for `shared` bytes; 0 when the builder accepted it, -1 when it refused. */
static int build(uint32_t shared, uint32_t q[OMEGA_BW_QMD_WORDS]) {
    OmegaBlackwellQmdConfig c = base();
    c.shared_bytes = shared;
    memset(q, 0, OMEGA_BW_QMD_BYTES);
    return omega_blackwell_build_qmd1(q, &c) == 0 ? 0 : -1;
}

int main(void) {
    uint32_t q[OMEGA_BW_QMD_WORDS], q_def[OMEGA_BW_QMD_WORDS];

    if (build(0, q_def) != 0) { printf("FAIL: the default QMD was refused\n"); return 1; }
    const uint32_t min = FIELD_MIN(q_def[36]), mx = FIELD_MAX(q_def[36]), target = FIELD_TARGET(q_def[36]);
    printf("word 36 default: shared7 %u min %u max %u target %u\n", FIELD_SHARED7(q_def[36]), min, mx, target);
    CHECK(FIELD_SHARED7(q_def[36]) == 8, "shared_bytes 0 keeps 1024 bytes (field 8), got %u", FIELD_SHARED7(q_def[36]));

    /* Every size the builder accepts encodes size / 128 exactly and leaves the SM config alone. */
    static const uint32_t sizes[] = { 1024, 4096, 16384, 32768 };
    for (size_t i = 0; i < sizeof sizes / sizeof sizes[0]; i++) {
        int rc = build(sizes[i], q);
        CHECK(rc == 0, "%u bytes refused", sizes[i]);
        if (rc) continue;
        CHECK(FIELD_SHARED7(q[36]) == sizes[i] / 128u, "%u bytes encode as %u (want %u)", sizes[i], FIELD_SHARED7(q[36]), sizes[i] / 128u);
        CHECK(FIELD_MIN(q[36]) == min && FIELD_MAX(q[36]) == mx && FIELD_TARGET(q[36]) == target,
              "%u bytes left the SM config fields alone (min %u max %u target %u)", sizes[i], FIELD_MIN(q[36]), FIELD_MAX(q[36]), FIELD_TARGET(q[36]));
    }

    /* The rest of the descriptor must not move with the shared size: bytes 0..35 and 37.. are the
     * same words as the 0-byte build (word 36 is the only shared-memory carrier). */
    build(32768, q);
    for (int w = 0; w < OMEGA_BW_QMD_WORDS; w++) if (w != 36) CHECK(q[w] == q_def[w], "word %d changed with the shared size", w);

    /* Negative control: one byte over the cap is refused, and the same request rounded down to the
     * cap is accepted, so the refusal is the cap and not something else. */
    CHECK(build(OMEGA_BW_QMD_MAX_SHARED_BYTES + 1u, q) == -1, "%u bytes (cap + 1) was accepted", OMEGA_BW_QMD_MAX_SHARED_BYTES + 1u);
    CHECK(build(OMEGA_BW_QMD_MAX_SHARED_BYTES, q) == 0, "the cap itself (%u bytes) was refused", OMEGA_BW_QMD_MAX_SHARED_BYTES);

    /* The production FP8 grouped GEMM: omega_tc2_shared_bytes = 4096 * rt, rt 8 at the shipped shape. */
    CHECK(4096u * 8u <= OMEGA_BW_QMD_MAX_SHARED_BYTES, "the tc2 production shape (rt 8, 32768 bytes) fits the cap %u", OMEGA_BW_QMD_MAX_SHARED_BYTES);
    CHECK(FIELD_SHARED7(q_def[36]) * 128u <= OMEGA_BW_QMD_MAX_SHARED_BYTES, "the cap is inside what the 11-bit field can name");

    printf("qmd shared: %d checks, %d failed: %s\n", g_checks, g_failed, g_failed ? "FAIL" : "PASS");
    return g_failed ? 1 : 0;
}
