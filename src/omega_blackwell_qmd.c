#include "omega_blackwell_qmd.h"
#include <string.h>

/*
 * Driver constant buffer 0 base template (224 words, 896 bytes).
 * Empirically captured from physical Blackwell GB10 compute channel bring-up.
 */
static const uint32_t CBANK0_DRIVER_TEMPLATE[OMEGA_BW_CBANK_DRIVER_WORDS] = {
    0x26420000, 0x00000003, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x94fe5d00, 0xe811c39b, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x000fffff, 0x00000000,
    0x20000000, 0x00000003, 0x22000000, 0x00000003, 0x24280000, 0x00000003, 0x24010000, 0x00000003,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000001, 0x00000000, 0x0400c000, 0x00000002, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000400, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x04ba32c8, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000001, 0x00000000, 0x00000000, 0x00000000, 0x00000120, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x25b66900, 0x00000003,
    0x00000001, 0x00000001, 0x00000001, 0x00000000, 0x3f800000, 0x3f800000, 0x3f800000, 0x00000400,
    0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x04ba32c8, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000001, 0x00000000, 0x00000000, 0x00000000, 0x00000120, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x25b66900, 0x00000003,
    0x00000001, 0x00000001, 0x00000001, 0x00000000, 0x3f800000, 0x3f800000, 0x3f800000, 0x00000400,
    0x00000001, 0x00000001, 0x00000001, 0x00000001, 0x00000030, 0x00000000, 0x00000000, 0x01000000,
    0x00000000, 0x0000e9c0, 0x00000000, 0x00000000, 0x00000000, 0x0000e9c0, 0x01000000, 0x0000e9bf,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x24280380, 0x00000003, 0x2428039c, 0x00000003, 0x00000000, 0x00000000,
    0x00000040, 0x00000001, 0x00000001, 0x00000000, 0x00000001, 0x00000001, 0x00000001, 0x00fffdc0,
};

int omega_blackwell_build_qmd0(uint32_t *qmd0_words, uint64_t qmd0_va, uint64_t qmd1_va) {
    if (!qmd0_words) return -1;
    (void)qmd0_va;
    memset(qmd0_words, 0, OMEGA_BW_QMD_BYTES);

    /* Word 4: DEPENDENCE_COUNTER=2, QMD_TYPE=1 (GRID_NULL) */
    qmd0_words[4] = 0x00800002;

    /* Word 10: DEPENDENT_QMD0_ENABLE=1, ACTION=1 (SCHEDULE), SELF_COPY=1 */
    qmd0_words[10] = 0x00230000;

    /* Word 12: DEPENDENT_QMD0_POINTER = qmd1_va >> 8 */
    qmd0_words[12] = (uint32_t)(qmd1_va >> 8);

    /* Word 14: QMD_MAJOR_VERSION=5, SAMPLER_INDEX=1 */
    qmd0_words[14] = 0x00500200;

    /* Word 34: CTA_THREAD_DIMENSION0=1, CTA_THREAD_DIMENSION1=1 */
    qmd0_words[34] = 0x00010001;

    /* Word 35: CTA_THREAD_DIMENSION2=1, REGISTER_COUNT=8 */
    qmd0_words[35] = 0x00000801;

    return 0;
}

int omega_blackwell_build_qmd1(uint32_t *qmd1_words, const OmegaBlackwellQmdConfig *cfg) {
    if (!qmd1_words || !cfg) return -1;
    memset(qmd1_words, 0, OMEGA_BW_QMD_BYTES);

    uint32_t tx = cfg->threads_x ? cfg->threads_x : (cfg->threads_per_block ? cfg->threads_per_block : 64);
    uint32_t ty = cfg->threads_y ? cfg->threads_y : 1;
    uint32_t tz = 1;
    uint32_t gx = cfg->grid_x ? cfg->grid_x : (cfg->grid_width ? cfg->grid_width : (cfg->num_elements + tx - 1) / tx);
    if (gx == 0) gx = 1;
    uint32_t gy = cfg->grid_y ? cfg->grid_y : 1;
    uint32_t gz = 1;

    /* Word 4: QMD_GROUP_ID=0x3f, QMD_TYPE=2 (GRID_CTA) */
    qmd1_words[4] = 0x013f0000;

    /* Word 9: RELEASE_ENABLE=1, RELEASE_STRUCTURE_SIZE=1 (ONE_WORD) */
    qmd1_words[9] = 0x00000003;

    /* Word 10: DEPENDENT_QMD0_ENABLE=1, ACTION=4 (DECREMENT_DEPENDENCE), PREFETCH=1 */
    qmd1_words[10] = 0x00190000;

    /* Word 12: DEPENDENT_QMD0_POINTER = qmd0_va >> 8 */
    qmd1_words[12] = (uint32_t)(cfg->qmd0_va >> 8);

    /* Word 14: QMD_VERSION=5, SASS_VERSION=0xa4, SAMPLER=1, cache invalidation flags */
    qmd1_words[14] = 0x2f5003a4;

    /* Words 15-17: RELEASE_SEMAPHORE0 (writes 6 to sem_va upon completion) */
    qmd1_words[15] = (uint32_t)cfg->sem_va;
    qmd1_words[16] = (uint32_t)(cfg->sem_va >> 32);
    qmd1_words[17] = 0x00000006;

    /* Word 19: CWD_MEMBAR_TYPE=1, FREE_CTA_SLOTS=24, PRE_EXIT=1 */
    qmd1_words[19] = 0x81810000;

    /* Words 20, 22: ARRIVE_AT_LATCH (ID 56, VALID=1) */
    qmd1_words[20] = 0x00000038;
    qmd1_words[22] = 0x04000000;

    /* Words 32-33: PROGRAM_ADDRESS (code_va >> 4) and prefetch size 0xa (bits 29:21) */
    qmd1_words[32] = (uint32_t)(cfg->code_va >> 4);
    qmd1_words[33] = (uint32_t)(((cfg->code_va >> 36) & 0x1fffff) | (0xa << 21));

    /* Words 34-35: CTA thread dimensions, registers, 1 barrier */
    qmd1_words[34] = (tx & 0xffff) | ((ty & 0xffff) << 16);
    uint32_t gpr = cfg->gpr_count ? cfg->gpr_count : 16;
    if (gpr < 16) gpr = 16;
    qmd1_words[35] = (tz & 0xff) | ((gpr & 0x1ff) << 8) | (1u << 17);

    /* Word 36: SHARED_MEMORY_SIZE_SHIFTED7 in bits 10:0 (default 8 = 1024 bytes), then the
     * MIN/MAX/TARGET_SM_CONFIG_SHARED_MEM_SIZE fields (9, 26, 9) kept as they were. */
    if (cfg->shared_bytes > OMEGA_BW_QMD_MAX_SHARED_BYTES) return -1;
    uint32_t smem7 = cfg->shared_bytes ? (cfg->shared_bytes + 127u) / 128u : 8u;
    qmd1_words[36] = (0x04b44808u & ~0x7ffu) | smem7;

    /* Word 37: SHADER_LOCAL_MEMORY_HIGH_SIZE_SHIFTED4 = 0 (SKEDCHECK05 compliance) */
    qmd1_words[37] = 0x00000000;

    /* Words 39-41: Grid dimensions */
    qmd1_words[39] = gx;
    qmd1_words[40] = gy;
    qmd1_words[41] = gz;

    /* Words 42-43: Constant Buffer 0 (cbank_va, size 1024 bytes -> 0x40 shifted 19) */
    qmd1_words[42] = (uint32_t)(cfg->cbank_va >> 6);
    qmd1_words[43] = (uint32_t)(((cfg->cbank_va >> 38) & 0x7ffff) | (0x40 << 19));

    /* Words 44-45: Constant Buffer 1 (scratch_va, size 0x90 shifted 19) */
    qmd1_words[44] = (uint32_t)(cfg->scratch_va >> 6);
    qmd1_words[45] = (uint32_t)(((cfg->scratch_va >> 38) & 0x7ffff) | (0x90 << 19));

    /* Words 52-53: Constant Buffer 5 (cbank_va + 0x300, size 0x10 shifted 19) */
    qmd1_words[52] = (uint32_t)((cfg->cbank_va + 0x300) >> 6);
    qmd1_words[53] = (uint32_t)((((cfg->cbank_va + 0x300) >> 38) & 0x7ffff) | (0x10 << 19));

    /* Words 56-57: Constant Buffer 7 (scratch_va, size 0x1000 shifted 19) */
    qmd1_words[56] = (uint32_t)(cfg->scratch_va >> 6);
    qmd1_words[57] = (uint32_t)(((cfg->scratch_va >> 38) & 0x7ffff) | (0x1000 << 19));

    /* Word 58: Constant buffer valid mask (CB0, CB1, CB5, CB7) */
    qmd1_words[58] = 0x10300011;

    /* Word 59: PROGRAM_PREFETCH_ADDR_LOWER_SHIFTED8 = code_va >> 8 */
    qmd1_words[59] = (uint32_t)(cfg->code_va >> 8);

    return 0;
}

int omega_blackwell_build_cbank_driver(uint32_t *cbank_words, uint64_t cbank_va) {
    if (!cbank_words) return -1;
    memcpy(cbank_words, CBANK0_DRIVER_TEMPLATE, sizeof(CBANK0_DRIVER_TEMPLATE));

    /* Patch driver template addresses with actual cbank_va */
    cbank_words[68] = (uint32_t)(cbank_va >> 32);
    cbank_words[69] = (uint32_t)cbank_va;

    cbank_words[210] = (uint32_t)((cbank_va + 0x380) >> 32);
    cbank_words[211] = (uint32_t)(cbank_va + 0x380);
    cbank_words[212] = (uint32_t)((cbank_va + 0x39c) >> 32);
    cbank_words[213] = (uint32_t)(cbank_va + 0x39c);

    return 0;
}

int omega_blackwell_build_cbank_args(uint32_t *args_words, uint64_t a_va, uint64_t b_va,
                                    uint64_t c_va, uint32_t n) {
    if (!args_words) return -1;

    args_words[0] = (uint32_t)a_va;
    args_words[1] = (uint32_t)(a_va >> 32);
    args_words[2] = (uint32_t)b_va;
    args_words[3] = (uint32_t)(b_va >> 32);
    args_words[4] = (uint32_t)c_va;
    args_words[5] = (uint32_t)(c_va >> 32);
    args_words[6] = n;

    return 0;
}

int omega_blackwell_build_cbank_driver_2d(uint32_t *cbank_words, uint64_t cbank_va,
                                         uint32_t threads_x, uint32_t threads_y,
                                         uint32_t grid_x, uint32_t grid_y) {
    if (!cbank_words) return -1;
    if (omega_blackwell_build_cbank_driver(cbank_words, cbank_va) != 0) return -1;
    if (threads_x > 0) cbank_words[216] = threads_x;
    if (threads_y > 0) cbank_words[217] = threads_y;
    if (grid_x > 0) cbank_words[220] = grid_x;
    if (grid_y > 0) cbank_words[221] = grid_y;
    return 0;
}

int omega_blackwell_build_cbank_args_matmul(uint32_t *args_words, uint64_t a_va, uint64_t b_va,
                                           uint64_t c_va, uint32_t m, uint32_t k, uint32_t n) {
    if (!args_words) return -1;
    args_words[0] = (uint32_t)a_va;
    args_words[1] = (uint32_t)(a_va >> 32);
    args_words[2] = (uint32_t)b_va;
    args_words[3] = (uint32_t)(b_va >> 32);
    args_words[4] = (uint32_t)c_va;
    args_words[5] = (uint32_t)(c_va >> 32);
    args_words[6] = m;
    args_words[7] = k;
    args_words[8] = n;
    args_words[9] = 0;
    return 0;
}

int omega_blackwell_verify_qmd_invariants(const uint32_t *qmd1_words) {
    if (!qmd1_words) return -1;

    /* 1. QMD Major Version must be 5 (bits 471:468 in Word 14) */
    uint32_t version = (qmd1_words[14] >> 20) & 0xf;
    if (version != 5) return -1;

    /* 2. QMD Type must be GRID_CTA = 2 (bits 153:151 in Word 4) */
    uint32_t type = (qmd1_words[4] >> 23) & 0x7;
    if (type != 2) return -1;

    /* 3. SKEDCHECK05 compliance: local memory high size shifted 4 must be 0 */
    if (qmd1_words[37] != 0) return -1;

    /* 4. Release semaphore payload must be non-zero */
    if (qmd1_words[17] == 0) return -1;

    /* 5. Program prefetch size must be 0xa (bits 29:21 in Word 33) */
    uint32_t prefetch = (qmd1_words[33] >> 21) & 0x1ff;
    if (prefetch != 0xa) return -1;

    /* 6. Constant buffer valid mask must have CB0 enabled */
    if ((qmd1_words[58] & 0x1) == 0) return -1;

    return 0;
}
