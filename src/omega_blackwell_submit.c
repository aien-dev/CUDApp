#include <math.h>
#include "omega_blackwell_submit.h"
#include "omega_blackwell_qmd.h"
#include "omega_blackwell_encoder.h"
#include "m16_native.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static const uint32_t SETUP_WORDS[18] = {
    0x20012061, 0x0000cec0, 0x20012092, 0x00000001, 0x200120a8, 0x0000000f, 0x2001255d, 0x00000003,
    0x2001255e, 0x20000000, 0x2001255f, 0x000fffff, 0x20012557, 0x00000003, 0x20012558, 0x22000000,
    0x20012559, 0x00000000,
};

static uint64_t current_time_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000000000ULL + (uint64_t)t.tv_nsec;
}

int omega_blackwell_execute_vector(const OmegaVectorSpec *spec,
                                  const OmegaBlackwellRealization *real,
                                  const uint32_t *h_a,
                                  const uint32_t *h_b,
                                  uint32_t *h_c_out,
                                  OmegaBlackwellVectorExecution *exec_info) {
    if (!spec || !real || !h_a || !h_b || !h_c_out) return -1;
    uint32_t n = spec->element_count;
    if (n == 0) return -1;

    if (exec_info) {
        memset(exec_info, 0, sizeof(*exec_info));
        strncpy(exec_info->target_chip, "NVIDIA DGX Spark (Grace Blackwell GB10)", sizeof(exec_info->target_chip) - 1);
        exec_info->sm_architecture = real->sm_architecture;
        exec_info->element_count = n;
        exec_info->zero_libcuda_linkage = (omega_blackwell_verify_zero_libcuda_linkage(NULL) == 0);
        exec_info->zero_cuda_symbols = (omega_blackwell_verify_zero_cuda_symbols(NULL) == 0);
        exec_info->zero_libcuda_runtime = (omega_blackwell_verify_zero_libcuda_runtime() == 0);
    }

    M16NativeContext ctx;
    if (m16_native_open(&ctx) != 0) { return -1; }
    if (m16_native_create_channel(&ctx) != 0) { m16_native_close(&ctx); return -1; }

    /* Allocate larger pushbuffer ring for unified method stream */
    NvrmMem large_pb;
    if (nvrm_alloc(&ctx.rm, 0x10000, &large_pb) != 0) { m16_native_close(&ctx); return -1; }
    ctx.pb_mem = large_pb;

    /* Allocate device-visible UVM buffers */
    NvrmMem code_mem, cbank_mem, a_mem, b_mem, c_mem, marker_mem, qmd_mem;
    size_t vector_bytes = (n * sizeof(uint32_t) + 0xFFFULL) & ~0xFFFULL;
    if (vector_bytes < 0x1000) vector_bytes = 0x1000;

    if (nvrm_alloc(&ctx.rm, 0x1000, &code_mem) != 0) { m16_native_close(&ctx); return -1; }
    if (nvrm_alloc(&ctx.rm, 0x1000, &cbank_mem) != 0) { m16_native_close(&ctx); return -1; }
    if (nvrm_alloc(&ctx.rm, vector_bytes, &a_mem) != 0) { m16_native_close(&ctx); return -1; }
    if (nvrm_alloc(&ctx.rm, vector_bytes, &b_mem) != 0) { m16_native_close(&ctx); return -1; }
    if (nvrm_alloc(&ctx.rm, vector_bytes, &c_mem) != 0) { m16_native_close(&ctx); return -1; }
    if (nvrm_alloc(&ctx.rm, 0x1000, &marker_mem) != 0) { m16_native_close(&ctx); return -1; }
    if (nvrm_alloc(&ctx.rm, 0x10000, &qmd_mem) != 0) { m16_native_close(&ctx); return -1; }
    

    /* Emit machine code into code_mem */
    size_t encoded_len = 0;
    if (omega_blackwell_encode_vecadd(code_mem.cpu, code_mem.size, &encoded_len) != 0) {
        m16_native_close(&ctx);
        return -1;
    }

    /* Populate input buffers A, B and initialize C with poison */
    memcpy(a_mem.cpu, h_a, n * sizeof(uint32_t));
    memcpy(b_mem.cpu, h_b, n * sizeof(uint32_t));
    uint32_t *c_dev = (uint32_t *)c_mem.cpu;
    for (size_t i = 0; i < n; i++) {
        c_dev[i] = OMEGA_VECTOR_POISON_VALUE;
    }

    /* Build driver cbank data and kernel arguments */
    uint32_t cbank_data[OMEGA_BW_CBANK_DRIVER_WORDS];
    omega_blackwell_build_cbank_driver(cbank_data, cbank_mem.va);

    uint32_t cbank_args[OMEGA_BW_CBANK_ARGS_WORDS];
    omega_blackwell_build_cbank_args(cbank_args, a_mem.va, b_mem.va, c_mem.va, n);

    /* Direct coherent CPU copy into cbank_mem */
    memcpy(cbank_mem.cpu, cbank_data, sizeof(cbank_data));
    memcpy((uint8_t *)cbank_mem.cpu + 0x380, cbank_args, sizeof(cbank_args));

    /* Configure and build QMD 0 & QMD 1 */
    uint64_t qmd0_va = qmd_mem.va;
    uint64_t qmd1_va = qmd_mem.va + 0x1000;
    uint64_t sem_va  = qmd_mem.va + 0x2000;
    uint64_t scratch_va = qmd_mem.va + 0x4000;

    OmegaBlackwellQmdConfig qmd_cfg = {
        .code_va = code_mem.va,
        .cbank_va = cbank_mem.va,
        .scratch_va = scratch_va,
        .sem_va = sem_va,
        .qmd0_va = qmd0_va,
        .qmd1_va = qmd1_va,
        .num_elements = n,
        .threads_per_block = 64,
        .grid_width = (n + 63) / 64
    };
    if (qmd_cfg.grid_width == 0) qmd_cfg.grid_width = 1;

    uint32_t qmd0_words[OMEGA_BW_QMD_WORDS];
    uint32_t qmd1_words[OMEGA_BW_QMD_WORDS];
    omega_blackwell_build_qmd0(qmd0_words, qmd0_va, qmd1_va);
    omega_blackwell_build_qmd1(qmd1_words, &qmd_cfg);

    /* Verify QMD structural invariants before submission */
    if (omega_blackwell_verify_qmd_invariants(qmd1_words) != 0) { m16_native_close(&ctx); return -1; }

    memcpy(qmd_mem.cpu, qmd0_words, sizeof(qmd0_words));
    memcpy((uint8_t *)qmd_mem.cpu + 0x1000, qmd1_words, sizeof(qmd1_words));

    /* Initialize synchronization semaphores in coherent memory */
    volatile uint32_t *hsem = (volatile uint32_t *)((uint8_t *)qmd_mem.cpu + 0x2000);
    volatile uint32_t *hmarker = (volatile uint32_t *)marker_mem.cpu;
    *hsem = 0;
    *hmarker = 0;
    __asm__ volatile("dsb sy" ::: "memory");

    /* Assemble unified pushbuffer (481 words) */
    uint32_t pb[1024];
    size_t pb_len = 0;

    /* 1. Setup words (18 words) */
    memcpy(&pb[pb_len], SETUP_WORDS, sizeof(SETUP_WORDS));
    pb_len += sizeof(SETUP_WORDS) / 4;

    /* 2. Cbank driver data upload (224 words via DMA) */
    pb[pb_len++] = nvrm_mthd(1, 0x0188, 2);
    pb[pb_len++] = (uint32_t)(cbank_mem.va >> 32);
    pb[pb_len++] = (uint32_t)cbank_mem.va;
    pb[pb_len++] = nvrm_mthd(1, 0x0180, 2);
    pb[pb_len++] = 0x00000380;
    pb[pb_len++] = 0x00000001;
    pb[pb_len++] = nvrm_mthd(1, 0x01b0, 1);
    pb[pb_len++] = 0x00000041;
    pb[pb_len++] = (224 << 16) | (1 << 13) | (0x01b4 >> 2) | (6u << 28);
    memcpy(&pb[pb_len], cbank_data, 224 * 4);
    pb_len += 224;

    /* 3. Kernel args upload (7 words via DMA) */
    pb[pb_len++] = nvrm_mthd(1, 0x0188, 2);
    pb[pb_len++] = (uint32_t)((cbank_mem.va + 0x380) >> 32);
    pb[pb_len++] = (uint32_t)(cbank_mem.va + 0x380);
    pb[pb_len++] = nvrm_mthd(1, 0x0180, 2);
    pb[pb_len++] = 0x0000001c;
    pb[pb_len++] = 0x00000001;
    pb[pb_len++] = nvrm_mthd(1, 0x01b0, 1);
    pb[pb_len++] = 0x00000041;
    pb[pb_len++] = (7 << 16) | (1 << 13) | (0x01b4 >> 2) | (6u << 28);
    memcpy(&pb[pb_len], cbank_args, 7 * 4);
    pb_len += 7;

    /* 4. Inline QMD 0 (98 words: 2 addr + 96 QMD words) */
    pb[pb_len++] = (98 << 16) | (1 << 13) | (0x0318 >> 2) | (2u << 28);
    pb[pb_len++] = (1u << 30) | (uint32_t)((qmd0_va >> 40) & 0x1ff);
    pb[pb_len++] = (uint32_t)(qmd0_va >> 8);
    memcpy(&pb[pb_len], qmd0_words, 96 * 4);
    pb_len += 96;

    /* 5. Intermediate semaphore upload (1 word = 5 via DMA) */
    pb[pb_len++] = nvrm_mthd(1, 0x0188, 2);
    pb[pb_len++] = (uint32_t)(sem_va >> 32);
    pb[pb_len++] = (uint32_t)sem_va;
    pb[pb_len++] = nvrm_mthd(1, 0x0180, 2);
    pb[pb_len++] = 0x00000004;
    pb[pb_len++] = 0x00000001;
    pb[pb_len++] = nvrm_mthd(1, 0x01b0, 1);
    pb[pb_len++] = 0x00000041;
    pb[pb_len++] = (1 << 16) | (1 << 13) | (0x01b4 >> 2) | (6u << 28);
    pb[pb_len++] = OMEGA_BW_SEMAPHORE_INTERMEDIATE_INIT;

    /* 6. Inline QMD 1 (98 words: 2 addr + 96 QMD words) */
    pb[pb_len++] = (98 << 16) | (1 << 13) | (0x0318 >> 2) | (2u << 28);
    pb[pb_len++] = (1u << 30) | (uint32_t)((qmd1_va >> 40) & 0x1ff);
    pb[pb_len++] = (uint32_t)(qmd1_va >> 8);
    memcpy(&pb[pb_len], qmd1_words, 96 * 4);
    pb_len += 96;

    /* 7. Subchannel 0 completion release (WFI) */
    pb[pb_len++] = nvrm_mthd(0, 0x005c, 5);
    pb[pb_len++] = (uint32_t)marker_mem.va;
    pb[pb_len++] = (uint32_t)(marker_mem.va >> 32);
    pb[pb_len++] = OMEGA_BW_MARKER_COMPLETION_PAYLOAD;
    pb[pb_len++] = 0;
    pb[pb_len++] = 0x1 | (1u << 20); // RELEASE | WFI

    /* Record timestamp and submit methods to hardware GPFIFO ring */
    uint64_t t_start = current_time_ns();
    if (m16_native_submit_methods(&ctx, pb, pb_len) != 0) { m16_native_close(&ctx); return -1; }

    /* Wait for hardware completion marker */
    if (m16_native_wait_marker(hmarker, OMEGA_BW_MARKER_COMPLETION_PAYLOAD, 5000) != 0) { m16_native_close(&ctx); return -1; }
    uint64_t t_end = current_time_ns();

    /* Validate intermediate semaphore updated to 6 */
    if (*hsem != OMEGA_BW_SEMAPHORE_INTERMEDIATE_DONE) { m16_native_close(&ctx); return -1; }

    /* Copy device output back to host */
    memcpy(h_c_out, c_mem.cpu, n * sizeof(uint32_t));

    /* Verify against OMEGA semantic oracle */
    size_t mismatch_idx = 0;
    int parity_res = omega_vector_verify_oracle(h_a, h_b, h_c_out, n, &mismatch_idx);

    if (exec_info) {
        exec_info->launch_timestamp_ns = t_start;
        exec_info->completion_timestamp_ns = t_end;
        exec_info->elapsed_ns = (t_end > t_start) ? (t_end - t_start) : 0;
        exec_info->completion_marker = *hmarker;
        exec_info->intermediate_semaphore = *hsem;
        exec_info->parity_verified = (parity_res == 0);
    }

    m16_native_close(&ctx);
    return parity_res;
}

int omega_blackwell_execute_matmul(const OmegaMatMulSpec *spec,
                                  const OmegaBlackwellKernel *kernel,
                                  const uint32_t *h_a,
                                  const uint32_t *h_b,
                                  uint32_t *h_c_out,
                                  OmegaBlackwellMatMulExecution *exec_info) {
    if (!spec || !kernel || !h_a || !h_b || !h_c_out) return -1;
    if (!kernel->code || kernel->code_size == 0) return -1;
    if (spec->m == 0 || spec->k == 0 || spec->n == 0) return -1;

    if (exec_info) {
        memset(exec_info, 0, sizeof(*exec_info));
        strncpy(exec_info->target_chip, "NVIDIA DGX Spark (Grace Blackwell GB10)", sizeof(exec_info->target_chip) - 1);
        exec_info->sm_architecture = 121;
        exec_info->m = spec->m;
        exec_info->k = spec->k;
        exec_info->n = spec->n;
        exec_info->zero_libcuda_linkage = (omega_blackwell_verify_zero_libcuda_linkage(NULL) == 0);
        exec_info->zero_cuda_symbols = (omega_blackwell_verify_zero_cuda_symbols(NULL) == 0);
        exec_info->zero_libcuda_runtime = (omega_blackwell_verify_zero_libcuda_runtime() == 0);
    }

    M16NativeContext ctx;
    if (m16_native_open(&ctx) != 0) { return -1; }
    if (m16_native_create_channel(&ctx) != 0) { m16_native_close(&ctx); return -1; }

    NvrmMem large_pb;
    if (nvrm_alloc(&ctx.rm, 0x10000, &large_pb) != 0) { m16_native_close(&ctx); return -1; }
    ctx.pb_mem = large_pb;

    /* Sizing buffer allocations */
    size_t a_bytes = ((size_t)spec->m * spec->k * sizeof(uint32_t) + 0xFFFULL) & ~0xFFFULL;
    if (a_bytes < 0x1000) a_bytes = 0x1000;
    size_t b_bytes = ((size_t)spec->k * spec->n * sizeof(uint32_t) + 0xFFFULL) & ~0xFFFULL;
    if (b_bytes < 0x1000) b_bytes = 0x1000;
    size_t c_bytes = ((size_t)spec->m * spec->n * sizeof(uint32_t) + 0xFFFULL) & ~0xFFFULL;
    if (c_bytes < 0x1000) c_bytes = 0x1000;
    size_t code_bytes = (kernel->code_size + 0xFFFULL) & ~0xFFFULL;
    if (code_bytes < 0x1000) code_bytes = 0x1000;

    NvrmMem code_mem, cbank_mem, a_mem, b_mem, c_mem, marker_mem, qmd_mem;
    if (nvrm_alloc(&ctx.rm, code_bytes, &code_mem) != 0) { m16_native_close(&ctx); return -1; }
    if (nvrm_alloc(&ctx.rm, 0x1000, &cbank_mem) != 0) { m16_native_close(&ctx); return -1; }
    if (nvrm_alloc(&ctx.rm, a_bytes, &a_mem) != 0) { m16_native_close(&ctx); return -1; }
    if (nvrm_alloc(&ctx.rm, b_bytes, &b_mem) != 0) { m16_native_close(&ctx); return -1; }
    if (nvrm_alloc(&ctx.rm, c_bytes, &c_mem) != 0) { m16_native_close(&ctx); return -1; }
    if (nvrm_alloc(&ctx.rm, 0x1000, &marker_mem) != 0) { m16_native_close(&ctx); return -1; }
    if (nvrm_alloc(&ctx.rm, 0x10000, &qmd_mem) != 0) { m16_native_close(&ctx); return -1; }

    /* Copy dynamic machine code into code_mem */
    memcpy(code_mem.cpu, kernel->code, kernel->code_size);

    /* Populate input buffers A, B and initialize C with poison */
    memcpy(a_mem.cpu, h_a, (size_t)spec->m * spec->k * sizeof(uint32_t));
    memcpy(b_mem.cpu, h_b, (size_t)spec->k * spec->n * sizeof(uint32_t));
    uint32_t *c_dev = (uint32_t *)c_mem.cpu;
    for (size_t i = 0; i < (size_t)spec->m * spec->n; i++) {
        c_dev[i] = OMEGA_VECTOR_POISON_VALUE;
    }

    /* 2D Tile Configuration: 16x16 threads per block */
    uint32_t threads_x = 16;
    uint32_t threads_y = 16;
    uint32_t grid_x = (spec->n + 15) / 16;
    uint32_t grid_y = (spec->m + 15) / 16;

    uint32_t cbank_data[OMEGA_BW_CBANK_DRIVER_WORDS];
    omega_blackwell_build_cbank_driver_2d(cbank_data, cbank_mem.va, threads_x, threads_y, grid_x, grid_y);

    uint32_t cbank_args[OMEGA_BW_CBANK_MATMUL_ARGS_WORDS];
    omega_blackwell_build_cbank_args_matmul(cbank_args, a_mem.va, b_mem.va, c_mem.va, spec->m, spec->k, spec->n);

    memcpy(cbank_mem.cpu, cbank_data, sizeof(cbank_data));
    memcpy((uint8_t *)cbank_mem.cpu + 0x380, cbank_args, sizeof(cbank_args));

    uint64_t qmd0_va = qmd_mem.va;
    uint64_t qmd1_va = qmd_mem.va + 0x1000;
    uint64_t sem_va  = qmd_mem.va + 0x2000;
    uint64_t scratch_va = qmd_mem.va + 0x4000;

    OmegaBlackwellQmdConfig qmd_cfg = {
        .code_va = code_mem.va,
        .cbank_va = cbank_mem.va,
        .scratch_va = scratch_va,
        .sem_va = sem_va,
        .qmd0_va = qmd0_va,
        .qmd1_va = qmd1_va,
        .num_elements = spec->m * spec->n,
        .threads_per_block = 256,
        .grid_width = grid_x * grid_y,
        .threads_x = threads_x,
        .threads_y = threads_y,
        .grid_x = grid_x,
        .grid_y = grid_y,
        .gpr_count = (kernel->gpr_count > 16) ? ((kernel->gpr_count + 15) & ~15U) : 32
    };

    uint32_t qmd0_words[OMEGA_BW_QMD_WORDS];
    uint32_t qmd1_words[OMEGA_BW_QMD_WORDS];
    omega_blackwell_build_qmd0(qmd0_words, qmd0_va, qmd1_va);
    omega_blackwell_build_qmd1(qmd1_words, &qmd_cfg);

    if (omega_blackwell_verify_qmd_invariants(qmd1_words) != 0) { m16_native_close(&ctx); return -1; }

    memcpy(qmd_mem.cpu, qmd0_words, sizeof(qmd0_words));
    memcpy((uint8_t *)qmd_mem.cpu + 0x1000, qmd1_words, sizeof(qmd1_words));

    volatile uint32_t *hsem = (volatile uint32_t *)((uint8_t *)qmd_mem.cpu + 0x2000);
    volatile uint32_t *hmarker = (volatile uint32_t *)marker_mem.cpu;
    *hsem = 0;
    *hmarker = 0;
    __asm__ volatile("dsb sy" ::: "memory");

    /* Assemble unified pushbuffer */
    uint32_t pb[1024];
    size_t pb_len = 0;

    /* 1. Setup words (18 words) */
    memcpy(&pb[pb_len], SETUP_WORDS, sizeof(SETUP_WORDS));
    pb_len += sizeof(SETUP_WORDS) / 4;

    /* 2. Cbank driver data upload (224 words via DMA) */
    pb[pb_len++] = nvrm_mthd(1, 0x0188, 2);
    pb[pb_len++] = (uint32_t)(cbank_mem.va >> 32);
    pb[pb_len++] = (uint32_t)cbank_mem.va;
    pb[pb_len++] = nvrm_mthd(1, 0x0180, 2);
    pb[pb_len++] = 0x00000380;
    pb[pb_len++] = 0x00000001;
    pb[pb_len++] = nvrm_mthd(1, 0x01b0, 1);
    pb[pb_len++] = 0x00000041;
    pb[pb_len++] = (224 << 16) | (1 << 13) | (0x01b4 >> 2) | (6u << 28);
    memcpy(&pb[pb_len], cbank_data, 224 * 4);
    pb_len += 224;

    /* 3. Kernel args upload (10 words via DMA = 40 = 0x28 bytes) */
    pb[pb_len++] = nvrm_mthd(1, 0x0188, 2);
    pb[pb_len++] = (uint32_t)((cbank_mem.va + 0x380) >> 32);
    pb[pb_len++] = (uint32_t)(cbank_mem.va + 0x380);
    pb[pb_len++] = nvrm_mthd(1, 0x0180, 2);
    pb[pb_len++] = 0x00000028;
    pb[pb_len++] = 0x00000001;
    pb[pb_len++] = nvrm_mthd(1, 0x01b0, 1);
    pb[pb_len++] = 0x00000041;
    pb[pb_len++] = (10 << 16) | (1 << 13) | (0x01b4 >> 2) | (6u << 28);
    memcpy(&pb[pb_len], cbank_args, 10 * 4);
    pb_len += 10;

    /* 4. Inline QMD 0 (98 words: 2 addr + 96 QMD words) */
    pb[pb_len++] = (98 << 16) | (1 << 13) | (0x0318 >> 2) | (2u << 28);
    pb[pb_len++] = (1u << 30) | (uint32_t)((qmd0_va >> 40) & 0x1ff);
    pb[pb_len++] = (uint32_t)(qmd0_va >> 8);
    memcpy(&pb[pb_len], qmd0_words, 96 * 4);
    pb_len += 96;

    /* 5. Intermediate semaphore upload (1 word = 5 via DMA) */
    pb[pb_len++] = nvrm_mthd(1, 0x0188, 2);
    pb[pb_len++] = (uint32_t)(sem_va >> 32);
    pb[pb_len++] = (uint32_t)sem_va;
    pb[pb_len++] = nvrm_mthd(1, 0x0180, 2);
    pb[pb_len++] = 0x00000004;
    pb[pb_len++] = 0x00000001;
    pb[pb_len++] = nvrm_mthd(1, 0x01b0, 1);
    pb[pb_len++] = 0x00000041;
    pb[pb_len++] = (1 << 16) | (1 << 13) | (0x01b4 >> 2) | (6u << 28);
    pb[pb_len++] = OMEGA_BW_SEMAPHORE_INTERMEDIATE_INIT;

    /* 6. Inline QMD 1 (98 words: 2 addr + 96 QMD words) */
    pb[pb_len++] = (98 << 16) | (1 << 13) | (0x0318 >> 2) | (2u << 28);
    pb[pb_len++] = (1u << 30) | (uint32_t)((qmd1_va >> 40) & 0x1ff);
    pb[pb_len++] = (uint32_t)(qmd1_va >> 8);
    memcpy(&pb[pb_len], qmd1_words, 96 * 4);
    pb_len += 96;

    /* 7. Subchannel 0 completion release (WFI) */
    pb[pb_len++] = nvrm_mthd(0, 0x005c, 5);
    pb[pb_len++] = (uint32_t)marker_mem.va;
    pb[pb_len++] = (uint32_t)(marker_mem.va >> 32);
    pb[pb_len++] = OMEGA_BW_MARKER_COMPLETION_PAYLOAD;
    pb[pb_len++] = 0;
    pb[pb_len++] = 0x1 | (1u << 20); // RELEASE | WFI

    /* Submit methods to hardware GPFIFO ring */
    uint64_t t_start = current_time_ns();
    if (m16_native_submit_methods(&ctx, pb, pb_len) != 0) { m16_native_close(&ctx); return -1; }

    /* Wait for hardware completion marker */
    if (m16_native_wait_marker(hmarker, OMEGA_BW_MARKER_COMPLETION_PAYLOAD, 5000) != 0) { m16_native_close(&ctx); return -1; }
    uint64_t t_end = current_time_ns();

    /* Validate intermediate semaphore updated to 6 */
    if (*hsem != OMEGA_BW_SEMAPHORE_INTERMEDIATE_DONE) { m16_native_close(&ctx); return -1; }

    /* Copy device output back to host */
    memcpy(h_c_out, c_mem.cpu, (size_t)spec->m * spec->n * sizeof(uint32_t));

    /* Verify against OMEGA MatMul mathematical reference oracle */
    uint32_t *h_expected = (uint32_t *)malloc((size_t)spec->m * spec->n * sizeof(uint32_t));
    if (!h_expected) { m16_native_close(&ctx); return -1; }

    omega_matmul_cpu_oracle_i32(h_a, h_b, h_expected, spec->m, spec->k, spec->n);

    size_t mismatches = 0;
    for (size_t i = 0; i < (size_t)spec->m * spec->n; i++) {
        if (h_c_out[i] != h_expected[i]) {
            mismatches++;
        }
    }
    free(h_expected);

    if (exec_info) {
        exec_info->launch_timestamp_ns = t_start;
        exec_info->completion_timestamp_ns = t_end;
        exec_info->elapsed_ns = (t_end > t_start) ? (t_end - t_start) : 0;
        exec_info->completion_marker = *hmarker;
        exec_info->intermediate_semaphore = *hsem;
        exec_info->mismatch_count = mismatches;
        exec_info->parity_verified = (mismatches == 0);
    }

    m16_native_close(&ctx);
    return (mismatches == 0) ? 0 : -1;
}


int omega_blackwell_execute_matmul_tensor(const OmegaMatMulSpec *spec,
                                         const OmegaBlackwellKernel *kernel,
                                         const void *h_a,
                                         const void *h_b,
                                         float *h_c_out,
                                         OmegaBlackwellMatMulExecution *exec_info,
                                         float *out_max_abs_err,
                                         float *out_max_rel_err) {
    if (!spec || !kernel || !h_a || !h_b || !h_c_out) return -1;
    if (!kernel->code || kernel->code_size == 0) return -1;
    if (spec->m == 0 || spec->k == 0 || spec->n == 0) return -1;
    if (spec->precision != OMEGA_MATMUL_PRECISION_FP16 && spec->precision != OMEGA_MATMUL_PRECISION_BF16) return -1;

    if (exec_info) {
        memset(exec_info, 0, sizeof(*exec_info));
        strncpy(exec_info->target_chip, "NVIDIA DGX Spark (Grace Blackwell GB10)", sizeof(exec_info->target_chip) - 1);
        exec_info->sm_architecture = 121;
        exec_info->m = spec->m;
        exec_info->k = spec->k;
        exec_info->n = spec->n;
        exec_info->zero_libcuda_linkage = (omega_blackwell_verify_zero_libcuda_linkage(NULL) == 0);
        exec_info->zero_cuda_symbols = (omega_blackwell_verify_zero_cuda_symbols(NULL) == 0);
        exec_info->zero_libcuda_runtime = (omega_blackwell_verify_zero_libcuda_runtime() == 0);
    }

    M16NativeContext ctx;
    if (m16_native_open(&ctx) != 0) { return -1; }
    if (m16_native_create_channel(&ctx) != 0) { m16_native_close(&ctx); return -1; }

    NvrmMem large_pb;
    if (nvrm_alloc(&ctx.rm, 0x10000, &large_pb) != 0) { m16_native_close(&ctx); return -1; }
    ctx.pb_mem = large_pb;

    size_t a_elems = (size_t)spec->m * spec->k;
    size_t b_elems = (size_t)spec->k * spec->n;
    size_t c_elems = (size_t)spec->m * spec->n;

    size_t a_bytes = ((a_elems * 2) + 0xFFFULL) & ~0xFFFULL;
    if (a_bytes < 0x1000) a_bytes = 0x1000;
    size_t b_bytes = ((b_elems * 2) + 0xFFFULL) & ~0xFFFULL;
    if (b_bytes < 0x1000) b_bytes = 0x1000;
    size_t c_bytes = ((c_elems * 4) + 0xFFFULL) & ~0xFFFULL;
    if (c_bytes < 0x1000) c_bytes = 0x1000;
    size_t code_bytes = (kernel->code_size + 0xFFFULL) & ~0xFFFULL;
    if (code_bytes < 0x1000) code_bytes = 0x1000;

    NvrmMem code_mem, cbank_mem, a_mem, b_mem, c_mem, marker_mem, qmd_mem;
    if (nvrm_alloc(&ctx.rm, code_bytes, &code_mem) != 0) { m16_native_close(&ctx); return -1; }
    if (nvrm_alloc(&ctx.rm, 0x1000, &cbank_mem) != 0) { m16_native_close(&ctx); return -1; }
    if (nvrm_alloc(&ctx.rm, a_bytes, &a_mem) != 0) { m16_native_close(&ctx); return -1; }
    if (nvrm_alloc(&ctx.rm, b_bytes, &b_mem) != 0) { m16_native_close(&ctx); return -1; }
    if (nvrm_alloc(&ctx.rm, c_bytes, &c_mem) != 0) { m16_native_close(&ctx); return -1; }
    if (nvrm_alloc(&ctx.rm, 0x1000, &marker_mem) != 0) { m16_native_close(&ctx); return -1; }
    if (nvrm_alloc(&ctx.rm, 0x10000, &qmd_mem) != 0) { m16_native_close(&ctx); return -1; }

    /* Copy dynamic machine code into code_mem */
    memcpy(code_mem.cpu, kernel->code, kernel->code_size);

    /* Populate input buffers A, B and initialize C with poison */
    memcpy(a_mem.cpu, h_a, a_elems * 2);
    memcpy(b_mem.cpu, h_b, b_elems * 2);
    float *c_dev = (float *)c_mem.cpu;
    for (size_t i = 0; i < c_elems; i++) {
        c_dev[i] = -999.0f;
    }

    uint32_t threads_x = 32;
    uint32_t threads_y = 1;
    uint32_t grid_x = spec->n / 8;
    uint32_t grid_y = spec->m / 16;
    if (grid_x == 0) grid_x = 1;
    if (grid_y == 0) grid_y = 1;

    uint32_t cbank_data[OMEGA_BW_CBANK_DRIVER_WORDS];
    omega_blackwell_build_cbank_driver_2d(cbank_data, cbank_mem.va, threads_x, threads_y, grid_x, grid_y);

    uint32_t cbank_args[OMEGA_BW_CBANK_MATMUL_ARGS_WORDS];
    omega_blackwell_build_cbank_args_matmul(cbank_args, a_mem.va, b_mem.va, c_mem.va, spec->m, spec->k, spec->n);

    memcpy(cbank_mem.cpu, cbank_data, sizeof(cbank_data));
    memcpy((uint8_t *)cbank_mem.cpu + 0x380, cbank_args, sizeof(cbank_args));

    uint64_t qmd0_va = qmd_mem.va;
    uint64_t qmd1_va = qmd_mem.va + 0x1000;
    uint64_t sem_va  = qmd_mem.va + 0x2000;
    uint64_t scratch_va = qmd_mem.va + 0x4000;

    OmegaBlackwellQmdConfig qmd_cfg = {
        .code_va = code_mem.va,
        .cbank_va = cbank_mem.va,
        .scratch_va = scratch_va,
        .sem_va = sem_va,
        .qmd0_va = qmd0_va,
        .qmd1_va = qmd1_va,
        .num_elements = c_elems,
        .threads_per_block = 32,
        .grid_width = grid_x * grid_y,
        .threads_x = threads_x,
        .threads_y = threads_y,
        .grid_x = grid_x,
        .grid_y = grid_y,
        .gpr_count = 64
    };

    uint32_t qmd0_words[OMEGA_BW_QMD_WORDS];
    uint32_t qmd1_words[OMEGA_BW_QMD_WORDS];
    omega_blackwell_build_qmd0(qmd0_words, qmd0_va, qmd1_va);
    omega_blackwell_build_qmd1(qmd1_words, &qmd_cfg);

    if (omega_blackwell_verify_qmd_invariants(qmd1_words) != 0) { m16_native_close(&ctx); return -1; }

    memcpy(qmd_mem.cpu, qmd0_words, sizeof(qmd0_words));
    memcpy((uint8_t *)qmd_mem.cpu + 0x1000, qmd1_words, sizeof(qmd1_words));

    volatile uint32_t *hsem = (volatile uint32_t *)((uint8_t *)qmd_mem.cpu + 0x2000);
    volatile uint32_t *hmarker = (volatile uint32_t *)marker_mem.cpu;
    *hsem = 0;
    *hmarker = 0;
    __asm__ volatile("dsb sy" ::: "memory");

    uint32_t pb[1024];
    size_t pb_len = 0;

    memcpy(&pb[pb_len], SETUP_WORDS, sizeof(SETUP_WORDS));
    pb_len += sizeof(SETUP_WORDS) / 4;

    pb[pb_len++] = nvrm_mthd(1, 0x0188, 2);
    pb[pb_len++] = (uint32_t)(cbank_mem.va >> 32);
    pb[pb_len++] = (uint32_t)cbank_mem.va;
    pb[pb_len++] = nvrm_mthd(1, 0x0180, 2);
    pb[pb_len++] = 0x00000380;
    pb[pb_len++] = 0x00000001;
    pb[pb_len++] = nvrm_mthd(1, 0x01b0, 1);
    pb[pb_len++] = 0x00000041;
    pb[pb_len++] = (224 << 16) | (1 << 13) | (0x01b4 >> 2) | (6u << 28);
    memcpy(&pb[pb_len], cbank_data, 224 * 4);
    pb_len += 224;

    pb[pb_len++] = nvrm_mthd(1, 0x0188, 2);
    pb[pb_len++] = (uint32_t)((cbank_mem.va + 0x380) >> 32);
    pb[pb_len++] = (uint32_t)(cbank_mem.va + 0x380);
    pb[pb_len++] = nvrm_mthd(1, 0x0180, 2);
    pb[pb_len++] = 0x00000028;
    pb[pb_len++] = 0x00000001;
    pb[pb_len++] = nvrm_mthd(1, 0x01b0, 1);
    pb[pb_len++] = 0x00000041;
    pb[pb_len++] = (10 << 16) | (1 << 13) | (0x01b4 >> 2) | (6u << 28);
    memcpy(&pb[pb_len], cbank_args, 10 * 4);
    pb_len += 10;

    pb[pb_len++] = (98 << 16) | (1 << 13) | (0x0318 >> 2) | (2u << 28);
    pb[pb_len++] = (1u << 30) | (uint32_t)((qmd0_va >> 40) & 0x1ff);
    pb[pb_len++] = (uint32_t)(qmd0_va >> 8);
    memcpy(&pb[pb_len], qmd0_words, 96 * 4);
    pb_len += 96;

    pb[pb_len++] = nvrm_mthd(1, 0x0188, 2);
    pb[pb_len++] = (uint32_t)(sem_va >> 32);
    pb[pb_len++] = (uint32_t)sem_va;
    pb[pb_len++] = nvrm_mthd(1, 0x0180, 2);
    pb[pb_len++] = 0x00000004;
    pb[pb_len++] = 0x00000001;
    pb[pb_len++] = nvrm_mthd(1, 0x01b0, 1);
    pb[pb_len++] = 0x00000041;
    pb[pb_len++] = (1 << 16) | (1 << 13) | (0x01b4 >> 2) | (6u << 28);
    pb[pb_len++] = OMEGA_BW_SEMAPHORE_INTERMEDIATE_INIT;

    pb[pb_len++] = (98 << 16) | (1 << 13) | (0x0318 >> 2) | (2u << 28);
    pb[pb_len++] = (1u << 30) | (uint32_t)((qmd1_va >> 40) & 0x1ff);
    pb[pb_len++] = (uint32_t)(qmd1_va >> 8);
    memcpy(&pb[pb_len], qmd1_words, 96 * 4);
    pb_len += 96;

    pb[pb_len++] = nvrm_mthd(0, 0x005c, 5);
    pb[pb_len++] = (uint32_t)marker_mem.va;
    pb[pb_len++] = (uint32_t)(marker_mem.va >> 32);
    pb[pb_len++] = OMEGA_BW_MARKER_COMPLETION_PAYLOAD;
    pb[pb_len++] = 0;
    pb[pb_len++] = 0x1 | (1u << 20);

    uint64_t t_start = current_time_ns();
    if (m16_native_submit_methods(&ctx, pb, pb_len) != 0) { m16_native_close(&ctx); return -1; }

    if (m16_native_wait_marker(hmarker, OMEGA_BW_MARKER_COMPLETION_PAYLOAD, 5000) != 0) { m16_native_close(&ctx); return -1; }
    uint64_t t_end = current_time_ns();

    if (*hsem != OMEGA_BW_SEMAPHORE_INTERMEDIATE_DONE) { m16_native_close(&ctx); return -1; }

    memcpy(h_c_out, c_mem.cpu, c_elems * sizeof(float));

    float *h_expected = (float *)malloc(c_elems * sizeof(float));
    if (!h_expected) { m16_native_close(&ctx); return -1; }

    if (spec->precision == OMEGA_MATMUL_PRECISION_FP16) {
        omega_matmul_cpu_oracle_f16((const uint16_t *)h_a, (const uint16_t *)h_b, h_expected, spec->m, spec->k, spec->n);
    } else {
        omega_matmul_cpu_oracle_bf16((const uint16_t *)h_a, (const uint16_t *)h_b, h_expected, spec->m, spec->k, spec->n);
    }

    float max_abs = 0.0f;
    float max_rel = 0.0f;
    size_t mismatches = 0;
    for (size_t i = 0; i < c_elems; i++) {
        float exp = h_expected[i];
        float act = h_c_out[i];
        float diff = fabsf(exp - act);
        float rel = (fabsf(exp) > 1e-6f) ? (diff / fabsf(exp)) : diff;
        if (diff > max_abs) max_abs = diff;
        if (rel > max_rel) max_rel = rel;
        if (diff > 1e-4f) {
            mismatches++;
        }
    }
    free(h_expected);

    if (out_max_abs_err) *out_max_abs_err = max_abs;
    if (out_max_rel_err) *out_max_rel_err = max_rel;

    if (exec_info) {
        exec_info->launch_timestamp_ns = t_start;
        exec_info->completion_timestamp_ns = t_end;
        exec_info->elapsed_ns = (t_end > t_start) ? (t_end - t_start) : 0;
        exec_info->completion_marker = *hmarker;
        exec_info->intermediate_semaphore = *hsem;
        exec_info->mismatch_count = mismatches;
        exec_info->parity_verified = (mismatches == 0);
    }

    m16_native_close(&ctx);
    return (mismatches == 0) ? 0 : -1;
}

int omega_blackwell_verify_zero_libcuda_linkage(const char *binary_path) {
    char cmd[512];
    if (binary_path) {
        snprintf(cmd, sizeof(cmd), "ldd %s 2>/dev/null", binary_path);
    } else {
        snprintf(cmd, sizeof(cmd), "ldd /proc/self/exe 2>/dev/null");
    }

    FILE *f = popen(cmd, "r");
    if (!f) return -1;

    char line[512];
    bool found_cuda = false;
    while (fgets(line, sizeof(line), f)) {
        if (strstr(line, "libcuda.so") || strstr(line, "libcudart.so")) {
            found_cuda = true;
            break;
        }
    }
    pclose(f);
    return found_cuda ? -1 : 0;
}

int omega_blackwell_verify_zero_cuda_symbols(const char *binary_path) {
    char cmd[512];
    if (binary_path) {
        snprintf(cmd, sizeof(cmd), "nm -u %s 2>/dev/null", binary_path);
    } else {
        snprintf(cmd, sizeof(cmd), "nm -u /proc/self/exe 2>/dev/null");
    }

    FILE *f = popen(cmd, "r");
    if (!f) return -1;

    char line[512];
    bool found_cuda_sym = false;
    while (fgets(line, sizeof(line), f)) {
        char sym[256];
        if (sscanf(line, " %*s %255s", sym) == 1 || sscanf(line, "%255s", sym) == 1) {
            if (strncmp(sym, "cu", 2) == 0 || strncmp(sym, "cuda", 4) == 0) {
                found_cuda_sym = true;
                break;
            }
        }
    }
    pclose(f);
    return found_cuda_sym ? -1 : 0;
}

int omega_blackwell_verify_zero_libcuda_runtime(void) {
    FILE *f = fopen("/proc/self/maps", "r");
    if (!f) return -1;

    char line[512];
    bool found_cuda = false;
    while (fgets(line, sizeof(line), f)) {
        if (strstr(line, "libcuda") || strstr(line, "libcudart")) {
            found_cuda = true;
            break;
        }
    }
    fclose(f);
    return found_cuda ? -1 : 0;
}
