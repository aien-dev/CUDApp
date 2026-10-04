#ifndef OMEGA_BLACKWELL_SUBMIT_H
#define OMEGA_BLACKWELL_SUBMIT_H

#include "omega_types.h"
#include "omega_vector.h"
#include "omega_blackwell_realize.h"
#include "omega_blackwell_matmul.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define OMEGA_BW_MARKER_COMPLETION_PAYLOAD 0x44444444U
#define OMEGA_BW_MARKER2_PAYLOAD 0x46464646u /* C3 second release, after the L2 flush */
/* Shader-cache invalidate, emitted as the first compute method of every launch that
 * uploads kernel code (cut 1b finding: the SM instruction cache is not coherent with
 * host writes; after a kernel-cache slot was evicted and its code address reused, the
 * chip ran stale instructions). Method and bits from NVIDIA open-gpu-doc
 * classes/compute/clcec0.h (BLACKWELL_COMPUTE_B): NVCEC0_INVALIDATE_SHADER_CACHES 0x021c,
 * _INSTRUCTION 0:0, _DATA 4:4, _CONSTANT 12:12. The shipped driver tree's clcec0.h is a
 * 30-line class-id stub and does not list the method; chip-proven by receipt
 * FB1-CUT1B-4345406 (18/18 parity after the fix). Emit: nvrm_mthd(1, MTHD, 1), ALL. */
#define OMEGA_BW_MTHD_INVALIDATE_SHADER_CACHES 0x021cu
#define OMEGA_BW_INVALIDATE_SHADER_CACHES_ALL ((1u << 0) | (1u << 4) | (1u << 12))
#define OMEGA_BW_SEMAPHORE_INTERMEDIATE_INIT 5U
#define OMEGA_BW_SEMAPHORE_INTERMEDIATE_DONE 6U

typedef struct {
    char target_chip[64];
    uint32_t sm_architecture;
    uint32_t element_count;
    uint64_t launch_timestamp_ns;
    uint64_t completion_timestamp_ns;
    uint64_t elapsed_ns;
    uint32_t completion_marker;
    uint32_t intermediate_semaphore;
    bool parity_verified;
    bool zero_libcuda_linkage;
    bool zero_cuda_symbols;
    bool zero_libcuda_runtime;
} OmegaBlackwellVectorExecution;

typedef struct {
    char target_chip[64];
    uint32_t sm_architecture;
    uint32_t m, k, n;
    uint64_t launch_timestamp_ns;
    uint64_t completion_timestamp_ns;
    uint64_t elapsed_ns;
    uint32_t completion_marker;
    uint32_t intermediate_semaphore;
    bool parity_verified;
    size_t mismatch_count;
    bool zero_libcuda_linkage;
    bool zero_cuda_symbols;
    bool zero_libcuda_runtime;
} OmegaBlackwellMatMulExecution;

/* Execute vector addition on physical GB10 silicon. Since cut A3b1 (DRAFT, chip-unproven) the
 * device sequence runs through omega_gpu_execute with the Blackwell backend in
 * src/omega_blackwell_engine.c; this wrapper keeps the host oracle and the result fields. */
int omega_blackwell_execute_vector(const OmegaVectorSpec *spec,
                                  const OmegaBlackwellRealization *real,
                                  const uint32_t *h_a,
                                  const uint32_t *h_b,
                                  uint32_t *h_c_out,
                                  OmegaBlackwellVectorExecution *exec_info);

/* Execute dynamic matrix multiplication on physical GB10 silicon through frozen M16 native submission */
/* Execute dynamic tensor matrix multiplication on physical GB10 silicon through frozen M16 native submission */
int omega_blackwell_execute_matmul_tensor(const OmegaMatMulSpec *spec,
                                         const OmegaBlackwellKernel *kernel,
                                         const void *h_a,
                                         const void *h_b,
                                         float *h_c_out,
                                         OmegaBlackwellMatMulExecution *exec_info,
                                         float *out_max_abs_err,
                                         float *out_max_rel_err);

int omega_blackwell_execute_matmul(const OmegaMatMulSpec *spec,
                                  const OmegaBlackwellKernel *kernel,
                                  const uint32_t *h_a,
                                  const uint32_t *h_b,
                                  uint32_t *h_c_out,
                                  OmegaBlackwellMatMulExecution *exec_info);

/* Validate that omegatool binary has zero dynamic linkage to libcuda.so or libcudart.so */
int omega_blackwell_verify_zero_libcuda_linkage(const char *binary_path);

/* Validate that omegatool binary has zero undefined symbols starting with cu or cuda */
int omega_blackwell_verify_zero_cuda_symbols(const char *binary_path);

/* Validate that current running process has zero mapped regions of libcuda */
int omega_blackwell_verify_zero_libcuda_runtime(void);

#endif /* OMEGA_BLACKWELL_SUBMIT_H */
