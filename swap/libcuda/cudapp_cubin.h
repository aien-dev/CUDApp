/* SPDX-License-Identifier: MIT */
#ifndef CUDAPP_CUBIN_H
#define CUDAPP_CUBIN_H
/*
 * cudapp_cubin: reads the machine code a CUDA program carries, so CUDA++'s libcuda.so.1 can load it.
 *
 * Three containers reach cuModuleLoadData / cuLibraryLoadData:
 *   - a fat binary (magic 0xBA55ED50): a list of entries, each an ELF image for one GPU architecture or a PTX
 *     text, optionally LZ4-compressed;
 *   - the wrapper nvcc puts around it in a program (magic 0x466243b1, pointer to the fat binary);
 *   - a bare ELF image (a .cubin file).
 * From the ELF image it takes, per kernel: the machine code (.text.<name>), the register count (top byte of
 * that section's sh_info), the static shared memory (.nv.shared.<name>), and from .nv.info.<name> where the
 * parameters live in constant bank 0 and how big each one is. Anything it cannot place is not guessed: the kernel
 * is marked refused with the reason, and the launch says it.
 *
 * Written from the ELF format (System V gABI) and from reading nvcc's output on this machine; no NVIDIA code.
 * Pure C, no device: tests/swap/test_cubin.c runs it on the host.
 */
#include <stddef.h>
#include <stdint.h>

#define CUDAPP_FATBIN_MAGIC 0xBA55ED50u
#define CUDAPP_FATBIN_WRAPPER_MAGIC 0x466243b1u
#define CUDAPP_MAX_PARAMS 64
#define CUDAPP_NAME_MAX 256

/* The architectures whose machine code runs on GB10 (sm_121): sm_121 itself and sm_120 (same major version,
 * binary compatible upward within a major version). */
#define CUDAPP_ARCH_OK(a) ((a) == 120u || (a) == 121u)

typedef struct {
    char name[CUDAPP_NAME_MAX];
    const uint8_t *code;      /* points into the module's image copy */
    size_t code_size;
    uint32_t regs;            /* registers per thread (sh_info >> 24) */
    uint32_t shared_static;   /* bytes of .nv.shared.<name> */
    uint32_t param_base;      /* offset of the parameter block in constant bank 0 (0x380 on sm_12x) */
    uint32_t param_size;      /* bytes of the parameter block */
    uint32_t n_params;
    uint16_t param_off[CUDAPP_MAX_PARAMS]; /* offset of parameter i inside the block */
    uint16_t param_len[CUDAPP_MAX_PARAMS]; /* its bytes */
    uint32_t stack_bytes;     /* local memory the kernel needs per thread (max stack); 0 = none */
    uint32_t max_threads[3];  /* __launch_bounds__ / reqntid, 0 = none */
    char refused[160];        /* "" = loadable; else why this kernel cannot run on CUDA++ v0 */
} CudappKernelInfo;

typedef struct {
    uint32_t arch;            /* sm number of the image chosen (121, 120), 0 if none */
    uint8_t *image;           /* owned copy of the chosen ELF image (decompressed) */
    size_t image_size;
    uint32_t n_kernels;
    CudappKernelInfo *kernels; /* owned */
    uint32_t has_ptx;         /* the container also held PTX (not used: no PTX compiler yet) */
    uint32_t archs_seen[16];  /* every architecture entry in the container, for the refusal text */
    uint32_t n_archs_seen;
    char refused[200];        /* "" = image loaded; else why the module cannot load */
} CudappModuleImage;

/* Status codes (mapped to CUresult by the caller). */
enum {
    CUDAPP_IMG_OK = 0,
    CUDAPP_IMG_BAD = 1,          /* not a container we know, or a malformed one */
    CUDAPP_IMG_NO_BINARY = 2,    /* well-formed, but no machine code for sm_120/121 (PTX only, or other archs) */
    CUDAPP_IMG_NOMEM = 3,
};

/* Parses `data` (fat binary, wrapper or ELF; `size` 0 = unknown, the container's own header bounds it).
 * Fills *out (zeroed first). On any status but OK, out->refused names the reason. */
int cudapp_image_load(const void *data, size_t size, CudappModuleImage *out);
void cudapp_image_free(CudappModuleImage *m);
const CudappKernelInfo *cudapp_image_kernel(const CudappModuleImage *m, const char *name);

/* LZ4 block decompression (the fat binary compression). Returns bytes written, or -1 on malformed input or
 * overflow of dst_cap. Exposed for the tests. */
long cudapp_lz4_block_decompress(const uint8_t *src, size_t src_len, uint8_t *dst, size_t dst_cap);

#endif
