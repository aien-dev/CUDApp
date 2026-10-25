/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */
/*
 * omega_cubin: package Omega-generated sm_121 machine code into the NVIDIA
 * cubin ELF container that the CUDA driver (and MAX's device layer) can load.
 *
 * This is a container writer, not a GPU driver, compiler or verifier. It does
 * not know a device address, open a device, allocate memory or launch work. It
 * writes only the bytes that describe the kernel: text, kernel parameter
 * metadata, an empty constant bank (the loader fills parameters), and the
 * standard note/compat sections. The input text must already have been
 * produced and digest-bound by the existing CUDApp code generator.
 *
 * Layout facts below were established from a ptxas 13.0.88 -arch=sm_121
 * reference output and are re-checked by nvdisasm/cuobjdump in
 * tools/artifact_smoke.sh (the C test checks the digest/identity layer):
 *   - ELF64 little-endian, e_machine 190 (EM_CUDA), e_flags 0x6007902 (sm_121),
 *     OS/ABI 0x41 (NV), ABI version 8;
 *   - kernel parameters live in constant bank 0 starting at byte 0x380;
 *   - a function symbol must carry STO_CUDA_ENTRY (st_other 0x10);
 *   - .nv.info.<fn> carries EIATTR_KPARAM_INFO entries (format 0x1704), the
 *     parameter constant-bank size (0x1903), and the index of the function's
 *     constant-bank section symbol (0x0a04);
 *   - the module-level .nv.info names the first function symbol and the symbol
 *     count (0x2f04); .nv.compat and .note.nv.cuinfo are required by the
 *     offline tools.
 * Nothing here is copied from an NVIDIA binary; these are ABI facts checked
 * against a public tool's output on this host.
 *
 * Limits, deliberate and recorded:
 *   - e_phoff/e_phnum are 0: this is a section-based container. The ptxas
 *     reference also carries PT_LOAD program headers; whether cuModuleLoadData
 *     needs them is exactly what the NOT_RUN device probe must decide.
 *   - EIATTR_EXIT_INSTR_OFFSETS is omitted because this writer does not decode
 *     SASS and will not fabricate the EXIT byte offset. ptxas emits it; the
 *     driver treats it as optional metadata until a device load proves else.
 *   - sh_info of the .nv.* sections uses this writer's section indices, which
 *     cuobjdump accepts; it does not reproduce ptxas's symbol indices.
 */
#ifndef OMEGA_CUBIN_H
#define OMEGA_CUBIN_H

#include <stddef.h>
#include <stdint.h>

#define OMEGA_CUBIN_ABI_VERSION 8u
#define OMEGA_CUBIN_EFLAGS_SM121 0x6007902u
#define OMEGA_CUBIN_SM_121 121u
#define OMEGA_CUBIN_PARAM_BASE 0x380u
#define OMEGA_CUBIN_CBANK_MAX 1024u
#define OMEGA_CUBIN_MAX_PARAMS 32u
#define OMEGA_CUBIN_MAX_NAME 64u

/* Named refusals. 0 is the only acceptance. */
enum {
    OMEGA_CUBIN_OK = 0,
    OMEGA_CUBIN_E_ARG = 1,        /* NULL/empty input, empty or invalid name */
    OMEGA_CUBIN_E_TEXT = 2,       /* text length 0, not a multiple of 16, or too large */
    OMEGA_CUBIN_E_PARAMS = 3,     /* too many, unsorted, overlapping, wrong size/alignment */
    OMEGA_CUBIN_E_CBANK = 4,      /* parameter area exceeds the 1024-byte bank */
    OMEGA_CUBIN_E_GPR = 5,        /* register count above 255 */
    OMEGA_CUBIN_E_SHARED = 6,     /* reserved shared size not a multiple of 4 */
    OMEGA_CUBIN_E_MEM = 7         /* allocation failure */
};

/* One kernel parameter: its byte offset in constant bank 0, its size (8 for a
 * pointer/u64, 4 for u32, 2 for u16, 1 for u8), and whether it is a pointer
 * (which selects the parameter-info type word). */
typedef struct {
    uint32_t offset;
    uint32_t size;
    uint8_t is_pointer;
} OmegaCubinParam;

const char *omega_cubin_rc_name(int rc);

/* Pack text into a cubin. *out receives a malloc'd buffer of *out_len bytes;
 * free with omega_cubin_free. Refuses malformed descriptors rather than
 * emitting a best-effort container. */
int omega_cubin_pack(const char *fn_name,
                     const uint8_t *text, size_t text_len,
                     const OmegaCubinParam *params, size_t nparams,
                     uint32_t cbank_size, uint32_t gpr_count,
                     uint32_t shared_reserved,
                     uint8_t **out, size_t *out_len, char *err, size_t err_cap);

void omega_cubin_free(uint8_t *p);

#endif /* OMEGA_CUBIN_H */
