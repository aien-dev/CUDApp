/* Generated checked portable Omega declarations. Regenerate with om/units.sh.
 * Status: 0 done, 1..14 trap, 32 argument refused, 33 reentry.
 * Only a successful call writes result. Views use exact declared lengths.
 * Implementation is Omega lowered by omega-bridge, not the native AArch64 ABI. */
#ifndef CUDAPP_PORTABLE_OMEGA_H
#define CUDAPP_PORTABLE_OMEGA_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
#define OMEGA_OK 0u
#define OMEGA_BAD_ARG 32u
#define OMEGA_REENTRY 33u
uint32_t omega_bw_enc_sm121(uint64_t *result, const void *p0, uint64_t p0_n, const void *p1, uint64_t p1_n, uint64_t p2, uint64_t p3, void *p4, uint64_t p4_n);
uint32_t omega_cubin_elf(uint64_t *result, const void *p0, uint64_t p0_n, const void *p1, uint64_t p1_n, const void *p2, uint64_t p2_n, void *p3, uint64_t p3_n);
uint32_t omega_ampere_enc_sm86(uint64_t *result, const void *p0, uint64_t p0_n, const void *p1, uint64_t p1_n, uint64_t p2, uint64_t p3, void *p4, uint64_t p4_n);
uint32_t omega_amp_runtime(uint64_t *result, const void *p0, uint64_t p0_n, void *p1, uint64_t p1_n);
#ifdef __cplusplus
}
#endif
#endif
