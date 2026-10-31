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
uint32_t omega_native_repack(uint64_t *result, const void *p0, uint64_t p0_n, const void *p1, uint64_t p1_n, const void *p2, uint64_t p2_n, void *p3, uint64_t p3_n, void *p4, uint64_t p4_n);
uint32_t omega_native_quant(uint64_t *result, uint64_t p0, void *p1, uint64_t p1_n, void *p2, uint64_t p2_n);
uint32_t omega_native_compose(uint64_t *result, const void *p0, uint64_t p0_n, void *p1, uint64_t p1_n, void *p2, uint64_t p2_n, void *p3, uint64_t p3_n, void *p4, uint64_t p4_n);
uint32_t omega_bc_encode(uint64_t *result, const void *p0, uint64_t p0_n, const void *p1, uint64_t p1_n, uint64_t p2, uint64_t p3, void *p4, uint64_t p4_n);
uint32_t omega_native_cache(uint64_t *result, void *p0, uint64_t p0_n, const void *p1, uint64_t p1_n, const void *p2, uint64_t p2_n, uint64_t p3);
uint32_t omega_native_lease(uint64_t *result, void *p0, uint64_t p0_n, const void *p1, uint64_t p1_n, const void *p2, uint64_t p2_n, uint64_t p3);
uint32_t omega_native_slotrev(uint64_t *result, const void *p0, uint64_t p0_n, uint64_t p1, uint64_t p2);
uint32_t omega_native_event(uint64_t *result, void *p0, uint64_t p0_n, void *p1, uint64_t p1_n, uint64_t p2, uint64_t p3);
uint32_t omega_native_validate(uint64_t *result, const void *p0, uint64_t p0_n, const void *p1, uint64_t p1_n, const void *p2, uint64_t p2_n, const void *p3, uint64_t p3_n, const void *p4, uint64_t p4_n, void *p5, uint64_t p5_n);
uint32_t omega_native_key(uint64_t *result, const void *p0, uint64_t p0_n, uint64_t p1, void *p2, uint64_t p2_n);
uint32_t omega_native_codekey(uint64_t *result, const void *p0, uint64_t p0_n, uint64_t p1, void *p2, uint64_t p2_n);
uint32_t omega_native_templates(uint64_t *result, const void *p0, uint64_t p0_n, const void *p1, uint64_t p1_n, const void *p2, uint64_t p2_n, const void *p3, uint64_t p3_n);
uint32_t omega_native_graph(uint64_t *result, const void *p0, uint64_t p0_n, const void *p1, uint64_t p1_n, void *p2, uint64_t p2_n, const void *p3, uint64_t p3_n, const void *p4, uint64_t p4_n, void *p5, uint64_t p5_n);
uint32_t omega_native_context_lookup(uint64_t *result, const void *p0, uint64_t p0_n, uint64_t p1, const void *p2, uint64_t p2_n);
uint32_t omega_native_binding(uint64_t *result, const void *p0, uint64_t p0_n, const void *p1, uint64_t p1_n, const void *p2, uint64_t p2_n, void *p3, uint64_t p3_n);
uint32_t omega_moe_plan(uint64_t *result, const void *p0, uint64_t p0_n, const void *p1, uint64_t p1_n, void *p2, uint64_t p2_n, void *p3, uint64_t p3_n, void *p4, uint64_t p4_n, void *p5, uint64_t p5_n);
uint32_t omega_moe_group(uint64_t *result, const void *p0, uint64_t p0_n, void *p1, uint64_t p1_n);
uint32_t omega_moe_ids32(uint64_t *result, const void *p0, uint64_t p0_n, uint64_t p1, void *p2, uint64_t p2_n);
uint32_t omega_moe_maps32(uint64_t *result, const void *p0, uint64_t p0_n, const void *p1, uint64_t p1_n, void *p2, uint64_t p2_n);
uint32_t omega_moe_segs32(uint64_t *result, const void *p0, uint64_t p0_n, const void *p1, uint64_t p1_n, const void *p2, uint64_t p2_n, void *p3, uint64_t p3_n);
uint32_t omega_moe_windows(uint64_t *result, const void *p0, uint64_t p0_n, void *p1, uint64_t p1_n, void *p2, uint64_t p2_n);
uint32_t omega_moe_admit(uint64_t *result, const void *p0, uint64_t p0_n, const void *p1, uint64_t p1_n, const void *p2, uint64_t p2_n, void *p3, uint64_t p3_n, void *p4, uint64_t p4_n, void *p5, uint64_t p5_n);
uint32_t omega_native_mcpack(uint64_t *result, const void *p0, uint64_t p0_n, const void *p1, uint64_t p1_n, const void *p2, uint64_t p2_n, void *p3, uint64_t p3_n);
uint32_t omega_native_mcbind(uint64_t *result, const void *p0, uint64_t p0_n, const void *p1, uint64_t p1_n, const void *p2, uint64_t p2_n, const void *p3, uint64_t p3_n, uint64_t p4, void *p5, uint64_t p5_n);
uint32_t omega_native_mcblob(uint64_t *result, const void *p0, uint64_t p0_n, const void *p1, uint64_t p1_n, const void *p2, uint64_t p2_n, uint64_t p3, const void *p4, uint64_t p4_n, uint64_t p5);
uint32_t omega_native_mccover(uint64_t *result, const void *p0, uint64_t p0_n, const void *p1, uint64_t p1_n, const void *p2, uint64_t p2_n, const void *p3, uint64_t p3_n);
uint32_t omega_native_mcintent(uint64_t *result, const void *p0, uint64_t p0_n, const void *p1, uint64_t p1_n, const void *p2, uint64_t p2_n, const void *p3, uint64_t p3_n, const void *p4, uint64_t p4_n);
uint32_t omega_native_mcseal(uint64_t *result, const void *p0, uint64_t p0_n, const void *p1, uint64_t p1_n, uint64_t p2, void *p3, uint64_t p3_n);
uint32_t omega_native_mccode(uint64_t *result, const void *p0, uint64_t p0_n, uint64_t p1, void *p2, uint64_t p2_n);
uint32_t omega_native_mclaunch(uint64_t *result, const void *p0, uint64_t p0_n, const void *p1, uint64_t p1_n, const void *p2, uint64_t p2_n, const void *p3, uint64_t p3_n, void *p4, uint64_t p4_n);
uint32_t omega_native_mcdir(uint64_t *result, const void *p0, uint64_t p0_n, const void *p1, uint64_t p1_n, void *p2, uint64_t p2_n);
uint32_t omega_native_mcalloc(uint64_t *result, const void *p0, uint64_t p0_n, const void *p1, uint64_t p1_n, void *p2, uint64_t p2_n);
uint32_t omega_native_mcseries(uint64_t *result, const void *p0, uint64_t p0_n, const void *p1, uint64_t p1_n, const void *p2, uint64_t p2_n, void *p3, uint64_t p3_n);
uint32_t omega_native_revision(uint64_t *result, uint64_t p0);
uint32_t omega_native_epoch(uint64_t *result, uint64_t p0);
uint32_t omega_native_f16(uint64_t *result, uint64_t p0);
#ifdef __cplusplus
}
#endif
#endif
