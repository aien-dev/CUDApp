/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */
/* Portable host stand-in used only when the host cannot assemble the seeded
 * bindings. src/cudapp_om_units.c and src/cudapp_freest_units.c stay as
 * om/units.sh wrote them. This file is not that regeneration, and the AArch64
 * library does not link it.
 *
 * ADR 0043 missing feature: those generated units are AArch64 machine code, and
 * seeded Omega does not yet emit a binding the build host can assemble. This
 * file is migration-pending C. The library compiles it only on a non-AArch64
 * host, and only with CUDAPP_MIGRATION_PENDING_C_ENCODER and
 * CUDAPP_MIGRATION_PENDING_C_CUBIN, the switches that select the C encoder and
 * the C cubin writer. The host test defines BINDING_PORTABLE. Extension
 * proposal drafted, not yet filed on omega-mv:
 * evidence/HOST-X86-20261030/proposal-binding-portable.md
 *
 * omega_native_revision and omega_native_epoch follow the committed
 * freestanding units. Those units were executed on an x86-64 host with
 * qemu-aarch64 8.2.2 and aarch64-linux-gnu-gcc 13, linking src/cudapp_freest_units.c.
 * Observed: revision(0)=1, revision(1)=2, revision(UINT64_MAX-1)=UINT64_MAX,
 * revision(UINT64_MAX)=0; epoch(0)=1, epoch(3)=4, epoch(UINT64_MAX)=UINT64_MAX;
 * a null or misaligned result pointer returns 32. The same vectors are checked
 * by tests/omega/test-binding-portable.sh.
 *
 * Every other entry returns 34 (OMEGA_BAD_RT) and does not write *result.
 * That refusal is not a result from the seeded unit. Callers that need the
 * seeded encoder, cubin packer, or the other units are target-only on AArch64.
 * On this host the C encoder and the C cubin writer are selected instead, so
 * those two entries are not on the portable codegen path.
 */
#if !defined(CUDAPP_MIGRATION_PENDING_C_ENCODER) && !defined(CUDAPP_MIGRATION_PENDING_C_CUBIN) && !defined(BINDING_PORTABLE)
#error src/cudapp_binding_portable.c is migration-pending C for a non-AArch64 host
#endif
#include <stddef.h>
#include <stdint.h>
#include "cudapp_native.h"

static int result_refused(const uint64_t *result)
{
    uintptr_t p = (uintptr_t)result;
    return result == 0 || (p % 8u) != 0u || p > UINTPTR_MAX - 8u;
}

uint32_t omega_native_revision(uint64_t *result, uint64_t o_previous)
{
    if (result_refused(result))
        return OMEGA_BAD_ARG;
    *result = o_previous == UINT64_MAX ? 0u : o_previous + 1u;
    return OMEGA_OK;
}

uint32_t omega_native_epoch(uint64_t *result, uint64_t o_previous)
{
    if (result_refused(result))
        return OMEGA_BAD_ARG;
    *result = o_previous == UINT64_MAX ? UINT64_MAX : o_previous + 1u;
    return OMEGA_OK;
}

uint32_t omega_bw_enc_sm121(uint64_t *result, const void *o_ins, uint64_t o_ins_len, const void *o_phys, uint64_t o_phys_len, uint64_t o_n, uint64_t o_pad, void *o_code, uint64_t o_code_len)
{
    (void)result; (void)o_ins; (void)o_ins_len; (void)o_phys; (void)o_phys_len; (void)o_n; (void)o_pad; (void)o_code; (void)o_code_len;
    return OMEGA_BAD_RT;
}
uint32_t omega_cubin_elf(uint64_t *result, const void *o_m, uint64_t o_m_len, const void *o_nm, uint64_t o_nm_len, const void *o_tx, uint64_t o_tx_len, void *o_o, uint64_t o_o_len)
{
    (void)result; (void)o_m; (void)o_m_len; (void)o_nm; (void)o_nm_len; (void)o_tx; (void)o_tx_len; (void)o_o; (void)o_o_len;
    return OMEGA_BAD_RT;
}
uint32_t omega_native_repack(uint64_t *result, const void *o_d, uint64_t o_d_len, const void *o_src, uint64_t o_src_len, const void *o_sc, uint64_t o_sc_len, void *o_dst, uint64_t o_dst_len, void *o_ds, uint64_t o_ds_len)
{
    (void)result; (void)o_d; (void)o_d_len; (void)o_src; (void)o_src_len; (void)o_sc; (void)o_sc_len; (void)o_dst; (void)o_dst_len; (void)o_ds; (void)o_ds_len;
    return OMEGA_BAD_RT;
}
uint32_t omega_native_quant(uint64_t *result, uint64_t o_op, void *o_d, uint64_t o_d_len, void *o_wire, uint64_t o_wire_len)
{
    (void)result; (void)o_op; (void)o_d; (void)o_d_len; (void)o_wire; (void)o_wire_len;
    return OMEGA_BAD_RT;
}
uint32_t omega_native_compose(uint64_t *result, const void *o_cfg, uint64_t o_cfg_len, void *o_ins, uint64_t o_ins_len, void *o_meta, uint64_t o_meta_len, void *o_phys, uint64_t o_phys_len, void *o_out, uint64_t o_out_len)
{
    (void)result; (void)o_cfg; (void)o_cfg_len; (void)o_ins; (void)o_ins_len; (void)o_meta; (void)o_meta_len; (void)o_phys; (void)o_phys_len; (void)o_out; (void)o_out_len;
    return OMEGA_BAD_RT;
}
uint32_t omega_bc_encode(uint64_t *result, const void *o_ins, uint64_t o_ins_len, const void *o_phys, uint64_t o_phys_len, uint64_t o_n, uint64_t o_pad, void *o_code, uint64_t o_code_len)
{
    (void)result; (void)o_ins; (void)o_ins_len; (void)o_phys; (void)o_phys_len; (void)o_n; (void)o_pad; (void)o_code; (void)o_code_len;
    return OMEGA_BAD_RT;
}
uint32_t omega_native_cache(uint64_t *result, void *o_p, uint64_t o_p_len, const void *o_req, uint64_t o_req_len, const void *o_tok, uint64_t o_tok_len, uint64_t o_op)
{
    (void)result; (void)o_p; (void)o_p_len; (void)o_req; (void)o_req_len; (void)o_tok; (void)o_tok_len; (void)o_op;
    return OMEGA_BAD_RT;
}
uint32_t omega_native_lease(uint64_t *result, void *o_p, uint64_t o_p_len, const void *o_req, uint64_t o_req_len, const void *o_tok, uint64_t o_tok_len, uint64_t o_op)
{
    (void)result; (void)o_p; (void)o_p_len; (void)o_req; (void)o_req_len; (void)o_tok; (void)o_tok_len; (void)o_op;
    return OMEGA_BAD_RT;
}
uint32_t omega_native_slotrev(uint64_t *result, const void *o_p, uint64_t o_p_len, uint64_t o_slot, uint64_t o_gen)
{
    (void)result; (void)o_p; (void)o_p_len; (void)o_slot; (void)o_gen;
    return OMEGA_BAD_RT;
}
uint32_t omega_native_event(uint64_t *result, void *o_p, uint64_t o_p_len, void *o_lease, uint64_t o_lease_len, uint64_t o_epoch, uint64_t o_op)
{
    (void)result; (void)o_p; (void)o_p_len; (void)o_lease; (void)o_lease_len; (void)o_epoch; (void)o_op;
    return OMEGA_BAD_RT;
}
uint32_t omega_native_validate(uint64_t *result, const void *o_rec, uint64_t o_rec_len, const void *o_sizes, uint64_t o_sizes_len, const void *o_code, uint64_t o_code_len, const void *o_launch, uint64_t o_launch_len, const void *o_expected, uint64_t o_expected_len, void *o_out, uint64_t o_out_len)
{
    (void)result; (void)o_rec; (void)o_rec_len; (void)o_sizes; (void)o_sizes_len; (void)o_code; (void)o_code_len; (void)o_launch; (void)o_launch_len; (void)o_expected; (void)o_expected_len; (void)o_out; (void)o_out_len;
    return OMEGA_BAD_RT;
}
uint32_t omega_native_key(uint64_t *result, const void *o_data, uint64_t o_data_len, uint64_t o_n, void *o_out, uint64_t o_out_len)
{
    (void)result; (void)o_data; (void)o_data_len; (void)o_n; (void)o_out; (void)o_out_len;
    return OMEGA_BAD_RT;
}
uint32_t omega_native_codekey(uint64_t *result, const void *o_data, uint64_t o_data_len, uint64_t o_n, void *o_out, uint64_t o_out_len)
{
    (void)result; (void)o_data; (void)o_data_len; (void)o_n; (void)o_out; (void)o_out_len;
    return OMEGA_BAD_RT;
}
uint32_t omega_native_templates(uint64_t *result, const void *o_launch, uint64_t o_launch_len, const void *o_cubin, uint64_t o_cubin_len, const void *o_observed, uint64_t o_observed_len, const void *o_sizes, uint64_t o_sizes_len)
{
    (void)result; (void)o_launch; (void)o_launch_len; (void)o_cubin; (void)o_cubin_len; (void)o_observed; (void)o_observed_len; (void)o_sizes; (void)o_sizes_len;
    return OMEGA_BAD_RT;
}
uint32_t omega_native_graph(uint64_t *result, const void *o_q, uint64_t o_q_len, const void *o_rows, uint64_t o_rows_len, void *o_p, uint64_t o_p_len, const void *o_want, uint64_t o_want_len, const void *o_records, uint64_t o_records_len, void *o_out, uint64_t o_out_len)
{
    (void)result; (void)o_q; (void)o_q_len; (void)o_rows; (void)o_rows_len; (void)o_p; (void)o_p_len; (void)o_want; (void)o_want_len; (void)o_records; (void)o_records_len; (void)o_out; (void)o_out_len;
    return OMEGA_BAD_RT;
}
uint32_t omega_native_context_lookup(uint64_t *result, const void *o_keys, uint64_t o_keys_len, uint64_t o_count, const void *o_actual, uint64_t o_actual_len)
{
    (void)result; (void)o_keys; (void)o_keys_len; (void)o_count; (void)o_actual; (void)o_actual_len;
    return OMEGA_BAD_RT;
}
uint32_t omega_native_binding(uint64_t *result, const void *o_real, uint64_t o_real_len, const void *o_launch, uint64_t o_launch_len, const void *o_call, uint64_t o_call_len, void *o_out, uint64_t o_out_len)
{
    (void)result; (void)o_real; (void)o_real_len; (void)o_launch; (void)o_launch_len; (void)o_call; (void)o_call_len; (void)o_out; (void)o_out_len;
    return OMEGA_BAD_RT;
}
uint32_t omega_moe_plan(uint64_t *result, const void *o_c, uint64_t o_c_len, const void *o_ids, uint64_t o_ids_len, void *o_w, uint64_t o_w_len, void *o_sg, uint64_t o_sg_len, void *o_mp, uint64_t o_mp_len, void *o_m, uint64_t o_m_len)
{
    (void)result; (void)o_c; (void)o_c_len; (void)o_ids; (void)o_ids_len; (void)o_w; (void)o_w_len; (void)o_sg; (void)o_sg_len; (void)o_mp; (void)o_mp_len; (void)o_m; (void)o_m_len;
    return OMEGA_BAD_RT;
}
uint32_t omega_moe_group(uint64_t *result, const void *o_r, uint64_t o_r_len, void *o_o, uint64_t o_o_len)
{
    (void)result; (void)o_r; (void)o_r_len; (void)o_o; (void)o_o_len;
    return OMEGA_BAD_RT;
}
uint32_t omega_moe_ids32(uint64_t *result, const void *o_src, uint64_t o_src_len, uint64_t o_n, void *o_ids, uint64_t o_ids_len)
{
    (void)result; (void)o_src; (void)o_src_len; (void)o_n; (void)o_ids; (void)o_ids_len;
    return OMEGA_BAD_RT;
}
uint32_t omega_moe_maps32(uint64_t *result, const void *o_m, uint64_t o_m_len, const void *o_mp, uint64_t o_mp_len, void *o_out, uint64_t o_out_len)
{
    (void)result; (void)o_m; (void)o_m_len; (void)o_mp; (void)o_mp_len; (void)o_out; (void)o_out_len;
    return OMEGA_BAD_RT;
}
uint32_t omega_moe_segs32(uint64_t *result, const void *o_sg, uint64_t o_sg_len, const void *o_m, uint64_t o_m_len, const void *o_dir, uint64_t o_dir_len, void *o_out, uint64_t o_out_len)
{
    (void)result; (void)o_sg; (void)o_sg_len; (void)o_m; (void)o_m_len; (void)o_dir; (void)o_dir_len; (void)o_out; (void)o_out_len;
    return OMEGA_BAD_RT;
}
uint32_t omega_moe_windows(uint64_t *result, const void *o_c, uint64_t o_c_len, void *o_o, uint64_t o_o_len, void *o_m, uint64_t o_m_len)
{
    (void)result; (void)o_c; (void)o_c_len; (void)o_o; (void)o_o_len; (void)o_m; (void)o_m_len;
    return OMEGA_BAD_RT;
}
uint32_t omega_moe_admit(uint64_t *result, const void *o_r, uint64_t o_r_len, const void *o_q, uint64_t o_q_len, const void *o_a, uint64_t o_a_len, void *o_p, uint64_t o_p_len, void *o_o, uint64_t o_o_len, void *o_lease, uint64_t o_lease_len)
{
    (void)result; (void)o_r; (void)o_r_len; (void)o_q; (void)o_q_len; (void)o_a; (void)o_a_len; (void)o_p; (void)o_p_len; (void)o_o; (void)o_o_len; (void)o_lease; (void)o_lease_len;
    return OMEGA_BAD_RT;
}
uint32_t omega_native_mcpack(uint64_t *result, const void *o_s, uint64_t o_s_len, const void *o_h, uint64_t o_h_len, const void *o_w, uint64_t o_w_len, void *o_o, uint64_t o_o_len)
{
    (void)result; (void)o_s; (void)o_s_len; (void)o_h; (void)o_h_len; (void)o_w; (void)o_w_len; (void)o_o; (void)o_o_len;
    return OMEGA_BAD_RT;
}
uint32_t omega_native_mcbind(uint64_t *result, const void *o_e, uint64_t o_e_len, const void *o_a, uint64_t o_a_len, const void *o_q, uint64_t o_q_len, const void *o_r, uint64_t o_r_len, uint64_t o_role, void *o_o, uint64_t o_o_len)
{
    (void)result; (void)o_e; (void)o_e_len; (void)o_a; (void)o_a_len; (void)o_q; (void)o_q_len; (void)o_r; (void)o_r_len; (void)o_role; (void)o_o; (void)o_o_len;
    return OMEGA_BAD_RT;
}
uint32_t omega_native_mcblob(uint64_t *result, const void *o_w, uint64_t o_w_len, const void *o_a, uint64_t o_a_len, const void *o_c, uint64_t o_c_len, uint64_t o_n, const void *o_l, uint64_t o_l_len, uint64_t o_role)
{
    (void)result; (void)o_w; (void)o_w_len; (void)o_a; (void)o_a_len; (void)o_c; (void)o_c_len; (void)o_n; (void)o_l; (void)o_l_len; (void)o_role;
    return OMEGA_BAD_RT;
}
uint32_t omega_native_mccover(uint64_t *result, const void *o_c, uint64_t o_c_len, const void *o_roles, uint64_t o_roles_len, const void *o_counts, uint64_t o_counts_len, const void *o_owner, uint64_t o_owner_len)
{
    (void)result; (void)o_c; (void)o_c_len; (void)o_roles; (void)o_roles_len; (void)o_counts; (void)o_counts_len; (void)o_owner; (void)o_owner_len;
    return OMEGA_BAD_RT;
}
uint32_t omega_native_mcintent(uint64_t *result, const void *o_m, uint64_t o_m_len, const void *o_bank, uint64_t o_bank_len, const void *o_qmd, uint64_t o_qmd_len, const void *o_s, uint64_t o_s_len, const void *o_seal, uint64_t o_seal_len)
{
    (void)result; (void)o_m; (void)o_m_len; (void)o_bank; (void)o_bank_len; (void)o_qmd; (void)o_qmd_len; (void)o_s; (void)o_s_len; (void)o_seal; (void)o_seal_len;
    return OMEGA_BAD_RT;
}
uint32_t omega_native_mcseal(uint64_t *result, const void *o_bank, uint64_t o_bank_len, const void *o_qmd, uint64_t o_qmd_len, uint64_t o_n, void *o_out, uint64_t o_out_len)
{
    (void)result; (void)o_bank; (void)o_bank_len; (void)o_qmd; (void)o_qmd_len; (void)o_n; (void)o_out; (void)o_out_len;
    return OMEGA_BAD_RT;
}
uint32_t omega_native_mccode(uint64_t *result, const void *o_code, uint64_t o_code_len, uint64_t o_n, void *o_out, uint64_t o_out_len)
{
    (void)result; (void)o_code; (void)o_code_len; (void)o_n; (void)o_out; (void)o_out_len;
    return OMEGA_BAD_RT;
}
uint32_t omega_native_mclaunch(uint64_t *result, const void *o_meta, uint64_t o_meta_len, const void *o_bank, uint64_t o_bank_len, const void *o_qmdslot, uint64_t o_qmdslot_len, const void *o_submit, uint64_t o_submit_len, void *o_out, uint64_t o_out_len)
{
    (void)result; (void)o_meta; (void)o_meta_len; (void)o_bank; (void)o_bank_len; (void)o_qmdslot; (void)o_qmdslot_len; (void)o_submit; (void)o_submit_len; (void)o_out; (void)o_out_len;
    return OMEGA_BAD_RT;
}
uint32_t omega_native_mcdir(uint64_t *result, const void *o_meta, uint64_t o_meta_len, const void *o_entries, uint64_t o_entries_len, void *o_out, uint64_t o_out_len)
{
    (void)result; (void)o_meta; (void)o_meta_len; (void)o_entries; (void)o_entries_len; (void)o_out; (void)o_out_len;
    return OMEGA_BAD_RT;
}
uint32_t omega_native_mcalloc(uint64_t *result, const void *o_meta, uint64_t o_meta_len, const void *o_records, uint64_t o_records_len, void *o_out, uint64_t o_out_len)
{
    (void)result; (void)o_meta; (void)o_meta_len; (void)o_records; (void)o_records_len; (void)o_out; (void)o_out_len;
    return OMEGA_BAD_RT;
}
uint32_t omega_native_mcseries(uint64_t *result, const void *o_meta, uint64_t o_meta_len, const void *o_hashes, uint64_t o_hashes_len, const void *o_ordinals, uint64_t o_ordinals_len, void *o_out, uint64_t o_out_len)
{
    (void)result; (void)o_meta; (void)o_meta_len; (void)o_hashes; (void)o_hashes_len; (void)o_ordinals; (void)o_ordinals_len; (void)o_out; (void)o_out_len;
    return OMEGA_BAD_RT;
}
