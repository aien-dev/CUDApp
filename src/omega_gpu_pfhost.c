/* omega_gpu_pfhost.c: launchers for the whole-prompt router logits and MoE combine (W-L1G-D). See omega_gpu_pfhost.h. */
#include "omega_gpu_pfhost.h"
#include "omega_blackwell_codegen_pfhost.h"
#include <string.h>

#define KIND_RL 0x50u
#define KIND_MC 0x51u
static int rl_mutant, mc_mutant;
static uint32_t rl_rows;
void omega_gpu_pf_router_test_set_mutant(int m) { rl_mutant = m; }
void omega_gpu_pf_combine_test_set_mutant(int m) { mc_mutant = m; }
void omega_gpu_pf_router_set_rows(uint32_t t) { rl_rows = t; }
uint32_t omega_gpu_pf_router_rows(void) { return rl_rows ? rl_rows : 16u; }

size_t omega_gpu_pf_router_packed_bytes(uint32_t e, uint32_t k) { return (size_t)(k / 4u + 1u) * e * 16u; }
void omega_gpu_pf_router_pack(uint32_t e, uint32_t k, const float *w, float *wq) {
    if (!w || !wq) return;
    for (uint32_t g = 0; g < k / 4u; g++)
        for (uint32_t ex = 0; ex < e; ex++)
            for (uint32_t q = 0; q < 4u; q++) wq[((size_t)g * e + ex) * 4u + q] = w[(size_t)ex * k + 4u * g + q];
    memset(wq + (size_t)(k / 4u) * e * 4u, 0, (size_t)e * 16u);
}

static int gen_rl(const OmegaPfKey *key, void *prog) { OmegaRlSpec s; memcpy(&s, key->words, sizeof s); return omega_blackwell_codegen_rl_prog(&s, prog); }
static int gen_mc(const OmegaPfKey *key, void *prog) { OmegaMcSpec s; memcpy(&s, key->words, sizeof s); return omega_blackwell_codegen_mc_prog(&s, prog); }

int omega_gpu_pf_router_logits(uint32_t k, uint32_t e, uint32_t rows, uint64_t x_va, uint64_t wq_va, uint64_t logits_va, uint64_t *ns) {
    OmegaRlSpec s; memset(&s, 0, sizeof s); s.k = k; s.e = e; s.t = omega_gpu_pf_router_rows(); s.mutant = rl_mutant;
    if (!omega_rl_spec_ok(&s) || !x_va || !wq_va || !logits_va || (x_va & 15u) || (wq_va & 15u) || (logits_va & 3u)) return OMEGA_GPU_PF_BAD_ARGS;
    if (rows == 0) return OMEGA_GPU_PF_OK;
    _Static_assert(sizeof(OmegaRlSpec) <= sizeof(((OmegaPfKey *)0)->words), "spec fits the key");
    OmegaPfKey key; memset(&key, 0, sizeof key); key.kind = KIND_RL; memcpy(key.words, &s, sizeof s);
    const uint32_t args[OMEGA_RL_ARGS_WORDS] = { (uint32_t)x_va, (uint32_t)(x_va >> 32), (uint32_t)wq_va, (uint32_t)(wq_va >> 32), (uint32_t)logits_va, (uint32_t)(logits_va >> 32) };
    omega_gpu_session_lock();
    int rc = omega_gpu_pf_open_locked();
    OmegaPfKernel kern;
    if (rc == OMEGA_GPU_PF_OK) rc = omega_gpu_pf_kernel_locked(&key, gen_rl, &kern);
    if (rc == OMEGA_GPU_PF_OK) rc = omega_gpu_pf_launch_locked(&kern, e, (rows + s.t - 1u) / s.t, 1, omega_rl_shared_bytes(&s), args, OMEGA_RL_ARGS_WORDS, ns);
    omega_gpu_session_unlock();
    return rc;
}

int omega_gpu_pf_moe_combine(uint32_t h, uint32_t top_k, uint32_t rows, uint64_t down_va, uint64_t inv_va, uint64_t wt_va, uint64_t out_va, uint64_t *ns) {
    OmegaMcSpec s; memset(&s, 0, sizeof s); s.h = h; s.top_k = top_k; s.mutant = mc_mutant;
    if (!omega_mc_spec_ok(&s) || !down_va || !inv_va || !wt_va || !out_va) return OMEGA_GPU_PF_BAD_ARGS;
    if ((down_va | inv_va | wt_va | out_va) & 15u) return OMEGA_GPU_PF_BAD_ARGS;
    if (rows == 0) return OMEGA_GPU_PF_OK;
    _Static_assert(sizeof(OmegaMcSpec) <= sizeof(((OmegaPfKey *)0)->words), "spec fits the key");
    OmegaPfKey key; memset(&key, 0, sizeof key); key.kind = KIND_MC; memcpy(key.words, &s, sizeof s);
    const uint32_t args[OMEGA_MC_ARGS_WORDS] = { (uint32_t)down_va, (uint32_t)(down_va >> 32), (uint32_t)inv_va, (uint32_t)(inv_va >> 32),
                                                 (uint32_t)wt_va, (uint32_t)(wt_va >> 32), (uint32_t)out_va, (uint32_t)(out_va >> 32) };
    omega_gpu_session_lock();
    int rc = omega_gpu_pf_open_locked();
    OmegaPfKernel kern;
    if (rc == OMEGA_GPU_PF_OK) rc = omega_gpu_pf_kernel_locked(&key, gen_mc, &kern);
    if (rc == OMEGA_GPU_PF_OK) rc = omega_gpu_pf_launch_locked(&kern, h / 4u, rows, 1, 0, args, OMEGA_MC_ARGS_WORDS, ns);
    omega_gpu_session_unlock();
    return rc;
}
