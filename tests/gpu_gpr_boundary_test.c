/* gpu_gpr_boundary_test: host-only audit of the declared-register rule (OWN-30B, 2026-10-13).
 * Rule (omega_blackwell_codegen.h): declared = peak + 2 rounded up to a multiple of 8. A fp8 grouped GEMM kernel with
 * peak 103 declared 104 (the old peak + 1 rule) hung the chip; the same program declared 112 ran.
 * Checks: the helper for every peak; every launcher's own `declared` for every live kernel spec; the matmul family
 * (declared by the session from the raw peak) is listed and must not be reachable at a boundary from a Qwen3-30B-A3B shape. */
#include "omega_blackwell_codegen.h"
#include "omega_blackwell_codegen_moe.h"
#include "omega_blackwell_codegen_bf16.h"
#include "omega_blackwell_codegen_router.h"
#include "omega_blackwell_codegen_attn.h"
#include "omega_blackwell_codegen_gemm.h"
#include "omega_gpu_fp8_api.h"
#include "omega_gpu_bf16_gemv_api.h"
#include "omega_gpu_router_api.h"
#include "omega_gpu_decode_attn_api.h"
#include "omega_gpu_fp8gemm_api.h"
#include <stdio.h>
#include <string.h>

static int g_checks, g_failed;
#define CHECK(c, ...) do { g_checks++; if (!(c)) { g_failed++; printf("FAIL: "); printf(__VA_ARGS__); printf("\n"); } } while (0)
static BlackwellIRProgram g_prog;

int main(void) {
    for (uint32_t peak = 0; peak <= 126; peak++) {
        uint32_t d = omega_bw_gpr_declared(peak);
        CHECK(d >= peak + 2 && d % 8 == 0 && d - (peak + 2) < 8 && omega_bw_gpr_declared_ok(d, peak), "helper at peak %u gives %u", peak, d);
        CHECK(!omega_bw_gpr_declared_ok(((peak + 1u + 7u) / 8u) * 8u, peak) || peak % 8 != 7, "old rule at peak %u must be refused", peak);
    }
    CHECK(!omega_bw_gpr_declared_ok(104, 103) && omega_bw_gpr_declared_ok(112, 103), "the 103 / 104 hang case is refused, 112 accepted");
    CHECK(!omega_bw_gpr_declared_ok(64, 64) && !omega_bw_gpr_declared_ok(72, 72) && omega_bw_gpr_declared_ok(72, 64), "peak 64 and 72 need 72 and 80 / 72");

    /* fp8 GEMV: live shapes through the API's own shape / depth rules */
    struct { unsigned n, k, groups; } fs[] = { {4096,2048,1},{512,2048,1},{5120,2048,1},{2048,4096,1},{1536,2048,8},{2048,768,8},{1536,2048,1},{2048,768,1},{1u<<20,2048,1},{1u<<20,4096,1},{1u<<20,768,1} };
    for (unsigned i = 0; i < sizeof fs / sizeof *fs; i++) {
        OmegaFp8GemvSpec s; memset(&s, 0, sizeof s); s.k = fs[i].k;
        omega_gpu_fp8_shape_for(s.k, fs[i].n, &s.lane_bytes, &s.rows, &s.warps);
        if (fs[i].groups > 1) s.warps = 32;
        s.depth = omega_gpu_fp8_depth_for(&s, fs[i].n, fs[i].groups);
        uint32_t peak = 0, decl = 0;
        CHECK(omega_fp8_gemv_prog_stats(&s, NULL, &peak) == 0, "fp8 gemv %ux%u x%u builds", fs[i].n, fs[i].k, fs[i].groups);
        (void)omega_gpu_fp8_kernel_fits(&s, &decl);
        CHECK(omega_bw_gpr_declared_ok(decl, peak), "fp8 gemv %ux%u x%u: peak %u declared %u", fs[i].n, fs[i].k, fs[i].groups, peak, decl);
    }
    /* every buildable fp8 GEMV variant */
    { unsigned bad = 0, n = 0;
      for (unsigned k = 256; k <= 8192; k += 256) for (unsigned lb = 8; lb <= 16; lb += 8) for (unsigned rows = 1; rows <= 8; rows *= 2) for (unsigned w = 1; w <= 32; w *= 2) for (unsigned dp = 0; dp <= 8; dp += dp ? dp : 1) {
          OmegaFp8GemvSpec s = { .k = k, .lane_bytes = lb, .rows = rows, .warps = w, .depth = dp, .mutant = 0 }; uint32_t peak = 0, decl = 0;
          if (omega_fp8_gemv_prog_stats(&s, NULL, &peak) != 0) continue;
          n++; (void)omega_gpu_fp8_kernel_fits(&s, &decl);
          if (!omega_bw_gpr_declared_ok(decl, peak) && decl != 0) bad++;
      }
      CHECK(bad == 0, "fp8 gemv sweep: %u of %u variants declared below peak + 2", bad, n); printf("fp8 gemv sweep: %u variants, %u below peak + 2\n", n, bad); }
    /* bf16 GEMV */
    { unsigned ks[] = { 2048, 4096 }; for (unsigned i = 0; i < 2; i++) { OmegaBf16GemvSpec s; memset(&s, 0, sizeof s); s.k = ks[i]; uint32_t c; omega_gpu_bf16_shape_for(s.k, &s.rows, &s.warps, &s.depth, &c);
        uint32_t peak = 0, decl = 0; CHECK(omega_bf16_gemv_prog_stats(&s, NULL, &peak) == 0, "bf16 k%u builds", ks[i]); (void)omega_gpu_bf16_kernel_fits(&s, &decl);
        CHECK(omega_bw_gpr_declared_ok(decl, peak), "bf16 gemv k%u: peak %u declared %u", ks[i], peak, decl); } }
    /* fp8 grouped GEMM (all shapes we launch) */
    { unsigned ks[] = { 2048, 768, 4096 }, mts[] = { 4, 8 }; for (unsigned i = 0; i < 3; i++) for (unsigned j = 0; j < 2; j++) {
        OmegaFp8GemmSpec s = { ks[i], ks[i] % 512 ? 8u : 16u, mts[j], 16, 0 }; uint32_t peak = 0, decl = 0;
        CHECK(omega_fp8gemm_prog_stats(&s, NULL, &peak) == 0, "gemm k%u mt%u builds", ks[i], mts[j]); (void)omega_gpu_fp8gemm_kernel_fits(&s, &decl);
        CHECK(omega_bw_gpr_declared_ok(decl, peak), "gemm k%u mt%u: peak %u declared %u", ks[i], mts[j], peak, decl); } }
    /* router and decode attention: the launcher uses omega_bw_gpr_declared(peak) (static), so check the helper on the real peak */
    { OmegaRouterSpec s = { OMEGA_GPU_ROUTER_WARPS, OMEGA_GPU_ROUTER_DEPTH, 0 }; uint32_t peak = 0;
      CHECK(omega_router_prog_stats(&s, NULL, &peak) == 0 && omega_bw_gpr_declared_ok(omega_bw_gpr_declared(peak), peak), "router peak %u", peak); printf("router peak %u -> declared %u\n", peak, omega_bw_gpr_declared(peak)); }
    { OmegaDecodeAttnSpec s = { OMEGA_GPU_DATTN_DEPTH, 0 }; uint32_t peak = 0;
      CHECK(omega_decode_attn_prog_stats(&s, NULL, &peak) == 0 && omega_bw_gpr_declared_ok(omega_bw_gpr_declared(peak), peak), "decode attn peak %u", peak); printf("decode attn peak %u -> declared %u\n", peak, omega_bw_gpr_declared(peak)); }
    /* matmul / moe-window fragment kernels: declared by the session from the raw peak (max(64, peak rounded up to 8)).
     * Boundary combinations are reported; the shapes of Qwen3-30B-A3B (n a multiple of 32) never select them. */
    { unsigned kps[] = { 768, 2048, 4096 }, nps[] = { 8, 128, 1536, 2048, 4096, 5120, 151936 }; unsigned built = 0, boundary = 0, model_boundary = 0;
      for (unsigned a = 0; a < 3; a++) for (unsigned b = 0; b < 7; b++) for (unsigned rt = 1; rt <= 4; rt *= 2) for (unsigned ct = 1; ct <= 4; ct *= 2) {
          OmegaMatMulSpec s; if (omega_matmul_spec_init(&s, 16, kps[a], nps[b], OMEGA_MATMUL_PRECISION_BF16) != 0) continue;
          unsigned un = (rt > 1 || ct > 1) ? omega_matmul_frag_block_unroll(kps[a], rt, ct) : omega_matmul_frag_unroll(kps[a]);
          memset(&g_prog, 0, sizeof g_prog);
          int rc = (rt > 1 || ct > 1) ? omega_blackwell_codegen_matmul_frag_block_prog(&s, 48, un, rt, ct, 0, &g_prog) : omega_blackwell_codegen_matmul_frag_prog(&s, 48, un, 0, &g_prog);
          if (rc) continue;
          built++;
          uint32_t pk = g_prog.regalloc.peak_gpr_usage, decl = pk > 64 ? ((pk + 7) / 8) * 8 : 64;
          if (decl < pk + 2) { boundary++; unsigned mt = 4, nt = nps[b] / 8; uint32_t r, c; omega_matmul_frag_block_shape(mt, nt, &r, &c); if (r == rt && c == ct) model_boundary++; }
      }
      printf("matmul fragment kernels: %u built, %u at a boundary (session declares the raw peak), %u of those selected for a model shape\n", built, boundary, model_boundary);
      CHECK(model_boundary == 0, "a Qwen3-30B shape reaches a matmul kernel declared below peak + 2"); }
    printf("gpu_gpr_boundary_test: %d checks, %d failed: %s\n", g_checks, g_failed, g_failed ? "FAIL" : "PASS");
    return g_failed ? 1 : 0;
}
