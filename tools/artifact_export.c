/* SPDX-License-Identifier: AGPL-3.0-only */
/* artifact-export: offline handoff for one CUDApp-generated GPU realization.
 *
 *   artifact-export matmul    --m M --k K --n N --contract-id HEX --out PREFIX --commit HEX
 *   artifact-export attention --q-heads HQ --kv-heads HKV --contract-id HEX --out PREFIX --commit HEX
 *
 * Opens no device. Emits a cubin, zero-VA QMD templates, cbank bytes, a
 * canonical manifest and a JSON mirror. Exit: 0 ok, 2 usage, 3 refusal, 4 I/O. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "omega_blackwell_matmul.h"
#include "omega_blackwell_qmd.h"
#include "omega_gpu_attention_api.h"
#include "omega_native_artifact.h"
#include "sha256.h"

static void usage(void) {
    fprintf(stderr,
            "usage: artifact-export matmul --m M --k K --n N --contract-id HEX --out PREFIX --commit HEX\n"
            "       artifact-export attention --q-heads H --kv-heads H --contract-id HEX --out PREFIX --commit HEX\n");
}

static uint32_t num(const char *s) {
    char *e = NULL;
    unsigned long v = strtoul(s, &e, 0);
    return (e && *e == 0 && v <= 0xfffffffful) ? (uint32_t)v : 0;
}

static void hex32(const uint8_t b[32], char out[65]) {
    static const char h[] = "0123456789abcdef";
    for (int i = 0; i < 32; i++) { out[2 * i] = h[b[i] >> 4]; out[2 * i + 1] = h[b[i] & 15]; }
    out[64] = 0;
}

int main(int argc, char **argv) {
    const char *mode, *out = NULL, *contract = NULL, *commit = NULL;
    uint32_t m = 0, k = 0, n = 0, hq = 0, hkv = 0;
    if (argc < 2) { usage(); return 2; }
    mode = argv[1];
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) out = argv[++i];
        else if (strcmp(argv[i], "--contract-id") == 0 && i + 1 < argc) contract = argv[++i];
        else if (strcmp(argv[i], "--commit") == 0 && i + 1 < argc) commit = argv[++i];
        else if (strcmp(argv[i], "--m") == 0 && i + 1 < argc) m = num(argv[++i]);
        else if (strcmp(argv[i], "--k") == 0 && i + 1 < argc) k = num(argv[++i]);
        else if (strcmp(argv[i], "--n") == 0 && i + 1 < argc) n = num(argv[++i]);
        else if (strcmp(argv[i], "--q-heads") == 0 && i + 1 < argc) hq = num(argv[++i]);
        else if (strcmp(argv[i], "--kv-heads") == 0 && i + 1 < argc) hkv = num(argv[++i]);
        else { usage(); return 2; }
    }
    if (!out || !contract || strlen(contract) != 64) { usage(); return 2; }
    if (!commit || strlen(commit) != 40) { fprintf(stderr, "--commit must be a 40-hex source revision\n"); usage(); return 2; }
    for (const char *h = commit; *h; h++) {
        if (!((*h >= '0' && *h <= '9') || (*h >= 'a' && *h <= 'f'))) {
            fprintf(stderr, "--commit must be lowercase 40-hex\n"); return 2;
        }
    }
    /* The frozen v1 ids and bounds. An export under a frozen id but outside the
     * frozen range would let a consumer admit an artifact the contract forbids
     * (REVIEW C-11). */
    if (strcmp(mode, "matmul") == 0) {
        if (strcmp(contract, "647116fc1ba1e1394982fc8ba181ce565293f160a6b56afc0f02b2128a675704") != 0) {
            fprintf(stderr, "contract id is not matmul-bf16-v1\n"); return 3;
        }
        if (m > 8192u || k > 16384u || n > 8192u) {
            fprintf(stderr, "shape outside matmul-bf16-v1 bounds (m<=8192,k<=16384,n<=8192)\n"); return 3;
        }
    } else if (strcmp(mode, "attention") == 0) {
        if (strcmp(contract, "003696e422bba7274877ba9e01a61076ea0f862f32760cbaa06c60f85a115359") != 0) {
            fprintf(stderr, "contract id is not gqa-causal-v1\n"); return 3;
        }
        if (hq > 64u || hkv > 64u) {
            fprintf(stderr, "heads outside gqa-causal-v1 bounds (hq<=64,hkv<=64)\n"); return 3;
        }
    }

    OmegaNativeArtifact a;
    memset(&a, 0, sizeof a);
    OmegaBlackwellKernel kern;
    memset(&kern, 0, sizeof kern);
    uint32_t q0[OMEGA_ARTIFACT_QMD_WORDS], q1[OMEGA_ARTIFACT_QMD_WORDS];
    uint32_t cbank[32];
    OmegaCubinParam params[16];
    size_t nparams = 0;
    char spec_hex[65], *spec_hexp = spec_hex;
    memset(q0, 0, sizeof q0); memset(q1, 0, sizeof q1); memset(cbank, 0, sizeof cbank);
    OmegaBlackwellQmdConfig cfg;
    memset(&cfg, 0, sizeof cfg);

    if (strcmp(mode, "matmul") == 0) {
        OmegaMatMulSpec spec;
        memset(&spec, 0, sizeof spec);
        if (!m || !k || !n) { usage(); return 2; }
        if (omega_matmul_spec_init(&spec, m, k, n, OMEGA_MATMUL_PRECISION_BF16) != 0) { fprintf(stderr, "spec refused\n"); return 3; }
        if (omega_blackwell_codegen_matmul(&spec, &kern) != 0) { fprintf(stderr, "codegen refused\n"); return 3; }
        hex32(spec.spec_id, spec_hex);
        cfg.gpr_count = kern.gpr_count;
        cfg.threads_per_block = 32; cfg.threads_x = 32;
        cfg.grid_x = 1; cfg.grid_y = (m + 15u) / 16u;
        cfg.num_elements = m * n; cfg.shared_bytes = 1024;
        omega_blackwell_build_cbank_args_matmul(cbank, 0, 0, 0, m, k, n);
        params[0] = (OmegaCubinParam){ 0x380, 8, 1 };
        params[1] = (OmegaCubinParam){ 0x388, 8, 1 };
        params[2] = (OmegaCubinParam){ 0x390, 8, 1 };
        nparams = 3;
        a.op = "matmul"; a.contract_name = "matmul-bf16-v1";
        a.precision = "bf16"; a.accumulation = "fp32";
        a.tolerance = "abs<=1e-5*max(1,abs(ref))";
        a.grid_x = 1; a.grid_y = cfg.grid_y; a.block_x = 32; a.block_y = 1; a.block_z = 1;
        a.shared_bytes = 1024; a.cbank_words = 10;
    } else if (strcmp(mode, "attention") == 0) {
        if (!hq || !hkv || hq % hkv != 0) { usage(); return 2; }
        uint32_t ratio = hq / hkv, log2gqa = 0;
        while ((1u << log2gqa) < ratio) log2gqa++;
        if ((1u << log2gqa) != ratio) { fprintf(stderr, "GQA ratio is not a power of two\n"); return 3; }
        int rc = omega_gpu_attention_codegen_hd(true, 12, log2gqa, 128, &kern);
        if (rc != 0) { fprintf(stderr, "attention codegen refused: %d\n", rc); return 3; }
        {
            char specbuf[160];
            uint8_t d[32];
            int n = snprintf(specbuf, sizeof specbuf,
                             "omega-attention-spec/1\nhead_dim=128\nq_heads=%u\nkv_heads=%u\nlog2_gqa=%u\n",
                             hq, hkv, log2gqa);
            sha256_hash((const uint8_t *)specbuf, (size_t)n, d);
            for (int i = 0; i < 32; i++) sprintf(spec_hex + 2 * i, "%02x", d[i]);
            spec_hex[64] = 0;
        }
        (void)spec_hexp;
        uint32_t hpc = omega_gpu_attention_heads_per_cta(128, log2gqa);
        uint32_t shared = (3u * hpc + 1u) * 128u * 4u;
        cfg.gpr_count = kern.gpr_count;
        cfg.threads_per_block = 128; cfg.threads_x = 128;
        cfg.grid_x = hq / hpc; cfg.grid_y = 1;
        cfg.num_elements = hq * 128u; cfg.shared_bytes = shared;
        params[0] = (OmegaCubinParam){ 0x380, 8, 1 };
        params[1] = (OmegaCubinParam){ 0x388, 8, 1 };
        params[2] = (OmegaCubinParam){ 0x390, 8, 1 };
        params[3] = (OmegaCubinParam){ 0x398, 8, 1 };
        for (int i = 0; i < 8; i++) params[4 + i] = (OmegaCubinParam){ (uint32_t)(0x3a0 + 4 * i), 4, 0 };
        nparams = 12;
        a.op = "gqa-causal-attention"; a.contract_name = "gqa-causal-v1";
        a.precision = "f32-kv"; a.accumulation = "fp32";
        a.tolerance = "rel<=1e-3,abs<=1e-4";
        a.grid_x = cfg.grid_x; a.grid_y = 1; a.block_x = 128; a.block_y = 1; a.block_z = 1;
        a.shared_bytes = shared; a.cbank_words = 16;
        /* Offline template: pointers and runtime strides are loader-supplied;
         * the query-head count is a contract constant, so bind it. */
        cbank[14] = hq;
    } else {
        usage(); return 2;
    }
    if (omega_blackwell_build_qmd0(q0, 0, 0) != 0 || omega_blackwell_build_qmd1(q1, &cfg) != 0) {
        fprintf(stderr, "QMD template refused\n"); return 3;
    }
    static char shapebuf[128], boundsbuf[128];
    if (strcmp(mode, "matmul") == 0) {
        snprintf(shapebuf, sizeof shapebuf, "m=%u,k=%u,n=%u", m, k, n);
        snprintf(boundsbuf, sizeof boundsbuf, "m<=8192,k<=16384,n<=8192");
    } else {
        snprintf(shapebuf, sizeof shapebuf, "r>=1,base=0,hq=%u,hkv=%u,hd=128", hq, hkv);
        snprintf(boundsbuf, sizeof boundsbuf, "hq<=64,hkv<=64,hd=128,rows+base<=4096");
    }
    a.shape = shapebuf;
    a.bounds = boundsbuf;
    a.contract_id_hex = contract;
    a.semantic_spec_id_hex = spec_hex;
    a.kernel_function = (strcmp(mode, "matmul") == 0) ? "omega_matmul_bf16" : "omega_gqa_attention_causal";
    a.text = kern.code; a.text_len = kern.code_size;
    a.params = params; a.nparams = nparams;
    a.gpr_count = kern.gpr_count;
    a.cbank_offset = 0x380;
    a.cbank_args = cbank;
    a.qmd0 = q0; a.qmd1 = q1;
    a.min_driver = "580.173.02";
    a.requires = "sm_121,nvidia-cubin-abi-v8";
    a.source_commit = commit;
    a.tool_version = "cudapp-artifact-export/1";
    a.qualification = "offline-disassembly=PASS;device-load=NOT_RUN;device-exec=NOT_RUN";

    char id[OMEGA_ARTIFACT_ID_MAX], err[256];
    int rc = omega_native_artifact_export(&a, out, id, err, sizeof err);
    if (rc != OMEGA_ARTIFACT_OK) {
        fprintf(stderr, "artifact-export refused: %s: %s\n", omega_native_artifact_rc_name(rc), err);
        omega_blackwell_kernel_free(&kern);
        return 3;
    }
    printf("artifact_id=%s\nop=%s\ncontract_id=%s\ncubin=%s.cubin\n", id, a.op, contract, out);
    omega_blackwell_kernel_free(&kern);
    return 0;
}
