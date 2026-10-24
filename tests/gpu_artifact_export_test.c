/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */
/* Host-only test for the native artifact exporter and admission.
 *
 * It proves: byte-deterministic export, identity/digest recomputation, the
 * cubin container's ELF and text bindings, refusal of malformed descriptors,
 * and rejection of tampered binaries, QMDs, manifests, architectures and
 * drivers. It never opens a device: device load/execution remains NOT_RUN and
 * is reported by the artifact's qualification field. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "omega_blackwell_matmul.h"
#include "omega_blackwell_qmd.h"
#include "omega_cubin.h"
#include "omega_gpu_attention_api.h"
#include "omega_native_artifact.h"
#include "sha256.h"

static int checks, fails;
#define CHECK(cond, ...) do { checks++; if (!(cond)) { fails++; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

static uint8_t *readf(const char *p, size_t *n) {
    FILE *f = fopen(p, "rb");
    long sz;
    uint8_t *b;
    if (!f) return NULL;
    fseek(f, 0, SEEK_END); sz = ftell(f); fseek(f, 0, SEEK_SET);
    b = malloc((size_t)sz + 1); if (!b) { fclose(f); return NULL; }
    if (sz && fread(b, 1, (size_t)sz, f) != (size_t)sz) { free(b); fclose(f); return NULL; }
    b[sz] = 0; *n = (size_t)sz; fclose(f); return b;
}
static int writef(const char *p, const void *d, size_t n) {
    FILE *f = fopen(p, "wb"); if (!f) return -1;
    if (n && fwrite(d, 1, n, f) != n) { fclose(f); return -1; }
    return fclose(f);
}
static int same_file(const char *a, const char *b) {
    size_t na, nb; uint8_t *pa = readf(a, &na), *pb = readf(b, &nb);
    int eq = pa && pb && na == nb && memcmp(pa, pb, na) == 0;
    free(pa); free(pb); return eq;
}

static int copy_artifact(const char *src, const char *dst) {
    const char *suf[] = { ".cubin", ".text.bin", ".qmd0.bin", ".qmd1.bin", ".cbank_args.bin" };
    for (size_t i = 0; i < sizeof suf / sizeof suf[0]; i++) {
        char a[600], b[600]; size_t n;
        snprintf(a, sizeof a, "%s%s", src, suf[i]);
        snprintf(b, sizeof b, "%s%s", dst, suf[i]);
        uint8_t *d = readf(a, &n);
        if (!d || writef(b, d, n)) { free(d); return -1; }
        free(d);
    }
    return 0;
}

static void hex32(const uint8_t d[32], char out[65]) {
    static const char h[] = "0123456789abcdef";
    for (int i = 0; i < 32; i++) { out[2*i] = h[d[i]>>4]; out[2*i+1] = h[d[i]&15]; }
    out[64] = 0;
}

/* --- minimal ELF reader (little-endian, 64-bit) --- */
static uint16_t g16(const uint8_t *p) { return (uint16_t)(p[0] | (p[1] << 8)); }
static uint32_t g32(const uint8_t *p) { return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24); }
static uint64_t g64(const uint8_t *p) { return (uint64_t)g32(p) | ((uint64_t)g32(p + 4) << 32); }

static int elf_section(const uint8_t *e, size_t n, const char *want, const uint8_t **data, size_t *len) {
    uint64_t shoff; uint16_t shentsize, shnum, shstrndx;
    const uint8_t *sh, *st; size_t stlen;
    if (n < 64 || memcmp(e, "\177ELF", 4) != 0) return -1;
    shoff = g64(e + 40); shentsize = g16(e + 58); shnum = g16(e + 60); shstrndx = g16(e + 62);
    if (shentsize != 64 || shstrndx >= shnum || shoff + (uint64_t)shentsize * shnum > n) return -1;
    sh = e + shoff;
    st = e + g64(sh + (uint64_t)shstrndx * 64 + 24);
    stlen = (size_t)g64(sh + (uint64_t)shstrndx * 64 + 32);
    for (uint16_t i = 0; i < shnum; i++) {
        const uint8_t *s = sh + (uint64_t)i * 64;
        uint32_t no = g32(s);
        if (no < stlen && strcmp((const char *)st + no, want) == 0) {
            uint64_t off = g64(s + 24), sz = g64(s + 32);
            if (off + sz > n) return -1;
            *data = e + off; *len = (size_t)sz; return 0;
        }
    }
    return -1;
}

/* Rewrite one manifest value and recompute artifact_id over the identity block. */
static int rewrite_manifest(const char *src, const char *dst, const char *key, const char *val) {
    size_t n; uint8_t *m = readf(src, &n); char *out; size_t cap = n + strlen(val) + 128, o = 0;
    char *p, *idline, *ident, *prov;
    if (!m) return -1;
    out = malloc(cap); if (!out) { free(m); return -1; }
    idline = strstr((char *)m, "artifact_id=");
    ident = strstr((char *)m, "\nop="); if (ident) ident++;
    prov = strstr((char *)m, "provenance_tool=");
    if (!idline || !ident || !prov || prov <= ident) { free(m); free(out); return -1; }
    {
        size_t klen = strlen(key), vlen = strlen(val);
        p = (char *)m;
        while (*p) {
            char *nl = strchr(p, '\n'); size_t len = nl ? (size_t)(nl - p) : strlen(p);
            if ((size_t)len > klen && strncmp(p, key, klen) == 0 && p[klen] == '=') {
                memcpy(out + o, key, klen); o += klen; out[o++] = '=';
                memcpy(out + o, val, vlen); o += vlen; out[o++] = '\n';
            } else {
                memcpy(out + o, p, len); o += len; out[o++] = '\n';
            }
            if (!nl) break;
            p = nl + 1;
        }
        (void)vlen;
    }
    {
        uint8_t d[32]; char id[65];
        char *ni = strstr(out, "\nop="); if (ni) ni++;
        char *np = strstr(out, "provenance_tool=");
        size_t ilen = np && ni ? (size_t)(np - ni) : 0;
        char *idp = strstr(out, "artifact_id=");
        sha256_hash((const uint8_t *)ni, ilen, d); hex32(d, id);
        if (idp) memcpy(idp + strlen("artifact_id="), id, 64);
        (void)idline;
    }
    {
        int rc = writef(dst, out, o);
        free(m); free(out); return rc;
    }
}

/* Edit one line and deliberately leave artifact_id untouched: this is the
 * naive NOT_RUN -> PASS tamper the artifact id must catch. */
static int rewrite_manifest_raw(const char *src, const char *dst, const char *key, const char *val) {
    size_t n; uint8_t *m = readf(src, &n); char *out; size_t cap = n + strlen(val) + 128, o = 0;
    char *p;
    if (!m) return -1;
    out = malloc(cap); if (!out) { free(m); return -1; }
    size_t klen = strlen(key), vlen = strlen(val);
    p = (char *)m;
    while (*p) {
        char *nl = strchr(p, '\n'); size_t len = nl ? (size_t)(nl - p) : strlen(p);
        if ((size_t)len > klen && strncmp(p, key, klen) == 0 && p[klen] == '=') {
            memcpy(out + o, key, klen); o += klen; out[o++] = '=';
            memcpy(out + o, val, vlen); o += vlen; out[o++] = '\n';
        } else {
            memcpy(out + o, p, len); o += len; out[o++] = '\n';
        }
        if (!nl) break;
        p = nl + 1;
    }
    int rc = writef(dst, out, o);
    free(m); free(out); return rc;
}

static void build_matmul(OmegaBlackwellKernel *k, uint32_t q0[96], uint32_t q1[96], uint32_t cb[10],
                         OmegaCubinParam ps[3], OmegaBlackwellQmdConfig *cfg, char spec_hex[65]) {
    OmegaMatMulSpec spec;
    memset(&spec, 0, sizeof spec);
    memset(k, 0, sizeof *k);
    memset(q0, 0, 96 * 4); memset(q1, 0, 96 * 4); memset(cb, 0, 10 * 4);
    omega_matmul_spec_init(&spec, 16, 16, 8, OMEGA_MATMUL_PRECISION_BF16);
    omega_blackwell_codegen_matmul(&spec, k);
    hex32(spec.spec_id, spec_hex);
    omega_blackwell_build_qmd0(q0, 0, 0);
    memset(cfg, 0, sizeof *cfg);
    cfg->gpr_count = k->gpr_count; cfg->threads_per_block = 32; cfg->threads_x = 32;
    cfg->grid_x = 1; cfg->grid_y = 1; cfg->num_elements = 16 * 8; cfg->shared_bytes = 1024;
    omega_blackwell_build_qmd1(q1, cfg);
    omega_blackwell_build_cbank_args_matmul(cb, 0, 0, 0, 16, 16, 8);
    ps[0] = (OmegaCubinParam){ 0x380, 8, 1 };
    ps[1] = (OmegaCubinParam){ 0x388, 8, 1 };
    ps[2] = (OmegaCubinParam){ 0x390, 8, 1 };
}

static void fill_artifact(OmegaNativeArtifact *a, const char *contract, const char *spec_hex,
                          OmegaBlackwellKernel *k, OmegaCubinParam *ps, size_t nps,
                          uint32_t *cb, size_t cbn, uint32_t *q0, uint32_t *q1,
                          const char *op, const char *name, const char *fn,
                          uint32_t gx, uint32_t gy, uint32_t bx, uint32_t shared) {
    memset(a, 0, sizeof *a);
    a->op = op; a->shape = "m=16,k=16,n=8"; a->bounds = "m<=8192,k<=16384,n<=8192";
    a->contract_name = name; a->contract_id_hex = contract;
    a->semantic_spec_id_hex = spec_hex;
    a->kernel_function = fn; a->precision = "bf16"; a->accumulation = "fp32";
    a->tolerance = "abs<=1e-5*max(1,abs(ref))";
    a->text = k->code; a->text_len = k->code_size;
    a->params = ps; a->nparams = nps;
    a->gpr_count = k->gpr_count;
    a->grid_x = gx; a->grid_y = gy; a->block_x = bx; a->block_y = 1; a->block_z = 1;
    a->shared_bytes = shared; a->cbank_offset = 0x380;
    a->cbank_args = cb; a->cbank_words = cbn;
    a->qmd0 = q0; a->qmd1 = q1;
    a->min_driver = "580.173.02"; a->requires = "sm_121,nvidia-cubin-abi-v8";
    a->source_commit = "UNKNOWN"; a->tool_version = "cudapp-artifact-export/1";
    a->qualification = "offline-disassembly=PASS;device-load=NOT_RUN;device-exec=NOT_RUN";
}

int main(int argc, char **argv) {
    const char *dir = argc > 1 ? argv[1] : "/tmp/artifact-test";
    char p1[512], p2[512], path[600], path2[600], id[65], err[256];
    char contract[65];
    memset(contract, 'a', 64); contract[64] = 0;
    mkdir(dir, 0700);
    snprintf(p1, sizeof p1, "%s/m1", dir);
    snprintf(p2, sizeof p2, "%s/m2", dir);

    OmegaBlackwellKernel k; OmegaCubinParam ps[3]; OmegaBlackwellQmdConfig cfg;
    uint32_t q0[96], q1[96], cb[10]; char spec_hex[65];
    build_matmul(&k, q0, q1, cb, ps, &cfg, spec_hex);
    CHECK(k.code_size > 0 && k.gpr_count > 0, "matmul codegen failed");

    OmegaNativeArtifact a;
    fill_artifact(&a, contract, spec_hex, &k, ps, 3, cb, 10, q0, q1,
                  "matmul", "matmul-bf16-v1", "omega_matmul_bf16", 1, 1, 32, 1024);
    int rc = omega_native_artifact_export(&a, p1, id, err, sizeof err);
    CHECK(rc == OMEGA_ARTIFACT_OK, "export matmul: %s %s", omega_native_artifact_rc_name(rc), err);
    if (rc != OMEGA_ARTIFACT_OK) { printf("%d checks, %d failed\n", checks, fails); return 1; }
    {
        char rid[65]; int r2 = omega_native_artifact_admit(p1, "580.173.02", rid, err, sizeof err);
        CHECK(r2 == OMEGA_ARTIFACT_OK && strcmp(rid, id) == 0, "admit fresh: %s %s", omega_native_artifact_rc_name(r2), err);
    }

    /* Determinism. */
    {
        int r2 = omega_native_artifact_export(&a, p2, id, err, sizeof err);
        CHECK(r2 == OMEGA_ARTIFACT_OK, "second export: %s", err);
        const char *suf[] = { ".cubin", ".text.bin", ".qmd0.bin", ".qmd1.bin", ".cbank_args.bin", ".manifest", ".json", ".sha256" };
        for (size_t i = 0; i < sizeof suf / sizeof suf[0]; i++) {
            snprintf(path, sizeof path, "%s%s", p1, suf[i]);
            snprintf(path2, sizeof path2, "%s%s", p2, suf[i]);
            CHECK(same_file(path, path2), "non-deterministic %s", suf[i]);
        }
    }

    /* Cubin container bindings. */
    {
        size_t n; uint8_t *e = readf((snprintf(path, sizeof path, "%s.cubin", p1), path), &n);
        const uint8_t *sec; size_t slen;
        CHECK(e && n > 64, "cubin readable");
        if (e) {
            CHECK(g16(e + 18) == 190, "e_machine %u != EM_CUDA", g16(e + 18));
            CHECK(g32(e + 48) == 0x6007902u, "wrong e_flags 0x%x", g32(e + 48));
            CHECK(elf_section(e, n, ".text.omega_matmul_bf16", &sec, &slen) == 0 &&
                  slen == k.code_size && memcmp(sec, k.code, slen) == 0, "cubin text does not bind the generated bytes");
            CHECK(elf_section(e, n, ".nv.info.omega_matmul_bf16", &sec, &slen) == 0, "missing .nv.info function section");
            free(e);
        }
    }

    /* Tampered files. */
    {
        uint8_t *b; size_t n;
        b = readf((snprintf(path, sizeof path, "%s.cubin", p1), path), &n);
        b[n / 2] ^= 0x5a; writef(path, b, n);
        snprintf(path2, sizeof path2, "%s.cubin", p2);
        CHECK(omega_native_artifact_admit(p1, "580.173.02", id, err, sizeof err) == OMEGA_ARTIFACT_E_DIGEST, "tampered cubin accepted");
        b = readf(path2, &n); writef(path, b, n); free(b);
    }
    {
        uint8_t *b; size_t n;
        b = readf((snprintf(path, sizeof path, "%s.text.bin", p1), path), &n); b[0] ^= 0x01; writef(path, b, n);
        CHECK(omega_native_artifact_admit(p1, "580.173.02", id, err, sizeof err) == OMEGA_ARTIFACT_E_DIGEST, "tampered text accepted");
        snprintf(path2, sizeof path2, "%s.text.bin", p2); b = readf(path2, &n); writef(path, b, n); free(b);
    }
    {
        uint8_t *b; size_t n;
        b = readf((snprintf(path, sizeof path, "%s.qmd1.bin", p1), path), &n); b[32] ^= 0x01; writef(path, b, n);
        CHECK(omega_native_artifact_admit(p1, "580.173.02", id, err, sizeof err) == OMEGA_ARTIFACT_E_DIGEST, "tampered qmd accepted");
        snprintf(path2, sizeof path2, "%s.qmd1.bin", p2); b = readf(path2, &n); writef(path, b, n); free(b);
    }

    /* Structured manifest refusals with a recomputed id (digests still match). */
    {
        char m0[600], bad[600], badman[620];
        snprintf(m0, sizeof m0, "%s.manifest", p1);
        snprintf(bad, sizeof bad, "%s/bad", dir);
        snprintf(badman, sizeof badman, "%s.manifest", bad);
        CHECK(copy_artifact(p1, bad) == 0, "copy artifact for structured tests");
        CHECK(rewrite_manifest(m0, badman, "isa", "sm_120") == 0, "rewrite isa");
        CHECK(omega_native_artifact_admit(bad, "580.173.02", id, err, sizeof err) == OMEGA_ARTIFACT_E_ISA, "wrong ISA accepted");
        CHECK(rewrite_manifest(m0, badman, "qmd_template", "live-va-v1") == 0, "rewrite qmd_template");
        CHECK(omega_native_artifact_admit(bad, "580.173.02", id, err, sizeof err) == OMEGA_ARTIFACT_E_RELOC, "live-VA QMD accepted");
        CHECK(rewrite_manifest(m0, badman, "kernel_format", "ptx") == 0, "rewrite kernel_format");
        CHECK(omega_native_artifact_admit(bad, "580.173.02", id, err, sizeof err) == OMEGA_ARTIFACT_E_FORMAT, "PTX kernel accepted");
        CHECK(rewrite_manifest(m0, badman, "min_driver", "999.0.0") == 0, "rewrite min_driver");
        CHECK(omega_native_artifact_admit(bad, "580.173.02", id, err, sizeof err) == OMEGA_ARTIFACT_E_DRIVER, "too-new driver requirement accepted");
        CHECK(rewrite_manifest(m0, badman, "contract_id", "zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz") == 0, "rewrite contract");
        CHECK(omega_native_artifact_admit(bad, "580.173.02", id, err, sizeof err) == OMEGA_ARTIFACT_E_CONTRACT, "malformed contract id accepted");
        CHECK(rewrite_manifest_raw(m0, badman, "qualification",
              "offline-disassembly=PASS;device-load=PASS;device-exec=PASS") == 0, "raw qualification rewrite");
        CHECK(omega_native_artifact_admit(bad, "580.173.02", id, err, sizeof err) == OMEGA_ARTIFACT_E_DIGEST,
              "unhashed NOT_RUN -> PASS tamper accepted");
    }

    /* Export refusals. */
    {
        char c2[65]; memcpy(c2, contract, 65); c2[63] = 0;
        a.contract_id_hex = c2;
        CHECK(omega_native_artifact_export(&a, p2, id, err, sizeof err) == OMEGA_ARTIFACT_E_CONTRACT, "short contract id accepted");
        a.contract_id_hex = contract;
        a.op = "gemm";
        CHECK(omega_native_artifact_export(&a, p2, id, err, sizeof err) == OMEGA_ARTIFACT_E_UNSUPPORTED, "unknown op accepted");
        a.op = "matmul";
        q1[32] = 0x1234; q1[33] = 0;
        CHECK(omega_native_artifact_export(&a, p2, id, err, sizeof err) == OMEGA_ARTIFACT_E_RELOC, "live code VA accepted");
        q1[32] = 0; q1[33] = 0xa << 21;
        q1[56] = 0x12345678;
        CHECK(omega_native_artifact_export(&a, p2, id, err, sizeof err) == OMEGA_ARTIFACT_E_RELOC, "live scratch VA accepted");
        q1[56] = 0; q1[57] |= 1u;
        CHECK(omega_native_artifact_export(&a, p2, id, err, sizeof err) == OMEGA_ARTIFACT_E_RELOC, "live scratch high VA accepted");
        q1[57] &= ~0x7ffffu; q1[59] = 0xdeadbeefu;
        CHECK(omega_native_artifact_export(&a, p2, id, err, sizeof err) == OMEGA_ARTIFACT_E_RELOC, "live prefetch VA accepted");
        q1[59] = 0;
    }
    {
        uint8_t *out = NULL; size_t outn = 0;
        OmegaCubinParam badp[2] = { { 0x380, 8, 1 }, { 0x384, 8, 1 } };
        CHECK(omega_cubin_pack("k", k.code, k.code_size, badp, 2, 0x398, k.gpr_count, 64, &out, &outn, err, sizeof err) == OMEGA_CUBIN_E_PARAMS, "overlapping params accepted");
        CHECK(omega_cubin_pack("k", k.code, k.code_size - 1, ps, 3, 0x398, k.gpr_count, 64, &out, &outn, err, sizeof err) == OMEGA_CUBIN_E_TEXT, "unaligned text accepted");
        CHECK(omega_cubin_pack("k", k.code, k.code_size, ps, 3, 0x398, 300, 64, &out, &outn, err, sizeof err) == OMEGA_CUBIN_E_GPR, "gpr 300 accepted");
        CHECK(omega_cubin_pack("k", k.code, k.code_size, ps, 3, 0x390, k.gpr_count, 64, &out, &outn, err, sizeof err) == OMEGA_CUBIN_E_CBANK, "too-small cbank accepted");
        CHECK(omega_cubin_pack("1bad", k.code, k.code_size, ps, 3, 0x398, k.gpr_count, 64, &out, &outn, err, sizeof err) == OMEGA_CUBIN_E_ARG, "bad function name accepted");
    }

    /* Attention artifact: the exact kernel the rows API builds (offline codegen). */
    {
        OmegaBlackwellKernel ak; memset(&ak, 0, sizeof ak);
        int arc = omega_gpu_attention_codegen_hd(true, 12, 3, 128, &ak);
        CHECK(arc == 0 && ak.code_size > 0, "attention codegen failed: %d", arc);
        if (arc == 0) {
            uint32_t aq0[96], aq1[96], acb[16];
            OmegaCubinParam aps[12];
            OmegaBlackwellQmdConfig acfg;
            char aspec[65];
            memset(aq0, 0, sizeof aq0); memset(aq1, 0, sizeof aq1); memset(acb, 0, sizeof acb);
            memset(&acfg, 0, sizeof acfg);
            memset(aspec, 'b', 64); aspec[64] = 0;
            omega_blackwell_build_qmd0(aq0, 0, 0);
            acfg.gpr_count = ak.gpr_count; acfg.threads_per_block = 128; acfg.threads_x = 128;
            acfg.grid_x = 2; acfg.grid_y = 1; acfg.num_elements = 8 * 128; acfg.shared_bytes = 6656;
            omega_blackwell_build_qmd1(aq1, &acfg);
            for (int i = 0; i < 4; i++) aps[i] = (OmegaCubinParam){ (uint32_t)(0x380 + 8 * i), 8, 1 };
            for (int i = 0; i < 8; i++) aps[4 + i] = (OmegaCubinParam){ (uint32_t)(0x3a0 + 4 * i), 4, 0 };
            snprintf(p2, sizeof p2, "%s/a1", dir);
            fill_artifact(&a, contract, aspec, &ak, aps, 12, acb, 16, aq0, aq1,
                          "gqa-causal-attention", "gqa-causal-v1", "omega_gqa_attention_causal", 2, 1, 128, 6656);
            a.requires = "sm_121,nvidia-cubin-abi-v8";
            int r2 = omega_native_artifact_export(&a, p2, id, err, sizeof err);
            CHECK(r2 == OMEGA_ARTIFACT_OK, "attention export: %s %s", omega_native_artifact_rc_name(r2), err);
            if (r2 == OMEGA_ARTIFACT_OK) {
                int r3 = omega_native_artifact_admit(p2, "580.173.02", id, err, sizeof err);
                CHECK(r3 == OMEGA_ARTIFACT_OK, "attention admit: %s %s", omega_native_artifact_rc_name(r3), err);
            }
            omega_blackwell_kernel_free(&ak);
        }
    }

    omega_blackwell_kernel_free(&k);
    printf("test_gpu_artifact_export: %d checks, %d failed\n", checks, fails);
    return fails ? 1 : 0;
}
