/* SPDX-License-Identifier: AGPL-3.0-only */
/* omega_native_artifact: offline digest-bound handoff for one GPU realization.
 * See the header. No device, no launch, no live address. */
#include "omega_native_artifact.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sha256.h"
#include "omega_blackwell_realize.h"

#define MANIFEST_CAP 8192u

const char *omega_native_artifact_rc_name(int rc) {
    switch (rc) {
    case OMEGA_ARTIFACT_OK: return "OK";
    case OMEGA_ARTIFACT_E_ARG: return "ARG";
    case OMEGA_ARTIFACT_E_FORMAT: return "FORMAT";
    case OMEGA_ARTIFACT_E_CONTRACT: return "CONTRACT";
    case OMEGA_ARTIFACT_E_ISA: return "ISA";
    case OMEGA_ARTIFACT_E_DIGEST: return "DIGEST";
    case OMEGA_ARTIFACT_E_IO: return "IO";
    case OMEGA_ARTIFACT_E_RELOC: return "RELOC";
    case OMEGA_ARTIFACT_E_DRIVER: return "DRIVER";
    case OMEGA_ARTIFACT_E_UNSUPPORTED: return "UNSUPPORTED";
    case OMEGA_ARTIFACT_E_MEM: return "MEM";
    default: return "UNKNOWN";
    }
}

static void hex64(const uint8_t d[32], char out[65]) {
    static const char h[] = "0123456789abcdef";
    for (int i = 0; i < 32; i++) {
        out[2 * i] = h[d[i] >> 4];
        out[2 * i + 1] = h[d[i] & 15];
    }
    out[64] = 0;
}

static int is_hex64(const char *s) {
    if (!s || strlen(s) != 64) return 0;
    for (int i = 0; i < 64; i++) {
        char c = s[i];
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) return 0;
    }
    return 1;
}

static int append(char **p, size_t *rem, const char *fmt, ...) {
    va_list ap;
    int n;
    va_start(ap, fmt);
    n = vsnprintf(*p, *rem, fmt, ap);
    va_end(ap);
    if (n < 0 || (size_t)n >= *rem) return -1;
    *p += n;
    *rem -= (size_t)n;
    return 0;
}

static int write_file(const char *path, const void *data, size_t n) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    if (n && fwrite(data, 1, n, f) != n) { fclose(f); return -1; }
    if (fclose(f) != 0) return -1;
    return 0;
}

static uint8_t *read_file(const char *path, size_t *n) {
    FILE *f = fopen(path, "rb");
    long sz;
    uint8_t *b;
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    sz = ftell(f);
    if (sz < 0 || sz > (long)(1u << 26)) { fclose(f); return NULL; }
    if (fseek(f, 0, SEEK_SET) != 0) { fclose(f); return NULL; }
    b = malloc((size_t)sz + 1u);
    if (!b) { fclose(f); return NULL; }
    if (sz && fread(b, 1, (size_t)sz, f) != (size_t)sz) { free(b); fclose(f); return NULL; }
    b[sz] = 0;
    *n = (size_t)sz;
    fclose(f);
    return b;
}

/* The QMD is a template: every device address field must be zero. */
static int qmd_template_ok(const uint32_t q0[OMEGA_ARTIFACT_QMD_WORDS],
                           const uint32_t q1[OMEGA_ARTIFACT_QMD_WORDS]) {
    if (q0[12] != 0) return 0;
    if (q1[12] != 0 || q1[15] != 0 || q1[16] != 0) return 0;
    if (q1[32] != 0 || (q1[33] & ~(0xa << 21)) != 0) return 0;
    if (q1[42] != 0 || (q1[43] & 0x7ffffu) != 0) return 0;
    if (q1[44] != 0 || (q1[45] & 0x7ffffu) != 0) return 0;
    if (q1[52] != (0x300u >> 6) || (q1[53] & 0x7ffffu) != 0) return 0;
    return 1;
}

static void json_put(FILE *f, const char *key, const char *val, int last) {
    fprintf(f, "  \"%s\": \"%s\"%s\n", key, val, last ? "" : ",");
}

int omega_native_artifact_export(const OmegaNativeArtifact *a, const char *out_prefix,
                                 char out_id[OMEGA_ARTIFACT_ID_MAX], char *err, size_t err_cap) {
    char text_hex[65], cub_hex[65], q0_hex[65], q1_hex[65], cb_hex[65], id[65];
    uint8_t td[32], cd[32], q0d[32], q1d[32], cbd[32];
    uint8_t *cub = NULL, *manifest = NULL;
    size_t cub_len = 0, mlen;
    char machine_hex[65];
    SemanticId machine;
    char path[1024];
    int rc = OMEGA_ARTIFACT_OK;

    if (err && err_cap) err[0] = 0;
    if (out_id) out_id[0] = 0;
    if (!a || !out_prefix || !out_id || !a->op || !a->shape || !a->bounds || !a->contract_name || !a->kernel_function ||
        !a->text || !a->text_len || !a->params || !a->qmd0 || !a->qmd1 || !a->cbank_args ||
        !a->semantic_spec_id_hex || !a->contract_id_hex || !a->min_driver || !a->requires ||
        !a->precision || !a->accumulation || !a->tolerance || !a->source_commit || !a->tool_version ||
        !a->qualification) {
        if (err) snprintf(err, err_cap, "missing required artifact field");
        return OMEGA_ARTIFACT_E_ARG;
    }
    if (!is_hex64(a->contract_id_hex) || !is_hex64(a->semantic_spec_id_hex)) {
        if (err) snprintf(err, err_cap, "contract_id and semantic_spec_id must be 64 lowercase hex");
        return OMEGA_ARTIFACT_E_CONTRACT;
    }
    if (strcmp(a->op, "matmul") != 0 && strcmp(a->op, "gqa-causal-attention") != 0) {
        if (err) snprintf(err, err_cap, "unsupported op '%s'", a->op);
        return OMEGA_ARTIFACT_E_UNSUPPORTED;
    }
    if (!qmd_template_ok(a->qmd0, a->qmd1)) {
        if (err) snprintf(err, err_cap, "QMD contains a live device address; export a zero-VA template");
        return OMEGA_ARTIFACT_E_RELOC;
    }
    if (omega_blackwell_get_machine_id(&machine) != 0) {
        if (err) snprintf(err, err_cap, "machine profile unavailable");
        return OMEGA_ARTIFACT_E_ARG;
    }
    hex64(machine.bytes, machine_hex);

    {
        uint32_t cbank_end = OMEGA_CUBIN_PARAM_BASE;
        for (size_t i = 0; i < a->nparams; i++) cbank_end = a->params[i].offset + a->params[i].size;
        rc = omega_cubin_pack(a->kernel_function, a->text, a->text_len, a->params, a->nparams,
                              cbank_end, a->gpr_count, 64u, &cub, &cub_len, err, err_cap);
    }
    if (rc != OMEGA_CUBIN_OK) return OMEGA_ARTIFACT_E_UNSUPPORTED;
    if (a->cbank_offset != OMEGA_CUBIN_PARAM_BASE || a->cbank_words == 0 || a->cbank_words > 256) {
        free(cub);
        if (err) snprintf(err, err_cap, "cbank offset/words outside the supported layout");
        return OMEGA_ARTIFACT_E_UNSUPPORTED;
    }

    sha256_hash(a->text, a->text_len, td);
    sha256_hash(cub, cub_len, cd);
    sha256_hash((const uint8_t *)a->qmd0, OMEGA_ARTIFACT_QMD_WORDS * 4u, q0d);
    sha256_hash((const uint8_t *)a->qmd1, OMEGA_ARTIFACT_QMD_WORDS * 4u, q1d);
    sha256_hash((const uint8_t *)a->cbank_args, a->cbank_words * 4u, cbd);
    hex64(td, text_hex); hex64(cd, cub_hex); hex64(q0d, q0_hex); hex64(q1d, q1_hex); hex64(cbd, cb_hex);

    /* Identity block: exactly these lines, in this order, each newline-terminated. */
    {
        char idbuf[MANIFEST_CAP], *p = idbuf;
        size_t rem = sizeof idbuf;
        if (append(&p, &rem, "op=%s\n", a->op) ||
            append(&p, &rem, "shape=%s\n", a->shape) ||
            append(&p, &rem, "bounds=%s\n", a->bounds) ||
            append(&p, &rem, "contract=%s\n", a->contract_name) ||
            append(&p, &rem, "contract_id=%s\n", a->contract_id_hex) ||
            append(&p, &rem, "semantic_spec_id=%s\n", a->semantic_spec_id_hex) ||
            append(&p, &rem, "isa=sm_121\n") ||
            append(&p, &rem, "machine_profile=%s\n", machine_hex) ||
            append(&p, &rem, "kernel_format=nvidia-cubin-elf64\n") ||
            append(&p, &rem, "kernel_function=%s\n", a->kernel_function) ||
            append(&p, &rem, "text_sha256=%s\n", text_hex) ||
            append(&p, &rem, "text_size=%zu\n", a->text_len) ||
            append(&p, &rem, "cubin_sha256=%s\n", cub_hex) ||
            append(&p, &rem, "cubin_size=%zu\n", cub_len) ||
            append(&p, &rem, "qmd0_sha256=%s\n", q0_hex) ||
            append(&p, &rem, "qmd1_sha256=%s\n", q1_hex) ||
            append(&p, &rem, "qmd_template=zero-va-v1\n") ||
            append(&p, &rem, "cbank_offset=0x%x\n", a->cbank_offset) ||
            append(&p, &rem, "cbank_args_sha256=%s\n", cb_hex) ||
            append(&p, &rem, "grid=%u,%u,1\n", a->grid_x, a->grid_y) ||
            append(&p, &rem, "block=%u,%u,%u\n", a->block_x, a->block_y, a->block_z) ||
            append(&p, &rem, "shared_bytes=%u\n", a->shared_bytes) ||
            append(&p, &rem, "gpr_count=%u\n", a->gpr_count) ||
            append(&p, &rem, "precision=%s\n", a->precision) ||
            append(&p, &rem, "accumulation=%s\n", a->accumulation) ||
            append(&p, &rem, "tolerance=%s\n", a->tolerance) ||
            append(&p, &rem, "min_driver=%s\n", a->min_driver) ||
            append(&p, &rem, "requires=%s\n", a->requires)) {
            free(cub);
            if (err) snprintf(err, err_cap, "identity block overflow");
            return OMEGA_ARTIFACT_E_ARG;
        }
        mlen = (size_t)(p - idbuf);
        sha256_hash((const uint8_t *)idbuf, mlen, cd);
        hex64(cd, id);
        manifest = malloc(MANIFEST_CAP);
        if (!manifest) { free(cub); return OMEGA_ARTIFACT_E_MEM; }
        {
            char *mp = (char *)manifest;
            size_t mr = MANIFEST_CAP;
            if (append(&mp, &mr, "%s\n", OMEGA_ARTIFACT_FORMAT) ||
                append(&mp, &mr, "artifact_id=%s\n", id) ||
                append(&mp, &mr, "%s", idbuf) ||
                append(&mp, &mr, "provenance_tool=%s\n", a->tool_version) ||
                append(&mp, &mr, "provenance_source_commit=%s\n", a->source_commit) ||
                append(&mp, &mr, "qualification=%s\n", a->qualification)) {
                free(cub); free(manifest);
                if (err) snprintf(err, err_cap, "manifest overflow");
                return OMEGA_ARTIFACT_E_ARG;
            }
            mlen = (size_t)(mp - (char *)manifest);
        }
    }

    /* Files. Every write is checked; the manifest is written last. */
    snprintf(path, sizeof path, "%s.text.bin", out_prefix);
    if (write_file(path, a->text, a->text_len)) { rc = OMEGA_ARTIFACT_E_IO; goto done; }
    snprintf(path, sizeof path, "%s.cubin", out_prefix);
    if (write_file(path, cub, cub_len)) { rc = OMEGA_ARTIFACT_E_IO; goto done; }
    snprintf(path, sizeof path, "%s.qmd0.bin", out_prefix);
    if (write_file(path, a->qmd0, OMEGA_ARTIFACT_QMD_WORDS * 4u)) { rc = OMEGA_ARTIFACT_E_IO; goto done; }
    snprintf(path, sizeof path, "%s.qmd1.bin", out_prefix);
    if (write_file(path, a->qmd1, OMEGA_ARTIFACT_QMD_WORDS * 4u)) { rc = OMEGA_ARTIFACT_E_IO; goto done; }
    snprintf(path, sizeof path, "%s.cbank_args.bin", out_prefix);
    if (write_file(path, a->cbank_args, a->cbank_words * 4u)) { rc = OMEGA_ARTIFACT_E_IO; goto done; }

    snprintf(path, sizeof path, "%s.json", out_prefix);
    {
        FILE *f = fopen(path, "wb");
        if (!f) { rc = OMEGA_ARTIFACT_E_IO; goto done; }
        fprintf(f, "{\n  \"format\": \"%s\",\n", OMEGA_ARTIFACT_FORMAT);
        json_put(f, "artifact_id", id, 0);
        json_put(f, "op", a->op, 0);
        json_put(f, "shape", a->shape, 0);
        json_put(f, "bounds", a->bounds, 0);
        json_put(f, "contract", a->contract_name, 0);
        json_put(f, "contract_id", a->contract_id_hex, 0);
        json_put(f, "semantic_spec_id", a->semantic_spec_id_hex, 0);
        json_put(f, "isa", "sm_121", 0);
        json_put(f, "machine_profile", machine_hex, 0);
        json_put(f, "kernel_format", "nvidia-cubin-elf64", 0);
        json_put(f, "kernel_function", a->kernel_function, 0);
        json_put(f, "text_sha256", text_hex, 0);
        json_put(f, "cubin_sha256", cub_hex, 0);
        json_put(f, "qmd0_sha256", q0_hex, 0);
        json_put(f, "qmd1_sha256", q1_hex, 0);
        json_put(f, "cbank_args_sha256", cb_hex, 0);
        json_put(f, "qmd_template", "zero-va-v1", 0);
        json_put(f, "precision", a->precision, 0);
        json_put(f, "accumulation", a->accumulation, 0);
        json_put(f, "tolerance", a->tolerance, 0);
        json_put(f, "min_driver", a->min_driver, 0);
        json_put(f, "requires", a->requires, 0);
        json_put(f, "provenance_tool", a->tool_version, 0);
        json_put(f, "provenance_source_commit", a->source_commit, 0);
        json_put(f, "qualification", a->qualification, 1);
        fprintf(f, "}\n");
        if (fclose(f) != 0) { rc = OMEGA_ARTIFACT_E_IO; goto done; }
    }
    snprintf(path, sizeof path, "%s.manifest", out_prefix);
    if (write_file(path, manifest, mlen)) { rc = OMEGA_ARTIFACT_E_IO; goto done; }
    snprintf(path, sizeof path, "%s.sha256", out_prefix);
    {
        FILE *f = fopen(path, "wb");
        if (!f) { rc = OMEGA_ARTIFACT_E_IO; goto done; }
        /* Role names, never absolute or caller-chosen paths: the listing is
         * byte-identical for the same artifact exported under any prefix. */
        fprintf(f, "%s  artifact_id\n", id);
        fprintf(f, "%s  manifest\n", id);
        fprintf(f, "%s  cubin\n", cub_hex);
        fprintf(f, "%s  text\n", text_hex);
        fprintf(f, "%s  qmd0\n", q0_hex);
        fprintf(f, "%s  qmd1\n", q1_hex);
        fprintf(f, "%s  cbank_args\n", cb_hex);
        if (fclose(f) != 0) { rc = OMEGA_ARTIFACT_E_IO; goto done; }
    }
    strcpy(out_id, id);
done:
    free(cub);
    free(manifest);
    return rc;
}

/* ---- admission ------------------------------------------------------- */

static char *find_line(char *buf, const char *key) {
    size_t k = strlen(key);
    char *p = buf;
    while (*p) {
        if (strncmp(p, key, k) == 0) return p;
        char *nl = strchr(p, '\n');
        if (!nl) break;
        p = nl + 1;
    }
    return NULL;
}

static int copy_value(const char *line, char *out, size_t cap) {
    const char *eq = strchr(line, '=');
    const char *nl;
    size_t n;
    if (!eq) return -1;
    eq++;
    nl = strchr(eq, '\n');
    n = nl ? (size_t)(nl - eq) : strlen(eq);
    if (n + 1 > cap) return -1;
    memcpy(out, eq, n);
    out[n] = 0;
    return 0;
}

static int semver_ge(const char *have, const char *need) {
    int h[3] = {0,0,0}, n[3] = {0,0,0};
    sscanf(have, "%d.%d.%d", &h[0], &h[1], &h[2]);
    sscanf(need, "%d.%d.%d", &n[0], &n[1], &n[2]);
    for (int i = 0; i < 3; i++) {
        if (h[i] != n[i]) return h[i] > n[i];
    }
    return 1;
}

int omega_native_artifact_admit(const char *out_prefix, const char *current_driver,
                                char out_id[OMEGA_ARTIFACT_ID_MAX], char *err, size_t err_cap) {
    char path[1024], id[65];
    size_t mn = 0;
    uint8_t *man;
    char *ident, *end, *p;
    int rc = OMEGA_ARTIFACT_OK;
    if (err && err_cap) err[0] = 0;
    if (out_id) out_id[0] = 0;
    if (!out_prefix) return OMEGA_ARTIFACT_E_ARG;
    snprintf(path, sizeof path, "%s.manifest", out_prefix);
    man = read_file(path, &mn);
    if (!man) { if (err) snprintf(err, err_cap, "cannot read %s", path); return OMEGA_ARTIFACT_E_IO; }
    if (strncmp((char *)man, OMEGA_ARTIFACT_FORMAT "\n", strlen(OMEGA_ARTIFACT_FORMAT) + 1) != 0) {
        rc = OMEGA_ARTIFACT_E_FORMAT; goto done;
    }
    {
        char *l = find_line((char *)man, "artifact_id=");
        if (!l || copy_value(l, id, sizeof id) || !is_hex64(id)) { rc = OMEGA_ARTIFACT_E_FORMAT; goto done; }
    }
    ident = find_line((char *)man, "op=");
    if (!ident) { rc = OMEGA_ARTIFACT_E_FORMAT; goto done; }
    end = find_line(ident, "provenance_tool=");
    if (!end) { rc = OMEGA_ARTIFACT_E_FORMAT; goto done; }
    {
        uint8_t d[32]; char want[65];
        sha256_hash((const uint8_t *)ident, (size_t)(end - ident), d);
        hex64(d, want);
        if (strcmp(want, id) != 0) {
            if (err) snprintf(err, err_cap, "artifact id mismatch: manifest %s recomputed %s", id, want);
            rc = OMEGA_ARTIFACT_E_DIGEST; goto done;
        }
        strcpy(out_id, id);
    }
    /* Fields and their files. */
    {
        struct { const char *key; const char *suffix; size_t fixed; } checks[] = {
            { "text_sha256=", ".text.bin", 0 },
            { "cubin_sha256=", ".cubin", 0 },
            { "qmd0_sha256=", ".qmd0.bin", OMEGA_ARTIFACT_QMD_WORDS * 4u },
            { "qmd1_sha256=", ".qmd1.bin", OMEGA_ARTIFACT_QMD_WORDS * 4u },
            { "cbank_args_sha256=", ".cbank_args.bin", 0 },
        };
        for (size_t i = 0; i < sizeof checks / sizeof checks[0]; i++) {
            char v[65], fn[1100];
            uint8_t *b, d[32];
            char got[65];
            size_t n;
            p = find_line(ident, checks[i].key);
            if (!p || copy_value(p, v, sizeof v) || !is_hex64(v)) { rc = OMEGA_ARTIFACT_E_FORMAT; goto done; }
            snprintf(fn, sizeof fn, "%s%s", out_prefix, checks[i].suffix);
            b = read_file(fn, &n);
            if (!b) { if (err) snprintf(err, err_cap, "cannot read %s", fn); rc = OMEGA_ARTIFACT_E_IO; goto done; }
            if (checks[i].fixed && n != checks[i].fixed) { free(b); rc = OMEGA_ARTIFACT_E_DIGEST; goto done; }
            sha256_hash(b, n, d); free(b); hex64(d, got);
            if (strcmp(got, v) != 0) {
                if (err) snprintf(err, err_cap, "%s digest mismatch for %s", checks[i].key, fn);
                rc = OMEGA_ARTIFACT_E_DIGEST; goto done;
            }
        }
    }
    {
        char v[128];
        p = find_line(ident, "shape=");
        if (!p || copy_value(p, v, sizeof v) || v[0] == 0) { rc = OMEGA_ARTIFACT_E_FORMAT; goto done; }
        p = find_line(ident, "bounds=");
        if (!p || copy_value(p, v, sizeof v) || v[0] == 0) { rc = OMEGA_ARTIFACT_E_FORMAT; goto done; }
        p = find_line(ident, "isa=");
        if (!p || copy_value(p, v, sizeof v) || strcmp(v, "sm_121") != 0) { rc = OMEGA_ARTIFACT_E_ISA; goto done; }
        p = find_line(ident, "kernel_format=");
        if (!p || copy_value(p, v, sizeof v) || strcmp(v, "nvidia-cubin-elf64") != 0) { rc = OMEGA_ARTIFACT_E_FORMAT; goto done; }
        p = find_line(ident, "qmd_template=");
        if (!p || copy_value(p, v, sizeof v) || strcmp(v, "zero-va-v1") != 0) { rc = OMEGA_ARTIFACT_E_RELOC; goto done; }
        p = find_line(ident, "contract_id=");
        if (!p || copy_value(p, v, sizeof v) || !is_hex64(v)) { rc = OMEGA_ARTIFACT_E_CONTRACT; goto done; }
        p = find_line(ident, "min_driver=");
        if (!p || copy_value(p, v, sizeof v)) { rc = OMEGA_ARTIFACT_E_FORMAT; goto done; }
        if (current_driver && !semver_ge(current_driver, v)) {
            if (err) snprintf(err, err_cap, "driver %s is older than required %s", current_driver, v);
            rc = OMEGA_ARTIFACT_E_DRIVER; goto done;
        }
    }
done:
    free(man);
    return rc;
}
