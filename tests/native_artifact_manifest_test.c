/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */
/* Host-only canonical-manifest admission tests. The synthetic instruction bytes
 * are never executed and make no device correctness claim. */
#include <fcntl.h>
#include <sys/stat.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "omega_native_artifact.h"
#include "sha256.h"

static int checks, failures;
#define CHECK(c, msg) do { checks++; if (!(c)) { failures++; fprintf(stderr, "FAIL: %s\n", msg); } } while (0)
static void save(const char *path, const char *s, size_t n) {
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
    if (fd < 0) exit(2);
    FILE *f = fdopen(fd, "wb");
    if (!f) { close(fd); exit(2); }
    if (fwrite(s, 1, n, f) != n || fclose(f)) exit(2);
}
/* Test mutations must fail as fixture errors, not dereference a missing field. */
static char *field(char *m, const char *key) {
    char *p = strstr(m, key);
    if (!p) { fprintf(stderr, "fixture missing %s\n", key); exit(2); }
    return p;
}
static char *next_line(char *p) {
    char *nl = strchr(p, '\n');
    if (!nl) { fprintf(stderr, "fixture missing newline\n"); exit(2); }
    return nl + 1;
}
static void insert_record(char *m, size_t cap, const char *key, const char *record) {
    char *p = field(m, key);
    size_t n = strlen(m), extra = strlen(record);
    if (n >= cap || extra >= cap - n) exit(2);
    size_t off = (size_t)(p - m);
    if (off > n) exit(2);
    memmove(m + off + extra, m + off, n - off + 1);
    memcpy(m + off, record, extra);
}
static void rehash(char *m) {
    char *start = field(m, "\nop=") + 1, *end = field(start, "provenance_tool=");
    char *id = field(m, "artifact_id=") + 12;
    if (end <= start || (size_t)(next_line(id) - id) != 65) exit(2);
    static const char hex[] = "0123456789abcdef";
    uint8_t digest[32];
    sha256_hash((uint8_t *)start, (size_t)(end - start), digest);
    for (int i = 0; i < 32; i++) { id[2*i] = hex[digest[i] >> 4]; id[2*i+1] = hex[digest[i] & 15]; }
}
static void refuse(const char *prefix, const char *path, char *m, size_t n, const char *msg) {
    char id[65], err[256];
    save(path, m, n);
    int rc = omega_native_artifact_admit(prefix, "580.173.02", id, err, sizeof err);
    CHECK(rc == OMEGA_ARTIFACT_E_FORMAT, msg);
}
static const char *const suffixes[] = {".text.bin",".cubin",".qmd0.bin",".qmd1.bin",".cbank_args.bin",".manifest",".json",".sha256"};
static void remove_artifact(const char *prefix) {
    char p[1100];
    for (size_t i = 0; i < sizeof suffixes / sizeof suffixes[0]; i++) {
        snprintf(p, sizeof p, "%s%s", prefix, suffixes[i]);
        unlink(p);
    }
}
/* The QMD 1 launch words omega_blackwell_build_qmd1 writes for this launch. */
static void set_launch(uint32_t *q1, uint32_t gx, uint32_t gy, uint32_t bx, uint32_t by, uint32_t bz,
                       uint32_t shared, uint32_t gpr) {
    q1[34] = bx | (by << 16);
    q1[35] = bz | ((gpr < 16 ? 16 : gpr) << 8) | (1u << 17);
    q1[36] = 0x04b44808u & ~0x7ffu;
    q1[36] |= shared ? (shared + 127) / 128 : 8;
    q1[39] = gx; q1[40] = gy; q1[41] = 1;
}
static int file_text(const char *path, char *buf, size_t cap) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    size_t n = fread(buf, 1, cap - 1, f);
    fclose(f);
    buf[n] = 0;
    return (int)n;
}
/* Replace the value of key in manifest m (one line), then re-bind the id. */
static void set_field(char *m, const char *key, const char *val) {
    char *v = field(m, key) + strlen(key), *nl = next_line(v) - 1;
    size_t vl = strlen(val);
    memmove(v + vl, nl, strlen(nl) + 1);
    memcpy(v, val, vl);
    rehash(m);
}
static void hex_digest(const void *b, size_t n, char out[65]) {
    static const char hex[] = "0123456789abcdef";
    uint8_t d[32];
    sha256_hash((const uint8_t *)b, n, d);
    for (int i = 0; i < 32; i++) { out[2*i] = hex[d[i] >> 4]; out[2*i+1] = hex[d[i] & 15]; }
    out[64] = 0;
}
static int admit_rc(const char *prefix, const char *driver, char id[65]) {
    char err[256];
    int rc = omega_native_artifact_admit(prefix, driver, id, err, sizeof err);
    if (rc) fprintf(stderr, "  (refused as expected? rc=%d %s)\n", rc, err);
    return rc;
}
int main(void) {
    char dir[] = "/tmp/cudapp-manifest-XXXXXX", prefix[256], path[280], id[65], err[256];
    if (!mkdtemp(dir)) return 2;
    /* Even a permissive invoking umask must not create shared test files. */
    snprintf(path, sizeof path, "%s/mode-check", dir);
    mode_t mask = umask(0);
    save(path, "test", 4);
    umask(mask);
    struct stat st;
    CHECK(stat(path, &st) == 0 && (st.st_mode & 0777) == 0600, "fixture file mode is not private");
    unlink(path);
    snprintf(prefix, sizeof prefix, "%s/fixture", dir);
    snprintf(path, sizeof path, "%s.manifest", prefix);
    uint8_t text[16] = {0}; uint32_t q0[96] = {0}, q1[96] = {0}, cb[2] = {0};
    q1[52] = 0x300u >> 6;
    set_launch(q1, 1, 1, 1, 1, 1, 0, 1);
    OmegaCubinParam param = {0x380, 8, 1};
    char hash[65]; memset(hash, 'a', 64); hash[64] = 0;
    OmegaNativeArtifact a = {
        .op="matmul", .shape="m=1,k=1,n=1", .bounds="m<=1,k<=1,n<=1", .contract_name="test",
        .contract_id_hex=hash, .semantic_spec_id_hex=hash, .kernel_function="test",
        .precision="bf16", .accumulation="fp32", .tolerance="0", .text=text, .text_len=sizeof text,
        .params=&param, .nparams=1, .gpr_count=1, .grid_x=1, .grid_y=1, .block_x=1, .block_y=1, .block_z=1,
        .cbank_offset=0x380, .cbank_args=cb, .cbank_words=2, .qmd0=q0, .qmd1=q1,
        .min_driver="580.173.02", .requires="sm_121", .source_commit="UNKNOWN", .tool_version="test/1",
        .qualification="device-load=NOT_RUN"
    };
    CHECK(omega_native_artifact_export(&a,prefix,id,err,sizeof err)==0, "export fixture");
    CHECK(omega_native_artifact_admit(prefix,"580.173.02",id,err,sizeof err)==0, "admit canonical fixture");
    char original[8192], m[16384];
    FILE *f=fopen(path,"rb"); if (!f) return 2;
    size_t n=fread(original,1,sizeof original-1,f);
    if (ferror(f) || n == 0 || n == sizeof original-1) { fclose(f); return 2; }
    fclose(f); original[n]=0;

    /* Move a checked policy after the identity terminator; the old parser
     * accepts both policies under exactly the same artifact id. */
    char *line=field(original,"min_driver="), *nl=next_line(line);
    size_t before=(size_t)(line-original), len=(size_t)(nl-line);
    memcpy(m, original, before); strcpy(m+before, nl);
    strncat(m, line, len); rehash(m);
    refuse(prefix,path,m,strlen(m),"minimum driver outside identity accepted");
    char *policy=field(m,"min_driver=")+11;
    if ((size_t)(next_line(policy) - policy) != 11) return 2;
    memcpy(policy,"000.000.00",10);
    refuse(prefix,path,m,strlen(m),"changed unhashed minimum driver accepted under same identity");

    strcpy(m,original); strcat(m,"isa=sm_120\n");
    refuse(prefix,path,m,strlen(m),"duplicate key after terminator accepted");
    strcpy(m,original); insert_record(m,sizeof m,"shape=","shape=shadow\n"); rehash(m);
    refuse(prefix,path,m,strlen(m),"duplicate key inside identity accepted");
    strcpy(m,original); line=field(m,"qualification="); nl=next_line(line); memmove(line,nl,strlen(nl)+1); rehash(m);
    refuse(prefix,path,m,strlen(m),"missing qualification accepted");
    strcpy(m,original); line=field(m,"shape="); nl=next_line(line);
    char *next=next_line(nl); char swap[256]; size_t l1=(size_t)(nl-line),l2=(size_t)(next-nl);
    if (l1 > sizeof swap) return 2;
    memcpy(swap,line,l1); memmove(line,nl,l2); memcpy(line+l2,swap,l1); rehash(m);
    refuse(prefix,path,m,strlen(m),"reordered canonical fields accepted");
    strcpy(m,original); m[n]='\0'; memcpy(m+n+1,"isa=sm_120\n",11);
    refuse(prefix,path,m,n+12,"embedded NUL with trailing policy accepted");
    strcpy(m,original); line=field(m,"provenance_tool="); *line=0;
    refuse(prefix,path,m,strlen(m),"missing identity boundary accepted");
    strcpy(m,original); m[n-1]=0;
    refuse(prefix,path,m,n-1,"missing final newline accepted");

    save(path,original,n);
    CHECK(omega_native_artifact_admit(prefix,"580.173.02",id,err,sizeof err)==0,"canonical fixture still accepted");
    remove_artifact(prefix);

    /* ---- audit CudaAuditFable F5/F3/F7/F8 (2026-10-26): admission and export fail closed ---- */
    char man[16384], val[128], pfx[256];
    snprintf(pfx, sizeof pfx, "%s/adm", dir);
    CHECK(omega_native_artifact_export(&a,pfx,id,err,sizeof err)==0, "export admission fixture");
    snprintf(path, sizeof path, "%s.manifest", pfx);
    if (file_text(path, original, sizeof original) <= 0) return 2;
    /* F5: a minimum driver that does not parse was read as 0.0.0 and admitted. */
    const char *bad_versions[] = {"garbage", "x999.0.0", "", "9999999999.0.0", "580.173", "580.173.02x", "-1.0.0"};
    for (size_t i = 0; i < sizeof bad_versions / sizeof bad_versions[0]; i++) {
        strcpy(m, original); set_field(m, "min_driver=", bad_versions[i]); save(path, m, strlen(m));
        snprintf(val, sizeof val, "unparsable manifest min_driver '%s' admitted", bad_versions[i]);
        CHECK(admit_rc(pfx, "580.173.02", id) == OMEGA_ARTIFACT_E_FORMAT && id[0] == 0, val);
    }
    save(path, original, strlen(original));
    CHECK(admit_rc(pfx, "garbage", id) == OMEGA_ARTIFACT_E_ARG, "unparsable current driver admitted");
    CHECK(admit_rc(pfx, "580.173.01", id) == OMEGA_ARTIFACT_E_DRIVER && id[0] == 0, "older driver admitted or id reported");
    CHECK(admit_rc(pfx, "580.173.02", id) == 0 && id[0] != 0, "admission fixture refused after driver checks");
    /* F5b: cbank_words with a trailing suffix was read by strtoul and admitted. */
    strcpy(m, original); set_field(m, "cbank_words=", "2abc"); save(path, m, strlen(m));
    CHECK(admit_rc(pfx, "580.173.02", id) == OMEGA_ARTIFACT_E_FORMAT, "cbank_words '2abc' admitted");
    strcpy(m, original); set_field(m, "cbank_words=", "+2"); save(path, m, strlen(m));
    CHECK(admit_rc(pfx, "580.173.02", id) == OMEGA_ARTIFACT_E_FORMAT, "cbank_words '+2' admitted");
    /* F3 (cbank): 33 words exceed the 32 the launch path accepts, even with a matching file and digest. */
    {
        uint32_t big[33] = {0};
        char cbp[1100], dg[65];
        snprintf(cbp, sizeof cbp, "%s.cbank_args.bin", pfx);
        save(cbp, (const char *)big, sizeof big);
        hex_digest(big, sizeof big, dg);
        strcpy(m, original); set_field(m, "cbank_words=", "33"); set_field(m, "cbank_args_sha256=", dg);
        save(path, m, strlen(m));
        CHECK(admit_rc(pfx, "580.173.02", id) == OMEGA_ARTIFACT_E_FORMAT, "33-word constant bank admitted");
        save(cbp, (const char *)cb, sizeof cb);
    }
    /* F3 (QMD): every digest matches but QMD 1 launches 2^31-1 CTAs with 511 registers. */
    {
        uint32_t hostile[96];
        char qp[1100], dg[65];
        memcpy(hostile, q1, sizeof hostile);
        hostile[39] = 0x7fffffffu;
        hostile[35] = (hostile[35] & ~(0x1ffu << 8)) | (511u << 8);
        snprintf(qp, sizeof qp, "%s.qmd1.bin", pfx);
        save(qp, (const char *)hostile, sizeof hostile);
        hex_digest(hostile, sizeof hostile, dg);
        strcpy(m, original); set_field(m, "qmd1_sha256=", dg); save(path, m, strlen(m));
        CHECK(admit_rc(pfx, "580.173.02", id) == OMEGA_ARTIFACT_E_CONTRACT && id[0] == 0,
              "QMD launching 0x7fffffff CTAs under a 1x1 manifest admitted");
        save(qp, (const char *)q1, sizeof q1);
    }
    save(path, original, strlen(original));
    CHECK(admit_rc(pfx, "580.173.02", id) == 0, "admission fixture refused after restore");

    /* F8: an existing artifact is never overwritten, and a planted symlink is not followed. */
    CHECK(omega_native_artifact_export(&a,pfx,id,err,sizeof err)==OMEGA_ARTIFACT_E_IO, "export overwrote an existing artifact");
    remove_artifact(pfx);
    {
        char victim[1100], link[1100], vbuf[64];
        snprintf(victim, sizeof victim, "%s/victim", dir);
        snprintf(link, sizeof link, "%s.text.bin", pfx);
        save(victim, "keep", 4);
        CHECK(symlink(victim, link) == 0, "fixture symlink");
        CHECK(omega_native_artifact_export(&a,pfx,id,err,sizeof err)==OMEGA_ARTIFACT_E_IO, "export followed a planted symlink");
        CHECK(file_text(victim, vbuf, sizeof vbuf) == 4 && strcmp(vbuf, "keep") == 0, "symlink target was written");
        unlink(link); unlink(victim); remove_artifact(pfx);
    }
    /* F8: a 1023-byte prefix truncated every suffix away; all eight writes hit the bare prefix. */
    {
        char longp[1024];
        size_t at = (size_t)snprintf(longp, sizeof longp, "%s/", dir);
        while (at < 1021) { longp[at++] = '.'; longp[at++] = '/'; }
        while (at < 1023) longp[at++] = 'x';
        longp[at] = 0;
        CHECK(strlen(longp) == 1023, "fixture long prefix length");
        CHECK(omega_native_artifact_export(&a,longp,id,err,sizeof err)==OMEGA_ARTIFACT_E_ARG, "1023-byte prefix accepted");
        CHECK(access(longp, F_OK) != 0, "a file was written at the truncated prefix");
        unlink(longp);
        /* Positive control: the longest prefix whose every path fits is accepted. */
        size_t fit = 1023 - strlen(".cbank_args.bin");
        longp[fit] = 0;
        if (longp[fit - 1] == '/') longp[fit - 1] = 'x';
        CHECK(omega_native_artifact_export(&a,longp,id,err,sizeof err)==0, "longest fitting prefix refused");
        CHECK(admit_rc(longp, "580.173.02", id) == 0, "longest fitting prefix not admitted");
        remove_artifact(longp);
    }
    /* F5/F7: export refuses what admission would refuse, and escapes JSON. */
    {
        OmegaNativeArtifact b = a;
        b.min_driver = "garbage";
        CHECK(omega_native_artifact_export(&b,pfx,id,err,sizeof err)==OMEGA_ARTIFACT_E_ARG, "export accepted min_driver 'garbage'");
        b = a; b.tolerance = "0.1\nisa=sm_120";
        CHECK(omega_native_artifact_export(&b,pfx,id,err,sizeof err)==OMEGA_ARTIFACT_E_ARG, "export accepted a newline in a field");
        b = a; uint32_t cb33[33] = {0}; b.cbank_args = cb33; b.cbank_words = 33;
        CHECK(omega_native_artifact_export(&b,pfx,id,err,sizeof err)!=0, "export accepted 33 constant-bank words");
        b = a; uint32_t hq[96]; memcpy(hq, q1, sizeof hq); hq[39] = 0x7fffffffu; b.qmd1 = hq;
        CHECK(omega_native_artifact_export(&b,pfx,id,err,sizeof err)==OMEGA_ARTIFACT_E_CONTRACT, "export accepted a QMD that disagrees with grid");
        b = a; b.shared_bytes = 640;
        CHECK(omega_native_artifact_export(&b,pfx,id,err,sizeof err)==OMEGA_ARTIFACT_E_CONTRACT, "export accepted a QMD that disagrees with shared_bytes");
        CHECK(access(pfx, F_OK) != 0, "refused export left a file");
        snprintf(path, sizeof path, "%s.text.bin", pfx);
        CHECK(access(path, F_OK) != 0, "refused export left text.bin");
        b = a; b.tolerance = "a\"b\\c";
        CHECK(omega_native_artifact_export(&b,pfx,id,err,sizeof err)==0, "export refused a quote in tolerance");
        snprintf(path, sizeof path, "%s.json", pfx);
        CHECK(file_text(path, man, sizeof man) > 0 && strstr(man, "\"tolerance\": \"a\\\"b\\\\c\"") != NULL,
              "JSON string not escaped");
        remove_artifact(pfx);
    }
    rmdir(dir);
    printf("native_artifact_manifest: %d checks, %d failed\n",checks,failures);
    return failures ? 1 : 0;
}
