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
    const char *suffix[]={".text.bin",".cubin",".qmd0.bin",".qmd1.bin",".cbank_args.bin",".manifest",".json",".sha256"};
    for (size_t i=0;i<sizeof suffix/sizeof suffix[0];i++){ snprintf(path,sizeof path,"%s%s",prefix,suffix[i]); unlink(path); }
    rmdir(dir);
    printf("native_artifact_manifest: %d checks, %d failed\n",checks,failures);
    return failures ? 1 : 0;
}
