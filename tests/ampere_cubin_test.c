/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */
/* Host-only ELF/container checks. No driver, libcuda or device access. */
#include "omega_cubin.h"
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); return 1; } } while (0)
static uint16_t u16(const uint8_t *p) { return (uint16_t)(p[0] | (uint16_t)p[1] << 8); }
static uint32_t u32(const uint8_t *p) { return (uint32_t)u16(p) | (uint32_t)u16(p + 2) << 16; }
static uint64_t u64(const uint8_t *p) { return (uint64_t)u32(p) | (uint64_t)u32(p + 4) << 32; }

/* Independent metadata reader, with no inferred target from a filename. */
static int target_matches(const uint8_t *p, size_t n, uint32_t sm) {
    if (n < 64 || memcmp(p, "\177ELF", 4) || p[4] != 2 || p[5] != 1 || u16(p + 18) != 190) return 0;
    uint32_t flags = sm == 86 ? ((86u << 16) | 0x500u | 86u) : OMEGA_CUBIN_EFLAGS_SM121;
    if (sm != 86 && sm != 121) return 0;
    if (u32(p + 48) != flags || p[7] != (sm == 86 ? 0x33 : 0x41) || p[8] != (sm == 86 ? 7 : 8)) return 0;
    if (u32(p + 20) != (sm == 86 ? 130u : 1u)) return 0;
    uint64_t sh = u64(p + 40);
    size_t count = u16(p + 60), step = u16(p + 58);
    if (step != 64 || sh > n || count > (n - (size_t)sh) / step) return 0;
    unsigned note_seen = 0, bank_seen = 0;
    for (size_t i = 1; i < count; ++i) {
        const uint8_t *s = p + (size_t)sh + i * step;
        uint32_t ty = u32(s + 4);
        uint64_t off = u64(s + 24), len = u64(s + 32);
        if (ty != 8 && (off > n || len > n - off)) return 0;
        /* Blackwell-only capability/mercury sections must not leak into Ampere. */
        if (sm == 86 && (ty == 0x70000016u || ty == 0x70000082u || ty == 0x70000083u || ty == 0x70000085u)) return 0;
        if (ty == 0x70000000u) {
            for (uint64_t j = 0; j + 12 <= len; j += 4) {
                const uint8_t *attr = p + off + j;
                if (u16(attr) == 0x0a04 && u16(attr + 2) == 8) {
                    if (u32(attr + 4) != 3 || u16(attr + 8) !=
                        (sm == 86 ? OMEGA_CUBIN_PARAM_BASE_SM86 : OMEGA_CUBIN_PARAM_BASE)) return 0;
                    ++bank_seen;
                }
            }
        }
        if (ty == 7) {
            if (len != 32 || u32(p + off) != 12 || u32(p + off + 4) != 8 ||
                memcmp(p + off + 12, "NVIDIA Corp", 11) || u16(p + off + 26) != sm) return 0;
            ++note_seen;
        }
    }
    return note_seen == 1 && bank_seen == 1;
}

int main(int argc, char **argv) {
    /* Our EXIT word; offline nvdisasm is the independent decoding oracle.
     * No NVIDIA binary or listing is embedded in this test. */
    const uint8_t text[16] = {0x4d,0x79,0,0, 0,0,0,0, 0,0,0x80,3, 0,0xea,0x0f,0};
    OmegaCubinParam param = {OMEGA_CUBIN_PARAM_BASE_SM86, 8, 1};
    uint8_t *image = NULL, *old = NULL;
    size_t n = 0, oldn = 0;
    char err[256];
    int rc = omega_cubin_pack_target(86, "ampere_exit", text, sizeof text, &param, 1, 1024, 8, 0, &image, &n, err, sizeof err);
    CHECK(rc == OMEGA_CUBIN_OK);
    CHECK(target_matches(image, n, 86));
    CHECK(!target_matches(image, n, 121));
    image[48] ^= 1;
    CHECK(!target_matches(image, n, 86)); /* negative control: wrong e_flags */
    image[48] ^= 1;
    image[8] = 8;
    CHECK(!target_matches(image, n, 86)); /* negative control: wrong ABI */
    image[8] = 7;
    if (argc == 2) {
        /* Create new offline fixtures privately even with a permissive umask. */
        int fd = open(argv[1], O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
        FILE *f = fd >= 0 ? fdopen(fd, "wb") : NULL;
        if (!f && fd >= 0) close(fd);
        CHECK(f != NULL);
        int wrote = fwrite(image, 1, n, f) == n;
        int closed = fclose(f) == 0;
        CHECK(wrote && closed);
    } else CHECK(argc == 1);
    omega_cubin_free(image);
    image = NULL;
    CHECK(omega_cubin_pack_target(0, "bad", text, sizeof text, NULL, 0, 1024, 8, 0, &image, &n, err, sizeof err) == OMEGA_CUBIN_E_TARGET);
    CHECK(image == NULL && n == 0);
    CHECK(omega_cubin_pack_target(89, "bad", text, sizeof text, NULL, 0, 1024, 8, 0, &image, &n, err, sizeof err) == OMEGA_CUBIN_E_TARGET);
    CHECK(omega_cubin_pack_target(121, "bad", text, sizeof text, &param, 1, 1024, 8, 0, &image, &n, err, sizeof err) == OMEGA_CUBIN_E_PARAMS);
    param.offset = OMEGA_CUBIN_PARAM_BASE;
    CHECK(omega_cubin_pack("legacy", text, sizeof text, &param, 1, 1024, 8, 0, &old, &oldn, err, sizeof err) == 0);
    CHECK(omega_cubin_pack_target(121, "legacy", text, sizeof text, &param, 1, 1024, 8, 0, &image, &n, err, sizeof err) == 0);
    CHECK(n == oldn && memcmp(image, old, n) == 0 && target_matches(image, n, 121));
    omega_cubin_free(image);
    omega_cubin_free(old);
    puts("AMPERE CUBIN HOST PASS; device NOT_RUN");
    return 0;
}
