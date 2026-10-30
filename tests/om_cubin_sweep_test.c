/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */
/* CUDApp: the cubin packer sweep of `make test-om-cubin`. Built with CUDAPP_MIGRATION_PENDING_CUBIN_DIFF, every
 * omega_cubin_pack call below packs through om/cubin/cubin_elf.om and through the C writer and aborts on any
 * difference of return code, length, bytes or message. This driver only chooses the descriptors and checks that
 * the sweep reached every refusal class and enough images; it holds no expected bytes of its own. */
#include "omega_cubin.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned long long calls, by_rc[8], image_bytes;
static uint8_t text[1u << 20];
static uint64_t rng = 0x9e3779b97f4a7c15ull;

static uint32_t next(void) {
    rng ^= rng << 13;
    rng ^= rng >> 7;
    rng ^= rng << 17;
    return (uint32_t)(rng >> 32);
}

static int pack(const char *name, size_t tlen, const OmegaCubinParam *p, size_t np, uint32_t cbank, uint32_t gpr,
                uint32_t shared) {
    uint8_t *out = NULL;
    size_t len = 0;
    char err[256];
    int rc = omega_cubin_pack(name, name == NULL && tlen == 1 ? NULL : text, tlen, p, np, cbank, gpr, shared, &out,
                              &len, err, sizeof err);
    calls++;
    if (rc < 0 || rc > 7) {
        fprintf(stderr, "FAIL: return code %d\n", rc);
        exit(1);
    }
    by_rc[rc]++;
    if (rc == OMEGA_CUBIN_OK) {
        if (!out || len < 64 || memcmp(out, "\177ELF", 4) != 0) {
            fprintf(stderr, "FAIL: image without an ELF header\n");
            exit(1);
        }
        image_bytes += len;
    } else if (out || len || !err[0]) {
        fprintf(stderr, "FAIL: refusal %d left an image or no message\n", rc);
        exit(1);
    }
    omega_cubin_free(out);
    return rc;
}

/* n parameters of the sizes in sz (pointer when ptr bit i is set), packed in order with optional gaps. */
static size_t layout(OmegaCubinParam *p, size_t n, int gaps) {
    uint32_t end = OMEGA_CUBIN_PARAM_BASE;
    static const uint32_t sizes[4] = { 1, 2, 4, 8 };
    for (size_t i = 0; i < n; i++) {
        uint32_t s = sizes[next() % 4];
        uint32_t off = (end + s - 1) & ~(s - 1);
        if (gaps && next() % 3 == 0) off += s * (next() % 3);
        p[i].offset = off;
        p[i].size = s;
        p[i].is_pointer = (uint8_t)(s == 8 && next() % 2 ? 1 + next() % 255 : 0);
        end = off + s;
    }
    return end;
}

int main(void) {
    OmegaCubinParam p[40];
    char name[80];
    for (size_t i = 0; i < sizeof text; i++) text[i] = (uint8_t)next();

    /* Refusal classes, in the C order. */
    {
        uint8_t *out = NULL;
        size_t len = 0;
        char err[64];
        if (omega_cubin_pack("k", text, 16, NULL, 0, 1024, 8, 0, NULL, &len, err, sizeof err) != OMEGA_CUBIN_E_ARG ||
            omega_cubin_pack("k", text, 16, NULL, 0, 1024, 8, 0, &out, NULL, err, sizeof err) != OMEGA_CUBIN_E_ARG) {
            fprintf(stderr, "FAIL: null out accepted\n");
            return 1;
        }
        calls += 2;
        by_rc[OMEGA_CUBIN_E_ARG] += 2;
    }
    pack(NULL, 16, NULL, 0, 1024, 8, 0);
    pack(NULL, 1, NULL, 0, 1024, 8, 0); /* null text */
    static const char *bad[] = { "", "9k", "k-1", "k 1", "\xc3\xa9k", "_\x7f", "a.b" };
    for (size_t i = 0; i < sizeof bad / sizeof bad[0]; i++) pack(bad[i], 16, NULL, 0, 1024, 8, 0);
    static const size_t tbad[] = { 0, 1, 15, 17, 24, 33, (1u << 24) + 16, (size_t)1 << 32 };
    for (size_t i = 0; i < sizeof tbad / sizeof tbad[0]; i++) pack("k", tbad[i], NULL, 0, 1024, 8, 0);
    for (size_t n = 33; n <= 40; n++) pack("k", 16, p, n, 1024, 8, 0);
    pack("k", 16, NULL, 0, 1024, 256, 0);
    pack("k", 16, NULL, 0, 1024, 0xffffffffu, 0);
    for (uint32_t s = 1; s < 4; s++) pack("k", 16, NULL, 0, 1024, 8, s);
    pack("k", 16, NULL, 0, 1024, 8, 0xfffffffeu);
    /* parameter refusals: overlap, unsorted, below the base, bad size, short pointer, misaligned */
    for (int k = 0; k < 4; k++) {
        layout(p, 4, 0);
        p[3].offset = p[2].offset + p[2].size - 1;
        p[3].size = 1;
        p[3].is_pointer = 0;
        pack("k", 16, p, 4, 1024, 8, 0);
    }
    p[0] = (OmegaCubinParam){ 0x37c, 4, 0 };
    pack("k", 16, p, 1, 1024, 8, 0);
    for (uint32_t s = 0; s <= 16; s++) {
        p[0] = (OmegaCubinParam){ 0x380, s, 0 };
        pack("k", 16, p, 1, 1024, 8, 0);
        p[0].is_pointer = 1;
        pack("k", 16, p, 1, 1024, 8, 0);
    }
    for (uint32_t s = 2; s <= 8; s *= 2)
        for (uint32_t o = 1; o < s; o++) {
            p[0] = (OmegaCubinParam){ 0x380 + o, s, 0 };
            pack("k", 16, p, 1, 1024, 8, 0);
        }
    /* bank: parameter end beyond the bank, bank outside 0x380..0x400 */
    p[0] = (OmegaCubinParam){ 0x3f8, 8, 1 };
    pack("k", 16, p, 1, 0x3f4, 8, 0);
    p[0] = (OmegaCubinParam){ 0xfffffff8u, 8, 0 };
    pack("k", 16, p, 1, 1024, 8, 0);
    static const uint32_t cbad[] = { 0, 0x37f, 0x401, 0x800, 0xffffffffu };
    for (size_t i = 0; i < sizeof cbad / sizeof cbad[0]; i++) pack("k", 16, NULL, 0, cbad[i], 8, 0);

    /* Images. Parameter counts 0..32, three layouts each, the tight bank and the full one. */
    for (size_t n = 0; n <= OMEGA_CUBIN_MAX_PARAMS; n++)
        for (int k = 0; k < 3; k++) {
            uint32_t end = (uint32_t)layout(p, n, k);
            if (end <= 1024) {
                pack("kernel_sweep", 16 * (1 + next() % 64), p, n, 1024, next() % 256, 4 * (next() % 1024));
                if (end >= 0x380) pack("kernel_sweep", 32, p, n, end, 32, 0);
            } else {
                pack("kernel_sweep", 32, p, n, 1024, 32, 0);
            }
        }
    /* Name lengths 1..65: 64 is the longest name, 65 is refused. */
    for (size_t l = 1; l <= OMEGA_CUBIN_MAX_NAME + 1; l++) {
        for (size_t i = 0; i < l; i++) name[i] = "_aZz09Kq"[(i + l) % 8];
        name[0] = l % 2 ? '_' : 'k';
        name[l] = 0;
        size_t n = l % 6;
        layout(p, n, 1);
        pack(name, 16 * l, p, n, 1024, (uint32_t)l, 0);
    }
    /* Text lengths from 16: every multiple of 16 to 4096, then powers of two to 1 MiB. */
    for (size_t t = 16; t <= 4096; t += 16) pack("text_len", t, NULL, 0, 0x380, 1, 0);
    for (size_t t = 8192; t <= sizeof text; t *= 2) pack("text_len", t, NULL, 0, 0x400, 1, 0);
    /* gpr 0..255 */
    for (uint32_t g = 0; g <= 255; g++) {
        layout(p, 3, 0);
        pack("gpr", 16, p, 3, 1024, g, g * 4);
    }
    /* shared reservations, large ones included */
    static const uint32_t shared[] = { 0, 4, 8, 252, 4096, 49152, 65536, 1u << 20, 100u << 20 };
    for (size_t i = 0; i < sizeof shared / sizeof shared[0]; i++) pack("shared", 64, NULL, 0, 1024, 16, shared[i]);

    static const char *names[8] = { "OK", "ARG", "TEXT", "PARAMS", "CBANK", "GPR", "SHARED", "MEM" };
    printf("cubin sweep: %llu calls, each packed by cubin_elf.om and the C writer and compared;", calls);
    for (int r = 0; r < 8; r++) printf(" %s %llu", names[r], by_rc[r]);
    printf("; %llu image bytes\n", image_bytes);
    for (int r = 0; r < 7; r++)
        if (by_rc[r] == 0) {
            fprintf(stderr, "FAIL: the sweep reached no %s case\n", names[r]);
            return 1;
        }
    if (by_rc[OMEGA_CUBIN_OK] < 400) {
        fprintf(stderr, "FAIL: only %llu images\n", by_rc[OMEGA_CUBIN_OK]);
        return 1;
    }
    return 0;
}
