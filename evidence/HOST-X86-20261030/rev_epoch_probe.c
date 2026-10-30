#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
/* declarations */
uint32_t omega_native_revision(uint64_t *result, uint64_t o_previous);
uint32_t omega_native_epoch(uint64_t *result, uint64_t o_previous);
static void show(const char *n, uint32_t st, uint64_t r) {
    printf("%s st=%u r=%" PRIu64 "\n", n, st, r);
}
int main(void) {
    uint64_t r = 42;
    show("rev0", omega_native_revision(&r, 0), r);
    show("rev1", omega_native_revision(&r, 1), r);
    show("revmax-1", omega_native_revision(&r, UINT64_MAX - 1), r);
    show("revmax", omega_native_revision(&r, UINT64_MAX), r);
    uint64_t *bad = (uint64_t *)(uintptr_t)1;
    printf("revmis st=%u\n", omega_native_revision(bad, 0));
    printf("revnull st=%u\n", omega_native_revision(0, 0));
    r = 7;
    show("ep0", omega_native_epoch(&r, 0), r);
    show("ep3", omega_native_epoch(&r, 3), r);
    show("epmax-1", omega_native_epoch(&r, UINT64_MAX - 1), r);
    show("epmax", omega_native_epoch(&r, UINT64_MAX), r);
    printf("epnull st=%u\n", omega_native_epoch(0, 0));
    return 0;
}
