/* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception */
/* Simulated RM/UVM boundary, not device evidence. Never opens a driver node.
 * Compile nvrm.c into this TU to exercise its real allocation state machine. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <assert.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <fcntl.h>
#include <unistd.h>
static int sim_ioctl(int, unsigned long, ...);
static int sim_open(const char *, int, ...);
static int sim_close(int);
static void *sim_mmap(void *, size_t, int, int, int, off_t);
#define ioctl sim_ioctl
#define open sim_open
#define close sim_close
#define mmap sim_mmap
#include "../nvrm/nvrm.c"
#undef ioctl
#undef open
#undef close
#undef mmap
static unsigned fail_stage, allocs, frees, dma_maps, dma_unmaps, ranges, range_frees, maps;
static int sim_open(const char *path, int flags, ...) {
    (void)flags;
    assert(!strncmp(path, "/dev/nvidia", 11));
    return 700; /* fake descriptor, not an OS/driver open */
}
static int sim_close(int fd) { assert(fd == 700); return 0; }
static void *sim_mmap(void *va, size_t size, int prot, int flags, int fd, off_t offset) {
    if (fd == 700) return mmap(va, size, prot, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
    return mmap(va, size, prot, flags, fd, offset);
}
static int sim_ioctl(int fd, unsigned long request, ...) {
    (void)fd;
    va_list ap; va_start(ap, request); void *arg = va_arg(ap, void *); va_end(ap);
    if (request == UVM_CREATE_EXTERNAL_RANGE) {
        UVM_CREATE_EXTERNAL_RANGE_PARAMS *p = arg;
        if (fail_stage == 3) p->rmStatus = NV_ERR_NO_MEMORY;
        else ranges++;
        return 0;
    }
    if (request == UVM_MAP_EXTERNAL_ALLOCATION) {
        UVM_MAP_EXTERNAL_ALLOCATION_PARAMS *p = arg;
        if (fail_stage == 5) p->rmStatus = NV_ERR_NO_MEMORY;
        return 0;
    }
    if (request == UVM_FREE) { range_frees++; return 0; }
    assert(_IOC_TYPE(request) == NV_IOCTL_MAGIC);
    switch (_IOC_NR(request)) {
    case NV_ESC_RM_ALLOC: {
        NVOS21_PARAMETERS *p = arg;
        NV_MEMORY_ALLOCATION_PARAMS *mp = (void *)(uintptr_t)p->pAllocParms;
        assert(p->hClass == NV01_MEMORY_LOCAL_USER);
        assert(((mp->attr >> 25) & 3u) == NVOS32_ATTR_LOCATION_VIDMEM);
        assert(mp->size == 4096);
        allocs++;
        if (fail_stage == 1) p->status = NV_ERR_NO_MEMORY;
        return 0;
    }
    case NV_ESC_REGISTER_FD: return 0;
    case NV_ESC_RM_MAP_MEMORY: {
        nv_ioctl_nvos33_parameters_with_fd *p = arg;
        maps++;
        assert(p->fd == 700 && p->params.length == 4096);
        if (fail_stage == 2) p->params.status = NV_ERR_NO_MEMORY;
        return 0;
    }
    case NV_ESC_RM_MAP_MEMORY_DMA: {
        NVOS46_PARAMETERS *p = arg;
        if (fail_stage == 4) p->status = NV_ERR_NO_MEMORY;
        else dma_maps++;
        return 0;
    }
    case NV_ESC_RM_UNMAP_MEMORY_DMA: dma_unmaps++; return 0;
    case NV_ESC_RM_FREE: frees++; return 0;
    case NV_ESC_RM_VID_HEAP_CONTROL: {
        NVOS32_PARAMETERS *p = arg;
        assert(p->function == NVOS32_FUNCTION_INFO);
        assert(p->data.Info.attr == (NVOS32_ATTR_LOCATION_VIDMEM << 25));
        p->total = 8ull << 30; p->free = 4096;
        return 0;
    }
    default: assert(!"unexpected simulated driver operation"); return -1;
    }
}
static void fresh(Nvrm *rm) {
    memset(rm, 0, sizeof *rm);
    rm->target_sm = 86; rm->discrete = 1;
    rm->root = 1; rm->device = 2; rm->virtmem = 3; rm->next_handle = 100;
    rm->fd_ctl = rm->fd_uvm = 700;
    rm->va_next = rm->va_slot_base = acquire_va_slot();
    allocs = frees = dma_maps = dma_unmaps = ranges = range_frees = maps = 0;
}
static void class_and_bdf(void) {
    static Nvrm rm;
    const uint32_t amp[] = {OMEGA_AMPERE_COMPUTE_B, OMEGA_AMPERE_GPFIFO_A, VOLTA_USERMODE_A};
    const uint32_t bw[] = {BLACKWELL_COMPUTE_B, BLACKWELL_CHANNEL_GPFIFO_A, HOPPER_USERMODE_A};
    rm.sm_version = NV2080_CTRL_GR_INFO_SM_VERSION_8_06;
    assert(select_classes(&rm, amp, 3) == 0 && rm.target_sm == 86 && rm.discrete);
    assert(rm.compute_class == OMEGA_AMPERE_COMPUTE_B && rm.gpfifo_class == OMEGA_AMPERE_GPFIFO_A);
    assert(select_classes(&rm, amp, 2) != 0); /* missing usermode: no stale class reuse */
    assert(select_classes(&rm, bw, 3) != 0); /* a Blackwell class set cannot authorize SM86 */
    rm.sm_version = NV2080_CTRL_GR_INFO_SM_VERSION_12_01;
    assert(select_classes(&rm, bw, 3) == 0 && rm.target_sm == 121 && !rm.discrete);
    rm.sm_version = NV2080_CTRL_GR_INFO_SM_VERSION_8_09;
    assert(select_classes(&rm, amp, 3) != 0); /* Ada is not silently treated as GA10x */
    nv_ioctl_card_info_t cards[64] = {0};
    cards[0].valid = cards[1].valid = 1;
    cards[1].pci_info.bus = 1; cards[1].pci_info.slot = 2;
    assert(card_index(cards, NULL) == 0);
    assert(card_index(cards, "0000:01:02.0") == 1);
    assert(card_index(cards, "0000:00:00.0") == 0);
    assert(card_index(cards, "0:1:2.0") == -1);
    assert(card_index(cards, "0000:01:02.0junk") == -1);
    assert(card_index(cards, "0000:01:03.0") == -1);
}
static void allocation_lifetime(void) {
    static Nvrm rm;
    fresh(&rm); fail_stage = 0;
    NvrmMem mem = {0};
    assert(nvrm_alloc_vram(&rm, 1, &mem) == 0);
    assert(mem.size == 4096 && rm.live_count == 1 && allocs == 1 && maps == 1);
    const uint8_t in[] = {1, 0, 255, 3}; uint8_t out[sizeof in] = {0};
    assert(nvrm_copy_to(&rm, &mem, 7, in, sizeof in) == 0);
    assert(nvrm_copy_from(&rm, &mem, 7, out, sizeof out) == 0);
    assert(!memcmp(in, out, sizeof in));
    assert(nvrm_copy_to(&rm, &mem, 4095, in, sizeof in) != 0);
    NvrmMem forged = mem; forged.handle++;
    assert(nvrm_copy_from(&rm, &forged, 0, out, sizeof out) != 0);
    uint64_t free_bytes, total;
    assert(nvrm_vram_info(&rm, &free_bytes, &total) == 0 && free_bytes == 4096 && total == (8ull << 30));
    NvrmMem stale = mem;
    rm.inject_rm_free_failure = 1;
    assert(nvrm_free(&rm, &mem) != 0 && rm.live_count == 1 && rm.live[0].quarantined);
    assert(nvrm_copy_to(&rm, &mem, 0, in, sizeof in) != 0);
    assert(nvrm_free(&rm, &mem) == 0 && rm.live_count == 0);
    assert(nvrm_free(&rm, &mem) == 0); /* idempotent */
    assert(frees == 1 && dma_unmaps == 1 && range_frees == 1);
    assert(nvrm_free(&rm, &stale) != 0 && frees == 1);
}
static void failed_allocation_cleanup(void) {
    static Nvrm rm;
    for (fail_stage = 1; fail_stage <= 5; fail_stage++) {
        fresh(&rm);
        NvrmMem mem = {0};
        assert(nvrm_alloc_vram(&rm, 4096, &mem) != 0);
        assert(!mem.handle && rm.live_count == 0 && !rm.faulted);
        assert(allocs == 1 && frees == (fail_stage == 1 ? 0u : 1u));
        assert(dma_unmaps == dma_maps && range_frees == ranges);
    }
    fail_stage = 0;
}
static void qmd_and_submission_admission(void) {
    uint32_t q[64];
    assert(omega_ampere_build_qmd(q, 0x100000, 0x200000, 0x300000, 32, 1, 2, 1, 64, 0) == 0);
    /* Vendor clc7c0qmd.h QMDV03_00: major[583:580], minor[579:576],
     * CB0[1055:1024], program[1567:1536], release0 payload[863:832]. */
    assert(((q[18] >> 4) & 15) == 3 && (q[18] & 15) == 0);
    assert(q[32] == 0x200000 && q[48] == 0x100000 && q[26] == 1);
    assert(omega_ampere_build_qmd(q, 0x100001, 0x200000, 0x300000, 32, 1, 2, 1, 64, 0) != 0);
    assert(omega_ampere_build_qmd(q, 0x100000, 0x200000, 0x300000, 1024, 2, 2, 1, 64, 0) != 0);
    assert(omega_ampere_build_qmd(q, 0x100000, 0x200000, 0x300000, 32, 1, 1, 1, 64, 1) == 0);
    assert((q[17] & 0x3ffffu) == 256); /* shared memory is rounded to vendor granularity */
    assert(omega_ampere_build_qmd(q, 0x100000, 0x200000, 0x300000, 32, 1, 1, 1, 64, 99u << 10) == 0);
    assert(omega_ampere_build_qmd(q, 0x100000, 0x200000, 0x300000, 32, 1, 1, 1, 64, 100u << 10) != 0);
    assert(omega_ampere_build_qmd(q, 0x100000, 0x200000, 0x300001, 32, 1, 1, 1, 64, 0) != 0);
    static Nvrm rm; fresh(&rm);
    uint32_t good[] = {nvrm_mthd(1, 0, 1), OMEGA_AMPERE_COMPUTE_B,
        nvrm_mthd(1, 0x021c, 1), (1u << 0) | (1u << 4) | (1u << 12),
        nvrm_mthd(1, 0x02b4, 1), 0x1000, nvrm_mthd(1, 0x02c0, 1), 3};
    NvrmMem pb = {.cpu = good, .size = sizeof good};
    assert(validate_ampere_push(&rm, &pb, 0, sizeof good / sizeof good[0]) == 0);
    good[3] = 0; /* The supported cache method does not admit arbitrary values. */
    assert(validate_ampere_push(&rm, &pb, 0, sizeof good / sizeof good[0]) != 0);
    good[3] = (1u << 0) | (1u << 4) | (1u << 12);
    good[1] = BLACKWELL_COMPUTE_B;
    assert(validate_ampere_push(&rm, &pb, 0, sizeof good / sizeof good[0]) != 0);
    good[0] = 0x206220c6; /* existing Blackwell inline QMD5 packet */
    assert(validate_ampere_push(&rm, &pb, 0, sizeof good / sizeof good[0]) != 0);
    assert(allocs == 0 && frees == 0); /* admission itself makes no ioctl */
}
int main(void) {
    class_and_bdf(); allocation_lifetime(); failed_allocation_cleanup(); qmd_and_submission_admission();
    puts("Ampere simulated RM allocation/class/QMD tests passed (HOST_ONLY; device NOT_RUN)");
    return 0;
}
