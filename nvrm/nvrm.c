/*
 * nvrm.c -- PHYSICS native accelerator substrate (NVRM client) (shared by M16 qualification and M17).
 * See nvrm.h. Sequence follows the NVIDIA open kernel module ABI (580.173.02).
 */

#include "nvrm.h"

#include <errno.h>
#include <stddef.h>
#include <stdatomic.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#include "nvos.h"
#include "nv-ioctl.h"
#include "nv-ioctl-numbers.h"
#include "nv_escape.h"
#include "uvm_linux_ioctl.h"
#include "uvm_ioctl.h"
#include "nv-unix-nvos-params-wrappers.h"
#include "class/clc361.h"
#include "class/cl0000.h"
#include "class/cl0080.h"
#include "class/cl2080.h"
#include "class/cl0070.h"
#include "class/cl003e.h"
#include "class/cl0040.h"
#include "class/cl90f1.h"
#include "class/cl9067.h"
#include "class/cla06c.h"
#include "class/clc96f.h"
#include "class/clcec0.h"
#include "class/clc661.h"
#include "alloc/alloc_channel.h"
#include "ctrl/ctrl0000/ctrl0000gpu.h"
#include "ctrl/ctrl0080/ctrl0080gpu.h"
#include "ctrl/ctrl2080/ctrl2080gpu.h"
#include "ctrl/ctrl2080/ctrl2080gr.h"
#include "ctrl/ctrla06c.h"
#include "ctrl/ctrlc36f.h"

#define NV_IOWR(nr, sz) _IOC(_IOC_READ | _IOC_WRITE, NV_IOCTL_MAGIC, (nr), (sz))

static int fail(Nvrm *rm, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(rm->err, sizeof rm->err, fmt, ap);
    va_end(ap);
    return -1;
}

static int nv_ioctl(int fd, unsigned nr, void *arg, size_t sz) {
    return ioctl(fd, NV_IOWR(nr, sz), arg);
}

static int uvm(Nvrm *rm, int fd, unsigned long cmd, void *params, NV_STATUS *st);

static int rm_alloc(Nvrm *rm, uint32_t parent, uint32_t cls, void *params, uint32_t psz, uint32_t *out) {
    if (rm->rm_alloc_accepted == UINT64_MAX ||
        (cls != NV01_ROOT_CLIENT && rm->next_handle == UINT32_MAX))
        return fail(rm, "RM handle or accounting space exhausted");
    NVOS21_PARAMETERS p;
    memset(&p, 0, sizeof p);
    p.hRoot = rm->root;
    p.hObjectParent = parent;
    p.hObjectNew = (cls == NV01_ROOT_CLIENT) ? 0 : rm->next_handle++;
    p.hClass = cls;
    p.pAllocParms = NV_PTR_TO_NvP64(params);
    p.paramsSize = psz;
    if (nv_ioctl(rm->fd_ctl, NV_ESC_RM_ALLOC, &p, sizeof p) != 0)
        return fail(rm, "RM_ALLOC class 0x%x ioctl errno %d", cls, errno);
    if (p.status != 0) return fail(rm, "RM_ALLOC class 0x%x status 0x%x", cls, p.status);
    rm->rm_alloc_accepted++;
    *out = p.hObjectNew;
    return 0;
}

static int rm_free(Nvrm *rm, uint32_t parent, uint32_t obj) {
    if (!obj) return 0;
    if (rm->rm_free_accepted == UINT64_MAX) return fail(rm, "RM free accounting exhausted");
    if (rm->inject_rm_free_failure) {
        rm->inject_rm_free_failure = 0;
        return fail(rm, "injected RM_FREE failure for handle 0x%x", obj);
    }
    NVOS00_PARAMETERS f;
    memset(&f, 0, sizeof f);
    f.hRoot = rm->root;
    f.hObjectParent = parent;
    f.hObjectOld = obj;
    if (nv_ioctl(rm->fd_ctl, NV_ESC_RM_FREE, &f, sizeof f) != 0)
        return fail(rm, "RM_FREE handle 0x%x errno %d", obj, errno);
    if (f.status != NV_OK) return fail(rm, "RM_FREE handle 0x%x status 0x%x", obj, f.status);
    rm->rm_free_accepted++;
    return 0;
}

static int uvm_free_checked(Nvrm *rm, uint64_t va, uint64_t size) {
    if (rm->inject_uvm_free_failure) {
        rm->inject_uvm_free_failure = 0;
        return fail(rm, "injected UVM_FREE failure for VA 0x%lx", (unsigned long)va);
    }
    UVM_FREE_PARAMS uf;
    memset(&uf, 0, sizeof uf);
    uf.base = va;
    uf.length = size;
    return uvm(rm, rm->fd_uvm, UVM_FREE, &uf, &uf.rmStatus);
}

static int dma_unmap_checked(Nvrm *rm, uint32_t hmem, uint64_t va, uint64_t size) {
    NVOS47_PARAMETERS p;
    memset(&p, 0, sizeof p);
    p.hClient = rm->root;
    p.hDevice = rm->device;
    p.hDma = rm->virtmem;
    p.hMemory = hmem;
    p.dmaOffset = va;
    p.size = size;
    if (nv_ioctl(rm->fd_ctl, NV_ESC_RM_UNMAP_MEMORY_DMA, &p, sizeof p) != 0)
        return fail(rm, "RM_UNMAP_MEMORY_DMA errno %d", errno);
    if (p.status != NV_OK) return fail(rm, "RM_UNMAP_MEMORY_DMA status 0x%x", p.status);
    return 0;
}

static int rm_control(Nvrm *rm, uint32_t obj, uint32_t cmd, void *params, uint32_t psz) {
    NVOS54_PARAMETERS p;
    memset(&p, 0, sizeof p);
    p.hClient = rm->root;
    p.hObject = obj;
    p.cmd = cmd;
    p.params = NV_PTR_TO_NvP64(params);
    p.paramsSize = psz;
    if (nv_ioctl(rm->fd_ctl, NV_ESC_RM_CONTROL, &p, sizeof p) != 0)
        return fail(rm, "RM_CONTROL 0x%x ioctl errno %d", cmd, errno);
    if (p.status != 0) return fail(rm, "RM_CONTROL 0x%x status 0x%x", cmd, p.status);
    return 0;
}

static int uvm(Nvrm *rm, int fd, unsigned long cmd, void *params, NV_STATUS *st) {
    if (ioctl(fd, cmd, params) != 0) return fail(rm, "UVM 0x%lx errno %d", cmd, errno);
    if (st && *st != NV_OK) return fail(rm, "UVM 0x%lx status 0x%x", cmd, *st);
    return 0;
}

static int open_dev_fd(Nvrm *rm) {
    char path[32];
    snprintf(path, sizeof path, "/dev/nvidia%u", rm->minor);
    int fd = open(path, O_RDWR | O_CLOEXEC);
    if (fd < 0) return fail(rm, "open %s errno %d", path, errno);
    nv_ioctl_register_fd_t r = { .ctl_fd = rm->fd_ctl };
    if (nv_ioctl(fd, NV_ESC_REGISTER_FD, &r, sizeof r) != 0) { close(fd); return fail(rm, "REGISTER_FD errno %d", errno); }
    return fd;
}

/* Map an RM memory object into this process at a fixed CPU address. */
static void *map_to_cpu(Nvrm *rm, uint32_t hmem, uint64_t size, void *target, uint32_t flags, int system) {
    int fd = system ? open("/dev/nvidiactl", O_RDWR | O_CLOEXEC) : open_dev_fd(rm);
    if (fd < 0) { fail(rm, "map fd"); return MAP_FAILED; }
    nv_ioctl_nvos33_parameters_with_fd m;
    memset(&m, 0, sizeof m);
    m.fd = fd;
    m.params.hClient = rm->root;
    m.params.hDevice = rm->device;
    m.params.hMemory = hmem;
    m.params.length = size;
    m.params.flags = flags;
    if (nv_ioctl(rm->fd_ctl, NV_ESC_RM_MAP_MEMORY, &m, sizeof m) != 0 || m.params.status != 0) {
        fail(rm, "RM_MAP_MEMORY errno %d status 0x%x", errno, m.params.status);
        close(fd);
        return MAP_FAILED;
    }
    void *p = mmap(target, size, PROT_READ | PROT_WRITE, MAP_SHARED | (target ? MAP_FIXED : 0), fd, 0);
    close(fd);
    if (p == MAP_FAILED) fail(rm, "mmap errno %d", errno);
    return p;
}

static _Atomic uint64_t g_va_counter = 0x1000000000ull;
static atomic_flag g_va_slots_lock = ATOMIC_FLAG_INIT;
static uint64_t g_va_recycled[1024];
static uint32_t g_va_recycled_count;

static uint64_t acquire_va_slot(void) {
    while (atomic_flag_test_and_set_explicit(&g_va_slots_lock, memory_order_acquire)) { }
    uint64_t base = g_va_recycled_count ? g_va_recycled[--g_va_recycled_count] : 0;
    atomic_flag_clear_explicit(&g_va_slots_lock, memory_order_release);
    if (!base) {
        uint64_t old = atomic_load_explicit(&g_va_counter, memory_order_relaxed);
        for (;;) {
            if (old > UINT64_MAX - 0x100000000ull) return 0;
            if (atomic_compare_exchange_weak_explicit(&g_va_counter, &old,
                    old + 0x100000000ull, memory_order_relaxed, memory_order_relaxed)) {
                base = old;
                break;
            }
        }
    }
    return base;
}

static void recycle_va_slot(uint64_t base) {
    while (atomic_flag_test_and_set_explicit(&g_va_slots_lock, memory_order_acquire)) { }
    if (g_va_recycled_count < 1024) g_va_recycled[g_va_recycled_count++] = base;
    atomic_flag_clear_explicit(&g_va_slots_lock, memory_order_release);
}

int nvrm_open(Nvrm *rm) {
    memset(rm, 0, sizeof *rm);
    rm->next_handle = 0xcf000001u;
    rm->va_next = acquire_va_slot();
    if (!rm->va_next) return fail(rm, "VA slot space exhausted");
    rm->va_slot_base = rm->va_next;
    rm->fd_ctl = open("/dev/nvidiactl", O_RDWR | O_CLOEXEC);
    rm->fd_uvm = open("/dev/nvidia-uvm", O_RDWR | O_CLOEXEC);
    rm->fd_uvm2 = open("/dev/nvidia-uvm", O_RDWR | O_CLOEXEC);
    if (rm->fd_ctl < 0 || rm->fd_uvm < 0 || rm->fd_uvm2 < 0) return fail(rm, "open control nodes errno %d", errno);

    if (rm_alloc(rm, 0, NV01_ROOT_CLIENT, NULL, 0, &rm->root)) return -1;

    UVM_INITIALIZE_PARAMS ui;
    memset(&ui, 0, sizeof ui);
    if (uvm(rm, rm->fd_uvm, UVM_INITIALIZE, &ui, &ui.rmStatus)) return -1;
    UVM_MM_INITIALIZE_PARAMS mi;
    memset(&mi, 0, sizeof mi);
    mi.uvmFd = rm->fd_uvm;
    (void)ioctl(rm->fd_uvm2, UVM_MM_INITIALIZE, &mi); /* libcuda also tolerates failure here */

    nv_ioctl_card_info_t cards[64];
    memset(cards, 0, sizeof cards);
    if (nv_ioctl(rm->fd_ctl, NV_ESC_CARD_INFO, cards, sizeof cards) != 0) return fail(rm, "CARD_INFO errno %d", errno);
    int found = -1;
    for (int i = 0; i < 64; i++) if (cards[i].valid) { found = i; break; }
    if (found < 0) return fail(rm, "no GPU");
    rm->gpu_id = cards[found].gpu_id;
    rm->minor = cards[found].minor_number;

    rm->fd_dev = open_dev_fd(rm);
    if (rm->fd_dev < 0) return -1;

    NV0000_CTRL_GPU_GET_ID_INFO_V2_PARAMS idinfo;
    memset(&idinfo, 0, sizeof idinfo);
    idinfo.gpuId = rm->gpu_id;
    if (rm_control(rm, rm->root, NV0000_CTRL_CMD_GPU_GET_ID_INFO_V2, &idinfo, sizeof idinfo)) return -1;

    NV0080_ALLOC_PARAMETERS dp;
    memset(&dp, 0, sizeof dp);
    dp.deviceId = idinfo.deviceInstance;
    dp.hClientShare = rm->root;
    dp.vaMode = NV_DEVICE_ALLOCATION_VAMODE_OPTIONAL_MULTIPLE_VASPACES;
    if (rm_alloc(rm, rm->root, NV01_DEVICE_0, &dp, sizeof dp, &rm->device)) return -1;
    NV2080_ALLOC_PARAMETERS sp;
    memset(&sp, 0, sizeof sp);
    if (rm_alloc(rm, rm->device, NV20_SUBDEVICE_0, &sp, sizeof sp, &rm->subdevice)) return -1;
    NV_MEMORY_VIRTUAL_ALLOCATION_PARAMS vp;
    memset(&vp, 0, sizeof vp);
    vp.limit = 0x1ffffffffffffull;
    if (rm_alloc(rm, rm->device, NV01_MEMORY_VIRTUAL, &vp, sizeof vp, &rm->virtmem)) return -1;

    /* Class list: pick the Blackwell classes that this RM exposes. */
    NV0080_CTRL_GPU_GET_CLASSLIST_PARAMS cl;
    memset(&cl, 0, sizeof cl);
    if (rm_control(rm, rm->device, NV0080_CTRL_CMD_GPU_GET_CLASSLIST, &cl, sizeof cl)) return -1;
    uint32_t classes[512];
    if (cl.numClasses > 512) return fail(rm, "too many classes");
    cl.classList = NV_PTR_TO_NvP64(classes);
    if (rm_control(rm, rm->device, NV0080_CTRL_CMD_GPU_GET_CLASSLIST, &cl, sizeof cl)) return -1;
    for (uint32_t i = 0; i < cl.numClasses; i++) {
        if (classes[i] == BLACKWELL_COMPUTE_B) rm->compute_class = classes[i];
        if (classes[i] == BLACKWELL_CHANNEL_GPFIFO_A) rm->gpfifo_class = classes[i];
        if (classes[i] == HOPPER_USERMODE_A) rm->usermode_class = classes[i];
    }
    if (!rm->compute_class || !rm->gpfifo_class || !rm->usermode_class)
        return fail(rm, "missing class: compute 0x%x gpfifo 0x%x usermode 0x%x",
                    rm->compute_class, rm->gpfifo_class, rm->usermode_class);

    if (rm_alloc(rm, rm->subdevice, rm->usermode_class, NULL, 0, &rm->usermode)) return -1;
    void *mmio = map_to_cpu(rm, rm->usermode, 0x10000, NULL, 0, 0);
    if (mmio == MAP_FAILED) return -1;
    rm->usermode_cpu = mmio;
    rm->doorbell = (volatile uint32_t *)((uint8_t *)mmio + NVC361_NOTIFY_CHANNEL_PENDING);

    NV2080_CTRL_GR_INFO gi = { .index = NV2080_CTRL_GR_INFO_INDEX_SM_VERSION };
    NV2080_CTRL_GR_GET_INFO_PARAMS gp;
    memset(&gp, 0, sizeof gp);
    gp.grInfoListSize = 1;
    gp.grInfoList = NV_PTR_TO_NvP64(&gi);
    if (rm_control(rm, rm->subdevice, NV2080_CTRL_CMD_GR_GET_INFO, &gp, sizeof gp)) return -1;
    rm->sm_version = gi.data;

    NV_VASPACE_ALLOCATION_PARAMETERS vap;
    memset(&vap, 0, sizeof vap);
    vap.vaBase = 0x1000;
    vap.vaSize = 0x1fffffb000000ull;
    vap.flags = NV_VASPACE_ALLOCATION_FLAGS_ENABLE_PAGE_FAULTING | NV_VASPACE_ALLOCATION_FLAGS_IS_EXTERNALLY_OWNED;
    if (rm_alloc(rm, rm->device, FERMI_VASPACE_A, &vap, sizeof vap, &rm->vaspace)) return -1;

    NV2080_CTRL_GPU_GET_GID_INFO_PARAMS gid;
    memset(&gid, 0, sizeof gid);
    gid.flags = NV2080_GPU_CMD_GPU_GET_GID_FLAGS_FORMAT_BINARY;
    gid.length = 16;
    if (rm_control(rm, rm->subdevice, NV2080_CTRL_CMD_GPU_GET_GID_INFO, &gid, sizeof gid)) return -1;
    memcpy(rm->gpu_uuid, gid.data, 16);

    UVM_REGISTER_GPU_PARAMS rg;
    memset(&rg, 0, sizeof rg);
    memcpy(rg.gpu_uuid.uuid, rm->gpu_uuid, 16);
    rg.rmCtrlFd = -1;
    if (uvm(rm, rm->fd_uvm, UVM_REGISTER_GPU, &rg, &rg.rmStatus)) return -1;
    rm->gpu_registered = 1;
    UVM_REGISTER_GPU_VASPACE_PARAMS rv;
    memset(&rv, 0, sizeof rv);
    memcpy(rv.gpuUuid.uuid, rm->gpu_uuid, 16);
    rv.rmCtrlFd = rm->fd_ctl;
    rv.hClient = rm->root;
    rv.hVaSpace = rm->vaspace;
    if (uvm(rm, rm->fd_uvm, UVM_REGISTER_GPU_VASPACE, &rv, &rv.rmStatus)) return -1;
    rm->vas_registered = 1;
    return 0;
}

/* Return a freed VA range to the sorted, coalesced free list. */
static int va_release(Nvrm *rm, uint64_t va, uint64_t size) {
    uint64_t end;
    if (!size || __builtin_add_overflow(va, size, &end)) {
        rm->faulted = 1;
        return fail(rm, "invalid VA release extent");
    }
    uint32_t pos = 0;
    while (pos < rm->free_count && rm->free_list[pos].va < va) pos++;
    if ((pos && rm->free_list[pos - 1].va + rm->free_list[pos - 1].size > va) ||
        (pos < rm->free_count && end > rm->free_list[pos].va)) {
        rm->faulted = 1;
        return fail(rm, "overlapping VA release extent");
    }

    if (rm->free_count >= NVRM_MAX_FREE) {
        /* Free-list table full: leak this VA range (address space only --
         * no RM object, CPU mapping, or UVM range remains associated with it). */
        rm->faulted = 1;
        return fail(rm, "VA free-list full; VA 0x%lx quarantined", (unsigned long)va);
    }
    for (uint32_t i = rm->free_count; i > pos; i--) rm->free_list[i] = rm->free_list[i - 1];
    rm->free_list[pos].va = va;
    rm->free_list[pos].size = size;
    rm->free_count++;

    /* Coalesce with the following entry. */
    if (pos + 1 < rm->free_count && rm->free_list[pos].va + rm->free_list[pos].size == rm->free_list[pos + 1].va) {
        rm->free_list[pos].size += rm->free_list[pos + 1].size;
        for (uint32_t i = pos + 1; i + 1 < rm->free_count; i++) rm->free_list[i] = rm->free_list[i + 1];
        rm->free_count--;
    }
    /* Coalesce with the preceding entry. */
    if (pos > 0 && rm->free_list[pos - 1].va + rm->free_list[pos - 1].size == rm->free_list[pos].va) {
        rm->free_list[pos - 1].size += rm->free_list[pos].size;
        for (uint32_t i = pos; i + 1 < rm->free_count; i++) rm->free_list[i] = rm->free_list[i + 1];
        rm->free_count--;
    }
    return 0;
}

static uint64_t va_take(Nvrm *rm, uint64_t size, uint64_t align) {
    /* First-fit reuse from the free list (sorted ascending by va), splitting
     * off any leading alignment pad or trailing remainder as smaller free
     * entries. Falls back to the bump allocator when nothing fits. */
    for (uint32_t i = 0; i < rm->free_count; i++) {
        uint64_t base = rm->free_list[i].va;
        uint64_t sz = rm->free_list[i].size;
        uint64_t rounded;
        uint64_t range_end;
        if (__builtin_add_overflow(base, align - 1, &rounded) ||
            __builtin_add_overflow(base, sz, &range_end)) return 0;
        uint64_t aligned = rounded & ~(align - 1);
        uint64_t pad = aligned - base;
        if (pad > sz || size > sz - pad) continue;
        uint64_t used_end;
        if (__builtin_add_overflow(aligned, size, &used_end)) return 0;

        if (pad == 0 && used_end == range_end) {
            /* Exact fit: remove entry i. */
            for (uint32_t j = i; j + 1 < rm->free_count; j++) rm->free_list[j] = rm->free_list[j + 1];
            rm->free_count--;
        } else if (pad == 0) {
            /* Consume from the front: shrink entry i. */
            rm->free_list[i].va = used_end;
            rm->free_list[i].size = range_end - used_end;
        } else {
            /* Leading pad remains free at entry i. */
            rm->free_list[i].size = pad;
            if (used_end < range_end && rm->free_count < NVRM_MAX_FREE) {
                for (uint32_t j = rm->free_count; j > i + 1; j--) rm->free_list[j] = rm->free_list[j - 1];
                rm->free_list[i + 1].va = used_end;
                rm->free_list[i + 1].size = range_end - used_end;
                rm->free_count++;
            }
            else if (used_end < range_end) {
                /* Preserve the range intact rather than silently losing VA. */
                rm->free_list[i].size = sz;
                continue;
            }
        }
        return aligned;
    }

    uint64_t rounded;
    if (__builtin_add_overflow(rm->va_next, align - 1, &rounded)) return 0;
    uint64_t va = rounded & ~(align - 1);
    if (__builtin_add_overflow(va, size, &rm->va_next)) return 0;
    return va;
}

static int uvm_map(Nvrm *rm, uint64_t va, uint64_t size, uint32_t hmem,
                   uint8_t *uvm_created, uint8_t *dma_mapped) {
    UVM_CREATE_EXTERNAL_RANGE_PARAMS cr;
    memset(&cr, 0, sizeof cr);
    cr.base = va;
    cr.length = size;
    if (uvm(rm, rm->fd_uvm, UVM_CREATE_EXTERNAL_RANGE, &cr, &cr.rmStatus)) return -1;
    *uvm_created = 1;

    NVOS46_PARAMETERS d;
    memset(&d, 0, sizeof d);
    d.hClient = rm->root;
    d.hDevice = rm->device;
    d.hDma = rm->virtmem;
    d.hMemory = hmem;
    d.length = size;
    d.flags = (NVOS46_FLAGS_PAGE_SIZE_4KB << 8) | (NVOS46_FLAGS_CACHE_SNOOP_ENABLE << 4) | (NVOS46_FLAGS_DMA_OFFSET_FIXED_TRUE << 15);
    d.dmaOffset = va;
    if (nv_ioctl(rm->fd_ctl, NV_ESC_RM_MAP_MEMORY_DMA, &d, sizeof d) != 0 || d.status != 0)
        return fail(rm, "RM_MAP_MEMORY_DMA errno %d status 0x%x", errno, d.status);
    if (d.dmaOffset != va) return fail(rm, "dmaOffset mismatch");
    *dma_mapped = 1;

    static UVM_MAP_EXTERNAL_ALLOCATION_PARAMS ma;
    memset(&ma, 0, sizeof ma);
    ma.base = va;
    ma.length = size;
    ma.rmCtrlFd = rm->fd_ctl;
    ma.hClient = rm->root;
    ma.hMemory = hmem;
    ma.gpuAttributesCount = 1;
    memcpy(ma.perGpuAttributes[0].gpuUuid.uuid, rm->gpu_uuid, 16);
    ma.perGpuAttributes[0].gpuMappingType = UvmGpuMappingTypeReadWriteAtomic;
    return uvm(rm, rm->fd_uvm, UVM_MAP_EXTERNAL_ALLOCATION, &ma, &ma.rmStatus);
}

static void quarantine_allocation(Nvrm *rm, uint32_t h, uint64_t va, uint64_t size,
                                  void *cpu, uint8_t uvm_live, uint8_t dma_live) {
    NvrmLiveAlloc *e = &rm->live[rm->live_count++];
    *e = (NvrmLiveAlloc){ .handle = h, .va = va, .size = size, .cpu = cpu,
                         .uvm_live = uvm_live, .dma_live = dma_live,
                         .rm_live = 1, .cpu_live = 1, .quarantined = 1 };
    rm->faulted = 1;
}

static int alloc_with_cacheability(Nvrm *rm, uint64_t size, NvrmMem *out, uint32_t gpu_cacheable);

/* Coherent, CPU-mapped, GPU-mapped buffer at one shared virtual address. */
int nvrm_alloc(Nvrm *rm, uint64_t size, NvrmMem *out) {
    return alloc_with_cacheability(rm, size, out, NVOS32_ATTR2_GPU_CACHEABLE_YES);
}

/* Same buffer, but the graphics chip does not keep it in its L2. The CPU
 * mapping is already uncached, so a CPU store reaches memory; without this a
 * resident chip program polling the buffer keeps reading its own L2 copy and
 * never sees the store. For memory both sides poll, not for bulk data. */
int nvrm_alloc_gpu_uncached(Nvrm *rm, uint64_t size, NvrmMem *out) {
    return alloc_with_cacheability(rm, size, out, NVOS32_ATTR2_GPU_CACHEABLE_NO);
}

static int alloc_with_cacheability(Nvrm *rm, uint64_t size, NvrmMem *out, uint32_t gpu_cacheable) {
    if (!rm || !out || !size || rm->faulted || size > SIZE_MAX - 0xfffULL)
        return rm ? fail(rm, "invalid or overflowed allocation size") : -1;
    size = (size + 0xfff) & ~0xfffull;
    if (rm->live_count >= NVRM_MAX_LIVE) return fail(rm, "live allocation table full (%d)", NVRM_MAX_LIVE);
    uint64_t va = va_take(rm, size, 0x1000);
    if (!va) return fail(rm, "GPU VA exhausted or overflowed");
    /* Reserve the CPU range first so the fixed mapping can never clobber anything. */
    void *res = mmap((void *)va, size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
    if (res != (void *)va) {
        if (res != MAP_FAILED) munmap(res, size);
        rm->faulted = 1;
        return fail(rm, "VA 0x%lx unavailable", (unsigned long)va);
    }

    NV_MEMORY_ALLOCATION_PARAMS mp;
    memset(&mp, 0, sizeof mp);
    mp.owner = rm->root;
    mp.type = NVOS32_TYPE_IMAGE;
    mp.flags = NVOS32_ALLOC_FLAGS_MAP_NOT_REQUIRED | NVOS32_ALLOC_FLAGS_MEMORY_HANDLE_PROVIDED |
               NVOS32_ALLOC_FLAGS_ALIGNMENT_FORCE | NVOS32_ALLOC_FLAGS_IGNORE_BANK_PLACEMENT;
    /* GB10 has no local video memory: everything is coherent system memory. */
    mp.attr = (NVOS32_ATTR_PHYSICALITY_ALLOW_NONCONTIGUOUS << 27) | (NVOS32_ATTR_LOCATION_PCI << 25);
    mp.attr2 = (gpu_cacheable << 2) | NVOS32_ATTR2_ZBC_PREFER_NO_ZBC;
    mp.format = 6;
    mp.size = size;
    mp.alignment = 0x1000;
    mp.limit = size - 1;
    uint32_t h;
    if (rm_alloc(rm, rm->device, NV01_MEMORY_SYSTEM, &mp, sizeof mp, &h)) {
        if (munmap(res, size) != 0) rm->faulted = 1;
        rm->faulted = 1; /* unknown driver state: do not recycle this VA */
        return -1;
    }
    void *cpu = map_to_cpu(rm, h, size, (void *)va, 0, 1);
    if (cpu != (void *)va) {
        char why[sizeof rm->err];
        memcpy(why, rm->err, sizeof why);
        quarantine_allocation(rm, h, va, size, (void *)va, 0, 0);
        return fail(rm, "cpu map at 0x%lx failed: %s", (unsigned long)va, why);
    }
    uint8_t uvm_created = 0, dma_mapped = 0;
    if (uvm_map(rm, va, size, h, &uvm_created, &dma_mapped)) {
        quarantine_allocation(rm, h, va, size, cpu, uvm_created, dma_mapped);
        return -1;
    }
    out->handle = h;
    out->va = va;
    out->size = size;
    out->cpu = cpu;
    memset(cpu, 0, size);

    rm->live[rm->live_count].handle = h;
    rm->live[rm->live_count].va = va;
    rm->live[rm->live_count].size = size;
    rm->live[rm->live_count].cpu = cpu;
    rm->live[rm->live_count].uvm_live = 1;
    rm->live[rm->live_count].dma_live = 1;
    rm->live[rm->live_count].rm_live = 1;
    rm->live[rm->live_count].cpu_live = 1;
    rm->live_count++;
    return 0;
}

/* Reverse of nvrm_alloc(): drop the UVM external range, free the RM memory
 * object, unmap the CPU range, and return the VA range to the free list. */
int nvrm_free(Nvrm *rm, NvrmMem *m) {
    if (!rm || !m) return -1;

    /* Idempotent: freeing an already-zeroed NvrmMem is a silent no-op. */
    if (m->handle == 0 && m->va == 0 && m->size == 0 && m->cpu == NULL) return 0;

    /* Reject anything that is not a currently-live allocation (e.g. a stale
     * copy of an NvrmMem that was already freed elsewhere) without touching
     * the driver at all. */
    int idx = -1;
    for (uint32_t i = 0; i < rm->live_count; i++) {
        if (rm->live[i].va == m->va && rm->live[i].handle == m->handle && rm->live[i].size == m->size) {
            if (idx >= 0) { rm->faulted = 1; return fail(rm, "duplicate live allocation table entry"); }
            idx = (int)i;
        }
    }
    if (idx < 0)
        return fail(rm, "nvrm_free: va 0x%lx handle 0x%x size 0x%lx is not a live allocation",
                    (unsigned long)m->va, m->handle, (unsigned long)m->size);

    NvrmLiveAlloc *live = &rm->live[idx];
    if (live->cpu != m->cpu || !m->cpu || !m->size ||
        (m->va & 0xfff) || (m->size & 0xfff)) {
        rm->faulted = 1;
        return fail(rm, "nvrm_free: allocation table identity corrupt");
    }
    /* Every completed step is irreversible. Preserve the stage flags if the
     * next step fails, and quarantine the VA until all releases are proven. */
    if (live->uvm_live) {
        if (uvm_free_checked(rm, m->va, m->size) != 0) {
            live->quarantined = 1;
            rm->faulted = 1;
            return -1;
        }
        live->uvm_live = 0;
    }
    if (live->dma_live) {
        if (dma_unmap_checked(rm, m->handle, m->va, m->size) != 0) {
            live->quarantined = 1;
            rm->faulted = 1;
            return -1;
        }
        live->dma_live = 0;
    }
    if (live->rm_live) {
        if (rm_free(rm, rm->device, m->handle) != 0) {
            live->quarantined = 1;
            rm->faulted = 1;
            return -1;
        }
        live->rm_live = 0;
    }
    if (live->cpu_live) {
        if (munmap(m->cpu, m->size) != 0) {
            live->quarantined = 1;
            rm->faulted = 1;
            return fail(rm, "munmap VA 0x%lx errno %d", (unsigned long)m->va, errno);
        }
        live->cpu_live = 0;
    }

    /* Remove from the live table (swap with last). */
    rm->live[idx] = rm->live[rm->live_count - 1];
    rm->live_count--;

    if (va_release(rm, m->va, m->size) != 0) return -1;

    memset(m, 0, sizeof *m);
    return 0;
}

int nvrm_channel(Nvrm *rm) {
    if (!rm || rm->faulted || rm->chgroup || rm->channel_registered)
        return rm ? fail(rm, "channel construction requires clean state") : -1;
    NV_CHANNEL_GROUP_ALLOCATION_PARAMETERS cg;
    memset(&cg, 0, sizeof cg);
    cg.engineType = NV2080_ENGINE_TYPE_GRAPHICS;
    if (rm_alloc(rm, rm->device, KEPLER_CHANNEL_GROUP_A, &cg, sizeof cg, &rm->chgroup)) return -1;
    NV_CTXSHARE_ALLOCATION_PARAMETERS cs;
    memset(&cs, 0, sizeof cs);
    cs.hVASpace = rm->vaspace;
    cs.flags = NV_CTXSHARE_ALLOCATION_FLAGS_SUBCONTEXT_ASYNC;
    if (rm_alloc(rm, rm->chgroup, FERMI_CONTEXT_SHARE_A, &cs, sizeof cs, &rm->ctxshare)) return -1;

    rm->entries = 1024;
    /* CPU publishes new queue entries and USERD GPPut after each completion.
     * A GPU-cached allocation can retain their previous contents on GB10;
     * CPU store barriers alone do not invalidate that cached copy. */
    if (nvrm_alloc_gpu_uncached(rm, 0x10000, &rm->fifo)) return -1;
    if (nvrm_alloc(rm, 0x100000, &rm->notifier)) return -1;

    NV_CHANNEL_ALLOC_PARAMS gp;
    memset(&gp, 0, sizeof gp);
    gp.gpFifoOffset = rm->fifo.va;
    gp.gpFifoEntries = rm->entries;
    gp.hObjectError = rm->notifier.handle;
    gp.hObjectBuffer = rm->fifo.handle;
    gp.hUserdMemory[0] = rm->fifo.handle;
    rm->userd_off = rm->entries * 8;   /* USERD follows the ring in the same allocation */
    gp.userdOffset[0] = rm->userd_off;
    gp.engineType = 0;
    gp.hContextShare = rm->ctxshare;
    if (rm_alloc(rm, rm->chgroup, rm->gpfifo_class, &gp, sizeof gp, &rm->gpfifo)) return -1;
    if (rm_alloc(rm, rm->gpfifo, rm->compute_class, NULL, 0, &rm->compute_obj)) return -1;

    NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN_PARAMS tk = { .workSubmitToken = 0xffffffffu };
    if (rm_control(rm, rm->gpfifo, NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN, &tk, sizeof tk)) return -1;
    rm->token = tk.workSubmitToken;

    /* The VA space is UVM-owned: UVM must bind the channel's context buffers before RM will schedule it. */
    UVM_REGISTER_CHANNEL_PARAMS rc;
    memset(&rc, 0, sizeof rc);
    memcpy(rc.gpuUuid.uuid, rm->gpu_uuid, 16);
    rc.rmCtrlFd = rm->fd_ctl;
    rc.hClient = rm->root;
    rc.hChannel = rm->gpfifo;
    rc.base = va_take(rm, 0x4000000, 0x1000);
    if (!rc.base) { rm->faulted = 1; return fail(rm, "channel VA exhausted"); }
    rc.length = 0x4000000;
    rm->channel_va = rc.base;
    rm->channel_va_size = rc.length;
    if (uvm(rm, rm->fd_uvm, UVM_REGISTER_CHANNEL, &rc, &rc.rmStatus)) {
        rm->faulted = 1;
        return -1;
    }
    rm->channel_registered = 1;

    NVA06C_CTRL_GPFIFO_SCHEDULE_PARAMS sch = { .bEnable = 1 };
    if (rm_control(rm, rm->chgroup, NVA06C_CTRL_CMD_GPFIFO_SCHEDULE, &sch, sizeof sch)) return -1;
    rm->put = 0;
    rm->retired = 0;
    return 0;
}

int nvrm_channel_destroy(Nvrm *rm) {
    if (!rm) return -1;
    if (rm->channel_registered) {
        UVM_UNREGISTER_CHANNEL_PARAMS p;
        memset(&p, 0, sizeof p);
        memcpy(p.gpuUuid.uuid, rm->gpu_uuid, sizeof rm->gpu_uuid);
        p.hClient = rm->root;
        p.hChannel = rm->gpfifo;
        if (uvm(rm, rm->fd_uvm, UVM_UNREGISTER_CHANNEL, &p, &p.rmStatus)) {
            rm->faulted = 1;
            return -1;
        }
        rm->channel_registered = 0;
    }
    if (rm->compute_obj && rm_free(rm, rm->gpfifo, rm->compute_obj)) goto failed;
    rm->compute_obj = 0;
    if (rm->gpfifo && rm_free(rm, rm->chgroup, rm->gpfifo)) goto failed;
    rm->gpfifo = 0;
    if (rm->ctxshare && rm_free(rm, rm->chgroup, rm->ctxshare)) goto failed;
    rm->ctxshare = 0;
    if (rm->chgroup && rm_free(rm, rm->device, rm->chgroup)) goto failed;
    rm->chgroup = 0;
    if (rm->fifo.handle && nvrm_free(rm, &rm->fifo)) goto failed;
    if (rm->notifier.handle && nvrm_free(rm, &rm->notifier)) goto failed;
    if (rm->channel_va_size) {
        if (va_release(rm, rm->channel_va, rm->channel_va_size) != 0) goto failed;
        rm->channel_va = 0;
        rm->channel_va_size = 0;
    }
    rm->put = rm->retired = 0;
    return 0;
failed:
    rm->faulted = 1;
    return -1;
}

/* GP entry per clc96f.h: ENTRY0 GET 31:2 (VA low, dword aligned), ENTRY1 GET_HI 7:0,
 * LEVEL 9:9 = MAIN, LENGTH 30:10 in dwords, SYNC 31:31 = PROCEED. */
uint64_t nvrm_gp_entry(uint64_t va, uint32_t nwords) {
    return (va & ~3ull) | ((uint64_t)(nwords & 0x1fffff) << 42) | (1ull << 41);
}

volatile uint32_t *nvrm_userd_gpput(Nvrm *rm) {
    return (volatile uint32_t *)((uint8_t *)rm->fifo.cpu + rm->userd_off + offsetof(Nvc96fControl, GPPut));
}

int nvrm_enqueue(Nvrm *rm, const NvrmMem *pb, uint32_t off, uint32_t nwords) {
    if (!rm || !pb || rm->faulted || !rm->gpfifo || !rm->fifo.cpu || rm->entries < 2)
        return rm ? fail(rm, "channel not ready") : -1;
    if ((off & 3u) || nwords == 0 || nwords > 0x1fffff || (uint64_t)off + (uint64_t)nwords * 4u > pb->size)
        return fail(rm, "bad pushbuffer span");
    if (pb->va > UINT64_MAX - off) return fail(rm, "pushbuffer VA overflow");
    /* Blackwell USERD documents no GPGet (clc96f.h), so consumption is tracked by the
     * caller via nvrm_retire() after it observes completion. Never overrun unretired entries. */
    if (rm->put - rm->retired >= rm->entries - 1) return fail(rm, "GPFIFO full: %u entries unretired", rm->put - rm->retired);
    volatile uint64_t *ring = (volatile uint64_t *)rm->fifo.cpu;
    ring[rm->put % rm->entries] = nvrm_gp_entry(pb->va + off, nwords);
    rm->put++;
    __asm__ volatile("dsb sy" ::: "memory");
    *nvrm_userd_gpput(rm) = rm->put % rm->entries;
    __asm__ volatile("dsb sy" ::: "memory");
    return 0;
}

void nvrm_retire(Nvrm *rm, uint32_t upto) {
    if (upto - rm->retired <= rm->put - rm->retired) rm->retired = upto;
}

void nvrm_ring(Nvrm *rm) {
    *rm->doorbell = rm->token;  /* NVC361_NOTIFY_CHANNEL_PENDING */
}

int nvrm_submit(Nvrm *rm, const NvrmMem *pb, uint32_t off, uint32_t nwords) {
    if (nvrm_enqueue(rm, pb, off, nwords)) return -1;
    nvrm_ring(rm);
    return 0;
}

int nvrm_close(Nvrm *rm) {
    if (!rm) return -1;
    if (rm->root && nvrm_channel_destroy(rm)) rm->faulted = 1;
    /* Release every remaining allocation, including caller-owned buffers and
     * allocations left behind by an interrupted construction path. */
    for (uint32_t i = rm->live_count; i > 0;) {
        NvrmLiveAlloc e = rm->live[--i];
        NvrmMem m = { .handle = e.handle, .va = e.va, .size = e.size, .cpu = e.cpu };
        if (nvrm_free(rm, &m) != 0) {
            rm->faulted = 1;
            /* Closing the client tears down driver state. The CPU mapping is
             * process state, so explicitly unmap it even on teardown error. */
            if (e.cpu_live && e.cpu) (void)munmap(e.cpu, e.size);
        } else if (i < rm->live_count) i++;
    }
    if (rm->usermode_cpu) {
        if (munmap(rm->usermode_cpu, 0x10000) != 0) rm->faulted = 1;
        rm->usermode_cpu = NULL;
        rm->doorbell = NULL;
    }
    if (rm->vas_registered) {
        UVM_UNREGISTER_GPU_VASPACE_PARAMS p;
        memset(&p, 0, sizeof p);
        memcpy(p.gpuUuid.uuid, rm->gpu_uuid, sizeof rm->gpu_uuid);
        if (uvm(rm, rm->fd_uvm, UVM_UNREGISTER_GPU_VASPACE, &p, &p.rmStatus)) rm->faulted = 1;
        rm->vas_registered = 0;
    }
    if (rm->gpu_registered) {
        UVM_UNREGISTER_GPU_PARAMS p;
        memset(&p, 0, sizeof p);
        memcpy(p.gpu_uuid.uuid, rm->gpu_uuid, sizeof rm->gpu_uuid);
        if (uvm(rm, rm->fd_uvm, UVM_UNREGISTER_GPU, &p, &p.rmStatus)) rm->faulted = 1;
        rm->gpu_registered = 0;
    }
    /* Free root children explicitly so a clean close proves every RM object
     * was accepted for release; parent teardown alone hides leaked handles. */
    if (rm->usermode && rm_free(rm, rm->subdevice, rm->usermode)) rm->faulted = 1;
    rm->usermode = 0;
    if (rm->vaspace && rm_free(rm, rm->device, rm->vaspace)) rm->faulted = 1;
    rm->vaspace = 0;
    if (rm->virtmem && rm_free(rm, rm->device, rm->virtmem)) rm->faulted = 1;
    rm->virtmem = 0;
    if (rm->subdevice && rm_free(rm, rm->device, rm->subdevice)) rm->faulted = 1;
    rm->subdevice = 0;
    if (rm->device && rm_free(rm, rm->root, rm->device)) rm->faulted = 1;
    rm->device = 0;
    if (rm->root && rm_free(rm, rm->root, rm->root)) rm->faulted = 1;
    rm->root = 0;
    if (rm->fd_dev > 0) close(rm->fd_dev);
    if (rm->fd_uvm2 > 0) close(rm->fd_uvm2);
    if (rm->fd_uvm > 0) close(rm->fd_uvm);
    if (rm->fd_ctl > 0) close(rm->fd_ctl);
    if (!rm->faulted && rm->live_count == 0 && !rm->channel_registered &&
        rm->va_slot_base && rm->va_next >= rm->va_slot_base &&
        rm->va_next - rm->va_slot_base <= 0x100000000ull)
        recycle_va_slot(rm->va_slot_base);
    rm->va_slot_base = 0;
    return rm->faulted ? -1 : 0;
}
