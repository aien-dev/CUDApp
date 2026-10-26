/* SPDX-License-Identifier: MIT OR GPL-2.0-only */
/*
 * aien_gpu_uapi.h -- the /dev/aien-gpu interface (ABI 1). Shared by the module and user space.
 * A new, small compute ABI of our own, not NVIDIA's NV_ESC/UVM ioctls (docs/DESIGN.md section 3).
 * Every argument block has a fixed size and is copied into the kernel once; its checks are Omega
 * (om/abi.om) before the kernel acts on it. Little endian; reserved bytes must be zero.
 */
#ifndef AIEN_GPU_UAPI_H
#define AIEN_GPU_UAPI_H

#include <linux/ioctl.h>
#include <linux/types.h>

#define AGPU_ABI_VERSION 1

/* Device state, AGPU_INFO.state */
enum agpu_state {
	AGPU_STATE_BOUND = 1,	/* identity checked, BAR0 mapped, memory allocated; GSP not touched */
	AGPU_STATE_BOOTING = 2,	/* a start-up is running or stopped half way (never retried per load) */
	AGPU_STATE_GSP_UP = 3,	/* GSP-RM reported INIT_DONE */
	AGPU_STATE_RM_READY = 4,	/* stage C bring-up done: clients may allocate memory and channels */
	AGPU_STATE_FAILED = 5,	/* a stage stopped; see the boot or rm verdict; reload the module after a reset */
};

struct agpu_info {
	__u32 abi;		/* AGPU_ABI_VERSION */
	__u32 state;		/* enum agpu_state */
	__u32 boot0, boot42;	/* NV_PMC_BOOT_0, NV_PMC_BOOT_42 as read at probe */
	__u64 bar0_bytes;
	__u64 arena_bytes, arena_iova;	/* the boot arena: mmap offset 0, length arena_bytes */
	__u64 heap_bytes, heap_iova;	/* the GSP system-memory heap: device only, never mapped */
	__u64 fw_bytes;		/* gsp_ga10x.bin size as loaded at probe (0 if missing) */
	__u32 fw_ok;		/* size and sha256 equal the pins in kmod/aien_gpu.h */
	__u32 dma_bits;
	__u32 boot_attempts;	/* 0 or 1: one GSP start-up per load */
	__u32 reserved[9];
};

/* AGPU_BOOT: stage B. in = the 128-byte block om/abi.om checks (request 1); out = the verdict. */
enum agpu_boot_verdict {
	AGPU_BOOT_INIT_DONE = 1,	/* GSP-RM sent GSP_INIT_DONE */
	AGPU_BOOT_REFUSED = 2,		/* the block failed om/abi.om (out.code); the chip was not touched */
	AGPU_BOOT_MMIO_STOPPED = 3,	/* the 11 register steps stopped: out.stage says where */
	AGPU_BOOT_QUEUE_ERROR = 4,	/* a status-queue message failed om/mqget.om (out.code) */
	AGPU_BOOT_SEQ_REFUSED = 5,	/* a sequencer request failed om/seq.om (out.code) */
	AGPU_BOOT_SEQ_STOPPED = 6,	/* a sequencer list stopped: poll (out.code 1) or core operation (2) */
	AGPU_BOOT_TIMEOUT = 7,		/* no INIT_DONE within waitms */
	AGPU_BOOT_STATE = 8,		/* not in AGPU_STATE_BOUND (already attempted) */
};

struct agpu_boot_out {
	__u32 verdict;		/* enum agpu_boot_verdict */
	__u32 code;		/* refusal code of the Omega unit named by the verdict */
	__u32 stage;		/* om/boot.om stage: 0 ok, 1 arg, 2 busy, 3 emem, 4 sec2, 5 mask, 6 lockdown, 7 fmc, 8 riscv */
	__u32 polls[4];
	__u32 dwords, ememc_after;
	__u32 sec2_mb0, sec2_mb1, hwcfg2, gsp_mb0, gsp_mb1, cpuctl;
	__u32 msgs;		/* status-queue messages read */
	__u32 last_fn;		/* function of the last message */
	__u32 seq_runs;		/* sequencer requests performed */
	__u32 seq_ops;		/* operations of the last one performed */
	__u32 rx_seq;		/* next expected status-queue sequence number */
	__u64 mmio_us, wait_us;	/* time in the register steps, time waiting for INIT_DONE */
	__u32 reserved[4];
};

struct agpu_boot {
	__u8 in[128];		/* layout: om/abi.om header and tools/aien-gpu-boot.c */
	struct agpu_boot_out out;
};

/* AGPU_ARENA_READ: copy a range of the boot arena out (root; evidence such as GSP's log buffers after
 * the start-up, when the arena is no longer mappable). om/abi.om request 2 checks off/len. */
struct agpu_arena_read {
	__u64 off, len;		/* len 1..16 MiB, inside the arena */
	__u64 buf;		/* user pointer */
};

/*
 * Stage C. All GSP RPC payloads are NVIDIA's 580.173.02 structures (MIT headers), built and checked by
 * om/rm.om; any value that depends on what GSP-RM reports is read at run time, and a mismatch is a
 * named refusal (AGPU_RM_E_*), never a guess. Route: an RM-managed GPU address space whose page tables
 * this module writes (docs/STAGE-C.md); no UVM.
 */
enum agpu_rm_verdict {
	AGPU_RM_OK = 1,
	AGPU_RM_E_STATE = 2,	/* wrong device state for this call */
	AGPU_RM_E_ARG = 3,	/* argument block refused by om/abi.om (code) */
	AGPU_RM_E_QUEUE = 4,	/* message queue refusal (code: om/seal, om/mqput, om/mqget) */
	AGPU_RM_E_TIMEOUT = 5,	/* no reply within the RPC deadline */
	AGPU_RM_E_RESULT = 6,	/* GSP-RM answered rpc_result != 0 (code = rpc_result, fn = function) */
	AGPU_RM_E_STATUS = 7,	/* the RPC's own status field != NV_OK (code = status) */
	AGPU_RM_E_VERSION = 8,	/* SET_GUEST_SYSTEM_INFO answered another version than 580.173.02's */
	AGPU_RM_E_STATIC = 9,	/* GspStaticConfigInfo failed a check (code says which) */
	AGPU_RM_E_REPLY = 10,	/* a reply failed om/rm.om (code) */
	AGPU_RM_E_NOMEM = 11,	/* host memory or DMA mapping */
	AGPU_RM_E_UNSUPPORTED = 12,	/* GSP asked for something this driver does not do (code says what) */
};

struct agpu_rm_status {
	__u32 verdict;		/* enum agpu_rm_verdict */
	__u32 code;
	__u32 fn;		/* RPC function that failed, or 0 */
	__u32 step;		/* the numbered step of docs/STAGE-C.md that failed, or 0 */
};

/* AGPU_RM_INIT: once per load, after INIT_DONE (spec section 1). */
struct agpu_rm_init {
	struct agpu_rm_status st;
	__u32 vgx_major, vgx_minor;	/* what GSP-RM answered */
	__u32 h_client, h_device, h_subdevice;	/* GSP-RM's internal handles (static info 1728..1736) */
	__u32 gr_runlist;		/* runlist id of GR0 (FIFO device info table) */
	__u32 reserved[8];
	__u8 static_info[1792];		/* GspStaticConfigInfo as received, for the evidence */
};

/* AGPU_MEM_ALLOC: system memory the GPU sees at gpu_va == the CPU address the caller maps it at.
 * The caller reserves [va, va + size) first (mmap PROT_NONE, MAP_FIXED_NOREPLACE), then maps the
 * returned mmap_off over it with MAP_FIXED|MAP_SHARED. */
#define AGPU_MEM_GPU_UNCACHED 1u	/* GPU does not cache it in L2 (memory the CPU polls) */
struct agpu_mem_alloc {
	__u64 size;		/* multiple of 4096, 4 KiB .. 64 GiB */
	__u64 va;		/* 4096-aligned, below 2^48, outside [4 GiB, 4.5 GiB) (GSP's range) */
	__u32 flags;		/* AGPU_MEM_* */
	__u32 handle;		/* out */
	__u64 mmap_off;		/* out */
	struct agpu_rm_status st;	/* out */
};

struct agpu_mem_free {
	__u32 handle;
	__u32 reserved;
	struct agpu_rm_status st;
};

/* AGPU_CHAN_CREATE: one compute channel (TSG, context share, GPFIFO channel, BLACKWELL_COMPUTE_B,
 * scheduled). Ring and USERD live in caller memory from AGPU_MEM_ALLOC (GPU-uncached). */
struct agpu_chan_create {
	__u32 ring_handle;	/* memory holding the GPFIFO ring */
	__u32 entries;		/* ring entries, power of two, 8..65536 */
	__u64 ring_off;		/* byte offset of the ring in that memory, 8-aligned */
	__u32 userd_handle;	/* memory holding USERD (512 bytes) */
	__u32 err_handle;	/* memory receiving channel errors (error notifier) */
	__u64 userd_off;	/* 512-aligned */
	__u32 chid;		/* out */
	__u32 token;		/* out: work-submit token, (runlist << 16) | chid */
	__u64 doorbell_off;	/* out: mmap offset of the 64 KiB usermode page; doorbell word at +0x90 */
	__u32 channel;		/* out: handle for AGPU_CHAN_DESTROY */
	__u32 reserved;
	struct agpu_rm_status st;
};

struct agpu_chan_destroy {
	__u32 channel;
	__u32 reserved;
	struct agpu_rm_status st;
};

#define AGPU_INFO _IOR('A', 0x10, struct agpu_info)
#define AGPU_BOOT _IOWR('A', 0x11, struct agpu_boot)
#define AGPU_ARENA_READ _IOW('A', 0x12, struct agpu_arena_read)
#define AGPU_RM_INIT _IOWR('A', 0x20, struct agpu_rm_init)
#define AGPU_MEM_ALLOC _IOWR('A', 0x21, struct agpu_mem_alloc)
#define AGPU_MEM_FREE _IOWR('A', 0x22, struct agpu_mem_free)
#define AGPU_CHAN_CREATE _IOWR('A', 0x23, struct agpu_chan_create)
#define AGPU_CHAN_DESTROY _IOWR('A', 0x24, struct agpu_chan_destroy)

#endif
