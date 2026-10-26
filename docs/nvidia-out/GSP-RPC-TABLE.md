# GSP boot sequence, RPC table and serving-surface map (580.173.02, GB20B / GB10B)

Milestone 1b of the NoNvDriver lane. This is source-only research: no device was opened, no driver bound, nothing run on the GPU. Written 2026-10-26.

Source pin: NVIDIA open-gpu-kernel-modules 580.173.02 at `/home/drakestapleton/workspace/_ext/open-gpu-kernel-modules-580.173.02`, git `20e4e6e19cc26ba47b5cbe23130a396be100c427`.

Path shorthands used below:
- `N/` means `src/nvidia/`.
- `C/` means `src/common/`.
- `U/` means `kernel-open/nvidia-uvm/`.
- `KO/` means `kernel-open/nvidia/`.

Labels:
- **MEASURED**: read at the cited file:line in the pinned tree, or by the named command.
- **MODELED**: derived by bookkeeping, not compiled.
- **[INFERENCE]**: reasoned, not shown by a cited line.
- **NOT_VERIFIED**: open.
- **NOT_RUN**: nothing was executed on hardware for any row in this file.

Chip identity is still open. Only a PMC_BOOT_42 read in an authorised window can settle it.
- GB20B: `GPU_ARCHITECTURE_BLACKWELL_GB2XX` 0x01B0 (`C/inc/swref/published/nv_arch.h:53`), impl `GPU_IMPLEMENTATION_GB20B` 0x0B (`:127`).
- GB10B: `GB1XX` 0x01A0 (`:51`), impl 0x0B (`:111`).

One new piece of evidence bears on this:
- The client CUDApp allocates `BLACKWELL_COMPUTE_B`. That class appears in the GB20B engine class list (`N/generated/g_gpu_class_list.c:3635-3642`).
- The GB10B list has only `BLACKWELL_COMPUTE_A` (`:2426-2427`). MEASURED.
- So if CUDApp's COMPUTE_B allocation has ever succeeded on this Spark, the chip is not GB10B. [INFERENCE]
- I did not check for a CUDApp receipt of that allocation: NOT_VERIFIED.

---

## 1. Boot sequence (GB20B primary; GB10B differences marked)

**Signed boot chain: load unmodified only (SBC).** Marks a step where NVIDIA-signed code goes to the chip or runs on it. Our driver may only copy those bytes unmodified and point the chip at them.

**FWSEC / VBIOS ucode step:**
- **GB20B: yes, chip-side.** The driver does not load FWSEC. It waits for it:
  - `_ksec2WaitBootCond_GB20B` polls `NV_PMC_SCRATCH_RESET_PLUS_2 == 0xFF`. Its comment reads "FWSEC writes 0xFF ... after completion of boot" (`N/src/kernel/gpu/sec2/arch/blackwell/kernel_sec2_gb20b.c:540-550`).
  - `ksec2WaitForSecureBoot_GB20B` wraps that poll with a 4 s timeout (`:552-575`).
  - Status: MEASURED (source).
- **GB10B: no FWSEC wait anywhere in the SEC2 path.** `ksec2PrepareBootCommands_GB10B` has none (`kernel_sec2_gb10b.c:560-596`).
  - `kgspExtractVbiosFromRom` for GB10B is a stub. L332 cites `g_kernel_gsp_nvoc.c:1300` (re-cited from L332, not re-read: NOT_VERIFIED here).
  - Status: MEASURED.
- **Both:** `kgspWaitForGfwBootOk` is bound to `_ac1694`, a no-op returning OK, for GB10B | GB20B | GB20C (`N/generated/g_kernel_gsp_nvoc.c:1513`). The GFW-boot wait that dGPUs do through FSP (`kernel_gsp_gh100.c:240-254`) is not taken. MEASURED.

| # | Step | GB20B source | GB10B difference | SBC |
|---|---|---|---|---|
| 0 | Chip power-on: boot ROM, SEC2 firmware, FWSEC/GFW run with no driver. The driver only observes. | completion flag `kernel_sec2_gb20b.c:540-550` | no observable FWSEC flag; the driver checks GSP HWCFG2 ≠ `0xBADF41xx` (`kernel_sec2_gb10b.c:264-283`) | **SBC** (never touched) |
| 1 | Pick the firmware file `nvidia/580.173.02/gsp_ga10x.bin` and the family. impl ≥ GB20B gives GB20Y. | `N/arch/nvalloc/common/inc/nv-firmware-chip-family-select.h:66`; strings `nv-firmware.h:59-62` | impl == GB10B gives GB10Y (`:60`) | |
| 2 | Signature section is `.fwsignature_` + family, so `.fwsignature_gb20y` (`.fwsignature_cc_` prefix under CC) | `N/src/kernel/gpu/gsp/kernel_gsp.c:4807-4837,4869`; prefix HAL `arch/blackwell/kernel_gsp_gb100.c:95` bound at `g_kernel_gsp_nvoc.c:1811` | `.fwsignature_gb10y` | **SBC** (signature bytes) |
| 3 | `kgspInitRm_IMPL` starts. RPC infrastructure: `GspMsgQueuesInit` allocates 256 KiB command and 256 KiB status queues in sysmem (unprotected, noncontiguous) | `kernel_gsp.c:3984`, `:3662`, `:2565`; `message_queue_cpu.c:71-72,186,222-262` | same | |
| 4 | `kgspAllocBootArgs_HAL` → `GSP_FMC_BOOT_PARAMS` | `kernel_gsp.c:3682`; `arch/hopper/kernel_gsp_gh100.c:148-190` | same | |
| 5 | Prepare the GSP-RM ELF: version check, signature memdesc, radix-3 page table over `.fwimage` | `kernel_gsp.c:4117`, `:4843`, radix3 `:4884` | same | **SBC** (`.fwimage` ELF, unmodified) |
| 6 | LIBOS log buffers and `kgspSetupLibosInitArgs` (log regions, then RMARGS = `GSP_ARGUMENTS_CACHED`) | `kernel_gsp.c:4125,4156,5179-5215`; cached args filled `:4519-4571` | same | |
| 7 | `kgspQueueAsyncInitRpcs` pre-stuffs the command queue **before** boot with `GSP_SET_SYSTEM_INFO` (72), then `SET_REGISTRY` (73) | `kernel_gsp.c:4179` → `:3792-3815` (72 at `:3800`, 73 at `:3807`) | same | |
| 8 | `_kgspBootGspRm`: the WPR2-already-up check is skipped when `PDB_PROP_GPU_PREINITIALIZED_WPR_REGION` is set (true for GB10B, GB20B, GB20C) | `kernel_gsp.c:3902-3912`, `:4222` | same | |
| 9 | `kgspPopulateWprMeta_GB10B`: the GH100 layout (256 B `GspFwWprMeta`), then vgaWorkspace = pmuReserved = nonWprHeap = 0, "No VGA Workspace on iGPU" | `kernel_gsp.c:3920`; binding `g_kernel_gsp_nvoc.c:989` (GB10B\|GB20B\|GB20C); body `arch/blackwell/kernel_gsp_gb10b.c:62-88`; GH100 `kernel_gsp_gh100.c:295-388` (size 256 at `:305`) | same function | **SBC** (WprMeta names the GspRmBoot bootloader) |
| 10 | `kgspPrepareForBootstrap_GH100` takes the SEC2 branch → `ksec2PrepareBootCommands_GB20B`, which waits for FWSEC, checks SafeToSend, allocates the COT payload with version 2 | `kernel_gsp.c:3928`; `kernel_gsp_gh100.c:775-811` (SEC2 `:802`); `kernel_sec2_gb20b.c:952-1019`, SafeToSend `:892-937`; version `g_kernel_sec2_nvoc.c:485` | `_GB10B` `kernel_sec2_gb10b.c:560-596`: no FWSEC wait; SafeToSend checks only the BOOT_GSPFMC property (`:414-427`); version 1 (`nvoc:481`) | |
| 11 | `ksec2SetupGspImages_GB20B`: pick the archive (Prod `GspRmFmcGfwProdSigned_GB20B`, CC or Debug), copy the FMC image to sysmem, fill `NVDM_PAYLOAD_COT` (gspFmcSysmemOffset `:801`, hash `:803`, sig `:806-808`, pkey `:811-813`, bootArgs `:817`). `frtsSysmemSize` is left 0 | `kernel_sec2_gb20b.c:577-692,729-827`; struct `N/inc/kernel/gpu/fsp/kern_fsp_cot_payload.h:36-56` (860 B packed, MODELED sum) | `_GB10B` sets `frtsSysmemSize` = image size (`kernel_sec2_gb10b.c:508`); archive is Prod or Debug only, no CC (`:215-238`) | **SBC** (FMC image, SHA-384 hash, RSA-3K signature and public key from bindata) |
| 12 | `kgspBootstrap_GH100`: RISC-V mode, then **`ksec2SendBootCommands_HAL`** | `kernel_gsp.c:3946`; `kernel_gsp_gh100.c:833-985`, `:850`, **`:871`** | same call site | **SBC (handoff)** |
| 13 | GB20B send: MCTP+NVDM packet (NVDM_TYPE_COT 0x14, `sec2_nvdm_format.h:39`) into SEC2 EMEM channel 0 (1 KiB, `sec2_emem_channels.h:34,37`), then **wait for and parse the SEC2 response** (`ksec2SendAndReadMessage`) | `kernel_sec2_gb20b.c:1033-1084`, send `:1044`; MCTP header `:205-220`; EMEM MSGQ read `:77-109,432-538`; vendor check `:293-331`; error map `:1113-1136`; `kernel_sec2.c:282,427,496` | GB10B: fire-and-forget `ksec2SendMessage` (`kernel_sec2_gb10b.c:622`), then poll FALCON_MAILBOX0 ≠ `0xc001cafe` (`:285-297`, `:640`) | **SBC** (SEC2 verifies and starts the FMC on chip) |
| 14 | Wait for the GSP target mask release, then lockdown release or FMC error in MAILBOX0 | `kernel_gsp_gh100.c:901,926,944-948` | same | **SBC** (FMC/ACR run, set up WPR2, load GspRmBoot and GSP-RM) |
| 15 | Program `FALCON_OS` = descriptor appVersion; check RISC-V active | `kernel_gsp_gh100.c:956,959` | same | |
| 16 | `GspStatusQueueInit`, then `kgspWaitForRmInitDone`: `rpcRecvPoll(GSP_INIT_DONE 0x1001)`, check rpc_result | `kernel_gsp_gh100.c:977,980`; `kernel_gsp.c:5252-5280` (`:5267`); `message_queue_cpu.c:317` | same | |
| 17 | While waiting, GSP may send `GSP_RUN_CPU_SEQUENCER` (0x1002). The CPU executes its register ops (WRITE/MODIFY/POLL/DELAY/STORE/CORE_RESET/START/WAIT_HALT) | `kernel_gsp.c:5297`; opcodes `:5331-5420`; boot-safe event list `:1503-1509` | same | |
| 18 | `NV_RM_RPC_GET_GSP_STATIC_INFO` (65), then log polling | `kernel_gsp.c:4270,4285` | same | |
| 19 | Unload: `NV_RM_RPC_UNLOADING_GUEST_DRIVER` (47) | `kernel_gsp.c:4339` | same | |

Shared HAL bindings for GB10B and GB20B (MEASURED, `g_kernel_gsp_nvoc.c`):
- `PrepareForBootstrap_GH100` `:698`
- `Bootstrap_GH100` `:719`
- `ResetHw_GB100` `:1091`
- `SetupGspFmcArgs_GH100` `:1830`
- `SetCmdQueueHead_TU102` `:664-678` (writes `NV_PGSP_QUEUE_HEAD(i)`, `arch/turing/kernel_gsp_tu102.c:343-354`)
- `ksec2SendPacket_GB10B` for both (`g_kernel_sec2_nvoc.c:804`)
- `PDB_PROP_GPU_ZERO_FB` true for GB10B | GB20B | GB20C (`N/generated/g_gpu_nvoc.c:212-218`)

---

## 2. RPC table

### 2.1 Framing (both chips)

| Item | Layout | Size | Source |
|---|---|---|---|
| Shared memory | page table, cmd queue (`msgqTxHeader` + entries), status queue (`msgqTxHeader` + entries), each page-aligned | default 0x40000 per queue | `N/inc/kernel/gpu/gsp/message_queue_priv.h:35-41`; `message_queue_cpu.c:71-72` |
| `msgqTxHeader` | u32 version, size, msgSize, msgCount, writePtr, flags, rxHdrOff, entryOff | 32 B (MODELED) | `C/shared/msgq/inc/msgq/msgq_priv.h:49-59`; MSGQ_VERSION 0 `:38` |
| `msgqRxHeader` | u32 readPtr | 4 B | `msgq_priv.h:62-65` |
| `MESSAGE_QUEUE_INIT_ARGUMENTS` | u64 sharedMemPhysAddr; u32 pageTableEntryCount; NvLength cmdQueueOffset, statQueueOffset | — | `N/inc/kernel/gpu/gsp/gsp_init_args.h:29-34` |
| `GSP_MSG_QUEUE_ELEMENT` | u8 authTagBuffer[16]; u8 aadBuffer[16]; u32 checkSum; u32 seqNum; u32 elemCount; `NV_DECLARE_ALIGNED(rpc_message_header_v rpc, 8)` | header 48 B (44 + 4 pad; MODELED). Upstream bindgen shows the same 4-byte pad (nova `bindings.rs:869-877`) | `message_queue_priv.h:43-51` |
| Element size | min `RM_PAGE_SIZE` 4096, max 16 × that; RPC payload max = max − 48 | — | `message_queue_priv.h:91-103` |
| Send / receive | seqNum `:481`, elemCount `:482`, checksum `:515/519`; receive `:608` | — | `message_queue_cpu.c:456-519,608` |
| `rpc_message_header_v03_00` | u32 header_version, signature, length, function, rpc_result, rpc_result_private, sequence; union {u32 spare \| cpuRmGfid}; `rpc_message_data[]` | 32 B (MODELED) | `N/generated/g_rpc-message-header.h:33-54`; nova `bindings.rs:686-696` |
| Queues | 1 queue, RM task index 0 | — | `N/inc/kernel/gpu/gsp/message_queue.h:31-32` |
| Oversize | the function is split across `CONTINUATION_RECORD` (71) elements | — | enum `rpc_global_enums.h:81` |

### 2.2 Functions and events our driver needs

Enum file: `N/inc/kernel/vgpu/rpc_global_enums.h`. Structs: `N/generated/g_rpc-structures.h` (shown as `S:`). Senders: `N/src/kernel/vgpu/rpc.c` (shown as `R:`).

Upstream reference: nova-core `drivers/gpu/nova-core/gsp/fw/r570_144/bindings.rs`, fetched 2026-10-26 from `https://raw.githubusercontent.com/torvalds/linux/master/drivers/gpu/nova-core/gsp/fw/r570_144/bindings.rs`. That is the master branch with no commit pin; the fetched file's sha256 is `70d5d253ebffe5834b6cac75b6a37e6aa18d0c39a940162b5cb5d17e330f79e7`. These are **r570** numbers. Every number in the rows below matches 580.
- **ABI differs:** 580 has `NUM_FUNCTIONS` = 229 (`:239`) where r570 has 227 (`bindings.rs:289`).
- 580 has `NUM_EVENTS` = 0x1024 (`:288`) where r570 has 4131 = 0x1023 (`bindings.rs:326`).
- So r570 bindings must not be used to talk to 580 firmware.
- I did not fetch nouveau r570 (fetch budget): NOT_VERIFIED.

| Dir | Name | Number (enum line) | Param struct (S: line) | Sender / handler (580) | Phase | Upstream nova r570 |
|---|---|---|---|---|---|---|
| CPU→GSP | GSP_SET_SYSTEM_INFO | 72 (`:82`) | `rpc_gsp_set_system_info_v17_00` {u32 data} (`S:1484`); payload `GspSystemInfo` `N/inc/kernel/gpu/gsp/gsp_static_config.h:172-227` | `R:10493`; queued `kernel_gsp.c:3800` | boot (pre-queued) | `bindings.rs:132`; `GspSystemInfo` `:580` |
| CPU→GSP | SET_REGISTRY | 73 (`:83`) | `PACKED_REGISTRY_TABLE` `N/generated/g_os_nvoc.h:234-239`. The rpc struct is described at `S:8368` only; the exact struct is NOT_VERIFIED | `R:10666`; queued `kernel_gsp.c:3807` | boot (pre-queued) | `:133`; `PACKED_REGISTRY_TABLE` `:844` |
| CPU→GSP | GET_GSP_STATIC_INFO | 65 (`:75`) | `rpc_get_gsp_static_info_v14_00` {u32 data} (`S:440`); payload `GspStaticConfigInfo` (`gsp_static_config.h:169`) | `R:9661`; called `kernel_gsp.c:4270` | boot (after INIT_DONE) | `:125`; `GspStaticConfigInfo_t` `:507` |
| both | CONTINUATION_RECORD | 71 (`:81`) | raw continuation bytes | queue layer | any | `:131` |
| CPU→GSP | GSP_RM_ALLOC | 103 (`:113`) | `rpc_gsp_rm_alloc_v03_00` {hClient, hParent, hObject, hClass, status, paramsSize, flags, reserved[4], params[]} (`S:1491`) | `rpcRmApiAlloc_GSP` `R:11138` (header `R:11195`; NV01_ROOT params-size workaround `R:11182-11184`); bound as physical RMAPI `N/src/kernel/rmapi/rpc_common.c:74-79` | runtime | `:163` |
| CPU→GSP | GSP_RM_CONTROL | 76 (`:86`) | `rpc_gsp_rm_control_v03_00` {hClient, hObject, cmd, status, paramsSize, rmapiRpcFlags, rmctrlFlags, rmctrlAccessRight, u64 reserved0, params[]} (`S:1506`) | `rpcRmApiControl_GSP` `R:10853` (header `R:10984`) | runtime | `:136` |
| CPU→GSP | FREE | 10 (`:20`) | `rpc_free_v03_00` {NVOS00_PARAMETERS_v03_00} (`S:160`) | `rpcRmApiFree_GSP` `R:11311` (header `R:11333`) | runtime | `:70` |
| CPU→GSP | DUP_OBJECT | 21 (`:31`) | `rpc_dup_object_v03_00` (struct line NOT_VERIFIED) | `rpcRmApiDupObject_GSP` `R:11259` (header `R:11285`) | runtime (UVM dup paths, §3) | `:81` |
| CPU→GSP | UNLOADING_GUEST_DRIVER | 47 (`:57`) | `rpc_unloading_guest_driver_v1F_07` (`S:378`) | `R:9166`; called `kernel_gsp.c:4339` | teardown | `:107` |
| CPU→GSP | ALLOC_MEMORY | 4 (`:14`) | `rpc_alloc_memory_v13_01` (`S:79`) | `R:3458` (header `R:3470`). Under the GSP client only via `memRegisterWithGsp` (`N/src/kernel/mem_mgr/mem.c:515`, macro `inc/kernel/vgpu/rpc.h:113-150`), reached from `memdescSendMemDescToGSP` (`gpu/mem_mgr/mem_desc.c:4786,4865`). No serving-path caller found | runtime, **not on our path** | `:64` |
| CPU→GSP | UPDATE_BAR_PDE | 70 (`:80`) | `rpc_update_bar_pde_v15_00` (`S:447`) | `R:9701`; called `gpu/bus/kern_bus.c:880`. Whether it fires on an iGPU with no BAR2 FB: NOT_VERIFIED | boot | `:130` |
| GSP→CPU | GSP_INIT_DONE | 0x1001 (`:253`) | header only; rpc_result checked | `_kgspRpcRecvPoll` `kernel_gsp.c:1648,2338,2362-2367` | boot | `:292` (4097) |
| GSP→CPU | GSP_RUN_CPU_SEQUENCER | 0x1002 (`:254`) | `rpc_run_cpu_sequencer_v17_00` {bufferSizeDWord, cmdIndex, regSaveArea[8], commandBuffer[]} (`S:1535`) | `_kgspRpcRunCpuSequencer` `kernel_gsp.c:1520` → `:5297` | boot, suspend/resume | `:293`; struct `:979` |
| GSP→CPU | POST_EVENT | 0x1003 (`:255`) | `rpc_post_event_v17_00` (`S:1545`) | `_kgspRpcPostEvent` (dispatch ~`kernel_gsp.c:1495-1560`) | runtime (notifiers) | `:294` |
| GSP→CPU | RC_TRIGGERED | 0x1004 (`:256`) | `rpc_rc_triggered_v17_02` (`S:1560`) | `_kgspRpcRCTriggered` (same dispatch) | runtime (channel errors) | `:295` |
| GSP→CPU | MMU_FAULT_QUEUED | 0x1005 (`:257`) | header only (NOT_VERIFIED) | `_kgspRpcMMUFaultQueued` `kernel_gsp.c:1532` | runtime (non-replayable faults) | `:296` |
| GSP→CPU | OS_ERROR_LOG | 0x1006 (`:258`) | `rpc_os_error_log_v17_00` (`S:1579`) | boot-safe list `kernel_gsp.c:1503-1509` | any | `:297` |
| GSP→CPU | UCODE_LIBOS_PRINT | 0x100c (`:264`) | `rpc_ucode_libos_print_v1E_08` (`S:1656`) | boot-safe list | any | `:303` |
| GSP→CPU | GSP_LOCKDOWN_NOTICE | 0x101c (`:280`) | `rpc_gsp_lockdown_notice_v17_00` (`S:1817`) | boot-safe list | boot | `:319` |
| GSP→CPU | GSP_POST_NOCAT_RECORD | 0x1020 (`:284`) | `rpc_gsp_post_nocat_record_v01_00` (`S:1783`) | boot-safe list | any | `:323` |

**Count: 20 rows.** That is 11 CPU→GSP functions (9 needed on our path; ALLOC_MEMORY is not on the path; UPDATE_BAR_PDE is NOT_VERIFIED), 1 bidirectional CONTINUATION_RECORD, and 8 GSP→CPU events. `RECOVERY_ACTION` 0x1022 (`:286`) exists but is not wired here: NOT_VERIFIED.

### 2.3 Functions that the GSP client does NOT send (vGPU-guest only). Do not implement.

Under the GSP client, `IS_FW_CLIENT` (`N/generated/g_gpu_nvoc.h:5689`) sends the `NV_RM_RPC_*` macros to the physical RMAPI, and that RMAPI is `GSP_RM_ALLOC`/`GSP_RM_CONTROL`:
- `ALLOC_SHARE_DEVICE` → FWCLIENT macro `rpc.h:48-~100`, which allocates NV01_ROOT on the first device, then the device.
- `ALLOC_OBJECT` → `rpc.h:271-290`.
- `ALLOC_SUBDEVICE` → `rpc.h:367-375`.
- `CONTROL` → `rpc.h:223-230`.

So these dedicated function numbers never appear on the GSP-client wire: [INFERENCE] from the macro bodies; grep for `NV_RM_RPC_MAP_MEMORY_DMA` in `N/src/kernel` finds no caller.

| Name | Number (enum line) |
|---|---|
| ALLOC_CHANNEL_DMA | 6 (`:16`) |
| ALLOC_OBJECT | 9 (`:19`) |
| MAP_MEMORY_DMA | 14 (`:24`) |
| UNMAP_MEMORY_DMA | 15 (`:25`) |
| ALLOC_SUBDEVICE | 19 (`:29`) |
| ALLOC_SHARE_DEVICE | 32 (`:42`) |
| SET_PAGE_DIRECTORY | 54 (`:64`) |
| UNSET_PAGE_DIRECTORY | 79 (`:89`) |
| CTRL_GPFIFO_SCHEDULE | 97 (`:107`) |
| CTRL_GPFIFO_GET_WORK_SUBMIT_TOKEN | 186 (`:196`) |

---

## 3. Serving-surface map (CUDApp `nvrm/nvrm.c`, a520cd30)

Legend:
- **(a)** = becomes this GSP RPC.
- **(b)** = CPU-side work our driver must do itself.
- Both = both.

All RM-control routing claims rest on the export table flags word lacking `RMCTRL_FLAGS_ROUTE_TO_PHYSICAL` = 0x40 (`N/inc/kernel/rmapi/control.h:233`). Such a control is implemented in kernel RM. Kernel RM may then issue its own RPC.

| # | Surface item | Disposition | Evidence (MEASURED unless marked) |
|---|---|---|---|
| 1 | NV_ESC_CARD_INFO | (b) enumerate the PCI device (sysfs/VFIO) | `KO/nv.c:2523`; ioctl `kernel-open/common/inc/nv-ioctl-numbers.h:31` |
| 2 | NV_ESC_REGISTER_FD | (b) fd bookkeeping | `N/arch/nvalloc/unix/src/escape.c:844` |
| 3 | RM_ALLOC NV01_ROOT_CLIENT | (b) local client. (a) GSP_RM_ALLOC hClass NV01_ROOT is sent lazily with the first device | `resource_list.h:64`; `rpc.h:48-~100` (`bFirstDevice`, `AllocWithHandle(..., NV01_ROOT, ...)`) |
| 4 | RM_ALLOC NV01_DEVICE_0 | (a) GSP_RM_ALLOC | `gpu/device.c:185` → `rpc.h:198-206` FWCLIENT → `rpc_common.c:74-79` |
| 5 | RM_ALLOC NV20_SUBDEVICE_0 | (a) GSP_RM_ALLOC | `gpu/subdevice/subdevice.c:124` → `rpc.h:367-375` |
| 6 | RM_ALLOC NV01_MEMORY_VIRTUAL | (a) GSP_RM_ALLOC (VA range object) | `mem_mgr/virt_mem_range.c:127` → `rpc.h:271-290` |
| 7 | RM_ALLOC NV01_MEMORY_SYSTEM | **(b)**: pin pages and make an IOVA (VFIO_IOMMU_MAP_DMA, or the iommufd equivalent). No RPC under the GSP client; the `NV_RM_RPC_ALLOC_MEMORY` call sits inside `if (IS_VIRTUAL(pGpu))` | `mem_mgr/system_mem.c:462-490` |
| 8 | RM_ALLOC FERMI_VASPACE_A | Both. (a) GSP_RM_ALLOC (skipped for FLA/BAR1/host VAS). (b) CPU builds the page tables. The server-reserved PDEs are copied to GSP with `NV90F1_CTRL_CMD_VASPACE_COPY_SERVER_RESERVED_PDES` (GSP-client gate `gpu_vaspace.c:4048`, call `:4062`). The wire for that control is GSP_RM_CONTROL [INFERENCE] | `gpu/mem_mgr/vaspace_api.c:227,250-263`; `mem_mgr/gpu_vaspace.c:4040-4062` |
| 9 | RM_ALLOC KEPLER_CHANNEL_GROUP_A | (a) GSP_RM_ALLOC, issued explicitly by the constructor (the resource entry `resource_list.h:464` carries no RPC flag) | `gpu/fifo/kernel_channel_group_api.c:409,425-428` |
| 10 | RM_ALLOC FERMI_CONTEXT_SHARE_A | (a) GSP_RM_ALLOC (generic path) | `resource_list.h:406,413` RPC_TO_ALL; generic `rmapi/alloc_free.c:766,871` |
| 11 | RM_ALLOC BLACKWELL_CHANNEL_GPFIFO_A | Both. (a) GSP_RM_ALLOC via `_kchannelSendChannelAllocRpc`. (b) CPU allocates the instance block, USERD, RAMFC and error notifier, then puts their PAs into the alloc params | `gpu/fifo/kernel_channel.c:951-955`; PA fill `:562-575`. On GB20B, `BLACKWELL_CHANNEL_GPFIFO_B` also exists (`g_gpu_class_list.c:3634`) |
| 12 | RM_ALLOC BLACKWELL_COMPUTE_B | (a) GSP_RM_ALLOC (generic channel-descendant RPC). (b) the GR context buffers are allocated by the client RM when `gpuIsClientRmAllocatedCtxBufferEnabled` | class entry `resource_list.h:2114`; compute entries carry `RS_FLAGS_CHANNEL_DESCENDANT_COMMON` = GPUS_LOCK \| RPC_TO_ALL (`resource_desc_flags.h:41,45`; flags `resource_list.h:2109,2120`; which flags line belongs to 2114 is NOT_VERIFIED); `gpu/gr/kernel_graphics_object.c:507,623`. **GB10B lacks this class** (`g_gpu_class_list.c:2426-2427` vs `:3635`) |
| 13 | RM_ALLOC HOPPER_USERMODE_A (doorbell) | **(b)**: map the usermode page. BAR0 `NV_VIRTUAL_FUNCTION` window 0x30000-0x3FFFF (priv 0x0-0x2FFFF), or BAR1 when it is available (iGPU/coherent → BAR0 path). The doorbell write is the work-submit token to `NVC361_NOTIFY_CHANNEL_PENDING` 0x90. No RPC | `gpu/fifo/usermode_api.c:84-100`; `gpu/fifo/arch/hopper/kernel_fifo_gh100.c:111-131` (HAL for GB10B and GB20B, `g_kernel_fifo_nvoc.c:527`); `C/inc/swref/published/blackwell/gb100/dev_vm.h:27,31`; `C/sdk/nvidia/inc/class/clc361.h:33`; UVM rings it at `U/uvm_volta_host.c:44`. GB20B also lists `BLACKWELL_USERMODE_A` (`g_gpu_class_list.c:3659`) |
| 14 | CTRL NV0000 GPU_GET_ID_INFO_V2 | (b) kernel-RM local | `rmapi/client_resource.c:1405`; flags 0x109 (`g_client_resource_nvoc.c:811`) |
| 15 | CTRL NV0080 GPU_GET_CLASSLIST | (b) local class list | `gpu/device_ctrl.c:56` → `gpuGetClassList`; GB20B list `g_gpu_class_list.c:3620-3727` |
| 16 | CTRL NV2080 GPU_GET_GID_INFO | (b) local `gpuGetGidInfo`. Where the GID bytes come from (static info): NOT_VERIFIED | `gpu/subdevice/subdevice_ctrl_gpu_kernel.c:3627-3640`; flags 0x10118 (`g_subdevice_nvoc.c:569`) |
| 17 | CTRL NV2080 GR_GET_INFO | (b) served from the cached `KGRAPHICS_STATIC_INFO`, filled once at init by (a) GSP_RM_CONTROL `NV2080_CTRL_CMD_INTERNAL_STATIC_KGR_GET_INFO` | `gpu/gr/kernel_graphics.c:3004,3036,1231` |
| 18 | CTRL NVA06C GPFIFO_SCHEDULE | (a) GSP_RM_CONTROL (`IS_VIRTUAL \|\| IS_GSP_CLIENT` → `NV_RM_RPC_CONTROL`) | `gpu/fifo/kernel_channel_group_api.c:1065,1176-1183`; flags 0x10008 (`g_kernel_channel_group_api_nvoc.c:211`) |
| 19 | CTRL NVC36F GET_WORK_SUBMIT_TOKEN | **(b)**: `kfifoGenerateWorkSubmitToken` on the CPU, then a notifier write. The RPC only fires for vGPU | `gpu/fifo/kernel_channel.c:3302,3316-3317,3362-3367`; flags 0x10008 (`g_kernel_channel_nvoc.c:539`). Token formula HAL: NOT_VERIFIED |
| 20 | RM_FREE | (a) FREE (10) for objects that went to GSP; (b) local free otherwise | `R:11311` |
| 21 | RM_MAP_MEMORY (CPU mapping) | (b) mmap of sysmem or BAR0/BAR1 | `escape.c:507`; `rmapi/mapping_cpu.c:120` |
| 22 | RM_MAP_MEMORY_DMA / UNMAP | **(b) page tables**: `dmaAllocMapping_GM107` → `gvaspaceMap` → `mmuWalkMap` (the CPU writes PDE/PTE), plus a TLB invalidate. The invalidate mechanism under the GSP client is NOT_VERIFIED | `escape.c:624,640`; `gpu/mem_mgr/arch/maxwell/virt_mem_allocator_gm107.c:319`; `mem_mgr/gpu_vaspace.c:2190,2239` |
| 23 | UVM_INITIALIZE / UVM_MM_INITIALIZE | (b) | `U/uvm.c:927`, `:59` |
| 24 | UVM_REGISTER_GPU | Both. (a) the RM objects via `nvUvmInterfaceRegisterGpu`/`DeviceCreate`/`GetGpuInfo` (GSP_RM_ALLOC/CONTROL). (b) **fault buffer**: `nvUvmInterfaceInitFaultInfo` → `NVB069_CTRL_CMD_FAULTBUFFER_GET_SIZE`/`GET_REGISTER_MAPPINGS`, `NVC369_..._REGISTER_REPLAY_BUF`/`NON_REPLAY_BUF`; ownership `NV2080_CTRL_CMD_MC_CHANGE_REPLAYABLE_FAULT_OWNERSHIP`. Then UVM reads and parses the buffer on the CPU (Blackwell parser) | `U/uvm_gpu.c:3801,2927,1414,2932`; `U/uvm_gpu_replayable_faults.c:140,188,247,254`; `N/src/kernel/rmapi/nv_gpu_ops.c:9174,9225,9253,9273,9293,2382,2391`; `U/uvm_blackwell_fault_buffer.c:34,95` |
| 25 | UVM_REGISTER_GPU_VASPACE | Both. (a) `DupAddressSpace` (DUP_OBJECT/ALLOC), `SetPageDirectory` → `NV0080_CTRL_CMD_DMA_SET_PAGE_DIRECTORY`. (b) **UVM owns the page tree** with the integrated-GPU MMU mode for both GB10B and GB20B. Only one iGPU per VA space ("T264/GB10B" TODO) | `U/uvm_gpu.c:3831,1305`; `U/uvm_va_space.c:1531,815-819`; `nv_gpu_ops.c:8779,8871`; `U/uvm_mmu.c:1116`; `U/uvm_hal.c:433-445` |
| 26 | UVM_REGISTER_CHANNEL | Both. (a) `RetainChannel` (inside RM: GET_WORK_SUBMIT_TOKEN, `nv_gpu_ops.c:10242`) and `BindChannelResources` → `NV2080_CTRL_CMD_GPU_PROMOTE_CTX` (GSP_RM_CONTROL [INFERENCE]). (b) map the channel context buffers into the VA space from RM-supplied PTEs | `U/uvm_user_channel.c:738,141,467`; `nv_gpu_ops.c:10126,10827,10898`; `U/uvm_map_external.c:180` |
| 27 | UVM_CREATE_EXTERNAL_RANGE | (b) | `U/uvm_map_external.c:622` |
| 28 | UVM_MAP_EXTERNAL_ALLOCATION | Both. (a) `GetExternalAllocPtes` and `DupMemory` from RM. (b) the CPU writes PTEs into UVM's tree, then a TLB invalidate pushed on a channel (Blackwell host methods) | `U/uvm_map_external.c:1086,187,909,280`; `U/uvm_blackwell_host.c:31,83`; `U/uvm_hal.c:311-327` |
| 29 | UVM_FREE | (b) unmap, invalidate | `U/uvm_va_range.c:740` |
| 30 | UVM_UNREGISTER_{GPU,GPU_VASPACE,CHANNEL} | Both. (a) `UnsetPageDirectory` / FREE. (b) teardown | `U/uvm_gpu.c:3824,3842`; `U/uvm_user_channel.c:914`; `U/uvm_va_space.c:1431` |
| 31 | Interrupts | **(b)**: MSI-X through VFIO eventfds (8 vectors, lead's measurement). Service the GSP message-queue interrupt plus the replayable-fault top half; RM-owned interrupts are drained through `nvUvmInterfaceServiceDeviceInterruptsRM` | `U/uvm_gpu_isr.c:218,282,598`; `U/uvm_gpu.c:1342` |

**CPU-side (b) list for our driver:**
1. VFIO_IOMMU_MAP_DMA for every sysmem object (#7).
2. GMMU page tables for the RM VA space and the UVM page tree (#8, #22, #25, #28).
3. TLB invalidates (#22, #28).
4. Instance block, USERD, RAMFC and error notifier allocation, with their PAs passed to GSP (#11).
5. The usermode BAR0 page (0x30000 window) and the doorbell write of the token to +0x90 (#13).
6. Work-submit token generation (#19).
7. UVM channel registration, mapping context buffers into the VA space (#26).
8. Fault buffer read and parse, plus replay (#24).
9. MSI-X interrupts (#31).
10. CARD_INFO, REGISTER_FD and CPU mmap (#1, #2, #21).
11. The GSP queue memory itself and the CPU sequencer executor (§1 steps 3 and 17).

---

## 4. What nova / nouveau already prove (4 of 4 web operations used)

1. nova-core chip list (`https://raw.githubusercontent.com/torvalds/linux/master/drivers/gpu/nova-core/gpu.rs`, fetched 2026-10-26, master branch with no commit pin, sha256 `374867636b5480b507114e29a467e9c77c64e02d47bde2d8c0ed495507f2edb6`).
   - Blackwell entries: GB100 and GB102 (`:138-140`), and GB202/203/205/206/207 (`:141-146`). They map to `BlackwellGB10x` / `BlackwellGB20x` (`:163-165`).
   - **No GB10B and no GB20B.** Upstream nova lacks our chip (MEASURED on the fetched file).
2. nova-core r570 bindings (§2.2). They confirm the RPC numbers and the 48-byte element header for r570 only, not for the 580 firmware on this box.
3. nouveau on GB10: David Airlie's experimental patches, roughly 100 patches, rework nouveau for the R610 firmware with DGX Spark boot/display support. The source for this is **secondary**: The Linux Camp, 2026-09-09, `https://thelinuxcamp.com/news/nouveau-nvk-experimental-patches-enable-nvidia-dgx-spark-gb10-support-mttzmgbr`, citing Phoronix.
   - Not merged according to that article. Patch contents are NOT_VERIFIED.
   - The article does not say GB10B or GB20B, and does not say whether it uses the SEC2-COT path.
4. Nouveau r570 mainline GB10/GB20B support: NOT_VERIFIED (no fetch left). The L332 inventory reports that NVIDIA's `nouveau/extract-firmware-nouveau.py` knows gh100, gb100 and gb202 FMC images but not gb10b (inventory lines 148-149, citing `:655-662,775-781`). This was re-cited, not re-read.

Net: no upstream driver is shown to boot GB10B/GB20B through the SEC2 COT path. The 580 open source is the only cited reference for that path.

---

## 5. L332 check

L332 files: `/home/drakestapleton/workspace/aienos-own-gpu-driver/docs/GB10_FIRMWARE_INVENTORY.md` (859 lines) and `native/kernel/tools/ck_gsp_rpc_encode.c` (594 lines).

**Correct and reused** (my reads agree):
- Signature prefix HAL `g_kernel_gsp_nvoc.c:1809-1813`.
- BOOT_GSPFMC chip list `g_kernel_sec2_nvoc.c:446-450`.
- No-FSP list `g_kern_fsp_nvoc.c:227-236`; MEASURED by me: GB10B and GB20B are both absent from the FSP list.
- `WaitForGfwBootOk_ac1694` for GB10B | GB20B | GB20C.
- `PopulateWprMeta_GB10B` zeroes.
- COT payload 860 B with hash 48 / sig 384 / key 384.
- RPC numbers 71/72/73/76/103.
- Events 0x1001/0x1002/0x1004/0x101c/0x1020 (`ck_gsp_rpc_encode.c:145,459-460`).
- Element header 48 / max RPC 65488 (inventory line 702).

**Defect found (applies to GB10B, the chip L332 targets):**
- The inventory gives the GB10B MCTP dword as `0xC0000000` "(SOM=1, EOM=1 ...)" (inventory line 440, also lines 756-757 "MCTP dword `0xC0000000`").
- In 580.173.02, `ksec2CreateMctpHeader` for **GB10B** is bound to `_539ab4` (`g_kernel_sec2_nvoc.c:786-788`), which is `return 1;` (`g_kernel_sec2_nvoc.h:853-854`). So GB10B sends MCTP dword **0x00000001**. Only GB20B | GB20C build SOM/EOM/SEID/SEQ (`:790-792`, `kernel_sec2_gb20b.c:206-220`). The call site is `kernel_sec2.c:319-321`.
- Fixing `ck_gsp_cot_encode` / `gsp_boot_build` in L332's tree is outside this task's write scope: lane L332 owns `feat/own-gpu-driver`. Reported to the lead.

**Every L332 GB10B assumption where GB20B differs** (all MEASURED, 580 source):

| # | Item | GB10B (L332's value) | GB20B | Source |
|---|---|---|---|---|
| 1 | Arch | arch 0x1A0 | 0x1B0 (impl 0x0B both) | `nv_arch.h:51,53,111,127`; inventory line 68 |
| 2 | Firmware family / signature section | `.fwsignature_gb10y` | `.fwsignature_gb20y` | family-select `:60` / `:66`; inventory line 89 |
| 3 | Bindata archives | `GspRmBoot_GB10B`, `GspRmFmcGfwProdSigned_GB10B` (the inventory's eight arrays and their sha256 values) | the `_GB20B` archives. GB20B's sha256 values were not measured: NOT_RUN | `g_kernel_gsp_nvoc.c:790/794`, `:926/930`; inventory lines 107-120 |
| 4 | FWSEC | "no FWSEC step" | the driver waits ≤4 s for FWSEC's `SCRATCH_RESET_PLUS_2 == 0xFF` | `kernel_sec2_gb20b.c:540-575`; inventory lines 26-28, 180-181, 197 |
| 5 | COT version | 1 | 2 | `g_kernel_sec2_nvoc.c:481/485`; inventory line 453 |
| 6 | `frtsSysmemSize` | image size (195,912) | 0 (not set) | `kernel_sec2_gb10b.c:508` vs `kernel_sec2_gb20b.c:729-827`; inventory line 457 |
| 7 | MCTP header | constant 1 (see defect above) | SOM\|EOM\|SEID\|SEQ fields | `g_kernel_sec2_nvoc.c:786-792` |
| 8 | Completion | fire-and-forget, then MAILBOX0 ≠ `0xc001cafe` | `ksec2SendAndReadMessage`: read the EMEM response, check the vendor and the SEC2_RESPONSE error code | `kernel_sec2_gb10b.c:622,640`; `kernel_sec2_gb20b.c:1044,293-361,1113-1136`; inventory line 470 |
| 9 | SafeToSend | BOOT_GSPFMC property only | also silicon, IS_MISSING and DISABLE_GSPFMC checks | `kernel_sec2_gb10b.c:414-427` vs `kernel_sec2_gb20b.c:892-937`; inventory lines 775-777 |
| 10 | `PDB_PROP_KSEC2_RM_BOOT_GSP` | TRUE | FALSE | `g_kernel_sec2_nvoc.c:464,469` |
| 11 | CC archive | GB100 variant; GB10B SEC2 has no CC branch | `CcFmcGfwProdSigned_GB202` (`nvoc:961`) plus a CC branch in `_ksec2GetGspUcodeArchive` (`kernel_sec2_gb20b.c:608`) | |
| 12 | SEC2 receive/read HALs | stubs | `ReadPacket_GB20B` (`g_kernel_sec2_nvoc.c:827`), `WaitForSecureBoot_GB20B` (`:860`), `NvdmToSeid_GB20B` (COT→SEID 0, same value; `kernel_sec2_gb20b.c:170-190`) | |
| 13 | Published register headers | `C/inc/swref/published/blackwell/gb10b/` (inventory lines 233, 473) | a separate `blackwell/gb20b/` directory (dev_boot, dev_ce_base, dev_falcon_v4, dev_gsp, dev_mmu, dev_perf, dev_sec_pri, dev_xtl_ep_pcfg_gpu, kind_macros). The register offsets were not diffed: NOT_VERIFIED | `ls` of both dirs |
| 14 | Compute class | `BLACKWELL_COMPUTE_A` only | `BLACKWELL_COMPUTE_B` (CUDApp's class), plus `BLACKWELL_CHANNEL_GPFIFO_B` and `BLACKWELL_USERMODE_A` | `g_gpu_class_list.c:2425-2446` vs `:3633-3677` |

**Same on both** (L332's values carry over): Bootstrap_GH100, SetupGspFmcArgs_GH100, ResetHw_GB100, PopulateWprMeta_GB10B (heap and ZERO_FB: `g_gpu_nvoc.c:212-218` lists GB10B | GB20B | GB20C), SendPacket_GB10B (EMEM write), GfwBootOk no-op, PREINITIALIZED_WPR_REGION, the queue/RPC framing, and the usermode HAL `kfifoConstructUsermodeMemdescs_GH100`.
