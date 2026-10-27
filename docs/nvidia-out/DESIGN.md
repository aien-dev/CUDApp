> **WARNING: This can and may permanently damage your system. No warranty of any kind. Do not use it unless you are a professional lab that is prepared to lose the hardware.**
>
> This notice covers every part of this work that would hand NVIDIA's signed boot images to the GPU (the SEC2 → GSP-FMC boot). Nothing in this branch does that; the code here only parses files on disk.

# Taking NVIDIA's kernel code out of the DGX Spark: design and window review

Date: 2026-10-26. Author: NoNvDriver lane (boot-up team). Builds on L332 (plan `~/handoffs/2026-10-24-nvidia-driver-out-omega.md`,
aien-dev/aienux branch `feat/own-gpu-driver` c9678d7b); does not replace it.
Every claim is labelled MEASURED (with the path or command), MODELED, or NOT_RUN. No GPU device, BAR, `/dev/nvidia*` or
`/dev/vfio*` was opened for this document, and no module was loaded or unloaded.

## Boot-chain policy (Drake, 2026-10-26, verbatim intent)

- **This Spark: the signed boot chain is NEVER touched.** Signed GSP/booter/FMC firmware, VBIOS, fuses, flash: no writes, ever.
- **Future: replacing the signed boot chain happens only if someone sponsors a dedicated DGX Spark.**
- **Boot-chain code is experimental and carries this notice at the top: "WARNING: This can and may permanently damage your system. No warranty of any kind. Do not use it unless you are a professional lab that is prepared to lose the hardware."**
- **Boot-chain research stays read-only: parsing and documentation only.**

Our driver may *load* NVIDIA's signed firmware unmodified, as nouveau and nova do. That load is itself the risky step: without
`nvidia.ko`, our code must run the SEC2 → GSP-FMC boot. Drake decides before any run.

## Layers, boot upward

| Layer | NVIDIA code present today | Replacement | Status |
|---|---|---|---|
| L0 firmware / boot chain | Yes: AMI BIOS 5.36_0ACUM018 on NVIDIA board P4242, VBIOS 9a.0b.2d.00.00, GSP firmware 580.173.02 (MEASURED, LAYERS-0-3.md) | None, by policy | Out of scope (read-only) |
| L1 bootloader | shim 1.58 and GRUB 1.202.5 are Canonical builds; 9 NVIDIA-shipped GRUB settings files set the NVIDIA-specific command line (MEASURED, LAYERS-0-3.md) | Drop the NVIDIA settings files once L2/L3 no longer need them | PROPOSED |
| L2 kernel | `linux-image-7.0.0-1019-nvidia`, Canonical-built on upstream 7.0.14, 89 config lines differ from generic; nouveau and Rust removed (MEASURED, LAYERS-0-3.md). Source patches: NOT_RUN | A generic or mainline kernel | NOT_RUN: nobody has booted one on a Spark |
| L3 kernel drivers | 12 loaded modules NVIDIA/Mellanox-authored, 4 of them the GPU driver (nvidia, nvidia_uvm, nvidia_modeset, nvidia_drm 580.173.02); built-in nvidia-ffa, tegra241-cmdqv (MEASURED, LAYERS-0-3.md) | GPU: vfio-pci (upstream Linux) + L4. Others: PROPOSED per row in LAYERS-0-3.md | GPU: PROPOSED; window 0 never ran |
| L4 GPU driver | nvidia.ko does GSP boot, memory, channels | Seeded Omega in user space over vfio-pci (route a). First host milestone: `.om` parser for the GSP firmware image (`native/gsp/fwimage.om`) and the GSP RPC table (`GSP-RPC-TABLE.md`) | Parser: see appendix; boot: NOT_RUN |
| L5 user space | CUDA++ uses no NVIDIA user-space library (only libc/libm, MEASURED in the CUDApp qualification lane); it talks to nvidia.ko through `/dev/nvidia*` ioctls | Same CUDA++ kernels, driven through L4 instead of the NVRM ioctls | PROPOSED |

**Wording rule.** "GPU driver replaced" must not be read as "no NVIDIA code on the Spark": L0 stays NVIDIA's by policy, and L1–L3 keep
NVIDIA-written settings, kernel config and built-in drivers until each row above moves.

## Route choice for L4

- **(a) Omega in user space over vfio-pci: recommended.** First-party code stays `.om`; the kernel side is upstream vfio-pci.
  Two facts L332's plan does not have yet (MEASURED, `/boot/config-7.0.0-1019-nvidia`):
  `CONFIG_VFIO_CONTAINER` is **not set**; `CONFIG_IOMMUFD=m`, `CONFIG_IOMMUFD_VFIO_CONTAINER=y`, `CONFIG_VFIO_DEVICE_CDEV=y`.
  So the legacy `/dev/vfio/vfio` type-1 container that the plan's request list uses (`_IO(';', 100+n)`, SET_IOMMU,
  GROUP_SET_CONTAINER) exists only as iommufd's compatibility layer, and only once `iommufd` is loaded. The Omega checked
  device calls should target the cdev + iommufd API (`/dev/vfio/devices/vfioN`, `/dev/iommu`, IOMMU_IOAS_ALLOC/MAP,
  VFIO_DEVICE_BIND_IOMMUFD) and the QEMU fixture must use the same kernel options, or it tests a path the Spark lacks.
- **(b) nouveau / nova: not usable now.** The running kernel builds neither (nouveau and Rust removed in the config,
  MEASURED); their support for this chip is unverified.
- **(c) our own kernel module:** L332's `aien_gpu.ko` (C) is used in windows 0 and 1 as a narrow bootstrap shim. It is a
  deviation from `.om`-first and should be retired once (a) can do the same reads.

## Chip identity: GB10B or GB20B? (blocks window 1)

L332's package and docs assume the chip is **GB10B**. In the 580.173.02 source:

- `nv_ref.h:153,170-171,182-183`: CHIP_ID = PMC_BOOT_42 bits 29:20; architecture 0x1A = GB100, 0x1B = GB200; implementation
  0xB in both. So GB10B = 0x1AB, GB20B = 0x1BB.
- `uvm_va_space.c:817` names GB10B "T264/GB10B" (Thor), and GB10B is the only chip marked `CLKS_IN_TEGRA_SOC`.
- The Spark reports compute capability sm_121 (MEASURED by CUDA++ builds); Thor is sm_110.

- CUDA++ (`nvrm/nvrm.c:353-360`) refuses to run unless the driver's class list contains `BLACKWELL_COMPUTE_B`, and it has
  passing chip receipts on this Spark (MEASURED: `evidence/FB1-CUT5-HD128-7dd92dc/FB1-CUT5-7dd92dc/receipt.json`,
  device "NVIDIA GB10, 580.173.02"). In 580.173.02 the GB10B class list has `BLACKWELL_COMPUTE_A` only
  (`g_gpu_class_list.c:2418-2487`); `COMPUTE_B` is listed only for GB202–GB207 (discrete), GB20B and GB20C.

INFERENCE (strong, two independent source facts plus a chip receipt): the Spark is GB20B (CHIP_ID 0x1BB) or GB20C (0x1BC),
not GB10B. Not measured: PMC_BOOT_42 has never been read (window 0 reads and logs it).
The GB20B boot images differ from GB10B (MEASURED: `g_bindata_kgspGetBinArchiveGspRmBoot_GB20B.c` 2,671,061 B vs GB10B
2,466,261 B; the FMC images compare different). L332's `window1-package.sh` and `ck_gsp_bindata_extract.c` hardcode GB10B and
`.fwsignature_gb10y`.

**Fix applied (aienux `feat/nvidia-out-layers`):** `window1-pre.sh` now refuses unless window 0's log holds a
PMC_BOOT_42 value with CHIP_ID 0x1AB. EXERCISED on synthetic logs: 0x1ab → pass, 0x1bb → refused, no value → refused.
If window 0 reads 0x1BB or 0x1BC, window 1 waits until GB20B images and the `gb20y` signature are packaged.

**Second defect fixed (aienux `feat/nvidia-out-layers`):** L332's SEC2 packet used the MCTP dword `0xC0000000`, which is the
GB20B form. On GB10B the 580.173.02 driver sends the constant `0x00000001` (`kernel_sec2.c:320` → `ksec2CreateMctpHeader_539ab4`
= `return 1;`, `g_kernel_sec2_nvoc.c:786-788`, `g_kernel_sec2_nvoc.h:853-855`). So the window-1 package was a hybrid that matched
neither chip. Builder, simulator, COT tool and tests now use `0x00000001` for GB10B (EXERCISED host-only: COT selftest PASS,
test_gb10_gsp_boot 257 checks PASS, test_gsp_mmio 67 PASS, nvhdr 167 PASS, mutants killed; negative control: putting
`0xC0000000` back in the builder fails test_gb10_gsp_boot:197). Found by the GSP-RPC-TABLE review (`GSP-RPC-TABLE.md`).

## Window procedure review (L332 rung 2)

| # | Risk in the reviewed scripts | Change made | Status |
|---|---|---|---|
| 1 | Hold-back file in `/etc/modprobe.d` survives a crash or power loss; every later boot comes up without NVIDIA until someone deletes it by hand | One-shot GRUB entry (`grub-reboot`), GRUB default untouched; hold-back file written to `/run/modprobe.d` (tmpfs) only in that boot | IMPLEMENTED; `window-boot.sh check` EXERCISED (PASS, read-only); install/arm NOT_RUN |
| 2 | No watchdog: a hung window boot needs a wall power cycle | `systemd.watchdog_sec=30` on the window entry; SBSA Generic Watchdog present (MEASURED `/sys/class/watchdog/watchdog0`) | IMPLEMENTED, NOT_RUN. Whether an SBSA reset also resets GSP/SEC2: NOT_RUN |
| 3 | `kernel.panic` only set partway through part 2 | `panic=10 oops=panic` on the window command line | IMPLEMENTED, NOT_RUN |
| 4 | No time limit on the window | 25-minute lease unit reboots the window boot | IMPLEMENTED, NOT_RUN |
| 5 | Window 1 hands SEC2 GB10B images without checking the chip | Chip-identity gate (above) | EXERCISED on synthetic logs |
| 6 | Initramfs check passes silently if `lsinitramfs` fails (pipeline into `grep -q`) | Fail closed in window0-pre, window1-pre, window-boot.sh | IMPLEMENTED; current initramfs has no nvidia `.ko` (MEASURED, 4,173 entries listed) |
| 7 | Recovery by reset has never been tested; window 1 can leave GSP half started | Optional FLR test in window 0 (`AIEN_W0_FLR=1`): with no driver bound, reset via the kernel's `reset_method`, then the stock driver must bind and answer | IMPLEMENTED, NOT_RUN; Drake decides whether window 0 runs it |
| 8 | Window 0 deleted the hold-back first thing, so something could load NVIDIA beside `aien_gpu` | Hold-back kept until `aien_gpu` is removed (window 0) and for the whole boot (window 1); post scripts refuse to run outside the window boot | IMPLEMENTED |
| 9 | Display | The window boot has no `nvidia_drm`; only simpledrm card0 remains (INFERENCE from LAYERS-0-3.md L3). Run headless over ssh | Noted |

Receipts follow the Q27 window standard (`~/handoffs/Q27/q27-window.sh`): every step timed into the window log, the log
committed pass or fail, restore checked by `window1-check.sh` (now also: not the window boot, no pending `next_entry`).

**Still open and stated plainly.** Window 1 runs NVIDIA's signed SEC2 → GSP-FMC boot from our code. The chip checks the
signatures, but a wrong sequence can leave GSP stopped until a reset, and whether a warm reboot resets it is NOT_RUN (L332
RUNBOOK). The last resort is a power cycle at the wall by Drake. Drake picks whether and when.

## Technical appendix

- Source tree: `~/workspace/_ext/open-gpu-kernel-modules-580.173.02` at 20e4e6e1.
- Firmware image: `/lib/firmware/nvidia/580.173.02/gsp_ga10x.bin`, 75,012,080 B (MEASURED `stat`).
- Boot facts for the window entry (MEASURED): `/etc/default/grub` GRUB_DEFAULT=0, GRUB_TIMEOUT=0, hidden menu;
  `/etc/grub.d/00_header` honours `next_entry`; `/etc/grub.d/41_custom` sources `custom.cfg`; `/boot/grub` on the ext4 root;
  systemd 255 supports `systemd.watchdog_sec=`; kmod reads `/run/modprobe.d`.
- Layer inventory: `LAYERS-0-3.md`. GSP messages and boot sequence: `GSP-RPC-TABLE.md`. Parser: `native/gsp/README.md`.
- aienux changes: `evidence/own-gpu-driver/rung2/{window-boot.sh,window-boot/*.service,window0-pre.sh,window0-post.sh,window1-pre.sh,window1-post.sh,window1-check.sh,RUNBOOK.md}`.
