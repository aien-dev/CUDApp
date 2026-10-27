# NVIDIA code on the Spark: layers 0-3 (firmware, bootloader, kernel flavour, kernel drivers)

Host `spark-b87b`, read-only inventory taken 2026-10-26 by lane NoNvDriver.BootLayers.
No device access was performed: no `/dev/nvidia*`, `/dev/vfio*` or BAR opens, no bind/unbind, no module loads, no sudo.
Labels: **MEASURED** (command or path given), **INFERENCE** (reasoned, not observed), **NOT_RUN** (not checked, reason given).
Status column: PROPOSED / IMPLEMENTED / EXERCISED / QUALIFIED. `n/a (policy)` means Drake's rule forbids replacing it on this Spark.

Short forms used in the source column:
- `dpkg-s X` = `dpkg -s X | grep -E '^(Package|Version|Maintainer|Source)'`
- `apt-policy X` = `apt-cache policy X`
- `mi X` = `modinfo X` (fields: filename, author, license, version, signer)
- `rc` = refcount, column 3 of `/proc/modules`

---

## Layer 0: firmware and signed boot chain (READ-ONLY, never modified on this Spark)

Platform (MEASURED, `/sys/class/dmi/id/*`): sys_vendor `NVIDIA`, product_name `NVIDIA_DGX_Spark`, product_family `DGX Spark`, product_version `A.7`, board `P4242` rev `A04`, bios_vendor `American Megatrends International, LLC.`, bios_version `5.36_0ACUM018`, bios_date `08/06/2025`. Kernel log line: `Hardware name: NVIDIA NVIDIA_DGX_Spark/P4242, BIOS 5.36_0ACUM018 08/06/2025` (`journalctl -k -b`).
UEFI: `/sys/firmware/efi` present with `efivars esrt fw_platform_size mok-variables systab`; `fw_platform_size` = 64 (MEASURED). `mokutil --sb-state` → `SecureBoot disabled` (MEASURED).

| Component | Who wrote it | NVIDIA-written? | Replacement route | Status |
|---|---|---|---|---|
| UEFI system firmware (AMI Aptio-class BIOS 5.36_0ACUM018) | AMI (bios_vendor) for NVIDIA (board/sys vendor NVIDIA). Source: `/sys/class/dmi/id/bios_vendor`, `bios_version`, `board_vendor` (MEASURED) | partly (AMI codebase, NVIDIA platform build) | none: boot chain is never touched on this Spark (Drake, 2026-10-26) | n/a (policy) |
| ESRT firmware-update entries | 4 entries `entry0..entry3` under `/sys/firmware/efi/esrt/entries` (MEASURED); `fw_class`/`fw_version` are root-only ("Permission denied") | NOT_RUN (contents unreadable without root) | none | n/a (policy) |
| `EFI/UpdateCapsule` on ESP | directory exists and is empty (`ls -la /boot/efi/EFI/UpdateCapsule`, MEASURED); kernel has `CONFIG_EFI_CAPSULE_LOADER=y` (generic: `=m`) | no content present | none; leave capsule path alone | n/a (policy) |
| SoC firmware below UEFI (BPMP/MB1/TF-A-class, fuses) | not visible from Linux userspace | NOT_RUN (no read path without root/device access) | none | n/a (policy) |
| GPU VBIOS `9a.0b.2d.00.00` | NVIDIA. Source: `/proc/driver/nvidia/gpus/000f:01:00.0/information` → `Video BIOS: 9a.0b.2d.00.00`, `GPU Firmware: 580.173.02`, `Model: Unknown` (MEASURED) | yes | none: only loaded/used as shipped | n/a (policy) |
| GSP-RM firmware `gsp_ga10x.bin` (75,012,080 B) and `gsp_tu10x.bin` (30,471,256 B) in `/lib/firmware/nvidia/580.173.02/` | NVIDIA binary, packaged by Ubuntu: `nvidia-firmware-580-580.173.02` 580.173.02-0ubuntu0.24.04.1, Source `nvidia-graphics-drivers-580`, Maintainer `Ubuntu Core Developers` (`dpkg -S`, `dpkg-s`, MEASURED). `modinfo nvidia` lists both as `firmware:` | yes (signed) | our driver LOADS it unmodified (nouveau/nova model); the loader is first-party .om | PROPOSED |
| Other `/lib/firmware/nvidia/*` (gk20a…gb202, tegra124–194; e.g. `gb100/gsp`, `gb202/gsp` hold `bootloader-570.144`, `fmc-570.144`, `gsp-570.144`) | `linux-firmware-nvidia-graphics` (`dpkg -S /lib/firmware/nvidia/gb202/gsp`); linux-firmware 20240318.git3b128b60.0ubuntu3.1, Maintainer `Ubuntu Kernel Team` (MEASURED). These are nouveau/nova-format files; there is no `gb10b`/`gb20b` directory (`ls /lib/firmware/nvidia`, MEASURED) | yes (NVIDIA blobs redistributed upstream) | not used by the running stack; no GB10 entry exists, so a nova-style loader would have to unpack the GB10B/GB20B images from `gsp_ga10x.bin` or the bindata below | PROPOSED |
| GSP boot/FMC images embedded in `nvidia.ko` (bindata) | NVIDIA. Source tree has 54 `g_bindata_*` files, including `g_bindata_kgspGetBinArchiveGspRmBoot_{GB10B,GB20B}.c` and `…GspRmFmcGfw{Prod,Debug}Signed_{GB10B,GB20B}.c` (`find …/open-gpu-kernel-modules-580.173.02 -name 'g_bindata_*'`, MEASURED). Running `nvidia.ko` is 19,231,198 B (`ls -la /lib/modules/7.0.0-1019-nvidia/kernel/nvidia-580-open/`, MEASURED) | yes (signed binary payloads) | extract and load unmodified; which pair applies depends on the unresolved GB10B vs GB20B identity (PMC_BOOT_42 read, needs a window) | PROPOSED |
| ConnectX-7 (mlx5) firmware | NVIDIA/Mellanox, on the NIC flash. **The CX7 is not enumerated right now**: `lspci -nn` shows 10 functions, none Mellanox; `/sys/bus/pci/drivers/mlx5_core/` has no bound device; `/sys/class/infiniband` is empty (MEASURED) | yes | none (NIC flash, not ours to rewrite) | n/a |
| Realtek r8127 NIC (`0007:01:00.0 [10ec:8127]`), MediaTek MT7925 Wi-Fi/BT (`0009:01:00.0 [14c3:7925]`) firmware | Realtek / MediaTek (`lspci -nn`, MEASURED) | no | out of scope (not NVIDIA) | n/a |

Layer 0 count: 9 rows; NVIDIA-written yes: 5, partly: 1, no: 1, NOT_RUN: 2. Replaceable under policy: 0. Our only planned contact is *loading* the GSP/FMC images (3 rows PROPOSED).

---

## Layer 1: bootloader and ESP

ESP `/boot/efi/EFI` (`ls -la`, MEASURED): `AIENOS/`, `BOOT/`, `UpdateCapsule/` (empty), `ubuntu/`.

`efibootmgr -v` (readable without root, MEASURED): `BootCurrent: 0001`, `BootOrder: 0001,0003,0004`, Timeout 1 s.
- `Boot0000* AIENOS Allen` → `\EFI\AIENOS\ALLEN.EFI` (not in BootOrder)
- `Boot0001* ubuntu` → `\EFI\ubuntu\shimaa64.efi` (current)
- `Boot0002* AIENOSRECOV` → `\EFI\BOOT\BOOTAA64.EFI` on GPT partition `f227357c-…` (not in BootOrder)
- `Boot0003* UEFI: PXE IPv4 Realtek PCIe 10 GBE`
- `Boot0004* UEFI: USB USB Hard Drive, Partition 1` → partition `f227357c-…` (same partition as Boot0002)
- `Boot0005* AIENOS Core RO` → `\EFI\AIENOS\CORE.EFI` (not in BootOrder)

`EFI/AIENOS/STAGED.TXT` (MEASURED, read only): `aienos_commit 26e929f3…`, `image_sha256 216952c3…b8bd`, staged 2026-10-20T22:55:59Z, `aienos_boot_entry: 0004`, `recovery_entry: Boot0002`. `sha256sum EFI/AIENOS/aienos-handoff.efi` = `216952c35e325a71385feb1668f7e72d76e4a4017bc3d85591d0281f9167b8bd`, matching the recorded hash (MEASURED). Discrepancy to note: STAGED.TXT names Boot0004 as the AIENOS entry, but Boot0004 now describes a USB hard-drive partition and no AIENOS ESP file is in BootOrder (MEASURED, both sources above). `BOOTREPORT.TXT` is 0 bytes.

| Component | Who wrote it | NVIDIA-written? | Replacement route | Status |
|---|---|---|---|---|
| shim `EFI/ubuntu/shimaa64.efi` = `EFI/BOOT/BOOTAA64.EFI` (sha256 `706f15b9…fe44`, both) | identical to `/usr/lib/shim/shimaa64.efi.signed` (alternatives link → `.signed.latest`, same sha). Package `shim-signed` 1.58+15.8-0ubuntu1, Source `shim-signed (1.58)`, Maintainer `Steve Langasek`, archive `ports.ubuntu.com noble/main` (`sha256sum`, `dpkg -L`, `dpkg-s`, `apt-policy`, MEASURED) | no (rhboot shim, Canonical/Microsoft-signed) | first-party .om EFI loader (AIENOS) | PROPOSED |
| MokManager `mmaa64.efi`, fallback `fbaa64.efi` | `shim-signed` (`dpkg -S /usr/lib/shim/mmaa64.efi`, `fbaa64.efi`; sha256 match, MEASURED) | no | same as shim | PROPOSED |
| GRUB `EFI/ubuntu/grubaa64.efi` (sha256 `b9d9dd81…abfcc`) | identical to `/usr/lib/grub/arm64-efi-signed/grubaa64.efi.signed`; `grub-efi-arm64-signed` 1.202.5+2.12-1ubuntu7.3, Source `grub2-signed (1.202.5)`, Maintainer `Colin Watson`, `noble-updates/main` (MEASURED) | no | first-party .om loader | PROPOSED |
| `EFI/ubuntu/grub.cfg` stub (search UUID d27bfd26…, `configfile /boot/grub/grub.cfg`) and `BOOTAA64.CSV` | Ubuntu grub install (content MEASURED via `cat`) | no | n/a | none |
| `/boot/grub/grub.cfg` menu titles | root-only (`Permission denied`) | NOT_RUN | — | — |
| `/etc/default/grub`: `GRUB_DEFAULT=0`, `GRUB_TIMEOUT=0`, `GRUB_TIMEOUT_STYLE=hidden`, `GRUB_CMDLINE_LINUX_DEFAULT="console=tty0 console=ttyS0,921600"` | local config (MEASURED `grep`) | no | n/a | none |
| `grub.d/menu.cfg` (`GRUB_DISTRIBUTOR="DGX OS"`, `quiet splash`, theme `/boot/themes/dgx`) | `dgxstation-grub` 25.02-1, Source `dgx-grub`, Maintainer `cudatools <cudatools@nvidia.com>`, repo `repo.download.nvidia.com/baseos … noble-updates/dgx` (MEASURED) | yes (config) | drop with GRUB | PROPOSED |
| `grub.d/grubserial.cfg` (serial efi0 921600) | `nv-grubserial` 26.03-1, Source `nvhelpers`, Maintainer `cudatools@nvidia.com`, NVIDIA baseos repo (MEASURED) | yes (config) | drop with GRUB | PROPOSED |
| `grub.d/no-grubmenu.cfg` (timeout 0, hidden) | `nv-no-grubmenu` 26.01-1, Source `nvhelpers`, `cudatools@nvidia.com` (MEASURED) | yes (config) | drop with GRUB | PROPOSED |
| `grub.d/iommu.cfg` → `iommu.passthrough=0` | file unowned; written by `/var/lib/dpkg/info/nv-iommu.postinst`; `nv-iommu` 25.12-1, Source `nvhelpers`, `cudatools@nvidia.com` (`grep -l iommu …postinst`, `dpkg-s`, MEASURED) | yes (config) | keep the setting (needed for VFIO/IOMMU isolation); own it in our loader config | PROPOSED |
| `grub.d/disable-init-on-alloc.cfg` → `init_on_alloc=0` | unowned; written by `nvidia-disable-init-on-alloc.postinst`; package 24.10-1, Maintainer `dgx-dev@nvidia.com` (MEASURED) | yes (config) | decide per security trade-off in our loader | PROPOSED |
| `grub.d/nvidia-console-settings.cfg` (`console=tty0 plymouth.ignore-serial-consoles plymouth.use-simpledrm`) | `nvidia-console-settings` 25.07-1, `dgx-dev@nvidia.com` (dpkg info list, MEASURED) | yes (config) | own in loader config | PROPOSED |
| `grub.d/nvidia-earlycon.cfg` (`earlycon=uart,mmio32,0x16A00000`) | `nvidia-earlycon` 25.06-1, `dgx-dev@nvidia.com` (MEASURED) | yes (config; carries platform UART address) | own in loader config | PROPOSED |
| `grub.d/nvidia-spark-initcall-bl.cfg` (`initcall_blacklist=tegra234_cbb_init`) | `nvidia-spark-initcall-bl` 1.0-1, `dgx-spark-pkgs@nvidia.com`, NVIDIA baseos `dgx` (MEASURED) | yes (config; platform workaround) | keep the workaround, own it | PROPOSED |
| `grub.d/nvidia-spark-pci.cfg` (`pci=pcie_bus_safe`) | `nvidia-spark-grub-pci` 1.0-1, `cudatools@nvidia.com` (MEASURED) | yes (config; platform workaround) | keep, own it | PROPOSED |
| `grub.d/kdump-tools.cfg` (`crashkernel=1G-:0M`) | `kdump-tools` 1:1.10.3ubuntu2, Ubuntu Developers (MEASURED) | no | n/a | none |
| systemd-boot | not installed (`apt-policy systemd-boot`: Installed (none), MEASURED) | — | — | — |
| `EFI/AIENOS/ALLEN.EFI` (417,792 B, sha `4670820c…ae8b`), `CORE.EFI` (159,744 B, sha `9bae7aee…fae2`), `aienos-handoff.efi` (144,384 B) | first-party AIENOS images staged on the ESP (file presence and hashes MEASURED). Their build provenance and boot behaviour were not checked in this lane | no (first-party) | this IS the replacement path for shim+GRUB | file staged; boot behaviour NOT_RUN here, and no AIENOS entry is in BootOrder |

Live cmdline (MEASURED `/proc/cmdline`) carries every NVIDIA-package argument above: `init_on_alloc=0 iommu.passthrough=0 console=tty0 plymouth.ignore-serial-consoles plymouth.use-simpledrm earlycon=uart,mmio32,0x16A00000 … initcall_blacklist=tegra234_cbb_init pci=pcie_bus_safe`.

Layer 1 count: 18 rows; NVIDIA-written: 9 (all are GRUB drop-in config from NVIDIA DGX/baseos packages, no NVIDIA-written executable); not NVIDIA: 7; NOT_RUN/absent: 2. Loader binaries (shim, MokManager, fallback, GRUB) are Canonical/upstream, not NVIDIA.

---

## Layer 2: kernel flavour

Running `7.0.0-1019-nvidia #19~24.04.2-Ubuntu SMP PREEMPT_DYNAMIC` (`uname -a`, MEASURED). `CONFIG_VERSION_SIGNATURE="Ubuntu 7.0.0-1019.19~24.04.2-nvidia 7.0.14"` (MEASURED, `/boot/config-7.0.0-1019-nvidia`), so the base is upstream 7.0.14.

| Component | Who wrote it | NVIDIA-written? | Replacement route | Status |
|---|---|---|---|---|
| `linux-image-7.0.0-1019-nvidia` 7.0.0-1019.19~24.04.2 | Source `linux-signed-nvidia-7.0`, Maintainer `Canonical Kernel Team`, archive `ports.ubuntu.com noble-updates/main` and `noble-security/main` (`dpkg-s`, `apt-policy`, MEASURED) | partly (Canonical-built Linux plus NVIDIA platform enablement, see config diff) | stock Ubuntu generic or mainline kernel if it boots GB10; else carry only the needed platform patches | PROPOSED |
| `linux-modules-7.0.0-1019-nvidia` | Source `linux-nvidia-7.0`, Maintainer `Ubuntu Kernel Team` (MEASURED) | partly | same | PROPOSED |
| Meta `linux-nvidia-hwe-24.04` / `linux-image-nvidia-hwe-24.04` | Source `linux-meta-nvidia-7.0`; image meta depends on `linux-firmware` and `linux-modules-nvidia-fs-7.0.0-1019-nvidia` (MEASURED) | no (packaging) | — | — |
| Older flavours kept installed: 6.11.0-1014-nvidia, 6.17.0-1032-nvidia | `dpkg -l`, `/boot/vmlinuz-*` (MEASURED) | partly | — | — |
| Ubuntu generic arm64 kernel | not installed; candidates `linux-image-generic` 6.8.0-147.147, `linux-image-generic-hwe-24.04` 7.0.0-38.38~24.04.4 (`apt-policy`, MEASURED) | no | the comparison baseline; whether it boots the Spark is NOT_RUN (needs reboot, forbidden) | NOT_RUN |

### Config diff: -nvidia vs Ubuntu generic of the same base

Method (MEASURED): `apt-get download linux-headers-7.0.0-38-generic` (7.0.0-38.38~24.04.4, `CONFIG_VERSION_SIGNATURE="Ubuntu 7.0.0-38.38~24.04.4-generic 7.0.14"`, same 7.0.14 base) into `/tmp/bootlayers`, extracted `.config`, compared `^CONFIG_…=` lines: 12,577 set options (-nvidia) vs 12,568 (generic), **89 differing lines** (`diff g.txt n.txt | grep -c '^[<>]'`). Patch-level (source) diff: NOT_RUN (no source package fetched); config shows which drivers are switched on, not which SAUCE patches exist.

Only in -nvidia (or changed), grouped:
- **NVIDIA/DGX platform**: `CONFIG_NVIDIA_FFA_EC=y` (driver `nvidia-ffa`, bound to ACPI `MSFT000C:00`, `/sys/bus/platform/drivers/nvidia-ffa`, MEASURED); `CONFIG_MTK_PCIE_HOTPLUG=m` (module `mtk_pcie_hotplug`, description "CX7 PCIe Hotplug Driver for NVIDIA DGX Systems", path `drivers/platform/arm64/nvidia/`); `CONFIG_NVGRACE_EGM=m`; `CONFIG_PINCTRL_MT8901=y`; `CONFIG_R8127=m` (Realtek vendor driver 11.014.00-NAPI); `CONFIG_ARM64_MPAM_RESCTRL_FS`, `RESCTRL_FS`, `PROC_CPU_RESCTRL`; seven `ARM64_ERRATUM_*` and two `TRBE` workarounds; 13 `CORESIGHT_*` options.
- **IOMMU/VFIO**: `CONFIG_IOMMU_DEFAULT_PASSTHROUGH=y` (generic: `IOMMU_DEFAULT_DMA_STRICT=y`; overridden at boot by `iommu.passthrough=0` from nv-iommu); `CONFIG_ARM_SMMU_V3_IOMMUFD=y`, `CONFIG_IOMMUFD_VFIO_CONTAINER=y` while generic's `CONFIG_VFIO_CONTAINER=y` and `CONFIG_VFIO_IOMMU_TYPE1=m` are **absent** in -nvidia. Consequence (INFERENCE from config): the legacy `/dev/vfio/vfio` container is served by iommufd's compatibility layer, not vfio_iommu_type1; a VFIO driver should target iommufd/cdev (`CONFIG_VFIO_DEVICE_CDEV=y` per lead).
- **GPU drivers removed in -nvidia**: `CONFIG_DRM_NOUVEAU=m` (+ `NOUVEAU_PLATFORM_DRIVER`, CH7006, SIL164) present in generic, **not set** in -nvidia. `CONFIG_RUST=y` and `RUST_FW_LOADER_ABSTRACTIONS=y` present in generic, **not set** in -nvidia (no Rust kernel code; `DRM_TYR` dropped too). Neither config has any `NOVA`/`DRM_NOVA` option (MEASURED grep). So the shipped flavour cannot host nouveau or a Rust nova-core without a rebuild.
- **Misc**: `CMA_SIZE_MBYTES` 32 → 0; cpufreq default governor ondemand → performance; `ARM_FFA_TRANSPORT`, `TCG_ARM_CRB_FFA`, `EFI_CAPSULE_LOADER`, `MTD`, `MTD_NAND_CORE`, `BCH`, `MT76_NPU` built in (`=y`) instead of modules; `MEMCG_V1`, `CPUSETS_V1` dropped.

Common to both configs, so not -nvidia-specific (MEASURED grep of both): `ARCH_TEGRA_{132,186,194,210,234,241,264}_SOC`, `TEGRA241_CMDQV=y`, `NVGRACE_GPU_VFIO_PCI=m`, `ARM_SMMU_V3=y`.

### Does a non-NVIDIA kernel boot the Spark? (web, 3 queries used)
- NVIDIA: DGX OS 7 "Includes … the NVIDIA-optimized Linux kernel 7.0 (DGX servers based on ARM64)" — https://docs.nvidia.com/dgx/dgx-os-7-user-guide/introduction.html ; DGX Spark release notes: "This update supports the Linux 7.0 kernel (previously 6.17)" — https://docs.nvidia.com/dgx/dgx-spark/release-notes.html
- Red Hat (2026-06-23): GB10 "requires a custom kernel build to enable compatibility with RHEL 10"; "released RHEL 10.y versions do not currently install on the GB10 platform"; kernel tree `gitlab.com/redhat/edge/kernel/nvidia-gb10` — https://developers.redhat.com/articles/2026/06/23/building-custom-red-hat-enterprise-linux-kernel-nvidia-dgx-spark
- Community guide: "The DGX Spark runs the NVIDIA/Ubuntu HWE kernel, not the standard generic kernel" — https://github.com/timothystewart6/ubuntu-gb10/blob/main/docs/02-nvidia-stack.md
- Unpatched mainline booting a DGX Spark: no primary source found; NOT_RUN. The sources above say vendor kernels are needed, which is evidence of missing upstream pieces, not a measured boot failure.

Layer 2 count: 5 rows; partly NVIDIA: 3, not NVIDIA: 1, NOT_RUN: 1. 89 config lines differ from generic; patch-level diff NOT_RUN.

---

## Layer 3: kernel drivers

`lsmod`: 174 modules loaded (175 lines incl. header; MEASURED). No `/lib/modules/7.0.0-1019-nvidia/updates` and no `/var/lib/dkms` (MEASURED): nothing is DKMS-built. Every module is a `.ko`/`.ko.zst` shipped by a Canonical package.

### NVIDIA- or Mellanox-authored modules (loaded)

| Module | File | Author / license / version (`mi`) | In-tree? | rc | NVIDIA-written? | Replacement route | Status |
|---|---|---|---|---|---|---|---|
| `nvidia` | `kernel/nvidia-580-open/nvidia.ko` (19,231,198 B) | no author; Dual MIT/GPL; 580.173.02; signer `Canonical Ltd. Kernel Module Signing`. Package `linux-modules-nvidia-580-open-7.0.0-1019-nvidia`, Source `linux-restricted-signatures-nvidia-7.0`, Maintainer `Canonical Kernel Team`, dpkg status **hold** | out-of-tree (Canonical prebuilt of NVIDIA open-gpu-kernel-modules) | 955 | yes | seeded-.om userspace driver over VFIO/iommufd, loading NVIDIA GSP firmware unmodified | PROPOSED |
| `nvidia_uvm` | `…/nvidia-uvm.ko` (3,051,046 B) | Dual MIT/GPL; 580.173.02; Canonical-signed | out-of-tree | 4 | yes | UVM surface (11 ioctls used by CUDApp) reimplemented in .om | PROPOSED |
| `nvidia_modeset` | `…/nvidia-modeset.ko` (2,635,102 B) | Dual MIT/GPL; 580.173.02; Canonical-signed | out-of-tree | 4 | yes | not needed by the serving path; display question below | PROPOSED |
| `nvidia_drm` | `…/nvidia-drm.ko` (234,494 B) | Dual MIT/GPL; 580.173.02; Canonical-signed | out-of-tree | 0 | yes | not needed by the serving path | PROPOSED |
| `mlx5_core` | `drivers/net/ethernet/mellanox/mlx5/core/mlx5_core.ko.zst` | Eli Cohen <eli@mellanox.com>; Dual BSD/GPL; pkg `linux-modules-7.0.0-1019-nvidia` | in-tree upstream | 2 | yes (Mellanox, now NVIDIA) | out of GPU scope; currently bound to no device (CX7 not enumerated) | none |
| `mlx5_ib` | `drivers/infiniband/hw/mlx5/mlx5_ib.ko.zst` | Eli Cohen <eli@mellanox.com>; Dual BSD/GPL | in-tree | 0 | yes | out of GPU scope | none |
| `mlx5_fwctl` | `drivers/fwctl/mlx5/mlx5_fwctl.ko.zst` | Saeed Mahameed <saeedm@nvidia.com>; Dual BSD/GPL | in-tree | 0 | yes | out of GPU scope | none |
| `mlxfw` | `drivers/net/ethernet/mellanox/mlxfw/mlxfw.ko.zst` | Yotam Gigi <yotamg@mellanox.com>; Dual BSD/GPL | in-tree | 1 | yes | out of GPU scope | none |
| `mstflint_access` | `kernel/mstflint/mstflint_access.ko.zst` | Mahmoud Hasan; Dual BSD/GPL; "2.0.0 (Nov-27-2012)"; description "MST Module"; pkg `linux-modules-7.0.0-1019-nvidia` | **not** under `kernel/drivers`: out-of-tree Mellanox source built into the Ubuntu flavour package | 0 | yes (Mellanox) | out of GPU scope | none |
| `tls` | `net/tls/tls.ko.zst` | `Mellanox Technologies`; Dual BSD/GPL | in-tree | 1 | partly (Mellanox-authored upstream core networking) | out of scope | none |
| `mtk_pcie_hotplug` | `drivers/platform/arm64/nvidia/mtk-pcie-hotplug.ko.zst` | no author; GPL; "CX7 PCIe Hotplug Driver for NVIDIA DGX Systems"; build-time key | in this flavour's tree; `CONFIG_MTK_PCIE_HOTPLUG` absent from generic config (MEASURED); mainline status NOT_RUN | 0 | yes (by description/path) | out of GPU scope | none |
| `snd_hda_codec_tegrahdmi` | `sound/hda/codecs/hdmi/snd-hda-codec-tegrahdmi.ko.zst` | no author; GPL; "Nvidia Tegra HDMI HD-audio codec"; aliases `hdaudio:v10DE003x` | in-tree (in generic too, INFERENCE from path; not diffed) | 1 | partly (NVIDIA HDMI-audio support in upstream ALSA) | goes with display | PROPOSED (with display) |

Installed but not loaded (MEASURED): `nvidia-peermem.ko` (7,942 B, same dir); `/lib/modules/7.0.0-1019-nvidia/ubuntu/nvidia-fs/nvidia-fs.ko.zst` from `linux-modules-nvidia-fs-7.0.0-1019-nvidia` (Source `linux-nvidia-7.0`).

### NVIDIA-authored code built into the kernel image (not modules)

| Component | Source | NVIDIA-written? | Replacement route | Status |
|---|---|---|---|---|
| `nvidia-ffa` (EC over Arm FF-A) | `CONFIG_NVIDIA_FFA_EC=y`; `/sys/module/nvidia-ffa-ec`; bound to `MSFT000C:00` (MEASURED). Absent from generic config | yes (by name; upstream status NOT_RUN) | platform code; outside GPU lane | none |
| Tegra241 CMDQV extension of arm-smmu-v3 | `CONFIG_TEGRA241_CMDQV=y` (both configs). Whether CMDQV is active on this SoC: NOT_RUN (`dmesg` denied; `journalctl -k` grep for `cmdqv` returned nothing) | yes (NVIDIA-contributed upstream) | keep (IOMMU we rely on for VFIO) | none |
| Tegra platform drivers registered (`tegra-bpmp`, `tegra-fuse`, `tegra-mc`, `tegra194-cbb`, `tegra-hsp`, …) | `/sys/bus/platform/drivers` (MEASURED); `tegra-fuse` and `tegra194-cbb` have no bound device (MEASURED) | yes (NVIDIA-contributed upstream) | none needed | none |

### Display (card0 / card1)

MEASURED:
- `/sys/class/drm/card0` → `/sys/devices/platform/simple-framebuffer.0`, driver `simple-framebuffer` (simpledrm, `CONFIG_DRM_SIMPLEDRM=y`, `CONFIG_SYSFB_SIMPLEFB=y`); connector `card0-Unknown-1` status `connected`; `/proc/fb` lists only `0 simpledrmdrmfb`.
- `/sys/class/drm/card1` → `000f:01:00.0`, driver `nvidia`; card1 has **no connector entries** (`ls /sys/class/drm` shows no `card1-*`).
- modprobe config conflict: `/etc/modprobe.d/nvidia-graphics-drivers-kms.conf` (`nvidia-kernel-common-580`) sets `options nvidia_drm modeset=1`; `/etc/modprobe.d/zz-nvidia-drm-override.conf` (package `nvidia-drm-options-modeset0` 25.07-1, `cudatools@nvidia.com`, NVIDIA baseos repo) sets `options nvidia-drm modeset=0`. Live `/sys/module/nvidia_drm/parameters/modeset` is root-only: effective value NOT_RUN. No connectors on card1 is consistent with modeset=0 (INFERENCE).

Consequence (INFERENCE, not measured): the local console is drawn by simpledrm into the framebuffer that UEFI GOP set up on the GPU (lead: GPU `boot_vga=1`). Removing `nvidia_drm` alone probably changes nothing visible, since it drives no connector. Unbinding `nvidia` from the GPU, an FLR, or a new driver re-initialising the display engine can stop scan-out of that framebuffer. The screen then goes dark, and simpledrm keeps writing to memory nobody displays. Plan any window for headless operation (SSH/serial console `ttyS0,921600`).

Layer 3 count: 174 modules loaded; 12 rows of NVIDIA/Mellanox-authored loaded modules (NVIDIA-written yes: 10, partly: 2); 4 are the GPU driver stack (out-of-tree, Canonical-signed, held); 3 built-in NVIDIA platform components; 0 DKMS modules.

Footnote, userspace (MEASURED): `dpkg -l | grep -i -E 'nvidia|cuda'` gives 167 packages in `ii`/`hi` state, 24 of them held (`hi`). `apt-mark showhold` lists 43 holds, 24 matching nvidia/cuda (e.g. `nvidia-driver-580-open`, `nvidia-kernel-common-580`, `libnvidia-compute-580`, `linux-modules-nvidia-580-open-7.0.0-1019-nvidia`, `nvidia-container-toolkit`). `nvidia-driver-580-open` is installed from the Ubuntu archive build 580.173.02-0ubuntu0.24.04.1, with NVIDIA CUDA sbsa repos also configured.

---

## Claims scoping

"GPU driver replaced" must never be reported as "zero NVIDIA code". After a GPU-driver-only cutover (our .om userspace driver replaces `nvidia`, `nvidia_uvm`, `nvidia_modeset`, `nvidia_drm`), this NVIDIA-written code would still be on the Spark (sources: tables above):

1. **Layer 0, by policy (never touched):** NVIDIA-built AMI UEFI 5.36_0ACUM018, SoC boot firmware, VBIOS 9a.0b.2d.00.00, and the NVIDIA-signed GSP-RM/booter/FMC images. Our driver still loads the GSP/FMC images unmodified, so NVIDIA's GSP-RM keeps running on the GPU's RISC-V core.
2. **Layer 0 files:** `nvidia-firmware-580-580.173.02` (`gsp_ga10x.bin`, `gsp_tu10x.bin`), `linux-firmware-nvidia-graphics`, CX7 NIC flash firmware.
3. **Layer 1:** nine NVIDIA DGX/baseos GRUB drop-ins (dgxstation-grub, nv-grubserial, nv-no-grubmenu, nv-iommu, nvidia-disable-init-on-alloc, nvidia-console-settings, nvidia-earlycon, nvidia-spark-initcall-bl, nvidia-spark-grub-pci) that set the live cmdline. These are configuration, not executables; shim and GRUB are not NVIDIA-written.
4. **Layer 2:** the `linux-nvidia-7.0` flavour itself: Canonical-built, with NVIDIA platform enablement (89 config lines differ from generic). It is the only kernel shown to boot this box (NVIDIA docs; Red Hat needs a custom GB10 tree). Booting generic or mainline instead is NOT_RUN.
5. **Layer 3, upstream GPL/BSD but NVIDIA- or Mellanox-authored:** `mlx5_core`, `mlx5_ib`, `mlx5_fwctl`, `mlxfw`, `tls`, `snd_hda_codec_tegrahdmi`, built-in `tegra241-cmdqv`, Tegra platform drivers. **Flavour-only NVIDIA pieces:** `nvidia-ffa` EC driver, `mtk_pcie_hotplug` (DGX CX7 hotplug), out-of-tree `mstflint_access`. Installed but unloaded: `nvidia-peermem.ko`, `nvidia-fs.ko`.
6. **Userspace:** 167 nvidia/cuda packages (footnote above), outside layers 0-3 but still NVIDIA code.

The accurate claim after the GPU cutover is: "the four NVIDIA GPU kernel modules are replaced by first-party Omega code; NVIDIA firmware, platform kernel code and the NVIDIA kernel flavour remain". Each layer is PROPOSED today: nothing in this inventory is IMPLEMENTED, EXERCISED or QUALIFIED on the Spark.
