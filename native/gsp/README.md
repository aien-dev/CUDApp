WARNING: This can and may permanently damage your system. No warranty of any kind. Do not use it unless you are a professional lab that is prepared to lose the hardware.

# native/gsp: GSP firmware container parser (read-only)

This directory holds read-only research into the GPU boot chain. `fwimage.om` reads one firmware file and
writes a text report. It never opens a device, maps a BAR, writes to a device or runs anything on one.

On Drake's DGX Spark the signed boot chain is never modified: UEFI and platform firmware, NVIDIA-signed
GSP/booter/FMC firmware, VBIOS and fuses. A future first-party driver may only load NVIDIA's signed firmware
unmodified, as nouveau and nova do. Replacing the boot chain is future work, and only if a dedicated,
sponsored DGX Spark exists for it.

## Files

| Path | Role |
|---|---|
| `native/gsp/fwimage.om` | Seeded Omega (OM-2) parser. It is compiled together with omega-mv `om/tls/sha2.om` (SHA-256). |
| `tests/gsp/run-fwimage-host.sh` | Host runner: pins, compile, oracle comparison, negative controls. |
| `tests/gsp/expected-580.173.02.txt` | Expected report, generated with `readelf -SW`, `dd` and `sha256sum` only. |

Build and run, with the pinned omc from omega-mv `369dbc96daa1684db2d7c2c7085bbf16b3fd77ee`
(`seed/out/omc` sha256 `2c82480596bbdc50cda9626866209ef662d5834f278f839ff4ce0250c495029f`):

```sh
cat native/gsp/fwimage.om "$OMEGA_MV/om/tls/sha2.om" | "$OMEGA_MV/seed/out/omc" > /tmp/fwimage
chmod 700 /tmp/fwimage
/tmp/fwimage /lib/firmware/nvidia/580.173.02/gsp_ga10x.bin gb20y
tests/gsp/run-fwimage-host.sh "$OMEGA_MV/seed/out/omc" /lib/firmware/nvidia/580.173.02/gsp_ga10x.bin
```

The runner refuses (exit 69) unless the omc, `sha2.om` (`5cf0f981…6fdb33`) and the firmware
(`32cb7cf3…d963`) match their pins. It prints `FWIMAGE PASS` only when every check below holds:

1. The committed expectation equals a fresh `readelf`/`dd`/`sha256sum` reference.
2. The fwimage output equals that expectation, with no family, with gb20y and with gb10y.
3. An expectation with one altered digest digit is rejected (a negative control on the checker itself).
4. The C oracle `ck_gsp_fw_inspect.c` from lane L332 agrees on the lines both tools print. The runner
   builds it into its temporary directory, and it is used only as a verification oracle.
5. Six mutated inputs are refused, each with its documented code: truncated to 1000 bytes (12), ELF magic
   byte changed (10), `.fwversion` sh_offset past end of file (14), `.fwversion` renamed (16), a duplicate
   section name (18), and an unknown family (17).

## Report and exit codes

```
SECTION <name> off=<hex> size=<hex>     every section header after index 0, in table order
FWVERSION <text>
SHA256 <name> <hex>                     .fwimage and every .fwsignature_* section, in table order
SELECT family=<f> image_off=<hex> image_size=<hex> sig_section=.fwsignature_<f> sig_sha256=<hex>
```

All checks run before any output, so a refusal prints only `FWIMAGE REFUSE <reason>` on stderr.

| Exit | Meaning |
|---|---|
| 0 | report written |
| 2 | usage |
| 3 | I/O: open, seek, short read or stdout write failed |
| 10 | not ELF (magic) |
| 11 | not ELF64 / little-endian / version 1 / RISC-V (e_machine 243) |
| 12 | truncated: file shorter than 64 bytes, or the section header table passes end of file |
| 13 | malformed header: e_ehsize or e_shentsize not 64, e_shnum 0, e_shstrndx 0 or ≥ e_shnum, name table not SHT_STRTAB or empty |
| 14 | section or name out of bounds (offset + size > file size, overflow-checked; name outside the table; table not zero-terminated) |
| 15 | missing `.fwimage` |
| 16 | missing `.fwversion` |
| 17 | unknown family (only gb20y, gb10y) |
| 18 | duplicate section names |
| 19 | capacity: more than 1024 sections or a name table above 64 KiB |
| 20 | selected `.fwsignature_<family>` absent |
| 21 | `.fwimage` or a `.fwsignature_*` section is SHT_NOBITS or empty |
| 22 | `.fwversion` malformed: size 0 or above 63, no final zero byte, or a non-printable byte |
| 100+n | Omega runtime trap n (none expected) |

## Container format, as the 580.173.02 driver reads it

Citations are paths under `open-gpu-kernel-modules` 580.173.02
(git `20e4e6e19cc26ba47b5cbe23130a396be100c427`).

- **File choice.** Families GB10Y=11 and GB20Y=12 (`kernel-open/common/inc/nv-firmware.h:52-53`) have the
  names `gb10y` and `gb20y` (`:63-64`). Both, together with GB10X/GB20X/GH100/AD10X/GA10X, load `gsp_ga10x`
  (`:94-102`).
- **Container.** The firmware is an ELF64 image. `_kgspFwContainerGetSection`
  (`src/nvidia/src/kernel/gpu/gsp/kernel_gsp.c:5078`) checks:
  - the magic (`:5116`), little-endian (`:5118`) and ELFCLASS64 (`:5120`);
  - `shentsize == 64` (`:5123`) and an overflow-checked section header table inside the buffer (`:5124-5126`);
  - `shstrndx <= shnum` (`:5127`).

  It then scans the headers backward from `shnum - 1` (`:5139`), bounds-checks each name offset (`:5145`)
  and each section's offset + size (`:5158`).
- **Section names** (`src/nvidia/generated/g_kernel_gsp_nvoc.h:269-274`): `.fwversion`, `.fwimage`,
  `.fwsignature_` (prefix), `.note.gnu.build-id`, `.fwsignature_cc_` (prefix for confidential compute).
- **Version.** `_kgspFwContainerVerifyVersion` (`kernel_gsp.c:4735`) requires:
  - a size of 1..63 that ends in a zero byte (`:4763-4766`);
  - text equal to the driver's `NV_VERSION_STRING`, with size = length + 1 (`:4768-4772`).

  In this file the value is `580.173.02` (11 bytes).
- **Image and signature.** `_kgspPrepareGspRmBinaryImage` (`kernel_gsp.c:4843`) verifies the version, takes
  `.fwimage`, and takes the signature section named prefix + family. That name comes from
  `_kgspGetSectionNameForPrefix` (`:4796`), into a 32-byte buffer (`:4850`). The function then builds the
  radix3 descriptor over the image (`:4884`).
- **Signature prefix.** For Blackwell the prefix comes from `kgspGetSignatureSectionNamePrefix_GB100`
  (`src/nvidia/generated/g_kernel_gsp_nvoc.c:1811`; `src/nvidia/src/kernel/gpu/gsp/arch/blackwell/kernel_gsp_gb100.c:95-108`).
  It returns `.fwsignature_cc_` when confidential compute is enabled, otherwise `.fwsignature_`.
  This file has no `.fwsignature_cc_gb10y` or `.fwsignature_cc_gb20y`, so on this chip the selection with
  CC enabled would fail. That last point is an inference from the section list and is not run.
- **Boot.** SEC2 boots GSP-FMC for GB10B, GB20B and GB20C (`PDB_PROP_KSEC2_BOOT_GSPFMC`,
  `src/nvidia/generated/g_kernel_sec2_nvoc.c:446-455`).

Measured on `/lib/firmware/nvidia/580.173.02/gsp_ga10x.bin` (75,012,080 bytes) with `readelf -SW`:

- 17 sections; e_shoff 0x47893b0; e_shstrndx 16.
- `.fwimage` at 0x40, size 0x477f000.
- Ten `.fwsignature_*` sections of 0x1000 bytes each.
- The full list is in `tests/gsp/expected-580.173.02.txt`.

Where `fwimage.om` is stricter than the driver:

| Check | Driver | fwimage.om |
|---|---|---|
| e_shstrndx bound | accepts `shstrndx == shnum` (`:5127`), which indexes one past the table | requires `0 < shstrndx < shnum` |
| duplicate section names | backward scan, so the last header with a name wins | refuses (18) |
| e_machine (243) and e_version | not checked | checked |

## Chip identity: gb20y or gb10y (unresolved)

The family string picks the signature section, so a wrong family selects the wrong signature.
Family selection (`src/nvidia/arch/nvalloc/common/inc/nv-firmware-chip-family-select.h`):

| Chip | Architecture | Implementation | Family | Source |
|---|---|---|---|---|
| GB10B | GB1XX = 0x1A0 (`src/common/inc/swref/published/nv_arch.h:51`) | 0x0B (`:111`) | gb10y | `nv-firmware-chip-family-select.h:58-62` |
| GB20B | GB2XX = 0x1B0 (`nv_arch.h:53`) | ≥ GB20B = 0x0B (`:127`) | gb20y | `nv-firmware-chip-family-select.h:64-68` |

Lane L332 assumed GB10B (gb10y). Lead evidence points to GB20B (gb20y):

- CUDApp measures sm_121.
- The UVM source pairs GB10B with T264 (Thor).
- `PDB_PROP_GPU_CLKS_IN_TEGRA_SOC` is set for GB10B only.

Only a `PMC_BOOT_42` read during an authorized window settles it; that read is not run. Until then the
parser and runner treat both families and report both selections.

The two signature digests differ:

| Family | Signature digest |
|---|---|
| gb20y | `6d38023b…f9a1` |
| gb10y | `08541c3a…8f2d` |
