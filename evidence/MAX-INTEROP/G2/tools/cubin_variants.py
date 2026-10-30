#!/usr/bin/env python3
"""Gate-2 load-matrix fixtures (diagnostic, host only).

Builds cubin variants that separate the candidate causes of CUDApp#68's
cuModuleLoadData error 200: missing program headers, missing capmerc/merc
sections, missing .note.nv.tkinfo, .nv.callgraph, symtab sh_info.
Writes variants/<name>.cubin and variants/MATRIX.tsv (name, fn, mode,
expectation, sha256, what it isolates).
"""
import hashlib
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CTL = HERE / "ctl.cubin"
STALE = Path("/dev/shm/cudapp-max/evidence/MAX-INTEROP/68/matmul-16x16x8.cubin")
MAIN = HERE / "main/matmul-16x16x8.cubin"
SM80 = Path.home() / "workspace/maxinterop/max-source/max/kernels/test/gpu/device_context/data/vec_add.sm80.cubin"
OUT = HERE / "variants"
FN = "omega_matmul_bf16"

SH = struct.Struct("<IIQQQQIIQQ")  # name type flags addr off size link info align entsize


class Elf:
    def __init__(self, raw: bytes):
        self.b = bytearray(raw)
        assert self.b[:4] == b"\x7fELF" and self.b[4] == 2, "not ELF64"

    def u(self, fmt, off):
        return struct.unpack_from(fmt, self.b, off)[0]

    def p(self, fmt, off, v):
        struct.pack_into(fmt, self.b, off, v)

    @property
    def shoff(self):
        return self.u("<Q", 0x28)

    @property
    def shnum(self):
        return self.u("<H", 0x3C)

    def shdrs(self):
        return [list(SH.unpack_from(self.b, self.shoff + i * 64)) for i in range(self.shnum)]

    def name_of(self, hdrs, i):
        st = hdrs[self.u("<H", 0x3E)]
        s = st[4] + hdrs[i][0]
        return bytes(self.b[s : self.b.index(0, s)]).decode()

    def index(self, name):
        h = self.shdrs()
        return next(i for i in range(len(h)) if self.name_of(h, i) == name)

    def set_sh(self, i, field, value):
        h = self.shdrs()
        h[i][field] = value
        SH.pack_into(self.b, self.shoff + i * 64, *h[i])

    def add_section(self, name, typ, flags, align, data, link=0, info=0, entsize=0):
        """Append a section at the end; shstrtab and the header table move to EOF."""
        h = self.shdrs()
        sx = self.u("<H", 0x3E)
        old = bytes(self.b[h[sx][4] : h[sx][4] + h[sx][5]])
        new_name_off = len(old)
        strtab = old + name.encode() + b"\0"
        h[sx][4], h[sx][5] = self._append(strtab, 1), len(strtab)
        off = self._append(data, max(align, 1))
        h.append([new_name_off, typ, flags, 0, off, len(data), link, info, align, entsize])
        shoff = self._append(b"".join(SH.pack(*r) for r in h), 8)
        self.p("<Q", 0x28, shoff)
        self.p("<H", 0x3C, len(h))
        return len(h) - 1

    def _append(self, data, align):
        while len(self.b) % align:
            self.b.append(0)
        off = len(self.b)
        self.b += data
        return off


def tkinfo_from(ctl: Elf):
    h = ctl.shdrs()
    i = ctl.index(".note.nv.tkinfo")
    return bytes(ctl.b[h[i][4] : h[i][4] + h[i][5]]), h[i]


def own_tkinfo(template: bytes):
    """Same note header/fields as ptxas, tool strings replaced with CUDApp's."""
    namesz, descsz, ntype = struct.unpack_from("<III", template, 0)
    name = template[12 : 12 + namesz]
    desc = bytearray(template[12 + ((namesz + 3) & ~3) :][:descsz])
    # desc = six u32 (2, 0, off_tool, off_version, off_build, off_options) then a
    # string blob that starts with NUL; offsets index the blob (read off ptxas 13.0.88).
    head = list(struct.unpack_from("<6I", desc, 0))
    blob = b"\0"
    offs = []
    for s in (b"cudapp", b"CUDApp omega_cubin sm_121", b"Build b5746b94", b"-arch sm_121"):
        offs.append(len(blob))
        blob += s + b"\0"
    head[2:6] = offs
    desc = struct.pack("<6I", *head) + blob
    desc += b"\0" * ((-len(desc)) % 4)
    return struct.pack("<III", namesz, len(desc), ntype) + name + b"\0" * ((-namesz) % 4) + desc


def main():
    OUT.mkdir(exist_ok=True)
    ctl = Elf(CTL.read_bytes())
    stale = STALE.read_bytes()
    main_raw = MAIN.read_bytes()
    tk, tkh = tkinfo_from(ctl)
    cg_i = ctl.index(".nv.callgraph")
    cgh = ctl.shdrs()[cg_i]
    callgraph = bytes(ctl.b[cgh[4] : cgh[4] + cgh[5]])
    rows = []

    def emit(name, data, fn, mode, expect, isolates):
        (OUT / f"{name}.cubin").write_bytes(data)
        rows.append((name, fn, mode, expect, hashlib.sha256(data).hexdigest(), isolates))

    # Positive control: ptxas 13.0.88 cubin for the same ABI.
    emit("A-ctl", bytes(ctl.b), FN, "run", "PASS", "probe+driver+ABI known-good")
    # Capmerc hypothesis: drop the trailing 7 capmerc/merc sections (indices 15..21;
    # nothing in .symtab or the remaining sections points at them).
    nomerc = Elf(bytes(ctl.b))
    first_merc = nomerc.index(".nv.capmerc.text." + FN)
    assert all(nomerc.name_of(nomerc.shdrs(), i).startswith(".nv.merc") or i == first_merc
               for i in range(first_merc, nomerc.shnum))
    nomerc.p("<H", 0x3C, first_merc)
    emit("B-ctl-nomerc", bytes(nomerc.b), FN, "run", "PASS if capmerc not required", "capmerc/merc sections")
    # The #68 evidence artifact (bb9be0b1, before PHDR commit 130a101f): no program headers.
    # 2026-10-24 window 20261024T043818Z-aa5c86 returned 200 for this path (transcript only).
    emit("C-omega-stale-nophdr", stale, FN, "reject", "REJECT 200 (seen 10-24, no receipt)", "program headers absent")
    # Same kernel from main b5746b94 writer (PHDR + 5 headers), text/cbank bytes identical.
    # 10-24 window 20261024T044814Z-4dc098 returned 200 for a PHDR build (/tmp/phdr-mat.cubin).
    emit("D-omega-main", main_raw, FN, "run", "REJECT 200 expected (10-24 phdr-mat)", "current CUDApp writer")
    # Symtab convention: ptxas sets symtab sh_info to the symbol count; CUDApp sets 2
    # with a local symbol after the global one.
    e = Elf(main_raw)
    st = e.index(".symtab")
    e.set_sh(st, 7, e.shdrs()[st][5] // 24)
    emit("E1-omega-symfix", bytes(e.b), FN, "run", "UNKNOWN", "symtab sh_info")
    # tkinfo hypothesis (omega-mv#864 body: ABI 8 without tkinfo -> 200 on other GPUs).
    e = Elf(main_raw)
    ti = e.add_section(".note.nv.tkinfo", tkh[1], tkh[2], tkh[8], tk)
    e.set_sh(e.index(".note.nv.cuinfo"), 6, ti)
    emit("E2-omega-tkinfo-ptxas", bytes(e.b), FN, "run", "UNKNOWN", ".note.nv.tkinfo (ptxas bytes)")
    e = Elf(main_raw)
    ti = e.add_section(".note.nv.tkinfo", tkh[1], tkh[2], tkh[8], own_tkinfo(tk))
    e.set_sh(e.index(".note.nv.cuinfo"), 6, ti)
    emit("E3-omega-tkinfo-own", bytes(e.b), FN, "run", "UNKNOWN", ".note.nv.tkinfo (own strings)")
    e = Elf(main_raw)
    ti = e.add_section(".note.nv.tkinfo", tkh[1], tkh[2], tkh[8], tk)
    e.set_sh(e.index(".note.nv.cuinfo"), 6, ti)
    e.add_section(".nv.callgraph", cgh[1], cgh[2], cgh[8], callgraph, link=e.index(".symtab"), entsize=cgh[9])
    st = e.index(".symtab")
    e.set_sh(st, 7, e.shdrs()[st][5] // 24)
    emit("E4-omega-tkinfo-cg-symfix", bytes(e.b), FN, "run", "UNKNOWN", "all three metadata gaps")
    # Negative controls: each must be rejected.
    emit("N1-ctl-truncated", bytes(ctl.b[:1500]), FN, "reject", "REJECT", "truncated image")
    emit("N2-ctl-wrong-entry", bytes(ctl.b), "omega_matmul_bf16_missing", "reject", "REJECT (500 NOT_FOUND)", "entry name")
    bad = Elf(bytes(ctl.b))
    flags = bad.u("<I", 0x30)
    bad.p("<I", 0x30, (flags & ~0xFF00) | (80 << 8))
    emit("N3-ctl-eflags-sm80", bytes(bad.b), FN, "reject", "REJECT", "e_flags arch field")
    if SM80.exists():
        emit("N4-max-vecadd-sm80", SM80.read_bytes(), "vec_add", "reject", "REJECT (209 NO_BINARY_FOR_GPU)", "wrong-arch real cubin")
    with open(OUT / "MATRIX.tsv", "w") as f:
        f.write("name\tfn\tmode\texpect\tsha256\tisolates\n")
        for r in rows:
            f.write("\t".join(r) + "\n")
    print((OUT / "MATRIX.tsv").read_text(), end="")


if __name__ == "__main__":
    sys.exit(main())
