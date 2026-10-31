#!/usr/bin/env python3
"""Gate-2 load-matrix fixtures (diagnostic, host only).

Builds cubin variants that separate the candidate causes of CUDApp#68's
cuModuleLoadData error 200: missing program headers, missing capmerc/merc
sections, missing .note.nv.tkinfo, .nv.callgraph, symtab sh_info, the
remaining ptxas-only metadata (E5), and container versus machine code (F, G:
each kernel's SASS inside the other writer's container).
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
MAIN = HERE / "omega-main-artifact/matmul-16x16x8.cubin"
SM80 = Path.home() / "workspace/maxinterop/max-source/max/kernels/test/gpu/device_context/data/vec_add.sm80.cubin"
OUT = HERE / "variants"
FN = "omega_matmul_bf16"

SH = struct.Struct("<IIQQQQIIQQ")  # name type flags addr off size link info align entsize
PH = struct.Struct("<IIQQQQQQ")  # type flags off vaddr paddr filesz memsz align
SYM = struct.Struct("<IBBHQQ")  # name info other shndx value size
EXIT_OFFS, CRS_STACK, MAXREG, REGCOUNT = 0x1C, 0x1E, 0x1B, 0x2F  # EIATTR codes (cuobjdump 13.0 names)
ISA_CLASS = 0x02  # EICOMPAT_ATTR_ISA_CLASS in .nv.compat (cuobjdump 13.0); ptxas 13.0 sm_121/a/f writes 1, CUDApp 2


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

    def data(self, i):
        h = self.shdrs()[i]
        return bytes(self.b[h[4] : h[4] + h[5]])

    def replace(self, i, data, align):
        """Move section i's contents to EOF; a caller moving a loaded section fixes its segment."""
        self.set_sh(i, 4, self._append(data, max(align, 1)))
        self.set_sh(i, 5, len(data))

    def phdr(self, j):
        return list(PH.unpack_from(self.b, self.u("<Q", 0x20) + j * 56))

    def set_phdr(self, j, row):
        PH.pack_into(self.b, self.u("<Q", 0x20) + j * 56, *row)

    def text_phdr(self, text_off):
        return next(j for j in range(self.u("<H", 0x38)) if self.phdr(j)[0] == 1 and self.phdr(j)[2] == text_off and self.phdr(j)[1] & 1)

    def sym(self, k):
        h = self.shdrs()[self.index(".symtab")]
        return list(SYM.unpack_from(self.b, h[4] + k * 24))

    def set_sym(self, k, row):
        h = self.shdrs()[self.index(".symtab")]
        SYM.pack_into(self.b, h[4] + k * 24, *row)


def nvinfo(data):
    """EIFMT_SVAL (4) entries carry a u16 size and a payload; the others a u16 value."""
    out, o = [], 0
    while o < len(data):
        fmt, attr = data[o], data[o + 1]
        n = struct.unpack_from("<H", data, o + 2)[0] if fmt == 4 else 0
        out.append([fmt, attr, bytes(data[o + 4 : o + 4 + n]) if fmt == 4 else bytes(data[o + 2 : o + 4])])
        o += 4 + n
    assert o == len(data), "nv.info overrun"
    return out


def nvinfo_pack(entries):
    return b"".join(bytes([f, a]) + (struct.pack("<H", len(v)) if f == 4 else b"") + v for f, a, v in entries)


def exit_offsets(text):
    """Offsets of EXIT (opcode bits 0..11 = 0x94d) in 16-byte sm_121 instructions."""
    return [o for o in range(0, len(text), 16) if struct.unpack_from("<Q", text, o)[0] & 0xFFF == 0x94D]


def set_attr(entries, attr, value):
    hit = [e for e in entries if e[1] == attr]
    assert len(hit) == 1, f"attr {attr:#x} occurs {len(hit)} times"
    hit[0][2] = value
    return entries


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
    # E5: E4 plus every remaining ptxas-only item except merc and .debug_frame:
    # EXIT_INSTR_OFFSETS + CRS_STACK_SIZE, the two reserved-smem symbols, and
    # .nv.constant0 sh_link/align as ptxas writes them.
    ctl_text = ctl.data(ctl.index(".text." + FN))
    assert exit_offsets(ctl_text) == [0x30, 0x780], "EXIT scan disagrees with cuobjdump on the control"
    main_e = Elf(main_raw)
    omega_text = main_e.data(main_e.index(".text." + FN))
    omega_exits = exit_offsets(omega_text)
    assert omega_exits == [0x400], omega_exits  # nvdisasm 13.0: /*0400*/ EXIT
    ki = e.index(".nv.info." + FN)
    ents = nvinfo(e.data(ki))
    at = next(n for n, x in enumerate(ents) if x[1] == 0x4A) + 1
    ents[at:at] = [[4, EXIT_OFFS, struct.pack(f"<{len(omega_exits)}I", *omega_exits)], [4, CRS_STACK, b"\0" * 4]]
    e.replace(ki, nvinfo_pack(ents), 4)
    sti, sx = e.index(".symtab"), e.index(".strtab")
    strtab = e.data(sx)
    names = [len(strtab), len(strtab) + len(".nv.reservedSmem.offset0") + 1]
    e.replace(sx, strtab + b".nv.reservedSmem.offset0\0__nv_reservedSMEM_offset_0_alias\0", 1)
    syms = e.data(sti) + SYM.pack(names[0], 0x21, 0, 0, 0x40, 4) + SYM.pack(names[1], 0x20, 0xA0, e.index(".nv.shared.reserved.0"), 0x40, 0)
    e.replace(sti, syms, 8)
    e.set_sh(sti, 7, len(syms) // 24)
    c0 = e.index(".nv.constant0." + FN)
    e.set_sh(c0, 6, 0)
    e.set_sh(c0, 8, 4)
    emit("E5-omega-allmeta", bytes(e.b), FN, "run", "UNKNOWN", "E4 + exit offsets, crs, smem syms, cbank0 link/align")
    e5 = bytes(e.b)
    # F: the control's container without merc (B) carrying Omega's machine code.
    # The text shrinks in place (0x880 -> 0x480); its segment, symbol size and
    # EXIT offsets follow. REGCOUNT stays the control's 40: Omega's code writes R0..R31
    # (peak 32), ptxas declares max register + 3 (control R37 -> 40, mma R15 -> 18). cbank0 is all zero in both.
    f = Elf(bytes(nomerc.b))
    ti = f.index(".text." + FN)
    th = f.shdrs()[ti]
    f.b[th[4] : th[4] + th[5]] = omega_text + b"\0" * (th[5] - len(omega_text))
    f.set_sh(ti, 5, len(omega_text))
    j = f.text_phdr(th[4])
    ph = f.phdr(j)
    ph[5] = ph[6] = len(omega_text)
    f.set_phdr(j, ph)
    k = th[7]  # .text sh_info = the function symbol
    s = f.sym(k)
    s[5] = len(omega_text)
    f.set_sym(k, s)
    ki = f.index(".nv.info." + FN)
    f.replace(ki, nvinfo_pack(set_attr(nvinfo(f.data(ki)), EXIT_OFFS, struct.pack(f"<{len(omega_exits)}I", *omega_exits))), 4)
    emit("F-ctlcontainer-omega-sass", bytes(f.b), FN, "run", "UNKNOWN: load => Omega SASS accepted", "machine code vs container (ptxas container)")
    # G: the current CUDApp container (D) carrying the control's machine code.
    # The text moves to EOF (0x480 -> 0x880) with its segment; symbol size,
    # REGCOUNT 40 and MAXREG 0xff follow the control's code. The old Omega
    # bytes are zeroed, so each of F and G holds exactly one kernel's code.
    g = Elf(main_raw)
    ti = g.index(".text." + FN)
    th = g.shdrs()[ti]
    j = g.text_phdr(th[4])
    g.replace(ti, ctl_text, 128)
    g.b[th[4] : th[4] + th[5]] = b"\0" * th[5]
    ph = g.phdr(j)
    ph[2], ph[5], ph[6] = g.shdrs()[ti][4], len(ctl_text), len(ctl_text)
    g.set_phdr(j, ph)
    k = th[7]
    s = g.sym(k)
    s[5] = len(ctl_text)
    g.set_sym(k, s)
    mi = g.index(".nv.info")
    g.replace(mi, nvinfo_pack(set_attr(nvinfo(g.data(mi)), REGCOUNT, struct.pack("<II", k, 40))), 4)
    ki = g.index(".nv.info." + FN)
    g.replace(ki, nvinfo_pack(set_attr(nvinfo(g.data(ki)), MAXREG, b"\xff\0")), 4)
    emit("G-omegacontainer-ctl-sass", bytes(g.b), FN, "run", "UNKNOWN: load => CUDApp container accepted", "machine code vs container (CUDApp container)")
    # H: register count and ISA class, the only E5-vs-F metadata differences left
    # (static diff 2026-10-31: same text, tkinfo, cuinfo, cbank0, e_flags).
    def with_meta(base, regs=None, maxreg=None, isa=None):
        v = Elf(base)
        def put(name, ents):
            i = v.index(name)
            h = v.shdrs()[i]
            new = nvinfo_pack(ents)
            assert len(new) == h[5]
            v.b[h[4] : h[4] + h[5]] = new
        if regs is not None:
            ents = nvinfo(v.data(v.index(".nv.info")))
            sym = struct.unpack_from("<I", next(x for x in ents if x[1] == REGCOUNT)[2])[0]
            put(".nv.info", set_attr(ents, REGCOUNT, struct.pack("<II", sym, regs)))
        if maxreg is not None:
            put(".nv.info." + FN, set_attr(nvinfo(v.data(v.index(".nv.info." + FN))), MAXREG, struct.pack("<H", maxreg)))
        if isa is not None:
            put(".nv.compat", set_attr(nvinfo(v.data(v.index(".nv.compat"))), ISA_CLASS, struct.pack("<H", isa)))
        return bytes(v.b)
    e3 = (OUT / "E3-omega-tkinfo-own.cubin").read_bytes()
    emit("H1-e5-reg40", with_meta(e5, regs=40, maxreg=40), FN, "run", "PASS if register count", "REGCOUNT+MAXREG 40 (peak+2 rounded to 8)")
    emit("H2-e5-regcount40", with_meta(e5, regs=40), FN, "run", "UNKNOWN", "REGCOUNT 0x2f only")
    emit("H3-e5-maxreg40", with_meta(e5, maxreg=40), FN, "run", "UNKNOWN", "MAXREG 0x1b only")
    emit("H4-e5-reg34", with_meta(e5, regs=34, maxreg=34), FN, "run", "UNKNOWN", "REGCOUNT+MAXREG 34 (peak+2, ptxas rule)")
    emit("H5-e5-isa1", with_meta(e5, isa=1), FN, "run", "715 if register count", "nv.compat ISA_CLASS 1 (ptxas value)")
    emit("H6-e3-reg40", with_meta(e3, regs=40, maxreg=40), FN, "run", "PASS if tkinfo+regs suffice", "minimal writer fix: own tkinfo + regs 40")
    emit("H7-d-reg40", with_meta(main_raw, regs=40, maxreg=40), FN, "reject", "REJECT 200 (no tkinfo)", "regs alone do not fix the load")
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
