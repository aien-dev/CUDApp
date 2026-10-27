#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Draw the README figures (docs/img/*.svg) from this repository's own data.

Run from the repository root:   python3 tools/figures/make_figures.py
Check they are current:         python3 tools/figures/make_figures.py --check

Nothing here is drawn by an image generator. The machine-code figures read the
instruction words straight out of src/omega_blackwell_encoder.c and decode them
with the field positions my encoder uses (src/omega_blackwell_codegen.c,
"Control words"). The numbers on the other figures are the receipts named on
each figure. Colours are the aienos.com "Turing look" tokens.
"""
import os
import re
import sys
from html import escape

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "docs", "img")

# ---- Turing look tokens (aienos.com public/assets/turing-look.css) ----------------
BG, CARD, CARD2 = "#08090c", "#0e1219", "#131822"
LINE, LINE2 = "#1c2536", "#2a3648"
INK, INK2, INK3 = "#f1f5f9", "#94a3b8", "#64748b"
BLUE, BLUE_SOFT, GREEN = "#2d7ff9", "#7db4ff", "#00e599"
AMBER, VIOLET, ROSE, CYAN = "#f59e0b", "#a78bfa", "#fb7185", "#22d3ee"

SANS = "Inter, ui-sans-serif, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, 'Liberation Mono', monospace"


# ---- tiny SVG writer ---------------------------------------------------------------
class Svg:
    def __init__(self, w, h, title, desc):
        self.w, self.h = w, h
        self.parts = []
        self.title, self.desc = title, desc

    def add(self, s):
        self.parts.append(s)

    def rect(self, x, y, w, h, fill="none", stroke="none", rx=0.0, sw=1.0, op=1.0, dash=None, extra=""):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        o = f' opacity="{op}"' if op != 1 else ""
        self.add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" '
                 f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}{o}{extra}/>')

    def text(self, x, y, s, size=16.0, fill=INK, weight=400, anchor="start", mono=False, op=1.0, ls=None, italic=False, strike=False):
        fam = MONO if mono else SANS
        size = max(size, 12.5)  # README width scales figures to about 0.73: keep the smallest text readable
        o = f' opacity="{op}"' if op != 1 else ""
        l = f' letter-spacing="{ls}"' if ls is not None else ""
        it = ' font-style="italic"' if italic else ""
        it += ' text-decoration="line-through"' if strike else ""
        self.add(f'<text x="{x:.1f}" y="{y:.1f}" font-family="{fam}" font-size="{size}" font-weight="{weight}" '
                 f'fill="{fill}" text-anchor="{anchor}"{o}{l}{it}>{escape(s)}</text>')

    def lines(self, x, y, rows, size=15.0, fill=INK2, gap=None, **kw):
        gap = gap or size * 1.4
        for i, r in enumerate(rows):
            self.text(x, y + i * gap, r, size=size, fill=fill, **kw)

    def line(self, x1, y1, x2, y2, stroke=LINE2, sw=1.5, dash=None, arrow=False, op=1.0):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        a = ' marker-end="url(#arrow)"' if arrow else ""
        o = f' opacity="{op}"' if op != 1 else ""
        self.add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" '
                 f'stroke-width="{sw}"{d}{a}{o}/>')

    def path(self, d, stroke=LINE2, sw=1.5, fill="none", arrow=False, dash=None, op=1.0):
        a = ' marker-end="url(#arrow)"' if arrow else ""
        ds = f' stroke-dasharray="{dash}"' if dash else ""
        o = f' opacity="{op}"' if op != 1 else ""
        self.add(f'<path d="{d}" stroke="{stroke}" stroke-width="{sw}" fill="{fill}"{a}{ds}{o}/>')

    def circle(self, x, y, r, fill=INK, stroke="none", sw=1.0):
        self.add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')

    def pill(self, x, y, label, color, size=13.0, h=24.0, filled=False, w=None):
        w = w or (len(label) * size * 0.62 + 22)
        if filled:
            self.rect(x, y, w, h, fill=color, rx=h / 2, op=0.16)
            self.rect(x, y, w, h, stroke=color, rx=h / 2, sw=1, op=0.7)
        else:
            self.rect(x, y, w, h, stroke=color, rx=h / 2, sw=1.2)
        self.text(x + w / 2, y + h / 2 + size * 0.36, label, size=size, fill=color, weight=600, anchor="middle",
                  mono=True)
        return w

    def kicker(self, x, y, s, color=BLUE_SOFT):
        self.text(x, y, s.upper(), size=14, fill=color, weight=600, mono=True, ls=1.5)

    def frame(self):
        self.rect(0.5, 0.5, self.w - 1, self.h - 1, fill=BG, stroke=LINE, rx=22)
        self.rect(0.5, 0.5, self.w - 1, self.h - 1, fill="url(#glow)", rx=22)

    def svg(self):
        head = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" '
                f'viewBox="0 0 {self.w} {self.h}" role="img" aria-labelledby="t d">'
                f'<title id="t">{escape(self.title)}</title><desc id="d">{escape(self.desc)}</desc>'
                '<defs>'
                '<radialGradient id="glow" cx="22%" cy="-10%" r="85%">'
                f'<stop offset="0" stop-color="{BLUE}" stop-opacity="0.16"/>'
                f'<stop offset="0.55" stop-color="{BLUE}" stop-opacity="0.03"/>'
                f'<stop offset="1" stop-color="{BLUE}" stop-opacity="0"/></radialGradient>'
                '<radialGradient id="glow2" cx="92%" cy="8%" r="45%">'
                f'<stop offset="0" stop-color="{GREEN}" stop-opacity="0.08"/>'
                f'<stop offset="1" stop-color="{GREEN}" stop-opacity="0"/></radialGradient>'
                '<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
                f'orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="{INK2}"/></marker>'
                '</defs>')
        return head + "".join(self.parts) + "</svg>\n"


def card(s, x, y, w, h, accent=None, fill=CARD, rx=16.0, sw=1.0, dash=None):
    s.rect(x, y, w, h, fill=fill, stroke=accent or LINE2, rx=rx, sw=sw, dash=dash)


# ---- the first kernel's words, read from the encoder source ------------------------
def vecadd_words():
    src = open(os.path.join(ROOT, "src", "omega_blackwell_encoder.c")).read()
    rows = re.findall(r'/\* ([0-9a-f]{4}) \*/ \{ \{ (0x[0-9a-f]+), (0x[0-9a-f]+), (0x[0-9a-f]+), (0x[0-9a-f]+) \}, '
                      r'"([^"]+)" \}', src)
    if len(rows) != 32:
        sys.exit(f"expected 32 vecadd instructions in src/omega_blackwell_encoder.c, found {len(rows)}")
    out = []
    for addr, w0, w1, w2, w3, mn in rows:
        w = [int(v, 16) for v in (w0, w1, w2, w3)]
        bits = w[0] | (w[1] << 32) | (w[2] << 64) | (w[3] << 96)
        out.append({"addr": addr, "w": w, "bits": bits, "mn": mn, **decode_control(w[3])})
    return out


def decode_control(w3):
    # src/omega_blackwell_codegen.c, "Control words": stall = bits 9..12 of w[3], yield = 13,
    # write barrier = 14..16, read barrier = 17..19, wait mask = 20..25; 7 in a barrier field means none.
    return {"stall": (w3 >> 9) & 0xF, "yield": (w3 >> 13) & 1, "wbar": (w3 >> 14) & 7,
            "rbar": (w3 >> 17) & 7, "wait": [i for i in range(6) if (w3 >> (20 + i)) & 1],
            "reuse": (w3 >> 26) & 0xF}


# Bit fields as drawn (bit 0 = lowest bit of word 0).
FIELDS = [  # (lo, hi, name, colour)
    (0, 11, "opcode and operand form", BLUE),
    (12, 15, "guard predicate", VIOLET),
    (16, 31, "destination and first source register", GREEN),
    (32, 63, "second source: register, immediate or constant address", CYAN),
    (64, 104, "third source, predicate operands, modifiers", INK2),
    (105, 127, "scheduling: stall, yield, scoreboards, reuse", AMBER),
]


def field_colour(b):
    for lo, hi, _, c in FIELDS:
        if lo <= b <= hi:
            return c
    return INK2


def short_mn(mn):
    pred = ""
    if mn.startswith("@"):
        pred, mn = mn.split(" ", 1)
        pred += " "
    return pred + mn.split(" ")[0]


# ---- figure 1: hero ------------------------------------------------------------------
def fig_hero(words):
    s = Svg(1280, 680, "CUDA++: my own path from a program to an NVIDIA GPU",
            "Title, three measured facts, and the 32 instructions of the first kernel CUDA++ ran, drawn bit by bit "
            "from src/omega_blackwell_encoder.c.")
    s.frame()
    s.rect(0.5, 0.5, s.w - 1, s.h - 1, fill="url(#glow2)", rx=22)
    s.kicker(64, 92, "aien-dev / CUDApp  ·  MIT  ·  NVIDIA GB10 (sm_121)")
    s.text(58, 196, "CUDA++", size=104, weight=800, ls=-3)
    s.lines(64, 250, ["My own path from a program to an NVIDIA GPU:",
                      "machine code, runtime, libraries and driver client.",
                      "No CUDA toolkit, no nvcc, no PTX, no libcuda."], size=22, fill=INK2, gap=32)
    facts = [("0", GREEN, "CUDA libraries loaded in the live", "35B model server (2026-10-21)"),
             ("115.14", GREEN, "TFLOPS tensor-core peak, measured", "by my own probe kernel"),
             ("59", GREEN, "instruction kinds my encoder", "writes for sm_121")]
    for i, (num, col, l1, l2) in enumerate(facts):
        y = 372 + i * 80
        card(s, 64, y, 500, 66, fill=CARD)
        s.rect(64, y, 4, 66, fill=col, rx=2)
        s.text(88, y + 44, num, size=34, weight=800, fill=col, ls=-1)
        s.lines(232, y + 28, [l1, l2], size=16, fill=INK2, gap=21)

    x0, y0, cw, rh = 620, 168, 3.4, 13.5
    s.kicker(x0, 92, "the first kernel: vector add", INK2)
    s.text(x0, 116, "32 instructions × 128 bits, every bit written by my encoder", size=15, fill=INK3)
    for i, (lo, hi, name, col) in enumerate(FIELDS):
        label = {BLUE: "opcode", VIOLET: "predicate", GREEN: "registers", CYAN: "operand",
                 INK2: "modifiers", AMBER: "scheduling"}[col]
        lx = x0 + i * 100
        s.rect(lx, 133, 10, 10, fill=col, rx=2)
        s.text(lx + 15, 142, label, size=13, fill=INK2)
    for r, ins in enumerate(words):
        y = y0 + r * rh
        nop = ins["mn"] == "NOP"
        for lo, hi, _, col in FIELDS:
            s.rect(x0 + lo * cw, y, (hi - lo + 1) * cw - 0.8, rh - 3, fill=col, op=0.07 if not nop else 0.04)
        for b in range(128):
            if (ins["bits"] >> b) & 1:
                s.rect(x0 + b * cw, y, cw - 0.8, rh - 3, fill=field_colour(b), op=0.95 if not nop else 0.35)
        s.text(x0 + 128 * cw + 14, y + rh - 4.5, ins["addr"], size=11.5, fill=INK3, mono=True)
        s.text(x0 + 128 * cw + 52, y + rh - 4.5, short_mn(ins["mn"]), size=11.5, fill=INK2 if not nop else INK3,
               mono=True, op=1 if not nop else 0.6)
    for b in (0, 32, 64, 96, 127):
        s.line(x0 + b * cw + cw / 2, y0 + 32 * rh - 1, x0 + b * cw + cw / 2, y0 + 32 * rh + 4, stroke=INK3, sw=1)
        s.text(x0 + b * cw + cw / 2, y0 + 32 * rh + 17, str(b), size=11, fill=INK3, anchor="middle", mono=True)
    s.text(x0, y0 + 32 * rh + 40, "bit 0 to bit 127 of each instruction · lit = 1 · src/omega_blackwell_encoder.c",
           size=13, fill=INK2, mono=True)
    return s


# ---- figure 2: the two stacks ----------------------------------------------------------
def fig_stack():
    s = Svg(1200, 930, "CUDA and CUDA++, layer by layer",
            "Left: the NVIDIA CUDA stack. Right: CUDA++. The four closed layers above the kernel module are "
            "replaced by my own code; the open kernel module and the signed GPU firmware are still NVIDIA's.")
    s.frame()
    s.kicker(48, 60, "the stack")
    s.text(48, 100, "What runs between your program and the GPU", size=30, weight=800, ls=-0.5)
    lx, rx, cw, mid = 48, 664, 488, 600
    s.text(lx, 150, "CUDA", size=22, weight=700, fill=INK2)
    s.text(rx, 150, "CUDA++", size=22, weight=700, fill=GREEN)
    rows = [
        ("program", ("Your program", "CUDA C++ kernels and host code"),
         ("Your program", "C calls to the kernel builder; Omega's languages next (bar C1)")),
        ("closed", ("nvcc · NVVM · PTX · ptxas", "compiler, closed"),
         ("Kernel builder · allocator · sm_121 encoder", "src/omega_blackwell_codegen*.c, _encoder.c, _qmd.c")),
        ("closed", ("cuBLAS · cuDNN · CUB", "maths libraries, closed"),
         ("Matmul · attention · MoE · reductions", "src/omega_gpu_*_api.c")),
        ("closed", ("libcudart", "runtime API, closed"),
         ("Session · command lists · job engine", "src/omega_gpu_session.c, _cmdlist.c, _engine.c")),
        ("closed", ("libcuda", "driver API, closed"),
         ("Driver client: nvrm + m16", "nvrm/nvrm.c, m16/: ioctls from the open headers")),
        ("open", ("nvidia.ko · nvidia-uvm.ko", "NVIDIA's open kernel module"),
         ("nvidia.ko · nvidia-uvm.ko", "stays; GB10 support goes upstream to Nova, nouveau")),
        ("fw", ("GSP firmware", "runs inside the GPU, signed by NVIDIA"),
         ("GSP firmware", "stays NVIDIA's: the chip checks its signature")),
        ("hw", ("NVIDIA GB10 GPU", "DGX Spark, sm_121"), ("NVIDIA GB10 GPU", "DGX Spark, sm_121")),
    ]
    marks = {"closed": ("replaced", GREEN), "open": ("stays", VIOLET), "fw": ("stays", AMBER)}
    y, rh, gap = 172, 72, 14
    for i, (kind, left, right) in enumerate(rows):
        if i == 5:
            y += 30
            s.line(36, y - 22, s.w - 36, y - 22, stroke=INK3, sw=1, dash="6 6")
            w = 196
            s.rect(mid - w / 2, y - 33, w, 22, fill=BG)
            s.text(mid, y - 17, "user space ↑   kernel ↓", size=13, fill=INK2, anchor="middle", mono=True)
        # left
        if kind == "closed":
            card(s, lx, y, cw, rh, accent=LINE2, fill=CARD)
            s.text(lx + 22, y + 32, left[0], size=19, weight=650, fill=INK3, strike=True)
            s.text(lx + 22, y + 56, left[1], size=14, fill=INK3)
        else:
            acc = {"program": LINE2, "open": BLUE, "fw": AMBER, "hw": LINE2}[kind]
            card(s, lx, y, cw, rh, accent=acc, fill=CARD)
            s.text(lx + 22, y + 32, left[0], size=19, weight=650)
            s.text(lx + 22, y + 56, left[1], size=14, fill=INK2)
        # right
        acc = {"program": LINE2, "closed": GREEN, "open": BLUE, "fw": AMBER, "hw": LINE2}[kind]
        fill = "#0b1a17" if kind == "closed" else CARD
        card(s, rx, y, cw, rh, accent=acc, fill=fill, sw=1.4 if kind == "closed" else 1)
        s.text(rx + 22, y + 32, right[0], size=19, weight=650, fill=INK)
        s.text(rx + 22, y + 56, right[1], size=14 if kind != "closed" else 13.5, fill=INK2,
               mono=(kind in ("closed",)))
        if kind in marks:
            lab, col = marks[kind]
            w = len(lab) * 8.6 + 18
            s.line(lx + cw + 4, y + rh / 2, mid - w / 2, y + rh / 2, stroke=col, sw=1.6, op=0.8)
            s.line(mid + w / 2, y + rh / 2, rx - 10, y + rh / 2, stroke=col, sw=1.6, arrow=True, op=0.8)
            s.rect(mid - w / 2, y + rh / 2 - 12, w, 24, fill=BG, rx=12)
            s.pill(mid - w / 2, y + rh / 2 - 12, lab, col, size=13, w=w, filled=True)
        y += rh + gap
    ly = y + 16
    for lab, col, x in (("closed, replaced by CUDA++", INK3, 48), ("first-party, Apache-2.0+LLVM", GREEN, 300),
                        ("NVIDIA open source, stays", BLUE, 440), ("NVIDIA signed firmware, stays", AMBER, 740)):
        s.rect(x, ly - 11, 14, 14, stroke=col, fill="#0b1a17" if col == GREEN else CARD, rx=3, sw=1.5)
        s.text(x + 22, ly + 1, lab, size=14, fill=INK2)
    return s


# ---- figure 3: one instruction, every bit ---------------------------------------------
def fig_instruction(words):
    ins = next(w for w in words if w["mn"].startswith("IADD3"))
    s = Svg(1200, 800, "One instruction, all 128 bits: IADD3 R9, PT, PT, R2, R5, RZ",
            "The add of vector add, decoded field by field, with its scheduling bits enlarged.")
    s.frame()
    s.kicker(48, 60, "one instruction, all 128 bits")
    s.text(48, 102, "IADD3 R9, PT, PT, R2, R5, RZ", size=32, weight=800, mono=True, ls=-0.5)
    s.text(48, 134, f"R9 = R2 + R5: the add in vector add, at offset 0x{ins['addr']}. "
                    "Sixteen bytes, four 32-bit words.", size=17, fill=INK2)
    x0, y0, cw, ch, wg = 56, 196, 8.2, 38, 10

    def bx(b):
        return x0 + b * cw + (b // 32) * wg

    for k in range(4):
        s.text(bx(32 * k), y0 - 12, f"word {k}  0x{ins['w'][k]:08x}", size=14, fill=INK2, mono=True)
    for b in range(128):
        on = (ins["bits"] >> b) & 1
        col = field_colour(b)
        s.rect(bx(b), y0, cw - 1.2, ch, fill=col, op=0.95 if on else 0.12, rx=1.5)
    for b in (0, 31, 32, 63, 64, 95, 96, 127):
        s.text(bx(b) + cw / 2, y0 + ch + 16, str(b), size=11.5, fill=INK3, anchor="middle", mono=True)

    def bracket(lo, hi, row, title, value, col):
        x1, x2 = bx(lo) + 1.5, bx(hi) + cw - 2.7
        yb = y0 + ch + 30 + row * 60
        s.path(f"M{x1:.1f} {yb - 6} L{x1:.1f} {yb} L{x2:.1f} {yb} L{x2:.1f} {yb - 6}", stroke=col, sw=1.6)
        s.line((x1 + x2) / 2, yb, (x1 + x2) / 2, yb + 8, stroke=col, sw=1.6)
        s.text(x1, yb + 26, value, size=16, weight=700, fill=col, mono=True)
        s.text(x1, yb + 45, title, size=13, fill=INK2)

    rd, ra = (ins["w"][0] >> 16) & 0xFF, (ins["w"][0] >> 24) & 0xFF
    rb, rc = ins["w"][1] & 0xFF, ins["w"][2] & 0xFF
    g = (ins["w"][0] >> 12) & 7
    reg = lambda r: "RZ" if r == 255 else f"R{r}"
    bracket(0, 11, 0, "opcode + operand form", f"0x{ins['w'][0] & 0xfff:03x} IADD3", BLUE)
    bracket(12, 15, 1, "guard predicate", "@PT" if g == 7 else f"@P{g}", VIOLET)
    bracket(16, 23, 2, "destination", reg(rd), GREEN)
    bracket(24, 31, 0, "source A", reg(ra), GREEN)
    bracket(32, 39, 1, "source B", reg(rb), CYAN)
    bracket(64, 71, 0, "source C", reg(rc), INK2)
    bracket(72, 104, 1, "carry predicates, modifiers", "PT, PT", INK2)
    bracket(105, 127, 0, "scheduling (enlarged below)", "control", AMBER)

    # enlarged control bits
    zy, zx, zw = 530, 70, 44
    card(s, 40, zy - 40, 1120, 240, fill=CARD)
    s.text(60, zy - 12, "bits 105 to 127: how the warp scheduler treats this instruction", size=16, weight=700,
           fill=AMBER)
    for i, b in enumerate(range(105, 128)):
        on = (ins["bits"] >> b) & 1
        s.rect(zx + i * zw, zy + 8, zw - 4, 40, fill=AMBER, op=0.9 if on else 0.12, rx=4)
        s.text(zx + i * zw + (zw - 4) / 2, zy + 35, str(on), size=17, weight=700, fill=BG if on else INK2,
               anchor="middle", mono=True)
        s.text(zx + i * zw + (zw - 4) / 2, zy + 64, str(b), size=11, fill=INK3, anchor="middle", mono=True)
    c = ins
    sub = [(105, 108, "stall", f"{c['stall']} cycles"), (109, 109, "yield", str(c["yield"])),
           (110, 112, "sets SB", "none" if c["wbar"] == 7 else f"SB{c['wbar']}"),
           (113, 115, "read SB", "none" if c["rbar"] == 7 else f"SB{c['rbar']}"),
           (116, 121, "waits on", ", ".join(f"SB{i}" for i in c["wait"]) or "none"),
           (122, 125, "reuse", str(c["reuse"])), (126, 127, "unused", "0")]
    for lo, hi, name, val in sub:
        x1 = zx + (lo - 105) * zw
        x2 = zx + (hi - 105) * zw + zw - 4
        yb = zy + 76
        s.path(f"M{x1} {yb} L{x1} {yb + 6} L{x2} {yb + 6} L{x2} {yb}", stroke=AMBER, sw=1.4)
        s.text(x1, yb + 30, val, size=15, weight=700, fill=INK, mono=True)
        s.text(x1, yb + 49, name, size=13, fill=INK2)
    s.text(60, zy + 180, "SB4 is the scoreboard both loads (LDG.E R2 and LDG.E R5) set: the add waits until both "
                         "values have arrived from memory.", size=14, fill=INK2)
    s.text(48, s.h - 24, "Field positions: src/omega_blackwell_codegen.c (\"Control words\"), learned from nvcc output "
                         "on my own probe kernels and Mesa NAK, believed only after the GB10 ran them.",
           size=13, fill=INK3)
    return s


# ---- figure 4: scoreboards in vector add -------------------------------------------------
def fig_listing(words):
    body = words[:21]
    rh, top = 30, 196
    s = Svg(1200, top + len(body) * rh + 90, "Vector add, instruction by instruction, with its scoreboards",
            "The 21 real instructions of the first kernel. Lines on the left join an instruction that sets a "
            "scoreboard to the later instruction that waits on it; amber bars are fixed stall cycles.")
    s.frame()
    s.kicker(48, 60, "the first kernel, decoded")
    s.text(48, 100, "How vector add waits for its data", size=30, weight=800, ls=-0.5)
    s.lines(48, 132, ["Slow instructions (constant loads, special registers, memory loads) set one of six scoreboards. "
                      "A later instruction",
                      "that needs the result waits on it. Fast instructions stall a fixed number of cycles instead. "
                      "My encoder writes these bits."], size=15, fill=INK2, gap=21)
    sbcol = [CYAN, VIOLET, GREEN, ROSE, AMBER, BLUE]
    lane = lambda k: 228 - k * 34
    hy = top - 14
    for lab, x in (("offset", 270), ("instruction", 336), ("stall", 760), ("sets", 940), ("waits on", 1040)):
        s.text(x, hy, lab.upper(), size=12, fill=INK3, mono=True, ls=1)
    for k in range(5):
        s.text(lane(k), hy, f"SB{k}", size=12, fill=sbcol[k], anchor="middle", mono=True, weight=700)
    ry = lambda r: top + r * rh + rh / 2
    for r, ins in enumerate(body):
        if r % 2 == 0:
            s.rect(250, top + r * rh, 920, rh, fill=CARD, rx=6)
        y = ry(r) + 5
        s.text(270, y, ins["addr"], size=14, fill=INK3, mono=True)
        s.text(336, y, ins["mn"], size=14.5, fill=INK, mono=True)
        if ins["stall"]:
            s.rect(760, ry(r) - 6, ins["stall"] * 11, 12, fill=AMBER, rx=3, op=0.85)
            s.text(760 + ins["stall"] * 11 + 8, y, str(ins["stall"]), size=12, fill=INK3, mono=True)
        if ins["wbar"] != 7:
            s.pill(940, ry(r) - 11, f"SB{ins['wbar']}", sbcol[ins["wbar"]], size=12, h=22, filled=True, w=52)
        for j, k in enumerate(ins["wait"]):
            s.pill(1040 + j * 60, ry(r) - 11, f"SB{k}", sbcol[k], size=12, h=22, w=52)
    # lanes: every wait joins the setters of that scoreboard since its previous wait
    pending = {k: [] for k in range(6)}
    for r, ins in enumerate(body):
        for k in ins["wait"]:
            setters = pending[k]
            if setters:
                x = lane(k)
                s.line(x, ry(setters[0]), x, ry(r), stroke=BG, sw=7)
                s.line(x, ry(setters[0]), x, ry(r), stroke=sbcol[k], sw=2.4)
                for sr in setters:
                    s.line(x, ry(sr), 246, ry(sr), stroke=BG, sw=6)
                    s.line(x, ry(sr), 246, ry(sr), stroke=sbcol[k], sw=1.4, op=0.7, dash="3 4")
                    s.circle(x, ry(sr), 5, fill=sbcol[k], stroke=BG, sw=2)
                s.line(x, ry(r), 252, ry(r), stroke=BG, sw=6)
                s.line(x, ry(r), 252, ry(r), stroke=sbcol[k], sw=1.8, arrow=True)
            pending[k] = []
        if ins["wbar"] != 7:
            pending[ins["wbar"]].append(r)
    s.text(48, s.h - 30, "Instruction words: src/omega_blackwell_encoder.c. Decoded by tools/figures/make_figures.py with the "
                         "encoder's own field positions.", size=13, fill=INK3)
    return s


# ---- figure 5: how the machine code was learned ---------------------------------------------
def fig_reverse():
    s = Svg(1200, 760, "How I learned the GB10's machine code",
            "Three sources and one rule: NVIDIA's compiler output on my own probe kernels, Mesa's open NAK "
            "compiler, and the chip itself. Nothing is believed until the GPU runs it and gives the right answer.")
    s.frame()
    s.kicker(48, 60, "the reverse engineering")
    s.text(48, 100, "How I learned the GB10's machine code", size=30, weight=800, ls=-0.5)
    s.text(48, 132, "NVIDIA publishes no instruction encoding for its GPUs. I worked it out, and checked every "
                    "answer on the chip.", size=16, fill=INK2)

    def box(x, y, w, h, n, title, sub, col):
        card(s, x, y, w, h, accent=col, fill=CARD)
        s.text(x + 18, y + 30, n, size=13, fill=col, mono=True, weight=700)
        s.text(x + 18, y + 56, title, size=17, weight=700)
        s.lines(x + 18, y + 80, sub, size=13.5, fill=INK2, gap=19)

    s.text(48, 180, "LEARN", size=13, fill=INK3, mono=True, ls=2, weight=700)
    W, H, y1 = 258, 150, 194
    xs = [48, 334, 620, 906]
    box(xs[0], y1, W, H, "01", "My own probe kernel", ["a few lines of CUDA C++ I", "wrote, e.g. one add"], BLUE_SOFT)
    box(xs[1], y1, W, H, "02", "nvcc, once, offline", ["nvcc 13.0.88 -arch=sm_121", "-cubin. Never run by", "CUDA++ itself."], INK2)
    box(xs[2], y1, W, H, "03", "Read the bytes back", ["cuobjdump -sass: 16 bytes", "next to each mnemonic"], INK2)
    box(xs[3], y1, W, H, "04", "Change one detail", ["a register, an immediate, a", "predicate, a modifier; diff", "the bits that moved"], BLUE_SOFT)
    for a, b in zip(xs, xs[1:]):
        s.line(a + W + 4, y1 + H / 2, b - 6, y1 + H / 2, stroke=INK2, sw=1.6, arrow=True)

    s.text(48, 404, "PROVE", size=13, fill=INK3, mono=True, ls=2, weight=700)
    y2 = 418
    box(48, y2, 258, 150, "05", "Field hypothesis", ["bits 16..23 are the", "destination register,", "and so on"], GREEN)
    box(334, y2, 258, 150, "06", "My encoder writes it", ["src/omega_blackwell_", "encoder.c, codegen*.c"], GREEN)
    box(620, y2, 258, 150, "07", "The GB10 runs it", ["a real launch on the chip,", "answers checked on the CPU"], GREEN)
    box(906, y2, 258, 150, "08", "Golden words", ["kept in the tests; nvdisasm", "cross-check when installed"], GREEN)
    s.path(f"M{xs[3] + W / 2} {y1 + H + 4} C {xs[3] + W / 2} 392, 177 380, 177 {y2 - 8}", stroke=INK2, sw=1.6,
           arrow=True)
    s.line(306 + 4, y2 + 75, 334 - 6, y2 + 75, stroke=INK2, sw=1.6, arrow=True)
    s.line(592 + 4, y2 + 75, 620 - 6, y2 + 75, stroke=INK2, sw=1.6, arrow=True)
    s.line(878 + 4, y2 + 75, 906 - 6, y2 + 75, stroke=GREEN, sw=2, arrow=True)
    s.text(892, y2 - 10, "right, bit for bit ↘", size=12.5, fill=GREEN, anchor="middle", mono=True)
    s.path(f"M749 {y2 + H + 4} C 749 {y2 + H + 52}, 177 {y2 + H + 52}, 177 {y2 + H + 8}", stroke=ROSE, sw=1.6,
           arrow=True, dash="5 5")
    s.text(463, y2 + H + 56, "wrong answer: back to the hypothesis", size=13, fill=ROSE, anchor="middle", mono=True)

    yb = 652
    card(s, 48, yb, 540, 84, accent=VIOLET, fill=CARD)
    s.text(66, yb + 30, "Mesa NAK (MIT), cited, no code copied", size=15, weight=700, fill=VIOLET)
    s.lines(66, yb + 54, ["sm70_encode.rs opcodes and bit positions, sm120 latencies,", "qmd.rs launch descriptor layout"],
            size=13, fill=INK2, gap=18)
    card(s, 612, yb, 552, 84, accent=AMBER, fill=CARD)
    s.text(630, yb + 30, "Arithmetic the documents leave open", size=15, weight=700, fill=AMBER)
    s.lines(630, yb + 54, ["tensor-core BF16: 2,048 chosen cases until one model matched", "every output bit (tests/gpu_tc_probe_test.c)"],
            size=13, fill=INK2, gap=18)
    return s


# ---- figure 6: the driver client's objects ------------------------------------------------
def fig_driver():
    s = Svg(1200, 1160, "The driver client: every object CUDA++ asks NVIDIA's kernel module for",
            "Device files on the left, the resource-manager object tree in the middle, the unified-memory "
            "registrations on the right, all from nvrm/nvrm.c.")
    s.frame()
    s.kicker(48, 60, "the driver client, part by part")
    s.text(48, 100, "What replaces libcuda: nvrm/nvrm.c", size=30, weight=800, ls=-0.5)
    s.lines(48, 132, ["About 900 lines of C that talk to NVIDIA's open kernel module through ioctls, written from its "
                      "MIT headers. Each box is",
                      "one object CUDA++ allocates (RM_ALLOC), with its class name, the header that defines it, and the "
                      "function that creates it."], size=15, fill=INK2, gap=21)

    # device files
    s.text(48, 196, "DEVICE FILES", size=12, fill=INK3, mono=True, ls=1.5, weight=700)
    devs = [("/dev/nvidiactl", ["resource manager: allocate", "objects, controls, free"], BLUE),
            ("/dev/nvidia0", ["this GPU: CPU mappings of", "memory and the doorbell page"], BLUE),
            ("/dev/nvidia-uvm", ["unified memory: GPU page", "tables of the address space"], VIOLET)]
    for i, (name, sub, col) in enumerate(devs):
        y = 210 + i * 118
        card(s, 48, y, 236, 100, accent=col)
        s.text(66, y + 32, name, size=16, weight=700, mono=True, fill=col)
        s.lines(66, y + 58, sub, size=13, fill=INK2, gap=18)

    # object tree
    s.text(316, 196, "RESOURCE-MANAGER OBJECTS", size=12, fill=INK3, mono=True, ls=1.5, weight=700)
    nodes = [  # depth, title, class, header, note, colour, creator
        (0, "Root client", "NV01_ROOT_CLIENT", "cl0000.h", "the handle that owns everything below", BLUE, "nvrm_open"),
        (1, "Device", "NV01_DEVICE_0", "cl0080.h", "the GPU; its class list names the Blackwell classes", BLUE, ""),
        (2, "Subdevice", "NV20_SUBDEVICE_0", "cl2080.h", "SM version (sm_121) and GPU UUID", BLUE, ""),
        (3, "Usermode page", "HOPPER_USERMODE_A", "clc661.h", "64 KiB of GPU registers; holds the doorbell", AMBER, ""),
        (2, "Virtual memory", "NV01_MEMORY_VIRTUAL", "cl0070.h", "the GPU address range CUDA++ may use", BLUE, ""),
        (2, "Address space", "FERMI_VASPACE_A", "cl90f1.h", "page faulting, handed to UVM to manage", VIOLET, ""),
        (2, "Memory, one per buffer", "NV01_MEMORY_SYSTEM", "cl003e.h", "GB10 has no VRAM: same address for CPU and GPU", GREEN, "nvrm_alloc"),
        (2, "Channel group", "KEPLER_CHANNEL_GROUP_A", "cla06c.h", "scheduled on the graphics/compute engine", BLUE, "nvrm_channel"),
        (3, "Context share", "FERMI_CONTEXT_SHARE_A", "cl9067.h", "binds the channel to the address space", BLUE, ""),
        (3, "Command ring (GPFIFO)", "BLACKWELL_CHANNEL_GPFIFO_A", "clc96f.h", "1,024 entries + USERD control page; submit token", GREEN, ""),
        (4, "Compute object", "BLACKWELL_COMPUTE_B", "clcec0.h", "the engine every kernel launch runs on", GREEN, ""),
    ]
    tx, nw, nh, ng, ind = 316, 560, 64, 12, 30
    pos = []
    for i, (d, title, cls, hdr, note, col, fn) in enumerate(nodes):
        x, y = tx + d * ind, 210 + i * (nh + ng)
        w = nw - d * ind
        card(s, x, y, w, nh, accent=col if col != BLUE else LINE2)
        s.rect(x, y + 10, 3, nh - 20, fill=col, rx=1.5)
        s.text(x + 16, y + 25, title, size=15.5, weight=700)
        if fn:
            s.text(x + 22 + len(title) * 8.4, y + 25, fn + "()", size=12, fill=INK3, mono=True)
        s.text(x + w - 14, y + 25, cls, size=12, fill=col, mono=True, anchor="end")
        s.text(x + 16, y + 48, note, size=13, fill=INK2)
        s.text(x + w - 14, y + 48, hdr, size=11.5, fill=INK3, mono=True, anchor="end")
        pos.append((d, x, y))
    # connectors: each node joins its parent (the closest earlier node one level up)
    for i, (d, x, y) in enumerate(pos):
        if d == 0:
            continue
        p = next(j for j in range(i - 1, -1, -1) if pos[j][0] == d - 1)
        px = pos[p][1] + 14
        s.path(f"M{px} {pos[p][2] + nh} L{px} {y + nh / 2} L{x} {y + nh / 2}", stroke=LINE2, sw=1.4)
    # device files feed the objects: control node to the root, GPU node to the usermode mapping
    s.path(f"M284 260 C 300 260, 300 242, {tx - 4} 242", stroke=BLUE, sw=1.3, dash="3 4", arrow=True)
    s.path(f"M284 378 C 300 378, 300 {pos[3][2] + 32}, {pos[3][1] - 4} {pos[3][2] + 32}", stroke=AMBER, sw=1.3,
           dash="3 4", arrow=True)

    # UVM registrations
    ux = 912
    s.text(ux, 196, "UNIFIED MEMORY (UVM) CALLS", size=12, fill=INK3, mono=True, ls=1.5, weight=700)
    uvm = [("UVM_INITIALIZE", "open the UVM session", 210),
           ("UVM_REGISTER_GPU", "this GPU, by UUID", 286),
           ("UVM_REGISTER_GPU_VASPACE", "the address space", 590),
           ("UVM_CREATE_EXTERNAL_RANGE", "per buffer: reserve the range", 666),
           ("UVM_MAP_EXTERNAL_ALLOCATION", "per buffer: map it for the GPU", 742),
           ("UVM_REGISTER_CHANNEL", "64 MiB for the channel's context", 894)]
    for name, sub, y in uvm:
        card(s, ux, y, 250, 64, accent=VIOLET)
        s.text(ux + 14, y + 26, name, size=11.8, weight=700, fill=VIOLET, mono=True)
        s.text(ux + 14, y + 48, sub, size=13, fill=INK2)
    # dotted links from UVM calls to the objects they bind
    links = [(286, 2), (590, 5), (666, 6), (742, 6), (894, 9)]
    for uy, ni in links:
        d, x, y = pos[ni]
        s.line(ux - 4, uy + 32, tx + nw + 6, y + nh / 2, stroke=VIOLET, sw=1.5, dash="4 4", op=0.95)

    by = 210 + len(nodes) * (nh + ng) + 8
    card(s, 48, by, 1104, 70, accent=INK3, fill=CARD2, dash="6 5")
    s.text(68, by + 30, "Below this line: NVIDIA's open kernel module (nvidia.ko, nvidia-uvm.ko) and the GSP "
                        "firmware inside the GPU.", size=15, fill=INK)
    s.text(68, by + 53, "Both stay: GB10 support goes upstream to Nova and nouveau (L332 is a research harness); "
                        "the signed firmware stays NVIDIA's.", size=14, fill=INK2)
    return s


# ---- figure 7: life of one launch ------------------------------------------------------------
def fig_launch():
    s = Svg(1200, 1010, "The life of one kernel launch",
            "Five CPU steps write into memory the CPU and GPU share, one store rings the doorbell, the GPU reads "
            "the ring, the pushbuffer and the launch descriptor, runs the grid and writes a completion marker the "
            "CPU polls.")
    s.frame()
    s.kicker(48, 60, "the launch path")
    s.text(48, 100, "The life of one kernel launch", size=30, weight=800, ls=-0.5)
    s.text(48, 132, "On the GB10 the CPU and the GPU share one memory, and CUDA++ maps every buffer at the same "
                    "address on both. Nothing is copied.", size=15.5, fill=INK2)
    cx, mx, gx = 48, 440, 820
    cw, mw, gw = 350, 300, 332
    s.text(cx, 186, "CPU (your process)", size=13, fill=BLUE_SOFT, mono=True, weight=700, ls=1)
    s.text(mx, 186, "SHARED MEMORY", size=13, fill=GREEN, mono=True, weight=700, ls=1)
    s.text(gx, 186, "GPU", size=13, fill=AMBER, mono=True, weight=700, ls=1)
    s.rect(mx - 14, 168, mw + 28, 740, fill=GREEN, rx=18, op=0.035)

    def step(x, y, w, n, title, sub, col, h=104):
        card(s, x, y, w, h, accent=col)
        s.circle(x + 26, y + 28, 14, fill=col)
        s.text(x + 26, y + 33, str(n), size=14, weight=800, fill=BG, anchor="middle")
        s.text(x + 50, y + 33, title, size=16, weight=700)
        s.lines(x + 18, y + 60, sub, size=13, fill=INK2, gap=18)

    def buf(y, title, sub, col=GREEN, h=68, dash=None):
        card(s, mx, y, mw, h, accent=col, dash=dash)
        s.text(mx + 16, y + 27, title, size=15, weight=700, fill=col)
        s.text(mx + 16, y + 50, sub, size=12.5, fill=INK2, mono=True)

    rows = [206, 326, 446, 566, 686]
    step(cx, rows[0], cw, 1, "Build the kernel", ["IR instructions, register allocation,", "16-byte machine words"], BLUE_SOFT)
    step(cx, rows[1], cw, 2, "Describe the launch", ["QMD: grid, block, registers, shared", "memory, code address. Arguments", "go in constant bank 0."], BLUE_SOFT, h=110)
    step(cx, rows[2], cw, 3, "Write the methods", ["INVALIDATE_SHADER_CACHES,", "WAIT_FOR_IDLE, SEND_PCAS_A (the", "QMD), then the completion tail"], BLUE_SOFT, h=110)
    step(cx, rows[3], cw, 4, "Queue one ring entry", ["8 bytes: pushbuffer address + length;", "then advance USERD.GPPut"], BLUE_SOFT)
    step(cx, rows[4], cw, 5, "Ring the doorbell", ["one 32-bit store of the channel's", "work-submit token"], AMBER)

    buf(rows[0] + 18, "Code buffer", "written once, executed by the GPU")
    buf(rows[1] + 18, "QMD + constant bank 0", "384 B QMD · 224 driver words · args")
    buf(rows[2] + 18, "Pushbuffer", "method headers + data words")
    buf(rows[3] + 18, "GPFIFO ring + USERD", "1,024 entries · GPPut (GPU-uncached)")
    buf(rows[4] + 18, "Usermode page (GPU registers)", "doorbell register", col=AMBER, dash="5 4")
    buf(826, "Marker page (GPU-uncached)", "0x44444444, then 0x46464646")

    for r in range(5):
        s.line(cx + cw + 4, rows[r] + 52, mx - 6, rows[r] + 52, stroke=BLUE_SOFT if r < 4 else AMBER, sw=1.6,
               arrow=True)

    gsteps = [(286, 6, "Host engine fetches", ["reads the new ring entry, then", "the pushbuffer's methods"]),
              (446, 7, "Compute engine launches", ["invalidates caches, reads the QMD", "and constant bank, starts the", "grid on the SMs"]),
              (606, 8, "The grid finishes", ["semaphore release writes the", "marker, L2_FLUSH_DIRTY, then", "the second marker"])]
    for y, n, t, sub in gsteps:
        step(gx, y, gw, n, t, sub, AMBER, h=118)
    # GPU reads: two buses on the right of the memory column, one per GPU step
    bx6, bx7 = mx + mw + 18, mx + mw + 40
    for r in (2, 3):
        s.line(mx + mw, rows[r] + 52, bx6, rows[r] + 52, stroke=INK2, sw=1.3)
    s.line(bx6, rows[3] + 52, bx6, 345, stroke=INK2, sw=1.3)
    s.line(bx6, 345, gx - 6, 345, stroke=INK2, sw=1.3, arrow=True)
    for r in (0, 1):
        s.line(mx + mw, rows[r] + 52, bx7, rows[r] + 52, stroke=INK2, sw=1.3)
    s.line(bx7, rows[0] + 52, bx7, 505, stroke=INK2, sw=1.3)
    s.line(bx7, 505, gx - 6, 505, stroke=INK2, sw=1.3, arrow=True)
    for r in (2, 3):
        s.circle(bx6, rows[r] + 52, 3.5, fill=INK2)
    for r in (0, 1):
        s.circle(bx7, rows[r] + 52, 3.5, fill=INK2)
    s.text(gx + 4, 278, "reads ring + pushbuffer", size=11.5, fill=INK3, mono=True)
    s.text(gx + 4, 438, "reads QMD, bank, code", size=11.5, fill=INK3, mono=True)
    # the doorbell wakes the host engine
    dy = rows[4] + 52
    s.path(f"M{mx + mw + 4} {dy} L802 {dy} L802 395 L{gx - 6} 395", stroke=BG, sw=7)
    s.path(f"M{mx + mw + 4} {dy} L802 {dy} L802 395 L{gx - 6} 395", stroke=AMBER, sw=1.8, arrow=True)
    s.add(f'<text transform="translate(796 {dy - 30}) rotate(-90)" font-family="{MONO}" font-size="12" '
          f'fill="{AMBER}" text-anchor="start">wakes the GPU</text>')
    s.path(f"M{gx + gw / 2} {606 + 118 + 4} L{gx + gw / 2} 860 L{mx + mw + 6} 860", stroke=AMBER, sw=1.8,
           arrow=True)
    s.text(gx + gw / 2 - 8, 848, "writes the markers", size=12, fill=AMBER, mono=True, anchor="end")
    step(cx, 806, cw, 9, "Wait, then retire", ["poll the marker with a hard deadline", "and a stall deadline; results are", "already in shared memory"], BLUE_SOFT, h=110)
    s.line(mx - 4, 860, cx + cw + 6, 860, stroke=BLUE_SOFT, sw=1.6, arrow=True)
    s.text(48, 958, "Blue: the CPU writes. Amber: the GPU acts. Dashed: GPU registers mapped into the process, "
                    "not memory.", size=13, fill=INK2)
    s.text(48, 980, "Measured on the GB10: 24 to 31 µs per extra dependent launch with wait-for-idle "
                    "(evidence/OWN30B-LAUNCH-COST-20261014T0120Z). Command lists put many launches in one submission.",
           size=13, fill=INK3)
    return s


# ---- figure 8: replacement map -------------------------------------------------------------
STATUS_COL = {"PASS": GREEN, "PARTIAL": BLUE_SOFT, "BUILDING": VIOLET, "MISSING": INK2, "STAYS": AMBER}


def fig_status():
    rows = [  # piece, job, replacement, status, note
        ("nvcc · NVVM · PTX · ptxas", "compile kernels", "sm_121 encoder + register allocator", "PASS", "every kernel this stack runs"),
        ("libcuda", "user-space driver", "nvrm + m16 driver client", "PARTIAL", "M16 raw logs unavailable"),
        ("libcudart", "device, memory, launch", "session, command lists, job engine", "PARTIAL", "one device, one process"),
        ("cuBLAS", "matrix multiply", "matmul, tensor-core GEMM, FP8, BF16", "PARTIAL", "model shapes; any-shape GEMM is C3"),
        ("cuDNN · FlashAttention", "attention, layers", "decode and prefill attention", "PARTIAL", "model-shaped"),
        ("CUB · Thrust", "reductions, scans, sorts", "frozen-order reductions", "PARTIAL", "E1-CLOSURE; scans and sorts are C3"),
        ("cuobjdump · nvdisasm", "read machine code back", "my own sm_121 disassembler", "BUILDING", "Omega branch c6/disasm: 331 kernels agree"),
        ("Nsight", "measure", "ceiling probes", "PARTIAL", "115.14 TFLOPS, 195.1 GB/s"),
        ("cuFFT · cuRAND", "FFT, random numbers", "none yet", "MISSING", ""),
        ("nvidia.ko", "kernel driver", "stays: NVIDIA's open kernel module", "STAYS", "GB10 support goes upstream: Nova, nouveau"),
        ("nvidia-uvm.ko", "page tables, faults", "stays: NVIDIA's open kernel module", "STAYS", "L332 is a research harness only"),
        ("nvidia-smi · NVML", "status, clocks, power", "not started", "MISSING", ""),
        ("GSP firmware", "runs inside the GPU", "stays NVIDIA's: signature-checked", "STAYS", "never shipped by CUDA++"),
    ]
    rh, top = 52, 200
    s = Svg(1200, top + len(rows) * rh + 80, "Every NVIDIA piece on the Spark and what replaces it",
            "One row per NVIDIA component with the CUDA++ replacement and its status (ROADMAP.md).")
    s.frame()
    s.kicker(48, 60, "the replacement map")
    s.text(48, 100, "Every NVIDIA piece on the Spark, and what replaces it", size=30, weight=800, ls=-0.5)
    s.text(48, 132, "PASS means a receipt on a named commit. The full table with every receipt is ROADMAP.md.",
           size=15.5, fill=INK2)
    hx = [48, 330, 560, 900, 1040]
    for lab, x in zip(("NVIDIA piece", "its job", "CUDA++", "status", ""), hx):
        s.text(x, top - 16, lab.upper(), size=12.5, fill=INK2, mono=True, ls=1, weight=700)
    for i, (p, job, rep, st, note) in enumerate(rows):
        y = top + i * rh
        if i % 2 == 0:
            s.rect(36, y, 1128, rh, fill=CARD, rx=8)
        s.text(48, y + 22, p, size=15, weight=700, mono=True, fill=INK if st != "STAYS" else AMBER)
        s.text(330, y + 22, job, size=14, fill=INK2)
        s.text(560, y + 22, rep, size=14.5, fill=INK)
        if note:
            s.text(560, y + 41, note, size=12.5, fill=INK2, op=0.85)
        col = STATUS_COL[st]
        s.pill(900, y + 12, st, col, size=12, h=24, filled=True, w=108)
    tot = {k: sum(1 for r in rows if r[3] == k) for k in STATUS_COL}
    x = 48
    yb = top + len(rows) * rh + 40
    for k, col in STATUS_COL.items():
        s.rect(x, yb - 12, 14, 14, fill=col, rx=3, op=0.85)
        lab = f"{k}  {tot[k]}"
        s.text(x + 22, yb, lab, size=14, fill=INK2, mono=True)
        x += 40 + len(lab) * 9
    return s


# ---- figure 9: ceilings -------------------------------------------------------------------------
def fig_ceiling():
    import math
    peak, bw = 115.14e12, 195.1e9  # evidence/OWN30B-T3-TC-PEAK-BW-5faa0f70-20261013T2236Z
    s = Svg(1200, 660, "The two ceilings I measured on the GB10",
            "Roofline: attainable speed against operations per byte, from the measured 115.14 TFLOPS tensor-core "
            "peak and 195.1 GB/s read bandwidth.")
    s.frame()
    s.kicker(48, 60, "speed against the machine's ceiling")
    s.text(48, 100, "The two ceilings I measured on the GB10", size=30, weight=800, ls=-0.5)
    s.text(48, 132, "Every speed claim is held against these: the tensor-core peak and the memory read rate my "
                    "own probes reached.", size=15.5, fill=INK2)
    px, py, pw, ph = 130, 170, 700, 380
    xmin, xmax, ymin, ymax = -2, 4, 9, 15  # log10 FLOP/byte, log10 FLOP/s
    X = lambda v: px + (math.log10(v) - xmin) / (xmax - xmin) * pw
    Y = lambda v: py + ph - (math.log10(v) - ymin) / (ymax - ymin) * ph
    s.rect(px, py, pw, ph, fill=CARD, rx=10)
    for e in range(xmin, xmax + 1):
        s.line(X(10 ** e), py, X(10 ** e), py + ph, stroke=LINE, sw=1)
        s.text(X(10 ** e), py + ph + 22, f"{10 ** e:g}", size=12.5, fill=INK3, anchor="middle", mono=True)
    for e in range(ymin, ymax + 1):
        s.line(px, Y(10 ** e), px + pw, Y(10 ** e), stroke=LINE, sw=1)
        lab = {9: "1 G", 10: "10 G", 11: "100 G", 12: "1 T", 13: "10 T", 14: "100 T", 15: "1 P"}[e]
        s.text(px - 12, Y(10 ** e) + 4, lab, size=12.5, fill=INK3, anchor="end", mono=True)
    s.text(px + pw / 2, py + ph + 48, "operations per byte moved (log)", size=13.5, fill=INK2, anchor="middle")
    s.add(f'<text transform="translate({px - 70} {py + ph / 2}) rotate(-90)" font-family="{SANS}" font-size="13.5" '
          f'fill="{INK2}" text-anchor="middle">operations per second (log)</text>')
    ridge = peak / bw
    pts = [(10 ** xmin, bw * 10 ** xmin), (ridge, peak), (10 ** xmax, peak)]
    d = " ".join(("M" if i == 0 else "L") + f"{X(a):.1f} {Y(b):.1f}" for i, (a, b) in enumerate(pts))
    s.path(d, stroke=GREEN, sw=3)
    s.line(X(ridge), Y(peak), X(ridge), py + ph, stroke=GREEN, sw=1, dash="4 5", op=0.6)
    s.text(X(ridge) + 8, py + ph - 10, f"ridge: {ridge:.0f} ops/byte", size=12.5, fill=GREEN, mono=True)
    s.text(X(ridge) - 14, Y(peak) + 5, "tensor-core peak 115.14 TFLOPS", size=14, fill=GREEN, mono=True, weight=700,
           anchor="end")
    ang = math.degrees(math.atan((ph / (ymax - ymin)) / (pw / (xmax - xmin))))
    s.add(f'<text transform="translate({X(0.012):.1f} {Y(bw * 0.012) - 12:.1f}) rotate(-{ang:.2f})" '
          f'font-family="{MONO}" font-size="14" font-weight="700" fill="{GREEN}">memory reads 195.1 GB/s</text>')
    marks = [(1 / 12, "vector add", "1 add per 12 bytes moved", CYAN),
             (1.0, "decode, BF16 matrix-vector", "multiply + add per 2-byte weight", AMBER),
             (4096 / 3, "large BF16 GEMM (4096³)", "each byte reused many times", ROSE)]
    for v, lab, sub, col in marks:
        yv = min(bw * v, peak)
        s.circle(X(v), Y(yv), 6, fill=col, stroke=BG, sw=2)
    # legend panel
    lx = 880
    for i, (v, lab, sub, col) in enumerate(marks):
        y = 196 + i * 92
        s.circle(lx + 8, y - 5, 6, fill=col)
        s.text(lx + 24, y, lab, size=15, weight=700, fill=col)
        s.text(lx + 24, y + 21, sub, size=13, fill=INK2)
        bound = "memory-bound" if v < ridge else "compute-bound"
        num = f"{v:,.0f}" if v >= 10 else (f"{v:.0f}" if v == 1 else f"{v:.3f}")
        s.text(lx + 24, y + 40, f"{num} ops/byte: {bound}", size=12.5, fill=INK2, mono=True)
    s.lines(lx, 488, ["A generated word reads every active", "weight once, so its speed is set by", "the memory line, not the tensor cores."],
            size=13.5, fill=INK2, gap=19)
    s.text(48, s.h - 40, "The two lines are measured: evidence/OWN30B-T3-TC-PEAK-BW-5faa0f70-20261013T2236Z "
                         "(peak.log, bw.log). The dots are arithmetic, not measurements.", size=13, fill=INK2)
    s.text(48, s.h - 20, "One op is one add or one multiply; a fused multiply-add counts as two.", size=13, fill=INK3)
    return s


# ---- figure 10: timeline ------------------------------------------------------------------------
def fig_timeline():
    ev = [("2026-09-26", "First native kernel", ["vector add qualified on the", "GB10; first tensor-core matmul"]),
          ("2026-10-02", "Numerics closed on chip", ["reductions, 9 functions over", "all 2^32 inputs, 148 ld/st kernels"]),
          ("2026-10-13", "Ceilings measured", ["115.14 TFLOPS tensor peak,", "195.1 GB/s reads, own probes"]),
          ("2026-10-21", "A 35B model, no CUDA", ["served on this code with zero", "CUDA libraries in the process"])]
    s = Svg(1200, 270, "Under four weeks, one DGX Spark",
            "Milestones from the first native kernel to a 35B model served with no CUDA library loaded.")
    s.frame()
    s.kicker(48, 56, "under four weeks, one DGX Spark")
    y = 120
    s.path(f"M80 {y} L950 {y}", stroke=GREEN, sw=2.5, op=0.6)
    for i, (d, t, sub) in enumerate(ev):
        x = 80 + i * 290
        s.circle(x, y, 9, fill=BG, stroke=GREEN, sw=3)
        anchor = "start"
        s.text(x, y - 22, d, size=13.5, fill=GREEN, mono=True, anchor=anchor, weight=700)
        s.text(x, y + 40, t, size=17, weight=700, anchor=anchor)
        for j, l in enumerate(sub):
            s.text(x, y + 64 + j * 20, l, size=14, fill=INK2, anchor=anchor)
    return s


FIGS = {
    "hero.svg": lambda w: fig_hero(w),
    "stack.svg": lambda w: fig_stack(),
    "instruction.svg": lambda w: fig_instruction(w),
    "vecadd-scoreboards.svg": lambda w: fig_listing(w),
    "reverse-engineering.svg": lambda w: fig_reverse(),
    "driver-objects.svg": lambda w: fig_driver(),
    "launch-path.svg": lambda w: fig_launch(),
    "replacement-map.svg": lambda w: fig_status(),
    "ceilings.svg": lambda w: fig_ceiling(),
    "timeline.svg": lambda w: fig_timeline(),
}


def main():
    check = "--check" in sys.argv
    words = vecadd_words()
    os.makedirs(OUT, exist_ok=True)
    stale = []
    for name, fn in FIGS.items():
        data = fn(words).svg()
        path = os.path.join(OUT, name)
        if check:
            if not os.path.exists(path) or open(path).read() != data:
                stale.append(name)
        else:
            with open(path, "w") as f:
                f.write(data)
            print(f"wrote docs/img/{name} ({len(data)} bytes)")
    if check:
        if stale:
            sys.exit("stale figures (run python3 tools/figures/make_figures.py): " + ", ".join(stale))
        print(f"figures current: {len(FIGS)}")


if __name__ == "__main__":
    main()
