#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Verification-only oracle for fp8.om, nvfp4.om and mxdot.om; never runtime.

References, each independent of the candidate's bit operations:
  spec   : OCP OFP8 rev 1.0 (2023-12-01) section 5.1 value formulas, Table 2
           rows and section 5.2.1 / Table 3 conversion rules, evaluated with
           exact rationals (fractions.Fraction).
  ml     : ml_dtypes (float8_e4m3fn, float8_e5m2, float4_e2m1fn), the
           published OCP-format implementation used by JAX.
  exact  : Fraction sums for the block dot products; binary64 sequential sums.
Needs numpy and ml_dtypes; the versions are printed with the receipt.
"""
import argparse
import hashlib
import math
from fractions import Fraction
from pathlib import Path
import struct
import subprocess
import tempfile

import ml_dtypes
import numpy as np

PIN = '2c82480596bbdc50cda9626866209ef662d5834f278f839ff4ce0250c495029f'
ROOT = Path(__file__).resolve().parents[2]
UNITS = ROOT / 'native/models'
TESTS = ROOT / 'tests/models'
QNAN = 0x7FC00000
F4 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)   # OCP MX v1.0 Table 5 (E2M1)
U = Fraction(1, 2 ** 53)


class Fail(Exception):
    pass


def need(cond, what):
    if not cond:
        raise Fail(what)


# ---------------------------------------------------------------- build/run
def build(compiler, parts, target):
    src = b''.join(parts)
    r = subprocess.run([compiler], input=src, capture_output=True, timeout=120)
    if r.returncode or r.stderr:
        raise Fail(f'compile rc={r.returncode} {r.stderr.decode()!r}')
    target.write_bytes(r.stdout)
    target.chmod(0o700)
    return str(target)


def run(exe, data, timeout=600):
    r = subprocess.run([exe], input=data, capture_output=True, timeout=timeout)
    return r.returncode, r.stdout


def fmtcall(exe, ops, aux, vals):
    rec = np.zeros((len(vals), 6), np.uint8)
    rec[:, 0] = ops
    rec[:, 1] = aux
    rec[:, 2:] = np.asarray(vals, '<u4').view(np.uint8).reshape(-1, 4)
    rc, out = run(exe, rec.tobytes())
    need(rc == 0 and len(out) == 4 * len(vals), f'fmt driver rc={rc} bytes={len(out)}')
    return np.frombuffer(out, '<u4')


# ---------------------------------------------------------------- spec oracle
SPEC = {  # name: (exponent bits, mantissa bits, bias)
    'e4m3': (4, 3, 7),
    'e5m2': (5, 2, 15),
}


def spec_value(fmt, code):
    """OFP8 5.1: value of one code as Fraction, or 'nan' / ('inf', sign)."""
    eb, mb, bias = SPEC[fmt]
    s = -1 if code & 128 else 1
    e = (code >> mb) & ((1 << eb) - 1)
    m = code & ((1 << mb) - 1)
    if fmt == 'e4m3' and e == 15 and m == 7:
        return 'nan'
    if fmt == 'e5m2' and e == 31:
        return ('inf', s) if m == 0 else 'nan'
    if e == 0:
        return s * Fraction(m, 1 << mb) * Fraction(2) ** (1 - bias)
    return s * (1 + Fraction(m, 1 << mb)) * Fraction(2) ** (e - bias)


TABLE2 = {  # OFP8 Table 2 rows (magnitudes), checked against spec_value
    'e4m3': {0x7E: Fraction(448), 0x08: Fraction(1, 2 ** 6),
             0x07: Fraction(7, 8) / 2 ** 6, 0x01: Fraction(1, 2 ** 9)},
    'e5m2': {0x7B: Fraction(57344), 0x04: Fraction(1, 2 ** 14),
             0x03: Fraction(3, 4) / 2 ** 14, 0x01: Fraction(1, 2 ** 16)},
}
MAXCODE = {'e4m3': 0x7E, 'e5m2': 0x7B}
OVF = {'e4m3': 0x7F, 'e5m2': 0x7C}       # non-saturating overflow code
NANC = {'e4m3': 0x7F, 'e5m2': 0x7E}      # NaN source code (sign kept)


def f32bits(x):
    return struct.unpack('<I', struct.pack('<f', x))[0]


def bitsf32(b):
    return struct.unpack('<f', struct.pack('<I', int(b)))[0]


def frac_to_f32bits(v):
    f = float(v)
    b = f32bits(f)
    need(Fraction(bitsf32(b)) == v, f'value {v} not exact in binary32')
    if v == 0 and f32bits(f) == 0 and isinstance(v, Fraction):
        return 0
    return b


def expected_decode(fmt, code, table):
    v = table[code]
    s = (code & 128) << 24
    if v == 'nan':
        return s | QNAN
    if isinstance(v, tuple):
        return s | 0x7F800000
    if v == 0:
        return s
    return frac_to_f32bits(v)


def spec_encoder(fmt):
    """Returns enc(xbits, sat) per 5.2.1/Table 3 using exact rationals."""
    eb, mb, bias = SPEC[fmt]
    vals = []
    for c in range(MAXCODE[fmt] + 1):
        vals.append((spec_value(fmt, c), c))
    top = vals[-1][0]
    ulp = Fraction(2) ** ((MAXCODE[fmt] >> mb) - bias - mb)
    nxt = top + ulp                            # next value, unbounded exponent
    nxt_even = ((MAXCODE[fmt] + 1) & 1) == 0   # its mantissa parity
    if (MAXCODE[fmt] & ((1 << mb) - 1)) == (1 << mb) - 1:
        nxt_even = True                        # carry into a new binade: M = 0
    mags = [v for v, _ in vals]

    def enc(xb, sat):
        s = (xb >> 24) & 128
        a = xb & 0x7FFFFFFF
        if a > 0x7F800000:
            return s | NANC[fmt]
        if a == 0x7F800000:
            return s | (MAXCODE[fmt] if sat else OVF[fmt])
        x = Fraction(bitsf32(a))
        if x > nxt or (x == nxt) or (x > top and (x - top > nxt - x or
                                                  (x - top == nxt - x and nxt_even))):
            return s | (MAXCODE[fmt] if sat else OVF[fmt])
        lo, hi = 0, len(mags) - 1
        while lo < hi:                         # first mag >= x
            mid = (lo + hi) // 2
            if mags[mid] >= x:
                hi = mid
            else:
                lo = mid + 1
        c = lo
        if mags[c] == x:
            return s | c
        below, above = c - 1, c
        db, da = x - mags[below], mags[above] - x
        if db < da or (db == da and below % 2 == 0):
            return s | below
        return s | above
    return enc


# ---------------------------------------------------------------- ml oracle
MLT = {'e4m3': ml_dtypes.float8_e4m3fn, 'e5m2': ml_dtypes.float8_e5m2}


def ml_decode_table(fmt):
    vals = np.arange(256, dtype=np.uint8).view(MLT[fmt]).astype(np.float64)
    out = {}
    for c, v in enumerate(vals):
        if math.isnan(v):
            out[c] = 'nan'
        elif math.isinf(v):
            out[c] = ('inf', -1 if v < 0 else 1)
        else:
            out[c] = Fraction(v) if not (v == 0 and math.copysign(1, v) < 0) else -Fraction(0)
    return out


def ml_encode(fmt, xbits, sat):
    x = np.asarray(xbits, np.uint32).view(np.float32)
    with np.errstate(invalid='ignore'):        # NaN inputs cast to NaN codes
        c = x.astype(MLT[fmt]).view(np.uint8).astype(np.uint32)
    if sat:
        a = np.asarray(xbits, np.uint32) & 0x7FFFFFFF
        notnan = a <= 0x7F800000
        mag = c & 127
        over = notnan & ((mag == OVF[fmt]) | ((fmt == 'e4m3') & (mag == 0x7F)))
        c = np.where(over, (c & 128) | MAXCODE[fmt], c)
    return c


def sweep_bits():
    """Stated sweep: every sign, every binary32 exponent field, every pattern
    of the top 12 mantissa bits, low 11 bits in {0,1,0x3FF,0x400,0x401,0x7FF}."""
    s = np.arange(2, dtype=np.uint32) << 31
    e = np.arange(256, dtype=np.uint32) << 23
    t = np.arange(4096, dtype=np.uint32) << 11
    lo = np.array([0, 1, 0x3FF, 0x400, 0x401, 0x7FF], np.uint32)
    return (s[:, None, None, None] | e[None, :, None, None] |
            t[None, None, :, None] | lo[None, None, None, :]).ravel()


def bf16_bits():
    return (np.arange(65536, dtype=np.uint32) << 16)


# ---------------------------------------------------------------- FP8 checks
def check_fp8(exe, flip=False):
    n = 0
    for op, fmt in ((0, 'e4m3'), (1, 'e5m2')):
        spec = {c: spec_value(fmt, c) for c in range(256)}
        ml = ml_decode_table(fmt)
        for c in range(256):                    # the two references agree
            a, b = spec[c], ml[c]
            need(a == b if not isinstance(a, Fraction) else
                 (isinstance(b, Fraction) and a == b), f'{fmt} ref disagree {c}')
        for c, mag in TABLE2[fmt].items():
            need(spec[c] == mag and spec[c | 128] == -mag, f'{fmt} Table 2 row {c:#x}')
        if flip and fmt == 'e4m3':
            ml[0x38] = Fraction(9, 8)           # negative control: 1.0 -> 1.125
        exp = [expected_decode(fmt, c, ml) for c in range(256)]
        got = fmtcall(exe, op, 0, range(256))
        bad = [c for c in range(256) if int(got[c]) != exp[c]]
        need(not bad, f'{fmt} decode mismatch codes {bad[:8]}')
        n += 256
    return n


def check_fp8_encode(exe, mutated=False):
    counts = {}
    sweep = sweep_bits()
    bf = bf16_bits()
    for fmt, op in (('e4m3', 2), ('e5m2', 3)):
        enc = spec_encoder(fmt)
        for sat in (0, 1):
            # exact-rational spec oracle over every bfloat16 input
            exp = np.array([enc(int(b), sat) for b in bf], np.uint32)
            got = fmtcall(exe, op, sat, bf)
            bad = np.nonzero(got != exp)[0]
            need(len(bad) == 0, f'{fmt} sat={sat} spec mismatch at {[hex(int(bf[i])) for i in bad[:4]]}')
            mlx = ml_encode(fmt, bf, sat)
            need(np.array_equal(mlx, exp), f'{fmt} sat={sat} spec and ml_dtypes disagree')
            # ml_dtypes over the structured binary32 sweep
            got = fmtcall(exe, op, sat, sweep)
            exp2 = ml_encode(fmt, sweep, sat)
            bad = np.nonzero(got != exp2)[0]
            need(len(bad) == 0, f'{fmt} sat={sat} sweep mismatch at {[hex(int(sweep[i])) for i in bad[:4]]}')
            counts[(fmt, sat)] = len(bf) + len(sweep)
            # round trip: every finite code decodes and re-encodes to itself
            dec = fmtcall(exe, op - 2, 0, range(256))
            back = fmtcall(exe, op, sat, dec)
            for c in range(256):
                v = spec_value(fmt, c)
                if v == 'nan':
                    want = (c & 128) | NANC[fmt]
                elif isinstance(v, tuple):
                    want = (c & 128) | (MAXCODE[fmt] if sat else OVF[fmt])
                else:
                    want = c
                need(int(back[c]) == want, f'{fmt} sat={sat} round trip {c:#x} -> {int(back[c]):#x}')
    return counts


# ---------------------------------------------------------------- NVFP4
def e2m1(n):
    v = Fraction(F4[n & 7]).limit_denominator(2)
    return -v if n & 8 else v


def ue4m3(c):
    return spec_value('e4m3', c)


def check_nvfp4(exe):
    # decode: every element x valid scale pair, against ml_dtypes product
    nib = np.repeat(np.arange(16, dtype=np.uint32), 127)
    sc = np.tile(np.arange(127, dtype=np.uint32), 16)
    got = fmtcall(exe, 4, sc, nib)
    f4 = nib.astype(np.uint8).view(ml_dtypes.float4_e2m1fn).astype(np.float64)
    f8 = sc.astype(np.uint8).view(ml_dtypes.float8_e4m3fn).astype(np.float64)
    prod = (f4 * f8).astype(np.float32)        # exact: 6 significant bits
    need(np.all((f4 * f8) == prod.astype(np.float64)), 'nvfp4 product not exact')
    exp = prod.view(np.uint32)
    for i in range(len(nib)):                  # exact-rational cross-check
        need(Fraction(float(prod[i])) == e2m1(int(nib[i])) * ue4m3(int(sc[i])), 'nvfp4 ref')
    bad = np.nonzero(got != exp)[0]
    need(len(bad) == 0, f'nvfp4 decode mismatch at {[(int(nib[i]), int(sc[i])) for i in bad[:4]]}')
    dec_pairs = len(nib)
    # encode: every bfloat16 non-NaN input x every valid scale
    bf = bf16_bits()
    bf = bf[(bf & 0x7FFFFFFF) <= 0x7F800000]
    total = 0
    mids = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
    for s in range(127):
        got = fmtcall(exe, 5, s, bf)
        x = bf.view(np.float32).astype(np.float64)
        neg = (bf >> 31).astype(np.uint32)
        sv = float(ue4m3(s))
        if sv == 0:
            exp = neg << 3
        else:
            q = np.abs(x) / sv                 # no false tie: see report
            c = np.searchsorted(mids, q, side='left').astype(np.uint32)
            tie = np.isin(q, mids)
            c = np.where(tie & (c % 2 == 1), c + 1, c)  # tie: even neighbour
            c = np.minimum(c, 7)
            exp = (neg << 3) | c
        bad = np.nonzero(got != exp)[0]
        need(len(bad) == 0, f'nvfp4 enc scale {s} mismatch at {[hex(int(bf[i])) for i in bad[:4]]}')
        total += len(bf)
    # spec cross-check of the vector oracle near every threshold (Fraction)
    for s in range(1, 127):
        sv = ue4m3(s)
        pts = []
        for m in (Fraction(1, 4), Fraction(3, 4), Fraction(5, 4), Fraction(7, 4),
                  Fraction(5, 2), Fraction(7, 2), Fraction(5), Fraction(6), Fraction(7)):
            b = f32bits(float(m * sv))
            pts += [b - 1, b, b + 1]
        got = fmtcall(exe, 5, s, pts)
        for b, g in zip(pts, got):
            x = Fraction(bitsf32(b)) / sv
            best = min(range(8), key=lambda k: (abs(x - Fraction(F4[k])), k % 2))
            need(int(g) == best, f'nvfp4 enc Fraction check s={s} b={b:#x}')
    # round trip for every element x nonzero scale
    rt = 0
    for s in range(1, 127):
        dec = fmtcall(exe, 4, s, range(16))
        back = fmtcall(exe, 5, s, dec)
        need(list(map(int, back)) == list(range(16)), f'nvfp4 round trip scale {s}')
        rt += 16
    # block scale: amax over nonneg bfloat16 and a random binary32 set
    rng = np.random.default_rng(20261026)
    am = np.concatenate([bf16_bits()[:0x7F81],
                         rng.integers(0, 0x7F800001, 200000, dtype=np.uint32)]).astype(np.uint32)
    got = fmtcall(exe, 6, 0, am)
    q = (am.view(np.float32) / np.float32(6.0)).astype(np.float32)
    exp = ml_encode('e4m3', q.view(np.uint32), 1)
    need(np.array_equal(got, exp), 'nvfp4scale mismatch')
    return dec_pairs, total, rt, len(am)


def check_nvblk(exe, fmtexe):
    rng = np.random.default_rng(7)
    n = 20000
    mag = rng.uniform(-30, 10, (n, 16))
    x = (rng.choice([-1, 1], (n, 16)) * np.exp2(mag) * rng.uniform(1, 2, (n, 16))).astype(np.float32)
    x[::97] = 0
    x[1::101, 3] = np.inf
    bits = x.view(np.uint32)
    rc, out = run(exe, bits.astype('<u4').tobytes())
    need(rc == 0 and len(out) == 9 * n, f'nvblk rc={rc}')
    out = np.frombuffer(out, np.uint8).reshape(n, 9)
    amax = (bits & 0x7FFFFFFF).max(axis=1).astype(np.uint32)
    sc = fmtcall(fmtexe, 6, 0, amax)
    need(np.array_equal(out[:, 0], sc.astype(np.uint8)), 'nvblk scale')
    for r in range(0, n, 37):                  # element codes via op 5
        e = fmtcall(fmtexe, 5, int(sc[r]), bits[r])
        packed = (e[0::2] | (e[1::2] << 4)).astype(np.uint8)
        need(np.array_equal(packed, out[r, 1:]), f'nvblk packing row {r}')
    return n


# ---------------------------------------------------------------- mxdot
BS = {0: 16, 1: 32, 2: 32}


def elem_frac(fmt, A, i):
    if fmt == 2:
        return spec_value('e4m3', int(A[i]))
    return e2m1((int(A[i >> 1]) >> (4 * (i & 1))) & 15)


def scale_frac(fmt, S, j):
    if fmt == 0:
        return ue4m3(int(S[j]))
    return Fraction(2) ** (int(S[j]) - 127)


def mx_reference(fmt, m, n, k, A, SA, B, SB):
    kb = k // BS[fmt]
    C, Cseq, Babs = [], [], []
    for i in range(m):
        for j in range(n):
            terms = []
            for t in range(kb):
                ia, ib = i * kb + t, j * kb + t
                inner = sum(elem_frac(fmt, A, ia * BS[fmt] + e) * elem_frac(fmt, B, ib * BS[fmt] + e)
                            for e in range(BS[fmt]))
                terms.append(inner * scale_frac(fmt, SA, ia) * scale_frac(fmt, SB, ib))
            acc = 0.0
            for tm in terms:
                f = float(tm)
                need(Fraction(f) == tm, 'block term not exact in binary64')
                acc = acc + f
            C.append(sum(terms, Fraction(0)))
            Cseq.append(acc)
            Babs.append(sum((abs(tm) for tm in terms), Fraction(0)))
    return C, Cseq, Babs, kb


def gamma(j):
    return j * U / (1 - j * U)


def mx_case(rng, fmt, m, n, k, wide):
    A = np.zeros(4096, np.uint8)
    B = np.zeros(4096, np.uint8)
    SA = np.zeros(256, np.uint8)
    SB = np.zeros(256, np.uint8)
    kb = k // BS[fmt]
    eb = k // 2 if fmt < 2 else k
    if fmt == 2:
        ok = np.array([c for c in range(256) if (c & 127) != 127], np.uint8)
        A[:eb * m] = rng.choice(ok, eb * m)
        B[:eb * n] = rng.choice(ok, eb * n)
    else:
        A[:eb * m] = rng.integers(0, 256, eb * m)
        B[:eb * n] = rng.integers(0, 256, eb * n)
    if fmt == 0:
        SA[:kb * m] = rng.integers(0, 127, kb * m)
        SB[:kb * n] = rng.integers(0, 127, kb * n)
    else:
        lo, hi = (0, 255) if wide else (117, 137)
        SA[:kb * m] = rng.integers(lo, hi, kb * m)
        SB[:kb * n] = rng.integers(lo, hi, kb * n)
    hdr = bytes([fmt, m, n, 0, k & 255, k >> 8, 0, 0])
    return hdr + A.tobytes() + SA.tobytes() + B.tobytes() + SB.tobytes(), (fmt, m, n, k, A, SA, B, SB)


def check_mxdot(exe, ncases=600):
    rng = np.random.default_rng(1026)
    shapes = []
    for fmt in (0, 1, 2):
        bs = BS[fmt]
        kmax = {0: 4096, 1: 8192, 2: 4096}[fmt]   # scale and element capacity
        shapes += [(fmt, 1, 1, bs), (fmt, 1, 1, kmax),
                   (fmt, 4, 4, 8 * bs), (fmt, 16, 16, bs), (fmt, 2, 3, 4 * bs)]
    cases = []
    while len(cases) < ncases:
        shp = shapes[len(cases) % len(shapes)]
        fmt, m, n, k = shp
        cases.append(mx_case(rng, fmt, m, n, k, wide=(len(cases) // len(shapes)) % 2 == 0))
    data = b''.join(c[0] for c in cases)
    rc, out = run(exe, data, timeout=900)
    need(rc == 0, f'mxdot rc={rc}')
    pos, worst, dots = 0, Fraction(0), 0
    bitexact = 0
    for _, (fmt, m, n, k, A, SA, B, SB) in cases:
        C, Cseq, Babs, kb = mx_reference(fmt, m, n, k, A, SA, B, SB)
        got = np.frombuffer(out[pos:pos + 8 * m * n], '<f8')
        pos += 8 * m * n
        for g, ex, sq, ba in zip(got, C, Cseq, Babs):
            gf = float(g)
            need(not math.isnan(gf), 'mxdot NaN')
            err = abs(Fraction(gf) - ex)
            bound = gamma(kb - 1) * ba
            need(err <= bound, f'mxdot bound fmt={fmt} k={k} err={float(err)} bound={float(bound)}')
            need(gf == sq, f'mxdot differs from left-to-right binary64 sum fmt={fmt}')
            bitexact += 1
            if ba:
                worst = max(worst, err / ba)
            dots += 1
    need(pos == len(out), 'mxdot trailing output')
    return len(cases), dots, bitexact, worst


def mx_refusals(exe):
    rng = np.random.default_rng(3)
    good, _ = mx_case(rng, 0, 1, 1, 16, False)
    ok = 0
    bad = []
    hdr = bytearray(good[:8])
    hdr[0] = 3
    bad.append(bytes(hdr) + good[8:])                      # unknown fmt
    hdr = bytearray(good[:8])
    hdr[4] = 17
    bad.append(bytes(hdr) + good[8:])                      # k not a block multiple
    c = bytearray(good)
    c[8 + 4096] = 127
    bad.append(bytes(c))                                   # UE4M3 NaN scale
    g1, _ = mx_case(rng, 1, 1, 1, 32, False)
    c = bytearray(g1)
    c[8 + 4096] = 255
    bad.append(bytes(c))                                   # E8M0 NaN scale
    g2, _ = mx_case(rng, 2, 1, 1, 32, False)
    c = bytearray(g2)
    c[8] = 127
    bad.append(bytes(c))                                   # E4M3 NaN element
    hdr = bytearray(good[:8])
    hdr[1], hdr[2] = 17, 16
    bad.append(bytes(hdr) + good[8:])                      # m n > 256
    for b in bad:
        rc, out = run(exe, b)
        need(rc == 105 and not out, f'mxdot refusal rc={rc}')
        ok += 1
    rc, out = run(exe, good[:100])
    need(rc == 3 and not out, f'mxdot truncated rc={rc}')
    return ok + 1


def fmt_refusals(exe):
    n = 0
    for op, aux, v in [(0, 0, 256), (1, 0, 256), (2, 2, 0), (3, 2, 0), (4, 0, 16),
                       (5, 0, 0x7FC00000), (6, 0, 0x80000000), (6, 0, 0x7FC00000)]:
        rec = bytes([op, aux]) + struct.pack('<I', v)
        rc, out = run(exe, rec)
        need(rc == 105 and not out, f'refusal op={op} rc={rc}')
        n += 1
    for s in range(127, 256):                  # every invalid UE4M3 scale code
        for op in (4, 5):
            rc, out = run(exe, bytes([op, s]) + struct.pack('<I', 1))
            need(rc == 105 and not out, f'nvfp4 scale {s} op {op} accepted rc={rc}')
            n += 1
    rc, out = run(exe, bytes([9, 0, 0, 0, 0, 0]))
    need(rc == 6, f'unknown op rc={rc}')
    rc, out = run(exe, bytes([0, 0, 0]))
    need(rc == 3, f'truncated rc={rc}')
    return n + 2


# ---------------------------------------------------------------- main
def expect_fail(label, fn):
    try:
        fn()
    except Fail as e:
        return f'{label}: rejected ({e})'
    raise Fail(f'negative control accepted: {label}')


def mutate(src, old, new):
    out = src.replace(old, new, 1)
    need(out != src, f'mutant not applied: {old!r}')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('compiler', type=Path)
    args = ap.parse_args()
    compiler = str(args.compiler.resolve())
    digest = hashlib.sha256(Path(compiler).read_bytes()).hexdigest()
    need(digest == PIN, f'compiler pin mismatch: {digest}')
    fp8 = (UNITS / 'fp8.om').read_bytes()
    nv = (UNITS / 'nvfp4.om').read_bytes()
    mx = (UNITS / 'mxdot.om').read_bytes()
    fdrv = (TESTS / 'fmt-driver.om').read_bytes()
    bdrv = (TESTS / 'nvblk-driver.om').read_bytes()
    mdrv = (TESTS / 'mxdot-driver.om').read_bytes()
    with tempfile.TemporaryDirectory(prefix='mxfmt-check-') as tmp:
        t = Path(tmp)
        fexe = build(compiler, [fp8, nv, fdrv], t / 'fmt')
        bexe = build(compiler, [fp8, nv, bdrv], t / 'nvblk')
        mexe = build(compiler, [fp8, nv, mx, mdrv], t / 'mxdot')
        n_dec = check_fp8(fexe)
        enc = check_fp8_encode(fexe)
        nvd, nve, nvrt, nvsc = check_nvfp4(fexe)
        nblk = check_nvblk(bexe, fexe)
        mcases, mdots, mbit, worst = check_mxdot(mexe)
        nref = fmt_refusals(fexe) + mx_refusals(mexe)
        # negative controls
        neg = []
        neg.append(expect_fail('flipped reference table entry E4M3 0x38', lambda: check_fp8(fexe, flip=True)))
        rhaz = mutate(fp8, b'if gt rm hf\nret add q 1', b'if ge rm hf\nret add q 1')
        rexe = build(compiler, [rhaz, nv, fdrv], t / 'rhaz')
        neg.append(expect_fail('round-half-away-from-zero FP8 encoder', lambda: check_fp8_encode(rexe)))
        nvrhaz = mutate(nv, b'if eq and k 1 1\nc = add k 1', b'if eq k k\nc = add k 1')
        nexe = build(compiler, [fp8, nvrhaz, fdrv], t / 'nvrhaz')
        neg.append(expect_fail('round-half-away-from-zero NVFP4 encoder', lambda: check_nvfp4(nexe)))
        flip = mutate(nv, b'if eq k 5\nret 14', b'if eq k 5\nret 13')
        texe = build(compiler, [fp8, flip, fdrv], t / 'nvflip')
        neg.append(expect_fail('flipped NVFP4 midpoint table entry 14->13', lambda: check_nvfp4(texe)))
        dflip = mutate(fp8, b'ret or s f32norm m -9', b'ret or s f32norm m -8')
        dexe = build(compiler, [dflip, nv, fdrv], t / 'decflip')
        neg.append(expect_fail('E4M3 subnormal exponent mutant', lambda: check_fp8(dexe)))
        mmut = mutate(mx, b'p = add add ue4m3exp ca ue4m3exp cb -2', b'p = add add ue4m3exp ca ue4m3exp cb -1')
        mmexe = build(compiler, [fp8, nv, mmut, mdrv], t / 'mxmut')
        neg.append(expect_fail('NVFP4 dot scale mutant', lambda: check_mxdot(mmexe, ncases=30)))
        print(f'compiler_sha256={digest}')
        for name, data in (('fp8.om', fp8), ('nvfp4.om', nv), ('mxdot.om', mx)):
            print(f'{name}_sha256={hashlib.sha256(data).hexdigest()}')
        for name, exe in (('fmt', fexe), ('nvblk', bexe), ('mxdot', mexe)):
            print(f'{name}_exe_sha256={hashlib.sha256(Path(exe).read_bytes()).hexdigest()}')
        print(f'numpy={np.__version__} ml_dtypes={ml_dtypes.__version__}')
        print(f'FP8 decode exhaustive: {n_dec} codes (E4M3 256, E5M2 256) == ml_dtypes == OFP8 5.1 formula; Table 2 rows checked')
        for (fmt, sat), c in enc.items():
            print(f'FP8 encode {fmt} sat={sat}: {c} inputs (65536 bfloat16 vs exact spec + ml_dtypes; '
                  f'{c - 65536} structured binary32 vs ml_dtypes); round trip 256 codes')
        print(f'NVFP4 decode exhaustive: {nvd} element x scale pairs (16 x 127) == ml_dtypes product == exact')
        print(f'NVFP4 encode: {nve} (non-NaN bfloat16 x 127 scales); threshold +-1ulp Fraction check; round trip {nvrt}')
        print(f'NVFP4 block scale: {nvsc} amax values; block encode {nblk} blocks')
        print(f'MX dot/matmul: {mcases} cases, {mdots} dot products, error <= gamma(nb-1) sum|t| for all; '
              f'{mbit} bit-equal to left-to-right binary64; worst err/sum|t| = {float(worst):.3e}')
        print(f'refusals: {nref}')
        for line in neg:
            print('NEG ' + line)
        print('GPU: NOT_RUN')


if __name__ == '__main__':
    try:
        main()
    except Fail as e:
        print(f'FAIL: {e}')
        raise SystemExit(1)
