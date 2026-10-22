#!/usr/bin/env python3
"""Independent CSR SpMV oracle. Each product and sum is rounded to binary32."""
import argparse, struct, subprocess
from pathlib import Path

def f32(u):
    return struct.unpack('<f', struct.pack('<I', u & 0xffffffff))[0]
def bits(x):
    return struct.unpack('<I', struct.pack('<f', x))[0]
def fadd(a, b):
    return bits(f32(a) + f32(b))
def fmul(a, b):
    return bits(f32(a) * f32(b))
def b(x):
    return bits(float(x))

def spmv(rows, cols, rp, ci, vals, x):
    y = []
    for i in range(rows):
        acc = 0
        for p in range(rp[i], rp[i + 1]):
            acc = fadd(acc, fmul(vals[p], x[ci[p]]))
        y.append(acc)
    return y

def raw(rows, cols, nnz, reserved, rp, ci, vals, x, extra=b''):
    return (struct.pack('<IIII', rows, cols, nnz, reserved)
            + struct.pack('<%dI' % len(rp), *rp)
            + struct.pack('<%dI' % len(ci), *ci)
            + struct.pack('<%dI' % len(vals), *vals)
            + struct.pack('<%dI' % len(x), *x)
            + extra)

def run(exe, data):
    return subprocess.run([str(exe)], input=data, capture_output=True, timeout=10)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('binary', type=Path)
    args = ap.parse_args()
    assert args.binary.is_file(), 'sparse implementation executable is missing'
    checks = 0
    cases = [
        (1, 1, [0, 1], [0], [b(3)], [b(4)], 'one entry'),
        (2, 2, [0, 2, 3], [0, 1, 1], [b(1), b(2), b(5)], [b(3), b(4)], 'two rows'),
        (2, 3, [0, 0, 2], [1, 2], [b(1.5), b(-2)], [b(4), b(0.5), b(1)], 'empty first row'),
        (1, 2, [0, 2], [0, 1], [b(1e-20), b(1)], [b(1), b(1)], 'underflow product then one'),
        (3, 1, [0, 1, 1, 2], [0, 0], [b(2), b(-3)], [b(7)], 'rectangular'),
        (1, 1, [0, 0], [], [], [b(9)], 'explicit zero row'),
    ]
    for rows, cols, rp, ci, vals, x, name in cases:
        want = spmv(rows, cols, rp, ci, vals, x)
        r = run(args.binary, raw(rows, cols, rp[-1], 0, rp, ci, vals, x))
        assert r.returncode == 0 and r.stdout == struct.pack('<%dI' % rows, *want), (name, r)
        checks += 1
    # A zero-size product is a legal empty result, not a guess.
    r = run(args.binary, raw(0, 0, 0, 0, [0], [], [], []))
    assert r.returncode == 0 and r.stdout == b'', r
    checks += 1
    bad = [
        b'',
        bytes(15),
        raw(65, 1, 0, 0, [0] * 66, [], [], [b(1)]),
        raw(1, 65, 0, 0, [0, 0], [], [], [b(1)] * 65),
        raw(1, 1, 1, 0, [0, 1], [1], [b(1)], [b(1)]),
        raw(2, 1, 2, 0, [0, 2, 1], [0, 0], [b(1), b(1)], [b(1)]),
        raw(1, 1, 1, 0, [0, 1], [0], [b(1)], [b(1)], b'\x00'),
        raw(1, 1, 1, 1, [0, 1], [0], [b(1)], [b(1)]),
        raw(1, 0, 1, 0, [0, 1], [0], [b(1)], []),
    ]
    for data in bad:
        r = run(args.binary, data)
        assert r.returncode != 0 and r.stdout == b'', (len(data), r)
        checks += 1
    print(f'SPARSE PASS {checks} checks (CSR binary32, empty rows, refusals)')

if __name__ == '__main__':
    main()
