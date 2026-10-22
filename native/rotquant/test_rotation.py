#!/usr/bin/env python3
"""Independent explicit-matrix oracle; never used by the native executable."""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import random
import struct
import subprocess

H = [[-1 if (i & j).bit_count() % 2 else 1 for j in range(128)] for i in range(128)]
FINITE = [b for b in range(256) if b & 127 != 127]


def units(b):
    sign = -1 if b >> 7 else 1
    exponent, fraction = (b >> 3) & 15, b & 7
    if b & 127 == 127:
        raise ValueError('NaN')
    value = Fraction(fraction, 8) * Fraction(2) ** -6 if exponent == 0 else Fraction(8 + fraction, 8) * Fraction(2) ** (exponent - 7)
    return int(sign * value * 512)


def transform(values):
    return [sum(s * x for s, x in zip(row, values)) for row in H]


def encode(data, mode):
    assert len(data) % 128 == 0 and mode in (0, 1)
    result = bytearray(struct.pack('<4sII', b'RQP1', len(data)//128, mode))
    for p in range(0, len(data), 128):
        v = [units(x) for x in data[p:p+128]]
        if mode:
            v = transform(v)
        m = max(map(abs, v))
        q = [round(Fraction(7*x, m)) if m else 0 for x in v]
        result += struct.pack('<I', m)
        result += bytes((q[i]+7) | ((q[i+1]+7) << 4) for i in range(0, 128, 2))
    return bytes(result)


def decode(packed):
    magic, count, mode = struct.unpack_from('<4sII', packed)
    assert magic == b'RQP1' and 1 <= count <= 256 and mode in (0, 1)
    assert len(packed) == 12 + count*68
    denom = 3584 if mode == 0 else 458752
    output = bytearray(struct.pack('<4sIII', b'RQR1', count, mode, denom))
    for b in range(count):
        p = 12 + b*68
        m = struct.unpack_from('<I', packed, p)[0]
        assert m <= (229376 if mode == 0 else 29360128)
        codes = [n for byte in packed[p+4:p+68] for n in (byte & 15, byte >> 4)]
        assert 15 not in codes
        q = [n-7 for n in codes]
        assert (m == 0 and all(x == 0 for x in q)) or (m > 0 and max(map(abs, q)) == 7)
        if mode:
            q = transform(q)
        output += struct.pack('<128q', *(m*x for x in q))
    return bytes(output)


def request(data, mode):
    return struct.pack('<4sII', b'RQU1', len(data)//128, mode) + data


def cases():
    vectors = [('zero', bytes(128)), ('negative-zero', bytes([128])*128),
               ('positive-max', bytes([126])*128), ('negative-max', bytes([254])*128),
               ('all-finite', bytes(FINITE + [0, 128])),
               ('ties', bytes(([14, 1, 3, 5, 7, 9, 11, 13, 129, 131, 133, 135, 137, 139, 141, 142]*8))),
               ('impulse', bytes([126])+bytes(127)),
               ('alternating', bytes([126, 254])*64),
               ('capacity', bytes([0, 128])*16384),
               ('capacity-nonzero', bytes(FINITE[i % len(FINITE)] for i in range(32768)))]
    rng = random.Random(49049)
    for i in range(20):
        count = 1 + i % 3
        vectors.append((f'random-{i}', bytes(rng.choice(FINITE) for _ in range(128*count))))
    for name, data in vectors:
        for mode in (0, 1):
            packed = encode(data, mode)
            yield name+f'-mode{mode}-encode', request(data, mode), 0, packed
            yield name+f'-mode{mode}-decode', packed, 0, decode(packed)
    for mode in (0, 1):
        raw = request(bytes(128), mode)
        packed = encode(bytes(128), mode)
        for kind, frame in [('raw', raw), ('packed', packed)]:
            for i in range(len(frame)):
                yield f'{kind}-{mode}-prefix{i}', frame[:i], 65, b''
            yield f'{kind}-{mode}-trailing', frame+b'X', 65, b''
            yield f'{kind}-{mode}-magic', b'BAD!'+frame[4:], 65, b''
            for count in (0, 257, 0xffffffff):
                bad = bytearray(frame)
                struct.pack_into('<I', bad, 4, count)
                yield f'{kind}-{mode}-count{count}', bytes(bad), 65, b''
            bad = bytearray(frame)
            struct.pack_into('<I', bad, 8, 2)
            yield f'{kind}-{mode}-bad-mode', bytes(bad), 65, b''
        for code in (127, 255):
            for position in (0, 127, 255):
                data = bytearray(256)
                data[position] = code
                yield f'nan-{mode}-{code}-{position}', request(data, mode), 65, b''
        bad = bytearray(packed)
        bad[16] = 15
        yield f'bad-nibble-{mode}', bytes(bad), 65, b''
        bad = bytearray(packed)
        bad[16] = 0x77 ^ 1
        yield f'zero-scale-nonzero-{mode}', bytes(bad), 65, b''
        bad = bytearray(packed)
        struct.pack_into('<I', bad, 12, 1)
        yield f'nonzero-scale-no-peak-{mode}', bytes(bad), 65, b''
        for val in (229377 if mode == 0 else 29360129, 0xffffffff):
            bad = bytearray(packed)
            struct.pack_into('<I', bad, 12, val)
            yield f'scale-limit-{mode}-{val}', bytes(bad), 65, b''
    yield 'overlong', bytes(32781), 65, b''
    # A structurally valid maximum numerator and negative serialization oracle.
    for mode in (0, 1):
        m = 229376 if mode == 0 else 29360128
        packed = struct.pack('<4sIII', b'RQP1', 1, mode, m)+bytes(64)
        yield f'negative-numerator-bound-{mode}', packed, 0, decode(packed)


def suite(binary):
    rows = []
    for name, data, rc, expected in cases():
        p = subprocess.run([str(binary)], input=data, capture_output=True, timeout=15)
        rows.append({'case': name, 'pass': p.returncode == rc and p.stdout == expected and not p.stderr,
                     'rc': p.returncode, 'expected_rc': rc,
                     'input_sha256': hashlib.sha256(data).hexdigest(),
                     'expected_sha256': hashlib.sha256(expected).hexdigest(),
                     'actual_sha256': hashlib.sha256(p.stdout).hexdigest(),
                     'stderr': p.stderr.decode(errors='replace')[:200]})
    return {'total': len(rows), 'passed': sum(r['pass'] for r in rows),
            'failed': sum(not r['pass'] for r in rows), 'cases': rows}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('binary', type=Path)
    p.add_argument('--out', required=True, type=Path)
    args = p.parse_args()
    result = suite(args.binary.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'cases'}))
    print('FAILURES', [r['case'] for r in result['cases'] if not r['pass']][:12])
    return int(result['failed'] != 0)


if __name__ == '__main__':
    raise SystemExit(main())
