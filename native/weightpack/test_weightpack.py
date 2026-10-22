#!/usr/bin/env python3
"""Independent test oracle, fixtures and process harness, not a deployed codec."""
import argparse
import hashlib
import json
import random
import struct
import subprocess
from pathlib import Path

LIMIT = 1048576

def raw_frame(rows, cols, axis, data):
    return struct.pack('<4sIII', b'ZCR1', rows, cols, axis) + data


def channels(data, rows, cols, axis):
    if axis == 0:
        return [data[i * cols:(i + 1) * cols] for i in range(rows)]
    return [data[i::cols] for i in range(cols)]


def reference_encode(frame):
    magic, rows, cols, axis = struct.unpack('<4sIII', frame[:16])
    assert magic == b'ZCR1' and axis in (0, 1)
    assert 1 <= rows <= 4096 and 1 <= cols <= 4096
    data = frame[16:]
    assert rows * cols == len(data) <= LIMIT
    payload = bytearray()
    for channel in channels(data, rows, cols, axis):
        if any(b & 127 for b in channel):
            payload += b'\x02' + channel
        elif any(channel):
            bits = bytearray((len(channel) + 7) // 8)
            for i, b in enumerate(channel):
                bits[i // 8] |= (b >> 7) << (i % 8)
            payload += b'\x01' + bits
        else:
            payload += b'\x00'
    mode = int(len(payload) < len(data))
    return struct.pack('<4sIIII', b'ZCP1', rows, cols, axis, mode) + (payload if mode else data)


def reference_decode(frame):
    magic, rows, cols, axis, mode = struct.unpack('<4sIIII', frame[:20])
    assert magic == b'ZCP1' and axis in (0, 1) and mode in (0, 1)
    assert 1 <= rows <= 4096 and 1 <= cols <= 4096 and rows * cols <= LIMIT
    payload = frame[20:]
    if mode == 0:
        assert len(payload) == rows * cols
        data = payload
    else:
        assert len(payload) < rows * cols
        count, width = (rows, cols) if axis == 0 else (cols, rows)
        pos = 0
        data = bytearray(rows * cols)
        for c in range(count):
            tag = payload[pos]
            pos += 1
            assert tag in (0, 1, 2)
            if tag == 0:
                values = bytes(width)
            elif tag == 1:
                size = (width + 7) // 8
                bits = payload[pos:pos + size]
                assert len(bits) == size and any(bits)
                if width % 8:
                    assert bits[-1] < (1 << (width % 8))
                values = bytes(128 if bits[i // 8] & (1 << (i % 8)) else 0 for i in range(width))
                pos += size
            else:
                values = payload[pos:pos + width]
                assert len(values) == width and any(b & 127 for b in values)
                pos += width
            for j, value in enumerate(values):
                index = c * cols + j if axis == 0 else j * cols + c
                data[index] = value
        assert pos == len(payload)
    return raw_frame(rows, cols, axis, bytes(data))


def valid_fixtures():
    fixtures = []
    for rows, cols in [(1, 1), (1, 7), (7, 1), (3, 9), (9, 3), (8, 8), (17, 16), (32, 32)]:
        n = rows * cols
        for axis in (0, 1):
            for label, data in [('plus-zero', bytes(n)), ('minus-zero', bytes([128]) * n),
                                ('signs', bytes([0, 128][i % 2] for i in range(n))),
                                ('all-codes', bytes(i % 256 for i in range(n)))]:
                fixtures.append((f'{rows}x{cols}-axis{axis}-{label}', raw_frame(rows, cols, axis, data)))
    rng = random.Random(60048)
    for i in range(64):
        rows, cols, axis = rng.randint(1, 37), rng.randint(1, 37), i % 2
        count, width = (rows, cols) if axis == 0 else (cols, rows)
        data = bytearray(rows * cols)
        for c in range(count):
            kind = rng.randrange(3)
            values = [rng.randrange(256) if kind == 2 else (128 * rng.randrange(2) if kind else 0) for _ in range(width)]
            for j, value in enumerate(values):
                data[c * cols + j if axis == 0 else j * cols + c] = value
        fixtures.append((f'random-{i}', raw_frame(rows, cols, axis, bytes(data))))
    for axis in (0, 1):
        fixtures.append((f'limit-{axis}', raw_frame(512, 2048, axis, bytes(LIMIT))))
        fixtures.append((f'dense-limit-{axis}', raw_frame(512, 2048, axis, bytes(range(256)) * (LIMIT // 256))))
    return fixtures


def cases():
    good = valid_fixtures()
    for label, raw in good:
        packed = reference_encode(raw)
        assert reference_decode(packed) == raw
        yield label + '-encode', raw, 0, packed
        yield label + '-decode', packed, 0, raw
    # Explicitly accept valid dense fallback even if sparse would be smaller.
    yield 'nonminimal-dense-accepted', struct.pack('<4sIIII', b'ZCP1', 3, 9, 0, 0) + bytes(27), 0, raw_frame(3, 9, 0, bytes(27))
    malformed = []
    for magic, header in [(b'ZCR1', 16), (b'ZCP1', 20)]:
        base = struct.pack('<4sIIII', magic, 3, 9, 0, 0)[:header] + bytes(27)
        malformed.extend((f'{magic!r}-prefix-{i}', base[:i]) for i in range(len(base)))
        malformed.extend([(f'{magic!r}-trailing', base + b'X'), (f'{magic!r}-badmagic', b'BAD!' + base[4:])])
        for field in (1, 2, 3):
            for val in ([0, 4097, 0xffffffff] if field != 3 else [2, 0xffffffff]):
                b = bytearray(base)
                struct.pack_into('<I', b, field * 4, val)
                malformed.append((f'{magic!r}-field{field}-{val}', bytes(b)))
        b = bytearray(base)
        struct.pack_into('<II', b, 4, 4096, 4096)
        malformed.append((f'{magic!r}-product-limit', bytes(b)))
    ph = struct.pack('<4sIIII', b'ZCP1', 3, 9, 0, 1)
    for label, payload in [('invalid-tag', b'\x03\0\0'), ('sign-empty', b'\x01\0\0\0\0'),
                           ('sign-padding', b'\x01\0\x80\0\0'), ('raw-allzero', b'\x02' + bytes(9) + b'\0\0'),
                           ('missing-channel', b'\0\0'), ('extra-channel', b'\0\0\0\0'),
                           ('sign-truncated', b'\x01\x80'), ('raw-truncated', b'\x02\x01')]:
        malformed.append((label, ph + payload))
    malformed.append(('sparse-not-smaller', struct.pack('<4sIIII', b'ZCP1', 1, 1, 0, 1) + b'\0'))
    malformed.append(('bad-mode', struct.pack('<4sIIII', b'ZCP1', 3, 9, 0, 2) + bytes(27)))
    malformed.append(('overlong-input', bytes(LIMIT + 21)))
    sample = reference_encode(raw_frame(3, 17, 0, b'\x80' * 17 + bytes(range(17)) + bytes(17)))
    malformed.extend((f'sparse-prefix-{i}', sample[:i]) for i in range(len(sample)))
    malformed.append(('sparse-trailing', sample + b'\0'))
    # Bounded arbitrary byte fuzz: reject unless one of the two framed magics.
    rng = random.Random(48)
    for i in range(32):
        malformed.append((f'garbage-{i}', bytes(rng.randrange(256) for _ in range(rng.randrange(1, 100)))))
    for label, data in malformed:
        yield label, data, 65, b''


def run_suite(binary):
    records = []
    for label, data, rc, expected in cases():
        p = subprocess.run([str(binary)], input=data, capture_output=True, timeout=15)
        ok = p.returncode == rc and p.stdout == expected and not p.stderr
        records.append({'case': label, 'pass': ok, 'expected_rc': rc, 'rc': p.returncode,
                        'input_sha256': hashlib.sha256(data).hexdigest(),
                        'expected_sha256': hashlib.sha256(expected).hexdigest(),
                        'actual_sha256': hashlib.sha256(p.stdout).hexdigest(),
                        'stderr': p.stderr.decode(errors='replace')[:300]})
    return {'binary_sha256': hashlib.sha256(Path(binary).read_bytes()).hexdigest(),
            'total': len(records), 'passed': sum(x['pass'] for x in records),
            'failed': sum(not x['pass'] for x in records), 'cases': records}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('binary', type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    result = run_suite(args.binary.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'cases'}))
    for row in [r for r in result['cases'] if not r['pass']][:12]:
        print('FAIL', row['case'], 'expected_rc', row['expected_rc'], 'rc', row['rc'])
    return int(result['failed'] != 0)


if __name__ == '__main__':
    raise SystemExit(main())
