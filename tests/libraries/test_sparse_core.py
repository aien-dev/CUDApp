#!/usr/bin/env python3
"""Independent behavioral oracles for native Omega sparse primitives.
Conversion: stable sorting, not count/scatter. Product: per-column path enumeration.
Solves: separately rounded substitution plus literal exactly constructed systems.
"""
import argparse
import hashlib
import random
import struct
import subprocess
from pathlib import Path
from test_csr_transpose import COMPILER_SHA, compile_program

ROOT = Path(__file__).resolve().parents[2]
PARTS = ['binary32', 'fdiv', 'sparse_core', 'sparse_convert', 'sparse_gemm', 'sparse_solve']


def words(values):
    return struct.pack('<%dI' % len(values), *values)


def bits(value):
    try:
        return struct.unpack('<I', struct.pack('<f', value))[0]
    except OverflowError:
        return 0x7f800000 if value > 0 else 0xff800000


def value(word):
    return struct.unpack('<f', struct.pack('<I', word))[0]


def rounded(number):
    result = bits(number)
    return 0x7fc00000 if result & 0x7fffffff > 0x7f800000 else result


def add(a, b):
    return rounded(value(a) + value(b))


def mul(a, b):
    return rounded(value(a) * value(b))


def packet(mode, rows, cols, a, inner=0, b=None, flags=0, rhs=()):
    first = words([mode, rows, inner, cols, len(a[1]), 0 if b is None else len(b[1]), flags, 0])
    result = first + words(list(a[0]) + list(a[1]) + list(a[2]))
    if b is not None:
        result += words(list(b[0]) + list(b[1]) + list(b[2]))
    return result + words(rhs)


class Checks:
    def __init__(self, binary):
        self.binary, self.count = binary, 0

    def run(self, data):
        return subprocess.run([str(self.binary)], input=data, capture_output=True, timeout=5)

    def good(self, data, expected, name):
        result = self.run(data)
        assert result.returncode == 0 and result.stdout == expected, (name, result, expected.hex())
        self.count += 1

    def bad(self, data, name):
        result = self.run(data)
        assert result.returncode == 2 and not result.stdout, (name, result)
        self.count += 1


def coo_answer(rows, rr, cc, vv):
    entries = sorted(zip(rr, cc, vv), key=lambda entry: entry[0])
    ptr = [sum(row < i for row, _, _ in entries) for i in range(rows + 1)]
    return words([len(rr)] + ptr + [col for _, col, _ in entries] + [v for _, _, v in entries])


def coo_from_csr(rows, a):
    rp, ci, av = a
    rr = [row for row in range(rows) for _ in range(rp[row], rp[row + 1])]
    return words([len(ci)] + rr + list(ci) + list(av))


def malformed(checks, fixture):
    checks.bad(fixture + b'\0', 'trailing byte')
    for cut in range(len(fixture)):
        checks.bad(fixture[:cut], 'truncation ' + str(cut))
    header = list(struct.unpack('<8I', fixture[:32]))
    for field, val in [(0, 0), (0, 5), (1, 65), (2, 65), (3, 65),
                       (4, 257), (5, 257), (6, 4), (7, 1), (1, 0xffffffff)]:
        changed = header.copy()
        changed[field] = val
        checks.bad(words(changed) + fixture[32:], 'bad header ' + str((field, val)))
    checks.bad(bytes(8192), 'full input buffer')


def convert(checks):
    rr, cc, vv = [2, 0, 2, 0, 0], [1, 3, 1, 0, 3], [7, 0x80000000, 0x7fc01234, 0, 0xffffffff]
    fixture = packet(1, 3, 4, (rr, cc, vv))
    literal = words([5, 0, 3, 3, 5, 3, 0, 3, 1, 1,
                     0x80000000, 0, 0xffffffff, 7, 0x7fc01234])
    checks.good(fixture, literal, 'literal stable COO grouping')
    a = ([0, 3, 3, 5], [3, 0, 3, 1, 1], [0x80000000, 0, 0xffffffff, 7, 0x7fc01234])
    checks.good(packet(2, 3, 4, a), words([5] + [0, 0, 0, 2, 2] + a[1] + a[2]), 'literal CSR expansion')
    for rows, cols in [(0, 0), (0, 64), (64, 0), (64, 64)]:
        checks.good(packet(1, rows, cols, ([], [], [])), words([0] + [0] * (rows + 1)), 'empty COO')
        checks.good(packet(2, rows, cols, ([0] * (rows + 1), [], [])), words([0]), 'empty CSR')
    checks.good(packet(1, 64, 64, ([63] * 256, [63] * 256, list(range(256)))),
                coo_answer(64, [63] * 256, [63] * 256, list(range(256))), 'maximum COO')
    rng = random.Random(454)
    for case in range(100):
        rows, cols, n = rng.randrange(1, 65), rng.randrange(1, 65), rng.randrange(257)
        rr, cc, vv = [rng.randrange(rows) for _ in range(n)], [rng.randrange(cols) for _ in range(n)], [rng.getrandbits(32) for _ in range(n)]
        expected = coo_answer(rows, rr, cc, vv)
        checks.good(packet(1, rows, cols, (rr, cc, vv)), expected, 'random COO ' + str(case))
        arr = list(struct.unpack('<%dI' % (len(expected) // 4), expected))
        a = (arr[1:rows + 2], arr[rows + 2:rows + 2 + n], arr[rows + 2 + n:])
        checks.good(packet(2, rows, cols, a), coo_from_csr(rows, a), 'random CSR ' + str(case))
    for data in [packet(1, 1, 1, ([1], [0], [7])), packet(1, 1, 1, ([0], [1], [7])),
                 packet(1, 1, 1, ([0xffffffff], [0], [7])), packet(1, 0, 1, ([0], [0], [7])),
                 packet(2, 1, 1, ([1, 1], [0], [7])), packet(2, 1, 1, ([0, 0], [0], [7])),
                 packet(2, 2, 1, ([0, 2, 1], [0], [7])),
                 packet(2, 2, 1, ([0, 0xffffffff, 1], [0], [7])),
                 packet(2, 1, 1, ([0, 1], [0xffffffff], [7]))]:
        checks.bad(data, 'invalid coordinates or CSR')
    malformed(checks, fixture)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--compiler', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=ROOT / 'build/sparse-core')
    parser.add_argument('--stage', choices=['convert', 'gemm', 'solve', 'all'], default='all')
    parser.add_argument('--mutants', action='store_true')
    args = parser.parse_args()
    compiler = args.compiler.resolve()
    assert hashlib.sha256(compiler.read_bytes()).hexdigest() == COMPILER_SHA, 'unqualified compiler'
    args.out.mkdir(parents=True, exist_ok=True)
    sources = {name: (ROOT / ('omega/libraries/' + name + '.om')).read_text() for name in PARTS}
    driver = (ROOT / 'tests/libraries/sparse_core_driver.om').read_text()
    def build(name, changed=None):
        target = args.out.resolve() / name
        content = dict(sources)
        if changed:
            part, before, after = changed
            assert content[part].count(before) == 1, ('mutation anchor', name)
            content[part] = content[part].replace(before, after)
        compile_program(compiler, '\n'.join(content.values()) + '\n' + driver, target)
        return target
    target = build('sparse-core')
    stages = {'convert': convert}
    if args.stage != 'all' and args.stage not in stages:
        raise AssertionError('stage not implemented: ' + args.stage)
    selected = stages if args.stage == 'all' else {args.stage: stages[args.stage]}
    for name, exercise in selected.items():
        checks = Checks(target)
        exercise(checks)
        print(name.upper(), 'PASS', checks.count, 'checks', flush=True)
    if args.mutants:
        mutations = [
            ('conversion-value', 'sparse_convert', 'put cval dest get sv p', 'put cval dest xor get sv p 1', 'convert'),
            ('conversion-row', 'sparse_convert', 'put sr p row', 'put sr p add row 1', 'convert'),
        ]
        for name, part, before, after, stage in mutations:
            if stage not in selected:
                continue
            binary = build(name, (part, before, after))
            try:
                selected[stage](Checks(binary))
            except AssertionError:
                print('MUTANT CAUGHT', name, flush=True)
            else:
                raise AssertionError('surviving mutant ' + name)
    print('SPARSE CORE PASS compiler=' + COMPILER_SHA, flush=True)


if __name__ == '__main__':
    main()
