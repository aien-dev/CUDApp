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


def product_answer(rows, cols, a, b):
    # Enumerate the contributing paths for each output column. No dense accumulator.
    rp, ci, av = a
    bp, bi, bv = b
    outp, outi, outv = [0], [], []
    for row in range(rows):
        for col in range(cols):
            terms = [mul(av[p], bv[q])
                     for p in range(rp[row], rp[row + 1])
                     for q in range(bp[ci[p]], bp[ci[p] + 1]) if bi[q] == col]
            if terms:
                acc = 0
                for term in terms:
                    acc = add(acc, term)
                outi.append(col)
                outv.append(acc)
        outp.append(len(outi))
    return words([len(outi)] + outp + outi + outv)


def random_csr(rng, rows, cols):
    rp, ci, av = [0], [], []
    for _ in range(rows):
        for _ in range(rng.randrange(7) if cols else 0):
            ci.append(rng.randrange(cols))
            av.append(bits(rng.randrange(-8, 9) / 4))
        rp.append(len(ci))
    return rp, ci, av


def gemm(checks):
    a = ([0, 2, 3], [2, 0, 1], [bits(1), bits(2), bits(-1)])
    b = ([0, 1, 3, 4], [1, 1, 0, 0], [bits(3), bits(4), bits(5), bits(6)])
    fixture = packet(3, 2, 2, a, inner=3, b=b)
    checks.good(fixture, words([4, 0, 2, 4, 0, 1, 0, 1] + [bits(v) for v in [6, 6, -5, -4]]),
                'literal rectangular sparse multiplication')
    a = ([0, 3], [0, 0, 0], [bits(16777216), bits(1), bits(-16777216)])
    b = ([0, 1], [0], [bits(1)])
    checks.good(packet(3, 1, 1, a, inner=1, b=b), words([1, 0, 1, 0, 0]),
                'duplicates separately rounded cancellation retains structural zero')
    a = ([0, 1], [0], [0])
    b = ([0, 1], [0], [0x7f800000])
    checks.good(packet(3, 1, 1, a, inner=1, b=b), words([1, 0, 1, 0, 0x7fc00000]), 'explicit zero times infinity')
    a = ([0, 0], [], [])
    checks.good(packet(3, 1, 1, a, inner=1, b=b), words([0, 0, 0]), 'absent entry does not multiply infinity')
    for rows, inner, cols in [(0, 0, 0), (0, 64, 64), (64, 0, 64), (64, 64, 0), (64, 64, 64)]:
        a, b = ([0] * (rows + 1), [], []), ([0] * (inner + 1), [], [])
        checks.good(packet(3, rows, cols, a, inner=inner, b=b), words([0] + [0] * (rows + 1)), 'empty product')
    a = (list(range(17)), [0] * 16, [bits(1)] * 16)
    b = ([0, 16], list(range(16)), [bits(2)] * 16)
    checks.good(packet(3, 16, 16, a, inner=1, b=b), product_answer(16, 16, a, b), 'exactly 256 output entries')
    a = (list(range(18)), [0] * 17, [bits(1)] * 17)
    b = ([0, 17], list(range(17)), [bits(2)] * 17)
    checks.bad(packet(3, 17, 17, a, inner=1, b=b), 'structural output exceeds capacity')
    a, b = ([0] * 64 + [1], [63], [bits(3)]), ([0] * 64 + [1], [63], [bits(2)])
    checks.good(packet(3, 64, 64, a, inner=64, b=b), product_answer(64, 64, a, b), 'last row and column')
    rng = random.Random(455)
    for case in range(100):
        rows, inner, cols = (rng.randrange(9) for _ in range(3))
        a, b = random_csr(rng, rows, inner), random_csr(rng, inner, cols)
        checks.good(packet(3, rows, cols, a, inner=inner, b=b), product_answer(rows, cols, a, b), 'random product ' + str(case))
    edge = [0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x00800000,
            0x7f7fffff, 0xff7fffff, 0x7f800000, 0xff800000, 0x7fc01234, bits(1)]
    for case in range(40):
        a, b = random_csr(rng, 3, 3), random_csr(rng, 3, 3)
        a = (a[0], a[1], [rng.choice(edge) for _ in a[2]])
        b = (b[0], b[1], [rng.choice(edge) for _ in b[2]])
        checks.good(packet(3, 3, 3, a, inner=3, b=b), product_answer(3, 3, a, b),
                    'special/subnormal payload product ' + str(case))
    valid = ([0, 1], [0], [bits(1)])
    for bad in [([1, 1], [0], [bits(1)]), ([0, 0], [0], [bits(1)]),
                ([0, 1], [1], [bits(1)]), ([0, 1], [0xffffffff], [bits(1)])]:
        checks.bad(packet(3, 1, 1, bad, inner=1, b=valid), 'invalid A')
        checks.bad(packet(3, 1, 1, valid, inner=1, b=bad), 'invalid B')
    malformed(checks, fixture)


def solve_answer(n, a, rhs, upper=False, unit=False):
    rp, ci, av = a
    solution = [0] * n
    for row in (range(n - 1, -1, -1) if upper else range(n)):
        acc, diag = rhs[row], bits(1)
        for p in range(rp[row], rp[row + 1]):
            if ci[p] == row:
                if not unit:
                    diag = av[p]
            else:
                acc = add(acc, mul(av[p], solution[ci[p]]) ^ 0x80000000)
        solution[row] = acc if unit else rounded(value(acc) / value(diag))
    return words(solution)


def solve(checks):
    a = ([0, 1, 3, 6], [0, 1, 0, 1, 0, 2], [bits(v) for v in [2, 4, 3, 2, -1, 8]])
    rhs = [bits(v) for v in [2, 11, 27]]
    fixture = packet(4, 3, 3, a, rhs=rhs)
    checks.good(fixture, words([bits(v) for v in [1, 2, 3]]), 'literal lower solve')
    a = ([0, 3, 5, 6], [2, 0, 1, 2, 1, 2], [bits(v) for v in [-1, 2, 3, 2, 4, 8]])
    checks.good(packet(4, 3, 3, a, flags=1, rhs=[bits(v) for v in [5, 14, 24]]),
                words([bits(v) for v in [1, 2, 3]]), 'literal upper solve')
    checks.good(packet(4, 2, 2, ([0, 0, 2], [0, 0], [bits(2), bits(-1)]), flags=2, rhs=[bits(1), bits(3)]),
                words([bits(1), bits(2)]), 'unit missing diagonal duplicate off-diagonals')
    checks.good(packet(4, 2, 2, ([0, 2, 3], [1, 0, 1], [bits(2), 0, bits(17)]), flags=3, rhs=[bits(5), bits(2)]),
                words([bits(1), bits(2)]), 'unit upper ignores stored finite diagonal')
    checks.good(packet(4, 1, 1, ([0, 1], [0], [bits(3)]), rhs=[bits(1)]), words([bits(1 / 3)]), 'rounded division')
    for flags in range(4):
        checks.good(packet(4, 0, 0, ([0], [], []), flags=flags), b'', 'empty solve')
    checks.good(packet(4, 64, 64, (list(range(65)), list(range(64)), [bits(2)] * 64), rhs=[bits(2 * i) for i in range(64)]),
                words([bits(i) for i in range(64)]), 'maximum dimension solve')
    rp, ci, av = [0], [], []
    for row in range(64):
        for _ in range(0 if row == 0 else (6 if row == 63 else 3)):
            ci.append(0)
            av.append(bits(1))
        ci.append(row)
        av.append(bits(2))
        rp.append(len(ci))
    rhs = [bits(2)] + [bits(5)] * 62 + [bits(8)]
    checks.good(packet(4, 64, 64, (rp, ci, av), rhs=rhs), words([bits(1)] * 64), '256-entry solve')
    rng = random.Random(456)
    for case in range(100):
        n, upper, unit = rng.randrange(1, 11), bool(rng.randrange(2)), bool(rng.randrange(2))
        known = [rng.randrange(-4, 5) for _ in range(n)]
        rp, ci, av, rhs = [0], [], [], []
        for row in range(n):
            available = list(range(row + 1, n) if upper else range(row))
            pairs = [(row, rng.choice([1, 2, 4]))] if not unit or rng.randrange(2) else []
            for _ in range(rng.randrange(5) if available else 0):
                pairs.append((rng.choice(available), rng.randrange(-3, 4)))
            rng.shuffle(pairs)
            diagonal = 1 if unit else next(v for j, v in pairs if j == row)
            rhs.append(bits(diagonal * known[row] + sum(v * known[j] for j, v in pairs if j != row)))
            ci.extend(j for j, _ in pairs)
            av.extend(bits(v) for _, v in pairs)
            rp.append(len(ci))
        flags = int(upper) | (int(unit) << 1)
        checks.good(packet(4, n, n, (rp, ci, av), flags=flags, rhs=rhs),
                    words([bits(v) for v in known]), 'constructed exact solve ' + str(case))
    for case in range(50):
        n, upper = 4, bool(case % 2)
        rp, ci, av = [0], [], []
        rhs = [bits(rng.uniform(-8, 8)) for _ in range(n)]
        for row in range(n):
            pairs = [(row, bits(rng.choice([0.5, 1.5, 3, 7])))]
            for col in (range(row + 1, n) if upper else range(row)):
                pairs.append((col, bits(rng.uniform(-2, 2))))
            rng.shuffle(pairs)
            ci.extend(j for j, _ in pairs)
            av.extend(v for _, v in pairs)
            rp.append(len(ci))
        a = (rp, ci, av)
        checks.good(packet(4, n, n, a, flags=int(upper), rhs=rhs),
                    solve_answer(n, a, rhs, upper), 'rounded random solve ' + str(case))
    a = ([0, 1, 5], [0, 0, 0, 0, 1], [bits(v) for v in [1, 16777216, 1, -16777216, 3]])
    rhs = [bits(1), bits(1)]
    checks.good(packet(4, 2, 2, a, rhs=rhs), solve_answer(2, a, rhs), 'stored-order subtraction rounding')
    for a, n, flags, rhs in [
        (([0, 1], [0], [0]), 1, 0, [bits(1)]),
        (([0, 1], [0], [0x80000000]), 1, 0, [bits(1)]),
        (([0, 0], [], []), 1, 0, [bits(1)]),
        (([0, 2], [0, 0], [bits(1), bits(2)]), 1, 0, [bits(1)]),
        (([0, 2], [0, 0], [bits(1), bits(2)]), 1, 2, [bits(1)]),
        (([0, 2, 3], [0, 1, 1], [bits(1)] * 3), 2, 0, [bits(1)] * 2),
        (([0, 1, 3], [0, 0, 1], [bits(1)] * 3), 2, 1, [bits(1)] * 2),
        (([0, 1], [0xffffffff], [bits(1)]), 1, 0, [bits(1)]),
        (([0, 1], [0], [0x7f800000]), 1, 0, [bits(1)]),
        (([0, 1], [0], [0x7fc01234]), 1, 2, [bits(1)]),
        (([0, 1], [0], [bits(1)]), 1, 0, [0xff800000]),
        (([0, 1], [0], [bits(1)]), 1, 0, [0x7fc00000]),
        (([0, 1], [0], [1]), 1, 0, [0x7f7fffff]),
        (([0, 1, 3], [0, 0, 1], [bits(1), 0x7f7fffff, bits(1)]), 2, 0, [0x7f7fffff, bits(1)]),
    ]:
        checks.bad(packet(4, n, n, a, flags=flags, rhs=rhs), 'invalid or nonfinite solve')
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
    def build(name, changed=None, adapter=driver):
        target = args.out.resolve() / name
        content = dict(sources)
        if changed:
            part, before, after = changed
            assert content[part].count(before) == 1, ('mutation anchor', name)
            content[part] = content[part].replace(before, after)
        compile_program(compiler, '\n'.join(content.values()) + '\n' + adapter, target)
        image = target.read_bytes()
        assert len(image) >= 64 and image[:4] == b'\x7fELF' and image[18:20] == b'\xb7\x00', ('not an AArch64 ELF', name)
        return target
    target = build('sparse-core')
    stages = {'convert': convert, 'gemm': gemm, 'solve': solve}
    if args.stage != 'all' and args.stage not in stages:
        raise AssertionError('stage not implemented: ' + args.stage)
    selected = stages if args.stage == 'all' else {args.stage: stages[args.stage]}
    total = 0
    for name, exercise in selected.items():
        checks = Checks(target)
        exercise(checks)
        print(name.upper(), 'PASS', checks.count, 'checks', flush=True)
        total += checks.count
    state = (ROOT / 'tests/libraries/sparse_core_state.om').read_text()
    def check_state(binary):
        result = subprocess.run([str(binary)], capture_output=True, timeout=5)
        assert result.returncode == 0 and not result.stdout, ('core state', result)
    check_state(build('state-check', adapter=state))
    print('CORE STATE PASS output preservation on capacity/arithmetic/validation failures and repeated calls', flush=True)
    if args.mutants:
        mutations = [
            ('conversion-value', 'sparse_convert', 'put cval dest get sv p', 'put cval dest xor get sv p 1', 'convert'),
            ('conversion-row', 'sparse_convert', 'put sr p row', 'put sr p add row 1', 'convert'),
            ('product-math', 'sparse_gemm', 'prod = fmul get aval p get bval q', 'prod = fadd get aval p get bval q', 'gemm'),
            ('product-structural-zero', 'sparse_gemm', 'if ne get wseen col 0', 'if ne get wacc col 0', 'gemm'),
            ('product-partial-publish', 'sparse_gemm', 'if ne gmstore row cols 0\nret 1', 'if ne gmstore row cols 0\npublish rows\nret 1', 'gemm'),
            ('solve-diagonal', 'sparse_solve', 'acc = fdiv acc diag', 'acc = fmul acc diag', 'solve'),
            ('solve-order', 'sparse_solve', 'if ne upper 0', 'if eq upper 0', 'solve'),
            ('solve-triangle-check', 'sparse_solve', 'if and eq upper 0 gt col row', 'if false', 'solve'),
        ]
        for name, part, before, after, stage in mutations:
            if stage not in selected:
                continue
            binary = build(name, (part, before, after))
            mutant_state = build(name + '-state', (part, before, after), adapter=state)
            try:
                selected[stage](Checks(binary))
                check_state(mutant_state)
            except AssertionError:
                print('MUTANT CAUGHT', name, flush=True)
            else:
                raise AssertionError('surviving mutant ' + name)
    print('SPARSE CORE PASS', total, 'oracle/refusal checks compiler=' + COMPILER_SHA, flush=True)


if __name__ == '__main__':
    main()
