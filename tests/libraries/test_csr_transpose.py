#!/usr/bin/env python3
"""Oracle uses stable tuple grouping, not the implementation's count/scatter.
Catches wrong destination, lost duplicates, reordered values, unchecked CSR bounds.
"""
import argparse
import hashlib
import random
import struct
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COMPILER_SHA = '10d7565239ca6bd3cca344b65a03c0f3498acd95e82ab3d420bd01408b01398e'


def words(xs):
    return struct.pack('<%dI' % len(xs), *xs)


def request(rows, cols, rp, ci, av, reserved=0):
    return words([rows, cols, len(ci), reserved] + rp + ci + av)


def oracle(rows, cols, rp, ci, av):
    entries = [(ci[p], i, av[p]) for i in range(rows)
               for p in range(rp[i], rp[i + 1])]
    entries.sort(key=lambda entry: entry[0])  # stable, preserves duplicates
    cp = [sum(col < j for col, _, _ in entries) for j in range(cols + 1)]
    return words(cp + [i for _, i, _ in entries] + [v for _, _, v in entries])


def exercise(binary):
    checks = 0
    def run(data):
        return subprocess.run([str(binary)], input=data, capture_output=True, timeout=5)
    def good(rows, cols, rp, ci, av, name):
        nonlocal checks
        result = run(request(rows, cols, rp, ci, av))
        expected = oracle(rows, cols, rp, ci, av)
        assert result.returncode == 0 and result.stdout == expected, (name, result, expected.hex())
        checks += 1
    # Literal non-square case, duplicates, empty rows/columns, NaN payload and signed zero.
    literal = request(3, 4, [0, 3, 3, 5], [2, 0, 2, 3, 0],
                      [0x80000000, 7, 0x7fc01234, 0x7f800000, 0xff800000])
    expected = words([0, 2, 2, 4, 5, 0, 2, 0, 0, 2,
                      7, 0xff800000, 0x80000000, 0x7fc01234, 0x7f800000])
    result = run(literal)
    assert result.returncode == 0 and result.stdout == expected, ('literal transpose', result)
    checks += 1
    for dims in [(0, 0), (0, 64), (64, 0), (64, 64), (1, 1)]:
        rows, cols = dims
        good(rows, cols, [0] * (rows + 1), [], [], 'empty ' + str(dims))
    good(64, 64, [0] + [256] * 64, [63] * 256,
         list(range(256)), 'max duplicates and empty tail')
    good(64, 64, [0] * 64 + [1], [63], [0xffffffff],
         'last row last column full-width value')
    rng = random.Random(452)
    for n in range(120):
        rows, cols = rng.randrange(1, 65), rng.randrange(1, 65)
        rp, ci, av = [0], [], []
        budget = rng.randrange(257)
        for _ in range(rows):
            count = rng.randrange(min(12, budget) + 1)
            ci.extend(rng.randrange(cols) for _ in range(count))
            av.extend(rng.getrandbits(32) for _ in range(count))
            budget -= count
            rp.append(len(ci))
        good(rows, cols, rp, ci, av, 'random ' + str(n))
    bad = [b'', bytes(15), literal + b'\0', literal[:-1],
           request(65, 1, [0] * 66, [], []),
           request(1, 65, [0, 0], [], []),
           request(1, 1, [0, 257], [0] * 257, [1] * 257),
           request(1, 1, [1, 1], [0], [1]),
           request(2, 1, [0, 2, 1], [0], [1]),
           request(2, 1, [0, 0xffffffff, 1], [0], [1]),
           request(1, 1, [0, 0], [0], [1]),
           request(1, 1, [0, 1], [1], [1]),
           request(1, 1, [0, 1], [0xffffffff], [1]),
           request(1, 0, [0, 1], [0], [1]),
           request(0, 1, [1], [], []),
           request(1, 1, [0, 0], [], [], reserved=1),
           words([0xffffffff, 1, 0, 0]),
           bytes(4096)]
    # Every truncation of a valid request must refuse without publishing partial output.
    bad.extend(literal[:i] for i in range(len(literal)))
    for index, data in enumerate(bad):
        result = run(data)
        assert result.returncode == 2 and result.stdout == b'', ('refusal', index, result)
        checks += 1
    return checks


def compile_program(compiler, source, target):
    result = subprocess.run([str(compiler)], input=source.encode(), capture_output=True, timeout=30)
    assert result.returncode == 0, ('compile', result.returncode, result.stderr)
    target.write_bytes(result.stdout)
    target.chmod(0o755)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--compiler', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=ROOT / 'build/csr-transpose')
    parser.add_argument('--mutants', action='store_true')
    args = parser.parse_args()
    compiler = args.compiler.resolve()
    actual = hashlib.sha256(compiler.read_bytes()).hexdigest()
    assert actual == COMPILER_SHA, ('unqualified compiler', actual)
    print('COMPILER sha256=' + actual, flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    library = (ROOT / 'omega/libraries/csr_transpose.om').read_text()
    driver = (ROOT / 'tests/libraries/csr_transpose_driver.om').read_text()
    target = args.out.resolve() / 'transpose'
    compile_program(compiler, library + '\n' + driver, target)
    count = exercise(target)
    print('CSR TRANSPOSE PASS', count, 'checks', flush=True)
    state = (ROOT / 'tests/libraries/csr_transpose_state.om').read_text()
    state_binary = args.out.resolve() / 'state-check'
    compile_program(compiler, library + '\n' + state, state_binary)
    result = subprocess.run([str(state_binary)], capture_output=True, timeout=5)
    assert result.returncode == 0 and not result.stdout, ('direct state check', result)
    print('CORE STATE PASS invalid-input output/scratch preservation and repeated calls', flush=True)
    if args.mutants:
        mutations = [
            ('value-corruption', 'put tva dest get tav p', 'put tva dest xor get tav p 1'),
            ('row-corruption', 'put tri dest row', 'put tri dest add row 1'),
            ('missing-first-offset-check', 'if ne get trp 0 0', 'if false'),
        ]
        for name, before, after in mutations:
            assert library.count(before) == 1, ('mutation anchor', name)
            mutant = args.out.resolve() / name
            compile_program(compiler, library.replace(before, after) + '\n' + driver, mutant)
            try:
                exercise(mutant)
            except AssertionError:
                print('MUTANT CAUGHT', name, flush=True)
            else:
                raise AssertionError('surviving mutant: ' + name)


if __name__ == '__main__':
    main()
