#!/usr/bin/env python3
"""Host qualification only. Native product logic lives in rotquant.om."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import select
import struct
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'weightpack'))
import qualify as common
import test_rotation as oracle


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def compile_native(compiler, source, target):
    p = subprocess.run([str(compiler)], input=source, capture_output=True, timeout=30)
    require(p.returncode == 0 and not p.stderr, repr((p.returncode, p.stderr)))
    # Static little-endian ELF64 AArch64 EXEC from the fingerprinted seed.
    require(len(p.stdout) >= 64 and p.stdout[:6] == b'\x7fELF\x02\x01', 'invalid ELF64')
    require(struct.unpack_from('<HH', p.stdout, 16) == (2, 183), 'wrong ELF target/type')
    target.write_bytes(p.stdout)
    target.chmod(0o700)
    return target


def io_controls(binary):
    raw = bytes([14, 1, 3, 5]*32)
    data = oracle.request(raw, 1)
    expected = oracle.encode(raw, 1)
    rows = []
    for name, fd in [('closed-read', 0), ('closed-write', 1)]:
        p = subprocess.run([str(binary)], input=data, capture_output=True,
                           preexec_fn=lambda: os.close(fd), timeout=10)
        rows.append({'case': name, 'rc': p.returncode,
                     'pass': p.returncode == 74 and not p.stdout and not p.stderr})
    with open('/dev/full', 'wb') as full:
        p = subprocess.run([str(binary)], input=data, stdout=full, stderr=subprocess.PIPE, timeout=10)
        rows.append({'case': 'full-output-device', 'rc': p.returncode, 'pass': p.returncode == 74 and not p.stderr})
    p = subprocess.Popen([str(binary)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p.stdin.write(data[:17])
    p.stdin.flush()
    early = bool(select.select([p.stdout], [], [], 0.03)[0])
    output, err = p.communicate(data[17:], timeout=10)
    rows.append({'case': 'fragmented-input-no-early-output', 'rc': p.returncode,
                 'pass': not early and p.returncode == 0 and output == expected and not err})
    require(all(r['pass'] for r in rows), repr(rows))
    return rows


def mutants(compiler, source, build, evidence):
    mutations = [('lose-fp8-sign', b'v = neg v\n', b'v = v\n'),
                 ('wrong-inverse-normalization', b'denom = 458752\n', b'denom = 3584\n'),
                 ('ties-away-instead-of-even', b'if or gt twice m and eq twice m ne rem q 2 0\n', b'if ge twice m\n')]
    rows = []
    for name, before, after in mutations:
        require(source.count(before) == 1, name+' mutation not unique')
        changed = source.replace(before, after)
        binary = compile_native(compiler, changed, build/('mutant-'+name))
        result = oracle.suite(binary)
        write_json(evidence/('mutant-'+name+'.json'), result)
        require(result['failed'] > 0, name+' escaped')
        rows.append({'name': name, 'source_sha256': sha(changed), 'binary_sha256': sha(binary.read_bytes()),
                     'caught': result['failed'], 'total': result['total'],
                     'change_before': before.decode(), 'change_after': after.decode()})
    return rows


def qualify_stream(compiler, source, stream_source, build, evidence):
    import test_stream
    native_source = source.replace(b'fn main to int\n', b'fn single_main to int\n')+stream_source
    require(source.count(b'fn main to int\n') == 1, 'main must be unique')
    stream = compile_native(compiler, native_source, build/'rotquant-stream')
    result = test_stream.suite(build/'rotquant', stream)
    write_json(evidence/'v2-stream-suite.json', result)
    require(result['failed'] == 0, 'v2/stream suite failed')
    capacity = []
    raw = test_stream.request(bytes(128), 0)
    expected = test_stream.packed(bytes(128), 0)*999999
    for count, status in [(999999, 0), (1000000, 65)]:
        p = subprocess.run([str(stream)], input=raw*count, capture_output=True, timeout=60)
        row = {'transactions': count, 'actual_rc': p.returncode, 'expected_rc': status,
               'actual_sha256': sha(p.stdout), 'expected_sha256': sha(expected),
               'actual_bytes': len(p.stdout), 'expected_bytes': len(expected),
               'pass': p.returncode == status and p.stdout == expected and not p.stderr}
        capacity.append(row)
    write_json(evidence/'stream-capacity.json', capacity)
    require(all(row['pass'] for row in capacity), 'stream boundary failure')
    mutations = [('lost-selector', b'tag = add tag 2147483648\n', b'tag = tag\n'),
                 ('lost-identity-normalization', b'm = mul m 128\n', b'm = m\n'),
                 ('rotate-through-zero', b'if eq get vec i 0\n', b'if eq get vec i 99999999\n')]
    caught = []
    for name, before, after in mutations:
        require(source.count(before) == 1, name+' mutation not unique')
        mutant = compile_native(compiler, source.replace(before, after), build/('mutant-'+name))
        verdict = test_stream.suite(mutant)
        write_json(evidence/('mutant-'+name+'.json'), verdict)
        require(verdict['failed'] > 0, name+' escaped')
        caught.append({'name': name, 'caught': verdict['failed'], 'total': verdict['total'],
                       'binary_sha256': sha(mutant.read_bytes())})
    return {'source_sha256': sha(native_source), 'binary_sha256': sha(stream.read_bytes()),
            'suite_total': result['total'], 'suite_passed': result['passed'],
            'capacity_cases': capacity, 'io_controls': io_controls(stream), 'mutants': caught}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--compiler', type=Path)
    parser.add_argument('--invalidate-only', action='store_true')
    parser.add_argument('--build', required=True, type=Path)
    parser.add_argument('--evidence', required=True, type=Path)
    parser.add_argument('--model', type=Path)
    args = parser.parse_args()
    evidence = common.retire_current_receipt(args.evidence)
    if args.invalidate_only:
        return
    # Independent Python references use asserts: never let -O disable qualification.
    require(__debug__ and sys.flags.optimize == 0, 'Python optimization is forbidden for qualification')
    build = common.safe_directory(args.build)
    require(args.compiler is not None, '--compiler required')
    compiler = args.compiler.resolve()
    require(sha(compiler.read_bytes()) == common.COMPILER_SHA, 'compiler fingerprint mismatch')
    build.mkdir(parents=True, exist_ok=True)
    build = build.resolve()
    here = Path(__file__).resolve().parent
    source = (here/'rotquant.om').read_bytes()
    binary = compile_native(compiler, source, build/'rotquant')
    result = oracle.suite(binary)
    write_json(evidence/'qualified-suite.json', result)
    require(result['failed'] == 0, 'native qualification failed')
    summary = {'scope': 'offline approximate representation; no inference/GPU qualification',
               'compiler_sha256': common.COMPILER_SHA, 'source_sha256': sha(source),
               'binary_sha256': sha(binary.read_bytes()), 'suite_total': result['total'],
               'suite_passed': result['passed'], 'io_controls': io_controls(binary),
               'mutants': mutants(compiler, source, build, evidence)}
    summary['stream'] = qualify_stream(compiler, source, (here/'stream.om').read_bytes(), build, evidence)
    dependency_paths = list(here.glob('*.py')) + [here/'rotquant.om', here/'stream.om', here/'CONTRACT.md',
                        here.parent/'weightpack/qualify.py', here.parent/'weightpack/test_weightpack.py',
                        here.parent/'weightpack/weightpack.om', here.parents[1]/'mk/native-rotquant.mk']
    summary['source_closure'] = {str(p.relative_to(here.parents[1])): sha(p.read_bytes()) for p in sorted(dependency_paths)}
    if args.model:
        import experiment
        # Rebuild the exact comparator rather than accepting a caller's executable.
        lossless = compile_native(compiler, (here.parent/'weightpack/weightpack.om').read_bytes(), build/'weightpack-comparator')
        comparison = experiment.experiment(args.model.resolve(), binary, lossless)
        write_json(evidence/'matrix-comparison.json', comparison)
        summary['real_matrix_count'] = len(comparison['matrices'])
        summary['comparison_sha256'] = sha((evidence/'matrix-comparison.json').read_bytes())
    write_json(evidence/'qualification.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
