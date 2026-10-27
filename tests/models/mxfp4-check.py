#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Verification-only oracle; never part of the runtime or installed engine."""
import argparse
import hashlib
import math
from pathlib import Path
import struct
import subprocess
import tempfile

PIN = "189ada82e00bb440cb41351bd252cc68c54a1b1050051cc42a1df700081c3b54"
# OpenAI gpt_oss/torch/weights.py FP4_VALUES. ldexp models its torch.ldexp;
# struct packs IEEE binary32 once, independently of the candidate bit operations.
VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
          -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)
ROOT = Path(__file__).resolve().parents[2]


def build(compiler, source, target):
    result = subprocess.run([compiler], input=source, capture_output=True, timeout=60)
    if result.returncode:
        raise RuntimeError(result.stderr.decode())
    target.write_bytes(result.stdout)
    target.chmod(0o700)
    return str(target)


def oracle(packed, lane, scale):
    value = math.ldexp(VALUES[(packed >> (lane * 4)) & 15], scale - 127)
    try:
        return struct.pack('<f', value)
    except OverflowError:
        return struct.pack('<f', math.copysign(math.inf, value))


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('compiler', type=Path)
    args = parser.parse_args()
    compiler = str(args.compiler.resolve())
    digest = hashlib.sha256(Path(compiler).read_bytes()).hexdigest()
    if digest != PIN:
        raise RuntimeError(f'compiler pin mismatch: {digest}')
    unit = (ROOT / 'native/models/mxfp4.om').read_bytes()
    driver = (ROOT / 'tests/models/mxfp4-driver.om').read_bytes()
    records = [(packed, lane, scale) for scale in range(255)
               for packed in range(256) for lane in range(2)]
    data = bytes(value for record in records for value in record)
    expected = b''.join(oracle(*record) for record in records)
    with tempfile.TemporaryDirectory(prefix='mxfp4-check-') as tmp:
        exe = build(compiler, unit + driver, Path(tmp) / 'candidate')
        result = subprocess.run([exe], input=data, capture_output=True, timeout=90)
        if result.returncode or result.stdout != expected:
            raise RuntimeError(f'candidate mismatch: rc={result.returncode}, bytes={len(result.stdout)}')
        for bad, status in [(b'\x00\x02\x7f', 105), (b'\x00\x00\xff', 105), (b'\x00', 3)]:
            r = subprocess.run([exe], input=bad, capture_output=True, timeout=10)
            if r.returncode != status or r.stdout:
                raise RuntimeError(f'refusal mismatch: {r.returncode}, {r.stdout!r}')
        for call in ['mxfp4bits 16 127', 'mxfp4bits 0 256', 'mxfp4byte 256 0 127']:
            main = ('fn main to int\nx = ' + call + '\nret int x\nend\n').encode()
            negative = build(compiler, unit + main, Path(tmp) / 'negative')
            r = subprocess.run([negative], capture_output=True, timeout=10)
            if r.returncode != 105:
                raise RuntimeError(f'out-of-range accepted: {call}')
        changed = unit.replace(b'sign = shl and nib 8 28', b'sign = uint 0')
        if changed == unit:
            raise RuntimeError('mutant was not applied')
        mutant = build(compiler, changed + driver, Path(tmp) / 'mutant')
        r = subprocess.run([mutant], input=data, capture_output=True, timeout=90)
        if r.returncode or len(r.stdout) != len(expected) or r.stdout == expected:
            raise RuntimeError('sign mutant did not run to a full mismatching result')
        print(f'compiler_sha256={digest}')
        print(f'source_sha256={hashlib.sha256(unit).hexdigest()}')
        print(f'candidate_sha256={hashlib.sha256(Path(exe).read_bytes()).hexdigest()}')
        print(f'oracle_sha256={hashlib.sha256(expected).hexdigest()}')
        print(f'MXFP4 native CPU exact matches={len(records)}; refusals=6; sign mutant rejected')
        print('NVIDIA/Intel/AMD GPU: NOT_RUN; model logits: NOT_RUN')


if __name__ == '__main__':
    run()
