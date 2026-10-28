#!/usr/bin/env python3
"""Host acceptance only. Existing C code is the independent verification oracle."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'build/components'
OMC = Path(os.environ.get('OMEGA_OMC', '/home/drakestapleton/workspace/omega-mcp/build/omc'))
ENC = Path(os.environ.get('COMPONENT_ENCODER_DIR', ROOT))
PIN = '2c82480596bbdc50cda9626866209ef662d5834f278f839ff4ce0250c495029f'
BASE = ['bw_alloc.om', 'bw_effect.om', 'bw_check.om', 'bw_project.om']

def command(args, data=None, success=0):
    p = subprocess.run([str(x) for x in args], cwd=ROOT, input=data, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != success:
        raise RuntimeError(f'{args}: exit {p.returncode}, expected {success}: {p.stderr.decode(errors="replace")}')
    return p.stdout

def compile_unit(name, modules, caller, encoder=False):
    parts = [(ROOT / 'om/components' / p).read_bytes() for p in modules]
    if encoder:
        parts += [(ENC / 'om/encoder/bw_enc_sm121.om').read_bytes()]
        spec = (ENC / 'om/encoder/bw_spec_sm121.om').read_bytes()
        marker = b'\nfn main to int\n'
        if spec.count(marker) != 1:
            raise RuntimeError('independent encoder standalone-driver boundary changed')
        parts += [spec.split(marker)[0]]
        parts += [(ROOT / 'om/components/encode.om').read_bytes()]
    parts += [(ROOT / 'tests/components' / caller).read_bytes()]
    source = b'\n'.join(parts)
    (OUT / f'{name}.om').write_bytes(source)
    binary = command([OMC], source)
    path = OUT / name
    path.write_bytes(binary)
    path.chmod(0o755)
    return path

def oracle(name, generator):
    path = OUT / f'{name}-oracle'
    command(['cc', '-std=gnu11', '-O2', '-Wall', '-Wextra', '-Werror', '-DOMEGA_GPU_NO_POSIX',
             '-ffunction-sections', '-fdata-sections', '-Wl,--gc-sections', '-Isrc',
             f'tests/components/{name}-oracle.c', f'src/{generator}', 'src/omega_blackwell_codegen.c', '-o', path])
    return path

def payload(spec, size):
    return struct.pack('<' + 'Q' * size, *(list(spec) + [0] * (size - len(spec))))

def code_from(frame, prefix=0):
    header = struct.unpack_from('<6Q', frame, prefix)
    n, nv, nu = header[:3]
    start = prefix + 48 + (nv + nu) * 16 + n * 64
    return frame[start:]

def main():
    os.chdir(ROOT)
    OUT.mkdir(parents=True, exist_ok=True)
    if hashlib.sha256(OMC.read_bytes()).hexdigest() != PIN:
        raise RuntimeError('compiler pin mismatch')
    dis = shutil.which('nvdisasm')
    if not dis:
        raise RuntimeError('NOT_RUN independent whole-output disassembly: nvdisasm absent')
    unit = compile_unit('component-unit', BASE, 'component-unit.om')
    command([unit])
    root = compile_unit('compose-unit', BASE + ['rowpipe.om','staged.om','compose.om'], 'compose-unit.om')
    command([root])
    direct = compile_unit('direct', BASE, 'direct-caller.om', True)
    rowpipe = compile_unit('rowpipe', BASE + ['rowpipe.om'], 'rowpipe-caller.om', True)
    staged = compile_unit('staged', BASE + ['staged.om'], 'staged-caller.om', True)
    refs = {'direct': oracle('direct', 'omega_blackwell_codegen_moe.c'),
            'rowpipe': oracle('rowpipe', 'omega_blackwell_codegen_moe2.c'),
            'staged': oracle('staged', 'omega_blackwell_codegen_gemm.c')}
    cases = {'direct': [(256,8,1,4,0),(256,8,1,4,1),(512,16,1,4,0),(512,16,1,4,1),
                        (2048,8,2,8,0),(2048,8,2,8,1),(4096,16,4,16,0),(4096,16,4,16,1)],
             'rowpipe': [(256,8,1,1),(768,8,1,5),(512,16,8,16),(4096,16,1,32),
                         (256,8,2,1),(256,8,4,1),(256,8,8,1),(512,16,1,1),(512,16,2,3),
                         (512,16,4,3),(1024,8,1,2),(2048,16,1,4),(1536,16,1,5),(1024,16,2,3)],
             'staged': [(2048,16,4,16,0),(2048,16,8,16,0),(2048,16,4,16,1),
                        (768,8,4,8,0),(1024,16,4,16,0)]}
    binaries = {'direct': direct, 'rowpipe': rowpipe, 'staged': staged}
    report = {'compiler': PIN, 'physical_execution': 'NOT_RUN: no GPU authority',
              'independent_numerical_execution': 'NOT_RUN: unchanged scalar/warp oracle not yet wired to native frames',
              'composer_root': 'PASS direct/changed-direct/rowpipe/dense-staged/mapped-staged and mutant refusal',
              'cases': [], 'negative_controls': []}
    for family, specs in cases.items():
        for index, spec in enumerate(specs):
            data = payload(spec, 16 if family == 'staged' else 8)
            actual = command([binaries[family]], data)
            expected = command([refs[family], *map(str, spec)])
            (OUT / f'{family}-{index}.actual').write_bytes(actual)
            (OUT / f'{family}-{index}.expected').write_bytes(expected)
            if actual != expected:
                raise RuntimeError(f'{family} spec{spec}: whole IR/allocation/code differs; artifacts retained')
            code = code_from(actual, 8 if family == 'staged' else 0)
            kernel = OUT / f'{family}-{index}.bin'
            kernel.write_bytes(code)
            sass = command([dis, '-b', 'SM121', kernel])
            if b'INVALID' in sass.upper() or b'UNKNOWN' in sass.upper():
                raise RuntimeError(f'{kernel}: disassembler rejected an instruction')
            (OUT / f'{family}-{index}.sass').write_bytes(sass)
            report['cases'].append({'family': family, 'spec': spec, 'whole_frame_sha256': hashlib.sha256(actual).hexdigest(),
                                    'code_sha256': hashlib.sha256(code).hexdigest(), 'code_bytes': len(code), 'result': 'PASS'})
        bad = list(specs[0]) + [0] * ((16 if family == 'staged' else 8) - len(specs[0]))
        reserved = 6 if family == 'staged' else 5
        bad[reserved] = 1
        if family == 'staged':
            if command([binaries[family]], payload(bad,16)) != struct.pack('<Q',1):
                raise RuntimeError('staged malformed component was accepted')
        else:
            command([binaries[family]], payload(bad,8), 66)
        report['negative_controls'].append(f'{family}: reserved component refusal PASS')
    reference = command([rowpipe], payload(cases['rowpipe'][0],8))
    for mutation in ['--sb-b','--sb-rd','--branch']:
        broken = command([refs['rowpipe'], *map(str,cases['rowpipe'][0]), mutation])
        if broken == reference:
            raise RuntimeError(f'ineffective oracle mutation {mutation}')
        report['negative_controls'].append(f'rowpipe {mutation}: mismatch detected PASS')
    resource = payload((4096,8,1,16),8)
    if command([rowpipe],resource,67) or command([refs['rowpipe']],resource,66):
        raise RuntimeError('resource refusal leaked output')
    for data in [resource[:-1],resource+b'\0']:
        if command([rowpipe],data,65):
            raise RuntimeError('malformed frame leaked output')
    report['negative_controls'].append('rowpipe native/oracle register exhaustion and malformed frames refuse without output PASS')
    report['negative_controls'].append('native component-unit malformed op/bundle/wait/branch/resource/MMA controls PASS')
    (OUT / 'receipt.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
