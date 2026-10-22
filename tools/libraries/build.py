#!/usr/bin/env python3
"""Build the host-native Omega library tests using a pinned seed-born omc.
Requires Linux AArch64. No C compiler, assembler, linker or CUDA tool is invoked.
"""
import argparse, hashlib, io, platform, subprocess, tarfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
PIN='f623467e21f9ea8a862a7c92965227fb67263e8e'
EXPECTED='10d7565239ca6bd3cca344b65a03c0f3498acd95e82ab3d420bd01408b01398e'
PARTS={'philox':['omega/libraries/philox.om','tests/libraries/philox_driver.om'],
       'binary32':['omega/libraries/binary32.om','tests/libraries/binary32_driver.om'],
       'fft':['omega/libraries/binary32.om','omega/libraries/twiddle1024.om','omega/libraries/fft.om','tests/libraries/fft_driver.om'],
       'sparse':['omega/libraries/binary32.om','omega/libraries/sparse.om','tests/libraries/sparse_driver.om'],
       'arith':['omega/libraries/binary32.om','omega/libraries/fdiv.om','omega/libraries/fsqrt.om','tests/libraries/arith_check.om'],
       'potrf':['omega/libraries/binary32.om','omega/libraries/fdiv.om','omega/libraries/fsqrt.om','omega/libraries/potrf.om','tests/libraries/potrf_check.om'],
       'npp':['omega/libraries/binary32.om','omega/libraries/npp.om','tests/libraries/npp_check.om'],
       'blas1':['omega/libraries/binary32.om','omega/libraries/fdiv.om','omega/libraries/fsqrt.om','omega/libraries/blas1.om','tests/libraries/blas1_check.om'],
       'idct':['omega/libraries/binary32.om','omega/libraries/idct.om','tests/libraries/idct_check.om'],
       'solvers':['omega/libraries/binary32.om','omega/libraries/fdiv.om','omega/libraries/fsqrt.om','omega/libraries/potrf.om','omega/libraries/solvers.om','tests/libraries/solvers_check.om'],
       'more':['omega/libraries/binary32.om','omega/libraries/fdiv.om','omega/libraries/fsqrt.om','omega/libraries/more.om','tests/libraries/more_check.om']}
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def run(args, **kw): return subprocess.run([str(x) for x in args],check=True,**kw)
def compile_source(compiler,src,target):
    p=run([compiler],input=src.encode(),capture_output=True)
    target.write_bytes(p.stdout); target.chmod(0o755)
    return target

def bootstrap(repo,out):
    out.mkdir(parents=True,exist_ok=True)
    # Archive the immutable commit, never consume the other worker's dirty checkout.
    raw=subprocess.check_output(['git','-C',str(repo),'archive',PIN,'seed'])
    with tarfile.open(fileobj=io.BytesIO(raw)) as tar:
        # Only regular files/directories inside seed, no symlinks from the source archive.
        for m in tar.getmembers():
            if not (m.name=='seed' or m.name.startswith('seed/')) or '..' in Path(m.name).parts or not (m.isfile() or m.isdir()):
                raise ValueError('unexpected seed archive member: '+m.name)
        tar.extractall(out,filter='data')
    seed=out/'seed'
    def exe(name,compiler,inputs):
        target=out/name
        p=run([compiler],input=b''.join(f.read_bytes() for f in inputs),capture_output=True)
        target.write_bytes(p.stdout); target.chmod(0o755); return target
    def same(a,b):
        if a.read_bytes()!=b.read_bytes(): raise AssertionError(f'seed mismatch: {a} != {b}')
    loader=out/'seed0.load'
    run(['sh',seed/'rung0/load0.sh',seed/'rung0/seed0.hex',loader]); loader.chmod(0o755)
    s0=exe('seed0',loader,[seed/'rung0/seed0.hex']); same(s0,seed/'rung0/seed0.bin')
    h1=exe('hex1',s0,[seed/'rung1/hex1.hex']); same(h1,seed/'rung1/hex1.bin')
    a2=exe('asm2',h1,[seed/'rung2/asm2.hex1']); same(a2,seed/'rung2/asm2.bin')
    o4=exe('om4',a2,[seed/p for p in ['rung2/table.s2','rung3/table3.s2','rung4/table4.s2','rung4/om4.s']]); same(o4,seed/'rung4/om4.bin')
    stage1=exe('stage1',o4,[seed/'omc/omc.om'])
    omc=exe('omc',stage1,[seed/'omc/omc.om'])
    stage3=exe('stage3',omc,[seed/'omc/omc.om']); same(omc,stage3)
    assert sha(omc)==EXPECTED, 'pinned compiler hash mismatch'
    print('SEED PASS',PIN,'omc',sha(omc),flush=True)
    return omc

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--omega-source',type=Path,required=True)
    ap.add_argument('--out',type=Path,default=ROOT/'build/fft-rng-native'); a=ap.parse_args()
    if platform.system()!='Linux' or platform.machine() not in ('aarch64','arm64'):
        ap.error('the pinned seed compiler emits Linux AArch64 programs')
    out=a.out.resolve(); omc=bootstrap(a.omega_source.resolve(),out/'compiler')
    for name,parts in PARTS.items():
        src='\n'.join((ROOT/p).read_text() for p in parts)
        target=compile_source(omc,src,out/name)
        print('BUILD',name,'bytes',target.stat().st_size,'sha256',sha(target),flush=True)
if __name__=='__main__': main()
