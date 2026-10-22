#!/usr/bin/env python3
"""Execute value-level negative controls; compile success is mandatory."""
import argparse, importlib.util, subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('libbuild',ROOT/'tools/libraries/build.py')
build=importlib.util.module_from_spec(spec); spec.loader.exec_module(build)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('out',type=Path); a=ap.parse_args(); out=a.out.resolve()
    mutants=[('philox','for r ulong 10','for r ulong 9','round-count'),
             ('binary32','if or gt r 4 and eq r 4 ne and v 1 0','if or gt r 5 and eq r 4 ne and v 1 0','rounding'),
             ('fft','if inv\nwi = xor wi 2147483648','if not inv\nwi = xor wi 2147483648','fft-sign'),
             ('sparse','acc = fadd acc fmul get av p get xv j','acc = fadd acc get av p','sparse-scale')]
    for name,old,new,label in mutants:
        src='\n'.join((ROOT/p).read_text() for p in build.PARTS[name]); assert src.count(old)==1
        exe=build.compile_source(out/'compiler/omc',src.replace(old,new),out/('mutant-'+label))
        p=subprocess.run(['python3',str(ROOT/'tests/libraries'/('test_'+name+'.py')),str(exe)],capture_output=True,text=True,timeout=90)
        (out/(label+'.log')).write_text(p.stdout+p.stderr)
        assert p.returncode==1 and 'AssertionError' in p.stderr,(label,p.returncode,p.stdout,p.stderr)
        # These tests distinguish wrong output from process failure by the diagnostic.
        assert 'returncode=' not in p.stderr or 'returncode=0' in p.stderr, p.stderr
        print('MUTANT CAUGHT',label,'compiled and failed value comparison')
if __name__=='__main__': main()
