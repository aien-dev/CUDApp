#!/usr/bin/env python3
import argparse, cmath, math, random, struct, subprocess
from pathlib import Path

def f32(x): return struct.unpack('<f',struct.pack('<f',x))[0]
def dft(x, direction):
    n=len(x)
    return [sum(v*cmath.exp(direction*2j*math.pi*k*j/n) for j,v in enumerate(x)) for k in range(n)]
def run(exe,x,direction):
    data=struct.pack('<Ii',len(x),direction)+b''.join(struct.pack('<ff',z.real,z.imag) for z in x)
    r=subprocess.run([str(exe)],input=data,capture_output=True,timeout=15)
    assert r.returncode==0,(len(x),direction,r)
    assert len(r.stdout)==len(x)*8
    return [complex(*z) for z in struct.iter_unpack('<ff',r.stdout)]
def check(got,want,x):
    err=math.sqrt(sum(abs(a-b)**2 for a,b in zip(got,want)))
    norm=math.sqrt(sum(abs(a)**2 for a in want))
    assert err <= 5e-6*max(norm,1e-35),(err,norm)
    assert max(abs(a-b) for a,b in zip(got,want)) <= 5e-6*max(sum(map(abs,x)),1e-35)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('binary',type=Path); a=ap.parse_args()
    assert a.binary.is_file(),'FFT implementation executable is missing'
    rng=random.Random(428); checks=0
    for n in [1,2,4,8,16,32,64,128,256,512,1024]:
        cases=[[1+0j]+[0j]*(n-1),[1+2j]*n,[complex((-1)**j,0) for j in range(n)]]
        for scale in [1,1e-20,1e20]:
            cases.append([complex(f32(rng.uniform(-1,1)*scale),f32(rng.uniform(-1,1)*scale)) for _ in range(n)])
        for x in cases:
            for direction in [-1,1]:
                got=run(a.binary,x,direction); check(got,dft(x,direction),x); checks+=1
            forward=run(a.binary,x,-1); back=run(a.binary,forward,1)
            check(back,[n*z for z in x],x); checks+=1
    for data in [b'',bytes(7),struct.pack('<Ii',0,-1),struct.pack('<Ii',3,-1)+bytes(24),struct.pack('<Ii',2048,-1),struct.pack('<Ii',2,0)+bytes(16),struct.pack('<Ii',1,-1)+bytes(7),struct.pack('<Ii',1,-1)+bytes(9),struct.pack('<Ii',0xffffffff,-1)]:
        r=subprocess.run([str(a.binary)],input=data,capture_output=True,timeout=5)
        assert r.returncode!=0 and r.stdout==b'',r; checks+=1
    print(f'FFT PASS {checks} checks; C2C f32 N=1..1024, both signs, unnormalized inverse, DFT and refusals')
if __name__=='__main__': main()
