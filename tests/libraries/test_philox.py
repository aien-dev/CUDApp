#!/usr/bin/env python3
"""Independent Philox oracle; published Random123 KATs pin its round semantics."""
import argparse, random, struct, subprocess
from pathlib import Path
MASK=(1<<32)-1
KATS=[([0]*4,[0]*2,[0x6627e8d5,0xe169c58d,0xbc57ac4c,0x9b00dbd8]),([MASK]*4,[MASK]*2,[0x408f276d,0x41c83b0e,0xa20bc7c6,0x6d5451fd]),([0x243f6a88,0x85a308d3,0x13198a2e,0x03707344],[0xa4093822,0x299f31d0],[0xd16cfe09,0x94fdcceb,0x5001e420,0x24126ea1])]
def ref(c,k):
    c=list(c); k=list(k)
    for _ in range(10):
        p0=0xd2511f53*c[0]; p1=0xcd9e8d57*c[2]
        c=[((p1>>32)^c[1]^k[0])&MASK,p1&MASK,((p0>>32)^c[3]^k[1])&MASK,p0&MASK]
        k=[(k[0]+0x9e3779b9)&MASK,(k[1]+0xbb67ae85)&MASK]
    return c

def run(exe, data):
    return subprocess.run([str(exe)],input=data,capture_output=True,timeout=10)
def request(mode,a,b,c,d): return struct.pack('<IIQQQQ',mode,0,a,b,c,d)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('binary',type=Path); args=ap.parse_args()
    assert args.binary.is_file(), 'Philox implementation executable is missing'
    count=0
    for c,k,want in KATS:
        assert ref(c,k)==want
        r=run(args.binary,request(0,c[0]|c[1]<<32,c[2]|c[3]<<32,k[0]|k[1]<<32,0))
        assert r.returncode==0 and r.stdout==struct.pack('<4I',*want),(c,r)
        count+=1
    rng=random.Random(428)
    for _ in range(200):
        c=[rng.getrandbits(32) for _ in range(4)]; k=[rng.getrandbits(32) for _ in range(2)]
        r=run(args.binary,request(0,c[0]|c[1]<<32,c[2]|c[3]<<32,k[0]|k[1]<<32,0))
        assert r.returncode==0 and r.stdout==struct.pack('<4I',*ref(c,k)),(c,r)
        count+=1
    for seed in [0,1,0xffffffffffffffff,0x123456789abcdef0]:
        for sub in [0,1,0xffffffffffffffff]:
            for off in [0,1,3,4,7,(1<<32)-2,(1<<34)-3,(1<<64)-130]:
                n=129; want=[]
                for i in range(n):
                    block=(off+i)//4
                    want.append(ref([block&MASK,block>>32,sub&MASK,sub>>32],[seed&MASK,seed>>32])[(off+i)%4])
                r=run(args.binary,request(1,seed,sub,off,n))
                assert r.returncode==0 and r.stdout==struct.pack('<%dI'%n,*want),(seed,sub,off,r)
                # Split calls must yield the same bytes, including partial blocks.
                a=run(args.binary,request(1,seed,sub,off,17)); b=run(args.binary,request(1,seed,sub,off+17,n-17))
                assert a.returncode==b.returncode==0 and a.stdout+b.stdout==r.stdout
                count+=2
    for n in [0,1,65536]:
        r=run(args.binary,request(1,7,9,0,n)); assert r.returncode==0 and len(r.stdout)==n*4; count+=1
    for data in [b'',bytes(39),bytes(41),request(2,0,0,0,0),request(1,0,0,0,65537),request(1,0,0,(1<<64)-1,2),struct.pack('<IIQQQQ',1,1,0,0,0,0)]:
        r=run(args.binary,data); assert r.returncode!=0 and r.stdout==b'',(len(data),r); count+=1
    print(f'PHILOX PASS {count} checks (published KATs, random counters, offset/carry, split calls, bounds)')
if __name__=='__main__': main()
