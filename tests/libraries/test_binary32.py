#!/usr/bin/env python3
import argparse, math, random, struct, subprocess
from pathlib import Path
def val(x): return struct.unpack('<f',struct.pack('<I',x))[0]
def bits(x):
    try: return struct.unpack('<I',struct.pack('<f',x))[0]
    except OverflowError: return 0xff800000 if x<0 else 0x7f800000
def isnan(x): return (x&0x7fffffff)>0x7f800000
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('binary',type=Path); a=ap.parse_args()
    assert a.binary.is_file(),'binary32 implementation executable is missing'
    edges=[0,0x80000000,1,2,0x7fffff,0x800000,0x800001,0x3f000000,0x3f800000,0x3f800001,0x40000000,0x7f7fffff,0x7f800000,0xff800000,0x7fc00000,0x7f800001,0xbf800000,0x80800000,0x807fffff,0x80000001]
    pairs=[(op,x,y) for op in (0,1) for x in edges for y in edges]
    rng=random.Random(428)
    pairs.extend((op,rng.getrandbits(32),rng.getrandbits(32)) for op in (0,1) for _ in range(20000))
    # Half-ulp ties and just either side, cancellation, underflow rounding.
    pairs.extend((op,x,y) for op in (0,1) for x in range(0x3f7ffffb,0x3f800005) for y in range(0x337ffffb,0x33800005))
    data=b''.join(struct.pack('<III',*p) for p in pairs)
    r=subprocess.run([str(a.binary)],input=data,capture_output=True,timeout=30)
    assert r.returncode==0,(r.returncode,r.stderr)
    assert len(r.stdout)==4*len(pairs),(len(r.stdout),len(pairs))
    for (op,x,y),(got,) in zip(pairs,struct.iter_unpack('<I',r.stdout)):
        want=bits(val(x)+val(y) if op==0 else val(x)*val(y))
        assert (isnan(got) and isnan(want)) or got==want,(op,hex(x),hex(y),hex(got),hex(want))
    for bad in [b'x',bytes(11),struct.pack('<III',2,0,0)]:
        r=subprocess.run([str(a.binary)],input=bad,capture_output=True,timeout=5)
        assert r.returncode!=0 and r.stdout==b''
    print(f'BINARY32 PASS {len(pairs)} arithmetic cases + 3 malformed requests; exact bits except NaN payload')
if __name__=='__main__': main()
