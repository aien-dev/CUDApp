#!/usr/bin/env python3
"""Independent v2/zero-support/stream oracle, outside native construction."""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import random
import select
import struct
import subprocess
import test_rotation as old


def packed(raw, mode):
    out = bytearray(struct.pack('<4sII', b'RQP2', len(raw)//128, mode))
    for start in range(0, len(raw), 128):
        v = [old.units(b) for b in raw[start:start+128]]
        rotate = mode == 1 or (mode == 2 and 0 not in v)
        z = old.transform(v) if rotate else v
        maximum = max(map(abs, z))
        q = [round(Fraction(7*x, maximum)) if maximum else 0 for x in z]
        tag = maximum | ((1 << 31) if mode == 2 and rotate else 0)
        out.extend(struct.pack('<I', tag))
        out.extend(bytes((q[i]+7) | ((q[i+1]+7) << 4) for i in range(0,128,2)))
    return bytes(out)


def reconstructed(frame):
    magic, count, mode = struct.unpack_from('<4sII', frame)
    assert magic == b'RQP2' and mode in (0,1,2) and len(frame) == 12+68*count
    denominator = 3584 if mode == 0 else 458752
    out = bytearray(struct.pack('<4sIII', b'RQR2', count, mode, denominator))
    for b in range(count):
        maximum = struct.unpack_from('<I', frame, 12+68*b)[0]
        rotate = mode == 1 or (mode == 2 and maximum >> 31 == 1)
        if mode == 2:
            maximum &= 0x7fffffff
        q = [n-7 for byte in frame[16+68*b:80+68*b] for n in (byte & 15, byte >> 4)]
        values = old.transform(q) if rotate else q
        factor = 128 if mode == 2 and not rotate else 1
        out.extend(struct.pack('<128q', *(maximum*x*factor for x in values)))
    return bytes(out)


def request(raw, mode):
    return struct.pack('<4sII',b'RQU2',len(raw)//128,mode)+raw


def cases():
    rng = random.Random(49600)
    nonzero = [b for b in old.FINITE if b & 127]
    vectors = [('zero',bytes(128)),('negzero',bytes([128])*128),
               ('nonzero',bytes([1,126,254,255-2])*32),
               ('mixed',bytes([0,126])*64+bytes([126,254])*64),
               ('all-finite',bytes(old.FINITE+[0,128])),
               ('ties',bytes([14,1,3,5,7,9,11,13])*16),
               ('capacity',bytes(old.FINITE[i%254] for i in range(32768))),
               ('capacity-nozero',bytes(nonzero[i%len(nonzero)] for i in range(32768)))]
    for i in range(8):
        raw=bytes(rng.choice(old.FINITE if i%2 else nonzero) for _ in range(128*(i%3+1)))
        vectors.append((f'random{i}',raw))
    for label,raw in vectors:
        for mode in (0,1,2):
            enc=packed(raw,mode); dec=reconstructed(enc)
            if mode==2:
                values=struct.unpack_from('<'+str(len(raw))+'q',dec,16)
                assert all(v==0 for b,v in zip(raw,values) if b & 127 ==0)
            yield label+f'-m{mode}-pack',request(raw,mode),0,enc
            yield label+f'-m{mode}-unpack',enc,0,dec
    for mode in (0,1,2):
        base=packed(bytes([126])*128,mode)
        for maximum in (0xffffffff,29360129 if mode==1 else 229377):
            bad=bytearray(base);struct.pack_into('<I',bad,12,maximum)
            yield f'badmaximum-{mode}-{maximum}',bytes(bad),65,b''
        for code in (127,255):
            raw=bytes([code])+bytes(127)
            yield f'nan-{mode}-{code}',request(raw,mode),65,b''
        bad=bytearray(base);bad[16]=0xff
        yield f'badnibble-{mode}',bytes(bad),65,b''
    # New mode remains invalid under version1; unknown version remains invalid.
    for version in (b'1',b'3'):
        raw=request(bytes(128),2)
        yield 'badversion-'+version.decode(),raw[:3]+version+raw[4:],65,b''
    zero=packed(bytes(128),2)
    for maximum in (1,2147483649):
        bad=bytearray(zero);struct.pack_into('<I',bad,12,maximum)
        yield f'noncanonical-{maximum}',bytes(bad),65,b''
    for data in (request(bytes(128),2),zero):
        for cut in (0,1,11,12,len(data)-1):
            yield f'truncated-{len(data)}-{cut}',data[:cut],65,b''
        yield f'trailing-{len(data)}',data+b'x',65,b''
        for value,offset in [(0,4),(257,4),(0xffffffff,4),(3,8)]:
            bad=bytearray(data);struct.pack_into('<I',bad,offset,value)
            yield f'badfield-{len(data)}-{offset}-{value}',bytes(bad),65,b''


def suite(single, stream=None):
    rows=[]
    def record(label,data,rc,expected,binary):
        p=subprocess.run([str(binary)],input=data,capture_output=True,timeout=30)
        rows.append({'case':label,'pass':p.returncode==rc and p.stdout==expected and not p.stderr,
                     'actual_rc':p.returncode,'expected_rc':rc,'input_sha256':hashlib.sha256(data).hexdigest(),
                     'expected_sha256':hashlib.sha256(expected).hexdigest(),'actual_sha256':hashlib.sha256(p.stdout).hexdigest()})
    for label,data,rc,out in cases():record(label,data,rc,out,single)
    if stream:
        frames=[request(bytes([i])*128,2) for i in (0,1,126,128,254)]
        outputs=[packed(bytes([i])*128,2) for i in (0,1,126,128,254)]
        sequence=[v for pair in zip(frames,outputs) for v in pair]
        expect=[v for frame in outputs for v in (frame,reconstructed(frame))]
        record('stream-repeated',b''.join(sequence),0,b''.join(expect),stream)
        record('stream-empty',b'',0,b'',stream)
        for suffix in (b'x',frames[0][:-1],request(bytes([127])*128,2)):
            record('stream-valid-prefix-failure-'+str(len(suffix)),frames[1]+suffix,65,outputs[1],stream)
        p=subprocess.Popen([str(stream)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        p.stdin.write(frames[1][:20]);p.stdin.flush()
        early=bool(select.select([p.stdout],[],[],0.05)[0])
        p.stdin.write(frames[1][20:]);p.stdin.flush()
        ready=bool(select.select([p.stdout],[],[],5)[0])
        data=p.stdout.read(len(outputs[1])) if ready else b''
        tail,err=p.communicate(frames[2],timeout=15)
        rows.append({'case':'stream-frame-before-eof','pass':not early and ready and data==outputs[1] and tail==outputs[2] and p.returncode==0 and not err})
    return {'total':len(rows),'passed':sum(r['pass'] for r in rows),'failed':sum(not r['pass'] for r in rows),'cases':rows}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('single',type=Path);p.add_argument('--stream',type=Path);p.add_argument('--out',required=True,type=Path);a=p.parse_args()
    result=suite(a.single,a.stream);a.out.write_text(json.dumps(result,indent=2)+'\n')
    print({k:v for k,v in result.items() if k!='cases'});print('FAILURES',[r['case'] for r in result['cases'] if not r['pass']][:8])
    raise SystemExit(bool(result['failed']))
