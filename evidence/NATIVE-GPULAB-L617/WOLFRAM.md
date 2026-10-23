# Wolfram calculations used by the native GPU lab

These are actual WolframLanguageEvaluator inputs and returned values from this
session, not estimates. Wolfram is external validation only, never a runtime
dependency. The Omega tests embed the resulting independent literals.

## Signed dot-product bounds and fixtures

```wolfram
64*128*128
{-128,127,-1,0,1,17,-23,64}.{127,-128,-1,1,0,-19,29,64}
Mod[{-128,127,-1,0,1,17,-23,64}.{127,-128,-1,1,0,-19,29,64},2^32]
{64*(-128)*(-128), Mod[64*127*(-128),2^32],64*127*127}
```

Returned: `1048576`, `-29405`, `4294937891`,
`{1048576,4293926912,1032256}` respectively.

## AMD instruction-word interpretation

```wolfram
Map[FromDigits[#,16]&, {"7e000200","4a000000","d72c0000",
 "dc520000","dc6a0000","bf8903f7","bfb00000","bf800000",
 "fe000000","ffffe000","ffffff00","0003ffff"}]
```

Returned:
`{2113929728,1241513984,3609985024,3696361472,3697934336,3213427703,3215982592,3212836864,4261412864,4294959104,4294967040,262143}`.

LLVM byte-list conversion:

```wolfram
FromDigits[Reverse[#],256]& /@
 {{128,2,12,126},{7,0,44,215},{8,19,2,0},{6,15,12,74},
  {4,0,82,220},{0,0,124,8},{247,3,137,191},{0,0,106,220},
  {4,6,124,0},{0,0,176,191}}
```

Returned:
`{2114716288,3609985031,135944,1242304262,3696361476,142344192,3213427703,3697934336,8128004,3215982592}`.

## Apple independent byte fixtures and field calculations

```wolfram
FromDigits[Reverse[FromDigits[#,16]& /@ StringPartition[#,2]],256]& /@
 {"620500000000","0e1501c024000000","9e09c4b200000000",
  "0509200e00c01200","4509040e00c01200"}
```

Returned:
`{1378,157840119054,2999191966,5277656050305285,5277656048470341}`.

```wolfram
{14+BitShiftLeft[42,7]+BitShiftLeft[1,16]+BitShiftLeft[588,28],
 30+BitShiftLeft[19,7]+BitShiftLeft[708,16]+BitShiftLeft[11,28]}
```

Returned `{157840119054,2999191966}`, agreeing with the recorded instruction bytes.

```wolfram
{FromDigits["0012c0000e000105",16],FromDigits["0012c0000e000145",16]}
BitOr[BitShiftLeft[31,10],BitShiftLeft[2047,16],BitShiftLeft[2047,28]]
BitOr[BitShiftLeft[31,10],BitShiftLeft[2^32-1,16]]
BitOr[BitShiftLeft[31,11],BitShiftLeft[15,16],BitShiftLeft[7,21],BitShiftLeft[3,32]]
```

Returned `{5277656048206085,5277656048206149}`,
`549621562368`, `281474976676864`, `12900628480`.

## Byte capacity and wire-integration fixtures

```wolfram
{4+64*(8+8+4+8+4)+8+4+4,18+64*50}
{32*64,2048+64,3072+32}
Table[Table[r+k,{k,0,3}].{1,2,3,4},{r,0,2}]
354+6*2^10
{20+4*32,18+4*50}
```

Returned `{2068,3218}`, `{2048,2112,3104}`, `{20,30,40}`, `6498`,
`{148,218}`. Both generated maximum-depth programs fit the 4096-byte code buffer.
These are capacities and exact arithmetic, **not throughput calculations**.
