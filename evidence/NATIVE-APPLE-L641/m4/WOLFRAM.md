# Independent Wolfram calculations for the G16 profile

Evaluated with WolframLanguageEvaluator, not mental arithmetic:

```wolfram
<|"M4Length" -> Table[{k,2+k(14+14+2+12)+14+2+4},{k,{1,4,64}}],
  "Wait0" -> FromDigits["0406",16],
  "M4MadOffsets" -> {2+28+2,2+28+2+4,2+28+2+8},
  "M4LateWeightOffset" -> {2+3(14+14+2+12),
    2+3(14+14+2+12)+9,2+3(14+14+2+12)+10},
  "M4OffsetEncoding" -> IntegerDigits[BitShiftLeft[16,13],256,6]|>
```

Actual result:

```
M4Length -> {{1,64},{4,190},{64,2710}}
Wait0 -> 1030
M4MadOffsets -> {32,36,40}
M4LateWeightOffset -> {128,137,138}
M4OffsetEncoding -> {0,0,0,2,0,0}
```

The IntegerDigits output is most-significant-byte first; the emitted words are
little-endian. The final-lane mutation changes the last K4 weight offset from
12 to 16, at instruction start128 and stream bytes137/138. Existing independent
M3 field constants and signed-byte numerical bounds remain unchanged.
