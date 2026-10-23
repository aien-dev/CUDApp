# Independent Wolfram calculations for G15

The WolframLanguageEvaluator was actually called with these exact expressions.
These are calculations, not hardware measurements or a runtime dependency.

```wolfram
vals=<|"loadLo"->FromDigits["0020000000540067",16],"storeLo"->FromDigits["00200000005400e7",16],"loadHi"->FromDigits["004600000011",16],"storeHi"->FromDigits["001180000011",16],"memMask"->(31*2^25+3*2^32+31*2^40),"loadOffMask"->(65535*2^13),"storeOffMask"->(65535*2^11),"madLo"->FromDigits["000000020054109f",16],"madHi"->FromDigits["002a21d0",16],"madMask"->(15*2^25+15*2^42+15*2^51+15*2^60),"waitLong"->FromDigits["0006000600060406",16],"bytesK64"->28+48*64|>; vals
```

Returned:

```
loadLo -> 9007199260246119
storeLo -> 9007199260246247
loadHi -> 300647710737
storeHi -> 75161927697
memMask -> 34098785550336
loadOffMask -> 536862720
storeOffMask -> 134215680
madLo -> 8595443871
madHi -> 2761168
madMask -> 17327665537508966400
waitLong -> 1688875630461958
bytesK64 -> 3100
```

Second call:

```wolfram
<|"golden32"->(FromDigits[#,16]& /@ {"10540067","00200000","0c54109f","60482002"}),"lastLoadStart"->2+3*48,"offset16high"->16*2^13,"lastLoadOffsetBytes"->IntegerDigits[16*2^13,256,6]|>
```

Returned:

```
golden32 -> {273940583, 2097152, 206835871, 1615339522}
lastLoadStart -> 146
offset16high -> 131072
lastLoadOffsetBytes -> {0, 0, 0, 2, 0, 0}
```

The last list is big-endian as requested by IntegerDigits. In the little-endian
instruction, offset16 sets instruction-relative byte10 to 2 and byte9 to zero.
This changes K4's last weight address from row*4+3 to row*4+4. The native gate
checks that lanes0 and1 finish, lane2 faults, and a full dispatch leaves every
public output poisoned.
