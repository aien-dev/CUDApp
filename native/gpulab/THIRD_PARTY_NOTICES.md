# Reference sources and notices

All `.om` implementations in this directory were handwritten for this lab. No
prior C simulator/encoder, Python generator, Metal bridge or inference engine is
compiled, linked, invoked as the implementation, or translated into Omega.

## AMD instruction oracle

LLVM `llvm-mc-18 -triple=amdgcn -mcpu=gfx1100` is an independently installed
validation tool. `amd-oracle-input.om` emits assembly names, not binary words;
LLVM's `.text` is compared byte-for-byte against our generated shader by
`test-amd-oracle.om`. LLVM is not on the emitted-program or simulator execution
path. Its version and raw output are retained in the receipts. LLVM code is not
vendored here.

## Apple instruction-layout reference

- Mesa `src/asahi/isa/AGX2.xml` at
  `ba3dacb9832bd073198d22c402f4e0d75ec3a09a`.
- Existing AIEN AGX reference at aien-dev/aienux
  `4b11bb96426cca4969bf77c9cde97b87d97f6b8c`:
  `native/gpu/agx/agx_isa_tables.c`,
  `native/gpu/agx/tests/fixtures/corpus.dis` and its recorded Apple binaries.
- Read-only field references: ldimm32, iadd, imul, load/store, stop/wait;
  dalu64, add/mul operand encodings, memory/membase/memindex.

The frozen byte fixtures in the Apple gate are the numeric interpretation of
these recorded bytes (little endian):

```
620500000000          ldimm r1, 0
0e1501c024000000      iadd r5, 1, r6
9e09c4b200000000      imul $r2, ^r2, 11
0509200e00c01200      load r1, du0, r1, i32, x, a
4509040e00c01200      store r1, du2, r0, i32, x, a
```

Hinted operands in the multiply fixture check bit placement in the encoder;
the simulator deliberately supports only the plain-register arithmetic subset.
The reference tables were inspected as documentation, not executed by this task.
The earlier broader C simulation/real-Metal evidence is historical and is not
presented as execution of this new Omega code.

For the AGX field descriptions derived from Mesa:

Copyright 2025 Valve Corporation

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
