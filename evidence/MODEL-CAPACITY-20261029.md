<!-- SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception -->
# Full-width model capacity guards

Host-only, Linux AArch64, 2026-10-29. Fixes #129. No GPU opened.
Base: b348277e, with the independently reviewed component-capacity repair from #173.

The old multiplication-first guards admit the reported full-width dimensions.
The replacement divides the capacity by a known nonzero factor before comparing.
The output guard handles zero columns separately. Positive block length and valid
nonzero K make both scale and element divisors nonzero. Zero-sized output retains
its existing policy and does not write the output array.

## Executed evidence

`OMEGA_OMC=<pinned compiler> make test-model-capacity`

- With original main's mxdot.om and the new fixture: make exits 2;
  full-width-output-product expected 105, actual 0.
- With corrected mxdot.om: exit 0, MODEL CAPACITY PASS.
- Individual original full-width output/left/right cases: all exit 0.
- Corrected full-width output/left/right cases: all exit 105.
- Exact output capacity computes all zero results over sentinel-filled output.
- Zero-left and zero-right shapes return 0 and preserve sentinel output.
- Output, scale and element capacity overflow cases each exit 105.

The fixture calls real mxmm with full-width dimensions, not the older byte-sized
wire driver. Reinstating the original source is the negative control. The guard
and fixture are included in the complete host-test manifest.

## Numeric checks

WolframLanguageEvaluator query:
`With[{m=2^63,n=2,k=32},Mod[{m*n,(k/16)*m,(k/16)*n,(k/2)*m,(k/2)*n},2^64]]`
returned `{0,0,4,0,32}` (dimensionless counts).

Query `2^63` returned `9223372036854775808` (dimensionless count).

Query
`And@@Flatten[Table[Equivalent[a*b<=c,a<=Quotient[c,b]],{a,{0,1,16,256,4096,2^63,2^64-1}},{b,{1,2,16,32,256,4096,2^63}},{c,{256,4096}}]]`
returned `True`. This checks the selected integer examples, not a universal proof.
An attempted universal Resolve query remained unevaluated and is not claimed as proof.

## Tested source identities
8f076133f4ef7a12050943c703bae3845bb31a3886084ee219dcaff38917721d  native/models/mxdot.om
a72b834df231089b42850b078c6f007e838792b1d907a91c1c26dc5e4b3a0dcb  tests/models/mxdot-capacity.om
052f0380f0b137db97ba94fa69d3bde4e46dde6329fd0ed7945a65c6b5b31063  tests/models/capacity.sh
2c82480596bbdc50cda9626866209ef662d5834f278f839ff4ce0250c495029f  pinned-omc
