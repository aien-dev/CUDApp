# Native lossless weight packing: first implementation slice

Refs CUDApp #48; related #49/#50; research aien-architecture-mv #297; ledger L600.
Personally implemented and reviewed in this Hermes chat. No subagent or worker.
Base: 232332422bfdc31c8918085b50f13c1e18ab702e. Branch feat/zero-channel-native.
No GPU, driver, model-serving or numerical-execution change. NOT production integration.

## Verified results
- Native codec: 464/464 deterministic cases pass.
- Four real I/O controls pass. Missing and wrong compilers are refused.
- Three compiled mutants caught: 85, 161, 2 failing cases respectively.
- Fresh source-only seed bootstrap reproduces the identical compiler and codec.
- Existing `quiet-build make -j2 test`: exit 0, CUDA++ HOST TESTS PASS (33 targets).
- Header-only census: 30,720 routed expert matrices across 40 layers, matching config-derived expectation. No full payload census yet.
- Five frozen real matrices encode identically to the separate Python oracle and reconstruct every byte. No model weights retained in this evidence.

| Layer/expert/projection | Zero channels | Raw weight bytes | Packed bytes incl. frame | Saved bytes |
|---|---:|---:|---:|---:|
| 0/0/gate_proj | 280 | 1,048,576 | 547,348 | 501,228 |
| 0/0/up_proj | 280 | 1,048,576 | 547,348 | 501,228 |
| 0/0/down_proj | 280 | 1,048,576 | 547,348 | 501,228 |
| 20/128/up_proj | 0 | 1,048,576 | 1,048,596 | -20 |
| 39/255/down_proj | 0 | 1,048,576 | 1,048,596 | -20 |

For the first expert's three matrices, 3,145,728 raw bytes become 1,642,044
packed bytes: 47.800827% fewer bytes including framing and negative-zero sign
masks. With the unchanged 384 scale bytes counted on both sides, 47.794993%.
This is a five-matrix selected replay, not whole-model compression prevalence.
The two non-sparse examples expand by the 20-byte frame header; a future loader
must keep its existing unframed dense representation when total cost does not win.
No model-quality, token-speed, memory-traffic or energy result is claimed.

## Pins / replay
- Product source SHA256: `9b841b643de67b001da4f1e7576e30e46e9b827a39166b7179460d249dce727c`.
- Compiler SHA256: `191b776ef98821597c3e30655da6d680a9b66edbc9ceaca3c39c61ea6df098e9`.
- Native binary SHA256: `939fef5d9ee207a9a5e504e599f735f756022010741640040b1a99bef374586f`.
- Frozen bootstrap package: omega-bridge d2010ab927b608931800c20d73a4c2b25f00a9fa.
- Compiler source origin: omega-mv 29b12760f4c962606411d6bf25032aedef063301.

See the [format and replay instructions](../../native/weightpack/CONTRACT.md).
The compiler is an explicit external prerequisite, not an implicit download or
new C backend. `make test-native-weightpack OMEGA_NATIVE_COMPILER=/path/to/omc`
runs the native build and independent controls. `qualify.py --model ...` adds
metadata census and the fixed five-matrix replay. Receipts include source/model,
header, tensor, scale, binary and reconstruction hashes.

## Retained failures and repairs
`red-empty.json`: 464/464 cases failed before implementation.
`compile*.log`: register-pressure refusals before splitting bounded helpers.
`compile-root-cause.json`: minimal syntax probe; the operator was valid, the
compiler's sixteen-register allocation limit was the actual compile refusal.
`first-native.json`: 462/464 passed; two maximum-sized dense copies hit the
runtime loop limit. `loop-bound-value-probes.json` distinguishes literal loops
from dynamically bounded loops. Chunked copies and explicit I/O ceilings repair
that limit; all 464 cases subsequently pass. No gates were skipped or waived.

## Scope still open
Full checkpoint payload inventory, integrated container/loader accounting,
load/decode cost, model-level exactness and coordinated GPU/runtime qualification.
Rotation-aware approximation and targeted factorization have issues but are not
implemented here. Review was personal, not independent. No merge or deployment.
