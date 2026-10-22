# Native weight representation development (L600)

Owner: this Hermes chat personally. No delegated implementation or review.
Issues: CUDApp #48 exact; #49 approximate rotation; #50 conditional factorization.
Research: aien-architecture-mv #297. Existing #16/#17 define downstream gates.

## Decision and staged scope
Start with reversible storage, not arithmetic skipping. A bounded native Omega
codec accepts framed FP8 matrix bytes; Python only supplies fixtures, metadata
I/O and an independent oracle. No changes to C kernels, compiler or serving.
Use a fresh CUDApp worktree at 232332422bfdc31c8918085b50f13c1e18ab702e.
Freeze omega-bridge d2010ab927b608931800c20d73a4c2b25f00a9fa, rebuild its seed,
and require compiler SHA256 191b776ef98821597c3e30655da6d680a9b66edbc9ceaca3c39c61ea6df098e9.

Alternatives: instruction skipping is deferred because zeros do not establish
nonfinite/signed-zero execution equivalence. Blanket factorization has poor
measured reconstruction. Rotation is explicitly approximate and not mixed in.

## First slice protocol / architecture
One request per process through stdin/stdout. No model paths inside the product.
All header integers little-endian u32. Matrix data are row-major bytes.
ZCR1: magic, rows, columns, axis (0=row channels, 1=column channels), raw bytes.
ZCP1: same fields plus mode (0=dense, 1=channel encoding), payload.
Rows/columns 1..4096, product <=1048576. Encoded payload <= dense payload.
A sparse channel tag is 0=all positive zero; 1=zero magnitudes plus packed sign
bits (LSB-first, zero unused bits, at least one negative zero); 2=raw channel
with at least one nonzero magnitude. Dense mode wins ties and expansion cases.
No reordering of original row-major output. Every one of the 256 byte patterns
is legal and losslessly retained, including both NaN encodings. Scales are not
part of this byte-only format and must be retained unchanged by later loaders.
Malformed/truncated/trailing/overlimit input: exit65 and no successful output.
Read/write transport failure: exit74; partial writes may exist on transport
failure, so orchestration must publish only after successful exit.
This framing is not an authenticated container: arbitrary valid payload changes
need an external content hash. No CRC is presented as corruption-proofing.

## Execution plan (TDD)
1. Write independent reference encoder/decoder and fixture suite first. Run the
   suite against a deliberately empty executable to record expected failures.
2. Build seeded native codec. Require byte-for-byte encode equality and exact
   decode reconstruction for all deterministic fixtures. Add limit/orientation,
   signed-zero, all-byte, truncation, canonical form and real I/O controls.
3. Compile planted mutants (signed-zero loss, wrong orientation, trailing-input
   acceptance) and require the suite to catch each. Preserve initial failures.
4. Replay fixed checkpoint matrices, recording source/tensor offsets/hashes,
   encoded/decoded hashes and full framing overhead. No retained model weights.
5. Rebuild the compiler from frozen source and repeat qualification; verify
   source tree/binary identity and repository host regressions where practical.
6. Publish source+raw receipts, verify exact remote commit and update #48. Keep
   all three issues OPEN. No delegated implementation or review is launched;
   integration remains pending under the existing PR ownership policy.

## Later work, not acceptance for the first slice
Full manifest-driven checkpoint scan with exact totals and channel alignment;
loader integration and exact model-output/runtime checks (#48); calibrated
native rotation quantizer with held-out model quality and full-cost accounting
(#49); post-zero structured factorization with residual accounting (#50).
No model-quality, full-checkpoint prevalence, GPU speed or energy claims from
this first slice. No live service/driver change or independent-review claim.
