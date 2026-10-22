# Personal source review (not independent review)

Per Drake's instruction, no subagent, external worker or reviewer was launched.
Reviewed the native codec, independent oracle, process harness and protocol.

- Header words decoded only after the minimum prefix exists; dimensions bounded
  before product calculation or channel indexing; axis validated before shape use.
- All index/payload lengths bounded before reads; maximum legal matrix cases pass.
- Sparse channels write only to private output until the entire payload and EOF
  boundary are accepted. Transport failure is distinguished from syntax refusal.
- Zero signs preserved; all 256 encodings covered without floating-point decode.
- Packed zero padding and raw/zero canonical tags checked; dense ties stay dense.
- No checksum/authentication claim. Invalid structure rejected, arbitrary valid
  content mutation requires external hash validation.
- Oracle constructs channel vectors independently; native encode equals oracle
  bytes, and both independent decode and native decode reproduce original frames.
- Three compiled mutants are actually killed; no file edits masquerade as mutants.
- Data from five real matrices stay local; committed receipts contain hashes,
  dimensions, indices and counts only. Full model metadata is explicitly not a
  full payload scan. Scale bytes are unchanged and counted separately.
- Existing C kernels and live serving/driver paths are untouched. Optional make
  target runs qualification without adding an unavailable compiler to defaults.

Follow-up review exposed stale-PASS and linked-output hazards in the host
qualification runner. Five controls failed before repair, then passed after
archiving the previous receipt before preflight and refusing unsafe output
trees. Full codec/model replay passed afterward. This protects preexisting
outputs under single-user preflight, not hostile concurrent filesystem races.

Limitations: no independent review, no whole-model payload scan, no loader/GPU
integration, no model-quality or performance qualification. Source-only seed
replay and host tests do not remove these gates.
