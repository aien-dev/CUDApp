# Personal source review (not independent review)

Author reviewed this branch personally, respecting the user's no-delegation
request. No independent reviewer was launched and no merge approval is implied.

- FP8 decode: all 254 finite codes independently match ml-dtypes; both NaN codes
  refused, signed zeros explicitly become numerical zero in the approximate lane.
- Quantization: exact integer maxima and symmetric nearest-even rounding,
  including explicit positive/negative half ties. No division by zero.
- Hadamard: native seven-stage butterfly cross-checked against an explicit sign
  matrix. H*H=128I and reconstruction denominators verified by Wolfram.
- Range: M<=29360128 and numerator magnitude<=26306674688 fit signed64; negative
  output uses two's-complement byte construction without signed/unsigned overflow.
- Domain: count/mode/length/NaNs/codes/scales checked before any stdout. Decoder
  validates positive-scale saturation; accepted frames need not prove provenance.
- Capacity: retained zero maximum-capacity cases and added nonzero maximum-capacity
  encode/decode cases so zero output cannot hide a missing tail of reconstruction.
- I/O: fragmented requests, closed descriptors and full output device exercised.
- Qualification: previous PASS retired before preflight; named parent links,
  hardlinks and special outputs refused; -O/PYTHONOPTIMIZE explicitly refused before
  running assert-based oracles. No hostile concurrent filesystem-race claim.
- Model input: immutable five selections, bounded safetensors metadata, payload and
  BF16 scale hashes, scale-tile alignment, positive finite scales, unchanged file
  identity/size/mtime across reads. No weights are published in evidence.
- References: NumPy/ml-dtypes used only outside the native product. They are not an
  inference runtime. Native code and exact reconstruction agree with a separate
  dense sign-matrix oracle over every value in all five matrices.
- Storage: equal frame/scale bytes for identity and rotation. Lossless zero packing
  is a separately labeled exact comparator. No final GPU padding, full-container
  size or resident-memory claim is made.
- Quality: synthetic vectors are explicitly not real activations or model quality.
  The down-projection regression and loss of all 280 exact zero columns are kept,
  not averaged away. No rotation is moved through SwiGLU or another nonlinearity.
- Cost: 896 add/sub per 128-value transform is a mathematical count, not latency.
  Native rational reconstruction expands to int64 numerators for validation; this
  is not a ready compact-resident INT4 inference kernel.

Disposition: offline component is qualified for its bounded stated contract.
No default path, GPU driver, live model, compiler worker or production service was
changed. Full model quality, real activation capture, calibration, native GPU
integration and measured end-to-end latency/energy remain open under #49.
