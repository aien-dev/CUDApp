# Weightpack v1: reversible FP8 byte representation

Refs [CUDApp #48](https://github.com/aien-dev/CUDApp/issues/48).
Experimental standalone Linux AArch64 host utility, written in seeded native
Omega. It does not load a model into the inference engine or change GPU kernels.

## Protocol
One complete frame on stdin, EOF required. One result on stdout after validation.
All integers below are unsigned little-endian 32-bit words. Bytes are unchanged
FP8 storage encodings, not numbers decoded and requantized by this program.

| Frame | Header | Payload |
|---|---|---|
| `ZCR1` (encode request / decode result) | magic, rows, columns, axis | row-major matrix bytes |
| `ZCP1` (encode result / decode request) | magic, rows, columns, axis, mode | dense or channel payload |

Each dimension is 1..4096 and their product is at most 1,048,576.
Axis 0 means row channels; axis 1 means column channels. Mode 0 is dense
row-major storage; mode 1 lists channels in increasing index order.

Each sparse channel has a one-byte tag:
* 0: every byte is positive zero (`00`); no following bytes.
* 1: every magnitude is zero, with at least one negative zero (`80`); followed
  by ceil(channel_length/8) sign bytes, LSB-first. Unused high bits must be zero.
* 2: channel_length original bytes, with at least one nonzero magnitude.

The encoder chooses sparse only if its payload is strictly smaller than dense.
The decoder accepts valid dense payloads even when sparse could save bytes.
The sparse decoder rejects noncanonical zero tags and non-saving payloads.

**Exact contract:** decode(encode(raw_frame)) equals raw_frame byte for byte,
including signed zeros, all finite codes and both FP8 NaN codes. No new floating
point operations are introduced. Block scales are outside this frame and must
be retained unchanged with their original layout and identity. This is not
permission to omit zero multiplications from an execution graph.

**Overhead:** packed framing is 20 bytes, raw framing is 16 bytes. A dense packed
frame is therefore 20 bytes larger than the unframed matrix; the future loader
must retain its existing unframed representation when total cost does not win.
Metadata, hashes, scale storage, alignment, loader tables and decode workspace
are not free. The current experiment includes frame bytes and reports scales
separately, but does not claim an integrated container/storage saving.

**Refusals:** exit 65, no stdout for malformed, unsupported, overlong, truncated
or trailing input. Exit 74 on read/write failure; OS signal termination (for
example SIGPIPE) remains possible. Failed transport can leave partial stdout.
Consumers must stage and publish only after successful exit. Waiting for EOF is
intentional, not streaming multiple requests. No filenames/argv/network access.

**Integrity:** the format is not authenticated. A changed byte that remains a
valid encoding is not detectable from syntax. Bind inputs and outputs to content
hashes in the container/loader; the qualification harness records SHA256.

## Build and replay
Requires the seeded Omega compiler with SHA256
`191b776ef98821597c3e30655da6d680a9b66edbc9ceaca3c39c61ea6df098e9`.
Source origin: `omega-mv@29b12760f4c962606411d6bf25032aedef063301`,
`seed/omc/omc.om`. This run rebuilt the compiler from the frozen source-only
bootstrap package `omega-bridge@d2010ab927b608931800c20d73a4c2b25f00a9fa`.
The bootstrap and fixpoint were replayed; no prebuilt compiler was trusted.
The external compiler is an explicit prerequisite, not silently downloaded by
the CUDApp build. Compiler source/package publication is a separate workstream.

```sh
make test-native-weightpack OMEGA_NATIVE_COMPILER=/absolute/path/to/omc
```

This optional target creates `build/weightpack/weightpack` and receipts under
`build/weightpack/evidence`. It does not change the default host test dependency
or install anything. Python is only the independent test oracle/orchestrator;
the produced executable's codec runs entirely in native Omega.

For the fixed real-checkpoint replay (read-only, no retained weights):

```sh
python3 native/weightpack/qualify.py   --compiler /absolute/path/to/omc   --build build/weightpack   --evidence build/weightpack/checkpoint-evidence   --model /absolute/path/to/qwen3.6-35b-a3b-fp8-95a723d0
```

The harness checks all routed-expert weight metadata against config-derived
counts. **Metadata enumeration is not a full-checkpoint payload scan.** The five
frozen matrices are the previously selected research samples, not a statistical
sample. No quality, token latency, GPU traffic, energy or throughput claim follows.

## Qualification output safety
The runner and Make entry archive an old `qualification.json` before new
preflight checks, so a failed rerun cannot expose the previous PASS as current.
Published receipts are immutable history; `qualification.previous-*.json` files
are retained history, not current qualification. Output-directory symlinks,
hardlinks and special files are refused before build writes. This is single-user
preflight, not protection against hostile concurrent filesystem replacement.

```sh
python3 native/weightpack/test_build_safety.py --compiler /path/to/omc \
  --out build/weightpack/build-safety.json
```

## Implementation notes
The pinned seed assigns at most sixteen registers to a function; helpers bound
expression pressure. Runtime-bounded loops trap above one million iterations,
whereas constant-bound loops differ. Dense copies are chunked, and I/O uses
explicit constant ceilings large enough for one-byte progress. The initial
compile refusals and maximum-size failures are retained in evidence.

Future work remains under #48: full payload inventory, total-cost/container
choice, loader integration, then exact model-level and hardware qualification.
Approximate rotation (#49) and conditional factorization (#50) remain separate.
