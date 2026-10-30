# Rebased docs repair verification

Host-only, Linux x86_64. Rebased base:
`67f3135ef88d8f520dfc15a2b2bc3264f53f0d6a`.
Tested head: `865c4d22c01d4af9ce6e2108268557e23db7ff22`.
Source repair commit: `b2c78f9f9ea3c5c39ea2b37f5287172dd02a80f0`.

The prior local-only source and receipt commits were rebased without conflict:

```
1:  8b02205 = 1:  b2c78f9 docs: align current provenance and repair stack legend spacing
2:  d56c5c7 = 2:  865c4d2 test(docs): record legend regression and host verification limits
```

An independent reviewer confirmed this range-diff and the unchanged repair.
The earlier receipt remains an immutable record of the local pre-publication
snapshot. Its old commit IDs identify that local snapshot; use the new IDs above
for published source. Upstream README first-kernel instructions are retained.
No currently open PR edited the affected docs or figure files at the owner check.

Fresh verification:

- `python3 tools/figures/test_stack_legend.py -v`: both tests pass, including the
  deliberate-overlap negative control. See `focused.log`.
- `python3 tools/figures/make_figures.py --check`: `figures current: 10`.
- `git diff origin/main...HEAD --check`: exit 0.
- Inkscape re-render of the bottom legend was visually inspected. The labels
  remain separated; no figure content or dimensions changed during the rebase.
- `make OUT_DIR=build/docs-rebase test`: exit 2 before suite completion, unchanged
  x86_64 assembler refusal of AArch64 barriers in `omega_blackwell_engine.c:393`
  and `:558`. This used a fresh output directory. See `make-test.log`.
- `make OUT_DIR=build/docs-rebase -j2`: exit 2. The parallel build also reaches
  `omega_blackwell_submit.c:284`, `:377`, `:541` and `:622`, with the same `ds`
  operand errors. See `make-build.log`.

The local build and full host suite have not passed. No barrier, compiler flag,
gate, runtime, swap path or driver code was changed by this repair. GPU/chip
execution and the trace-emitter runner remain NOT_RUN here. CI results, including
native ARM64 checks, are separate evidence for the published PR head.
