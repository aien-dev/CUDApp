---
name: Add a library
about: A new kernel or a wider one (any-shape GEMM, scan, sort, FFT, random numbers ...).
title: "Library: "
labels: library
---

**What it computes**, and which CUDA library does this today.

**Shapes and types** it must cover.

**The host reference**: how the exact answer is computed on the CPU.

**Done when** (the command anyone can run, and what it must print):

**Checklist**
- [ ] host reference and test, seen failing first
- [ ] mutation test: a deliberately broken copy fails
- [ ] simulator run, where the simulator covers the instructions
- [ ] chip receipt under `evidence/`, with speed as a fraction of the measured ceiling
