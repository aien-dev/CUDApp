# Third-party material in CUDA++

The first-party material in the transition candidate is Apache-2.0 WITH LLVM-exception (`LICENSE`). This repository also contains and cites third-party material under separate terms. Those grants and notices remain in force; they are not replaced by the project license.

## 1. NVIDIA open GPU kernel module headers (copied, unchanged)

`third_party/nvidia-open-580.173.02/` holds 67 header files from NVIDIA's open-gpu-kernel-modules,
release 580.173.02 (the driver version on the GB10 this was built on). They describe the ioctl
interface of the open kernel module, the object classes and the control calls. `SHA256SUMS` in that
folder lists every file. Each file keeps its own NVIDIA copyright and MIT notice (60 carry
`SPDX-License-Identifier: MIT`; 7 older ones carry the MIT permission text without the SPDX line).
`third_party/nvidia-open-580.173.02/COPYING` is NVIDIA's licence file: the files are MIT, and only
when linked together into a Linux kernel module is the result dual MIT/GPLv2. CUDA++ builds no kernel
module from them; it uses them as user-space interface definitions.

    Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
    (full MIT notice in each file and in COPYING)

## 2. Mesa (NVK Vulkan driver and NAK compiler): facts cited in comments, no code copied

Some comments in `src/` cite Mesa's open NVIDIA driver as the source of a hardware fact: instruction
opcodes and bit positions (`nak/sm70_encode.rs`), instruction latencies (`nak/sm120_instr_latencies.rs`),
the QMD layout (`nak/qmd.rs`), reconvergence scopes (`nak_nir_lower_cf.c`), the compute dispatch and
cache-invalidate sequence (`nvk_cmd_dispatch.c`), the shader-heap prefetch note (`nvk_device.c`), and
class method offsets (`clcec0.h`). The encoder, scheduler and command-list code in CUDA++ is my own;
the latency values used by `src/omega_gpu_elementwise_api.c` were taken from Mesa's tables. These
Mesa files are MIT (checked 2026-10-21 at gitlab.freedesktop.org/mesa/mesa, branch main, first lines of
each file):

    nak/sm70_encode.rs, nak/qmd.rs, nak_nir_lower_cf.c   Copyright (c) 2022-2024 Collabora, Ltd.   SPDX-License-Identifier: MIT
    nak/sm120_instr_latencies.rs                          Copyright (c) 2025 Red Hat.               SPDX-License-Identifier: MIT
    vulkan/nvk_device.c, vulkan/nvk_cmd_dispatch.c         Copyright (c) 2022 Collabora Ltd. and Red Hat Inc.   SPDX-License-Identifier: MIT

    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
    associated documentation files (the "Software"), to deal in the Software without restriction,
    including without limitation the rights to use, copy, modify, merge, publish, distribute,
    sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions: The above copyright notice and this
    permission notice shall be included in all copies or substantial portions of the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.

## 3. Other open sm_12x work

These projects map the same instruction set or the same driver interface in the open. Apart from the
Mesa facts and latency values in section 2, no code or data from them is in CUDA++; they're listed so
readers can find and compare them. Cross-checking CUDA++ against them is PROPOSED.

    Mesa NAK SM120 encoder        gitlab.freedesktop.org/mesa/mesa (src/nouveau/compiler)   MIT
    cubit                         github.com/kacper-daftcode/cubit                          MIT
    blackwell-isa                 github.com/kacper-daftcode/blackwell-isa                  MIT
    nvidia-sass-document          github.com/cicuvc/nvidia-sass-document                    no licence shown
    tinygrad NV backend           github.com/tinygrad/tinygrad (tinygrad/runtime/ops_nv.py) MIT

## Public-source quotations

`docs/PAIN_POINTS.md` contains attributed quotations from the public sources listed there. They remain third-party text; the repository license does not relicense them.

## Not in CUDA++

- No NVIDIA firmware, no libcuda, no cubin or fatbin, no PTX. The GPU's signed GSP firmware is loaded
  by NVIDIA's kernel module from the system's own firmware folder; CUDA++ never ships it.
- Some host tests print a listing through NVIDIA's `nvdisasm` when it happens to be installed, as a
  cross-check. No listing made by NVIDIA tools is stored in this repository (one such file was kept out
  of the history at extraction, see `docs/AUDIT.md`). My own disassembler is on the roadmap (C6).
