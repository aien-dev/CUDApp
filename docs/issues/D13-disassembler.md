# D13: our own disassembler, in plain text a person can read

Tracking: [#10](https://github.com/aien-dev/CUDApp/issues/10). Keep open until its exact-commit acceptance gate passes.

Labels: D13

**The pain.** PTX "is only partially documented, constantly shifting between hardware generations, and effectively a black box for developers." (Modular). CUDA++'s host tests still call NVIDIA's `nvdisasm` as a cross-check when it is installed.

**What is left.** A disassembler built from CUDA++'s own encoder tables, printing each instruction with its control word (stall, barriers) in words, and a timing view like the motor's. Omega bar C6, ledger L386.

**Done when.** It agrees with `nvdisasm` on every library kernel, and nothing in the tree calls `nvdisasm`.
