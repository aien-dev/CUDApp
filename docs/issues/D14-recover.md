# D14: recover from a fault without restarting the process

Labels: D14

**The pain.** "Pretty much all asynchronous errors (these are errors that occur as a result of device code execution) are sticky." (NVIDIA forums). After one fault the whole process has to restart. `docs/PAIN_POINTS.md` section 2.3.

**What exists.** CUDA++ latches the process on an unconfirmed completion and parks buffers the GPU might still touch, instead of reusing them (`docs/runtime/API.md` section 3). Recovery is open (omega-mv#518).

**What is left.** Stop a faulting or endless kernel and keep the same process usable. Omega bar C5, ledger L386.

**Done when.** A kernel that never ends is timed out, and the same process then runs hello-gpu to PASS; chip receipt.
