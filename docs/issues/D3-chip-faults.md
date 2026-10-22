# D3, chip half: read the GPU's own fault record after a fault

Labels: pain-point, D3

**The pain.** CUDA reports a bad memory access at a later, unrelated call: "might be asynchronously reported at some other API call". vLLM's maintainers: "In our experience the python stack traces for these types of exceptions are basically always incorrect and pretty worthless." `docs/PAIN_POINTS.md` section 2.3.

**What exists.** CUDA++ launches wait for their own completion by default, so an error comes back from the call that caused it. On the host simulator a bad access is named in full: kernel, launch, instruction, block, thread, address, the buffer it missed and by how much (`hello_gpu --sim --bug=past-end`).

**What is left.** On the chip, a faulting kernel today shows up as a launch that does not complete (a timeout), and the process latches. Read the fault itself from the driver (the MMU fault's address, access type and engine, which the open kernel module records) and print it at the launch, with the kernel name and the next command (rerun the same launch under `--sim`).

**Done when.** A deliberately bad kernel run on the GB10 inside a chip window prints the faulting address and the kernel name at the launch that caused it; receipt under `evidence/`.
