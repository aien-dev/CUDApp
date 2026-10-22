---
name: Port a chip
about: You want CUDA++ to run on your NVIDIA GPU. Open this first so nobody duplicates the work.
title: "Port: <GPU name> (<PCI id>)"
labels: port
---

**GPU** (name, and the PCI id `make doctor` prints, for example 0x2b85)

**Kernel module version** (first line of `cat /proc/driver/nvidia/version`)

**Output of `make doctor`**

```
```

**Output of `make test`** (last 20 lines are enough if it passes)

```
```

**Plan.** Follow `docs/runtime/PORTABILITY.md` section 5. Tick as you go, with a receipt path for each:

- [ ] target row (PCI id, architecture, class numbers, launch descriptor version, memory model)
- [ ] encoder words checked on this chip
- [ ] launch descriptor and class numbers checked
- [ ] `make hello-gpu` PASS (receipt under `evidence/`)
- [ ] chip row added to `src/omega_gpu_chip.c` with the receipt path
