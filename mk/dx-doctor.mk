# SPDX-License-Identifier: AGPL-3.0-only
# Developer bar D2 and D6 (docs/PAIN_POINTS.md): `make doctor` says, in sentences, whether this
# machine can run CUDA++ and what to do if not. It reads files under /proc, /sys and /dev only:
# it opens no device and needs no GPU window. `make hello-gpu` runs it first.
DOCTOR = $(OUT_DIR)/cudapp-doctor
# The one version CUDA++ has to match: the NVIDIA open kernel module whose headers it is built
# against. Taken from the folder name, so there is exactly one place it is written.
CUDAPP_NVIDIA_OPEN_ABI := $(patsubst nvidia-open-%,%,$(notdir $(NVIDIA_OPEN)))

.PHONY: doctor test-doctor
$(DOCTOR): tools/doctor/cudapp_doctor.c $(wildcard src/omega_gpu_chip.c src/omega_gpu_chip.h) | $(OUT_DIR)
	$(CC) $(CFLAGS) -DCUDAPP_NVIDIA_OPEN_ABI='"$(CUDAPP_NVIDIA_OPEN_ABI)"' -o $@ tools/doctor/cudapp_doctor.c $(wildcard src/omega_gpu_chip.c)
doctor: $(DOCTOR)
	$(DOCTOR)
test-doctor: $(DOCTOR)
	sh tests/doctor/run.sh $(DOCTOR)
