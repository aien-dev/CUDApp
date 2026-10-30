# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Portable backend, first non-NVIDIA code path (docs/ARCHITECTURE.md §3, PROPOSED split).
# OMC is the absolute path to the pinned seeded omc (aien-dev/omega-mv 369dbc96 seed/out/omc).
#   make doctor OMC=...            also prints the target table from backend/targets.om
#   make test-multitarget OMC=...  host gate tests/multitarget/run.sh (SPIR-V on PoCL/lavapipe CPUs,
#                                  ORACLE gfx1151 HSACO validation, every negative control). Not in
#                                  HOST_TESTS: it needs omc, spirv-val, llvm-mc and ld.lld; a missing
#                                  tool reports NOT_RUN. Every GPU target stays hardware NOT_RUN.
# The C launcher tools/multitarget/spirv-host.c is migration-pending (default OFF): pass
# CUDAPP_MIGRATION_PENDING_SPIRV_HOST=1 plus POCL_LIBRARY to execute the modules. Its .om successors are
# backend/opencl/host.om and backend/vulkan/host.om once Omega has checked shared-library calls
# (E2, aien-dev/omega-mv#888).
OMC ?=
.PHONY: doctor-targets test-multitarget
doctor-targets:
	sh tools/multitarget/targets-doctor.sh '$(OMC)' $(OUT_DIR)/multitarget
test-multitarget:
	@test -n '$(OMC)' || { echo 'NOT_RUN: test-multitarget needs OMC=/absolute/path/to/pinned/omc'; exit 77; }
	sh tests/multitarget/run.sh '$(OMC)'
