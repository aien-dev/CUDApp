# Omega runtime R3 (omega-mv#596): the portability table (docs/runtime/portability_table.tsv) must match src/.
# Picked up by `-include mk/*.mk`. Host only: no compiler, no device, no physics headers.
#   test-runtime-portability-check     every src/omega_gpu_*, omega_blackwell_*, omega_bw_* file has a row, every row names a
#                                      file, a PORTABLE row's file includes no Linux/NVRM or Blackwell header, and the
#                                      os_binding column agrees with the include scan and the freestanding file list
#   test-runtime-portability-controls  breaks one thing at a time in scratch copies; the check must FAIL each (and PASS the
#                                      guarded-include control)
#   test-runtime-portability           both
.PHONY: test-runtime-portability test-runtime-portability-check test-runtime-portability-controls
test-runtime-portability-check:
	sh tools/runtime_portability_check.sh . docs/runtime/portability_table.tsv mk/omega-gpu-engine.mk
test-runtime-portability-controls:
	sh tools/runtime_portability_controls.sh build/runtime_r3_controls docs/runtime/portability_table.tsv mk/omega-gpu-engine.mk
test-runtime-portability: test-runtime-portability-check test-runtime-portability-controls
