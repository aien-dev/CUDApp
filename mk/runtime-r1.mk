# Omega runtime R1 (omega-mv#594): the Runtime API manifest checked against the headers. Host only, no GPU, no physics checkout.
# Picked up by `-include mk/*.mk` at the top of the Makefile, so no extra include line is needed.
RUNTIME_API_MANIFEST ?= docs/runtime/omega_runtime_api.manifest
RUNTIME_API_MD ?= docs/runtime/API.md
RUNTIME_API_WORK ?= $(OUT_DIR)/runtime-api
test-runtime-api:
	CC="$(CC)" tools/runtime_api_check.sh $(RUNTIME_API_MANIFEST) . $(RUNTIME_API_WORK)/real $(RUNTIME_API_MD)
	CC="$(CC)" tools/runtime_api_negative.sh $(RUNTIME_API_MANIFEST) . $(RUNTIME_API_WORK) $(RUNTIME_API_MD)
