# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Omega runtime R1 (omega-mv#594): the Runtime API manifest checked against the headers. Host only, no GPU, no physics checkout.
# Picked up by `-include mk/*.mk` at the top of the Makefile, so no extra include line is needed.
RUNTIME_API_MANIFEST ?= docs/runtime/omega_runtime_api.manifest
RUNTIME_API_MD ?= docs/runtime/API.md
RUNTIME_API_WORK ?= $(OUT_DIR)/runtime-api
test-runtime-api:
	CC="$(CC)" tools/runtime_api_check.sh $(RUNTIME_API_MANIFEST) . $(RUNTIME_API_WORK)/real $(RUNTIME_API_MD)
	CC="$(CC)" tools/runtime_api_negative.sh $(RUNTIME_API_MANIFEST) . $(RUNTIME_API_WORK) $(RUNTIME_API_MD)

# The version-bump check: the manifest against the one on RUNTIME_API_BASE (a git revision; CI passes the pull request's
# base). Fails when an entry changed without the bump docs/runtime/API.md section 2 asks for. Host only.
RUNTIME_API_BASE ?= origin/main
.PHONY: test-runtime-api test-runtime-api-semver
test-runtime-api-semver:
	mkdir -p $(RUNTIME_API_WORK)
	git show $(RUNTIME_API_BASE):$(RUNTIME_API_MANIFEST) > $(RUNTIME_API_WORK)/base.manifest
	tools/runtime_api_semver.sh $(RUNTIME_API_WORK)/base.manifest $(RUNTIME_API_MANIFEST)
