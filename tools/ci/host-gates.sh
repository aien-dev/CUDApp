#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Fresh-checkout host gates. Reads no GPU device node and does not run make hello-gpu.
#
#   tools/ci/host-gates.sh --self-test   checker negative controls (no compile)
#   tools/ci/host-gates.sh --figures     README figure byte check
#   tools/ci/host-gates.sh               make -j4 and the host commands below
#
# make test is the host-tests job in .github/workflows/ci.yml. This script does not repeat it.
# make test-runtime-api is .github/workflows/runtime-api.yml (#189). An x86-64 host-tests
# job is #196. Neither is started here.
set -euo pipefail
export LC_ALL=C

root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$root"
out=build/host-gates
mkdir -p "$out"

# Simulator output of c = a + b over 2^20 floats. Measured by build/hello_gpu --sim.
# Pinned so a silent answer change fails even when the program still prints PASS.
output_sha='output sha256 72d0c1b2928779e83518f0b4c7d101d933cf89f210cd4627e138c33c9ce9b6a6'

fail=0
summary=$out/summary.txt

note() { printf '%s\n' "$*" | tee -a "$summary"; }

run_log() {
	local log=$1
	shift
	set +e
	"$@" >"$log" 2>&1
	local rc=$?
	set -e
	cat "$log"
	printf '%s\n' "$rc" >"$log.rc"
}

rc_of() { cat "$1.rc"; }

archives_ok() {
	local rc=$1 dir=$2
	[ "$rc" -eq 0 ] || return 1
	[ -f "$dir/libomega_gpu.a" ] && [ -f "$dir/libcudapp.a" ]
}

artifact_ok() {
	local rc=$1 log=$2 line
	[ "$rc" -eq 0 ] || return 1
	line=$(grep -E '^test_gpu_artifact_export: ' "$log" | tail -n 1 || true)
	[[ "$line" =~ ^test_gpu_artifact_export:\ [0-9]+\ checks,\ 0\ failed$ ]]
}

zero_cuda_ok() {
	local rc=$1 log=$2
	[ "$rc" -eq 0 ] || return 1
	grep -qF 'ZERO_CUDA SELF-TEST PASS (CUDA probe rejected, clean library accepted)' "$log" || return 1
	grep -qF 'ZERO_CUDA FAIL' "$log" || return 1
	grep -qF 'libcudart.so' "$log" || return 1
	grep -qF 'cudaMalloc' "$log" || return 1
	grep -qF 'ZERO_CUDA PASS build/hello_gpu needs=' "$log" || return 1
	grep -qF 'ZERO_CUDA PASS build/gpu_matmul_api_test needs=' "$log" || return 1
	grep -qF 'ZERO_CUDA SELF-TEST BROKEN' "$log" && return 1
	return 0
}

sim_ok() {
	local rc=$1 log=$2
	[ "$rc" -eq 0 ] || return 1
	grep -qF 'on the host simulator (no GPU)' "$log" || return 1
	grep -qF 'checked 1048576 of 1048576 elements: 0 wrong, 0 never written' "$log" || return 1
	grep -qF 'work: 12582912 bytes moved (8388608 read, 4194304 written), 1048576 floating-point operations' "$log" || return 1
	grep -qF "$output_sha" "$log" || return 1
	grep -qxF 'PASS' "$log" || return 1
	grep -qE '^machine code sha256 [0-9a-f]{64}$' "$log" || return 1
}

# The fault path exits 1 from inside the launch loop. It prints FAULT and does not print PASS.
past_end_ok() {
	local rc=$1 log=$2
	[ "$rc" -eq 1 ] || return 1
	grep -qF 'FAULT at launch 8 of 8 of kernel vecadd (simulator): global load outside every buffer' "$log" || return 1
	grep -qF '0 bytes past the end of buffer a' "$log" || return 1
	grep -qF '0 wrong, 0 never written' "$log" && return 1
	grep -qxF 'PASS' "$log" && return 1
	return 0
}

gpu_bug_refused_ok() {
	local rc=$1 log=$2
	[ "$rc" -eq 2 ] || return 1
	grep -qF -e '--bug=past-end is only for --sim: on the GPU it would write past a buffer' "$log" || return 1
}

# make exits 2 when the doctor recipe exits 1. Exit 0 would be READY, which this runner must not claim.
doctor_ok() {
	local rc=$1 log=$2
	[ "$rc" -eq 2 ] || return 1
	grep -qF 'cudapp doctor: is this machine ready for CUDA++?' "$log" || return 1
	grep -qF 'FAIL  NVIDIA kernel module: not loaded' "$log" || return 1
	grep -qF 'FAIL  device files: /dev/nvidiactl is missing' "$log" || return 1
	grep -qF 'FAIL  device files: /dev/nvidia0 is missing' "$log" || return 1
	grep -qF 'FAIL  device files: /dev/nvidia-uvm is missing' "$log" || return 1
	grep -qF 'FAIL  GPU: no NVIDIA GPU found in /sys/bus/pci/devices' "$log" || return 1
	grep -qF 'NOT READY:' "$log" || return 1
	grep -qF 'NOT_RUN: the target table is seeded Omega' "$log" || return 1
	grep -qF 'Error 1' "$log" || return 1
	if grep -qxF 'READY. Next: make hello-gpu' "$log"; then
		return 1
	fi
	return 0
}

figures_ok() {
	local rc=$1 log=$2
	[ "$rc" -eq 0 ] || return 1
	grep -qE '^figures current: [0-9]+$' "$log" || return 1
	grep -qF 'stale figures' "$log" && return 1
	return 0
}

expect_ok() {
	local name=$1
	shift
	if "$@"; then
		echo "PASS $name"
		return 0
	fi
	echo "FAIL $name"
	fail=1
	return 0
}

expect_reject() {
	local name=$1
	shift
	if "$@"; then
		echo "FAIL $name (checker accepted a case that must be rejected)"
		fail=1
		return 0
	fi
	echo "PASS $name"
}

self_test() {
	local t log
	t=$(mktemp -d)
	log=$t/log

	mkdir -p "$t/both"
	: >"$t/both/libomega_gpu.a"
	: >"$t/both/libcudapp.a"
	expect_ok archives-both archives_ok 0 "$t/both"
	expect_reject archives-missing archives_ok 0 "$t"
	expect_reject archives-bad-exit archives_ok 2 "$t/both"

	printf '%s\n' 'test_gpu_artifact_export: 47 checks, 0 failed' >"$log"
	expect_ok artifact-clean artifact_ok 0 "$log"
	printf '%s\n' 'FAIL tamper caught' 'test_gpu_artifact_export: 47 checks, 0 failed' >"$log"
	expect_ok artifact-fail-word-kept artifact_ok 0 "$log"
	printf '%s\n' 'test_gpu_artifact_export: 47 checks, 1 failed' >"$log"
	expect_reject artifact-failed-count artifact_ok 0 "$log"
	printf '%s\n' 'test_gpu_artifact_export: 47 checks, 0 failed' >"$log"
	expect_reject artifact-bad-exit artifact_ok 1 "$log"

	cat >"$log" <<'EOF'
ZERO_CUDA FAIL /tmp/probe.so needs: libcudart.so.13
ZERO_CUDA FAIL /tmp/probe.so imports: cudaMalloc
ZERO_CUDA SELF-TEST PASS (CUDA probe rejected, clean library accepted)
ZERO_CUDA PASS build/hello_gpu needs=[libm.so.6 libc.so.6]
ZERO_CUDA PASS build/gpu_matmul_api_test needs=[libc.so.6]
EOF
	expect_ok zero-cuda-with-visible-fail zero_cuda_ok 0 "$log"
	expect_reject zero-cuda-bad-exit zero_cuda_ok 1 "$log"
	printf '%s\n' 'ZERO_CUDA PASS build/hello_gpu needs=[libc.so.6]' >"$log"
	expect_reject zero-cuda-hidden-control zero_cuda_ok 0 "$log"

	cat >"$log" <<EOF
hello_gpu: c = a + b over 1048576 floats on the host simulator (no GPU), built with Omega's own kernel builder
machine code sha256 9e28bd9c46f80584d77238a752e99cecddbc8d622ffbe320885ed2f499e1f075
checked 1048576 of 1048576 elements: 0 wrong, 0 never written
work: 12582912 bytes moved (8388608 read, 4194304 written), 1048576 floating-point operations
$output_sha
PASS
EOF
	expect_ok sim-pass sim_ok 0 "$log"
	printf '%s\n' 'PASS' >"$log"
	expect_reject sim-pass-only sim_ok 0 "$log"
	cat >"$log" <<EOF
on the host simulator (no GPU)
checked 1048576 of 1048576 elements: 0 wrong, 0 never written
work: 12582912 bytes moved (8388608 read, 4194304 written), 1048576 floating-point operations
$output_sha
machine code sha256 9e28bd9c46f80584d77238a752e99cecddbc8d622ffbe320885ed2f499e1f075
FAIL
EOF
	expect_reject sim-printed-fail sim_ok 0 "$log"

	cat >"$log" <<'EOF'
FAULT at launch 8 of 8 of kernel vecadd (simulator): global load outside every buffer
  address: 0x1000 (4 bytes), 0 bytes past the end of buffer a = [0x0, 0x1000) (4194304 bytes)
EOF
	expect_ok past-end past_end_ok 1 "$log"
	printf '%s\n' \
		'FAULT at launch 8 of 8 of kernel vecadd (simulator): global load outside every buffer' \
		'0 bytes past the end of buffer a' \
		'checked 1048576 of 1048576 elements: 0 wrong, 0 never written' \
		'PASS' >"$log"
	expect_reject past-end-continued past_end_ok 1 "$log"
	expect_reject past-end-exit0 past_end_ok 0 "$log"

	printf '%s\n' '--bug=past-end is only for --sim: on the GPU it would write past a buffer' >"$log"
	expect_ok gpu-bug-refused gpu_bug_refused_ok 2 "$log"
	expect_reject gpu-bug-exit0 gpu_bug_refused_ok 0 "$log"

	cat >"$log" <<'EOF'
cudapp doctor: is this machine ready for CUDA++?
  FAIL  NVIDIA kernel module: not loaded (/proc/driver/nvidia/version does not exist)
  FAIL  device files: /dev/nvidiactl is missing
  FAIL  device files: /dev/nvidia0 is missing
  FAIL  device files: /dev/nvidia-uvm is missing
  FAIL  GPU: no NVIDIA GPU found in /sys/bus/pci/devices
NOT READY: 5 problems above. Fix the first FAIL line and run make doctor again.
  NOT_RUN: the target table is seeded Omega and needs the pinned omc.
make: *** [mk/dx-doctor.mk:14: doctor] Error 1
EOF
	expect_ok doctor-not-ready doctor_ok 2 "$log"
	expect_reject doctor-exit0 doctor_ok 0 "$log"
	printf '%s\n' 'READY. Next: make hello-gpu' >>"$log"
	expect_reject doctor-ready-line doctor_ok 2 "$log"
	grep -qF 'NOT READY:' "$log" || { echo "FAIL doctor fixture lost NOT READY"; fail=1; }

	printf '%s\n' 'figures current: 10' >"$log"
	expect_ok figures-current figures_ok 0 "$log"
	printf '%s\n' 'stale figures (run python3 tools/figures/make_figures.py): stack.svg' >"$log"
	expect_reject figures-stale figures_ok 1 "$log"
	printf '%s\n' 'figures current: 10' >"$log"
	expect_reject figures-bad-exit figures_ok 1 "$log"

	rm -rf "$t"
	if [ "$fail" -ne 0 ]; then
		echo "CUDA++ HOST GATES SELF-TEST FAIL" | tee "$out/self-test.txt"
		exit 1
	fi
	echo "CUDA++ HOST GATES SELF-TEST PASS" | tee "$out/self-test.txt"
	exit 0
}

finish() {
	echo "----- summary -----"
	cat "$summary"
	if [ "$fail" -ne 0 ]; then
		echo "CUDA++ HOST GATES FAIL"
		exit 1
	fi
	echo "CUDA++ HOST GATES PASS"
	exit 0
}

record() {
	# name, exit, result, detail
	note "$(printf '%s\texit=%s\t%s\t%s' "$1" "$2" "$3" "$4")"
	if [ "$3" = FAIL ]; then
		fail=1
	fi
}

figures_mode() {
	local log=$out/figures.log rc
	: >"$summary"
	run_log "$log" python3 tools/figures/make_figures.py --check
	rc=$(rc_of "$log")
	if figures_ok "$rc" "$log"; then
		record 'python3 tools/figures/make_figures.py --check' "$rc" PASS "$(grep -E '^figures current: ' "$log")"
	else
		record 'python3 tools/figures/make_figures.py --check' "$rc" FAIL 'checker rejected the figure output'
	fi
	if command -v inkscape >/dev/null 2>&1; then
		local legend=$out/stack-legend.log
		run_log "$legend" python3 tools/figures/test_stack_legend.py -v
		rc=$(rc_of "$legend")
		if [ "$rc" -eq 0 ]; then
			record 'python3 tools/figures/test_stack_legend.py' "$rc" PASS 'rendered legend'
		else
			record 'python3 tools/figures/test_stack_legend.py' "$rc" FAIL 'legend test failed'
		fi
	else
		record 'python3 tools/figures/test_stack_legend.py' '-' NOT_RUN 'inkscape is not installed'
	fi
	finish
}

identity() {
	{
		echo "source_sha $(git rev-parse HEAD)"
		if [ -n "${HOST_GATES_HEAD_SHA:-}" ]; then
			echo "head_sha ${HOST_GATES_HEAD_SHA}"
		fi
		echo "uname $(uname -s -m -r)"
		if [ -r /etc/os-release ]; then
			# shellcheck disable=SC1091
			. /etc/os-release
			echo "os ${PRETTY_NAME:-unknown}"
		fi
		echo "gcc $(gcc --version | head -n 1)"
		echo "make $(make --version | head -n 1)"
		echo "python3 $(python3 --version 2>&1)"
		echo "jobs 4"
		echo "utc $(date -u +%Y-%m-%dT%H:%M:%SZ)"
	} | tee "$out/identity.txt"
}

gates() {
	local log rc
	: >"$summary"
	identity
	note 'make test	exit=-	ENFORCED_ELSEWHERE	CI job host-tests in .github/workflows/ci.yml'
	note 'make test-runtime-api	exit=-	ENFORCED_ELSEWHERE	.github/workflows/runtime-api.yml (#189)'
	note 'x86-64 make test	exit=-	NOT_RUN	#196 owns that CI job'
	note 'make hello-gpu	exit=-	NOT_RUN	no GB10 and no /dev/aien-gpu on this runner'

	log=$out/make-j4.log
	run_log "$log" make -j4
	rc=$(rc_of "$log")
	if archives_ok "$rc" build; then
		sha256sum build/libomega_gpu.a build/libcudapp.a | tee "$out/archives.sha256"
		record 'make -j4' "$rc" PASS "$(sed -n '1,2p' "$out/archives.sha256" | tr '\n' ' ')"
	else
		record 'make -j4' "$rc" FAIL 'library archives were not produced'
	fi

	if [ -f build/libomega_gpu.a ]; then
		log=$out/native-artifact.log
		run_log "$log" make -j4 test-native-artifact
		rc=$(rc_of "$log")
		if artifact_ok "$rc" "$log"; then
			record 'make test-native-artifact' "$rc" PASS "$(grep -E '^test_gpu_artifact_export: ' "$log" | tail -n 1)"
		else
			record 'make test-native-artifact' "$rc" FAIL 'exit code or summary line rejected'
		fi

		log=$out/zero-cuda.log
		run_log "$log" make -j4 zero-cuda-gate
		rc=$(rc_of "$log")
		if zero_cuda_ok "$rc" "$log"; then
			record 'make zero-cuda-gate' "$rc" PASS 'self-test FAIL lines kept; binaries PASS'
		else
			record 'make zero-cuda-gate' "$rc" FAIL 'exit code or ZERO_CUDA lines rejected'
		fi
	else
		record 'make test-native-artifact' '-' NOT_RUN 'blocked: make -j4 did not produce the library'
		record 'make zero-cuda-gate' '-' NOT_RUN 'blocked: make -j4 did not produce the library'
	fi

	if [ -x build/hello_gpu ]; then
		log=$out/hello-sim.log
		run_log "$log" build/hello_gpu --sim
		rc=$(rc_of "$log")
		local log2=$out/hello-sim-2.log
		run_log "$log2" build/hello_gpu --sim
		local rc2
		rc2=$(rc_of "$log2")
		local h1 h2
		h1=$(grep -E '^(output|machine code) sha256 ' "$log" || true)
		h2=$(grep -E '^(output|machine code) sha256 ' "$log2" || true)
		if sim_ok "$rc" "$log" && sim_ok "$rc2" "$log2" && [ -n "$h1" ] && [ "$h1" = "$h2" ]; then
			record 'build/hello_gpu --sim' "$rc" PASS 'two runs, same output and machine-code hashes'
		else
			record 'build/hello_gpu --sim' "$rc" FAIL "repeat exit $rc2; hashes differ or checks missed"
		fi

		log=$out/hello-past-end.log
		run_log "$log" build/hello_gpu --sim --bug=past-end
		rc=$(rc_of "$log")
		if past_end_ok "$rc" "$log"; then
			record 'build/hello_gpu --sim --bug=past-end' "$rc" PASS 'negative control: FAULT, exit 1, no PASS'
		else
			record 'build/hello_gpu --sim --bug=past-end' "$rc" FAIL 'negative control did not fail closed'
		fi

		log=$out/hello-bug-refused.log
		run_log "$log" build/hello_gpu --bug=past-end
		rc=$(rc_of "$log")
		if gpu_bug_refused_ok "$rc" "$log"; then
			record 'build/hello_gpu --bug=past-end' "$rc" PASS 'refused off the simulator'
		else
			record 'build/hello_gpu --bug=past-end' "$rc" FAIL 'GPU bug path was not refused'
		fi
	else
		record 'build/hello_gpu --sim' '-' NOT_RUN 'blocked: build/hello_gpu was not built'
		record 'build/hello_gpu --sim --bug=past-end' '-' NOT_RUN 'blocked: build/hello_gpu was not built'
		record 'build/hello_gpu --bug=past-end' '-' NOT_RUN 'blocked: build/hello_gpu was not built'
	fi

	log=$out/doctor.log
	run_log "$log" make -j4 doctor
	rc=$(rc_of "$log")
	if doctor_ok "$rc" "$log"; then
		record 'make doctor' "$rc" PASS 'NOT READY on this runner; chip qualification NOT_RUN'
	else
		record 'make doctor' "$rc" FAIL 'expected the no-GPU refusal, not READY'
	fi
	finish
}

case "${1:-}" in
--self-test) self_test ;;
--figures) figures_mode ;;
'') gates ;;
*)
	echo "usage: $0 [--self-test|--figures]" >&2
	exit 2
	;;
esac
