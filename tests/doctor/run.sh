#!/bin/sh
# SPDX-License-Identifier: AGPL-3.0-only
# Host test for `cudapp-doctor` (CUDA++ developer bar D2 and D6, docs/PAIN_POINTS.md).
#   usage: tests/doctor/run.sh build/cudapp-doctor      (CUDAPP_DOCTOR_SHOW=1 also prints what each fake machine is told)
# No GPU and no device file is opened: every case is a fake machine built under a temporary
# folder and handed to the doctor through CUDAPP_DOCTOR_ROOT. Each case states the exit code
# and the sentences a person must see. A case that passes with the wrong text is a failure.
set -u
DOC=${1:?usage: run.sh path/to/cudapp-doctor}
T=$(mktemp -d "${TMPDIR:-/tmp}/cudapp-doctor-test.XXXXXX")
trap 'rm -rf "$T"' EXIT
pass=0; fail=0

# make_machine DIR [module-version-line|none] [pci-device-id|none] [dev nodes...]
make_machine() {
	d=$1; ver=$2; pci=$3; shift 3
	mkdir -p "$d/proc/driver/nvidia" "$d/sys/bus/pci/devices" "$d/dev"
	if [ "$ver" != none ]; then
		printf 'NVRM version: %s\nGCC version:  gcc version 13.3.0\n' "$ver" > "$d/proc/driver/nvidia/version"
	else
		rmdir "$d/proc/driver/nvidia"
	fi
	if [ "$pci" != none ]; then
		g="$d/sys/bus/pci/devices/000f:01:00.0"; mkdir -p "$g"
		echo 0x10de > "$g/vendor"; echo "$pci" > "$g/device"; echo 0x030000 > "$g/class"
	fi
	# an unrelated non-NVIDIA device the doctor must ignore
	o="$d/sys/bus/pci/devices/0000:00:00.0"; mkdir -p "$o"; echo 0x8086 > "$o/vendor"; echo 0x1234 > "$o/device"; echo 0x060000 > "$o/class"
	printf 'MemTotal:       127535484 kB\nMemFree:        1000000 kB\nMemAvailable:   42000000 kB\n' > "$d/proc/meminfo"
	for n in "$@"; do : > "$d/dev/$n"; chmod 666 "$d/dev/$n"; done
}
OPEN='NVIDIA UNIX Open Kernel Module for aarch64  580.173.02  Release Build  (test)  Tue Jun 23 08:34:19 UTC 2026'
NODES='nvidiactl nvidia0 nvidia-uvm'

# check NAME WANT_EXIT ROOT then one grep -F pattern per argument (all must appear)
check() {
	name=$1; want=$2; root=$3; shift 3
	out=$(CUDAPP_DOCTOR_ROOT="$root" "$DOC" 2>&1); got=$?
	[ -n "${CUDAPP_DOCTOR_SHOW:-}" ] && { echo "===== fake machine: $name (exit $got)"; printf "%s\n" "$out"; }
	ok=1
	[ "$got" -eq "$want" ] || { echo "  exit $got, want $want"; ok=0; }
	for pat in "$@"; do
		printf '%s\n' "$out" | grep -qF -- "$pat" || { echo "  missing: $pat"; ok=0; }
	done
	# D6: never the bare CUDA message, in any case
	printf '%s\n' "$out" | grep -qi 'no kernel image' && ! printf '%s\n' "$out" | grep -qF 'instead of failing later' && { echo "  bare 'no kernel image' text"; ok=0; }
	if [ $ok -eq 1 ]; then pass=$((pass+1)); echo "PASS $name"; else fail=$((fail+1)); echo "FAIL $name"; printf '%s\n' "$out" | sed 's/^/  | /'; fi
}

# 1. a ready GB10
make_machine "$T/gb10" "$OPEN" 0x2e12 $NODES
check gb10-ready 0 "$T/gb10" \
	'ok    NVIDIA kernel module: open module 580.173.02' \
	'ok    GPU 000f:01:00.0: NVIDIA GB10 (sm_121), qualified' \
	'evidence/m16-blackwell-native-path-requalification-receipt.json' \
	'ok    device files: /dev/nvidiactl /dev/nvidia0 /dev/nvidia-uvm' \
	'CUDA toolkit: not needed' \
	'READY. Next: make hello-gpu'

# 2. no NVIDIA kernel module loaded
make_machine "$T/nomod" none 0x2e12
check no-module 1 "$T/nomod" \
	'FAIL  NVIDIA kernel module: not loaded' \
	'sudo modprobe nvidia' \
	'NOT READY'

# 3. a different module version: our ioctl structures come from the 580.173.02 headers
make_machine "$T/oldmod" 'NVIDIA UNIX Open Kernel Module for aarch64  575.64.05  Release Build  (test)' 0x2e12 $NODES
check module-version 1 "$T/oldmod" \
	'FAIL  NVIDIA kernel module: version 575.64.05 is loaded; CUDA++ speaks the 580.173.02 interface' \
	'refuses instead of guessing' \
	'NOT READY'

# 4. the proprietary (closed) module at the right version: allowed, with a warning
make_machine "$T/closed" 'NVIDIA UNIX aarch64 Kernel Module  580.173.02  Release Build  (test)' 0x2e12 $NODES
check closed-module 0 "$T/closed" \
	'warn  NVIDIA kernel module: the closed module 580.173.02' \
	'READY. Next: make hello-gpu'

# 5. D6: a GPU CUDA++ has no machine code for. Refused with what is missing and what to do.
make_machine "$T/newchip" "$OPEN" 0x2b85 $NODES
check unknown-chip 1 "$T/newchip" \
	'FAIL  GPU 000f:01:00.0: NVIDIA device 0x2b85 is not one CUDA++ has qualified' \
	'instead of failing later with "no kernel image is available"' \
	'What is missing:' \
	'docs/runtime/PORTABILITY.md section 5' \
	'NOT READY'

# 6. no NVIDIA GPU at all
make_machine "$T/nogpu" "$OPEN" none $NODES
check no-gpu 1 "$T/nogpu" \
	'FAIL  GPU: no NVIDIA GPU found in /sys/bus/pci/devices' \
	'NOT READY'

# 7. the unified-memory device file is missing
make_machine "$T/nouvm" "$OPEN" 0x2e12 nvidiactl nvidia0
check no-uvm 1 "$T/nouvm" \
	'FAIL  device files: /dev/nvidia-uvm is missing' \
	'sudo modprobe nvidia-uvm' \
	'NOT READY'

# 8. a device file this user may not open
make_machine "$T/perm" "$OPEN" 0x2e12 $NODES
chmod 600 "$T/perm/dev/nvidia0"; chmod 000 "$T/perm/dev/nvidia0"
if [ -r "$T/perm/dev/nvidia0" ]; then
	echo "SKIP perm (running as root: every file is readable)"
else
	check no-permission 1 "$T/perm" \
		'FAIL  device files: /dev/nvidia0 exists but this user cannot open it for reading and writing' \
		'NOT READY'
fi

# 9. the memory line on a unified-memory chip
check memory-line 0 "$T/gb10" 'ok    memory: 121.6 GiB shared by the CPU and the GPU, 40.1 GiB available now'

echo "doctor tests: $pass passed, $fail failed"
[ "$fail" -eq 0 ]
