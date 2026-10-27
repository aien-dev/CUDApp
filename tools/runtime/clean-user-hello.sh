#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# clean-user-hello.sh BRANCH LOG [URL]   (bar C2: a stranger runs hello-GPU from a clean machine)
#
# Creates a brand-new local Linux user (empty home, no ~/workspace, empty environment), runs ONLY the commands of
# the README "Run your first kernel" bash block of BRANCH as that user, and deletes the user again. Needs sudo; every
# privileged command is printed with a leading "+" before it runs. The whole README block runs under strace (as
# root, following the stranger's processes) so the log can show that no CUDA file was opened or executed; a red
# control (`nvcc --version` as the same user, same trace) shows the trace check can see one.
#
# The GPU is used only while holding the lock ($CUDAPP_GPU_LOCK, default /tmp/aien-gb10.lock). If $CUDAPP_BOARD
# names a file, GPU start and end lines are appended to it. Exit code: that of the README block (0 = every command
# passed), 3 = setup failure, 4 = the trace saw a CUDA file opened or executed, 5 = the trace red control failed.
set -uo pipefail
branch=${1:?usage: clean-user-hello.sh BRANCH LOG [URL]}
log=${2:?usage: clean-user-hello.sh BRANCH LOG [URL]}
url=${3:-https://github.com/aien-dev/CUDApp}
lock=${CUDAPP_GPU_LOCK:-/tmp/aien-gb10.lock}
board=${CUDAPP_BOARD:-}
U=c2stranger$(date -u +%H%M%S)
cd / || exit 3
run() { echo "+ $*"; "$@"; }
note() { [ -n "$board" ] && echo "$(date -u +%FT%TZ) $*" >> "$board"; return 0; }
tr_dir=$(mktemp -d "${TMPDIR:-/tmp}/clean-user-trace.XXXXXX")
exec > >(tee "$log") 2>&1
echo "# clean-user-hello: branch=$branch url=$url user=$U"
echo "# date: $(date -u +%FT%TZ)  host: $(uname -srm)  loadavg: $(cut -d' ' -f1-3 /proc/loadavg)"
echo "# operator: $(id -un)  nvidia module: $(cat /sys/module/nvidia/version 2>/dev/null || echo none)"
cleanup() {
	if sudo test -s "$tr_dir/readme.trace"; then sudo gzip -9c "$tr_dir/readme.trace" > "$log.strace.gz" && echo "# full trace of the README block kept: $log.strace.gz"; fi
	echo "# --- delete the temporary user ---"
	run sudo pkill -u "$U" || true
	run sudo userdel -r "$U" 2>&1 | grep -v 'mail spool' || true
	sudo rm -rf "$tr_dir"
	if getent passwd "$U" >/dev/null || [ -e "/home/$U" ]; then echo "# CLEANUP FAILED: $U still present"; else echo "# user $U and /home/$U removed"; fi
}
trap cleanup EXIT
echo "# --- create a brand-new user ---"
run sudo useradd -m -s /bin/bash "$U" || exit 3
# the stranger's shell: an empty environment except what login(1) sets, then the distribution's login profile
S() { sudo -u "$U" env -i HOME="/home/$U" USER="$U" LOGNAME="$U" SHELL=/bin/bash TERM=dumb LANG=C.UTF-8 PATH=/usr/local/bin:/usr/bin:/bin bash -lc "cd ~; $1"; }
echo "# new user:"
S 'echo "#   id: $(id)"; echo "#   home: [$(ls -A ~ | tr "\n" " ")]"; [ -e ~/workspace ] && echo "#   ~/workspace EXISTS" || echo "#   ~/workspace absent"'
echo "# environment after login (env | sort):"
S 'env | sort | sed "s/^/#   /"'
echo "# --- the README of $branch, as a stranger reads it on GitHub ---"
readme=$(curl -fsSL "https://raw.githubusercontent.com/aien-dev/CUDApp/$branch/README.md") || { echo "FAIL: cannot fetch the README"; exit 3; }
cmds=$(printf '%s\n' "$readme" | awk '/^## Run your first kernel/{s=1;next} s&&/^```bash/{b=1;next} s&&b&&/^```/{exit} s&&b{print}')
[ -n "$cmds" ] || { echo "FAIL: no bash block under 'Run your first kernel'"; exit 3; }
echo "# README commands, verbatim:"; printf '%s\n' "$cmds" | sed 's/^/#   /'
# the one change: the clone takes the branch under test (none for main)
[ "$branch" = main ] || cmds=$(printf '%s\n' "$cmds" | sed -E "s#^git clone #git clone --branch $branch #")
script=$(printf 'set -e\nmkdir empty && cd empty\necho "# cwd: $(pwd), contents: [$(ls -A)]"\n'
	printf '%s\n' "$cmds" | awk -v sq="'" '{ q = $0; gsub(sq, sq "\\" sq sq, q); print "echo " sq "$ " q sq; print $0
		if ($0 ~ /^cd /) print "echo \"# commit: $(git rev-parse HEAD)\"" }')
# traced: execve and every file open, all processes of the stranger, recorded by root
T() { sudo strace -f -qq -e trace=execve,openat -o "$2" sudo -u "$U" env -i HOME="/home/$U" USER="$U" LOGNAME="$U" SHELL=/bin/bash TERM=dumb LANG=C.UTF-8 PATH=/usr/local/bin:/usr/bin:/bin bash -lc "cd ~; $1"; }
# a CUDA file that was really opened or executed: a successful call (no "= -1") on the CUDA toolkit tree, a CUDA or
# NVIDIA JIT user-space library, or an nvcc. The zero-CUDA gate's own red control builds a stub libcudart.so in a
# mktemp directory (/tmp/tmp.*) on purpose; those calls are counted apart and are not a CUDA use.
CUDA_RE='"(/usr/local/cuda[^"]*|/opt/nvidia[^"]*|[^"]*/(libcuda|libcudart|libcublas|libcudnn|libnvrtc|libnvJitLink|libnvidia-ptxjitcompiler)[^"/]*|[^"]*/nvcc)"'
cuda_hits() { sudo grep -E '(execve|openat)\(' "$1" | grep -vE '= -1 ' | grep -E "$CUDA_RE" | grep -vE '"/tmp/tmp\.[A-Za-z0-9]+/' || true; }
stub_hits() { sudo grep -E '(execve|openat)\(' "$1" | grep -vE '= -1 ' | grep -E "$CUDA_RE" | grep -cE '"/tmp/tmp\.[A-Za-z0-9]+/' || true; }
echo "# --- run the README block as $U, holding $lock ---"
note "GPU-START clean-user-hello $U branch=$branch (flock $lock)"
t0=$(date +%s)
echo "+ flock -w 900 $lock strace -f ... sudo -u $U (README block)"
export -f T; export U
flock -w 900 "$lock" bash -c 'T "$1" "$2"' _ "$script" "$tr_dir/readme.trace"
rc=$?
echo "# README block exit code: $rc  ($(( $(date +%s) - t0 )) s including the lock wait)"
note "GPU-END clean-user-hello $U rc=$rc, GPU released"
echo "# --- trace check: CUDA files opened or executed during the README block ---"
echo "#   traced calls: $(sudo grep -cE '(execve|openat)\(' "$tr_dir/readme.trace"), programs executed: $(sudo grep -E 'execve\(' "$tr_dir/readme.trace" | grep -vcE '= -1 ')"
hits=$(cuda_hits "$tr_dir/readme.trace")
if [ -n "$hits" ]; then echo "# TRACE FAIL: CUDA files used:"; printf '%s\n' "$hits" | head -20 | sed 's/^/#   /'; [ $rc -eq 0 ] && rc=4
else echo "# TRACE PASS: no CUDA file (/usr/local/cuda*, libcuda*, libnvrtc*, nvcc, ...) was opened or executed"; fi
echo "#   (the zero-CUDA gate's own stub libcudart in /tmp/tmp.*: $(stub_hits "$tr_dir/readme.trace") successful calls, its red control)"
echo "# --- trace red control: the same check on 'nvcc --version' run by the same user ---"
nvcc_bin=$(S 'command -v nvcc' 2>/dev/null); [ -n "$nvcc_bin" ] || { [ -x /usr/local/cuda/bin/nvcc ] && nvcc_bin=/usr/local/cuda/bin/nvcc; }
if [ -z "$nvcc_bin" ]; then echo "# TRACE RED CONTROL: NOT_RUN (no CUDA toolkit on this machine to trace)"; else
T "$nvcc_bin --version" "$tr_dir/red.trace" > /dev/null 2>&1
red=$(cuda_hits "$tr_dir/red.trace")
if [ -n "$red" ]; then echo "# TRACE RED CONTROL: caught as it must ($(printf '%s\n' "$red" | wc -l) CUDA file uses, first: $(printf '%s\n' "$red" | head -1 | cut -c1-160))"
else echo "# TRACE RED CONTROL BROKEN: the run of $nvcc_bin was not seen"; [ $rc -eq 0 ] && rc=5; fi; fi
echo "# clean-user-hello exit code: $rc"
exit $rc
