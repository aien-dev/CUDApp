#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Host-only chip-helper checks: every child is a dummy, every lock is temporary."""
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(os.environ.get("CUDAPP_CHIP_HELPER_ROOT", Path(__file__).resolve().parents[1]))


class ChipHelperTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="cudapp-chip-helpers-")
        self.addCleanup(self.tmp.cleanup)
        self.work = Path(self.tmp.name)
        self.calls = self.work / "calls"
        self.lock = self.work / "lock"
        self.out = self.work / "evidence"
        self.out.mkdir()
        self.home = self.work / "home"
        repo = self.home / "workspace/omega-gpu-channels"
        base = self.home / "workspace/omega-gpu-channels-base/build"
        (repo / "build").mkdir(parents=True)
        base.mkdir(parents=True)
        subprocess.run(["git", "init", "-q", str(repo)], check=True)
        subprocess.run(["git", "-C", str(repo), "-c", "user.name=Test", "-c",
                        "user.email=test@example.invalid", "commit", "--allow-empty",
                        "-qm", "host fixture"], check=True)
        self.env = dict(os.environ, HOME=str(self.home), LOCK=str(self.lock),
                        CALLS=str(self.calls), REAL_FLOCK=shutil.which("flock") or "",
                        CHILD_MODE="pass", AIEN_OMEGA_GPU_CHANNELS="unexpected")
        self.assertTrue(self.env["REAL_FLOCK"], "flock is required")
        dummy = '''#!/bin/sh
held=locked
if "$REAL_FLOCK" -n "$LOCK" true; then held=unlocked; fi
printf '%s %s %s\\n' "$(basename "$0")" "${AIEN_OMEGA_GPU_CHANNELS-unset}" "$held" >> "$CALLS"
case "$CHILD_MODE" in
    fail) echo 'dummy child failure'; exit 37 ;;
    late) if [ "${AIEN_OMEGA_GPU_CHANNELS-unset}" = 2 ]; then
              echo 'dummy child failure'; exit 37
          fi ;;
    silent) echo 'dummy child without a verdict'; exit 0 ;;
    pass) echo 'C4 streams chip: PASS (dummy host test, not GPU evidence)' ;;
esac
'''
        for folder in (repo / "build", base):
            for name in ("gpu_streams_test", "gpu_q38_graph_test", "gpu_cmdlist_test",
                         "gpu_decode_graph_test", "gpu_q38_channels_test"):
                path = folder / name
                path.write_text(dummy)
                path.chmod(0o755)
        self.env["BIN"] = str(repo / "build/gpu_streams_test")

    def run_helper(self, name, *args, **env):
        return subprocess.run(["sh", str(ROOT / "tools" / name), str(self.out), *args],
                              env=dict(self.env, **env), text=True, capture_output=True,
                              timeout=15)

    def invocations(self):
        return self.calls.read_text().splitlines() if self.calls.exists() else []

    def test_c4_refuses_zero_without_running_a_child(self):
        result = self.run_helper("c4_streams_chip.sh", "0")
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.invocations(), [])

    def test_c4_refuses_malformed_or_unrepresentable_counts(self):
        for count in ("", "00", "-1", "abc", "1.5", "999999999999999999999999999"):
            with self.subTest(count=count):
                self.calls.unlink(missing_ok=True)
                result = self.run_helper("c4_streams_chip.sh", count)
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(self.invocations(), [])

    def test_c4_runs_the_requested_number_under_the_lock(self):
        result = self.run_helper("c4_streams_chip.sh", "2")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.invocations(), ["gpu_streams_test unexpected locked"] * 2)
        self.assertIn("runs passing: 2 of 2", (self.out / "summary.txt").read_text())

    def test_c4_keeps_the_default_count_when_omitted(self):
        result = self.run_helper("c4_streams_chip.sh")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(len(self.invocations()), 10)

    def test_c4_rejects_child_failure_and_missing_verdict(self):
        for mode in ("fail", "silent"):
            with self.subTest(mode=mode):
                result = self.run_helper("c4_streams_chip.sh", "1", CHILD_MODE=mode)
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_l136_holds_the_lock_for_every_child(self):
        result = self.run_helper("l136_channel_gates.sh")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(len(self.invocations()), 11)
        self.assertTrue(all(line.endswith(" locked") for line in self.invocations()),
                        self.invocations())
        self.assertEqual([line.split()[1] for line in self.invocations()],
                         ["unset"] * 9 + ["2", "2"])

    def test_l136_records_failure_and_stops_with_the_child_status(self):
        result = self.run_helper("l136_channel_gates.sh", CHILD_MODE="fail")
        self.assertEqual(result.returncode, 37, result.stdout + result.stderr)
        self.assertEqual(len(self.invocations()), 1)
        summary = (self.out / "summary.txt").read_text()
        self.assertIn("g1-base-q38-l4 rc=37 dummy child failure", summary)
        self.assertNotIn("done", summary)

    def test_l136_propagates_a_late_channel_child_failure(self):
        result = self.run_helper("l136_channel_gates.sh", CHILD_MODE="late")
        self.assertEqual(result.returncode, 37, result.stdout + result.stderr)
        self.assertEqual(len(self.invocations()), 10)
        summary = (self.out / "summary.txt").read_text()
        self.assertIn("g2-l4 rc=37 dummy child failure", summary)
        self.assertNotIn("done", summary)

    def test_contended_lock_prevents_any_child(self):
        # Keep real flock semantics; shorten only the wait to make contention
        # deterministic without waiting for the production wait budget.
        shimdir = self.work / "bin"
        shimdir.mkdir()
        shim = shimdir / "flock"
        shim.write_text('''#!/bin/sh
if [ "$1" = -w ]; then shift; shift; exec "$REAL_FLOCK" -n "$@"; fi
exec "$REAL_FLOCK" "$@"
''')
        shim.chmod(0o755)
        with self.lock.open("a") as holder:
            fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
            for name in ("c4_streams_chip.sh", "l136_channel_gates.sh"):
                with self.subTest(helper=name):
                    result = self.run_helper(name, *(["1"] if name.startswith("c4") else []),
                                             PATH=str(shimdir) + os.pathsep + os.environ["PATH"])
                    self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertEqual(self.invocations(), [])


if __name__ == "__main__":
    unittest.main()
