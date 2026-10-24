#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Remove the shared-lock guard in a temporary copy; its test must fail."""
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


root = Path(__file__).resolve().parents[2]
with tempfile.TemporaryDirectory(prefix="localrouter-control-") as name:
    temp = Path(name)
    for relative in ("tools/localrouter/setup.py", "tests/localrouter/test_setup.py"):
        target = temp / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / relative, target)
    script = temp / "tools/localrouter/setup.py"
    text = script.read_text()
    before = 'require(cfg.get("gpu_lock") == "",'
    if text.count(before) != 1:
        sys.exit("FAIL: shared-lock guard was not found exactly once")
    script.write_text(text.replace(before, 'require(True,'))
    run = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s",
                          "tests/localrouter", "-p", "test_setup.py", "-v"],
                         cwd=temp, capture_output=True, text=True)
    output = run.stdout + run.stderr
    # A crash/import failure cannot count as detecting this control.
    if run.returncode == 0 or "AssertionError: ValueError not raised" not in output or "FAILED (failures=1)" not in output:
        print(output)
        sys.exit("FAIL: removing the shared-lock guard did not fail its intended check")
    print("PASS: shared-lock guard removed in scratch copy; intended rejection test failed")
    print("mutant exit code:", run.returncode)
