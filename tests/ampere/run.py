#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Explicit device-only Ampere checks. Never substitutes simulation or updates old receipts."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", action="store_true", help="authorize submissions on this machine")
    parser.add_argument("--bdf", required=True, help="exact GA102/GA104 full PCI BDF")
    parser.add_argument("--output", required=True, type=Path, help="new evidence directory; existing paths refuse")
    parser.add_argument("--build", type=Path, default=ROOT / "build")
    parser.add_argument("--full-math", action="store_true", help="all binary32 inputs per unary function; otherwise NOT_RUN")
    args = parser.parse_args()
    if not args.device:
        parser.error("explicit --device required; no driver was opened")
    args.output.mkdir(parents=True, exist_ok=False)
    names = ("doctor", "hello", "hello-negative", "math-smoke", "math-negative", "math-exhaustive", "memory-refusal")
    receipt = {"schema": 1, "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "bdf": args.bdf, "target": "sm_86", "evidence": "DEVICE",
               "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
               "checks": {name: {"status": "NOT_RUN"} for name in names}}
    def save():
        (args.output / "results.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    save()
    env = dict(os.environ, CUDAPP_GPU_BDF=args.bdf)
    def run(name, command, expected=0, marker=None):
        check = receipt["checks"][name]
        check["command"] = [str(x) for x in command]
        save()
        try:
            result = subprocess.run(check["command"], cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        except OSError as error:
            check.update(status="FAIL", error=str(error)); save(); return False
        (args.output / (name + ".log")).write_text(result.stdout)
        ok = result.returncode == expected and (marker is None or marker in result.stdout)
        check.update(status="PASS" if ok else "FAIL", exit_code=result.returncode, log=name + ".log")
        save()
        print(name + ": " + check["status"], flush=True)
        return ok
    if not run("doctor", [sys.executable, ROOT / "tests/ampere/doctor.py", "--bdf", args.bdf], marker='"preflight": "PASS"'):
        return 1
    build = args.build.resolve()
    for binary in ("hello_gpu", "hello_gpu_control", "ampere_math_device", "ampere_memory_refusal"):
        if not (build / binary).is_file():
            print("Missing device test binary: " + str(build / binary), file=sys.stderr)
            return 1
    sequence = [
        ("hello", [build / "hello_gpu", "--target=sm_86"], 0, "checked 1048576 of 1048576 elements: 0 wrong, 0 never written"),
        ("hello-negative", [build / "hello_gpu_control", "--target=sm_86", "--bug=short"], 1, "checked 1048576 of 1048576 elements: 1 wrong, 1 never written"),
        ("memory-refusal", [build / "ampere_memory_refusal", "--device"], 0, "AMPERE MEMORY REFUSAL PASS"),
        ("math-smoke", [build / "ampere_math_device", "--device", "--count", "65536"], 0, "SHARD PASS"),
        ("math-negative", [build / "ampere_math_device", "--device", "--count", "65536", "--inject-output-error"], 1, "AMPERE MATH FAIL op=rcp input=00000000"),
    ]
    for name, command, expected, marker in sequence:
        if not run(name, command, expected, marker):
            return 1
    if args.full_math and not run("math-exhaustive", [build / "ampere_math_device", "--device"], marker="AMPERE MATH log2 EXHAUSTIVE PASS"):
        return 1
    print("All requested checks passed. Unrequested checks remain NOT_RUN; no qualification registry was changed.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
