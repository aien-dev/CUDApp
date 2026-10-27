#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Host-only CLI refusal tests. Invalid selectors must never reach device work."""
import pathlib
import subprocess
import sys
import tempfile


def refused(run, receipt):
    text = run.stdout + run.stderr
    return (run.returncode == 2 and "INVALID" in text
            and "PASS" not in text and receipt == "unchanged\n")


def main():
    build = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "build").resolve()
    failures = 0
    # Negative control: a zero-case PASS and a refusal that overwrites a
    # receipt must both be rejected by the same check used below.
    fake_pass = subprocess.CompletedProcess([], 0, "GPU_GEMM: PASS\n", "")
    fake_refusal = subprocess.CompletedProcess([], 2, "", "INVALID shape\n")
    if refused(fake_pass, "unchanged\n") or refused(fake_refusal, "overwritten\n"):
        print("FAIL selector checker negative control")
        return 1
    print("PASS selector checker rejects zero-case PASS and overwritten receipt")
    with tempfile.TemporaryDirectory(prefix="gemm-selectors-") as tmp:
        for name in ("pfgemm", "qgemm", "fp8gemm"):
            binary = build / ("gpu_" + name + "_test")
            modes = [[], ["--host-only"], ["--sim"]]
            if name != "fp8gemm":
                modes[0] = ["--chip"]
            for mode in modes:
                for selector in ("typo", "", "GATEUP"):
                    out = pathlib.Path(tmp) / "receipt.json"
                    out.write_text("unchanged\n")
                    args = [str(binary), *mode, "--shape", selector]
                    if name == "fp8gemm":
                        args += ["--out", str(out)]
                    run = subprocess.run(args, capture_output=True, text=True)
                    text = run.stdout + run.stderr
                    ok = refused(run, out.read_text())
                    if not ok:
                        failures += 1
                        print(f"FAIL {name} {mode} shape={selector!r}: rc={run.returncode} {text.strip()}")
            # A known selector must remain accepted by the real host path, not
            # by a new validation-only surrogate or any GPU-capable mode.
            selectors = ["gateup", "down", "qkv", "o"]
            if name == "fp8gemm":
                selectors += ["all", "d768"]
            for selector in selectors:
                run = subprocess.run([str(binary), "--host-only", "--shape", selector],
                                     capture_output=True, text=True)
                if run.returncode or "PASS" not in run.stdout:
                    failures += 1
                    print(f"FAIL {name} valid host control {selector}: rc={run.returncode}\n{run.stdout}{run.stderr}")
            if name != "fp8gemm":
                run = subprocess.run([str(binary), "--host-only", "--sim"],
                                     capture_output=True, text=True)
                if run.returncode != 2 or "NOT_RUN" not in run.stderr or "PASS" in run.stdout:
                    failures += 1
                    print(f"FAIL {name} empty mode selection: rc={run.returncode}")
    print(f"GEMM_SELECTOR_TEST: {'FAIL' if failures else 'PASS'} ({failures} failures; host only)")
    return bool(failures)


if __name__ == "__main__":
    sys.exit(main())
