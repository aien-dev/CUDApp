#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""SIMULATED filesystem preflight controls. No GPU driver calls."""
import importlib.util
from pathlib import Path
import tempfile

spec = importlib.util.spec_from_file_location("ampere_doctor", Path(__file__).with_name("doctor.py"))
doctor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(doctor)

def put(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value)

with tempfile.TemporaryDirectory(prefix="ampere-doctor-") as directory:
    root = Path(directory)
    def result():
        return doctor.inspect(root, machine="x86_64")
    put(root, "proc/driver/nvidia/version", "NVRM version: NVIDIA UNIX Open Kernel Module for x86_64 580.173.02\n")
    for node in ("nvidia0", "nvidiactl", "nvidia-uvm"):
        put(root, "dev/" + node, "")
    gpu = "sys/bus/pci/devices/0000:01:00.0/"
    put(root, gpu + "vendor", "0x10de")
    put(root, gpu + "class", "0x030000")
    put(root, "proc/driver/nvidia/gpus/0000:01:00.0/information", "Device Minor: 0\n")
    for pci in doctor.CARDS:
        put(root, gpu + "device", pci)
        report = result()
        assert report["preflight"] == "PASS", report
        assert report["evidence"] == "SIMULATED" and report["device_execution"] == "NOT_RUN"
    put(root, gpu + "device", "0x2e12")
    assert result()["preflight"] == "FAIL", "negative control: GB10 must refuse"
    put(root, gpu + "device", "0x2204")
    put(root, "proc/driver/nvidia/version", "NVRM version: NVIDIA UNIX x86_64 Kernel Module 580.173.02\n")
    assert result()["preflight"] == "FAIL", "negative control: closed module must refuse"
    put(root, "proc/driver/nvidia/version", "NVRM version: NVIDIA UNIX Open Kernel Module 590.00.00\n")
    assert result()["preflight"] == "FAIL", "negative control: mismatched driver must refuse"
    put(root, "proc/driver/nvidia/version", "NVRM version: NVIDIA UNIX Open Kernel Module 580.173.02\n")
    (root / "dev/nvidia-uvm").unlink()
    assert result()["preflight"] == "FAIL", "negative control: missing UVM must refuse"
    put(root, "dev/nvidia-uvm", "")
    for key, value in (("vendor", "0x10de"), ("class", "0x030000"), ("device", "0x2486")):
        put(root, "sys/bus/pci/devices/0000:02:00.0/" + key, value)
    assert result()["preflight"] == "FAIL", "ambiguous GPU must refuse"
    assert doctor.inspect(root, "0000:01:00.0", "x86_64")["preflight"] == "PASS"
    assert doctor.inspect(root, "0000:01:00.0", "aarch64")["preflight"] == "FAIL"
print("AMPERE DOCTOR SIMULATED PASS (target IDs and refusal controls); DEVICE NOT_RUN")
