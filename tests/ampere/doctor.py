#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Read-only qualification preflight. Never opens a GPU or changes its driver."""
import argparse
import json
import os
from pathlib import Path
import platform
import re
import sys

# NVIDIA open-gpu-kernel-modules 580.173.02 README Compatible GPUs table.
# Only GA102/GA104 IDs requested by this port. Other 3060 Ti die variants refuse.
CARDS = {"0x2204": ("RTX 3090", "GA102"),
         "0x2486": ("RTX 3060 Ti", "GA104"),
         "0x2489": ("RTX 3060 Ti", "GA104")}
ABI = "580.173.02"

def inspect(root=Path("/"), bdf=None, machine=None):
    simulated = root.resolve() != Path("/")
    problems = []
    def text(name):
        try:
            return (root / name).read_text().strip()
        except OSError:
            return ""
    machine = machine or platform.machine()
    if machine != "x86_64":
        problems.append("host must be Linux x86_64 for this device plan")
    if platform.system() != "Linux":
        problems.append("host OS must be Linux")
    module = text("proc/driver/nvidia/version")
    open_module = "Open Kernel Module" in module
    version = re.search(r"\b([0-9]+\.[0-9]+\.[0-9]+)\b", module)
    version = version.group(1) if version else None
    if not open_module:
        problems.append("NVIDIA open kernel module must be loaded")
    if version != ABI:
        problems.append("driver ABI must match vendored " + ABI + "; do not force another version")
    cards = []
    for device in sorted((root / "sys/bus/pci/devices").glob("*")):
        try:
            vendor = (device / "vendor").read_text().strip().lower()
            kind = (device / "device").read_text().strip().lower()
            pci_class = (device / "class").read_text().strip().lower()
        except OSError:
            continue
        if vendor != "0x10de" or not pci_class.startswith("0x03"):
            continue
        if bdf and device.name != bdf:
            continue
        identity = CARDS.get(kind)
        cards.append({"bdf": device.name, "pci_device": kind,
                      "name": identity[0] if identity else "unsupported NVIDIA GPU",
                      "die": identity[1] if identity else None,
                      "target": "sm_86" if identity else None})
    if len(cards) != 1:
        problems.append("select exactly one target GPU with --bdf (or attach exactly one)")
    elif cards[0]["target"] != "sm_86":
        problems.append("selected PCI ID is not a requested GA102/GA104 target")
    nodes = ["nvidiactl", "nvidia-uvm"]
    if len(cards) == 1:
        info = text("proc/driver/nvidia/gpus/" + cards[0]["bdf"] + "/information")
        minor = re.search(r"^Device Minor:\s*([0-9]+)\s*$", info, re.MULTILINE)
        if not minor:
            problems.append("selected GPU has no driver Device Minor mapping")
        else:
            cards[0]["device_node"] = "/dev/nvidia" + minor.group(1)
            nodes.append("nvidia" + minor.group(1))
    for node in nodes:
        path = root / "dev" / node
        if not path.exists() or not os.access(path, os.R_OK | os.W_OK):
            problems.append("missing/inaccessible /dev/" + node)
    return {"preflight": "FAIL" if problems else "PASS", "evidence": "SIMULATED" if simulated else "OBSERVED",
            "device_execution": "NOT_RUN", "machine": machine, "driver_version": version,
            "open_module": open_module, "gpus": cards, "problems": problems,
            "note": "Preflight PASS is not hardware qualification or permission to submit work."}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/"), help="fake machine root; results labeled SIMULATED")
    parser.add_argument("--bdf", help="selected full PCI address, e.g. 0000:01:00.0")
    args = parser.parse_args()
    report = inspect(args.root, args.bdf)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["preflight"] == "PASS" else 1

if __name__ == "__main__":
    sys.exit(main())
