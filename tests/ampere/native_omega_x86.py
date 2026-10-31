#!/usr/bin/env python3
"""Explicit experimental #923 oracle. Host EMULATED only; never GPU evidence.

The large cubin private-ABI unit is compiled, not called: local Omega arrays
cannot supply its full 16 MiB/128 MiB view arguments. No signatures are shrunk.
"""
import argparse
import hashlib
from pathlib import Path
import subprocess

PIN = "becf5c817bcb7e0846d12ebfc2be91042d8ad89a"
COMPILER_SHA = "5f0faab7cd4fc12158cf8e099418098d278df644efc10e4e7e2e5ca6141fca86"
ROOT = Path(__file__).resolve().parents[2]
ENCODER_MAIN = """
fn main to int
ins = new ulong 8192 0
phys = new ulong 288 0
code = new uchar 16384 0
n = ENTRY view ins view phys 1 0 edit code
if ne n 16
ret 1
end
w = write 1 code 0 n
if ne w 16
ret 2
end
ret 0
end
"""
RUNTIME_MAIN = """
fn main to int
m = new ulong 16 0
q = new ulong 64 0
put m 0 2
put m 1 4096
put m 2 1
put m 3 4096
r = amp_runtime view m edit q
if ne r 0
ret 1
end
put m 3 0
r = amp_runtime view m edit q
if ne r 1
ret 2
end
ret 0
end
"""


def run(command, **kw):
    return subprocess.run(command, capture_output=True, timeout=180, **kw)


def compile_program(compiler, source, path):
    result = run([str(compiler)], input=source)
    if result.returncode:
        raise RuntimeError(f"compile refused {path.name}: {result.returncode}: {result.stderr!r}")
    path.write_bytes(result.stdout)
    path.chmod(0o755)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler-tree", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tree = args.compiler_tree.resolve()
    observed = subprocess.check_output(["git", "-C", str(tree), "rev-parse", "HEAD"], text=True).strip()
    compiler = tree / "seed/out/omc"
    if observed != PIN or hashlib.sha256(compiler.read_bytes()).hexdigest() != COMPILER_SHA:
        raise SystemExit("experimental compiler source/binary pin mismatch")
    args.output.mkdir(parents=True, exist_ok=True)
    units = (("om/encoder/bw_enc_sm121.om", "bw_enc_sm121"),
             ("om/encoder/ampere_enc_sm86.om", "ampere_enc_sm86"),
             ("om/encoder/amp_runtime.om", "amp_runtime"),
             ("om/cubin/cubin_elf.om", "cubin_elf"))
    for relative, entry in units:
        source = (ROOT / relative).read_bytes()
        bare = args.output / (entry + ".x86.unit")
        compile_program(compiler, b"target x86linux\n" + source, bare)
        print("EMULATED-ORACLE INPUT COMPILE", relative, hashlib.sha256(source).hexdigest(),
              "x86-unit-sha256", hashlib.sha256(bare.read_bytes()).hexdigest())
        if entry == "cubin_elf":
            print("NOT_RUN native-cubin private-ABI execution: full-size external buffer caller required")
            continue
        suffix = RUNTIME_MAIN if entry == "amp_runtime" else ENCODER_MAIN.replace("ENTRY", entry)
        program = source + suffix.encode()
        arm = args.output / (entry + ".arm")
        x86 = args.output / (entry + ".x86")
        compile_program(compiler, program, arm)
        compile_program(compiler, b"target x86linux\n" + program, x86)
        native = run([str(arm)])
        emulated = run(["qemu-x86_64", str(x86)])
        if native.returncode or emulated.returncode or native.stdout != emulated.stdout:
            raise RuntimeError(f"{entry} runtime parity failed: native={native.returncode}, x86={emulated.returncode}")
        mutant_suffix = suffix.replace("if ne r 1", "if ne r 0") if entry == "amp_runtime" else suffix.replace("view phys 1 0", "view phys 0 0")
        mutant = args.output / (entry + ".mutant.x86")
        compile_program(compiler, b"target x86linux\n" + source + mutant_suffix.encode(), mutant)
        negative = run(["qemu-x86_64", str(mutant)])
        expected = 2 if entry == "amp_runtime" else 1
        if negative.returncode != expected:
            raise RuntimeError(f"{entry} negative control was not caught: {negative.returncode}")
        print("EMULATED PASS", entry, "native-AArch64/x86-QEMU stdout-sha256",
              hashlib.sha256(emulated.stdout).hexdigest(), "negative-exit", negative.returncode)
    print("EXPERIMENTAL NATIVE X86 SMALL-UNIT PROBES PASS; native-cubin runtime NOT_RUN; device NOT_RUN")


if __name__ == "__main__":
    main()
