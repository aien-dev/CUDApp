#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Verifier only: compare every decoded instruction/operand against the SM121
reference, with the explicit documented target ABI/ISA spelling differences.
NVIDIA output is ephemeral; no driver is loaded and no oracle listings committed.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import re
import struct
import subprocess
import tempfile


def decode(exe, target, data):
    with tempfile.NamedTemporaryFile(suffix=".bin") as f:
        f.write(data)
        f.flush()
        result = subprocess.run([exe, "-b", target, "-hex", f.name], text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode:
        raise RuntimeError(f"nvdisasm {target} refused: {result.stderr.strip()}")
    instructions = re.findall(r"/\*[0-9a-fA-F]+\*/\s+([^;\n]+);", result.stdout)
    if len(instructions) * 16 != len(data):
        raise RuntimeError(f"{target}: decoded instruction count does not cover entire buffer")
    if re.search(r"INVALID|\.WORD|\?\?", "\n".join(instructions), re.I):
        raise RuntimeError(f"{target}: invalid instruction or operand")
    return instructions


def normalize(text, target):
    text = re.sub(r"\s+", " ", text.strip())
    text = re.sub(r"desc\[UR(?:Z|\d+)\]", "", text)
    text = text.replace("+URZ", "")
    text = text.replace("LDCU", "ULDC")
    if target == "SM121" and text.startswith("ULDC"):
        # SM121's LDCU uses an explicit zero uniform base; SM86's ULDC
        # encodes a direct cbank byte offset.
        text = text.replace("[URZ+", "[").replace("[URZ]", "[0x0]")
        if re.fullmatch(r"ULDC\.64 UR\d+, c\[0x0\]\[0x358\]", text):
            # Reserved Blackwell descriptor load: not consumed on Ampere,
            # whose shorter CB0 must not be read at this offset.
            return "NOP"
    if text.startswith("F2FP."):
        text = text.replace(".F32", "")
    if target == "SM121" and re.search(r"\b(?:LDC|ULDC)(?:\.| )", text):
        def parameter(match):
            offset = int(match[1], 16)
            if offset >= 0x380:
                offset = offset - 0x380 + 0x160
            return f"c[0x0][0x{offset:x}]"
        text = re.sub(r"c\[0x0\]\[0x([0-9a-f]+)\]", parameter, text)
    return text


def compare(exe, reference, actual, ops=()):
    expected = [normalize(x, "SM121") for x in decode(exe, "SM121", reference)]
    # Op21's IR contract is MMIO.GPU. Historical SM121 incorrectly emits EF;
    # repair that legacy encoder separately, never treat the two as aliases.
    # This target-specific expectation retains every register/address operand.
    for i, op in enumerate(ops):
        if op == 21:
            expected[i] = re.sub(r"\bLDG\.E\.EF\b", "LDG.E.MMIO.GPU", expected[i])
        if op == 14:
            # SM86 omits the two discarded carry outputs; SM121 prints PT.
            # Real predicate outputs are not removed and must still match.
            expected[i] = expected[i].replace(", PT, PT,", ",", 1)
        if op in (54, 55):
            # SM86 BSSY/BSYNC are reconvergence-barrier operations. SM121's
            # explicit RECONVERGENT qualifier is not an SM86 mnemonic;
            # retain each barrier register, guard, and exact join address.
            expected[i] = re.sub(r"\b(BSSY|BSYNC)\.RECONVERGENT\b", r"\1", expected[i])
    observed = [normalize(x, "SM86") for x in decode(exe, "SM86", actual)]
    if expected != observed:
        for i, (a, b) in enumerate(zip(expected, observed)):
            if a != b:
                raise RuntimeError(f"instruction {i}: expected {a!r}, observed {b!r}")
        raise RuntimeError("decoded lengths differ")
    return observed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--nvdisasm", default=os.environ.get("NVDISASM", "/usr/local/cuda/bin/nvdisasm"))
    args = parser.parse_args()
    pairs = sorted(args.corpus.glob("*.sm86.bin"))
    if not pairs:
        raise RuntimeError("empty SM86 corpus")
    coverage = set()
    mutation = None
    uniform_dst_mutation = None
    mmio_mutation = None
    subtraction_mutation = None
    branch_mutation = None
    reconvergence_mutation = None
    sync_barrier_mutation = None
    carry_mutation = None
    mismatches = []
    # compare() is a pure function of (reference, code, ops); kernels that several producers
    # emit identically are decoded once, and the distinct ones run concurrently (nvdisasm is
    # a separate process per call). Every pair is still checked and reported by its own name.
    loaded = []
    for actual in pairs:
        prefix = str(actual)[:-len("sm86.bin")]
        reference = Path(prefix + "sm121.bin").read_bytes()
        code = actual.read_bytes()
        metadata = Path(prefix + "ir").read_bytes()
        if len(metadata) % 8:
            raise RuntimeError("malformed IR metadata")
        loaded.append((actual, reference, code, metadata))

    def check(key):
        reference, code, metadata = key
        ops = [op for op, _ in struct.iter_unpack("=II", metadata)]
        try:
            return compare(args.nvdisasm, reference, code, ops), None
        except RuntimeError as error:
            return None, error

    unique = list(dict.fromkeys((reference, code, metadata) for _, reference, code, metadata in loaded))
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 1) as pool:
        results = dict(zip(unique, pool.map(check, unique)))
    for actual, reference, code, metadata in loaded:
        ops = [op for op, _ in struct.iter_unpack("=II", metadata)]
        decoded, error = results[(reference, code, metadata)]
        if error is not None:
            mismatches.append(f"{actual.name}: {error}")
            continue
        coverage.update(ops)
        if mutation is None:
            for i, text in enumerate(decoded):
                if re.search(r"^(?:@!?P\d+ )?[A-Z0-9_.]+ R\d+,", text):
                    changed = bytearray(code)
                    # Corrupt a real GPR destination, not padding or an ignored bit.
                    changed[i * 16 + 2] ^= 1
                    mutation = (reference, bytes(changed), ops)
                    break
        if uniform_dst_mutation is None:
            for i, text in enumerate(decoded):
                if text == "IMAD R12, R8, UR2, R16":
                    changed = bytearray(code)
                    # Reintroduce the old bug: a uniform SOURCE selected UR2's
                    # physical index for the GPR destination instead of R12.
                    changed[i * 16 + 2] = 2
                    uniform_dst_mutation = (reference, bytes(changed), ops)
                    break
        if mmio_mutation is None and 21 in ops:
            changed = bytearray(code)
            i = ops.index(21)
            # Historical EF word must not satisfy the Ampere MMIO contract.
            struct.pack_into("<I", changed, i * 16 + 8, 202250496)
            mmio_mutation = (reference, bytes(changed), ops)
        if subtraction_mutation is None and 31 in ops:
            changed = bytearray(code)
            changed[ops.index(31) * 16 + 7] &= 0x7f
            subtraction_mutation = (reference, bytes(changed), ops)
        if branch_mutation is None and 46 in ops:
            changed = bytearray(code)
            changed[ops.index(46) * 16 + 4] ^= 0x10
            branch_mutation = (reference, bytes(changed), ops)
        if reconvergence_mutation is None and 54 in ops:
            changed = bytearray(code)
            changed[ops.index(54) * 16 + 4] ^= 0x10
            reconvergence_mutation = (reference, bytes(changed), ops)
        if sync_barrier_mutation is None and 55 in ops:
            changed = bytearray(code)
            changed[ops.index(55) * 16 + 2] ^= 1
            sync_barrier_mutation = (reference, bytes(changed), ops)
        if carry_mutation is None and 14 in ops:
            changed = bytearray(code)
            changed[ops.index(14) * 16 + 10] ^= 2
            carry_mutation = (reference, bytes(changed), ops)
    if mismatches:
        raise RuntimeError("\n".join(mismatches))
    missing = set(range(60)) - coverage
    if missing:
        raise RuntimeError(f"opcode coverage missing: {sorted(missing)}")
    if mutation is None:
        raise RuntimeError("no operand-bearing instruction for negative control")
    try:
        compare(args.nvdisasm, *mutation)
    except RuntimeError:
        pass
    else:
        raise RuntimeError("negative control BROKEN: mutated register encoding passed")
    if uniform_dst_mutation is None:
        raise RuntimeError("uniform-source nonidentity GPR-map regression missing")
    try:
        compare(args.nvdisasm, *uniform_dst_mutation)
    except RuntimeError:
        pass
    else:
        raise RuntimeError("negative control BROKEN: old uniform-destination bug passed")
    if mmio_mutation is None:
        raise RuntimeError("MMIO contract regression missing")
    try:
        compare(args.nvdisasm, *mmio_mutation)
    except RuntimeError:
        pass
    else:
        raise RuntimeError("negative control BROKEN: EF load satisfied MMIO contract")
    if subtraction_mutation is None:
        raise RuntimeError("subtraction modifier regression missing")
    try:
        compare(args.nvdisasm, *subtraction_mutation)
    except RuntimeError:
        pass
    else:
        raise RuntimeError("negative control BROKEN: FADD satisfied the FSUB contract")
    if branch_mutation is None:
        raise RuntimeError("Ampere branch displacement regression missing")
    try:
        compare(args.nvdisasm, *branch_mutation)
    except RuntimeError:
        pass
    else:
        raise RuntimeError("negative control BROKEN: wrong branch destination accepted")
    if reconvergence_mutation is None:
        raise RuntimeError("Ampere reconvergence join regression missing")
    try:
        compare(args.nvdisasm, *reconvergence_mutation)
    except RuntimeError:
        pass
    else:
        raise RuntimeError("negative control BROKEN: wrong reconvergence join accepted")
    if sync_barrier_mutation is None:
        raise RuntimeError("Ampere reconvergence barrier regression missing")
    try:
        compare(args.nvdisasm, *sync_barrier_mutation)
    except RuntimeError:
        pass
    else:
        raise RuntimeError("negative control BROKEN: wrong reconvergence barrier accepted")
    if carry_mutation is None:
        raise RuntimeError("Ampere discarded-carry regression missing")
    try:
        compare(args.nvdisasm, *carry_mutation)
    except RuntimeError:
        pass
    else:
        raise RuntimeError("negative control BROKEN: unexpected carry output accepted")
    refused = list(args.corpus.glob("*.refused"))
    refused_markers = {imm for file in refused for op, imm in struct.iter_unpack("=II", file.read_bytes())
                       if (op, imm) in {(16, 0x51303832), (37, 0x46453433)}}
    if refused_markers != {0x51303832, 0x46453433}:
        raise RuntimeError("native FP8 QMMA and E4M3 conversion refusal coverage missing")
    print(f"SM86 offline oracle: {len(pairs)} kernels, every opcode/operand matched; mutated operand refused;")
    print(f"native FP8 refusals: {len(refused)} kernels. Device qualification: NOT_RUN")
    print("MMIO policy, subtraction sign, branch target, reconvergence join/barrier controls refused")
    print("IADD3's discarded PT outputs retained semantically; unexpected carry output refused")


if __name__ == "__main__":
    main()
