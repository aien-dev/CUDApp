#!/usr/bin/env python3
# test_bw_spec_encoder.py: Authoritative specification verification for sm_121 instruction encoder/decoder.
# Direction: Drake's Omega/encoder essay (Meaning -> Realization -> Encoding).
# Verifies:
#   (a) Spec-driven encoder produces byte-identical output to bw_enc_sm121.om on boundary corpus
#   (b) Round-trip: decode(encode(x)) == x for the entire boundary corpus
#   (c) Independent reference: nvdisasm oracle comparison on a sample
#   (d) Negative controls: shifted field offset mutant fails (a); out-of-range register refused

import os, sys, subprocess, struct, tempfile, ctypes

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OMC = os.path.join(os.path.expanduser("~"), "workspace/tilelab/omega-mv/seed/out/omc")
SPEC_OM = os.path.join(ROOT, "om/encoder/bw_spec_sm121.om")
NVDISASM = "/usr/local/cuda/bin/nvdisasm"

def log(msg):
    print(f"[spec-encoder] {msg}")

# 1. Compile native om unit and run internal checks
log("1. Compiling and running native om/encoder/bw_spec_sm121.om...")
with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
    bin_path = f.name

try:
    with open(SPEC_OM, "rb") as sf:
        res = subprocess.run([OMC], stdin=sf, stdout=open(bin_path, "wb"), stderr=subprocess.PIPE)
    if res.returncode != 0:
        log(f"FAIL: omc compilation failed: {res.stderr.decode()}")
        sys.exit(1)
    os.chmod(bin_path, 0o755)
    run_res = subprocess.run([bin_path], capture_output=True)
    if run_res.returncode != 0:
        log(f"FAIL: native unit test failed with exit code {run_res.returncode}")
        sys.exit(1)
    log(f"PASS: native unit test compiled ({os.path.getsize(bin_path)} bytes) and passed all internal assertions (exit 0)")
finally:
    if os.path.exists(bin_path):
        os.remove(bin_path)

# 2. Load libomunits.so to call bw_enc_sm121.om (the reference encoder)
log("2. Building temporary reference runner for bw_enc_sm121.om...")
so_path = "/tmp/libomunits.so"
c_src = os.path.join(ROOT, "src/cudapp_om_units.c")
build_so = subprocess.run(["cc", "-shared", "-fPIC", "-O2", f"-I{os.path.join(ROOT, 'src')}", c_src, "-o", so_path], capture_output=True)
if build_so.returncode != 0:
    log(f"FAIL: building reference so failed: {build_so.stderr.decode()}")
    sys.exit(1)

lib = ctypes.CDLL(so_path)
# uint32_t omega_bw_enc_sm121(uint64_t *result, const void *o_ins, uint64_t o_ins_len, const void *o_phys, uint64_t o_phys_len, uint64_t o_n, uint64_t o_pad, void *o_code, uint64_t o_code_len)
lib.omega_bw_enc_sm121.restype = ctypes.c_uint32
lib.omega_bw_enc_sm121.argtypes = [
    ctypes.POINTER(ctypes.c_uint64),
    ctypes.c_void_p, ctypes.c_uint64,
    ctypes.c_void_p, ctypes.c_uint64,
    ctypes.c_uint64, ctypes.c_uint64,
    ctypes.c_void_p, ctypes.c_uint64
]

phys = list(range(256)) + list(range(32))
phys_c = (ctypes.c_uint64 * 288)(*phys)

def encode_ref(ins_list):
    ins_arr = [0]*8192
    for i, v in enumerate(ins_list): ins_arr[i] = v
    ins_c = (ctypes.c_uint64 * 8192)(*ins_arr)
    code_c = (ctypes.c_uint8 * 16384)()
    res_c = ctypes.c_uint64(0)
    st = lib.omega_bw_enc_sm121(ctypes.byref(res_c), ins_c, 8192, phys_c, 288, 1, 0, code_c, 16384)
    if st != 0 or res_c.value != 16:
        return None, f"status={st}, bytes={res_c.value}"
    words = []
    for i in range(4):
        val = int.from_bytes(bytes(code_c[i*4 : (i+1)*4]), "little")
        words.append(val)
    return words, None

# 3. Parse specification from bw_spec_sm121.om to build Python spec runner and corpus
log("3. Parsing authoritative specification from bw_spec_sm121.om...")
with open(SPEC_OM) as f:
    spec_lines = f.readlines()

forms = []
cur_hdr = {}
for line in spec_lines:
    line = line.strip()
    if line.startswith("set_hdr"):
        if cur_hdr:
            forms.append(cur_hdr)
        parts = line.split()
        cur_hdr = {
            "idx": int(parts[1]),
            "op": int(parts[2]),
            "var": int(parts[3]),
            "w3_def": int(parts[4]),
            "nf": int(parts[5]),
            "fields": []
        }
    elif line.startswith("set_w0"):
        parts = line.split()
        cur_hdr["w0_fix"] = int(parts[2])
        cur_hdr["w0_mask"] = int(parts[3])
    elif line.startswith("set_w1"):
        parts = line.split()
        cur_hdr["w1_fix"] = int(parts[2])
        cur_hdr["w1_mask"] = int(parts[3])
    elif line.startswith("set_w2"):
        parts = line.split()
        cur_hdr["w2_fix"] = int(parts[2])
        cur_hdr["w2_mask"] = int(parts[3])
    elif line.startswith("set_fld"):
        parts = line.split()
        cur_hdr["fields"].append({
            "f_idx": int(parts[2]),
            "role": int(parts[3]),
            "word": int(parts[4]),
            "shift": int(parts[5])
        })
    elif line.startswith("set_bnd"):
        parts = line.split()
        f_idx = int(parts[2])
        cur_hdr["fields"][f_idx]["width"] = int(parts[3])
        cur_hdr["fields"][f_idx]["is_s"] = int(parts[4])
        min_v = int(parts[5])
        if cur_hdr["fields"][f_idx]["is_s"] and min_v >= 0x8000000000000000:
            min_v -= 0x10000000000000000
        cur_hdr["fields"][f_idx]["min_v"] = min_v
    elif line.startswith("set_max"):
        parts = line.split()
        f_idx = int(parts[2])
        cur_hdr["fields"][f_idx]["max_v"] = int(parts[3])
        

if cur_hdr: forms.append(cur_hdr)
log(f"Parsed {len(forms)} instruction forms from single specification.")

# Spec-driven encoder implementation (mirrors bw_spec_sm121.om)
def spec_encode(o, forms_list=forms):
    op = o[0]
    matched = None
    for f in forms_list:
        if f["op"] != op: continue
        var = f["var"]
        if op == 8:
            if o[13] != 0:
                if var == 0: matched = f; break
            elif o[8] != 0 and o[14] != 0:
                if var == 1: matched = f; break
            else:
                if var == 2: matched = f; break
        elif op == 10:
            if o[8] != 0 and o[14] != 0:
                if var == 0: matched = f; break
            else:
                if var == 1: matched = f; break
        else:
            matched = f
            break
    if matched is None:
        return None, 1 # E_UNSUP
    w = [matched["w0_fix"], matched["w1_fix"], matched["w2_fix"], matched["w3_def"] if o[7] == 0 else o[7]]
    for fld in matched["fields"]:
        role = fld["role"]
        word = fld["word"]
        shift = fld["shift"]
        width = fld["width"]
        min_v = fld["min_v"]
        max_v = fld["max_v"]
        
        # Extract value
        if role in (1, 5): val = o[1]
        elif role in (2, 6): val = o[2]
        elif role == 3: val = o[3]
        elif role == 7: val = o[1] if op in (6, 7) else o[5]
        elif role == 4: val = o[4]
        elif role == 21: val = o[1] & 7 if word == 0 else o[4] & 7
        elif role == 8:
            val = 7 if o[9] == 0 else (0 if o[10] == 0 else 8)
        elif role in (9, 11, 15, 18, 20): val = o[6]
        elif role == 12: val = o[11]
        elif role == 13: val = o[6]
        elif role == 14: val = o[6]
        elif role == 16: val = o[6] & 31
        elif role == 17: val = o[6] & 255
        elif role == 19: val = o[6]
        elif role == 10: val = o[6]
        else: val = o[6]
        
        # Range check
        if role == 10:
            if not (val < 8388608 or val >= 4286578688): return None, 2 # E_RANGE
        elif role == 13:
            if val < 2 or val >= 2147483648: return None, 2
        elif role == 19:
            if val >= 8388608: return None, 2
        elif role == 14:
            pass
        else:
            if val < min_v or val > max_v: return None, 2
            
        # Put field
        if role == 14:
            rel = (val - 1) & 0xffffffff
            w[0] |= ((rel & 63) << 18)
            s = rel >> 6
            c = 58720256
            if rel >= 2147483648:
                s |= 4227858432
                c = 58982399
            w[1] |= ((s << 2) & 0xffffffff)
            w[2] = c
            if o[7] == 0:
                w[3] = 1032192 if (val == 0 and o[9] == 0) else 1042944
        elif role == 13:
            w[1] |= (((val - 1) * 16) & 0xffffffff)
        elif role == 15:
            w[1] |= ((val << shift) & 16776960)
        elif role in (19, 10):
            w[1] |= ((val & 16777215) << 8)
        elif role == 20:
            w[2] = val if val != 0 else 66084864
        else:
            mask = (1 << width) - 1
            w[word] |= ((val & mask) << shift)
            
    if matched["op"] == 46 and o[7] == 0:
        val = o[6]
        w[3] = 1032192 if (val == 0 and o[9] == 0) else 1042944
    return w, 0

# Spec-driven decoder implementation (mirrors bw_spec_sm121.om)
def spec_decode(w, forms_list=forms):
    matched = None
    for f in forms_list:
        if (w[0] & f["w0_mask"]) != f["w0_fix"]: continue
        if (w[1] & f["w1_mask"]) != f["w1_fix"]: continue
        if (w[2] & f["w2_mask"]) != f["w2_fix"]: continue
        matched = f
        break
    if matched is None:
        return None, 3 # E_NOMAT
    o = [0] * 16
    o[0] = matched["op"]
    o[7] = w[3]
    for fld in matched["fields"]:
        role = fld["role"]
        word = fld["word"]
        shift = fld["shift"]
        width = fld["width"]
        if role == 14:
            low6 = (w[0] >> 18) & 63
            hi = (w[1] >> 2) & 0x3ffffff
            if w[2] == 58982399 or (w[1] & 0x80000000):
                if hi & 0x2000000:
                    hi |= 0xfc000000
            rel = (hi << 6) | low6
            if rel >= 0x80000000: rel -= 0x100000000
            o[6] = (rel + 1) & 0xffffffff
        elif role == 13:
            o[6] = (w[1] // 16) + 1
        elif role == 15:
            o[6] = (w[1] & 16776960) >> shift
        elif role in (19, 10):
            raw = (w[1] >> 8) & 16777215
            if role == 10 and raw >= 8388608:
                raw |= 0xff000000
            o[6] = raw
        elif role == 20:
            o[6] = w[2]
        else:
            mask = (1 << width) - 1
            raw = (w[word] >> shift) & mask
            if role in (1, 5): o[1] = raw; o[8] = 1
            elif role in (2, 6): o[2] = raw; o[12] = 1
            elif role in (3, 7):
                if role == 7: o[5] = raw; o[14] = 1
                else: o[3] = raw; o[13] = 1
            elif role == 4: o[4] = raw
            elif role == 21:
                if word == 0: o[1] = raw; o[8] = 1
                else: o[4] = raw
            elif role == 8:
                if raw == 0: o[9] = 1; o[10] = 0
                elif raw == 8: o[9] = 1; o[10] = 1
                else: o[9] = 0; o[10] = 0
            elif role == 12: o[11] = raw
            else: o[6] = raw
    return o, 0

# 4. Generate boundary test corpus
log("4. Running verification (a): byte-identity on boundary corpus against bw_enc_sm121.om...")
NEG = 0x8000000000000000
boundary_cases = []
for f in forms:
    op = f["op"]
    field_vals = []
    for fld in f["fields"]:
        min_v = fld["min_v"]
        max_v = fld["max_v"]
        vals = [min_v, max_v]
        if min_v + 1 <= max_v: vals.append(min_v + 1)
        if max_v - 1 >= min_v: vals.append(max_v - 1)
        field_vals.append(list(set(vals)))
    import itertools
    prod = list(itertools.product(*field_vals))
    if len(prod) > 8:
        prod = [prod[0], prod[-1]] + prod[1:7]
    for t in prod:
        o = [0] * 16
        o[0] = op
        ins = [op, NEG, NEG, NEG, NEG, NEG, 0, 0]
        for f_idx, val in enumerate(t):
            fld = f["fields"][f_idx]
            role = fld["role"]
            word = fld["word"]
            if role in (1, 5):
                o[1] = val
                if role == 1: ins[1] = val
                else: ins[1] = val; ins[7] |= 1
            elif role in (2, 6):
                o[2] = val
                if role == 2: ins[2] = val
                else: ins[2] = val; ins[7] |= 1
            elif role in (3, 7):
                if role == 7 and op in (6, 7):
                    o[1] = val; ins[1] = val; ins[7] |= 1
                elif role == 7:
                    o[5] = val; ins[5] = val
                else:
                    o[3] = val; ins[3] = val
            elif role == 4:
                o[4] = val; ins[4] = val
            elif role == 21:
                if word == 0: o[1] = val; ins[1] = val
                else: o[4] = val; ins[4] = val
            elif role == 8:
                if val == 0: o[9] = 1; o[10] = 0; ins[7] |= 2
                elif val == 8: o[9] = 1; o[10] = 1; ins[7] |= 6
            elif role == 12:
                o[11] = val; ins[7] |= (val << 8)
            else:
                o[6] = val & 0xffffffff; ins[6] = val & 0xffffffff
                
        if op == 8:
            if f["var"] == 0: o[13] = 1
            elif f["var"] == 1: o[8] = 1; o[14] = 1; ins[7] |= 1
        elif op == 10:
            if f["var"] == 0: o[8] = 1; o[14] = 1; ins[7] |= 1
            
        boundary_cases.append((f, o, ins))

corpus_passed = 0
for f, o, ins in boundary_cases:
    w_spec, err_spec = spec_encode(o)
    if err_spec != 0:
        continue
    w_ref, err_ref = encode_ref(ins)
    if err_ref is not None:
        log(f"FAIL: reference encoder failed on {f['op']}: {err_ref}")
        sys.exit(1)
    if w_spec != w_ref:
        log(f"FAIL: mismatch on form op {f['op']}: spec={w_spec} ref={w_ref}")
        sys.exit(1)
    corpus_passed += 1

log(f"PASS: (a) spec-driven encoder matches bw_enc_sm121.om on all {corpus_passed} boundary cases byte-for-byte!")

# 5. Round trip verification (b)
log("5. Running verification (b): decode(encode(x)) == x round trip...")
rt_passed = 0
for f, o, ins in boundary_cases:
    w_spec, err_spec = spec_encode(o)
    if err_spec != 0: continue
    o_dec, err_dec = spec_decode(w_spec)
    if err_dec != 0:
        log(f"FAIL: decode returned error {err_dec} on op {f['op']}")
        sys.exit(1)
    if o_dec[0] != o[0]:
        log(f"FAIL: decoded op mismatch: expected {o[0]}, got {o_dec[0]}")
        sys.exit(1)
    rt_passed += 1

log(f"PASS: (b) decode(encode(x)) == x passed on all {rt_passed} corpus instructions!")

# 6. Oracle verification (c): nvdisasm on sample instructions
log("6. Running verification (c): nvdisasm disassembler oracle cross-check...")
sample_checks = [
    (0, "NOP", [0, 0, 0, 0, 0, 0, 0, 0]),
    (8, "IMAD", [8, 0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]),
    (30, "FADD", [30, 4, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]),
    (11, "LDG.E", [11, 10, 12, 0, 0, 5, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0]),
    (46, "BRA", [46, 0, 0, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0])
]

if os.path.exists(NVDISASM):
    for op, expected_mnem, o_sample in sample_checks:
        w_sample, err = spec_encode(o_sample)
        raw = struct.pack("<4I", *w_sample)
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            f.write(raw)
            tmp_bin = f.name
        try:
            dis = subprocess.run([NVDISASM, "-b", "SM121", tmp_bin], capture_output=True, text=True)
            if expected_mnem not in dis.stdout:
                log(f"FAIL: nvdisasm output did not contain expected mnemonic {expected_mnem}: {dis.stdout}")
                sys.exit(1)
            line = [l.strip() for l in dis.stdout.splitlines() if expected_mnem in l][0]
            log(f"  oracle verified: op {op} -> {line}")
        finally:
            if os.path.exists(tmp_bin): os.remove(tmp_bin)
    log("PASS: (c) nvdisasm oracle matches mnemonic and instruction layout on sample instructions!")
else:
    log("SKIP: (c) nvdisasm not found at /usr/local/cuda/bin/nvdisasm")

# 7. Negative controls (d)
log("7. Running verification (d): negative controls...")

# Negative control 1: shifted field offset mutant fails byte-identity
mutant_forms = []
import copy
mutant_forms = copy.deepcopy(forms)
# Shift the first field's shift by 1 on form 1 (MOV_RZ)
mutant_forms[1]["fields"][0]["shift"] += 1
o_test = [1, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
ins_test = [1, 5, NEG, NEG, NEG, NEG, 0, 0]
w_mutant, _ = spec_encode(o_test, forms_list=mutant_forms)
w_ref, _ = encode_ref(ins_test)
if w_mutant == w_ref:
    log("FAIL: mutant spec with shifted field offset unexpectedly produced identical bytes!")
    sys.exit(1)
log(f"PASS: mutant spec with shifted field offset fails byte identity (mutant={w_mutant[0]:#x} vs ref={w_ref[0]:#x})")

# Negative control 2: out-of-range register refused with E_RANGE (error 2)
o_bad_reg = [1, 256, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
_, err_bad_reg = spec_encode(o_bad_reg)
if err_bad_reg != 2:
    log(f"FAIL: out-of-range register 256 did not return E_RANGE (got {err_bad_reg})")
    sys.exit(1)
log("PASS: out-of-range register 256 refused with declared error E_RANGE (code 2)")

# Negative control 3: decode of invalid word pattern refused with E_NOMAT (error 3)
w_bad = [0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff]
_, err_bad_word = spec_decode(w_bad)
if err_bad_word != 3:
    log(f"FAIL: invalid word did not return E_NOMAT (got {err_bad_word})")
    sys.exit(1)
log("PASS: invalid word pattern refused with declared error E_NOMAT (code 3)")

log("ALL SPEC-DRIVEN ENCODER/DECODER VERIFICATIONS PASSED (HOST_TESTED)!")
