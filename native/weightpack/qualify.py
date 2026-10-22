#!/usr/bin/env python3
"""Host qualification/orchestration. Product transformation stays in weightpack.om."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import select
import struct
import subprocess
import stat
import uuid
import test_weightpack as oracle

COMPILER_SHA = '191b776ef98821597c3e30655da6d680a9b66edbc9ceaca3c39c61ea6df098e9'
SELECTIONS = [(0, 0, 'gate_proj'), (0, 0, 'up_proj'), (0, 0, 'down_proj'),
              (20, 128, 'up_proj'), (39, 255, 'down_proj')]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def compile_source(compiler, source, target):
    p = subprocess.run([str(compiler)], input=source, capture_output=True, timeout=30)
    assert p.returncode == 0 and not p.stderr, (p.returncode, p.stderr)
    assert p.stdout[:4] == b'\x7fELF'
    target.write_bytes(p.stdout)
    target.chmod(0o700)
    return target


def run(binary, data):
    p = subprocess.run([str(binary)], input=data, capture_output=True, timeout=30)
    assert p.returncode == 0 and not p.stderr, (p.returncode, p.stderr)
    return p.stdout


def io_controls(binary):
    data = oracle.raw_frame(3, 9, 0, bytes(27))
    expected = oracle.reference_encode(data)
    rows = []
    for name, fd in [('closed-read', 0), ('closed-write', 1)]:
        p = subprocess.run([str(binary)], input=data, capture_output=True,
                           preexec_fn=lambda: os.close(fd), timeout=10)
        rows.append({'case': name, 'rc': p.returncode, 'pass': p.returncode == 74 and not p.stdout and not p.stderr})
    with open('/dev/full', 'wb') as full:
        p = subprocess.run([str(binary)], input=data, stdout=full, stderr=subprocess.PIPE, timeout=10)
        rows.append({'case': 'full-output-device', 'rc': p.returncode, 'pass': p.returncode == 74 and not p.stderr})
    p = subprocess.Popen([str(binary)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p.stdin.write(data[:17])
    p.stdin.flush()
    early = bool(select.select([p.stdout], [], [], 0.03)[0])
    out, err = p.communicate(data[17:], timeout=10)
    rows.append({'case': 'fragmented-input-no-early-output', 'rc': p.returncode,
                 'pass': not early and p.returncode == 0 and out == expected and not err})
    assert all(r['pass'] for r in rows), rows
    return rows


def mutants(compiler, source, build, evidence):
    changes = [('lose-zero-sign', b'if negative\nret 1\n', b'if negative\nret 0\n'),
               ('wrong-orientation', b'if eq get shape 2 0\n', b'if eq get shape 2 1\n'),
               ('accept-trailing-sparse', b'if ne p n\nret 65\nend\n', b'')]
    summary = []
    for name, before, after in changes:
        assert source.count(before) == 1
        changed = source.replace(before, after)
        binary = compile_source(compiler, changed, build / ('mutant-' + name))
        result = oracle.run_suite(binary)
        write_json(evidence / ('mutant-' + name + '.json'), result)
        assert result['failed'] > 0, name
        summary.append({'name': name, 'source_sha256': sha(changed),
                        'binary_sha256': result['binary_sha256'], 'caught': result['failed'],
                        'total': result['total'], 'change_before': before.decode(), 'change_after': after.decode()})
    return summary


def read_header(path):
    with path.open('rb') as stream:
        head = stream.read(8)
        assert len(head) == 8
        size = struct.unpack('<Q', head)[0]
        assert 0 < size <= 32 * 1024 * 1024
        raw = stream.read(size)
        assert len(raw) == size
    # JSON duplicate keys are refused, not silently overwritten.
    def pairs(items):
        result = {}
        for key, value in items:
            assert key not in result, key
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs), 8 + size, sha(head + raw)


def metadata_inventory(model):
    config = json.loads((model / 'config.json').read_text())
    text = config.get('text_config', config)
    layers, experts = text['num_hidden_layers'], text['num_experts']
    records = []
    seen = set()
    for layer in range(layers):
        path = model / f'layers-{layer}.safetensors'
        header, base, header_sha = read_header(path)
        expected = set()
        for expert in range(experts):
            for proj in ('gate_proj', 'up_proj', 'down_proj'):
                key = f'model.language_model.layers.{layer}.mlp.experts.{expert}.{proj}.weight'
                item = header[key]
                assert item['dtype'] == 'F8_E4M3'
                rows, cols = item['shape']
                lo, hi = item['data_offsets']
                assert 0 <= lo < hi <= path.stat().st_size - base
                assert hi - lo == rows * cols and rows * cols <= oracle.LIMIT
                assert 1 <= rows <= 4096 and 1 <= cols <= 4096
                assert key not in seen
                seen.add(key)
                expected.add(key)
        actual = {k for k in header if '.mlp.experts.' in k and k.endswith('.weight')}
        assert actual == expected, (layer, actual ^ expected)
        records.append({'layer': layer, 'path': path.name, 'header_sha256': header_sha,
                        'routed_weight_tensors': len(expected), 'payload_scanned': False})
    assert len(seen) == layers * experts * 3
    return {'config_sha256': sha((model / 'config.json').read_bytes()),
            'layers': layers, 'experts_per_layer': experts, 'expected_routed_weight_tensors': layers * experts * 3,
            'enumerated_routed_weight_tensors': len(seen), 'metadata_only': True, 'files': records}


def checkpoint_replay(model, binary):
    rows = []
    for layer, expert, projection in SELECTIONS:
        path = model / f'layers-{layer}.safetensors'
        header, base, header_sha = read_header(path)
        key = f'model.language_model.layers.{layer}.mlp.experts.{expert}.{projection}.weight'
        item = header[key]
        assert item['dtype'] == 'F8_E4M3'
        r, c = item['shape']
        lo, hi = item['data_offsets']
        assert hi - lo == r * c
        scale = header[key + '_scale_inv']
        slo, shi = scale['data_offsets']
        assert 0 <= slo <= shi <= path.stat().st_size - base
        with path.open('rb') as stream:
            stream.seek(base + lo)
            data = stream.read(hi - lo)
            stream.seek(base + slo)
            scales = stream.read(shi - slo)
        assert len(data) == r * c and len(scales) == shi - slo
        axis = int(projection == 'down_proj')
        raw = oracle.raw_frame(r, c, axis, data)
        packed = run(binary, raw)
        reference = oracle.reference_encode(raw)
        assert packed == reference
        restored = run(binary, packed)
        assert restored == raw == oracle.reference_decode(packed)
        vectors = oracle.channels(data, r, c, axis)
        zero_indices = [i for i, v in enumerate(vectors) if not any(x & 127 for x in v)]
        rows.append({'layer': layer, 'expert': expert, 'projection': projection, 'tensor': key,
                     'dtype': item['dtype'], 'shape': item['shape'], 'axis': axis, 'offsets': [lo, hi],
                     'header_sha256': header_sha, 'raw_sha256': sha(data), 'framed_raw_sha256': sha(raw),
                     'packed_sha256': sha(packed), 'reconstructed_sha256': sha(restored[16:]),
                     'raw_bytes': len(data), 'packed_bytes_including_header': len(packed),
                     'saved_bytes_including_header': len(data) - len(packed),
                     'scale_bytes_unchanged': len(scales), 'scale_sha256': sha(scales),
                     'original_with_scales': len(data) + len(scales),
                     'packed_with_scales': len(packed) + len(scales),
                     'mode': struct.unpack_from('<I', packed, 16)[0],
                     'zero_channels': len(zero_indices), 'zero_indices': zero_indices,
                     'all_byte_and_oracle_checks_passed': True})
    assert len(rows) == len(SELECTIONS) == 5
    aligned = [r['zero_indices'] for r in rows[:3]]
    assert aligned[0] == aligned[1] == aligned[2] and len(aligned[0]) == 280
    return {'selection_policy': 'Five fixed matrices inherited from the research; not representative prevalence.',
            'model_directory': model.name, 'config_sha256': sha((model / 'config.json').read_bytes()),
            'matrices': rows, 'matched_first_expert_zero_channels': len(aligned[0]),
            'weights_retained_in_evidence': False, 'inference_or_gpu_tested': False}


def safe_directory(path):
    # Inspect named parents before resolve(), which would hide directory symlinks.
    path = Path(os.path.abspath(path))
    for component in [*reversed(path.parents), path]:
        if component.exists() or component.is_symlink():
            mode = component.lstat().st_mode
            if not stat.S_ISDIR(mode):
                raise ValueError('unsafe output directory: ' + str(component))
    if path.exists():
        for item in path.rglob('*'):
            info = item.lstat()
            if stat.S_ISDIR(info.st_mode):
                continue
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise ValueError('unsafe output entry: ' + str(item))
    return path


def retire_current_receipt(evidence):
    evidence = safe_directory(evidence)
    evidence.mkdir(parents=True, exist_ok=True)
    receipt = evidence / 'qualification.json'
    if receipt.exists():
        receipt.rename(evidence / ('qualification.previous-' + uuid.uuid4().hex + '.json'))
    return evidence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--compiler', type=Path)
    parser.add_argument('--invalidate-only', action='store_true')
    parser.add_argument('--build', required=True, type=Path)
    parser.add_argument('--evidence', required=True, type=Path)
    parser.add_argument('--model', type=Path)
    args = parser.parse_args()
    args.evidence = retire_current_receipt(args.evidence)
    if args.invalidate_only:
        return
    args.build = safe_directory(args.build)
    if args.compiler is None:
        raise ValueError("--compiler is required for qualification")
    compiler = args.compiler.resolve()
    assert sha(compiler.read_bytes()) == COMPILER_SHA
    source = Path(__file__).with_name('weightpack.om').read_bytes()
    args.build.mkdir(parents=True, exist_ok=True)
    args.evidence.mkdir(parents=True, exist_ok=True)
    binary = compile_source(compiler, source, args.build.resolve() / 'weightpack')
    result = oracle.run_suite(binary)
    write_json(args.evidence / 'qualified-suite.json', result)
    assert result['failed'] == 0
    summary = {'source_sha256': sha(source), 'compiler_sha256': COMPILER_SHA,
               'binary_sha256': sha(binary.read_bytes()), 'suite_total': result['total'],
               'suite_passed': result['passed'], 'io_controls': io_controls(binary),
               'mutants': mutants(compiler, source, args.build.resolve(), args.evidence)}
    if args.model:
        inventory = metadata_inventory(args.model)
        write_json(args.evidence / 'metadata-inventory.json', inventory)
        replay = checkpoint_replay(args.model, binary)
        write_json(args.evidence / 'checkpoint-replay.json', replay)
        summary['metadata_tensor_count'] = inventory['enumerated_routed_weight_tensors']
        summary['real_matrix_replay_count'] = len(replay['matrices'])
    write_json(args.evidence / 'qualification.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
