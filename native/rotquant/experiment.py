#!/usr/bin/env python3
"""Offline independent NumPy/ml-dtypes experiment; not an inference runtime."""
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import numpy as np
import ml_dtypes

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'weightpack'))
import qualify as common

H = np.array([[-1 if (i & j).bit_count() % 2 else 1 for j in range(128)] for i in range(128)], dtype=np.int64)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def invoke(binary, data):
    p = subprocess.run([str(binary)], input=data, capture_output=True, timeout=30)
    assert p.returncode == 0 and not p.stderr, (p.returncode, p.stderr)
    return p.stdout


def native_matrix(binary, raw, shape, mode):
    decoded_fp8 = np.frombuffer(raw, dtype=np.uint8).view(ml_dtypes.float8_e4m3fn).astype(np.float64)
    assert np.isfinite(decoded_fp8).all()
    source_units = (decoded_fp8*512).astype(np.int64).reshape(-1, 128)
    assert np.array_equal(source_units.reshape(-1)/512, decoded_fp8)
    # Independent dense sign-matrix multiplication, not the product butterfly.
    transformed = source_units @ H if mode else source_units
    maxima = np.abs(transformed).max(axis=1)
    safe = np.maximum(maxima, 1)[:, None]
    magnitude = np.abs(transformed)*7
    quotient, remainder = np.divmod(magnitude, safe)
    rounded = quotient + ((2*remainder > safe) | ((2*remainder == safe) & (quotient % 2 != 0)))
    quantized = rounded * np.sign(transformed)
    assert np.abs(quantized).max() <= 7
    numerators = (quantized @ H if mode else quantized) * maxima[:, None]
    denominator = 458752 if mode else 3584
    restored = []
    packed_parts = []
    for start in range(0, len(source_units), 256):
        count = min(256, len(source_units)-start)
        block_raw = raw[start*128:(start+count)*128]
        request = struct.pack('<4sII', b'RQU1', count, mode) + block_raw
        packed = invoke(binary, request)
        expected = bytearray(struct.pack('<4sII', b'RQP1', count, mode))
        for b in range(start, start+count):
            q = quantized[b]+7
            expected += struct.pack('<I', int(maxima[b]))
            expected += (q[::2] | (q[1::2] << 4)).astype(np.uint8).tobytes()
        assert packed == bytes(expected)
        reconstructed = invoke(binary, packed)
        assert reconstructed[:16] == struct.pack('<4sIII', b'RQR1', count, mode, denominator)
        actual = np.frombuffer(reconstructed, dtype='<i8', offset=16).reshape(count, 128)
        assert np.array_equal(actual, numerators[start:start+count])
        restored.append(actual.copy())
        packed_parts.append(packed)
    numerator = np.concatenate(restored).reshape(shape)
    return numerator/denominator, {'frame_count': len(packed_parts), 'packed_bytes_including_frames': sum(map(len, packed_parts)),
                                  'packed_sha256': sha(b''.join(packed_parts)), 'zero_quantizer_blocks': int((maxima == 0).sum()),
                                  'all_native_codes_and_numerators_match_oracle': True}


def diagnostic_vectors(width):
    rng = np.random.default_rng(49001)
    rademacher = rng.choice([-1.0, 1.0], size=(width, 16))
    gaussian = rng.standard_normal((width, 16))
    for values in (rademacher, gaussian):
        values /= np.linalg.norm(values, axis=0)
    impulses = np.zeros((width, 8))
    indices = np.linspace(0, width-1, 8, dtype=int)
    impulses[indices, np.arange(8)] = 1
    return np.concatenate([rademacher, gaussian, impulses], axis=1)


def error_metrics(reference, candidate, vectors):
    original = reference @ vectors
    predicted = candidate @ vectors
    delta = predicted-original
    norms = np.linalg.norm(original, axis=0)
    error = np.linalg.norm(delta, axis=0)
    nonzero = norms != 0
    profiles = {}
    for name, a, b in [('rademacher', 0, 16), ('gaussian', 16, 32), ('impulses', 32, 40)]:
        denominator = np.linalg.norm(original[:, a:b])
        profiles[name] = float(np.linalg.norm(delta[:, a:b])/denominator) if denominator else None
    base = np.linalg.norm(reference)
    return {'relative_weight_frobenius_error': float(np.linalg.norm(candidate-reference)/base) if base else None,
            'relative_synthetic_output_error': float(np.linalg.norm(delta)/np.linalg.norm(original)) if np.linalg.norm(original) else None,
            'profile_output_errors': profiles, 'max_relative_vector_error': float(np.max(error[nonzero]/norms[nonzero])) if nonzero.any() else None,
            'zero_reference_vectors': int((~nonzero).sum()),
            'nonzero_outputs_for_zero_reference_vectors': int(np.count_nonzero(np.linalg.norm(predicted[:, ~nonzero], axis=0))),
            'max_absolute_error_on_zero_reference_vectors': float(error[~nonzero].max()) if (~nonzero).any() else 0.0,
            'reconstructed_zero_rows': int(np.count_nonzero(np.all(candidate == 0, axis=1))),
            'reconstructed_zero_columns': int(np.count_nonzero(np.all(candidate == 0, axis=0)))}


def experiment(model, binary, lossless_binary):
    records = []
    # Verify all 254 finite byte codes against an independently implemented dtype.
    from test_rotation import units
    finite = np.array([b for b in range(256) if b & 127 != 127], dtype=np.uint8)
    assert np.array_equal(finite.view(ml_dtypes.float8_e4m3fn).astype(np.float64)*512, [units(int(b)) for b in finite])
    for layer, expert, projection in common.SELECTIONS:
        path = model/f'layers-{layer}.safetensors'
        before = path.stat()
        header, base, header_sha = common.read_header(path)
        key = f'model.language_model.layers.{layer}.mlp.experts.{expert}.{projection}.weight'
        item, scale = header[key], header[key+'_scale_inv']
        rows, cols = item['shape']
        assert item['dtype'] == 'F8_E4M3' and scale['dtype'] == 'BF16'
        assert rows % 128 == cols % 128 == 0
        assert scale['shape'] == [rows//128, cols//128]
        lo, hi = item['data_offsets']
        slo, shi = scale['data_offsets']
        assert hi-lo == rows*cols and shi-slo == 2*(rows//128)*(cols//128)
        assert 0 <= lo < hi <= before.st_size-base and 0 <= slo < shi <= before.st_size-base
        with path.open('rb') as stream:
            stream.seek(base+lo)
            raw = stream.read(hi-lo)
            stream.seek(base+slo)
            scales = stream.read(shi-slo)
        after = path.stat()
        assert (before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_ino, after.st_size, after.st_mtime_ns)
        assert len(raw) == hi-lo and len(scales) == shi-slo
        grid = np.frombuffer(scales, dtype='<u2').view(ml_dtypes.bfloat16).astype(np.float64).reshape(scale['shape'])
        assert np.isfinite(grid).all() and (grid > 0).all()
        expanded = grid.repeat(128, axis=0).repeat(128, axis=1)
        reference = np.frombuffer(raw, dtype=np.uint8).view(ml_dtypes.float8_e4m3fn).astype(np.float64).reshape(rows, cols)*expanded
        assert np.isfinite(reference).all()
        vectors = diagnostic_vectors(cols)
        modes = {}
        for mode, name in [(0, 'identity'), (1, 'hadamard')]:
            reconstructed, receipt = native_matrix(binary, raw, (rows, cols), mode)
            receipt.update(error_metrics(reference, reconstructed*expanded, vectors))
            receipt['total_bytes_with_unchanged_scales'] = receipt['packed_bytes_including_frames']+len(scales)
            modes[name] = receipt
        assert modes['identity']['total_bytes_with_unchanged_scales'] == modes['hadamard']['total_bytes_with_unchanged_scales']
        axis = int(projection == 'down_proj')
        dense = common.oracle.raw_frame(rows, cols, axis, raw)
        packed = invoke(lossless_binary, dense)
        assert invoke(lossless_binary, packed) == dense
        original_zero = int(np.count_nonzero(np.all(reference == 0, axis=1 if axis == 0 else 0)))
        records.append({'layer': layer, 'expert': expert, 'projection': projection, 'tensor': key,
                        'shape': [rows, cols], 'dtype': item['dtype'], 'header_sha256': header_sha,
                        'raw_sha256': sha(raw), 'scale_sha256': sha(scales), 'scale_shape': scale['shape'],
                        'diagnostic_vectors_sha256': sha(vectors.astype('<f8').tobytes()),
                        'raw_bytes_with_scales': len(raw)+len(scales), 'original_zero_channels': original_zero,
                        'lossless_zero_packed_bytes_with_scales': len(packed)+len(scales),
                        'lossless_comparator_is_not_an_equal_bit_approximation': True, 'modes': modes,
                        'rotation_weight_error_ratio': modes['hadamard']['relative_weight_frobenius_error']/modes['identity']['relative_weight_frobenius_error'],
                        'rotation_synthetic_output_error_ratio': modes['hadamard']['relative_synthetic_output_error']/modes['identity']['relative_synthetic_output_error']})
    assert len(records) == len(common.SELECTIONS) == 5
    return {'scope': 'Offline representation and synthetic linear-output diagnostics only.',
            'model': model.name, 'config_sha256': sha((model/'config.json').read_bytes()),
            'numpy_version': np.__version__, 'ml_dtypes_version': ml_dtypes.__version__,
            'finite_format_codes_independently_verified': 254, 'vectors_per_matrix': 40, 'vector_rng_seed': 49001,
            'real_model_activations': False, 'model_quality_tested': False, 'timing_or_energy_tested': False,
            'source_weights_retained': False, 'matrices': records}
