import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import sys
import traceback
from pathlib import Path

import torch

OUT = Path('results/step64b-fa2-bias')
ORIGINAL_D = 96
AUGMENTED_D = 104
ALPHA = 0.01
BETA = 3.0
MAX_ERROR_LIMIT = 0.08
MEAN_ERROR_LIMIT = 0.015


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def stage(n, name):
    print(f'STEP64B_STAGE={n}/7 {name}', flush=True)


def weights(batch, length, device):
    w = torch.ones((batch, length), device=device, dtype=torch.float32)
    a, b = length // 5, 4 * length // 5
    w[:, :a] = ALPHA
    w[:, a:b] = BETA
    return w, a, b


def augment(q, k, v, w):
    """Input [B, sequence, heads, 96], Q/K/V output width 104."""
    d = q.shape[-1]
    D = AUGMENTED_D
    if d != ORIGINAL_D or k.shape[-1] != d or v.shape[-1] != d:
        raise ValueError('Unexpected Q/K/V head dimensions')
    if not (q.ndim == k.ndim == v.ndim == 4):
        raise ValueError('Expected 4D Q/K/V')
    if q.shape[0] != k.shape[0] or k.shape[:3] != v.shape[:3]:
        raise ValueError('Mismatched Q/K/V batch or KV heads')
    if tuple(w.shape) != (k.shape[0], k.shape[1]):
        raise ValueError('Wrong key-weight shape')
    if not torch.isfinite(w).all() or not (w > 0).all():
        raise ValueError('Weights must be finite and positive')

    q_ext = torch.zeros((*q.shape[:-1], D - d), device=q.device, dtype=q.dtype)
    q_ext[..., 0] = math.sqrt(D)
    k_ext = torch.zeros((*k.shape[:-1], D - d), device=k.device, dtype=k.dtype)
    k_ext[..., 0] = w.log().to(k.dtype)[:, :, None]
    v_ext = torch.zeros((*v.shape[:-1], D - d), device=v.device, dtype=v.dtype)
    q_new = torch.cat([q * math.sqrt(D / d), q_ext], dim=-1).contiguous()
    k_new = torch.cat([k, k_ext], dim=-1).contiguous()
    v_new = torch.cat([v, v_ext], dim=-1).contiguous()
    return q_new, k_new, v_new


def reference(q, k, v, w, causal=True):
    """FP32 reference; supports MHA, GQA and bottom-right causal alignment."""
    hq, hk = q.shape[2], k.shape[2]
    if hq % hk:
        raise ValueError('Query heads must be divisible by KV heads')
    qf = q.float().transpose(1, 2)
    kf = k.float().transpose(1, 2).repeat_interleave(hq // hk, dim=1)
    vf = v.float().transpose(1, 2).repeat_interleave(hq // hk, dim=1)
    scores = torch.matmul(qf, kf.transpose(-1, -2)) / math.sqrt(q.shape[-1])
    scores = scores + w.log()[:, None, None, :]
    if causal:
        lq, lk = q.shape[1], k.shape[1]
        qi = torch.arange(lq, device=q.device)[:, None]
        kj = torch.arange(lk, device=q.device)[None, :]
        allowed = kj <= qi + lk - lq
        scores = scores.masked_fill(~allowed[None, None], -torch.inf)
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, vf).transpose(1, 2), probs


def errors(a, b):
    delta = (a.float() - b.float()).abs()
    return {'max_abs': float(delta.max().item()),
            'mean_abs': float(delta.mean().item())}


def acceptable(e):
    return e['max_abs'] <= MAX_ERROR_LIMIT and e['mean_abs'] <= MEAN_ERROR_LIMIT


def cpu_identity_test():
    torch.manual_seed(64)
    q = torch.randn(2, 7, 4, ORIGINAL_D, dtype=torch.float64)
    k = torch.randn(2, 7, 4, ORIGINAL_D, dtype=torch.float64)
    w, a, b = weights(2, 7, 'cpu')
    w = w.double()
    d, D = ORIGINAL_D, AUGMENTED_D
    q_ext = torch.zeros(2, 7, 4, D - d, dtype=torch.float64)
    k_ext = torch.zeros_like(q_ext)
    q_ext[..., 0] = math.sqrt(D)
    k_ext[..., 0] = w.log()[:, :, None]
    qa = torch.cat([q * math.sqrt(D / d), q_ext], dim=-1)
    ka = torch.cat([k, k_ext], dim=-1)
    lhs = torch.einsum('bihd,bjhd->bhij', qa, ka) / math.sqrt(D)
    rhs = torch.einsum('bihd,bjhd->bhij', q, k) / math.sqrt(d)
    rhs += w.log()[:, None, None, :]
    max_error = float((lhs - rhs).abs().max().item())
    if max_error > 1e-11:
        raise AssertionError(f'CPU mathematical identity failed: {max_error}')
    return {'max_error_float64': max_error,
            'head_dim': d, 'augmented_head_dim': D,
            'alpha': ALPHA, 'beta': BETA,
            'roi_cut': a, 'nonroi_end': b}


def dense_gpu_case(name, batch, lq, lk, hq, hk, causal, flash_attn_func):
    device = 'cuda:0'
    q = (torch.randn(batch, lq, hq, ORIGINAL_D, device=device,
                     dtype=torch.bfloat16) * 0.4).contiguous()
    k = (torch.randn(batch, lk, hk, ORIGINAL_D, device=device,
                     dtype=torch.bfloat16) * 0.4).contiguous()
    v = (torch.randn(batch, lk, hk, ORIGINAL_D, device=device,
                     dtype=torch.bfloat16) * 0.4).contiguous()
    w, roi_end, nonroi_end = weights(batch, lk, device)
    neutral = torch.ones_like(w)
    qn, kn, vn = augment(q, k, v, neutral)
    qb, kb, vb = augment(q, k, v, w)
    with torch.inference_mode():
        native = flash_attn_func(q, k, v, dropout_p=0.0,
                                 softmax_scale=1 / math.sqrt(ORIGINAL_D), causal=causal)
        aug_neutral = flash_attn_func(qn, kn, vn, dropout_p=0.0,
                                      softmax_scale=1 / math.sqrt(AUGMENTED_D), causal=causal)
        aug_biased = flash_attn_func(qb, kb, vb, dropout_p=0.0,
                                     softmax_scale=1 / math.sqrt(AUGMENTED_D), causal=causal)
        torch.cuda.synchronize()
        ref_neutral, p0 = reference(q, k, v, neutral, causal)
        ref_biased, p1 = reference(q, k, v, w, causal)
    native_error = errors(aug_neutral[..., :ORIGINAL_D], native)
    neutral_error = errors(aug_neutral[..., :ORIGINAL_D], ref_neutral)
    biased_error = errors(aug_biased[..., :ORIGINAL_D], ref_biased)
    effect = errors(aug_biased[..., :ORIGINAL_D], aug_neutral[..., :ORIGINAL_D])
    roi_before = float(p0[:, :, -1, :roi_end].sum(-1).mean().item())
    roi_after = float(p1[:, :, -1, :roi_end].sum(-1).mean().item())
    nonroi_before = float(p0[:, :, -1, roi_end:nonroi_end].sum(-1).mean().item())
    nonroi_after = float(p1[:, :, -1, roi_end:nonroi_end].sum(-1).mean().item())
    checks = {
        'neutral_fa2_parity': acceptable(native_error),
        'neutral_fp32_reference_parity': acceptable(neutral_error),
        'biased_fp32_reference_parity': acceptable(biased_error),
        'attention_output_changes': effect['max_abs'] > 0.001,
        'roi_mass_decreases': roi_after < roi_before - 0.01,
        'nonroi_mass_increases': nonroi_after > nonroi_before + 0.01,
        'finite_outputs': bool(torch.isfinite(aug_biased).all().item()),
        'output_dimensions_preserved': aug_biased[..., :ORIGINAL_D].shape == q.shape,
    }
    return {'name': name, 'shapes': {'q': list(q.shape), 'kv': list(k.shape),
                                     'augmented_head_dim': AUGMENTED_D},
            'causal': causal, 'neutral_vs_native': native_error,
            'neutral_vs_fp32': neutral_error, 'biased_vs_fp32': biased_error,
            'output_effect': effect,
            'reference_last_query_roi_mass': [roi_before, roi_after],
            'reference_last_query_nonroi_mass': [nonroi_before, nonroi_after],
            'checks': checks, 'pass': all(checks.values())}


def varlen_gpu_case(flash_attn_varlen_func):
    device = 'cuda:0'
    lens = (64, 45)
    hq, hk, d = 32, 32, ORIGINAL_D
    q = torch.randn(sum(lens), hq, d, device=device, dtype=torch.bfloat16) * 0.4
    k = torch.randn(sum(lens), hk, d, device=device, dtype=torch.bfloat16) * 0.4
    v = torch.randn(sum(lens), hk, d, device=device, dtype=torch.bfloat16) * 0.4
    cu = torch.tensor((0, lens[0], sum(lens)), dtype=torch.int32, device=device)
    native_parts = []
    aug_parts = []
    ref_parts = []
    index = 0
    with torch.inference_mode():
        for length in lens:
            qi = q[index:index+length].unsqueeze(0)
            ki = k[index:index+length].unsqueeze(0)
            vi = v[index:index+length].unsqueeze(0)
            wi, _, _ = weights(1, length, device)
            q1, k1, v1 = augment(qi, ki, vi, wi)
            aug_parts.append((q1.squeeze(0), k1.squeeze(0), v1.squeeze(0)))
            ref_parts.append(reference(qi, ki, vi, wi, causal=True)[0].squeeze(0))
            index += length
        qa = torch.cat([x[0] for x in aug_parts]).contiguous()
        ka = torch.cat([x[1] for x in aug_parts]).contiguous()
        va = torch.cat([x[2] for x in aug_parts]).contiguous()
        out = flash_attn_varlen_func(qa, ka, va, cu_seqlens_q=cu,
                                     cu_seqlens_k=cu, max_seqlen_q=max(lens),
                                     max_seqlen_k=max(lens), dropout_p=0.0,
                                     softmax_scale=1 / math.sqrt(AUGMENTED_D), causal=True)
        torch.cuda.synchronize()
        ref = torch.cat(ref_parts)
        diff = errors(out[..., :ORIGINAL_D], ref)
    checks = {'bf16_varlen_matches_fp32_reference': acceptable(diff),
              'finite_outputs': bool(torch.isfinite(out).all().item()),
              'total_tokens_preserved': out.shape[0] == sum(lens)}
    return {'name': 'varlen_unpadded', 'lengths': list(lens),
            'biased_vs_fp32': diff, 'checks': checks, 'pass': all(checks.values())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    job = os.environ.get('SLURM_JOB_ID', 'local')
    path = OUT / f'step64b-{job}.json'
    report = {'stage': 'STEP64B', 'job': job,
              'scope': 'BF16/FA2 QK augmentation kernel feasibility (not Phi integration)',
              'source_sha256': file_sha256(__file__),
              'status': 'INCOMPLETE', 'tests': [], 'errors': []}
    try:
        stage(1, 'CPU_exact_mathematical_identity')
        report['cpu_identity'] = cpu_identity_test()
        print('STEP64B_CPU_IDENTITY=PASS', flush=True)
        if args.preflight:
            report['status'] = 'PREFLIGHT_PASS'
        else:
            stage(2, 'load_actual_flashattention2_GPU_kernel')
            if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
                raise RuntimeError('Exactly one visible CUDA GPU is required')
            from flash_attn import flash_attn_func, flash_attn_varlen_func
            report['environment'] = {
                'gpu': torch.cuda.get_device_name(0),
                'torch': torch.__version__, 'cuda': torch.version.cuda,
                'flash_attn': importlib.metadata.version('flash-attn'),
                'flash_func_module': flash_attn_func.__module__,
                'dtype': 'torch.bfloat16', 'head_dim': ORIGINAL_D,
                'augmented_dim': AUGMENTED_D,
            }
            torch.manual_seed(6402)
            torch.cuda.manual_seed_all(6402)
            cases = [
                (3, 'prefill_mha', 1, 128, 128, 32, 32, True),
                (4, 'decode_mha', 1, 1, 128, 32, 32, False),
                (5, 'prefill_gqa', 1, 64, 64, 32, 8, True),
            ]
            for number, name, batch, lq, lk, hq, hk, causal in cases:
                stage(number, name)
                result = dense_gpu_case(name, batch, lq, lk, hq, hk, causal, flash_attn_func)
                report['tests'].append(result)
                print(f"STEP64B_TEST {name}={'PASS' if result['pass'] else 'FAIL'} "
                      f"max_error={result['biased_vs_fp32']['max_abs']:.6f}", flush=True)
            stage(6, 'varlen_padding_equivalent_unpad_path')
            result = varlen_gpu_case(flash_attn_varlen_func)
            report['tests'].append(result)
            print(f"STEP64B_TEST {result['name']}={'PASS' if result['pass'] else 'FAIL'} "
                  f"max_error={result['biased_vs_fp32']['max_abs']:.6f}", flush=True)
            report['gpu_peak_allocated_gib'] = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
            report['status'] = ('GPU_PASS' if all(x['pass'] for x in report['tests'])
                                else 'FAILED_NUMERICS')
    except Exception as e:
        report['status'] = 'ERROR'
        report['errors'].append({'type': type(e).__name__, 'message': str(e),
                                 'traceback': traceback.format_exc()[-5000:]})
        print('STEP64B_ERROR=' + repr(e), flush=True)
    finally:
        stage(7, 'write_final_report')
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
        print('STEP64B_REPORT=' + str(path), flush=True)
        print('STEP64B_SOURCE_SHA256=' + report['source_sha256'], flush=True)
        print('STEP64B_STATUS=' + report['status'], flush=True)
    return 1 if report['status'] in ('ERROR', 'FAILED_NUMERICS') else 0


if __name__ == '__main__':
    sys.exit(main())
