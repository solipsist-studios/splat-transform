#!/usr/bin/env python3
"""
xz_to_omg4.py  –  Convert an OMG4 compressed 4DGS model (comp.xz) to the
compact .omg4 v2 binary format consumed by the supersplat-viewer.

OMG4 (arXiv 2510.03857) represents a dynamic scene with 4D space-time
Gaussians. Each Gaussian has a 4x4 covariance built from two quaternions
(a 4D rotor) and four scales (three spatial + one temporal). Rendering at
time t "slices" the 4D Gaussian:

    L      = R4(q_l, q_r) @ diag(exp(s_xyz), exp(s_t))
    Sigma  = L @ L^T                       (4x4)
    S11    = Sigma[:3,:3]  S12 = Sigma[:3,3]  Stt = Sigma[3,3]
    Sigma3D(t) = S11 - outer(S12, S12) / Stt      (constant in t)
    mean(t)    = xyz + (S12 / Stt) * (t - t_c)    (linear motion)
    opacity(t) = sigmoid(o) * exp(-0.5 * (t - t_c)^2 / Stt)

Color and (peak) opacity come from small MLPs evaluated on the contracted
position and normalised time. This exporter bakes every per-Gaussian
quantity once (evaluating the MLPs at each Gaussian's own temporal centre,
where it is most visible) and stores the temporal parameters explicitly so
the viewer can evaluate motion and temporal fade per rendered frame on the
GPU. There is no per-frame data: the file covers the full clip continuously.

Requirements: torch numpy dahuffman  (the OMG4 training environment)

Usage:
    python xz_to_omg4.py \
        --input  path/to/comp.xz \
        --output path/to/scene.omg4 \
        --time_min 0.0 --time_max 10.0 --fps 30

time_min/time_max MUST match the `time_duration` the model was trained
with (configs/dynerf/*.yaml in the OMG4 repo: [0.0, 10.0]); both the
temporal-opacity units and the MLP time normalisation depend on it.

.omg4 v2 file format (all values little-endian) — see splat4d_io.py for the
shared spec:
    Header (32 bytes):
        uint32  magic = 0x34474D4F ("OMG4")
        uint32  version = 2
        uint32  numSplats (N)
        uint32  flags = 0 (reserved)
        float32 timeMin, timeMax   (clip time range, seconds)
        float32 fps                (advisory, for UI only)
        uint32  reserved = 0
    Data: 19 SoA float32[N] arrays, in order:
        x y z                      position at t = t_center
        rot_0 rot_1 rot_2 rot_3    quaternion (w,x,y,z) of sliced 3D covariance
        scale_0 scale_1 scale_2    log-space sqrt eigenvalues of sliced 3D cov
        opacity                    logit-space peak opacity
        f_dc_0 f_dc_1 f_dc_2       SH DC coefficients
        vx vy vz                   velocity, scene units / second
        t_center                   temporal centre, seconds
        t_sigma                    temporal std-dev, seconds
"""

import argparse
import lzma
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from splat4d_io import OMG4_MAGIC, write_omg4_v2_header, report_output

try:
    import dahuffman
except ImportError:
    sys.exit("ERROR: 'dahuffman' package not found. Install it with: pip install dahuffman")


# ---------------------------------------------------------------------------
# SVQ / Huffman decode (mirrors utils/compress_utils.py + decode() in the
# OMG4 repository's scene/gaussian_model.py)
# ---------------------------------------------------------------------------

def huffman_decode(encoded_bytes, huffman_table, count):
    codec = dahuffman.HuffmanCodec(code_table=huffman_table)
    return np.fromiter(codec.decode(encoded_bytes), dtype=np.uint16, count=count)


def decode_all_layers(code_list, index_list, htable_list, count):
    """Decode a full VQ attribute (split into sub-vector slices)."""
    parts = []
    for codes, idx_bytes, htable in zip(code_list, index_list, htable_list):
        labels = huffman_decode(idx_bytes, htable, count)
        codes = np.asarray(codes, dtype=np.float32)
        parts.append(codes[labels])
    return np.concatenate(parts, axis=-1).astype(np.float32)


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().float().cpu().numpy()
    return np.asarray(x, dtype=np.float32)


# ---------------------------------------------------------------------------
# MLP forward pass.
#
# The published OMG4 checkpoints are trained with the TorchFallbackMLP in
# scene/gaussian_model.py (used when tiny-cuda-nn is not installed): plain
# nn.Linear layers WITH biases, no padding, hidden width 64, one hidden
# layer. The flat `params` vector is torch's parameters() order:
#   linear0.weight [64, n_in] | linear0.bias [64] |
#   linear1.weight [n_out, 64] | linear1.bias [n_out]
# Hidden activation: ReLU for mlp_cont, LeakyReLU(negative_slope=0.1) for
# mlp_dc / mlp_view / mlp_opacity. No output activation.
# ---------------------------------------------------------------------------

def unpack_linear_mlp(params_f16, n_in, n_hidden, n_out):
    p = np.asarray(params_f16, dtype=np.float32)
    expected = n_hidden * n_in + n_hidden + n_out * n_hidden + n_out
    if p.size != expected:
        raise ValueError(
            f"MLP param count mismatch: got {p.size}, expected {expected} "
            f"for Linear({n_in},{n_hidden})+Linear({n_hidden},{n_out}) with biases. "
            "Was this checkpoint trained with tiny-cuda-nn instead of the torch fallback?")
    o = 0
    w0 = p[o:o + n_hidden * n_in].reshape(n_hidden, n_in); o += n_hidden * n_in
    b0 = p[o:o + n_hidden]; o += n_hidden
    w1 = p[o:o + n_out * n_hidden].reshape(n_out, n_hidden); o += n_out * n_hidden
    b1 = p[o:o + n_out]
    return w0, b0, w1, b1


def mlp_forward(params_f16, x, n_hidden, n_out, hidden_activation):
    w0, b0, w1, b1 = unpack_linear_mlp(params_f16, x.shape[1], n_hidden, n_out)
    h = x @ w0.T + b0
    if hidden_activation == 'relu':
        h = np.maximum(h, 0.0)
    elif hidden_activation == 'leaky_relu':
        h = np.where(h >= 0.0, h, 0.1 * h)   # slope 0.1, matching TorchFallbackMLP
    else:
        raise ValueError(hidden_activation)
    return h @ w1.T + b1


def frequency_encode(x, n_frequencies=16):
    """TorchFallbackMLP ordering: per octave i, [sin(x*2^i*pi) (D dims), cos(...) (D dims)]."""
    out = []
    for i in range(n_frequencies):
        freq = (2.0 ** i) * math.pi
        out.append(np.sin(x * freq))
        out.append(np.cos(x * freq))
    return np.concatenate(out, axis=-1).astype(np.float32)


def contract_to_unisphere(x):
    """Scene contraction with the fixed aabb [-1,1]^3 used by the OMG4 renderer."""
    # aabb is [-1,1] so the normalisation-to-[-1,1] step is the identity
    x = x.copy()
    mag = np.linalg.norm(x, axis=-1, keepdims=True)
    mask = mag[:, 0] > 1
    x[mask] = (2 - 1 / mag[mask]) * (x[mask] / mag[mask])
    return x / 4 + 0.5


# ---------------------------------------------------------------------------
# 4D rotor / covariance math (mirrors utils/general_utils.py)
# ---------------------------------------------------------------------------

def build_rotation_4d(l, r):
    """4D rotation matrices [N,4,4] from left/right quaternions (unnormalised ok)."""
    q_l = l / np.linalg.norm(l, axis=-1, keepdims=True)
    q_r = r / np.linalg.norm(r, axis=-1, keepdims=True)

    a, b, c, d = q_l[:, 0], q_l[:, 1], q_l[:, 2], q_l[:, 3]
    p, q, r_, s = q_r[:, 0], q_r[:, 1], q_r[:, 2], q_r[:, 3]

    N = l.shape[0]
    M_l = np.stack([a, -b, -c, -d,
                    b, a, -d, c,
                    c, d, a, -b,
                    d, -c, b, a], axis=0).reshape(4, 4, N).transpose(2, 0, 1)
    M_r = np.stack([p, q, r_, s,
                    -q, p, -s, r_,
                    -r_, s, p, -q,
                    -s, -r_, q, p], axis=0).reshape(4, 4, N).transpose(2, 0, 1)
    A = M_l @ M_r
    # matches torch.flip(A, [1, 2])
    return A[:, ::-1, ::-1]


def quat_from_rotmat(R):
    """Batch rotation matrix [N,3,3] -> quaternion [N,4] (w,x,y,z), w >= 0."""
    m00, m01, m02 = R[:, 0, 0], R[:, 0, 1], R[:, 0, 2]
    m10, m11, m12 = R[:, 1, 0], R[:, 1, 1], R[:, 1, 2]
    m20, m21, m22 = R[:, 2, 0], R[:, 2, 1], R[:, 2, 2]
    tr = m00 + m11 + m22

    # Case selection: trace, or the largest diagonal element.
    case = np.where(tr > 0, 0,
           np.where((m00 >= m11) & (m00 >= m22), 1,
           np.where(m11 >= m22, 2, 3)))

    s0 = np.sqrt(np.maximum(tr + 1.0, 0.0)) * 2               # 4*qw
    s1 = np.sqrt(np.maximum(1.0 + m00 - m11 - m22, 0.0)) * 2  # 4*qx
    s2 = np.sqrt(np.maximum(1.0 - m00 + m11 - m22, 0.0)) * 2  # 4*qy
    s3 = np.sqrt(np.maximum(1.0 - m00 - m11 + m22, 0.0)) * 2  # 4*qz
    eps = 1e-12

    q0 = np.stack([0.25 * s0, (m21 - m12) / (s0 + eps), (m02 - m20) / (s0 + eps), (m10 - m01) / (s0 + eps)], axis=1)
    q1 = np.stack([(m21 - m12) / (s1 + eps), 0.25 * s1, (m01 + m10) / (s1 + eps), (m02 + m20) / (s1 + eps)], axis=1)
    q2 = np.stack([(m02 - m20) / (s2 + eps), (m01 + m10) / (s2 + eps), 0.25 * s2, (m12 + m21) / (s2 + eps)], axis=1)
    q3 = np.stack([(m10 - m01) / (s3 + eps), (m02 + m20) / (s3 + eps), (m12 + m21) / (s3 + eps), 0.25 * s3], axis=1)

    q = np.select([(case == 0)[:, None], (case == 1)[:, None], (case == 2)[:, None]], [q0, q1, q2], q3)
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    q[q[:, 0] < 0] *= -1.0
    return q.astype(np.float32)


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def convert(xz_path, out_path, time_min, time_max, fps, prune_threshold, include_sh=True):
    print(f"Loading {xz_path} …")
    with lzma.open(xz_path, "rb") as f:
        save_dict = pickle.load(f)

    xyz = to_numpy(save_dict['xyz'])                # [N,3]
    t_center = to_numpy(save_dict['t'])[:, 0]       # [N]
    N = xyz.shape[0]

    print(f"  {N:,} Gaussians | time range [{time_min}, {time_max}] s")

    scaling = decode_all_layers(save_dict['scale_code'], save_dict['scale_index'], save_dict['scale_htable'], N)          # [N,3] log
    rotation_l = decode_all_layers(save_dict['rotation_code'], save_dict['rotation_index'], save_dict['rotation_htable'], N)  # [N,4]
    rotation_r = decode_all_layers(save_dict['rotation_r_code'], save_dict['rotation_r_index'], save_dict['rotation_r_htable'], N)  # [N,4]
    scaling_t = decode_all_layers(save_dict['scaling_t_code'], save_dict['scaling_t_index'], save_dict['scaling_t_htable'], N)[:, 0]  # [N] log
    appearance = decode_all_layers(save_dict['app_code'], save_dict['app_index'], save_dict['app_htable'], N)             # [N,6]
    features_static = appearance[:, 0:3]

    # ── 4D covariance slicing ───────────────────────────────────────────────
    print("  Slicing 4D covariance …")
    R4 = build_rotation_4d(rotation_l, rotation_r).astype(np.float64)         # [N,4,4]
    s4 = np.concatenate([np.exp(scaling), np.exp(scaling_t)[:, None]], axis=1).astype(np.float64)  # [N,4]
    L = R4 * s4[:, None, :]                                                    # R4 @ diag(s)
    Sigma = L @ L.transpose(0, 2, 1)                                           # [N,4,4]

    S11 = Sigma[:, :3, :3]
    S12 = Sigma[:, :3, 3]                                                      # [N,3]
    Stt = np.maximum(Sigma[:, 3, 3], 1e-12)                                    # [N]

    velocity = (S12 / Stt[:, None]).astype(np.float32)                         # units/sec
    t_sigma = np.sqrt(Stt).astype(np.float32)
    Sigma3D = S11 - np.einsum('ni,nj->nij', S12, S12) / Stt[:, None, None]

    # Eigendecompose the sliced covariance -> principal axes + scales
    eigval, eigvec = np.linalg.eigh(Sigma3D)                                   # ascending
    eigval = np.maximum(eigval, 1e-18)
    # ensure right-handed rotation before quaternion conversion
    neg = np.linalg.det(eigvec) < 0
    eigvec[neg, :, 2] *= -1
    quat = quat_from_rotmat(eigvec)                                            # (w,x,y,z)
    log_scales = (0.5 * np.log(eigval)).astype(np.float32)                     # log(sqrt(eig))

    # ── MLP appearance at each Gaussian's own temporal centre ───────────────
    print("  Evaluating appearance MLPs …")
    t_eval = np.clip(t_center, time_min, time_max)
    t_norm = (t_eval - time_min) / max(time_max - time_min, 1e-9)

    xyz_c = contract_to_unisphere(xyz.astype(np.float64)).astype(np.float32)
    xyzt = np.concatenate([xyz_c, t_norm[:, None].astype(np.float32)], axis=1)  # [N,4]

    encoded = frequency_encode(xyzt, n_frequencies=16)                          # [N,128]
    cont_feat = mlp_forward(save_dict['MLP_cont'], encoded, 64, 13, 'relu')     # [N,13]
    space_feat = np.concatenate([cont_feat, features_static], axis=1)           # [N,16]

    f_dc = mlp_forward(save_dict['MLP_dc'], space_feat, 64, 3, 'leaky_relu')    # [N,3] SH DC
    opacity_logit = mlp_forward(save_dict['MLP_opacity'], space_feat, 64, 1, 'leaky_relu')[:, 0]  # [N]

    # ── View-dependent SH (optional) ────────────────────────────────────────
    # The reference model evaluates 48 4D-spherindrical-harmonic coefficients:
    # 16 spatial (deg 3) x 3 temporal cosine bands (utils/sh_utils.py
    # eval_shfs_4d). At dir_t = t - t_center = 0 both cosine bands equal 1, so
    # the effective spatial coefficient k collapses to
    #   eff[k] = sh[k] + sh[k+16] + sh[k+32]
    # which is exactly a standard 3-band 3DGS SH set. Coefficient 0 folds into
    # f_dc; coefficients 1..15 become f_rest (PLY channel-major layout).
    f_rest = None
    if include_sh:
        features_view = appearance[:, 3:6]
        view_feat = np.concatenate([cont_feat, features_view], axis=1)          # [N,16]
        view_sh = mlp_forward(save_dict['MLP_sh'], view_feat, 64, 141, 'leaky_relu').reshape(-1, 47, 3)
        # full coeff j (1..47) = view_sh[:, j-1]; temporal fold at t = t_center:
        f_dc = f_dc + view_sh[:, 15, :] + view_sh[:, 31, :]
        # eff[k] for k=1..15: view indices k-1, k+15, k+31
        f_rest = (view_sh[:, 0:15, :] + view_sh[:, 16:31, :] + view_sh[:, 32:47, :])  # [N,15,3]

    # ── Prune Gaussians that can never contribute inside the time range ─────
    # Peak temporal weight over [time_min, time_max]:
    dist = np.abs(t_center - np.clip(t_center, time_min, time_max))
    peak_weight = np.exp(-0.5 * (dist / np.maximum(t_sigma, 1e-9)) ** 2)
    peak_alpha = (1.0 / (1.0 + np.exp(-opacity_logit))) * peak_weight
    keep = peak_alpha >= prune_threshold
    kept = int(keep.sum())
    print(f"  Pruning: dropped {N - kept:,} / {N:,} Gaussians below alpha {prune_threshold:.5f} inside time range")

    arrays = [
        xyz[:, 0], xyz[:, 1], xyz[:, 2],
        quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3],
        log_scales[:, 0], log_scales[:, 1], log_scales[:, 2],
        opacity_logit,
        f_dc[:, 0], f_dc[:, 1], f_dc[:, 2],
        velocity[:, 0], velocity[:, 1], velocity[:, 2],
        t_center,
        t_sigma
    ]
    if f_rest is not None:
        # PLY channel-major order: f_rest_{ch*15 + k} = eff_rest[k][channel ch]
        for ch in range(3):
            for k in range(15):
                arrays.append(f_rest[:, k, ch])
    arrays = [np.ascontiguousarray(a[keep], dtype=np.float32) for a in arrays]

    # ── Diagnostics ─────────────────────────────────────────────────────────
    for t in np.linspace(time_min, time_max, 5):
        w = np.exp(-0.5 * ((t_center[keep] - t) / np.maximum(t_sigma[keep], 1e-9)) ** 2)
        alpha = (1.0 / (1.0 + np.exp(-opacity_logit[keep]))) * w
        print(f"    t={t:6.2f}s  active(alpha>1/255): {(alpha > 1 / 255).mean() * 100:5.1f}%")
    vmag = np.linalg.norm(velocity[keep], axis=1)
    print(f"    |velocity| p50/p95/p99: {np.percentile(vmag, [50, 95, 99]).round(4)}")
    print(f"    t_sigma    p50/p95    : {np.percentile(t_sigma[keep], [50, 95]).round(3)}")

    # ── Write ───────────────────────────────────────────────────────────────
    flags = 1 if f_rest is not None else 0
    with open(out_path, 'wb') as fp:
        write_omg4_v2_header(fp, OMG4_MAGIC, kept, time_min, time_max, fps, flags)
        for a in arrays:
            fp.write(a.tobytes())

    report_output(out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert OMG4 comp.xz checkpoint to compact .omg4 v2 for supersplat-viewer')
    parser.add_argument('--input', required=True, help='Path to comp.xz (OMG4 output)')
    parser.add_argument('--output', required=True, help='Destination .omg4 file')
    parser.add_argument('--time_min', type=float, default=0.0,
                        help='Training time_duration min in seconds (default: 0.0)')
    parser.add_argument('--time_max', type=float, default=10.0,
                        help='Training time_duration max in seconds (default: 10.0)')
    parser.add_argument('--fps', type=float, default=30.0, help='Advisory fps for UI (default: 30)')
    parser.add_argument('--prune_threshold', type=float, default=1.0 / 1024,
                        help='Drop Gaussians whose peak alpha inside the time range is below this (default: 1/1024; 0 disables)')
    parser.add_argument('--no_sh', action='store_true',
                        help='Skip baking the 3-band view-dependent SH coefficients (smaller file, flatter shading)')
    args = parser.parse_args()

    convert(args.input, args.output, args.time_min, args.time_max, args.fps,
            args.prune_threshold, include_sh=not args.no_sh)
