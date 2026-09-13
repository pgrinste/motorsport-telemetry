"""GPU Monte Carlo ideal-line ensemble.

Samples the vehicle/surface parameter space (grip mu, braking & drive
capability, aero drag) and runs the two-pass friction-circle solver for all
samples in parallel as a [N, n] tensor on one GPU. Outputs lap-time stats and
p10/p50/p90 speed bands (the uncertainty envelope shown in the frontend).

Usage: CUDA_VISIBLE_DEVICES=0 python src/mc_ideal_line.py
"""
from __future__ import annotations

import json
import math
import os
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
G = 9.80665
N_SAMPLES = 4096
MAX_SWEEPS = 16
TOL = 1e-3


def main():
    from physics import load_track, to_enu_m, resample_closed, signed_curvature

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if dev == "cuda" else "CPU fallback"
    print(f"device: {dev} ({gpu_name})")

    meta, pts = load_track(os.path.join(HERE, "data", "track_suzuka.json"))
    lat0, lon0 = float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))
    x, y, z = to_enu_m(pts[:, 0], pts[:, 1], pts[:, 2], lat0, lon0)
    xr, yr, zr, s, ds = resample_closed(x, y, z, 2.0)
    kappa = signed_curvature(xr, yr, ds)
    n = len(s)

    # Parameter distributions (F1-class envelopes).
    rng = np.random.default_rng(7)
    mu = torch.tensor(rng.uniform(1.00, 1.12, N_SAMPLES), dtype=torch.float32, device=dev)
    a_brake = torch.tensor(rng.uniform(35.0, 42.0, N_SAMPLES), dtype=torch.float32, device=dev)
    a_acc = torch.tensor(rng.uniform(10.0, 12.5, N_SAMPLES), dtype=torch.float32, device=dev)
    k_drag = torch.tensor(rng.uniform(0.0011, 0.0014, N_SAMPLES), dtype=torch.float32, device=dev)

    kt = torch.tensor(kappa, dtype=torch.float32, device=dev).unsqueeze(0)  # [1, n]
    vlat = torch.sqrt(mu[:, None] * G / torch.clamp(torch.abs(kt), min=1e-4))  # [N, n]
    est_vram_mb = (vlat.numel() * 4 * 6) / 1e6
    print(f"batch {N_SAMPLES} x {n} pts | est VRAM ~{est_vram_mb:.0f} MB")

    # Fixed rotation at the globally slowest point (per-sample argmin differs by <2%).
    i0 = int(np.argmin(vlat.mean(dim=0).cpu().numpy()))
    vlat_r = torch.roll(vlat, -i0, dims=1)
    ab_r = a_brake[:, None] * ds   # [N, 1] for the vectorized backward pass
    aa = a_acc                     # [N]
    kd = k_drag                    # [N]

    v = vlat_r.clone()
    t0 = time.time()
    for sweep in range(MAX_SWEEPS):
        vb = torch.sqrt(torch.clamp(torch.roll(v, -1, dims=1) ** 2 + 2 * ab_r, min=0.0))
        vn = torch.minimum(vlat_r, torch.minimum(v, vb))
        # Sequential forward sweep along the track (elementwise over the batch).
        for i in range(1, n):
            vp = vn[:, i - 1]
            c2 = vp * vp + 2.0 * ds * (aa - kd * vp * vp)
            cand = torch.sqrt(torch.clamp(c2, min=0.0))
            lim = torch.minimum(vlat_r[:, i], vb[:, i])
            vn[:, i] = torch.minimum(cand, lim)
        # Wrap seam.
        vp = vn[:, -1]
        c2 = vp * vp + 2.0 * ds * (aa - kd * vp * vp)
        cand = torch.sqrt(torch.clamp(c2, min=0.0))
        vn[:, 0] = torch.minimum(cand, torch.minimum(vlat_r[:, 0], vb[:, 0]))
        delta = float((vn - v).abs().max())
        v = vn
        if delta < TOL:
            break
    wall = time.time() - t0
    print(f"solver converged in {sweep + 1} sweeps ({wall:.1f}s)")

    lap_time = (ds / torch.clamp(v, min=0.5)).sum(dim=1)  # per-sample lap time
    v_unroll = torch.roll(v, i0, dims=1).cpu().numpy()

    step = 4  # downsample to ~8 m for storage
    out = {
        "meta": {
            "n_samples": N_SAMPLES,
            "device": dev,
            "gpu": gpu_name,
            "wall_s": round(wall, 1),
            "sweeps": sweep + 1,
            "distributions": {
                "mu": [1.00, 1.12], "a_brake_m_s2": [35.0, 42.0],
                "a_acc_m_s2": [10.0, 12.5], "k_drag": [0.0011, 0.0014]},
        },
        "lap_time_s": {
            "mean": round(float(lap_time.mean()), 3),
            "p10": round(float(torch.quantile(lap_time, 0.10)), 3),
            "p50": round(float(torch.quantile(lap_time, 0.50)), 3),
            "p90": round(float(torch.quantile(lap_time, 0.90)), 3),
        },
        "s": np.round(s[::step], 2).tolist(),
        "v_p10_ms": np.round(np.percentile(v_unroll[:, ::step], 10, axis=0), 2).tolist(),
        "v_p50_ms": np.round(np.percentile(v_unroll[:, ::step], 50, axis=0), 2).tolist(),
        "v_p90_ms": np.round(np.percentile(v_unroll[:, ::step], 90, axis=0), 2).tolist(),
    }
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    with open(os.path.join(HERE, "data", "mc_results.json"), "w") as f:
        json.dump(out, f)
    print(f"lap_time mean={out['lap_time_s']['mean']} s  "
          f"[{out['lap_time_s']['p10']}, {out['lap_time_s']['p90']}]")

    # Quick band figure.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 4))
    ss = np.array(out["s"]) / 1000
    p10, p50, p90 = (np.array(x) * 3.6 for x in (out["v_p10_ms"], out["v_p50_ms"], out["v_p90_ms"]))
    ax.plot(ss, p90, color="steelblue", lw=0.8, alpha=0.7)
    ax.fill_between(ss, p10, p90, color="steelblue", alpha=0.25, label="p10–p90 (4096 samples)")
    ax.plot(ss, p50, color="navy", lw=1.8, label="median")
    ax.set_xlabel("track position km"); ax.set_ylabel("speed km/h")
    ax.set_title(f"Monte Carlo ideal-line speed envelope — {N_SAMPLES} samples on {gpu_name}")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout()
    out_png = os.path.join(HERE, "outputs", "figures", "mc_band.png")
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=110)
    print("saved", out_png)


if __name__ == "__main__":
    main()
