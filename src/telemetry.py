"""Synthetic 100 Hz driver telemetry for the ideal-line reference.

Driver model
  - v_driver(s) = v_ideal(s) * skill(s): product of per-corner Gaussian error
    dips (persistent across laps — these create the delta-heatmap hot spots),
    a lap modifier (warm-up faster, degradation slower), and AR(1) pace noise.
  - Lateral offset: apex clipping toward the inside of each detected curvature
    peak plus smooth wander; gives the driver trace a real racing line.
  - Sensor noise: low-passed speed noise + GPS position jitter at sample time.

Outputs
  outputs/telemetry_driver.csv   lap,t,lat,lon,alt,speed_ms,g_lat,g_long,
                                 throttle,brake,tire_temp_C,ers_battery_MJ
  outputs/delta.json             per-lap local & cumulative delta vs ideal,
                                 aligned to the ideal-line s grid.
"""
from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R_EARTH = 6371000.0
DT_HZ = 0.01          # 100 Hz
N_LAPS = 3

# (center_s_m, depth, width_m): persistent driver errors
CORNER_ERRORS = [
    (480, 0.015, 60),     # T1 entry slightly flat-footed
    (1180, 0.010, 80),    # S-Curves
    (1790, 0.045, 90),    # hairpin EXIT — the big persistent error
    (3460, 0.025, 70),    # Spoon entry
    (4480, 0.030, 80),    # 130R
    (6520, 0.010, 60),    # Casio chicane
]
# Per-lap scaling of corner-error depth: warm-up -> consistent, degradation -> sloppy.
ERR_SCALE = [1.0, 0.5, 1.5]


def _enu_to_latlon(x, y, lat0, lon0):
    lon = lon0 + np.degrees(x / (R_EARTH * np.cos(np.radians(lat0))))
    lat = lat0 + np.degrees(y / R_EARTH)
    return lat, lon


def main():
    ideal = json.load(open(os.path.join(HERE, "outputs", "ideal_line.json")))
    trk = json.load(open(os.path.join(HERE, "data", "track_suzuka.json")))
    pts = np.array([(p["lat"], p["lon"]) for p in trk["points"]])
    lat0, lon0 = float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))

    s = np.array(ideal["s"]); x = np.array(ideal["x"]); y = np.array(ideal["y"])
    z = np.array(ideal["z"]); v_id = np.array(ideal["v_ms"])
    T_tire = np.array(ideal["tire_temp_C"]); batt = np.array(ideal["ers_battery_J"]) / 1e6
    ds = float(ideal["meta"]["ds_m"])
    L = s[-1] + ds
    n = len(s)

    # Track tangent / normal for lateral offsets.
    tx = (np.roll(x, -1) - np.roll(x, 1)) / (2 * ds)
    ty = (np.roll(y, -1) - np.roll(y, 1)) / (2 * ds)
    tn = np.hypot(tx, ty); tx /= tn; ty /= tn
    nx_, ny_ = -ty, tx

    # Curvature recovered from the ideal-line traces.
    kappa = np.array(ideal["a_lat"]) / np.maximum(v_id ** 2, 1e-6)

    # Apex-clipping lateral offset: inside of each detected curvature peak.
    peaks = trk.get("detected_peaks_s_m", [])
    off = np.zeros(n)
    for ps in peaks:
        i = int(np.argmin(np.abs(s - (ps % L))))
        amp = 1.6
        side = float(np.sign(kappa[i])) or 1.0
        w = 45.0
        d = np.minimum(np.abs(s - s[i]), L - np.abs(s - s[i]))
        off += side * amp * np.exp(-(d ** 2) / (2 * w ** 2))
    # smooth wander
    rng = np.random.default_rng(42)
    from scipy.ndimage import uniform_filter1d
    wander = np.cumsum(rng.normal(0, 0.05, n))
    wander -= uniform_filter1d(wander, size=15, mode="wrap")
    off += wander * 0.3
    xd = x + nx_ * off
    yd = y + ny_ * off

    # Per-corner error Gaussians on the s grid (depth applied per lap below).
    gauss = []
    for c, depth, w in CORNER_ERRORS:
        d = np.minimum(np.abs(s - (c % L)), L - np.abs(s - (c % L)))
        gauss.append(np.exp(-(d ** 2) / (2 * w ** 2)))

    laps_data = []
    delta_out = {"s": s.tolist(), "laps": {}}
    t_global = 0.0
    for lap in range(1, N_LAPS + 1):
        skill = np.ones(n)
        for g, (_, depth, _) in zip(gauss, CORNER_ERRORS):
            skill *= 1.0 - depth * ERR_SCALE[lap - 1] * g
        v_d = v_id * skill
        # AR(1) pace noise (std ~0.12%)
        phi, sig = 0.97, 0.0012
        e = np.zeros(n); eps = rng.normal(0, sig / np.sqrt(1 - phi ** 2), n)
        for i in range(1, n):
            e[i] = phi * e[i - 1] + eps[i]
        e -= e.mean()   # zero net pace bias per lap; ordering set by ERR_SCALE
        v_d = v_d * (1.0 + e)

        dt_grid = ds / np.maximum(v_d, 0.5)
        t_cum = np.concatenate([[0.0], np.cumsum(dt_grid)])[:-1]
        lap_time = float(dt_grid.sum())

        # Driver dynamics on the s grid.
        dvds = (np.roll(v_d, -1) - np.roll(v_d, 1)) / (2 * ds)
        a_long = v_d * dvds
        from scipy.ndimage import uniform_filter1d
        a_long = uniform_filter1d(a_long, size=3, mode="wrap")
        a_lat = kappa * v_d ** 2
        throttle = (a_long > 0.5).astype(float)
        brake = np.clip(-a_long / 38.0, 0.0, 1.0)

        # Sample at 100 Hz over the lap.
        n_samp = int(round(lap_time / DT_HZ))
        t_q = np.arange(n_samp) * DT_HZ
        idx = np.searchsorted(t_cum, t_q, side="right") % n
        frac = (t_q - t_cum[idx]) / ds * v_d[idx]  # ~0..1 within segment
        frac = np.clip(frac, 0.0, 0.999)
        xi = idx; xj = (idx + 1) % n

        def interp(arr):
            return arr[xi] + frac * (arr[xj] - arr[xi])

        # GPS jitter (AR(1), smooth, meters in ENU) + low-passed speed noise
        def ar1(std, phi=0.9):
            out = np.zeros(n_samp)
            for i in range(1, n_samp):
                out[i] = phi * out[i - 1] + rng.normal(0, std)
            return out
        jit_x, jit_y = ar1(0.25), ar1(0.25)
        sp_noise = np.convolve(rng.normal(0, 0.15, n_samp), np.ones(5) / 5, mode="same")
        lat, lon = _enu_to_latlon(interp(xd) + jit_x, interp(yd) + jit_y, lat0, lon0)
        laps_data.append({
            "lap": lap, "t0": t_global,
            "lat": lat, "lon": lon, "alt": interp(z),
            "speed": interp(v_d) + sp_noise,
            "g_lat": a_lat[xi] + frac * (a_lat[xj] - a_lat[xi]),
            "g_long": a_long[xi] + frac * (a_long[xj] - a_long[xi]),
            "throttle": throttle[xi], "brake": brake[xi],
            "tire": T_tire[xi] + frac * (T_tire[xj] - T_tire[xi]),
            "ers": batt[xi] + frac * (batt[xj] - batt[xi]),
        })
        t_global += lap_time

        # Delta vs ideal on the shared s grid.
        t_id_cum = np.concatenate([[0.0], np.cumsum(ds / np.maximum(v_id, 0.5))])[:-1]
        cum_ms = (t_cum - t_id_cum) * 1000.0          # time gap at position s
        local_ms = np.gradient(cum_ms, s) * 25.0      # ms gained/lost per 25 m
        delta_out["laps"][str(lap)] = {
            "lap_time_s": round(lap_time, 3),
            "cum_delta_ms": np.round(cum_ms, 2).tolist(),
            "local_delta_ms": np.round(local_ms, 2).tolist(),
        }

    os.makedirs(os.path.join(HERE, "outputs"), exist_ok=True)
    import pandas as pd
    frames = []
    for d in laps_data:
        n_s = len(d["lat"])
        frames.append(pd.DataFrame({
            "lap": d["lap"], "t_s": np.arange(n_s) * DT_HZ + d["t0"],
            "lat": d["lat"].round(7), "lon": d["lon"].round(7),
            "alt_m": d["alt"].round(2), "speed_ms": d["speed"].round(3),
            "g_lat": d["g_lat"].round(4), "g_long": d["g_long"].round(4),
            "throttle": d["throttle"].astype(int), "brake": d["brake"].round(3),
            "tire_temp_C": d["tire"].round(2), "ers_battery_MJ": d["ers"].round(3)}))
    df = pd.concat(frames, ignore_index=True)
    csv_path = os.path.join(HERE, "outputs", "telemetry_driver.csv")
    df.to_csv(csv_path, index=False)

    with open(os.path.join(HERE, "outputs", "delta.json"), "w") as f:
        json.dump(delta_out, f)

    for lap in range(1, N_LAPS + 1):
        d = delta_out["laps"][str(lap)]
        print(f"lap{lap}: {d['lap_time_s']} s   max_cum_delta={max(d['cum_delta_ms']):.0f} ms")
    print("rows:", len(df), "->", csv_path)


if __name__ == "__main__":
    main()
