"""Physics engine — Suzuka ideal line & vehicle-dynamics traces.

Pipeline
  1. Load OSM centerline (data/track_suzuka.json), project to local ENU meters.
  2. Resample to uniform arc length (default 2 m).
  3. Signed curvature via central differences + wrap-around smoothing.
  4. Minimum-time speed profile: two-pass friction-circle solver with quadratic
     aero drag (backward braking pass, forward accel+drag pass, iterated).
  5. Derived traces: G-forces, throttle/brake, ERS battery bookkeeping, tire temp.

Model assumptions (documented in README + notebook)
  - Point-mass vehicle; corner limit |a_lat| <= MU * g (friction circle, no fade).
  - Constant peak braking / drive capability in the base profile.
  - Quadratic aero drag calibrated to an F1-class car (~800 kg, Cd*A ~ 1.6 m^2),
    which caps top speed near 320 km/h instead of letting it run away.
  - ERS is first-order bookkeeping on the base profile (harvest braking energy,
    deploy a capped boost on straights); it does not feed back into v(s).
"""
from __future__ import annotations

import json
import math
import os

import numpy as np
from scipy.ndimage import uniform_filter1d

G = 9.80665          # m/s^2
R_EARTH = 6371000.0  # m

# Vehicle / surface parameters (F1-class, conservative)
MU = 1.05            # peak lateral friction coefficient
A_BRAKE = 38.0       # m/s^2 sustained braking (~3.9 g)
A_ACC = 11.0         # m/s^2 max drive acceleration incl. ERS assist (~1.1 g)
K_DRAG = 0.001225    # (rho*Cd*A/2m); terminal ~89 m/s at A_ACC
M_CAR = 800.0        # kg incl. driver

# ERS bookkeeping (first-order, F1 MGU-K class)
ERS_CAP_J = 4.0e6    # J per-lap deploy budget
ERS_HARV_EFF = 0.85  # braking-energy harvest efficiency
ERS_P_MAX_W = 600e3  # W harvest/deploy power cap
ERS_BOOST = 2.5      # m/s^2 extra drive while deploying

# Lumped tire-thermal model (rear-axle representative)
T_AMB_C = 35.0       # Suzuka summer ambient
T_START_C = 85.0     # warm, race-ready start
ALPHA_HEAT = 0.04    # C per unit |a_lat|*v slip-power proxy
BETA_COOL = 0.10     # 1/s cooling rate


def load_track(path: str):
    with open(path) as f:
        d = json.load(f)
    pts = np.array([(p["lat"], p["lon"], p["elev"]) for p in d["points"]], float)
    return d, pts


def to_enu_m(lat_deg, lon_deg, elev_m, lat0, lon0):
    """Local ENU (east, north, up) meters around reference (lat0, lon0)."""
    x = np.radians(lon_deg - lon0) * R_EARTH * np.cos(np.radians(lat0))
    y = np.radians(lat_deg - lat0) * R_EARTH
    return x, y, elev_m


def resample_closed(x, y, z, spacing: float):
    """Resample a closed loop to uniform arc length. Returns x,y,z,s (cyclic)."""
    dx = np.roll(x, -1) - x
    dy = np.roll(y, -1) - y
    dz = np.roll(z, -1) - z
    seg = np.hypot(dx, dy)  # planar segment length (elevation is small)
    L = float(seg.sum())
    s_old = np.concatenate([[0.0], np.cumsum(seg)])[:-1]  # start of each segment
    n_new = int(round(L / spacing))
    ds = L / n_new
    s_new = np.arange(n_new) * ds
    # Queries inside the closing segment get index N from searchsorted;
    # clamp to the last raw segment (node N-1 -> node 0) instead of wrapping.
    idx = np.minimum(np.searchsorted(s_old, s_new % L, side="right"), len(x) - 1)
    t = (s_new - s_old[idx]) / np.maximum(seg[idx], 1e-9)
    xr = x[idx] + t * (np.roll(x, -1)[idx] - x[idx])
    yr = y[idx] + t * (np.roll(y, -1)[idx] - y[idx])
    zr = z[idx] + t * (np.roll(z, -1)[idx] - z[idx])
    return xr, yr, zr, s_new, ds


def signed_curvature(x, y, ds: float, smooth_pts: int = 25):
    """Signed curvature (1/m) via central differences; +ve = left turn in ENU."""
    dx = (np.roll(x, -1) - np.roll(x, 1)) / (2 * ds)
    dy = (np.roll(y, -1) - np.roll(y, 1)) / (2 * ds)
    ddx = (np.roll(dx, -1) - 2 * dx + np.roll(dx, 1)) / ds ** 2
    ddy = (np.roll(dy, -1) - 2 * dy + np.roll(dy, 1)) / ds ** 2
    k = (dx * ddy - dy * ddx) / (dx * dx + dy * dy + 1e-9) ** 1.5
    return uniform_filter1d(k, size=smooth_pts, mode="wrap")


def ideal_speed_profile(kappa, ds: float, iters: int = 24):
    """Minimum-time speed profile via two-pass friction-circle solver.

    The track is rotated so the sequential forward sweep starts at the slowest
    point (tightest corner), where drag is negligible; one full sweep then
    propagates true predecessor values across the entire loop, including the
    wrap-around seam. Backward (braking) and forward (accel+drag) passes are
    iterated to a cyclic fixed point.
    """
    n = len(kappa)
    vlat = np.sqrt(MU * G / np.maximum(np.abs(kappa), 1e-4))
    i0 = int(np.argmin(vlat))
    vl = np.roll(vlat, -i0)
    v = vl.copy()
    for _ in range(iters):
        # Backward pass: can we brake from this point to the next one's limit?
        vb = np.sqrt(np.clip(np.roll(v, -1) ** 2 + 2 * A_BRAKE * ds, 0.0, None))
        vn = np.minimum(vl, np.minimum(v, vb))
        # Forward pass: sequential acceleration against quadratic aero drag.
        for i in range(1, n):
            vp = vn[i - 1]
            c2 = vp * vp + 2.0 * ds * (A_ACC - K_DRAG * vp * vp)
            cand = math.sqrt(c2) if c2 > 0.0 else 0.0
            lim = vl[i] if vl[i] < vb[i] else vb[i]
            vn[i] = cand if cand < lim else lim
        # Wrap seam: accelerate from last point back to the first.
        vp = vn[-1]
        c2 = vp * vp + 2.0 * ds * (A_ACC - K_DRAG * vp * vp)
        cand = math.sqrt(c2) if c2 > 0.0 else 0.0
        lim = vl[0] if vl[0] < vb[0] else vb[0]
        vn[0] = min(cand, lim)
        if float(np.max(np.abs(vn - v))) < 1e-3:
            return np.roll(vn, i0)
        v = vn
    return np.roll(v, i0)


def derive_traces(s, kappa, v, ds):
    dt = ds / np.maximum(v, 0.5)
    t = np.concatenate([[0.0], np.cumsum(dt)])[:-1]  # time AT each point (len n)
    dvds = (np.roll(v, -1) - np.roll(v, 1)) / (2 * ds)
    a_long = v * dvds            # + accel, - brake
    a_lat = kappa * v ** 2       # signed lateral
    # Light smoothing: the solver switches constraints sharply at brake points,
    # which central differences turn into 1-2 point spikes; real drivers modulate.
    a_long = uniform_filter1d(a_long, size=3, mode="wrap")
    a_lat = uniform_filter1d(a_lat, size=3, mode="wrap")
    throttle = np.where(a_long > 0.5, 1.0, 0.0)
    brake = np.clip(-a_long / A_BRAKE, 0.0, 1.0)
    return t, a_lat, a_long, throttle, brake


def tire_temp(a_lat, v, dt):
    T = np.empty(len(v))
    T[0] = T_START_C
    for i in range(1, len(v)):
        heat = ALPHA_HEAT * abs(a_lat[i]) * v[i]
        cool = BETA_COOL * (T[i - 1] - T_AMB_C)
        T[i] = float(np.clip(T[i - 1] + (heat - cool) * dt[i], 40.0, 150.0))
    return T


def run(track_path: str, out_path: str, spacing: float = 2.0):
    meta, pts = load_track(track_path)
    lat0, lon0 = float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))
    x, y, z = to_enu_m(pts[:, 0], pts[:, 1], pts[:, 2], lat0, lon0)
    xr, yr, zr, s, ds = resample_closed(x, y, z, spacing)
    kappa = signed_curvature(xr, yr, ds)
    v = ideal_speed_profile(kappa, ds)
    t, a_lat, a_long, throttle, brake = derive_traces(s, kappa, v, ds)

    # ERS: harvest on braking; deploy capped boost where not corner-limited.
    n = len(v)
    dt = ds / np.maximum(v, 0.5)
    batt = np.empty(n)
    e = ERS_CAP_J
    harvested = deployed = 0.0
    for i in range(n):
        if a_long[i] < -0.5:
            p = min(ERS_HARV_EFF * (-a_long[i]) * v[i] * M_CAR, ERS_P_MAX_W)
            e = min(e + p * dt[i], ERS_CAP_J)
            harvested += p * dt[i]
        corner_limited = abs(kappa[i]) * v[i] ** 2 > 0.5 * MU * G
        if throttle[i] == 1.0 and not corner_limited and e > 0:
            p = min(M_CAR * ERS_BOOST * max(v[i], 20.0), ERS_P_MAX_W)
            de = min(p * dt[i], e)
            e -= de
            deployed += de
        batt[i] = e

    T_tire = tire_temp(a_lat, v, dt)
    g_total = np.hypot(a_lat / G, a_long / G)
    dt_all = ds / np.maximum(v, 0.5)
    lap_time = float(dt_all.sum())

    out = {
        "meta": {
            "track": meta["name"],
            "length_m": float(s[-1] + ds),
            "n_points": n,
            "ds_m": float(ds),
            "params": {"mu": MU, "a_brake": A_BRAKE, "a_acc": A_ACC,
                       "k_drag": K_DRAG, "m_car": M_CAR},
            "lap_time_s": round(lap_time, 3),
            "top_speed_kmh": round(float(v.max()) * 3.6, 1),
            "max_g_lat": round(float(np.abs(a_lat).max() / G), 2),
            "max_g_long": round(float(np.abs(a_long).max() / G), 2),
            "ers_harvested_MJ": round(harvested / 1e6, 3),
            "ers_deployed_MJ": round(deployed / 1e6, 3),
            "tire_temp_peak_C": round(float(T_tire.max()), 1),
        },
        "s": np.round(s, 2).tolist(),
        "x": np.round(xr, 2).tolist(),
        "y": np.round(yr, 2).tolist(),
        "z": np.round(zr, 2).tolist(),
        "t": np.round(t, 4).tolist(),
        "v_ms": np.round(v, 3).tolist(),
        "a_lat": np.round(a_lat, 3).tolist(),
        "a_long": np.round(a_long, 3).tolist(),
        "g_total": np.round(g_total, 3).tolist(),
        "throttle": throttle.tolist(),
        "brake": np.round(brake, 3).tolist(),
        "ers_battery_J": np.round(batt, 0).tolist(),
        "tire_temp_C": np.round(T_tire, 2).tolist(),
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f)

    # Corner entry-speed sanity table.
    print(f"lap_time={lap_time:.2f}s  top_speed={v.max()*3.6:.0f} km/h  "
          f"max_g_lat={np.abs(a_lat).max()/G:.2f}g  max_g_long={np.abs(a_long).max()/G:.2f}g")
    print(f"ERS harvested={harvested/1e6:.2f} MJ deployed={deployed/1e6:.2f} MJ "
          f"tire_peak={T_tire.max():.0f} C")
    for c in meta.get("corners", []):
        i = int(np.argmin(np.abs(s - (c["s_m"] % (s[-1] + ds)))))
        print(f"  {c['name']:<14} s={c['s_m']:>7.0f} m  entry_v={v[i]*3.6:6.1f} km/h")
    return out


if __name__ == "__main__":
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run(os.path.join(here, "data", "track_suzuka.json"),
        os.path.join(here, "outputs", "ideal_line.json"))
