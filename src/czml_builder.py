"""Build build/czml/telemetry.czml from the ideal line + driver telemetry.

Two moving packages at 5 Hz. NOTE on encoding: this repo's Cesium 1.115 bundle
parses sampled positions via a flat `cartesian` array of interleaved groups
[seconds_offset_from_epoch, x, y, z] (ECEF meters) and scalar properties via
interleaved [offset, value] under the `number` key — do NOT use the spec's
epoch/interval/positions form: its numeric `interval` crashes this build's
time-interval parser (`iso8601.split is not a function`).
  - "driver"       : synthetic 100 Hz log downsampled, real GPS jitter included
  - "ideal_ghost"  : physics ideal line running in parallel for comparison
Clock spans all three laps; the frontend drives scrubbing / playback speed.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R_EARTH = 6371000.0
EPOCH = "2026-09-12T00:00:00Z"
HZ = 5.0

# WGS84 ellipsoid (Cesium's default)
_WGS_A = 6378137.0
_WGS_F = 1.0 / 298.257223563
_WGS_E2 = _WGS_F * (2.0 - _WGS_F)


def llh_to_ecef(lat_deg, lon_deg, h):
    """Batch WGS84 geodetic -> ECEF meters."""
    la = np.radians(np.asarray(lat_deg, dtype=float))
    lo = np.radians(np.asarray(lon_deg, dtype=float))
    h = np.asarray(h, dtype=float)
    N = _WGS_A / np.sqrt(1.0 - _WGS_E2 * np.sin(la) ** 2)
    x = (N + h) * np.cos(la) * np.cos(lo)
    y = (N + h) * np.cos(la) * np.sin(lo)
    z = (N * (1.0 - _WGS_E2) + h) * np.sin(la)
    return x, y, z


def interleaved(offsets_s, cols):
    """[t0, c0..., t1, c1...] flat list for this Cesium build's sampler."""
    out = []
    for i, off in enumerate(offsets_s):
        out.append(round(float(off), 6))
        for c in cols[i]:
            out.append(round(float(c), 3))
    return out


def main():
    ideal = json.load(open(os.path.join(HERE, "outputs", "ideal_line.json")))
    trk = json.load(open(os.path.join(HERE, "data", "track_suzuka.json")))
    pts = np.array([(p["lat"], p["lon"]) for p in trk["points"]])
    lat0, lon0 = float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))

    # --- driver from the 100 Hz CSV, downsampled to 5 Hz -------------------
    df = pd.read_csv(os.path.join(HERE, "outputs", "telemetry_driver.csv"))
    d = df.iloc[::20].reset_index(drop=True)  # 100 Hz -> 5 Hz
    t_end = float(d["t_s"].iloc[-1]) + 0.2

    t_off = (d["t_s"] - float(d["t_s"].iloc[0])).to_numpy()
    dx, dy, dz = llh_to_ecef(d["lat"], d["lon"], d["alt_m"])

    driver = {
        "id": "driver",
        "name": "Driver (100 Hz log @ 5 Hz)",
        "availability": f"{EPOCH}/{_iso(t_end)}",
        "position": {"epoch": EPOCH,
                     "cartesian": interleaved(t_off, list(zip(dx, dy, dz)))},
        "speed": {"epoch": EPOCH,
                  "number": interleaved(t_off, [[v] for v in d["speed_ms"]])},
        "path": {"show": True, "material": {"line": {"color": {
            "id": "driver_path_color", "rgba": [80, 160, 255, 190]}}},
                 "width": 3, "leadTime": 0.0, "trailTime": 4.0},
        "point": {"pixelSize": 10, "color": {"rgba": [80, 160, 255, 255]},
                  "outlineColor": {"rgba": [255, 255, 255, 255]}, "outlineWidth": 2},
        "label": {"text": "DRIVER", "font": "12px sans-serif", "showBackground": True,
                  "backgroundColor": {"rgba": [10, 20, 40, 160]},
                  "pixelOffset": [0, -18]},
    }

    # --- ideal ghost: interpolate the ideal line at 5 Hz --------------------
    s = np.array(ideal["s"]); x = np.array(ideal["x"]); y = np.array(ideal["y"])
    z = np.array(ideal["z"]); v_id = np.array(ideal["v_ms"])
    ds = float(ideal["meta"]["ds_m"])
    L = s[-1] + ds
    dt_grid = ds / np.maximum(v_id, 0.5)
    t_cum = np.concatenate([[0.0], np.cumsum(dt_grid)])[:-1]
    lap_time = float(dt_grid.sum())
    n_samp = int(round(lap_time * HZ))
    t_q = np.arange(n_samp) / HZ
    idx = np.searchsorted(t_cum, t_q % lap_time, side="right") % len(s)

    def enu2ll(xx, yy):
        lon = lon0 + np.degrees(xx / (R_EARTH * np.cos(np.radians(lat0))))
        lat = lat0 + np.degrees(yy / R_EARTH)
        return lat, lon

    glat, glon = enu2ll(x[idx], y[idx])
    # Ghost repeats the ideal lap (it is faster than the driver and finishes first).
    g_off = np.arange(3 * n_samp) / HZ          # continuous across all 3 laps
    gx, gy, gz = llh_to_ecef(np.tile(glat, 3), np.tile(glon, 3), np.tile(z[idx], 3))
    ghost_spd = list(v_id[idx]) * 3
    ghost_end = _iso(3.0 * lap_time + 0.2)
    ghost = {
        "id": "ideal_ghost",
        "name": "Ideal line (physics model)",
        "availability": f"{EPOCH}/{ghost_end}",
        "position": {"epoch": EPOCH,
                     "cartesian": interleaved(g_off, list(zip(gx, gy, gz)))},
        "speed": {"epoch": EPOCH,
                  "number": interleaved(g_off, [[v] for v in ghost_spd])},
        "point": {"pixelSize": 8, "color": {"rgba": [255, 210, 60, 255]},
                  "outlineColor": {"rgba": [255, 255, 255, 255]}, "outlineWidth": 2},
        "label": {"text": "IDEAL", "font": "11px sans-serif", "showBackground": True,
                  "backgroundColor": {"rgba": [40, 30, 5, 160]},
                  "pixelOffset": [0, -16]},
    }

    doc = [
        {"id": "document", "version": "1.0",
         "clock": {"interval": f"{EPOCH}/{_iso(t_end)}",
                   "currentTime": EPOCH, "multiplier": 1,
                   "range": "LOOP_STOP"}},
        driver, ghost,
    ]
    out = os.path.join(HERE, "build", "czml", "telemetry.czml")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(doc, f)
    print(f"saved {out} ({os.path.getsize(out)/1024:.0f} KB) | driver samples={len(d)} ghost={n_samp}")


def _iso(t_s: float) -> str:
    h = int(t_s // 3600); m = int((t_s % 3600) // 60); sec = t_s % 60
    return f"2026-09-12T{h:02d}:{m:02d}:{sec:08.5f}Z"


if __name__ == "__main__":
    main()
