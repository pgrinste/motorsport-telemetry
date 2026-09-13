"""Build build/czml/telemetry.czml from the ideal line + driver telemetry.

Two moving packages at 5 Hz (epoch/interval compact encoding):
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


def main():
    ideal = json.load(open(os.path.join(HERE, "outputs", "ideal_line.json")))
    trk = json.load(open(os.path.join(HERE, "data", "track_suzuka.json")))
    pts = np.array([(p["lat"], p["lon"]) for p in trk["points"]])
    lat0, lon0 = float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))

    # --- driver from the 100 Hz CSV, downsampled to 5 Hz -------------------
    df = pd.read_csv(os.path.join(HERE, "outputs", "telemetry_driver.csv"))
    d = df.iloc[::20].reset_index(drop=True)  # 100 Hz -> 5 Hz
    t_end = float(d["t_s"].iloc[-1]) + 0.2

    def positions(sub):
        return [f"{la:.7f} {lo:.7f} {al:.2f}" for la, lo, al in
                zip(sub["lat"], sub["lon"], sub["alt_m"])]

    driver = {
        "id": "driver",
        "name": "Driver (100 Hz log @ 5 Hz)",
        "availability": f"{EPOCH}/{_iso(t_end)}",
        "position": {"epoch": EPOCH, "interval": round(1.0 / HZ, 3),
                     "interpolatedRoute": True, "positions": positions(d)},
        "speed": {"epoch": EPOCH, "interval": round(1.0 / HZ, 3),
                  "values": [round(float(x), 2) for x in d["speed_ms"]]},
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
    ghost_pos = [f"{a:.7f} {b:.7f} {c:.2f}" for a, b, c in zip(glat, glon, z[idx])] * 3
    ghost_spd = [round(float(v), 2) for v in v_id[idx]] * 3
    ghost_end = _iso(3.0 * lap_time + 0.2)
    ghost = {
        "id": "ideal_ghost",
        "name": "Ideal line (physics model)",
        "availability": f"{EPOCH}/{ghost_end}",
        "position": {"epoch": EPOCH, "interval": round(1.0 / HZ, 3),
                     "interpolatedRoute": True,
                     "positions": ghost_pos},
        "speed": {"epoch": EPOCH, "interval": round(1.0 / HZ, 3),
                  "values": ghost_spd},
        "point": {"pixelSize": 8, "color": {"rgba": [255, 210, 60, 255]},
                  "outlineColor": {"rgba": [255, 255, 255, 255]}, "outlineWidth": 2},
        "label": {"text": "IDEAL", "font": "11px sans-serif", "showBackground": True,
                  "backgroundColor": {"rgba": [40, 30, 5, 160]},
                  "pixelOffset": [0, -16]},
    }

    doc = [
        {"id": "document", "version": "1.0",
         "clock": {"interval": f"{EPOCH}/{_iso(t_end)}",
                   "currentTime": {"date": EPOCH}, "multiplier": 1,
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
