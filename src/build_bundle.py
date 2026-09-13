"""Assemble build/bundle.json — everything the Cesium frontend needs.

Static bundle keeps the site deployable anywhere (GitHub Pages); the FastAPI
server only adds live WebSocket streaming on top.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ideal = json.load(open(os.path.join(HERE, "outputs", "ideal_line.json")))
    delta = json.load(open(os.path.join(HERE, "outputs", "delta.json")))
    mc = json.load(open(os.path.join(HERE, "data", "mc_results.json")))
    trk = json.load(open(os.path.join(HERE, "data", "track_suzuka.json")))
    df = pd.read_csv(os.path.join(HERE, "outputs", "telemetry_driver.csv"))

    # Track geometry (downsampled 2x for the ribbon).
    pts = trk["points"][::2]
    track = {"lat": [round(p["lat"], 7) for p in pts],
             "lon": [round(p["lon"], 7) for p in pts],
             "elev": [p["elev"] for p in pts]}

    # Ideal line traces at ~8 m resolution.
    step = 4
    ideal_ds = {k: np.round(np.array(ideal[k])[::step], 3).tolist()
                for k in ("s", "v_ms", "t")}

    # Driver speed vs track position per lap (for the profile chart).
    s_grid = np.array(delta["s"])
    L = float(s_grid[-1]) + float(ideal["meta"]["ds_m"])
    driver_v_s = {}
    for lap in (1, 2, 3):
        sub = df[df["lap"] == lap]
        # cumulative distance along the driven path
        dlat = np.radians(np.diff(sub["lat"].values)) * 6371000
        dlon = np.radians(np.diff(sub["lon"].values)) * 6371000 * \
            np.cos(np.radians(sub["lat"].values[:-1]))
        dist = np.concatenate([[0.0], np.cumsum(np.hypot(dlat, dlon))])
        v_at_s = np.interp(s_grid % L, dist % L, sub["speed_ms"].values)
        driver_v_s[str(lap)] = np.round(v_at_s[::8], 2).tolist()

    # Gap series (cumulative delta vs ideal) at 5 Hz for the HUD.
    # cum_delta_ms is indexed by track position s; map driver clock time ->
    # cumulative driven distance (~s) -> gap value.
    s_grid_full = np.array(delta["s"])
    L_full = float(s_grid_full[-1]) + float(ideal["meta"]["ds_m"])
    gap_series = {}
    for lap in ("1", "2", "3"):
        sub5 = df[df["lap"] == int(lap)].iloc[::20]  # 100 Hz -> 5 Hz
        t_local = (sub5["t_s"].values - sub5["t_s"].values[0])
        dlat = np.radians(np.diff(sub5["lat"].values)) * 6371000
        dlon = np.radians(np.diff(sub5["lon"].values)) * 6371000 * \
            np.cos(np.radians(sub5["lat"].values[:-1]))
        dist5 = np.concatenate([[0.0], np.cumsum(np.hypot(dlat, dlon))])
        cum_ms = np.array(delta["laps"][lap]["cum_delta_ms"])
        gap_series[lap] = {"t": np.round(t_local, 2).tolist(),
                           "gap_ms": np.interp(dist5 % L_full, s_grid_full,
                                               cum_ms).round(1).tolist(),
                           "speed_kmh": (sub5["speed_ms"].values * 3.6).round(1).tolist(),
                           "dist_m": dist5.round(1).tolist()}

    corners_geo = []
    for c in trk.get("corners", []):
        p = trk["points"][c["index"]]
        corners_geo.append({"name": c["name"], "s_m": c["s_m"],
                            "lat": p["lat"], "lon": p["lon"], "elev": p["elev"]})

    bundle = {
        "meta": ideal["meta"],
        "track": track,
        "corners": corners_geo,
        "ideal": ideal_ds,
        "delta": {"s": np.round(np.array(delta["s"])[::8], 1).tolist(),
                  "laps": {lap: {
                      "lap_time_s": delta["laps"][lap]["lap_time_s"],
                      "cum_delta_ms": np.round(
                          np.array(delta["laps"][lap]["cum_delta_ms"])[::8], 1).tolist()}
                  for lap in ("1", "2", "3")}},
        "driver_v_s": driver_v_s,
        "gap_series": gap_series,
        "mc": {"s": mc["s"], "v_p10": mc["v_p10_ms"], "v_p50": mc["v_p50_ms"],
               "v_p90": mc["v_p90_ms"], "lap_time_s": mc["lap_time_s"]},
    }
    out = os.path.join(HERE, "build", "bundle.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(bundle, f)
    print(f"saved {out} ({os.path.getsize(out)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
