"""Smoke-test figure for the ideal-line physics run (4 panels)."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    d = json.load(open(os.path.join(HERE, "outputs", "ideal_line.json")))
    trk = json.load(open(os.path.join(HERE, "data", "track_suzuka.json")))
    s = np.array(d["s"]); x = np.array(d["x"]); y = np.array(d["y"])
    v = np.array(d["v_ms"]); t = np.array(d["t"])
    a_lat = np.array(d["a_lat"]); a_long = np.array(d["a_long"])
    g_tot = np.array(d["g_total"]); T = np.array(d["tire_temp_C"])
    batt = np.array(d["ers_battery_J"]) / 1e6
    corners = trk.get("corners", [])

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    ax = axes[0][0]
    segs = [np.column_stack([x[i:i + 2], y[i:i + 2]]) for i in range(len(x) - 1)]
    lc = LineCollection(segs, cmap="viridis", norm=plt.Normalize(v.min(), v.max()), lw=2.5)
    lc.set_array(v[:-1])
    ax.add_collection(lc)
    L = s[-1] + d["meta"]["ds_m"]
    for c in corners:
        i = int(np.argmin(np.abs(s - (c["s_m"] % L))))
        ax.annotate(c["name"].split("/")[-1].strip(), (x[i], y[i]), fontsize=7,
                    ha="center", va="bottom")
    ax.set_aspect("equal"); ax.autoscale()
    sm = fig.colorbar(lc, ax=ax); sm.set_label("speed m/s")
    ax.set_title(f"{trk['name']} — ideal line (v colored)")

    ax = axes[0][1]
    ax.plot(s / 1000, v * 3.6, lw=1.5)
    for c in corners:
        sc = c["s_m"] % L
        i = int(np.argmin(np.abs(s - sc)))
        ax.axvline(sc / 1000, color="gray", lw=0.4, alpha=0.5)
        ax.annotate(c["name"].split("/")[-1].strip(), (sc / 1000, v[i] * 3.6 + 8),
                    fontsize=7, rotation=90, va="bottom")
    ax.set_xlabel("track position km"); ax.set_ylabel("speed km/h")
    ax.set_title(f"Speed profile — lap {d['meta']['lap_time_s']} s, top {d['meta']['top_speed_kmh']} km/h")

    ax = axes[1][0]
    ax.plot(t, a_lat / 9.80665, label="lat G", lw=1)
    ax.plot(t, a_long / 9.80665, label="long G", lw=1)
    ax.plot(t, g_tot, label="total G", lw=1.2, color="k")
    ax.set_xlabel("t s"); ax.set_ylabel("g"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title("G-force traces (ideal lap)")

    ax = axes[1][1]
    ax.plot(t, T, color="crimson", lw=1.5, label="tire temp C")
    ax2 = ax.twinx()
    ax2.plot(t, batt, color="steelblue", lw=1.2, ls="--", label="ERS battery MJ")
    ax.set_xlabel("t s"); ax.set_ylabel("C", color="crimson")
    ax2.set_ylabel("MJ", color="steelblue"); ax.grid(alpha=0.3)
    ax.set_title(f"Tire thermal + ERS (peak {d['meta']['tire_temp_peak_C']} C)")

    fig.tight_layout()
    out = os.path.join(HERE, "outputs", "figures", "ideal_line_smoke.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=110)
    print("saved", out)


if __name__ == "__main__":
    main()
