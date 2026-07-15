# -*- coding: utf-8 -*-
"""
Created on Fri May 22 14:14:39 2026

@author: Zemay
"""

"""
RTL & Waypoint Landing Accuracy — Mission Comparison
=====================================================

Compares two PX4 mission profiles on the same drone setup:

  Mission 1 — Takeoff → hover 10s → land (one touchdown per flight, vs home).
  Mission 2 — Multi-waypoint mission with a LAND command at each waypoint
              (multiple touchdowns per flight, each measured vs its own
              intended waypoint).

Outputs:
  • side_by_side.png  — both missions on one plot (different colors/markers)
  • drift_over_time.png — every touchdown colored by chronological order
                          (early = blue, late = red), to spot drift.

Usage
-----
  python compare_missions.py M1_FOLDER M2_FOLDER [--out OUTDIR]

Example (Windows):
  python compare_missions.py "C:\\path\\to\\Take off_hover10s_land" "C:\\path\\to\\mission2_logs" --out C:\\Users\\Zemay\\Downloads

Dependencies:  pip install pyulog matplotlib numpy
"""
import argparse
import glob
import math
import os
import sys
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from pyulog import ULog

DATASHEET_RADIUS_M = 1.5
MIN_PEAK_ABOVE_REST_M = 1.0   # must have flown >=1 m above resting altitude in last 5 s

# ---------------------------------------------------------------------------

def latlon_to_local(lat, lon, ref_lat, ref_lon):
    """Equirectangular projection — accurate for sub-km distances."""
    R = 6371000.0
    dlat = math.radians(lat - ref_lat)
    dlon = math.radians(lon - ref_lon)
    return dlat * R, dlon * R * math.cos(math.radians(ref_lat))


def log_start_time(ulog):
    """Best-effort wall-clock start time (used for chronological ordering)."""
    try:
        return ulog.start_timestamp
    except Exception:
        return 0


def parse_mission1(path):
    """Return list with one dict: the single RTL touchdown of a takeoff/hover/land flight."""
    ulog = ULog(path)
    lp = ulog.get_dataset('vehicle_local_position').data
    ld = ulog.get_dataset('vehicle_land_detected').data
    home = ulog.get_dataset('home_position').data
    alt = -lp['z']

    if alt.max() < MIN_PEAK_ABOVE_REST_M:
        return []   # never actually flew

    # Last 0->1 transition is the final touchdown
    landed = ld['landed']; ts = ld['timestamp']
    final_td_ts = None
    for i in range(1, len(landed)):
        if landed[i] == 1 and landed[i-1] == 0:
            final_td_ts = ts[i]
    if final_td_ts is None:
        return []

    home_x, home_y = float(home['x'][-1]), float(home['y'][-1])
    k = int(np.argmin(np.abs(lp['timestamp'] - final_td_ts)))
    actual_x, actual_y = float(lp['x'][k]), float(lp['y'][k])

    return [{
        'east':  actual_y - home_y,
        'north': actual_x - home_x,
        'kind':  'RTL→home',
        'log':   os.path.basename(path),
        'start_ts': log_start_time(ulog),
    }]


def parse_mission2(path):
    """Return a dict per touchdown, each measured vs its intended LAND waypoint."""
    ulog = ULog(path)
    lp = ulog.get_dataset('vehicle_local_position').data
    ld = ulog.get_dataset('vehicle_land_detected').data
    nm = ulog.get_dataset('navigator_mission_item').data
    home = ulog.get_dataset('home_position').data
    alt = -lp['z']

    home_lat = float(home['lat'][-1]); home_lon = float(home['lon'][-1])
    home_x   = float(home['x'][-1]);   home_y   = float(home['y'][-1])

    # LAND mission items (nav_cmd == 21)
    land_cmds = [(nm['timestamp'][i], float(nm['latitude'][i]), float(nm['longitude'][i]))
                 for i in range(len(nm['timestamp'])) if nm['nav_cmd'][i] == 21]

    # 0->1 transitions in 'landed', filtered: drone must have been
    # ≥1 m above its resting altitude in the preceding 5 s.
    valid_touchdowns = []
    for i in range(1, len(ld['landed'])):
        if ld['landed'][i] == 1 and ld['landed'][i-1] == 0:
            ts_check = ld['timestamp'][i]
            k = int(np.argmin(np.abs(lp['timestamp'] - ts_check)))
            rest_alt = alt[k]
            window = (lp['timestamp'] > ts_check - 5_000_000) & (lp['timestamp'] < ts_check)
            if np.any(window) and (alt[window] - rest_alt).max() > MIN_PEAK_ABOVE_REST_M:
                valid_touchdowns.append(ts_check)

    results = []
    log_start = log_start_time(ulog)
    MAX_GAP_S = 60.0   # touchdown must follow its LAND command within 60 s
    for td_ts in valid_touchdowns:
        candidates = [lc for lc in land_cmds if lc[0] <= td_ts]
        if candidates:
            cmd_ts, tlat, tlon = max(candidates, key=lambda c: c[0])
            gap_s = (td_ts - cmd_ts) / 1e6
            if gap_s > MAX_GAP_S:
                # touchdown too far from its LAND cmd -> aborted/partial mission, skip
                continue
            kind = "WP-LAND"
        else:
            tlat, tlon = home_lat, home_lon
            kind = "RTL→home"

        k = int(np.argmin(np.abs(lp['timestamp'] - td_ts)))
        actual_x, actual_y = float(lp['x'][k]), float(lp['y'][k])

        tgt_n, tgt_e = latlon_to_local(tlat, tlon, home_lat, home_lon)
        tgt_x = tgt_n + home_x
        tgt_y = tgt_e + home_y

        results.append({
            'east':  actual_y - tgt_y,
            'north': actual_x - tgt_x,
            'kind':  kind,
            'log':   os.path.basename(path),
            'start_ts': log_start,
        })
    return results


# ---------------------------------------------------------------------------

def collect(folder, parser_fn, label):
    paths = sorted(glob.glob(os.path.join(folder, "*.ulg")))
    if not paths:
        sys.exit(f"No .ulg files found in {folder!r}")
    all_points = []
    print(f"\n=== {label} : {len(paths)} log(s) ===")
    for p in paths:
        try:
            pts = parser_fn(p)
        except Exception as e:
            print(f"  [skip] {os.path.basename(p)}: {e}")
            continue
        if not pts:
            print(f"  [skip] {os.path.basename(p)}: no valid touchdown")
            continue
        for pt in pts:
            print(f"  {pt['log']:45s}  {pt['kind']:10s}  E={pt['east']:+.3f}  N={pt['north']:+.3f}")
        all_points.extend(pts)
    return all_points


def summarize(points, label):
    if not points:
        print(f"  {label}: no points")
        return
    radii = np.hypot([p['east'] for p in points], [p['north'] for p in points])
    inside = (radii <= DATASHEET_RADIUS_M).sum()
    print(f"  {label:12s} n={len(points):3d}  mean={radii.mean():.3f}m  "
          f"max={radii.max():.3f}m  inside_{DATASHEET_RADIUS_M}m={inside}/{len(points)}")


def figure_side_by_side(m1, m2, out_path):
    fig, ax = plt.subplots(figsize=(8, 8))

    ring = Circle((0, 0), DATASHEET_RADIUS_M, fill=False, linestyle="--",
                  linewidth=2, edgecolor="green",
                  label=f"Datasheet Accuracy ({DATASHEET_RADIUS_M} m)")
    ax.add_patch(ring)

    if m1:
        ax.scatter([p['east'] for p in m1], [p['north'] for p in m1],
                   s=110, marker='o', c='royalblue', edgecolors='black',
                   linewidths=0.6, alpha=0.6,
                   label=f"Mission 1 — hover & land  (n={len(m1)})", zorder=3)
    if m2:
        ax.scatter([p['east'] for p in m2], [p['north'] for p in m2],
                   s=110, marker='^', c='orange', edgecolors='black',
                   linewidths=0.6, alpha=0.7,
                   label=f"Mission 2 — waypoint landings  (n={len(m2)})", zorder=3)

    ax.scatter([0], [0], marker="X", s=240, c="red", linewidths=2,
               label="Intended Target", zorder=4)

    ax.axhline(0, color='black', linewidth=0.8)
    ax.axvline(0, color='black', linewidth=0.8)
    lim = 2.0
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_xlabel("East/West error (m)")
    ax.set_ylabel("North/South error (m)")
    ax.set_title("Touchdown accuracy — Mission 1 vs Mission 2 (Holybro M10)")
    ax.legend(loc="upper left", framealpha=0.95)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  saved {out_path}")
    plt.close(fig)


def figure_drift_over_time(points, out_path):
    """All touchdowns combined, colored by chronological order."""
    if not points:
        return
    # Sort by start_ts then by appearance order within the log
    pts_sorted = sorted(points, key=lambda p: p['start_ts'])
    n = len(pts_sorted)
    colors = plt.cm.viridis(np.linspace(0, 1, n))

    fig, ax = plt.subplots(figsize=(8, 8))
    ring = Circle((0, 0), DATASHEET_RADIUS_M, fill=False, linestyle="--",
                  linewidth=2, edgecolor="green",
                  label=f"Datasheet Accuracy ({DATASHEET_RADIUS_M} m)")
    ax.add_patch(ring)

    east  = [p['east']  for p in pts_sorted]
    north = [p['north'] for p in pts_sorted]
    sc = ax.scatter(east, north, c=range(n), cmap='viridis',
                    s=110, edgecolors='black', linewidths=0.5, zorder=3)
    ax.scatter([0], [0], marker="X", s=240, c="red", linewidths=2,
               label="Intended Target", zorder=4)

    cbar = fig.colorbar(sc, ax=ax, shrink=0.8)
    cbar.set_label("Touchdown order (earliest → latest)")

    ax.axhline(0, color='black', linewidth=0.8)
    ax.axvline(0, color='black', linewidth=0.8)
    lim = 2.0
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_xlabel("East/West error (m)")
    ax.set_ylabel("North/South error (m)")
    ax.set_title("Drift over time — all touchdowns in chronological order")
    ax.legend(loc="upper left", framealpha=0.95)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  saved {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mission1_folder", help="Folder with Mission 1 (hover+land) .ulg files")
    ap.add_argument("mission2_folder", help="Folder with Mission 2 (waypoints) .ulg files")
    ap.add_argument("--out", default=".", help="Output directory for the two PNGs")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    m1 = collect(args.mission1_folder, parse_mission1, "Mission 1")
    m2 = collect(args.mission2_folder, parse_mission2, "Mission 2")

    print("\n=== Summary ===")
    summarize(m1, "Mission 1")
    summarize(m2, "Mission 2")
    summarize(m1 + m2, "Combined")

    print("\n=== Figures ===")
    figure_side_by_side(m1, m2, os.path.join(args.out, "side_by_side.png"))
    figure_drift_over_time(m1 + m2, os.path.join(args.out, "drift_over_time.png"))


if __name__ == "__main__":
    main()