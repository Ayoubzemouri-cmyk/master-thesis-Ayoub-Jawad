"""
Mission 2 - Per-waypoint landing consistency  (thesis figure, clean layout)

Layout:
  * scatter plot on top
  * legend in a single row beneath
  * statistics table at the bottom
  * thin axes, restrained colour-blind-safe palette, 2 sigma ellipses
"""
import argparse
import glob
import math
import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
from pyulog import ULog

mpl.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["DejaVu Serif", "Times New Roman", "Times", "STIXGeneral"],
    "mathtext.fontset":  "stix",
    "axes.labelsize":    10,
    "axes.titlesize":    11,
    "axes.linewidth":    0.6,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "xtick.direction":   "out",
    "ytick.direction":   "out",
    "xtick.major.size":  3,
    "ytick.major.size":  3,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.fontsize":   9,
    "legend.frameon":    False,
    "savefig.bbox":      "tight",
    "savefig.dpi":       300,
    "pdf.fonttype":      42,
    "ps.fonttype":       42,
})

WP_COLORS = {1: "#0072B2", 2: "#D55E00", 3: "#009E73", 4: "#CC79A7",
             5: "#56B4E9", 6: "#E69F00"}

MIN_PEAK_ABOVE_REST_M = 1.0
MAX_GAP_S = 60.0


def latlon_to_local(lat, lon, ref_lat, ref_lon):
    R = 6371000.0
    dlat = math.radians(lat - ref_lat)
    dlon = math.radians(lon - ref_lon)
    return dlat * R, dlon * R * math.cos(math.radians(ref_lat))


def parse_log(path):
    ulog = ULog(path)
    lp = ulog.get_dataset("vehicle_local_position").data
    ld = ulog.get_dataset("vehicle_land_detected").data
    nm = ulog.get_dataset("navigator_mission_item").data
    home = ulog.get_dataset("home_position").data
    alt = -lp["z"]

    home_lat = float(home["lat"][-1]); home_lon = float(home["lon"][-1])
    home_x   = float(home["x"][-1]);   home_y   = float(home["y"][-1])

    land_cmds, counter = [], 0
    for i in range(len(nm["timestamp"])):
        if nm["nav_cmd"][i] == 21:
            counter += 1
            land_cmds.append((nm["timestamp"][i],
                              float(nm["latitude"][i]),
                              float(nm["longitude"][i]),
                              counter))

    valid = []
    for i in range(1, len(ld["landed"])):
        if ld["landed"][i] == 1 and ld["landed"][i-1] == 0:
            ts = ld["timestamp"][i]
            k = int(np.argmin(np.abs(lp["timestamp"] - ts)))
            rest = alt[k]
            win = (lp["timestamp"] > ts - 5_000_000) & (lp["timestamp"] < ts)
            if np.any(win) and (alt[win] - rest).max() > MIN_PEAK_ABOVE_REST_M:
                valid.append(ts)

    pts = []
    for td_ts in valid:
        cands = [lc for lc in land_cmds if lc[0] <= td_ts]
        if not cands:
            continue
        cmd_ts, tlat, tlon, wp_idx = max(cands, key=lambda c: c[0])
        if (td_ts - cmd_ts) / 1e6 > MAX_GAP_S:
            continue
        k = int(np.argmin(np.abs(lp["timestamp"] - td_ts)))
        ax_, ay_ = float(lp["x"][k]), float(lp["y"][k])
        tn, te = latlon_to_local(tlat, tlon, home_lat, home_lon)
        tx, ty = tn + home_x, te + home_y
        pts.append({"east": ay_ - ty, "north": ax_ - tx, "wp_idx": wp_idx})
    return pts


def confidence_ellipse(x, y, ax, n_std=2.0, **kwargs):
    if len(x) < 3:
        return None
    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    rx = np.sqrt(1 + pearson)
    ry = np.sqrt(1 - pearson)
    ell = Ellipse((0, 0), width=2 * rx, height=2 * ry, **kwargs)
    sx = np.sqrt(cov[0, 0]) * n_std
    sy = np.sqrt(cov[1, 1]) * n_std
    mx, my = np.mean(x), np.mean(y)
    transf = (mpl.transforms.Affine2D()
              .rotate_deg(45).scale(sx, sy).translate(mx, my))
    ell.set_transform(transf + ax.transData)
    return ax.add_patch(ell)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log_dir")
    ap.add_argument("--save")
    ap.add_argument("--title",
                    default="Touchdown error at sequential LAND waypoints, Mission 2")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.log_dir, "*.ulg")))
    if not paths:
        sys.exit(f"No .ulg files in {args.log_dir!r}")

    by_log = {}
    for p in paths:
        try:
            pts = parse_log(p)
        except Exception as e:
            print(f"[skip] {os.path.basename(p)}: {e}")
            continue
        if pts:
            by_log[os.path.basename(p)] = pts
    if not by_log:
        sys.exit("No valid touchdowns.")

    flight1 = sorted(by_log.keys())[0]
    by_wp = defaultdict(list)
    for name, pts in by_log.items():
        for pt in pts:
            by_wp[pt["wp_idx"]].append((pt["east"], pt["north"],
                                        name == flight1))

    all_e = [p[0] for v in by_wp.values() for p in v]
    all_n = [p[1] for v in by_wp.values() for p in v]
    span  = max(abs(min(all_e)), abs(max(all_e)),
                abs(min(all_n)), abs(max(all_n)))
    lim   = max(0.30, span * 1.30)

    fig = plt.figure(figsize=(5.6, 8.4))
    gs  = fig.add_gridspec(3, 1, height_ratios=[4.6, 0.55, 1.4], hspace=0.55)
    ax    = fig.add_subplot(gs[0])
    axLeg = fig.add_subplot(gs[1]); axLeg.axis("off")
    axT   = fig.add_subplot(gs[2]); axT.axis("off")

    ax.axhline(0, color="#888888", linewidth=0.5, zorder=2)
    ax.axvline(0, color="#888888", linewidth=0.5, zorder=2)

    for wp_idx in sorted(by_wp.keys()):
        col = WP_COLORS.get(wp_idx, "#666666")
        pts = by_wp[wp_idx]
        ex = np.array([p[0] for p in pts])
        ny = np.array([p[1] for p in pts])
        f1 = np.array([p[2] for p in pts])

        confidence_ellipse(ex, ny, ax, n_std=2.0,
                           facecolor=col, alpha=0.08,
                           edgecolor=col, linewidth=0.7,
                           linestyle=(0, (4, 2)), zorder=2)

        ax.plot(ex.mean(), ny.mean(), marker="+", markersize=9,
                markeredgewidth=1.2, color=col, zorder=4)

        ax.scatter(ex[~f1], ny[~f1], s=24, marker="o",
                   facecolor=col, edgecolor="white", linewidths=0.6,
                   zorder=3, label=f"WP{wp_idx}")
        ax.scatter(ex[f1], ny[f1], s=70, marker="D",
                   facecolor="white", edgecolor=col, linewidths=1.2,
                   zorder=5)

    ax.scatter([0], [0], marker="x", s=60, c="black",
               linewidths=1.3, zorder=6)

    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel(r"East-west error,  $\Delta E$  (m)")
    ax.set_ylabel(r"North-south error,  $\Delta N$  (m)")
    ax.set_title(args.title, pad=10)
    ax.grid(True, linestyle=":", linewidth=0.5, color="#CCCCCC", zorder=0)

    handles = [Line2D([0], [0], marker="o", linestyle="",
                      markerfacecolor=WP_COLORS[w], markeredgecolor="white",
                      markersize=6, label=f"WP{w}")
               for w in sorted(by_wp.keys())]
    handles += [
        Line2D([0], [0], marker="D", linestyle="",
               markerfacecolor="white", markeredgecolor="#444444",
               markersize=7, label="Flight 1 baseline"),
        Line2D([0], [0], marker="+", linestyle="", color="#444444",
               markersize=8, markeredgewidth=1.2, label="cluster mean"),
        Line2D([0], [0], linestyle=(0, (4, 2)), color="#444444",
               linewidth=0.9, label=r"$2\sigma$ ellipse"),
        Line2D([0], [0], marker="x", linestyle="", color="black",
               markersize=7, markeredgewidth=1.3, label="target"),
    ]
    axLeg.legend(handles=handles, loc="center",
                 ncol=5, handletextpad=0.4,
                 columnspacing=1.6, borderpad=0.0)

    header = ["Waypoint", r"$n$", r"$\bar{r}$ (cm)",
              r"$r_{\max}$ (cm)", r"$\sigma$ (cm)"]
    rows = []
    for wp_idx in sorted(by_wp.keys()):
        rad = np.hypot([p[0] for p in by_wp[wp_idx]],
                       [p[1] for p in by_wp[wp_idx]])
        rows.append([f"WP{wp_idx}", f"{len(rad)}",
                     f"{rad.mean()*100:.1f}",
                     f"{rad.max()*100:.1f}",
                     f"{rad.std()*100:.1f}"])
    all_rad = np.hypot(all_e, all_n)
    rows.append(["all", f"{len(all_rad)}",
                 f"{all_rad.mean()*100:.1f}",
                 f"{all_rad.max()*100:.1f}",
                 f"{all_rad.std()*100:.1f}"])

    tbl = axT.table(cellText=rows, colLabels=header,
                    cellLoc="center", colLoc="center",
                    loc="center", edges="horizontal")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(0.95, 1.35)
    for j in range(len(header)):
        tbl[(0, j)].set_text_props(fontweight="bold")
    for j in range(len(header)):
        tbl[(len(rows), j)].set_text_props(fontstyle="italic",
                                            color="#555555")

    print(f"Reference (Flight 1): {flight1}")
    print("\nPer-waypoint radial error:")
    for wp_idx in sorted(by_wp.keys()):
        rad = np.hypot([p[0] for p in by_wp[wp_idx]],
                       [p[1] for p in by_wp[wp_idx]])
        print(f"  WP{wp_idx}: n={len(rad)}  mean={rad.mean():.3f} m  "
              f"max={rad.max():.3f} m  std={rad.std():.3f} m")
    print(f"  ALL:  n={len(all_rad)}  mean={np.mean(all_rad):.3f} m  "
          f"max={np.max(all_rad):.3f} m  std={np.std(all_rad):.3f} m")

    if args.save:
        fig.savefig(args.save, facecolor="white")
        print(f"\nSaved figure to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()