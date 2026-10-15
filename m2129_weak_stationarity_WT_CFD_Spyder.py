# -*- coding: utf-8 -*-
"""Segmented pressure mean/std diagnostics for M2129 CSVs; run with F5.

Dependencies: numpy, pandas, scipy, matplotlib.
No pressure-to-velocity conversion or temperature is needed here.
Constant mean/variance are necessary, but not sufficient, for weak stationarity:
lag covariance must also be invariant. No automatic stationary verdict is made.
Rack_mean is the instantaneous arithmetic mean, not an area-weighted mean.
Its std is NOT the mean of individual probe stds (correlations matter).
Std uses ddof=0, i.e. RMS of fluctuations about each segment's own mean.
The incomplete last segment is discarded and its size is reported.
"""
from pathlib import Path
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

# ================= USER SETTINGS =================
# DATA_DIR = Path(r"D:/M2129_data")
DATA_DIR = Path(r"C:/Users/Tusas/Desktop/M2129")
WT_FILE = "WT_data.csv"       # Use None to skip WT
# WT_FILE = "experiment.csv"
CFD_PT_FILE = "CFD_Pt_data.csv"
CFD_PS_FILE = "CFD_Ps_data.csv"
RUN_WT = True
RUN_CFD = True
WT_TIME_WINDOW = (None, None)  # absolute start/end seconds, inclusive
CFD_TIME_WINDOW = (None, None)
SEGMENT_DURATION_S = 0.2  # same physical duration for WT and CFD
STUDY_DURATIONS_S = [0.02, 0.04, 0.1, 0.2, 0.4, 1.0]
MIN_SEGMENTS = 3
TIME_RELATIVE_TOLERANCE = 0.001
# None: display raw Pa. Otherwise supply a positive, fixed H0 in Pa.
# Do not use a time-varying reference: it changes the stationarity question.
NORMALIZATION_PA = {"WT": None, "CFD": None}
OUTPUT_DIR = DATA_DIR / "stationarity_results"
SHOW_FIGURES = True
DPI = 180
# =================================================
PT_NAMES = ([f"Pt_{i}_u" for i in range(1, 6)] + ["Pt_center"]
            + [f"Pt_{i}_l" for i in range(1, 6)])
PS_NAMES = ["Ps_1_u", "Ps_1_l", "Ps_1_left", "Ps_1_right"]


def read_csv(path):
    df = pd.read_csv(path, sep=",", encoding="utf-8-sig")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def extract(df, names):
    out = {}
    for name in names:
        matches = [c for c in df if not re.search(r"time\s*\(\s*s\s*\)", c, re.I) and re.search(
            r"(?<![A-Za-z0-9_])" + re.escape(name) + r"(?![A-Za-z0-9_])", c,
            flags=re.I)]
        if len(matches) != 1:
            raise ValueError(f"Expected one column for {name}; found {matches}")
        out[name] = pd.to_numeric(df[matches[0]], errors="raise").to_numpy(float)
    return pd.DataFrame(out)


def time_column(df):
    candidates = [c for c in df if re.search(r"time\s*\(\s*s\s*\)", c, re.I)]
    if len(candidates) != 1:
        raise ValueError(f"Cannot identify unique time column: {candidates}")
    t = pd.to_numeric(df[candidates[0]], errors="raise").to_numpy(float)
    if len(t) < 2 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError("Time must be finite and strictly increasing, with >=2 rows.")
    return t


def load_case(name):
    if name == "WT":
        df = read_csv(DATA_DIR / WT_FILE)
        t = time_column(df)
        data = extract(df, PT_NAMES + PS_NAMES)
        data["Ps_reference"] = data[PS_NAMES].mean(axis=1, skipna=False)
        window = WT_TIME_WINDOW
    else:
        df = read_csv(DATA_DIR / CFD_PT_FILE)
        t = time_column(df)
        data = extract(df, PT_NAMES)
        ps = read_csv(DATA_DIR / CFD_PS_FILE)
        ts = time_column(ps)
        cols = [c for c in ps if "ps_avg_aip_exp" in c.lower()]
        if len(cols) != 1:
            raise ValueError(f"Cannot identify CFD static pressure: {cols}")
        p = pd.to_numeric(ps[cols[0]], errors="raise").to_numpy(float)
        if not np.isfinite(p).all():
            raise ValueError("CFD static pressure contains nonfinite values.")
        # Independent pressure diagnostics: never interpolate or truncate Pt to Ps.
        # Static CFD is analyzed on its own time grid below.
        window = CFD_TIME_WINDOW
    data["Rack_mean"] = data[PT_NAMES].mean(axis=1, skipna=False)
    result = [(t, data, window)]
    if name == "CFD":
        result.append((ts, pd.DataFrame({"Ps_reference": p}), window))
    return result


def select_window(t, data, window):
    lo, hi = window
    mask = np.ones(len(t), dtype=bool)
    if lo is not None:
        mask &= t >= lo
    if hi is not None:
        mask &= t <= hi
    t, data = t[mask], data.loc[mask].reset_index(drop=True)
    if len(t) < 2 or not np.isfinite(data.to_numpy()).all():
        raise ValueError("Selected data are too short or contain NaN/Inf. No rows were silently removed.")
    dt = float(np.median(np.diff(t)))
    if np.max(np.abs(np.diff(t) - dt)) > TIME_RELATIVE_TOLERANCE * dt:
        raise ValueError("Nonuniform time grid: inspect missing samples/time precision.")
    return t, data, dt


def segments(t, y, n):
    count = len(y) // n
    if n < 2 or count < MIN_SEGMENTS:
        return None
    stop = count * n
    blocks = y[:stop].reshape(count, n)
    centers = t[:stop].reshape(count, n).mean(axis=1)
    return centers, blocks.mean(axis=1), blocks.std(axis=1, ddof=0)


def fit(x, y):
    if np.ptp(y) == 0:
        return 0.0, float(y[0]), np.nan
    reg = linregress(x - x[0], y)
    return reg.slope, reg.intercept, reg.rvalue ** 2


def save_figure(fig, name):
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / name, dpi=DPI)
    if not SHOW_FIGURES:
        plt.close(fig)


def run_analysis():
    if MIN_SEGMENTS < 3 or SEGMENT_DURATION_S <= 0:
        raise ValueError("Require MIN_SEGMENTS >=3 and positive segment duration.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_segments, summary, metadata = [], [], []
    results = {}
    for case, enabled in [("WT", RUN_WT), ("CFD", RUN_CFD)]:
        if not enabled:
            continue
        scale = NORMALIZATION_PA[case]
        if scale is not None and (not np.isfinite(scale) or scale <= 0):
            raise ValueError("Normalization must be None or a positive fixed pressure.")
        scale_value = 1.0 if scale is None else scale
        unit = "Pa" if scale is None else "P / H0"
        results[case] = {}
        for t0, data0, window in load_case(case):
            t, data, dt = select_window(t0, data0, window)
            n_main = max(2, int(round(SEGMENT_DURATION_S / dt)))
            metadata.append(dict(case=case, channels=", ".join(data.columns),
                                 start_s=t[0], end_s=t[-1], samples=len(t),
                                 fs_Hz=1/dt, normalization_Pa=scale,
                                 main_segment_samples=n_main))
            for channel in data:
                y = data[channel].to_numpy()
                durations = sorted(set([SEGMENT_DURATION_S] + list(STUDY_DURATIONS_S)))
                for duration in durations:
                    if not np.isfinite(duration) or duration <= 0:
                        raise ValueError("Study durations must be positive and finite.")
                    n = max(2, int(round(duration / dt)))
                    seg = segments(t, y, n)
                    if seg is None:
                        warnings.warn(f"{case}/{channel}: {duration:g}s skipped; too few complete segments.")
                        continue
                    x, means, stds = seg
                    record = dict(case=case, channel=channel, requested_duration_s=duration,
                                  actual_duration_s=n*dt, segment_samples=n,
                                  segments=len(x), discarded_samples=len(y) % n)
                    for key, values in [("mean", means), ("std", stds)]:
                        slope, intercept, r2 = fit(x, values / scale_value)
                        record.update({key+"_slope_per_s": slope, key+"_r2": r2,
                                       key+"_fit_change": slope*(x[-1]-x[0])})
                    summary.append(record)
                    if duration == SEGMENT_DURATION_S:
                        results[case][channel] = (x, means/scale_value, stds/scale_value)
                        all_segments.extend(dict(case=case, channel=channel, time_s=a,
                                                 mean_Pa=b, std_Pa=c)
                                            for a, b, c in zip(x, means, stds))
                        fig, axs = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
                        for ax, values, label in zip(axs, [means/scale_value, stds/scale_value],
                                                     ["Segment mean", "Fluctuation RMS (std)"]):
                            slope, intercept, r2 = fit(x, values)
                            ax.plot(x, values, "o", ms=3)
                            ax.plot(x, intercept+slope*(x-x[0]), "k--")
                            ax.set_ylabel(f"{label} [{unit}]")
                            ax.set_title(f"slope={slope:.3e}/s; R²={r2:.3f}")
                            ax.grid(alpha=.3)
                        axs[-1].set_xlabel("Physical time [s]")
                        fig.suptitle(f"{case}: {channel}; segment={n*dt:g} s")
                        save_figure(fig, f"{case}_{channel}.png")
        if not results[case]:
            raise ValueError(f"{case}: main duration produces too few segments; reduce SEGMENT_DURATION_S.")
        fig, axs = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
        for channel in PT_NAMES + ["Rack_mean"]:
            if channel not in results[case]:
                continue
            x, means, stds = results[case][channel]
            style = dict(color="black", lw=2.6) if channel == "Rack_mean" else dict(lw=.9, alpha=.65)
            axs[0].plot(x, means, label=channel, **style)
            axs[1].plot(x, stds, **style)
        axs[0].set_ylabel(f"Segment mean [{unit}]")
        axs[1].set_ylabel(f"Fluctuation RMS [{unit}]")
        axs[1].set_xlabel("Physical time [s]")
        axs[0].legend(ncol=4, fontsize=8)
        for ax in axs:
            ax.grid(alpha=.3)
        fig.suptitle(f"{case}: total pressure probes and instantaneous rack mean")
        save_figure(fig, f"{case}_all_probes.png")
    summary = pd.DataFrame(summary)
    segmented = pd.DataFrame(all_segments)
    summary.to_csv(OUTPUT_DIR / "segment_size_study.csv", index=False)
    segmented.to_csv(OUTPUT_DIR / "segment_statistics.csv", index=False)
    pd.DataFrame(metadata).to_csv(OUTPUT_DIR / "analysis_metadata.csv", index=False)
    if not summary.empty:
        fig, axs = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
        for case in results:
            subset = summary[(summary.case == case) & (summary.channel == "Rack_mean")]
            for ax, key in zip(axs, ["mean", "std"]):
                ax.plot(subset.actual_duration_s, subset[key+"_slope_per_s"], "o-", label=case)
                ax.set_ylabel(key + " slope [display units/s]")
                ax.set_xscale("log")
                ax.grid(alpha=.3)
                ax.legend()
        axs[-1].set_xlabel("Actual segment duration [s]")
        fig.suptitle("Segment-duration sensitivity: instantaneous rack mean")
        save_figure(fig, "rack_mean_segment_duration_study.png")
    print(f"Saved results: {OUTPUT_DIR}")
    print("Mean/std trend diagnostics only; no automatic stationarity verdict.")
    if SHOW_FIGURES:
        plt.show()
    return results, summary, segmented


if __name__ == "__main__":
    RESULTS, STUDY_SUMMARY, SEGMENT_STATISTICS = run_analysis()
