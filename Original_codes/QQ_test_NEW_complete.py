# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 18:12:26 2025

@author: Tusos
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
import pandas as pd
from scipy.stats import pearsonr
import os


def _quantiles_at_common_probs(x, y, n_q=None):
    """
    Pair X, Y by the same probability levels.
    If n_q is None, use min(len(x), len(y)).
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if n_q is None:
        n_q = max(50, min(len(x), len(y)))  # at least some resolution
    p = (np.arange(1, n_q + 1) - 0.5) / n_q  # (0,1) plotting positions

    # numpy changed 'interpolation' -> 'method'; support both
    try:
        xq = np.quantile(x, p, method="linear")
        yq = np.quantile(y, p, method="linear")
    except TypeError:
        xq = np.quantile(x, p, interpolation="linear")
        yq = np.quantile(y, p, interpolation="linear")
    return xq, yq, p


def _regression_with_ci(x, y, alpha=0.05, x_grid=None):
    """
    Simple linear regression y = a + b x with (1-alpha) CI for mean prediction.
    Returns dict with slope, intercept, r, r2, yhat(x_grid), lo, hi.
    """
    x = np.asarray(x); y = np.asarray(y)
    n = x.size
    lr = stats.linregress(x, y)  # slope, intercept, rvalue, pvalue, stderr

    if x_grid is None:
        x_grid = np.linspace(x.min(), x.max(), 200)

    # Predicted line at the original x (for residuals) and on grid (to draw band)
    yhat = lr.intercept + lr.slope * x
    yhat_grid = lr.intercept + lr.slope * x_grid

    # Standard error of regression
    s_err = np.sqrt(np.sum((y - yhat)**2) / (n - 2))
    x_mean = np.mean(x)
    Sxx = np.sum((x - x_mean)**2)

    # t critical
    tval = stats.t.ppf(1 - alpha / 2, df=n - 2)

    # CI for the mean response at each x_grid
    conf = tval * s_err * np.sqrt(1/n + (x_grid - x_mean)**2 / Sxx)
    lo = yhat_grid - conf
    hi = yhat_grid + conf

    return {
        "slope": lr.slope,
        "intercept": lr.intercept,
        "r": lr.rvalue,
        "r2": lr.rvalue**2,
        "x_grid": x_grid,
        "yhat_grid": yhat_grid,
        "lo": lo,
        "hi": hi,
    }


# -----------------------------------------------------------------------------
# Earlier version of qq_compare_plot (commented out in the photographs).
# It is intentionally left commented, as in the original source.
# -----------------------------------------------------------------------------
# def qq_compare_plot(
#         test_data, cfd_data,
#         n_q=None,
#         alpha=0.05,
#         acceptance=0.015,           # 1.5% default
#         acceptance_mode="relative", # "relative" -> +/- percentage; "absolute" -> +/- delta
#         annotate_stats=True,
#         title="Q-Q validation: CFD vs Test",
#         ax=None
# ):
#     """
#     Build a Q-Q plot comparing CFD (Y) to TEST (X) amplitudes.
#
#     Parameters
#     ----------
#     test_data : array-like
#         Test record (X).
#     cfd_data : array-like
#         CFD record (Y).
#     n_q : int or None
#         Number of paired quantiles. Default uses min(Nx, Ny).
#     alpha : float
#         For the regression confidence band (e.g., 0.05 -> 95%).
#     acceptance : float
#         If acceptance_mode == "relative": +/- percentage (e.g., 0.015 for +/-1.5%).
#         If acceptance_mode == "absolute": +/- absolute delta in data units.
#     acceptance_mode : {"relative", "absolute"}
#     """
#     xq, yq, p = _quantiles_at_common_probs(test_data, cfd_data, n_q=n_q)
#     reg = _regression_with_ci(xq, yq, alpha=alpha)
#
#     # Equality line and acceptance band on a grid spanning data
#     xg = reg["x_grid"]
#     eq = xg  # y = x
#
#     if acceptance_mode == "relative":
#         lower_acc = xg * (1 - acceptance)
#         upper_acc = xg * (1 + acceptance)
#         acc_label = f"+/-{acceptance*100:.2f}% Acceptance Interval"
#     elif acceptance_mode == "absolute":
#         lower_acc = xg - acceptance
#         upper_acc = xg + acceptance
#         acc_label = f"+/-{acceptance:g} Acceptance Interval"
#     else:
#         raise ValueError("acceptance_mode must be 'relative' or 'absolute'")
#
#     # Equivalence check: is the entire (1-alpha) CI inside the acceptance band?
#     ci_inside = np.all(reg["lo"] >= lower_acc) and np.all(reg["hi"] <= upper_acc)
#
#     # Plot
#     if ax is None:
#         fig, ax = plt.subplots(figsize=(8, 6), dpi=120)
#
#     ax.plot(xq, yq, "o", ms=4, mfc="none", mec="k", label="Q-Q Data")
#     ax.plot(xg, reg["yhat_grid"], "--", lw=1.5, label="Regression")
#     ax.plot(xg, reg["lo"], "--", lw=1.2, label=f"{int((1-alpha)*100)}% Confidence Interval")
#     ax.plot(xg, reg["hi"], "--", lw=1.2)
#     ax.plot(xg, eq, "-.", lw=1.5, label="Equivalent (Y = X)")
#     ax.plot(xg, lower_acc, "--", lw=1.2, label=acc_label)
#     ax.plot(xg, upper_acc, "--", lw=1.2)
#
#     # Mark data min/max (paired quantiles extremes)
#     ax.plot([xq.min(), xq.max()], [yq.min(), yq.max()], "^", ms=7,
#             color="tab:red", label="Data Min, Max")
#     ax.set_xlabel("Test Data (X)")
#     ax.set_ylabel("CFD Data (Y)")
#     ax.set_title(title)
#
#     # Stats box
#     if annotate_stats:
#         txt = (
#             f"Test Mean = {np.mean(test_data):.3f}\n"
#             f"CFD  Mean = {np.mean(cfd_data):.3f}\n"
#             f"Test Std. Dev. = {np.std(test_data, ddof=1):.3f}\n"
#             f"CFD  Std. Dev. = {np.std(cfd_data, ddof=1):.3f}\n"
#             f"R² (Q-Q) = {reg['r2']:.3f}\n"
#             f"CFD N = {len(cfd_data)}\n"
#             f"TEST N = {len(test_data)}\n"
#             f"Equivalence (CI <= accept.): {'YES' if ci_inside else 'NO'}"
#         )
#         ax.text(0.7, 0.02, txt,
#                 transform=ax.transAxes,
#                 va="bottom", ha="left",
#                 fontsize=9,
#                 bbox=dict(boxstyle="round,pad=0.35", alpha=0.15)
#                 )
#
#     ax.legend(loc="best", fontsize=9)
#     ax.grid(True, alpha=0.25)
#     # Keep square-ish aspect so 45 degree looks right
#     lim_min = min(ax.get_xlim()[0], ax.get_ylim()[0])
#     lim_max = max(ax.get_xlim()[1], ax.get_ylim()[1])
#     ax.set_xlim(lim_min, lim_max)
#     ax.set_ylim(lim_min, lim_max)
#     return ax


def qq_compare_plot(
        test_data, cfd_data,
        n_q=None,
        alpha=0.05,
        acceptance=0.015,     # 1.5% default
        acceptance_mode="relative",  # "relative" -> percentage; "absolute" -> +/- delta
        annotate_stats=True,
        title="Q-Q validation: CFD vs Test",
        ax=None
):
    # ---------- prepare Data ----------
    x = np.asarray(test_data).ravel()
    y = np.asarray(cfd_data).ravel()
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]

    # quantiles with same probability
    if n_q is None:
        n_q = max(50, min(len(x), len(y)))
    p = (np.arange(1, n_q + 1) - 0.5) / n_q
    try:
        xq = np.quantile(x, p, method="linear")
        yq = np.quantile(y, p, method="linear")
    except TypeError:
        xq = np.quantile(x, p, interpolation="linear")
        yq = np.quantile(y, p, interpolation="linear")

    # Regression and 95% band
    lr = stats.linregress(xq, yq)
    xg = np.linspace(np.min(xq), np.max(xq), 400)
    yhat_q = lr.intercept + lr.slope * xq
    yhat_g = lr.intercept + lr.slope * xg
    n = len(xq)
    s_err = np.sqrt(np.sum((yq - yhat_q)**2) / (n - 2))
    xbar = np.mean(xq)
    Sxx = np.sum((xq - xbar)**2)
    tval = stats.t.ppf(1 - alpha/2, df=n-2)
    conf = tval * s_err * np.sqrt(1/n + (xg - xbar)**2 / Sxx)
    lo, hi = yhat_g - conf, yhat_g + conf

    # Equality line and acceptance lines
    eq = xg
    if acceptance_mode == "relative":
        lower_acc = xg * (1 - acceptance)
        upper_acc = xg * (1 + acceptance)
        acc_label = f"+/-{acceptance*100:.2f}% Acceptance Interval"
    elif acceptance_mode == "absolute":
        lower_acc = xg - acceptance
        upper_acc = xg + acceptance
        acc_label = f"+/-{acceptance:g} Acceptance Interval"
    else:
        raise ValueError("acceptance_mode must be 'relative' or 'absolute'")

    ci_inside = np.all(lo >= lower_acc) and np.all(hi <= upper_acc)

    # Plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6), dpi=120)

    # Styles
    ax.plot(xq, yq, "o", ms=4, mfc="none", mec="k", label="QQ Data")  # points
    ax.plot(xg, yhat_g, "--", lw=1.6, color="k", label="Regression")  # regression
    ax.plot(xg, lo, "--", lw=1.3, color="r", label="95% Confidence Interval")  # CI low
    ax.plot(xg, hi, "--", lw=1.3, color="r", label="_nolegend_")  # CI high
    ax.plot(xg, eq, "--", lw=1.6, color="b", label="Equivalent")  # Y=X
    ax.plot(xg, lower_acc, "--", lw=1.6, color="g", label=acc_label)  # acceptance low
    ax.plot(xg, upper_acc, "--", lw=1.6, color="g", label="_nolegend_")  # acceptance high

    ax.set_xlabel("Test Data, PTij/PT0")
    ax.set_ylabel("CFD Data, PTij/PT0")
    ax.set_title(title)

    # Box in the low right corner
    if annotate_stats:
        txt = (
            f"Test Mean = {np.mean(x):.3f}\n"
            f"CFD Mean = {np.mean(y):.3f}\n"
            f"Test Stan. Dev. = {np.std(x, ddof=1):.3f}\n"
            f"CFD Stan. Dev. = {np.std(y, ddof=1):.3f}\n"
            f"R² (Q-Q) = {lr.rvalue**2:.3f}\n"
            f"CFD Sample Size = {len(y)}\n"
            f"Test Sample Size = {len(x)}"
        )
        ax.text(0.98, 0.02, txt, transform=ax.transAxes,
                ha="right", va="bottom", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.35", alpha=0.15))

    # grid and axes ratio
    ax.grid(True, alpha=0.25)
    vmin = min(ax.get_xlim()[0], ax.get_ylim()[0])
    vmax = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    ax.set_aspect("equal", adjustable="box")

    # ---------- NEW: Data min max over axes ----------
    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))
    xrng = xmax - xmin
    yrng = ymax - ymin
    xpad = 0.015 * xrng
    ypad = 0.015 * yrng

    # Vertical axis
    ax.plot([xmin - xpad, xmin], [ymin, ymax], marker="^", ms=9, color="red",
            linestyle="None", clip_on=False, label="Data Min, Max")
    ax.plot([xmin - xpad, xmin], [ymax, ymin], marker="^", ms=9, color="red",
            linestyle="None", clip_on=False, label="_nolegend_")

    # Horizontal axis
    ax.plot([xmin, xmax], [ymin - ypad, ymin - ypad], marker="^", ms=9,
            color="red", linestyle="None", clip_on=False, label="_nolegend_")
    ax.plot([xmax, xmin], [ymin - ypad, ymin - ypad], marker="^", ms=9,
            color="red", linestyle="None", clip_on=False, label="_nolegend_")

    # Legends
    ax.legend(loc="upper left", fontsize=9, frameon=True)

    return {
        "ax": ax,
        "slope": lr.slope,
        "intercept": lr.intercept,
        "r": lr.rvalue,
        "r2": lr.rvalue**2,
        "n": len(xq),
        "x_min": np.min(xq),
        "x_max": np.max(xq),
        "y_min": np.min(yq),
        "y_max": np.max(yq),
    }


# ---------------------- Example usage ------------------------
# x_test = ...  # 1D array of your TEST amplitude data
# y_cfd = ...   # 1D array of your CFD amplitude data
# ax = qq_compare_plot(x_test, y_cfd,
#                      n_q=None,               # auto
#                      alpha=0.05,             # 95% CI
#                      acceptance=0.015,       # +/-1.5%
#                      acceptance_mode="relative",
#                      title="Q-Q plot: CFD vs Test (PTij/PT0)")
# plt.show()

# NOTE TO USER: The following paths / run identifiers are transcribed from
# photographs; confirm them with the actual original files before execution.
# wdir = 'D:/GE_PACKAGE_1_Ma130_AoA0_AoS5/2016/'
# wt_runid = 'C1060Run024272016'
# cfd_runid = 'C1060Run091172016_CFD'
wdir = 'D:/GE_PACKAGE_1_Ma130_AoA0_AoS5/4_SSEC/2014/'
wt_runid = 'C1060Run024272014'
cfd_runid = 'C1060Run091172014_CFD'
os.chdir(wdir)
parts = wt_runid.split('T')
suffix = parts[1]

# wdir = '//vds/Bolumler/J13F0/VOLCANO/DATA/STATISTICS/Mach130_AoA0_AoS5'
# wdir = '//vds/Bolumler/J13F0/VOLCANO/DATA/STATISTICS/Mach130_AoA0_AoS0_VREMAN/1994'
# wdir = 'D:/Volcano_LES/Mach110_AoA0_AoS0'
# wdir = 'C:/Users/t26206/Desktop/BB_R1_DDES_WT'
# os.chdir(wdir)

# --- File Paths ---
wt_pressure_file = wt_runid + '.unsteady'
wt_steady_file = wt_runid + '.steady'
DES_pressure_file = cfd_runid + '.unsteady'
DES_steady_file = cfd_runid + '.steady'

# Read H0 (P_total_inf) from steady files
wt_steady = pd.read_csv(wt_steady_file, sep='\t')
cfd_steady = pd.read_csv(DES_steady_file, sep='\t')
pt0_wt = wt_steady.at[0, 'H0']
pt0_cfd = cfd_steady.at[0, 'H0']

# coords_file = 'ProbeCoordinates_WT.xlsx'
# --- Load Data ---
wt_pressure_df = pd.read_csv(wt_pressure_file, sep='\t')

# Rolling WT and CFD data: this is used where we are interested in engine stability;
# NOTE: Do not use this for the raw pressure statistics calculations and validations

# window_n = 16    # Mean average of 16 datapoints (rolling)
window_n = 1        # Skip mean average rolling
mask = wt_pressure_df.columns.str.startswith(('UHAD', 'USAD'))
wt_rolled_data = wt_pressure_df.copy()
wt_rolled_data.loc[:, mask] = (wt_pressure_df.loc[:, mask].rolling(window=window_n, min_periods=1).mean())


des_pressure_df = pd.read_csv(DES_pressure_file, sep='\t')
cfd_rolled_data = des_pressure_df.copy()
cfd_rolled_data.loc[:, mask] = (des_pressure_df.loc[:, mask].rolling(window=window_n, min_periods=1).mean())

x_test = wt_rolled_data['UHAD102']
x_test /= pt0_wt
y_cfd = cfd_rolled_data['UHAD102']
y_cfd /= pt0_cfd

# TEST PLOT ##########################################################
# ax = qq_compare_plot(x_test, y_cfd,
#                      n_q=None,                  # auto
#                      alpha=0.05,               # 95% CI
#                      acceptance=0.015,          # +/-1.5%
#                      acceptance_mode="relative",
#                      title="Q-Q plot: CFD vs Test (PTij/PT0)")
# plt.show()
######################################################################


# Q-Q Plots for all 40 probes
wt_pressure_df = pd.read_csv(wt_pressure_file, sep='\t')
des_pressure_df = pd.read_csv(DES_pressure_file, sep='\t')
sampling_rate = 25000
test_start = 0.0
period = 4.5
cfd_start = 0.00
cfd_period = 4.5

test_end = test_start + period
test_time = test_end - test_start
test_start_sample = int(sampling_rate * test_start)
test_time = round(test_time, 2)
test_end_sample = int(test_start_sample + sampling_rate * period)
wt_pressure_df = wt_pressure_df.iloc[test_start_sample:test_end_sample].copy()  # WT data from 1 to 1.21 second

cfd_start_sample = int(sampling_rate * cfd_start)
cfd_end_sample = int(cfd_start_sample + sampling_rate * cfd_period)
dfc = cfd_rolled_data.copy()
# dfc = pd.read_csv(cfd_path, sep='\t')
dfc = dfc.iloc[cfd_start_sample:cfd_end_sample].copy()  # WT data from 1 to 1.21 second
cfd_time = len(dfc) / sampling_rate
cfd_time = round(cfd_time, 2)

out_dir = f"qq_figs_T{suffix}_WT{test_time}sec_CFD{cfd_time}sec"
os.makedirs(out_dir, exist_ok=True)
cols = [f"UHAD{i}{k:02d}" for i in range(1, 9) for k in [1, 2, 3, 4, 5]]
print(f"{len(cols)} columns to process:", cols)
for col in cols:
    x = wt_pressure_df[col].to_numpy().copy()
    y = des_pressure_df[col].to_numpy().copy()
    x /= pt0_wt
    y /= pt0_cfd

    title = f"Q-Q: {col} (CFD vs Test)"
    result = qq_compare_plot(x, y, n_q=None, alpha=0.05,
                             acceptance=0.015, acceptance_mode="relative", title=title)

    # Now you can access the slope!
    print(f"{col}: Slope = {result['slope']:.4f}, R² = {result['r2']:.4f}")

    out_png = os.path.join(out_dir, f"qq_{col}.png")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()


# # Statistics for TEST and CFD export to .csv file
# # --- Prepare storage for statistics ---
# results = []
# wt_pressure_df = pd.read_csv(wt_pressure_file, sep='\t')
# des_pressure_df = pd.read_csv(DES_pressure_file, sep='\t')
# sampling_rate = 25000
# test_start = 0.1
# #period = 0.21
# period = 4.4
#
# test_end = test_start + period
# test_start_sample = int(sampling_rate*test_start)
# test_end_sample = int(test_start_sample + sampling_rate*period)
# wt_pressure_df = wt_pressure_df.iloc[test_start_sample:test_end_sample].copy()  # WT data from ...
#
# # --- Loop over each UHAD channel ---
# for col in cols:
#     # Extract data
#     x = wt_pressure_df[col].to_numpy()
#     y = des_pressure_df[col].to_numpy()
#
#     # Normalize by total pressure
#     x /= pt0_wt
#     y /= pt0_cfd
#
#     # Remove NaNs
#     x = x[~np.isnan(x)]
#     y = y[~np.isnan(y)]
#
#     # Skip if empty
#     if len(x) == 0 or len(y) == 0:
#         print(f"[Warning] Skipping {col}: empty data after NaN removal")
#         continue
#
#     # --- Compute TEST statistics ---
#     test_min = np.min(x)
#     test_max = np.max(x)
#     test_mean = np.mean(x)
#     test_std = np.std(x, ddof=1)  # sample std
#     test_skew = stats.skew(x, nan_policy='omit')
#     test_kurt = stats.kurtosis(x, nan_policy='omit')
#     test_n = len(x)
#
#     # --- Compute CFD statistics ---
#     cfd_min = np.min(y)
#     cfd_max = np.max(y)
#     cfd_mean = np.mean(y)
#     cfd_std = np.std(y, ddof=1)
#     cfd_skew = stats.skew(y, nan_policy='omit')
#     cfd_kurt = stats.kurtosis(y, nan_policy='omit')
#     cfd_n = len(y)
#
#     # --- Q-Q Regression (for R²) ---
#     # Use quantiles at common probabilities
#     try:
#         xq, yq, p = _quantiles_at_common_probs(x, y, n_q=None)
#         lr = stats.linregress(xq, yq)
#         r2 = lr.rvalue ** 2
#     except Exception as e:
#         print(f"[Error] Failed Q-Q regression for {col}: {e}")
#         r2 = np.nan
#     angle_deg = np.arctan(lr.slope) * (180 / np.pi)
#
#     # --- Append to results ---
#     results.append({
#         "Channel": col,
#         "Test_Min": test_min,
#         "Test_Max": test_max,
#         "Test_Mean": test_mean,
#         "Test_Std": test_std,
#         "Test_Skewness": test_skew,
#         "Test_Kurtosis": test_kurt,
#         "Test_Sample_Size": test_n,
#
#         "CFD_Min": cfd_min,
#         "CFD_Max": cfd_max,
#         "CFD_Mean": cfd_mean,
#         "CFD_Std": cfd_std,
#         "CFD_Skewness": cfd_skew,
#         "CFD_Kurtosis": cfd_kurt,
#         "CFD_Sample_Size": cfd_n,
#
#         "R2_QQ": r2,
#         "Slope": lr.slope,
#         "Angle": angle_deg
#     })
#
# # --- Convert to DataFrame ---
# df_results = pd.DataFrame(results)
#
# # --- Save to CSV ---
# output_csv = "uhad_channel_statistics.csv"
# df_results.to_csv(output_csv, index=False)
# print(f"\n✅ Statistics saved to: {output_csv}")
#
#
# # Correlation Matrix
#
# # Load the CSV
# df = pd.read_csv("uhad_channel_statistics.csv")
#
# # Select only TEST and CFD columns (exclude sample sizes and R2_QQ)
# test_cols = [col for col in df.columns if col.startswith("Test_") and not col.endswith("_Sample_Size")]
# cfd_cols = [col for col in df.columns if col.startswith("CFD_") and not col.endswith("_Sample_Size")]
#
# # Combine
# var_cols = test_cols + cfd_cols
# df_clean = df[var_cols].copy()
#
# # Compute correlation matrix
# corr_matrix = df_clean.corr(method="pearson")
#
# # Plot heatmap
# plt.figure(figsize=(20, 8))
# sns.heatmap(
#     corr_matrix,
#     annot=True,
#     fmt=".2f",
#     cmap="coolwarm",
#     center=0,
#     square=True,
#     cbar_kws={"shrink": 0.8},
#     linewidths=0.5
# )
# plt.title("Pearson Correlation: TEST vs CFD Variables (Across UHAD Channels)", fontsize=14, pad=20)
# plt.tight_layout()
# plt.savefig("correlation_matrix_test_vs_cfd_across_channels.png", dpi=300, bbox_inches="tight")
# plt.show()
#
# # Save to CSV
# corr_matrix.to_csv("correlation_matrix_test_vs_cfd_across_channels.csv")
# print("✅ Correlation matrix saved to: correlation_matrix_test_vs_cfd_across_channels.csv")
#
#
# # overall Correlation
# corr, pval = pearsonr(df["Test_Min"], df["CFD_Min"])
# print(f"🔗 Correlation (Test_Min vs CFD_Min): {corr:.4f}, p = {pval:.4f}")
#
# if pval < 0.001:
#     sig = "***"
# elif pval < 0.01:
#     sig = "**"
# elif pval < 0.05:
#     sig = "*"
# else:
#     sig = "ns"
# print(f"   Significance: {sig}")
#
#
# # Scatter plots
#
# # --- Load the CSV ---
# df = pd.read_csv("uhad_channel_statistics.csv")
#
# # --- Define variables ---
# # Top: Min, Max, Mean
# test_minmaxmean = ["Test_Min", "Test_Max", "Test_Mean"]
# cfd_minmaxmean = ["CFD_Min", "CFD_Max", "CFD_Mean"]
# titles_minmaxmean = ["Min", "Max", "Mean"]
#
# # Bottom: Std, Skewness, Kurtosis
# test_stdskewkurt = ["Test_Std", "Test_Skewness", "Test_Kurtosis"]
# cfd_stdskewkurt = ["CFD_Std", "CFD_Skewness", "CFD_Kurtosis"]
# titles_stdskewkurt = ["Std", "Skewness", "Kurtosis"]
#
# # Colors
# colors = ["#E63946", "#F1C40F", "#2ECC71"]  # Red, Yellow, Green
#
# # --- Determine shared axis ranges for top panel (Min/Max/Mean) ---
# vmin, vmax = 0.8, 1.2  # Typical range for pressure ratios
#
# # --- Re-evaluate and set optimized ranges for bottom panel ---
# # Use actual data min/max to center the plot
# ranges = {
#     "Test_Std": (df["Test_Std"].min() * 0.95, df["Test_Std"].max() * 1.05),
#     "Test_Skewness": (df["Test_Skewness"].min() * 1.1, df["Test_Skewness"].max() * 1.1),
#     "Test_Kurtosis": (df["Test_Kurtosis"].min() * 0.95, df["Test_Kurtosis"].max() * 1.05)
# }
#
# # --- Helper: Find channels with min and max deviation ---
# def get_extreme_channels(test_col, cfd_col):
#     """Return channel with largest and smallest deviation"""
#     df_temp = df.copy()
#     df_temp["Deviation"] = abs(df_temp[test_col] - df_temp[cfd_col])
#     max_dev = df_temp.loc[df_temp["Deviation"].idxmax()]
#     min_dev = df_temp.loc[df_temp["Deviation"].idxmin()]
#     return max_dev["Channel"], min_dev["Channel"]
#
# # --- Create figure with 2 rows: top (1x3), bottom (1x3) ---
# fig, (ax_top, ax_bottom) = plt.subplots(2, 3, figsize=(16, 8), sharex=False, sharey=False)
#
# # --- Top Panel: Min, Max, Mean ---
# for i, (t_col, c_col, title, color) in enumerate(zip(test_minmaxmean, cfd_minmaxmean, titles_minmaxmean, colors)):
#     ax = ax_top[i]
#
#     # Scatter plot (all points)
#     ax.scatter(df[t_col], df[c_col], alpha=0.7, color=color, s=50)
#
#     # Diagonal line (y = x)
#     ax.plot([vmin, vmax], [vmin, vmax], 'k--', lw=1, alpha=0.5)
#
#     # Set limits
#     ax.set_xlim(vmin, vmax)
#     ax.set_ylim(vmin, vmax)
#
#     # Labels
#     ax.set_xlabel("TEST", fontsize=10)
#     ax.set_ylabel("CFD", fontsize=10)
#     ax.set_title(f"Test vs CFD: {title}", fontsize=10, pad=6)
#
#     # Grid
#     ax.grid(True, alpha=0.3)
#
#     # Get extreme channels
#     max_ch, min_ch = get_extreme_channels(t_col, c_col)
#
#     # Plot largest deviation (star)
#     row_max = df[df["Channel"] == max_ch].iloc[0]
#     ax.scatter(row_max[t_col], row_max[c_col], color="black", s=120, marker="*", edgecolor="white", linewidth=1.2, zorder=5)
#
#     # Plot smallest deviation (circle)
#     row_min = df[df["Channel"] == min_ch].iloc[0]
#     ax.scatter(row_min[t_col], row_min[c_col], color="black", s=80, marker="o", edgecolor="white", linewidth=1.2, zorder=5)
#
#     # Store channel names for legend (only on first subplot)
#     if i == 0:
#         ax.plot([], [], '*', color='black', markersize=8, markerfacecolor='none', markeredgecolor='black', label=f"Largest Dev: {max_ch}")
#         ax.plot([], [], 'o', color='black', markersize=8, markerfacecolor='none', markeredgecolor='black', label=f"Smallest Dev: {min_ch}")
#
# # --- Bottom Panel: Std, Skewness, Kurtosis ---
# for i, (t_col, c_col, title, color) in enumerate(zip(test_stdskewkurt, cfd_stdskewkurt, titles_stdskewkurt, colors)):
#     ax = ax_bottom[i]
#
#     # Scatter plot (all points)
#     ax.scatter(df[t_col], df[c_col], alpha=0.7, color=color, s=50)
#
#     # Diagonal line
#     x_range = ranges[t_col]
#     ax.plot(x_range, x_range, 'k--', lw=1, alpha=0.5)
#     ax.set_xlim(x_range)
#     ax.set_ylim(x_range)
#
#     # Labels
#     ax.set_xlabel("TEST", fontsize=10)
#     ax.set_ylabel("CFD", fontsize=10)
#     ax.set_title(f"Test vs CFD: {title}", fontsize=10, pad=6)
#
#     # Grid
#     ax.grid(True, alpha=0.3)
#
#     # Get extreme channels
#     max_ch, min_ch = get_extreme_channels(t_col, c_col)
#
#     # Plot largest deviation (star)
#     row_max = df[df["Channel"] == max_ch].iloc[0]
#     ax.scatter(row_max[t_col], row_max[c_col], color="black", s=120, marker="*", edgecolor="white", linewidth=1.2, zorder=5)
#
#     # Plot smallest deviation (circle)
#     row_min = df[df["Channel"] == min_ch].iloc[0]
#     ax.scatter(row_min[t_col], row_min[c_col], color="black", s=80, marker="o", edgecolor="white", linewidth=1.2, zorder=5)
#
#     # Add legend only once (on first subplot)
#     if i == 0:
#         ax.plot([], [], 's', color='black', markersize=8, markerfacecolor='none', markeredgecolor='black', label=f"Largest Dev: {max_ch}")
#         ax.plot([], [], 'o', color='black', markersize=8, markerfacecolor='none', markeredgecolor='black', label=f"Smallest Dev: {min_ch}")
#
# # --- Add legend in empty space (top-left subplot) ---
# ax_top[0].legend(loc='upper right', fontsize=9, frameon=True, fancybox=True, shadow=True, borderpad=1)
#
# # --- Finalize layout ---
# plt.tight_layout(pad=2.5)
#
# # --- Save high-res image ---
# plt.savefig("test_vs_cfd_comprehensive_scatter_with_named_legend.png", dpi=300, bbox_inches="tight")
# plt.show()
#
# print("✅ Saved: test_vs_cfd_comprehensive_scatter_with_named_legend.png")
