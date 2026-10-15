# -*- coding: utf-8 -*-
"""
Reconstructed from user-provided screenshots of QQ_all40_with_statistical_box.py.
The code layout, formulas, and settings are preserved as closely as possible.
Please verify local directories, run IDs and column names before running.
"""

# Created on Tue Sep  9 15:31:25 2025
# @author: Tasos  # Author name is not completely legible in the photograph

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import math
matplotlib.use('Qt5Agg')  # Use Qt backend for interactive plots
from scipy import stats


# ---------------- Helpers from previous code ----------------
def _quantiles_at_common_probs(x, y, n_q=None):
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if n_q is None:
        n_q = max(100, min(len(x), len(y)))
    p = (np.arange(1, n_q + 1) - 0.5) / n_q
    try:
        xq = np.quantile(x, p, method='linear')
        yq = np.quantile(y, p, method='linear')
    except TypeError:
        xq = np.quantile(x, p, interpolation='linear')
        yq = np.quantile(y, p, interpolation='linear')
    return xq, yq, p


def _regression_with_ci(x, y, alpha=0.05, x_grid=None):
    x = np.asarray(x)
    y = np.asarray(y)
    n = x.size
    lr = stats.linregress(x, y)
    if x_grid is None:
        x_grid = np.linspace(x.min(), x.max(), 200)
    yhat = lr.intercept + lr.slope * x
    yhat_grid = lr.intercept + lr.slope * x_grid
    s_err = np.sqrt(np.sum((y - yhat)**2) / max(n - 2, 1))
    x_mean = np.mean(x)
    sxx = np.sum((x - x_mean)**2) if n > 1 else 1.0
    tval = stats.t.ppf(1 - alpha / 2, df=max(n - 2, 1))
    conf = tval * s_err * np.sqrt(1 / max(n, 1) + (x_grid - x_mean)**2 / max(sxx, 1e-12))
    lo = yhat_grid - conf
    hi = yhat_grid + conf
    return {
        'slope': lr.slope, 'intercept': lr.intercept, 'r2': lr.rvalue**2,
        'x_grid': x_grid, 'yhat_grid': yhat_grid, 'lo': lo, 'hi': hi,
    }


def summarize_moments(a):
    a = np.asarray(a).ravel()
    a = a[np.isfinite(a)]
    n = a.size
    mean = np.mean(a) if n > 0 else np.nan
    std = np.std(a, ddof=1) if n > 1 else np.nan
    skew = stats.skew(a, bias=False) if n > 2 else np.nan
    kurt_ex = stats.kurtosis(a, fisher=True, bias=False) if n > 3 else np.nan
    return {'n': n, 'mean': mean, 'std': std, 'skew': skew, 'kurt_excess': kurt_ex}


# ---------------- Main QQ plot (now supports fixed x/y limits) ----------------
def qq_compare_plot(
    test_data, cfd_data,
    n_q=None,
    alpha=0.05,
    acceptance=0.015,
    acceptance_mode='relative',  # 'relative' (%X) or 'absolute' (+/-8)
    annotate_stats=True,
    show_legend=True,
    title='Q-Q validation: CFD vs Test',
    xlim=None, ylim=None,
    ax=None
):
    xq, yq, _ = _quantiles_at_common_probs(test_data, cfd_data, n_q=n_q)

    # If xlim is given, it is used to draw the lines
    if xlim is not None:
        xg = np.linspace(xlim[0], xlim[1], 200)
    else:
        xg = np.linspace(np.min(xq), np.max(xq), 200)

    reg = _regression_with_ci(xq, yq, alpha=alpha, x_grid=xg)

    # Equality & acceptance band
    eq = xg
    if acceptance_mode == 'relative':
        lower_acc = xg * (1 - acceptance)
        upper_acc = xg * (1 + acceptance)
        acc_label = f'±{acceptance*100:.2f}% Acceptance Interval'
    elif acceptance_mode == 'absolute':
        lower_acc = xg - acceptance
        upper_acc = xg + acceptance
        acc_label = f'±{acceptance:g} Acceptance Interval'
    else:
        raise ValueError("acceptance_mode must be 'relative' or 'absolute'")

    ci_inside = np.all((reg['lo'] >= lower_acc) & (reg['hi'] <= upper_acc))
    stats_test = summarize_moments(test_data)
    stats_cfd = summarize_moments(cfd_data)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6), dpi=120)

    # Scatter of Q-Q points
    ax.plot(xq, yq, 'o', ms=3.5, mfc='none', mec='k', mew=0.5, label='Q-Q Data')
    # ax.plot(xq, yq, 'o', ms=3.5, mfc='none', mec='blue', label='Q-Q Data')
    # Regression + CI
    ax.plot(xg, reg['yhat_grid'], '-', color='gray', lw=1.0, label='Regression')
    ax.plot(xg, reg['lo'], '--', color='red', lw=0.5,
            label=f'{int((1-alpha)*100)}% CI')
    ax.plot(xg, reg['hi'], '--', color='red', lw=0.5)
    # Equality + acceptance
    ax.plot(xg, eq, '--', color='b', lw=1.0, label='Y = X')
    ax.plot(xg, lower_acc, '--', color='lime', lw=1.0, label=acc_label)
    ax.plot(xg, upper_acc, '--', color='lime', lw=1.0)

    ax.set_xlabel('Test (X)', fontsize=9)
    ax.set_ylabel('CFD (Y)', fontsize=9)
    ax.set_title(title, fontsize=9)
    ax.grid(True, alpha=0.25)

    # Make ranges similar
    if xlim is None:
        # xlim is not given it automatically takes the min and max from data
        vmin = min(ax.get_xlim()[0], ax.get_ylim()[0])
        vmax = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.set_xlim(vmin, vmax)
        ax.set_ylim(vmin, vmax)
    else:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim if ylim is not None else xlim)

    # Ratio of 1:1 to see the 45 degree line correctly
    ax.set_aspect('equal', adjustable='box')

    # Statistic box at the right bottom corner
    if annotate_stats:
        txt = (
            'Test:\n'
            f"μ={stats_test['mean']:.3f}, σ={stats_test['std']:.3f}, "
            f"skew={stats_test['skew']:.3f}, kurt(ex)={stats_test['kurt_excess']:.3f}\n"
            'CFD:\n'
            f"μ={stats_cfd['mean']:.3f}, σ={stats_cfd['std']:.3f}, "
            f"skew={stats_cfd['skew']:.3f}, kurt(ex)={stats_cfd['kurt_excess']:.3f}\n"
            f"R²(Q-Q)={reg['r2']:.3f}   "
            f"N_T={stats_test['n']}  N_C={stats_cfd['n']}  "
            f"Equiv: {'YES' if ci_inside else 'NO'}"
        )
        ax.text(0.98, 0.02, txt, transform=ax.transAxes,
                va='bottom', ha='right', fontsize=6,
                bbox=dict(boxstyle='round,pad=0.35', alpha=0.15))

    if show_legend:
        ax.legend(loc='best', fontsize=7)
    return ax


# ----------- Build a single 5x8 figure with identical limits -----------
def qq_grid_5x8(
    dfT, dfC, cols,
    pt0_wt, pt0_cfd,
    n_q=None,
    acceptance=0.015,
    acceptance_mode='relative',
    alpha=0.05,
    grid_title='Q-Q: CFD vs Test (All Probes)',
    out_path='QQ_grid_5x8.png',
    show=True,
    verbose=True,
    savefig_kwargs=None,
):
    # Pre-calculate global range
    global_min = np.inf
    global_max = -np.inf
    valid_flags = {}
    cleaned = {}

    for col in cols:
        if (col not in dfT.columns) or (col not in dfC.columns):
            valid_flags[col] = False
            continue
        x = dfT[col].to_numpy()
        y = dfC[col].to_numpy()
        x = x[np.isfinite(x)]
        y = y[np.isfinite(y)]

        x /= pt0_wt
        y /= pt0_cfd

        if len(x) < 10 or len(y) < 10:
            valid_flags[col] = False
            continue
        valid_flags[col] = True
        cleaned[col] = (x, y)

        xq, yq, _ = _quantiles_at_common_probs(x, y, n_q=n_q)
        global_min = min(global_min, np.min(xq), np.min(yq))
        global_max = max(global_max, np.max(xq), np.max(yq))

    if not np.isfinite(global_min) or not np.isfinite(global_max):
        raise RuntimeError('No valid columns found to plot.')

    # Add margin
    rng = global_max - global_min
    pad = 0.02 * rng if rng > 0 else 1e-6
    xlim = (global_min - pad, global_max + pad)

    # Create 5x8 subplot grid
    fig, axs = plt.subplots(5, 8, figsize=(13, 14), sharex=True, sharey=True)
    axs = axs.ravel()

    # Define the stats labels
    stats_labels = ['μ', 'σ', 'S', 'K']
    col_names = ['Test', 'CFD']

    for i, col in enumerate(cols):
        ax = axs[i]
        ax.set_aspect('equal', adjustable='box')

        if not valid_flags.get(col, False):
            ax.text(0.5, 0.5, f'{col}\n(no/insufficient data)',
                    ha='center', va='center', fontsize=8)
            ax.set_xlim(xlim); ax.set_ylim(xlim)
            ax.grid(True, alpha=0.2)
            continue

        x, y = cleaned[col]

        # Compute moments
        stats_test = summarize_moments(x)
        stats_cfd = summarize_moments(y)

        # Prepare data for table: 4 rows x 2 columns
        table_data = [
            [f"{stats_test['mean']:.3f}", f"{stats_cfd['mean']:.3f}"],
            [f"{stats_test['std']:.3f}", f"{stats_cfd['std']:.3f}"],
            [f"{stats_test['skew']:.3f}", f"{stats_cfd['skew']:.3f}"],
            [f"{stats_test['kurt_excess']:.3f}", f"{stats_cfd['kurt_excess']:.3f}"],
        ]

        # Create table in bottom-right corner
        table = ax.table(
            cellText=table_data,
            rowLabels=stats_labels,
            colLabels=col_names,
            cellLoc='center',
            loc='bottom',
            bbox=[0.55, 0.02, 0.42, 0.42]  # [left, bottom, width, height]
        )

        # Style the table
        table.set_zorder(100)  # Very high zorder -> on top
        table.auto_set_font_size(False)
        table.set_fontsize(5.5)
        table.scale(1.0, 1.9)  # Adjust row height

        # Style header (optional)
        for (row, coln), cell in table.get_celld().items():
            if row == 0:
                cell.set_text_props(weight='bold', color='black')
            if coln == 0:
                cell.set_text_props(weight='bold')
            cell.set_edgecolor('gray')

        # Plot Q-Q
        qq_compare_plot(
            x, y,
            n_q=n_q,
            alpha=alpha,
            acceptance=acceptance,
            acceptance_mode=acceptance_mode,
            annotate_stats=False,
            show_legend=False,
            title=col,
            xlim=xlim, ylim=xlim,
            ax=ax
        )

        # Remove x/y labels except on outer edges
        if i // 8 < 4:
            ax.set_xlabel('')
        if i % 8 != 0:
            ax.set_ylabel('')

    # Final title
    fig.suptitle(grid_title, fontsize=10, fontweight='bold')
    plt.subplots_adjust(
        left=0.05,   # Left margin
        right=0.95,  # Right margin
        bottom=0.05, # Bottom margin
        top=0.93,    # Top margin
        hspace=0.2,  # Vertical spacing between rows (default: 0.2)
        wspace=0.01  # Horizontal spacing between columns (default: 0.2)
    )
    plt.savefig(out_path, dpi=500, bbox_inches='tight')
    plt.close()
    print(f'[saved] {out_path}')
    return fig


# ---------------- Example usage ----------------
if __name__ == '__main__':
    # Define path for the data files
    # wdir = '//uds/Bolumler/313F0/VOLCANO/DATA/STATISTICS/Mach130_AoA0_AoS5/2014'
    # wdir = 'D:/Volcano_LES/Mach130_AoA0_AoS5/'
    # wdir = 'D:/TKE_spectrum/Mach130_AoA0_AoS5/'
    # wdir = 'D:/GE_PACKAGE_1_Ma130_AoA0_AoS5/2014/'
    wdir = 'D:/GE_PACKAGE_1_Ma130_AoA0_AoS5/5_SSEC/2014/'
    wt_runid = 'C1068Run0242T2014'
    cfd_runid = 'C1068Run091172014_CFD'
    os.chdir(wdir)

    parts = wt_runid.split('T')
    suffix = parts[1]
    wt_pressure_file = wt_runid + '.unsteady'
    wt_steady_file = wt_runid + '.steady'
    DES_pressure_file = cfd_runid + '.unsteady'
    DES_steady_file = cfd_runid + '.steady'

    # Read H0 (P_total_inf) from steady files
    wt_steady = pd.read_csv(wt_steady_file, sep='\t')
    cfd_steady = pd.read_csv(DES_steady_file, sep='\t')
    pt0_wt = wt_steady.at[0, 'H0']
    pt0_cfd = cfd_steady.at[0, 'H0']

    test_path = wt_pressure_file
    cfd_path = DES_pressure_file

    # dfT = pd.read_excel(test_path, sheet_name=0)
    # dfC = pd.read_excel(cfd_path, sheet_name=0)
    dfT = pd.read_csv(test_path, sep='\t')
    sampling_rate = 25000
    test_start = 0.0
    period = 4.5
    cfd_start = 0.00
    cfd_period = 4.5

    test_end = test_start + period
    test_time = test_end - test_start
    test_time = round(test_time, 2)
    test_start_sample = int(sampling_rate * test_start)
    test_end_sample = int(test_start_sample + sampling_rate * period)
    dfT = dfT.iloc[test_start_sample:test_end_sample].copy()

    cfd_start_sample = int(sampling_rate * cfd_start)
    cfd_end_sample = int(cfd_start_sample + sampling_rate * cfd_period)
    dfC = pd.read_csv(cfd_path, sep='\t')
    dfC = dfC.iloc[cfd_start_sample:cfd_end_sample].copy()
    cfd_time = len(dfC) / sampling_rate
    cfd_time = round(cfd_time, 2)

    # cols = [f'UHAD{k}f{i:02d}' for i in range(1, 9) for k in [1, 2, 3, 4, 5]]
    cols = []
    for k in range(1, 6):
        row = [f'UHAD{k}f{i:02d}' for i in range(1, 9)]
        cols.extend(row)

    # Generate the 8x5 figure with same range
    qq_grid_5x8(
        dfT, dfC, cols,
        pt0_wt=pt0_wt,
        pt0_cfd=pt0_cfd,
        n_q=None,                      # Automatic
        acceptance=0.015,             # +/-1.5%
        acceptance_mode='relative',
        alpha=0.05,
        # grid_title='Mach130_AoA0_AoS5 C1068Run0242T2016 Q-Q: WMLES1.0sec vs WT4.5sec',
        grid_title=f'Mach130_AoA0_AoS5 {wt_runid} Q-Q: WT{test_time}sec vs CFD{cfd_time}sec',
        out_path=f'{suffix}_WT{test_time}sec_CFD{cfd_time}sec_qq_grid_5x8.png'
    )
