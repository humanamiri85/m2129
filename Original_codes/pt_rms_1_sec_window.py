# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 20:35:43 2026

@author: t26206

Find the window with max RMS pressure on both WT and CFD data.

Transcribed from three photographed portions of the same script.
"""

import os
import re
import numpy as np
import pandas as pd

# ============================
# 0) Paths & Load
# ============================
# VERIFY AGAINST ORIGINAL: the path and run IDs are difficult to read clearly.
wdir = 'D:/GE_PACKAGE_1_Ma130_AoA0_Ao55/2016/'
wt_runid = 'C16069Run0242T2016'
cfd_runid = 'C16069Run0911T2016_CFD'
os.chdir(wdir)

TEST_FILE = wt_runid + '.unsteady'
CFD_FILE = cfd_runid + '.unsteady'

sampling_rate = 25000
window_n = 16  # Rolling average window
analysis_window_sec = 1.0  # Sliding window size in seconds
analysis_window_samples = int(sampling_rate * analysis_window_sec)


def process_unsteady_data(file_path, label):
    print(f"Processing {label}...")
    df = pd.read_csv(file_path, sep='\t')

    # 1. Filter only UHAD columns
    uhad_mask = df.columns.str.startswith('UHAD')
    uhad_data = df.loc[:, uhad_mask].copy()

    # 2. Apply Rolling Mean (Temporal smoothing)
    # We apply rolling mean to each probe independently first
    rolled_uhad = uhad_data.rolling(window=window_n, min_periods=1).mean()

    # 3. Spatial Average (Average of all UHAD channels at each time step)
    # This results in a single time series: P_avg(t)
    time_series_avg = rolled_uhad.mean(axis=1)

    # 4. Global RMS (over the whole period)
    # RMS = sqrt(mean((x - mean(x))**2))
    global_mean = time_series_avg.mean()
    global_rms = np.sqrt(((time_series_avg - global_mean)**2).mean())

    # 5. Sliding Window Analysis
    # We slide a 1-second window to find the maximum RMS
    max_rms = -1
    best_start_sample = 0
    best_end_sample = 0

    # Optimization: Use a loop to slide the window
    # To speed up: we can step by a few samples or use a rolling std
    # pandas .rolling().std() is mathematically equivalent to RMS of demeaned data
    # Original comment preserved above; see note about ddof in the response.
    rolling_std = time_series_avg.rolling(window=analysis_window_samples).std()

    # Find the index of the maximum value in the rolling standard deviation
    idx_max = rolling_std.idxmax()
    max_rms = rolling_std.max()

    # The window ends at idx_max and starts analysis_window_samples before it
    best_end_sample = idx_max
    best_start_sample = idx_max - analysis_window_samples + 1

    start_time = best_start_sample / sampling_rate
    end_time = best_end_sample / sampling_rate

    return {
        'global_rms': global_rms,
        'max_window_rms': max_rms,
        'start_time': start_time,
        'end_time': end_time,
        'time_series': time_series_avg
    }


# Execute for both datasets
wt_results = process_unsteady_data(TEST_FILE, "Wind Tunnel")
cfd_results = process_unsteady_data(CFD_FILE, "CFD")

# ============================
# Report Results
# ============================
# ... existing print statements ...

# ============================
# Plotting
# ============================
import matplotlib.pyplot as plt


def plot_max_rms_window(results, title, ax):
    # Create time axis
    time_axis = np.arange(len(results['time_series'])) / sampling_rate
    signal = results['time_series'].values

    # Plot the full signal in a neutral color
    ax.plot(time_axis, signal, color='gray', alpha=0.5, label='Full Signal', linewidth=0.8)

    # Extract the window for highlighting
    start_t = results['start_time']
    end_t = results['end_time']

    # Mask for the window
    mask = (time_axis >= start_t) & (time_axis <= end_t)

    # Plot the max RMS window in a bold color
    ax.plot(time_axis[mask], signal[mask], color='red', label='Max RMS Window', linewidth=1.2)

    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Avg UHAD Pressure")
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=False)

# The ends of these two lines were outside the visible photograph;
# the endings below are reconstructed from their visible beginning.
plot_max_rms_window(
    wt_results,
    f"WT: Spatial Avg Pressure (Max RMS Window: {wt_results['start_time']:.2f}s - {wt_results['end_time']:.2f}s)",
    ax1
)
plot_max_rms_window(
    cfd_results,
    f"CFD: Spatial Avg Pressure (Max RMS Window: {cfd_results['start_time']:.2f}s - {cfd_results['end_time']:.2f}s)",
    ax2
)

plt.tight_layout()
plt.savefig(os.path.join(wdir, f"Max_RMS_Window_{analysis_window_sec}sec.png"), dpi=300)
plt.show()
plt.close()
