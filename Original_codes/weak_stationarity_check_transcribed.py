# -*- coding: utf-8 -*-
"""
Segmented Mean and Std Dev Analysis for Raw Pressure Data
Author: Koci Uzmani
Description: Reads unsteady files, divides raw pressure signals into segments,
             normalizes both mean and std dev (RMS) by H0 from steady files,
             and plots results with linear regression.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
import os
from scipy import stats

# ----------------------------------------
# 1. Configuration & Paths
# ----------------------------------------
# IMPORTANT: Check the two run IDs and folder path against the original files.
# wdir = 'D:/GE_PACKAGE_1_Ma130_AoA0_AoS5/2011/'
wdir = 'D:/GE_PACKAGE_1_Ma130_AoA0_AoS5/4_SSEC/2024/'
os.chdir(wdir)

wt_runid = 'C1060Run0242T2014'
cfd_runid = 'C1060Run0911T2014_CFD'
parts = wt_runid.split('T')
suffix = parts[1]

# Steady file paths for H0 extraction
wt_steady_file = wt_runid + '.steady'
cfd_steady_file = cfd_runid + '.steady'

sampling_rate = 25000  # Hz
fs = sampling_rate
dt = 1 / fs
segment_size = 1000

wt_start, wt_period = 0.0, 4.50
cfd_start, cfd_period = 0.0, 4.50

cases = {
    'WT': {
        'unsteady': wt_runid + '.unsteady',
        'label': f'WT-{wt_period}sec',
        'time_window': (wt_start, wt_period),
        'color': 'blue'
    },
    'CFD': {
        'unsteady': cfd_runid + '.unsteady',
        'label': f'CFD-{cfd_period}sec',
        'time_window': (cfd_start, cfd_period),
        'color': 'red'
    }
}

# ----------------------------------------
# 2. Helper Functions
# ----------------------------------------
def load_pressure_data(case_data):
    unsteady_file = case_data['unsteady']
    time_window = case_data['time_window']
    df = pd.read_csv(unsteady_file, sep='\t')

    start_sec, duration_sec = time_window
    start_sample = int(start_sec * fs)
    end_sample = int(start_sample + (duration_sec * fs))

    if end_sample > len(df):
        end_sample = len(df)

    df = df.iloc[start_sample:end_sample].copy().reset_index(drop=True)
    n_rows = len(df)
    time_vec = np.arange(start_sec, start_sec + n_rows * dt, dt)[:n_rows]
    pres_cols = [col for col in df.columns if col.startswith('UHAD')]

    pres_df = pd.DataFrame({'Time': time_vec})
    pres_df[pres_cols] = df[pres_cols]
    return pres_df


def calculate_linear_fit(x, y):
    """Calculates linear regression and returns slope, intercept, and R^2."""
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    return slope, intercept, r_value ** 2


# ----------------------------------------
# 3. H0 Extraction from Steady Files
# ----------------------------------------
print('📂 Reading H0 values from steady files...')
wt_steady = pd.read_csv(wt_steady_file, sep='\t')
cfd_steady = pd.read_csv(cfd_steady_file, sep='\t')

h0_values = {
    'WT': wt_steady.at[0, 'H0'],
    'CFD': cfd_steady.at[0, 'H0']
}
print(f"✅ H0 WT: {h0_values['WT']:.2f}, H0 CFD: {h0_values['CFD']:.2f}")

# ----------------------------------------
# 4. Main Execution & Plotting
# ----------------------------------------
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
col_map = {'WT': 0, 'CFD': 1}

for name, case_data in cases.items():
    print(f'🔹 Processing {name} pressure data...')
    pres_df = load_pressure_data(case_data)
    pres_cols = [col for col in pres_df.columns if col.startswith('UHAD')]

    combined_pres = pres_df[pres_cols].mean(axis=1).values
    time_vec = pres_df['Time'].values

    means, stds, segment_times = [], [], []

    for i in range(0, len(combined_pres), segment_size):
        segment = combined_pres[i:i + segment_size]
        if len(segment) < segment_size:
            break
        means.append(np.mean(segment))
        stds.append(np.std(segment))
        segment_times.append(time_vec[i + segment_size // 2])

    x = np.array(segment_times)

    # Normalize both Mean and Std Dev (RMS) by H0
    h0 = h0_values[name]
    y_mean = np.array(means) / h0
    y_std = np.array(stds) / h0

    col = col_map[name]

    # --- Plot Mean ---
    ax_m = axs[0, col]
    ax_m.plot(x, y_mean, color=case_data['color'], marker='o', markersize=4,
              linestyle='None', label=case_data['label'], alpha=0.5)

    m_slope, m_int, m_r2 = calculate_linear_fit(x, y_mean)
    fit_m = m_slope * x + m_int
    ax_m.plot(x, fit_m, color='black', linestyle='--', linewidth=1.5)
    ax_m.text(0.05, 0.9,
              f'y = {m_slope:.4e}x + {m_int:.4e}\nR² = {m_r2:.4f}',
              transform=ax_m.transAxes, verticalalignment='top',
              bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax_m.set_title(f'{name} - Normalized Mean Pressure')
    ax_m.set_ylabel('Mean Pressure / H0')
    ax_m.grid(True, ls='--', alpha=0.6)

    # --- Plot Std Dev (RMS) ---
    ax_s = axs[1, col]
    ax_s.plot(x, y_std, color=case_data['color'], marker='s', markersize=4,
              linestyle='None', label=case_data['label'], alpha=0.5)

    s_slope, s_int, s_r2 = calculate_linear_fit(x, y_std)
    fit_s = s_slope * x + s_int
    ax_s.plot(x, fit_s, color='black', linestyle='--', linewidth=1.5)
    ax_s.text(0.05, 0.9,
              f'y = {s_slope:.4e}x + {s_int:.4e}\nR² = {s_r2:.4f}',
              transform=ax_s.transAxes, verticalalignment='top',
              bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax_s.set_title(f'{name} - Normalized RMS Pressure')
    ax_s.set_ylabel("P'_rms / H0")
    ax_s.set_xlabel('Time (s)')
    ax_s.grid(True, ls='--', alpha=0.6)

plt.suptitle(f'T{suffix} Weak Static Stationarity Analysis (Segment Size: {segment_size})',
             fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

out_path = os.path.join(wdir,
                        f'T{suffix}_Weak_Static_Stationarity_Analysis_segment_size({segment_size}).png')
plt.savefig(out_path, dpi=300)
plt.show()
plt.close()
print(f'✅ Plot saved to: {out_path}')


# ----------------------------------------------------------------------
# 5. Segment Size Study
# ----------------------------------------------------------------------
print('\n🚀 Starting Segment Size Study...')
study_sizes = [100, 200, 500, 1000, 2000, 5000]
results = {
    'WT': {'mean_slopes': [], 'rms_slopes': []},
    'CFD': {'mean_slopes': [], 'rms_slopes': []}
}

# Pre-load data to avoid reading files repeatedly in the loop
data_cache = {}
for name, case_data in cases.items():
    df_loaded = load_pressure_data(case_data)
    pres_cols = [col for col in df_loaded.columns if col.startswith('UHAD')]
    data_cache[name] = {
        'combined': df_loaded[pres_cols].mean(axis=1).values,
        'time': df_loaded['Time'].values,
        'h0': h0_values[name],
        'color': case_data['color'],
        'label': case_data['label']
    }

for size in study_sizes:
    print(f'Analyzing segment size: {size}...')
    for name in ['WT', 'CFD']:
        cache = data_cache[name]
        combined_pres = cache['combined']
        time_vec = cache['time']

        means, stds, segment_times = [], [], []
        for i in range(0, len(combined_pres), size):
            segment = combined_pres[i:i + size]
            if len(segment) < size:
                break
            means.append(np.mean(segment))
            stds.append(np.std(segment))
            segment_times.append(time_vec[i + size // 2])

        x = np.array(segment_times)
        y_mean = np.array(means) / cache['h0']
        y_std = np.array(stds) / cache['h0']

        m_slope, _, _ = calculate_linear_fit(x, y_mean)
        s_slope, _, _ = calculate_linear_fit(x, y_std)

        results[name]['mean_slopes'].append(m_slope)
        results[name]['rms_slopes'].append(s_slope)

# Plotting the Study Results
fig_study, (ax_mean, ax_rms) = plt.subplots(2, 1, figsize=(10, 12))

for name in ['WT', 'CFD']:
    cache = data_cache[name]
    ax_mean.plot(study_sizes, results[name]['mean_slopes'], marker='o',
                 color=cache['color'], label=f'Mean - {name}')
    ax_rms.plot(study_sizes, results[name]['rms_slopes'], marker='s',
                color=cache['color'], label=f'RMS - {name}')

ax_mean.set_title('Effect of Segment Size on Mean Pressure Slopes')
ax_mean.set_ylabel('Slope')
ax_mean.set_xscale('log')  # Log scale often better for size studies
ax_mean.grid(True, ls='--', alpha=0.6)
ax_mean.legend()

ax_rms.set_title('Effect of Segment Size on RMS Pressure Slopes')
ax_rms.set_xlabel('Segment Size (samples)')
ax_rms.set_ylabel('Slope')
ax_rms.set_xscale('log')
ax_rms.grid(True, ls='--', alpha=0.6)
ax_rms.legend()

plt.tight_layout()
study_out_path = os.path.join(wdir, f'T{suffix}_Segment_Size_Study.png')
plt.savefig(study_out_path, dpi=300)
plt.show()
print(f'✅ Study plot saved to: {study_out_path}')
