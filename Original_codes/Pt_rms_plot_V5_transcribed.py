# -*- coding: utf-8 -*-
"""
Created on Thu Sep 11 23:46:27 2025

@author: Tusas

Transcription of the supplied Spyder screenshots (lines 1–370).
The original code's logic and active/commented status are retained.
See the companion transcription notes for potentially uncertain text.
"""

import os
import re
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import sys
sys.path.append('D:/Volcano_LES/up-todated_scripts')
from plot_utils_3 import plotfunction


# ==========================
# 0) Paths & load
# ==========================
# wdir = 'C:/Users/t26206/Desktop/Volcano_LES/2SEC'
# wdir = '//vds/Bolumler/J13F0/VOLCANO/DATA/STATISTICS/Mach130_AoA0_AoS5'
wdir = 'D:/GE_PACKAGE_1_Ma130_AoA0_AoS5/4_SSEC/2014/'
wt_runid = 'C10668Run024272014'
cfd_runid = 'C10668Run091172014_CFD'

sampling_rate = 25000
test_start = 0.0
period = 4.50
cfd_start = 0.00
cfd_period = 4.50

os.chdir(wdir)
parts = wt_runid.split('T')
suffix = parts[1]

TEST_FILE = wt_runid + '_unsteady'
wt_steady_file = wt_runid + '_steady'
CFD_FILE = cfd_runid + '_unsteady'
cfd_steady_file = cfd_runid + '_steady'
coords_file = 'ProbeCoordinates_WT.xlsx'

# retain dynamic pressure value from wt and cfd steady files
wt_steady = pd.read_csv(wt_steady_file, sep='\t')
q0_wt_value = wt_steady['Q0'].item()
cfd_steady = pd.read_csv(cfd_steady_file, sep='\t')
q0_cfd_value = cfd_steady['Q0'].item()

# retain free stream pressure value from wt and cfd steady files
h0_wt_value = wt_steady['H0'].item()
h0_cfd_value = cfd_steady['H0'].item()

coords_df = pd.read_excel(coords_file, sheet_name='centered')  # Columns: No, x, y, z
wt_pressure_df = pd.read_csv(TEST_FILE, sep='\t')
des_pressure_df = pd.read_csv(CFD_FILE, sep='\t')


test_end = test_start + period
test_time = test_end - test_start
test_time = round(test_time, 2)
test_start_sample = int(sampling_rate * test_start)
test_end_sample = int(test_start_sample + sampling_rate * period)
wt_pressure_df = wt_pressure_df.iloc[test_start_sample:test_end_sample].copy()  # WT data from 1 to 1.21 second

cfd_start_sample = int(sampling_rate * cfd_start)
cfd_end_sample = int(cfd_start_sample + sampling_rate * cfd_period)
dfc = des_pressure_df
dfc = dfc.iloc[cfd_start_sample:cfd_end_sample].copy()  # WT data from 1 to 1.21 second
cfd_time = len(dfc) / sampling_rate
cfd_time = round(cfd_time, 2)

# Rolling WT and CFD data: this is used where we are interested in engine stability.
# NOTE: Do not use this for the raw pressure statistics calculations and validations (Q-Q, Auto Corr, Cross Corr)
window_n = 16  # Mean average of 16 datapoints (rolling)
#window_n = 1   # Skip mean average rolling
mask = wt_pressure_df.columns.str.startswith(('UHAD', 'USAD'))
wt_rolled_data = wt_pressure_df.copy()
wt_rolled_data.loc[:, mask] = (wt_pressure_df.loc[:, mask].rolling(window=window_n, min_periods=1).mean())

cfd_rolled_data = des_pressure_df.copy()
cfd_rolled_data.loc[:, mask] = (des_pressure_df.loc[:, mask].rolling(window=window_n, min_periods=1).mean())


# =============================
# 1) UHAD columns & cleaning
# =============================
uhad_columns_test = [c for c in wt_rolled_data.columns if c.startswith('UHAD')]
uhad_columns_cfd = [c for c in cfd_rolled_data.columns if c.startswith('UHAD')]
# Common channels between Test/CFD, preserving the TEST channel order
uhad_columns = [c for c in uhad_columns_test if c in uhad_columns_cfd]

# Extract probe number from column headers: UHAD101 -> 101
_probe_num = lambda col: int(re.search(r'UHAD(\d+)', col).group(1))
probe_numbers = [_probe_num(c) for c in uhad_columns]  # 40 total probe list

# Total pressure dataframe with probe headers
pressure_df_clean_test = wt_rolled_data[uhad_columns].copy()
pressure_df_clean_cfd = cfd_rolled_data[uhad_columns].copy()
pressure_df_clean_test.columns = probe_numbers
pressure_df_clean_cfd.columns = probe_numbers

# ============================
# 2) Polar coordinates on AIP
# ============================
coords_df['No'] = coords_df['No'].astype(int)
# The probes which exist in the data with the original order
coords_df = coords_df.set_index('No').loc[probe_numbers].reset_index()

# radius on y-z plane
coords_df['r'] = np.sqrt(coords_df['y']**2 + coords_df['z']**2)

# Angle theta according to port convention
def assign_theta(port):
    # port 508 ... 102 101
    last_digit = int(str(port)[2])  # * 1..8
    base_angle = {1: 90, 2: 45, 3: 0, 4: 315, 5: 270, 6: 225, 7: 180, 8: 135}
    return np.deg2rad(base_angle.get(last_digit, 0))

coords_df['theta'] = coords_df['No'].apply(assign_theta)
# Normalize theta to [0, 2pi)
coords_df['theta'] = np.mod(coords_df['theta'], 2 * np.pi)

# Sort by theta for better interpolation
coords_df = coords_df.sort_values('theta').reset_index(drop=True)

# ============================
# 3) Interp grid (polar)
# ============================
radial_point = 100
angular_point = 73
r_grid = np.linspace(coords_df['r'].min(), coords_df['r'].max(), radial_point)
theta_grid = np.linspace(0, 2*np.pi, angular_point)
R, Theta = np.meshgrid(r_grid, theta_grid)

# Hub masking (optional)
hub_radius = coords_df['r'].min() * 0.7
hub_mask = R < hub_radius


def interpolate_snapshot(values):
    theta_vals = coords_df['theta'].values
    r_vals = coords_df['r'].values
    values = np.asarray(values)

    # Make the theta periodic
    extended_theta = np.concatenate([theta_vals, theta_vals + 2*np.pi])
    extended_r = np.concatenate([r_vals, r_vals])
    extended_values = np.concatenate([values, values])

    points = np.column_stack((extended_theta, extended_r))
    Z = griddata(points, extended_values, (Theta, R), method='cubic')
    Z[hub_mask] = np.nan
    return Z


# ============================
# 4) RMS per probe
# ============================
def compute_rms_per_probe(df: pd.DataFrame, demean: bool = True) -> pd.Series:
    df_num = df.apply(pd.to_numeric, errors='coerce')
    if demean:
        centered = df_num - df_num.mean(axis=0, skipna=True)
        rms = (centered.pow(2).mean(axis=0, skipna=True)).pow(0.5)
    else:
        rms = (df_num.pow(2).mean(axis=0, skipna=True)).pow(0.5)
    rms.name = 'RMS'
    return rms


# =================================================
# find required probes (High/Low RMS, Ring 3, Ring 5)
# =================================================
def ring_of(no: int) -> int:
    # 101 -> 1, 508 -> 5
    return int(str(no)[0])


def detect_probe_groups(rms: pd.Series,
                        q_low: float = 0.15,
                        q_high: float = 0.85) -> dict:
    ql = rms.quantile(q_low)
    qh = rms.quantile(q_high)
    groups = {
        'High RMS (top 15%)': rms[rms >= qh].index.astype(int).tolist(),
        'Low RMS (bottom 15%)': rms[rms <= ql].index.astype(int).tolist(),
        'Mid ring (k=3)': [no for no in rms.index.astype(int) if ring_of(no) == 3],
        'Near-wall ring (k=5)': [no for no in rms.index.astype(int) if ring_of(no) == 5],
    }
    return groups


def overlay_probe_groups(coords_df: pd.DataFrame,
                         groups: dict,
                         label_points: bool = False, ax=None):
    if ax is None:
        ax = plt.gca()

    coords = coords_df.set_index('No')[['theta', 'r']]

    allowed = ('High RMS (top 15%)', 'Low RMS (bottom 15%)')
    styles = {
        'High RMS (top 15%)': dict(marker='^', s=180, facecolors='none',
                                   edgecolors='k', linewidths=1.2, zorder=5),
        'Low RMS (bottom 15%)': dict(marker='v', s=160, facecolors='none',
                                      edgecolors='k', linewidths=1.2, zorder=5),
    }
    for name in allowed:
        nos = groups.get(name, [])
        if not nos:
            continue
        sub = coords.loc[[n for n in nos if n in coords.index]]
        if sub.empty:
            continue
        ax.scatter(sub['theta'].values, sub['r'].values, label=name, **styles[name])

        if label_points:
            for no, row in coords.iterrows():
                ax.text(row['theta'], row['r'], str(no), fontsize=7,
                        ha='center', va='center')

    # Legend
    ax.legend(loc='upper left', bbox_to_anchor=(-0.25, 1.05), frameon=True, prop={'size': 8})
    plt.gcf().subplots_adjust(left=0.28, right=0.95)


# ============================
# 6) Run: RMS -> contour -> highlight
# ============================
# define radii
radii = [0.015811, 0.027386, 0.035355, 0.041833, 0.047434]

# --- RMS Calculation & Non-dimensionalization (Pt_rms / Q0) ---
rms_test_raw = compute_rms_per_probe(pressure_df_clean_test).reindex(probe_numbers)
rms_test_nd = rms_test_raw / q0_wt_value  # Non-dimensionalize by WT Q0
rms_test_interpolated = interpolate_snapshot(rms_test_nd.values)

rms_cfd_raw = compute_rms_per_probe(pressure_df_clean_cfd).reindex(probe_numbers)
rms_cfd_nd = rms_cfd_raw / q0_cfd_value  # Non-dimensionalize by CFD Q0
rms_cfd_interpolated = interpolate_snapshot(rms_cfd_nd.values)

# # FIXED LIMITS for ND RMS
# rms_vmin = 0.0
# rms_vmax = 0.2

# # Calculate global min/max for ND RMS to synchronize colorbars
# rms_vmin = min(rms_test_nd.min(), rms_cfd_nd.min()) * 0.99
# rms_vmax = max(rms_test_nd.max(), rms_cfd_nd.max()) * 1.01

# Calculate global min/max for ND RMS to synchronize colorbars
rms_vmin = min(rms_test_nd.min(), rms_cfd_nd.min())
rms_vmax = max(rms_test_nd.max(), rms_cfd_nd.max())

# --- RMS Figure (Combined) ---
# IMPORTANT: Use subplot_kw to make the axes polar
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7), subplot_kw=dict(polar=True))

plotfunction(
    Z=rms_test_interpolated,
    Theta=Theta,
    R=R,
    coords_df=coords_df,
    radii=radii,
    cmap='jet',
    title=f'{suffix}_Non-dim RMS Pressure (WT) [{test_time}sec]',
    pad=30,
    cbar_label=r'$P_{rms} / Q_0$',
    vmin=rms_vmin,
    vmax=rms_vmax,
    nlevels=11,
    show_points=True,
    draw_guides=True,
    show=False,  # Must be False
    ax=ax1  # Pass the first polar axis
)

# ADDED: Detect and overlay for WT
groups_test = detect_probe_groups(rms_test_raw, q_low=0.15, q_high=0.85)
overlay_probe_groups(coords_df, groups_test, label_points=False, ax=ax1)

plotfunction(
    rms_cfd_interpolated,
    Theta,
    R,
    coords_df,
    radii,
    cmap='jet',
    title=f'{suffix}_Non-dim RMS Pressure (CFD) [{cfd_time}sec]',
    pad=30,
    cbar_label=r'$P_{rms} / Q_0$',
    vmin=rms_vmin,
    vmax=rms_vmax,
    nlevels=11,
    show=False,  # Must be False
    ax=ax2  # Pass the second polar axis
)

# ADDED: Detect and overlay for CFD
groups_cfd = detect_probe_groups(rms_cfd_raw, q_low=0.15, q_high=0.85)
overlay_probe_groups(coords_df, groups_cfd, label_points=False, ax=ax2)

plt.tight_layout()
plt.savefig(os.path.join(wdir, f'Pt_rms_ND_WT{test_time}sec_CFD{cfd_time}sec.png'), dpi=300, bbox_inches='tight')
plt.show()
plt.close()
print(f'✅ PR Plot saved in {wdir}')


# --- Mean Pressure Field & Non-dimensionalization (P_mean / H0) ---
# Calculate mean per probe first
mean_test_raw = pressure_df_clean_test.mean(axis=0).reindex(probe_numbers)
mean_test_nd = mean_test_raw / h0_wt_value  # Non-dimensionalize by WT H0
pressure_mean_test_interp = interpolate_snapshot(mean_test_nd.values)

mean_cfd_raw = pressure_df_clean_cfd.mean(axis=0).reindex(probe_numbers)
mean_cfd_nd = mean_cfd_raw / h0_cfd_value  # Non-dimensionalize by CFD H0
pressure_mean_cfd_interp = interpolate_snapshot(mean_cfd_nd.values)

# FIXED LIMITS for Pressure Recovery
mean_vmin = 0.8
mean_vmax = 1.0

# # Calculate global min/max for ND Mean Pressure
# mean_vmin = min(mean_test_nd.min(), mean_cfd_nd.min()) * 0.95
# mean_vmax = max(mean_test_nd.max(), mean_cfd_nd.max()) * 1.05

# IMPORTANT: Use subplot_kw to make the axes polar
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7), subplot_kw=dict(polar=True))

plotfunction(
    pressure_mean_test_interp, Theta, R, coords_df, radii,
    cmap='jet', title=f'{suffix}_Pressure Recovery (WT) [{test_time}sec]', pad=30,
    cbar_label=r'$P_{mean} / H_0$', vmin=mean_vmin, vmax=mean_vmax, nlevels=9,
    show=False,  # Must be False
    ax=ax1  # Pass the first polar axis
)

plotfunction(
    pressure_mean_cfd_interp, Theta, R, coords_df, radii,
    cmap='jet', title=f'{suffix}_Pressure Recovery (CFD) [{cfd_time}sec]', pad=30,
    cbar_label=r'$P_{mean} / H_0$', vmin=mean_vmin, vmax=mean_vmax, nlevels=9,
    show=False,  # Must be False
    ax=ax2  # Pass the second polar axis
)

plt.tight_layout()
plt.savefig(os.path.join(wdir, f'PR_avg_WT{test_time}sec_CFD{cfd_time}sec.png'), dpi=300, bbox_inches='tight')
plt.show()
plt.close()
print(f'✅ Pt_rms Plot saved in {wdir}')
