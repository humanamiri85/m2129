# -*- coding: utf-8 -*-
"""
TKE Spectrum Analysis for WT, DES, and LES
Author: Köd Uzmanı
Description: Full implementation to load unsteady/steady files, compute velocity,
             calculate 1D TKE spectra, and plot with Kolmogorov slope.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
from scipy.signal import welch
import os

# -------------------------------
# 1. Configuration & Paths
# -------------------------------
# VERIFY: Folder spelling in the photo is not fully legible.
wdir = 'D:/GE_PACKAGE_1_Mat30_AoAO_AoSS/4_SSEC/2014/'
os.chdir(wdir)

# VERIFY: Run IDs should be checked against the original files.
wt_runid = 'C1060Run024272014'
cfd_runid = 'C1060Run092172014_CFD'
parts = wt_runid.split('7')
suffix = parts[1]

# Constants
sampling_rate = 25000  # Hz
fs = sampling_rate
dt = 1 / fs
gamma = 1.4
R = 287.15
n_window = 2048

# Time Window Settings
wt_start = 0.0
wt_period = 4.50
cfd_start = 0.0
cfd_period = 4.50

# File Definitions
wt_pressure_file = wt_runid + '.unsteady'
wt_steady_file = wt_runid + '.steady'
cfd_pressure_file = cfd_runid + '.unsteady'
cfd_steady_file = cfd_runid + '.steady'

# Calculate total durations for labels
wt_total_len = len(pd.read_csv(wt_pressure_file, sep='\t'))
wt_total_time = round(wt_total_len / sampling_rate, 2)

cfd_total_len = len(pd.read_csv(cfd_pressure_file, sep='\t'))
cfd_total_time = round(cfd_total_len / sampling_rate, 2)

# Case Dictionary
cases = {
    'WT': {
        'unsteady': wt_pressure_file,
        'steady': wt_steady_file,
        'label': f'WT-{wt_period}sec',
        'time_window': (wt_start, wt_period)
    },
    'CFD': {
        'unsteady': cfd_pressure_file,
        'steady': cfd_steady_file,
        'label': f'CFD-{cfd_period}sec',
        'time_window': (cfd_start, cfd_period)
    }
}

# -----------------------------------
# 2. Processing Function
# -----------------------------------
def load_and_process_case(case_data):
    """
    Loads data, computes velocity from pressure ratios, and applies time window.
    """
    unsteady_file = case_data['unsteady']
    steady_file = case_data['steady']
    time_window = case_data['time_window']

    # Read steady file for T_inf
    steady_df = pd.read_csv(steady_file, sep='\t')
    t_inf = steady_df.at[0, 'T30']
    c = np.sqrt(gamma * R * t_inf)

    # Read unsteady data
    unsteady_df = pd.read_csv(unsteady_file, sep='\t')

    # Apply time window slicing
    start_sec, duration_sec = time_window
    start_sample = int(start_sec * fs)
    end_sample = int(start_sample + (duration_sec * fs))

    if end_sample > len(unsteady_df):
        end_sample = len(unsteady_df)

    unsteady_df = unsteady_df.iloc[start_sample:end_sample].copy().reset_index(drop=True)

    # Create time vector
    n_rows = len(unsteady_df)
    time_vec = np.arange(start_sec, start_sec + n_rows * dt, dt)[:n_rows]

    # Identify columns
    uhad_cols = [col for col in unsteady_df.columns if col.startswith('UHAD')]
    usad_cols = [col for col in unsteady_df.columns if col.startswith('USAD')]

    vel_df = pd.DataFrame({'Time': time_vec})

    # Average USAD for the pressure ratio denominator
    usad_avg = unsteady_df[usad_cols].mean(axis=1)
    vel_df['USAD_avg'] = usad_avg

    # Velocity calculation: M = sqrt((2/(gamma-1)) * ((P/P_avg)^((gamma-1)/gamma) - 1))
    for col in uhad_cols:
        p_ratio = unsteady_df[col] / vel_df['USAD_avg']
        m_sq = (2 / (gamma - 1)) * ((p_ratio ** ((gamma - 1) / gamma)) - 1)
        m_sq = np.clip(m_sq, 0, None)  # Remove negative values from noise
        vel_df[f'{col}_vel'] = np.sqrt(m_sq) * c

    return vel_df, t_inf, c

# -----------------------------------
# 3. Main Execution & Analysis
# -----------------------------------
results = {}
print('🔎 Processing cases...')

for name, case_data in cases.items():
    print(f'  ➜ Processing {name}...')
    vel_df, t_inf, c = load_and_process_case(case_data)
    results[name] = {
        'vel_df': vel_df,
        't_inf': t_inf,
        'c': c,
        'label': case_data['label']
    }

# Plotting
fig, ax1 = plt.subplots(figsize=(12, 7))
all_U_means = []
all_kappas = []
all_E_kappas = []

for name, res in results.items():
    vel_df = res['vel_df']
    vel_cols = [col for col in vel_df.columns if col.endswith('_vel')]

    case_E_kappas = []
    case_U_means = []

    for col in vel_cols:
        u = vel_df[col].values
        u_mean = np.mean(u)
        case_U_means.append(u_mean)

        # Fluctuating component
        u_prime = u - u_mean

        # PSD using Welch
        f, psd = welch(u_prime, fs=fs, nperseg=n_window, scaling='density')

        # Filter DC
        mask = f > 0
        f, psd = f[mask], psd[mask]

        # Convert to Wavenumber (kappa) and Energy Spectrum (E_kappa)
        # kappa = 2*pi*f / U_mean
        # E_kappa = 0.5 * PSD * (U_mean / 2*pi)
        kappa = (2 * np.pi * f) / u_mean
        E_kappa = 0.5 * psd * (u_mean / (2 * np.pi))

        case_E_kappas.append(E_kappa)

    # Average across all probes for this case
    avg_E_kappa = np.mean(case_E_kappas, axis=0)
    avg_U_mean = np.mean(case_U_means)

    # Store for reference line and secondary axis
    all_U_means.append(avg_U_mean)
    all_kappas.append((2 * np.pi * f[0]) / avg_U_mean)  # just a sample
    all_E_kappas.append(avg_E_kappa)

    # Plotting the result
    # We need a consistent kappa axis for the plot
    final_kappa = (2 * np.pi * f) / avg_U_mean
    ax1.loglog(final_kappa, avg_E_kappa, label=res['label'], linewidth=2)

# -----------------------------------
# 4. Final Touches & Export
# -----------------------------------
# Kolmogorov -5/3 Slope
u_global_avg = np.mean(all_U_means)
# Use the last case's data to draw the slope
ref_kappa = (2 * np.pi * f) / all_U_means[-1]
ref_e = all_E_kappas[-1]
# Anchor the slope to the middle of the spectrum
mid_idx = len(ref_e) // 4
slope = ref_e[mid_idx] * (ref_kappa / ref_kappa[mid_idx]) ** (-5 / 3)
ax1.loglog(ref_kappa, slope, 'k--', label='Kolmogorov -5/3', linewidth=2)

ax1.set_xlabel('Wavenumber $\\kappa$ (rad/m)', fontsize=12)
ax1.set_ylabel('Energy Spectrum $E(\\kappa)$ (m^3/s^2)', fontsize=12)
ax1.grid(True, which='both', ls='--', alpha=0.5)
ax1.legend()

# Secondary X-axis for Frequency
def k_to_f(k): return k * u_global_avg / (2 * np.pi)
def f_to_k(f): return f * 2 * np.pi / u_global_avg
ax2 = ax1.secondary_xaxis('top', functions=(k_to_f, f_to_k))
ax2.set_xlabel(f'Frequency (Hz) (Avg U = {u_global_avg:.2f} m/s)')

plt.title(f'{suffix} 1-D TKE Spectrum Comparison WT({wt_period})sec_CFD({cfd_period})sec', fontsize=14)
plt.tight_layout()
plt.show()

# Save and Print
out_path = os.path.join(wdir, f'{suffix}_1-D_TKE_WT({wt_period})sec_CFD({cfd_period})sec.png')
plt.savefig(out_path, dpi=300)
# plt.close()
print(f'✅ Plot saved to: {out_path}')

print('\n--- Final Summary ---')
for i, name in enumerate(results.keys()):
    print(f'{name}: Avg Velocity = {all_U_means[i]:.2f} m/s')

# # Optional: Print summary
# print('\n📋 Summary of Cases:')
# for name, res in results.items():
#     print(f"  {name}: t_inf = {res['t_inf']:.2f} K, c = {res['c']:.2f} m/s, Avg U = {np.mean(all_U_means):.2f} m/s")


# # ---------------------------------------
# # 🔍 NEW SECTION: Automatic Inertial Subrange Detection & Normalized Energy Fraction
# # ---------------------------------------
# print("\n🔍 Detecting Inertial Subrange (Kolmogorov -5/3 Scaling) & Computing Energy Fraction...")
# print("  ➜ Automatically identifying wavenumber range with -5/3 slope.")

# # Define parameters for detection
# min_k_points = 5           # Minimum number of points in fit window
# max_window_size = 100      # Max window size (in index) for sliding fit
# slope_target = -5/3       # Target slope: -1.6667
# tolerance_slope = 0.3     # Acceptable deviation from -5/3
# min_r2 = 0.7              # Minimum R² for valid fit
# kappa_min = 1.0           # Minimum wavenumber to consider (avoid noise)
# kappa_max = 1e4           # Maximum wavenumber (avoid high-frequency noise)

# # Store results
# inertial_results = {}

# for name, res in results.items():
#     kappa = all_kappas[list(results.keys()).index(name)]
#     E_kappa = all_E_kappas[list(results.keys()).index(name)]
#     label = res['label']

#     # Filter out very low/high k values
#     valid_mask = (kappa >= kappa_min) & (kappa <= kappa_max)
#     kappa_filtered = kappa[valid_mask]
#     E_kappa_filtered = E_kappa[valid_mask]

#     if len(kappa_filtered) < min_k_points:
#         print(f"⚠️ Not enough data points in valid range for {name}. Skipping inertial subrange detection.")
#         inertial_results[name] = {
#             'k_min': np.nan, 'k_max': np.nan,
#             'slope': np.nan, 'r2': np.nan,
#             'integral_inertial': 0.0, 'fraction': 0.0,
#             'total_energy': np.trapz(E_kappa_filtered, x=kappa_filtered)
#         }
#         continue

#     # Convert to log space
#     log_k = np.log10(kappa_filtered)
#     log_E = np.log10(E_kappa_filtered)

#     best_slope = None
#     best_r2 = -1
#     best_k_min_idx = 0
#     best_k_max_idx = len(log_k) - 1

#     # Sliding window over log-k space
#     for i in range(len(log_k) - min_k_points + 1):
#         for j in range(i + min_k_points, min(i + max_window_size, len(log_k))):
#             k_window = log_k[i:j+1]
#             E_window = log_E[i:j+1]

#             # Linear fit: log(E) = slope * log(k) + intercept
#             coeffs = np.polyfit(k_window, E_window, 1)
#             slope = coeffs[0]
#             intercept = coeffs[1]

#             # Predicted values
#             E_pred = np.polyval(coeffs, k_window)
#             ss_res = np.sum((E_window - E_pred) ** 2)
#             ss_tot = np.sum((E_window - np.mean(E_window)) ** 2)
#             r2 = 1 - (ss_res / ss_tot)

#             # Check if slope is close to -5/3 and R² is high
#             if abs(slope - slope_target) < tolerance_slope and r2 > min_r2:
#                 if r2 > best_r2:
#                     best_r2 = r2
#                     best_slope = slope
#                     best_k_min_idx = i
#                     best_k_max_idx = j

#     # If no valid region found
#     if best_slope is None:
#         print(f"❌ No inertial subrange detected for {name} (no fit within tolerance).")
#         inertial_results[name] = {
#             'k_min': np.nan, 'k_max': np.nan,
#             'slope': np.nan, 'r2': np.nan,
#             'integral_inertial': 0.0, 'fraction': 0.0,
#             'total_energy': np.trapz(E_kappa_filtered, x=kappa_filtered)
#         }
#         continue

#     # Extract best inertial subrange
#     k_min_inertial = kappa_filtered[best_k_min_idx]
#     k_max_inertial = kappa_filtered[best_k_max_idx]

#     # Compute integral over inertial subrange
#     inertial_mask = (kappa >= k_min_inertial) & (kappa <= k_max_inertial)
#     E_inertial = E_kappa[inertial_mask]
#     kappa_inertial = kappa[inertial_mask]
#     integral_inertial = np.trapezoid(E_inertial, x=kappa_inertial)  # ✅ Correct & future-safe

#     # Total energy in valid range
#     total_energy = np.trapezoid(E_kappa_filtered, x=kappa_filtered)

#     # Fraction of total energy
#     fraction = (integral_inertial / total_energy) * 100 if total_energy > 0 else 0.0

#     # Store result
#     inertial_results[name] = {
#         'k_min': k_min_inertial,
#         'k_max': k_max_inertial,
#         'slope': best_slope,
#         'r2': best_r2,
#         'integral_inertial': integral_inertial,
#         'fraction': fraction,
#         'total_energy': total_energy
#     }

#     # Print summary
#     print(f"  ➜ {name}: "
#           f"Range [{k_min_inertial:.1f}, {k_max_inertial:.1f}] 1/m, "
#           f"Slope = {best_slope:.3f} (target: -1.667), R² = {best_r2:.3f}, "
#           f"Energy fraction = {fraction:.2f}%")

# # ---------------------------------------
# # 📊 Summary of Inertial Subrange Results
# # ---------------------------------------
# print("\n" + "-"*75)
# print("📊 INERTIAL SUBRANGE DETECTION & ENERGY FRACTION SUMMARY")
# print("-"*75)
# print(f"{'Case':<12} {'k_min (1/m)':<12} {'k_max (1/m)':<12} {'Slope':<10} {'R²':<8} {'Energy Fraction (%)':<18}")
# print("-" * 75)
# for name, data in inertial_results.items():
#     print(f"{name:<12} {data['k_min']:<12.1f} {data['k_max']:<12.1f} "
#           f"{data['slope']:<10.3f} {data['r2']:<8.3f} "
#           f"{data['fraction']:<18.2f}")
# print("-"*75)

# # Optional: Save to CSV
# csv_file_inertial = os.path.join(wdir, "TKE_Inertial_Subrange_Analysis.csv")
# inertial_df = pd.DataFrame([
#     {
#         'case': name,
#         'k_min_inertial_1m': data['k_min'],
#         'k_max_inertial_1m': data['k_max'],
#         'slope': data['slope'],
#         'r2': data['r2'],
#         'integral_inertial_m3s2': data['integral_inertial'],
#         'fraction_energy_percent': data['fraction'],
#         'total_energy_valid_range_m3s2': data['total_energy']
#     }
#     for name, data in inertial_results.items()
# ])
# inertial_df.to_csv(csv_file_inertial, index=False)
# print(f"✅ Inertial subrange results saved to: {csv_file_inertial}")


# # -----------------------------------------------
# # 🟡 NEW SECTION: Dedicated Plot with Color-Per-Case Inertial Subrange Regions
# # -----------------------------------------------
# print("\n🟡 Creating a dedicated plot with color-per-case inertial subrange regions...")

# # Define a consistent color palette (useful for publication)
# colors = {
#     'WT': '#1f77b4',        # Blue
#     'DES': '#ff7f0e',       # Orange
#     'LES': '#2ca02c',       # Green
#     'DES_cutcell': '#d62728'  # Red
# }

# # Create a new figure
# fig, ax = plt.subplots(figsize=(14, 8))

# # Plot each case with its own color
# for name, res in results.items():
#     kappa = all_kappas[list(results.keys()).index(name)]
#     E_kappa = all_E_kappas[list(results.keys()).index(name)]
#     label = res['label']
#     color = colors.get(name, 'gray')  # Fallback to gray if not in dict

#     # Plot full spectrum in light version of the color (light grayish)
#     ax.loglog(kappa, E_kappa, color=color, alpha=0.4, linewidth=1.2, label=f"{label} (full)")

#     # Get detected inertial subrange
#     data = inertial_results[name]
#     if np.isnan(data['k_min']) or np.isnan(data['k_max']):
#         continue

#     k_min = data['k_min']
#     k_max = data['k_max']
#     idx = np.where((kappa >= k_min) & (kappa <= k_max))[0]
#     if len(idx) == 0:
#         continue

#     # Extract E(k) in inertial range
#     E_inertial = E_kappa[idx]
#     kappa_inertial = kappa[idx]

#     # Shade the inertial subrange with a **gold fill**, but **match the case color for border**
#     ax.fill_between(kappa_inertial, 0, E_inertial,
#                     color='gold', alpha=0.5, edgecolor=color, linewidth=2.0,
#                     label=f"{label} (inertial subrange)")

#     # Add text label inside the region (in dark red for contrast)
#     ax.text(k_max * 0.7, E_inertial[-1] * 0.6, f"{name}",
#             fontsize=10, color='darkred', ha='center', va='center',
#             bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))

# # Add reference -5/3 Kolmogorov line (black dashed)
# u_avg_avg = np.mean(all_U_means)
# idx_ref = len(kappa) // 4
# k_ref = kappa[idx_ref]
# E_ref = all_E_kappas[list(results.keys()).index(name)][idx_ref]
# kolmogorov_line = E_ref * (kappa / k_ref)**(-5/3)
# ax.loglog(kappa, kolmogorov_line, 'k--', linewidth=2, alpha=0.8, label='-5/3 Kolmogorov Slope')

# # Formatting
# ax.set_xlabel('Wavenumber (1/m)', fontsize=12)
# ax.set_ylabel('1D Energy Spectrum (m³/s²)', fontsize=12)
# ax.grid(True, which='both', ls='--', alpha=0.5)
# ax.legend(loc='lower left', fontsize=10, frameon=True, fancybox=True, shadow=True)

# # Secondary x-axis: Frequency (Hz)
# def k_to_f(k):
#     return k * U_avg_avg / (2 * np.pi)

# def f_to_k(f):
#     return f * 2 * np.pi / U_avg_avg

# ax2 = ax.secondary_xaxis('top', functions=(k_to_f, f_to_k))
# ax2.set_xlabel(f'Frequency (Hz) - based on avg U = {U_avg_avg:.1f} m/s', fontsize=12)

# plt.title('🧭 Detected Inertial Subrange Regions (Color-Per-Case)\n'
#           '↪ Gold fill + colored border = detected region\n'
#           '↪ Dashed line = -5/3 reference | Red text = case label',
#           fontsize=14, pad=20)
# plt.tight_layout()

# # Save the plot
# out_png = os.path.join(wdir, "TKE_Inertial_Subrange_Detected_Regions_ColorPerCase.png")
# plt.savefig(out_png, dpi=300, bbox_inches='tight')
# plt.close()

# print(f"✅ Color-per-case inertial subrange visualization saved as '{out_png}'")
