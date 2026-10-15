"""Pressure-derived, one-component velocity energy spectra (not full 3-D TKE).

Spyder usage: edit USER SETTINGS below, then press F5.
Dependencies: numpy, pandas, scipy, matplotlib.
RESULTS, PROBE_SUMMARY and GROUP_SUMMARY remain in Variable Explorer.

Assumptions: pressures in Pa, absolute after adding the specified offsets;
uniform static temperature 20 C by default; isentropic subsonic pressure-to-speed
relation; wall-static average represents rake static pressure. The inferred speed
fluctuations are not independently measured axial velocity fluctuations.
Taylor conversion uses a single arithmetic rake mean speed for each case.
Group averages are equal-probe averages, not area/mass-flow weighted averages.
"""
from pathlib import Path
from types import SimpleNamespace
import json
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib
# Keep Spyder's configured graphics backend.
import matplotlib.pyplot as plt
from scipy.signal import welch

# %% USER SETTINGS - edit here, then press F5 in Spyder
DATA_DIR = Path(r"C:/Users/Tusas/Desktop/M2129")
WT_FILE = "WT_data.csv"       # Use None to skip WT
CFD_PT_FILE = "CFD_Pt_data.csv"   # Set both CFD files to None to skip CFD
CFD_PS_FILE = "CFD_Ps_data.csv"
OUTPUT_DIR = DATA_DIR / "spectrum_results"
WT_TEMPERATURE_C = 20.0
CFD_TEMPERATURE_C = 20.0
TEMPERATURE_TYPE = "static"      # "static" or "total"
WT_PRESSURE_OFFSET_PA = 0.0      # 0 assumes absolute pressures in Pa
CFD_PRESSURE_OFFSET_PA = 0.0     # For gauge pressures: enter reference pressure
WT_START_S, WT_END_S = None, None    # None means use the full available record
CFD_START_S, CFD_END_S = None, None  # End times are exclusive
NPERSEG = 2048
SHOW_PLOTS = True

# %% Analysis functions
PT = [f'Pt_{i}_u' for i in range(1, 6)] + ['Pt_center'] + [f'Pt_{i}_l' for i in range(1, 6)]
PS = ['Ps_1_u', 'Ps_1_l', 'Ps_1_left', 'Ps_1_right']
GROUPS = {'upper': PT[:5], 'lower': PT[6:], 'all': PT}
GAMMA, R = 1.4, 287.15


def read_pressure(path, kind):
    df = pd.read_csv(path, encoding='utf-8-sig')
    df.columns = [str(c).strip() for c in df.columns]
    time_cols = [c for c in df if 'time' in c.lower()]
    if len(time_cols) != 1:
        raise ValueError(f'{path}: expected exactly one time column, found {time_cols}')
    names = {time_cols[0]: 'time'}
    if kind in ('wt', 'pt'):
        for probe in PT + (PS if kind == 'wt' else []):
            matches = [c for c in df if c not in time_cols and re.search(r'(?<!\w)' + re.escape(probe) + r'(?!\w)', c, re.I)]
            if len(matches) != 1:
                raise ValueError(f'{path}: expected one column for {probe}, found {matches}')
            names[matches[0]] = probe
    else:
        cols = [c for c in df if c != time_cols[0] and not c.lower().startswith('unnamed')]
        if len(cols) != 1:
            raise ValueError(f'{path}: expected one CFD mean-static column, found {cols}')
        names[cols[0]] = 'Ps_ref'
    df = df[list(names)].rename(columns=names).apply(pd.to_numeric, errors='raise')
    if not np.isfinite(df.to_numpy()).all():
        raise ValueError(f'{path}: missing or nonfinite values; clean them before spectral analysis')
    if len(df) < 8 or np.any(np.diff(df.time) <= 0):
        raise ValueError(f'{path}: need at least 8 rows and strictly increasing unique times')
    return df


def spectrum_case(name, frame, args, out):
    start, end = getattr(args, name.lower() + '_start'), getattr(args, name.lower() + '_end')
    frame = frame.loc[(frame.time >= start) & (frame.time < end)].copy()
    if len(frame) < 8:
        raise ValueError(f'{name}: fewer than 8 samples in selected [start, end) interval')
    t = frame.time.to_numpy()
    intervals = np.diff(t)
    dt = float(np.median(intervals))
    if not np.allclose(intervals, dt, rtol=1e-4, atol=1e-10):
        raise ValueError(f'{name}: nonuniform sampling; no automatic resampling is performed')
    fs = 1 / dt
    offset = getattr(args, name.lower() + '_offset_pa')
    ps = frame.Ps_ref.to_numpy() + offset
    pt = frame[PT].to_numpy() + offset
    if np.any(ps <= 0) or np.any(pt <= 0):
        raise ValueError(f'{name}: nonpositive absolute pressure; check units and reference offset')
    bad = pt < ps[:, None]
    if np.any(bad):
        counts = dict(zip(PT, bad.sum(axis=0).tolist()))
        raise ValueError(f'{name}: Pt < Ps samples: {counts}. Check static reference/calibration; no clipping applied.')
    temp_c = getattr(args, name.lower() + '_temp_c')
    temperature = temp_c + 273.15
    if temperature <= 0:
        raise ValueError('Temperature must exceed absolute zero')
    m2 = 2 / (GAMMA - 1) * ((pt / ps[:, None]) ** ((GAMMA - 1) / GAMMA) - 1)
    if args.temperature_type == 'total':
        velocity = np.sqrt(GAMMA * R * temperature * m2 / (1 + (GAMMA - 1) / 2 * m2))
    else:
        velocity = np.sqrt(GAMMA * R * temperature * m2)
    mean = velocity.mean(axis=0)
    if np.any(mean <= 0):
        raise ValueError(f'{name}: zero mean speed prevents Taylor conversion')
    if np.any(m2 >= 1):
        warnings.warn(f'{name}: sonic/supersonic inferred samples; verify Pitot/shock relation applicability')
    fluctuations = velocity - mean
    nperseg = min(args.nperseg, len(t))
    segments = 1 + (len(t) - nperseg) // (nperseg - nperseg // 2)
    if segments < 4:
        warnings.warn(f'{name}: only {segments} Welch segments; spectrum has limited averaging')
    freq, psd = welch(fluctuations, fs=fs, window='hann', nperseg=nperseg,
                      noverlap=nperseg // 2, detrend='constant', scaling='density', axis=0)
    energy_f = 0.5 * psd
    # Includes DC and Nyquist bins; Welch variance can differ from record variance
    # because of segment detrending and Hann weighting.
    energy_integral = energy_f.sum(axis=0) * (fs / nperseg)
    u_ref = float(mean.mean())
    kappa = 2 * np.pi * freq / u_ref
    energy_k = energy_f * u_ref / (2 * np.pi)
    columns = PT.copy()
    for members in GROUPS.values():
        indices = [PT.index(p) for p in members]
        energy_f = np.column_stack([energy_f, energy_f[:, indices].mean(axis=1)])
        energy_k = np.column_stack([energy_k, energy_k[:, indices].mean(axis=1)])
    columns += ['mean_upper', 'mean_lower', 'mean_all']
    pd.DataFrame(energy_f, columns=columns).assign(frequency_Hz=freq).to_csv(out / f'{name}_energy_frequency.csv', index=False)
    pd.DataFrame(energy_k, columns=columns).assign(kappa_rad_per_m=kappa).to_csv(out / f'{name}_energy_wavenumber.csv', index=False)
    pd.DataFrame({'time_s': t, **{p: velocity[:, i] for i, p in enumerate(PT)}}).to_csv(out / f'{name}_velocity.csv', index=False)
    rows = []
    for i, probe in enumerate(PT):
        variance = float(np.mean(fluctuations[:, i] ** 2))
        rows.append(dict(case=name, probe=probe, mean_speed_m_s=mean[i], rms_speed_m_s=np.sqrt(variance),
                         fluctuation_percent=100 * np.sqrt(variance) / mean[i],
                         one_component_energy_m2_s2=0.5 * variance,
                         welch_energy_m2_s2=energy_integral[i]))
    metadata = dict(samples=len(t), first_time_s=t[0], last_time_s=t[-1], fs_Hz=fs,
                    nperseg=nperseg, welch_segments=segments, frequency_bin_Hz=fs/nperseg,
                    temperature_C=temp_c, taylor_speed_m_s=u_ref,
                    mean_static_pressure_Pa=float(ps.mean()))
    return dict(f=freq, k=kappa, ef=energy_f, ek=energy_k, columns=columns, rows=rows, metadata=metadata)


def plot_comparison(results, out, domain):
    xkey, ykey = ('f', 'ef') if domain == 'frequency' else ('k', 'ek')
    xlabel = 'Frequency (Hz)' if domain == 'frequency' else r'Wavenumber $\kappa$ (rad/m)'
    ylabel = r'$E_f$ (m$^2$/s$^2$/Hz)' if domain == 'frequency' else r'$E_\kappa$ (m$^3$/s$^2$)'
    fig, axes = plt.subplots(4, 3, figsize=(15, 15))
    for i, ax in enumerate(axes.flat):
        if i == 11:
            ax.axis('off')
            continue
        for name, res in results.items():
            ax.loglog(res[xkey][1:], res[ykey][1:, i], label=name, lw=1.2)
        ax.set(title=PT[i], xlabel=xlabel, ylabel=ylabel)
        ax.grid(True, which='both', alpha=0.25)
        ax.legend()
    fig.suptitle('Pressure-derived one-component energy: individual probes')
    fig.tight_layout()
    fig.savefig(out / f'probes_{domain}.png', dpi=180)
    if not SHOW_PLOTS:
        plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for j, (ax, group) in enumerate(zip(axes, GROUPS)):
        for name, res in results.items():
            x, y = res[xkey][1:], res[ykey][1:, 11+j]
            ax.loglog(x, y, label=name, lw=1.5)
        # Reference guide only: this is not inertial-range detection.
        valid = (x > 0) & (y > 0)
        if valid.sum() >= 2:
            xx, yy = x[valid], y[valid]
            idx = len(xx) // 3
            guide_x = np.geomspace(xx[idx], xx[-1], 50)
            ax.loglog(guide_x, yy[idx] * (guide_x / xx[idx]) ** (-5/3), 'k--', label='-5/3 guide')
        ax.set(title=f'Equal-probe mean: {group}', xlabel=xlabel, ylabel=ylabel)
        ax.grid(True, which='both', alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out / f'mean_spectra_{domain}.png', dpi=180)
    if not SHOW_PLOTS:
        plt.close(fig)


def main():
    def input_path(filename):
        return None if filename is None else str(DATA_DIR / filename)

    args = SimpleNamespace(
        wt=input_path(WT_FILE), cfd_pt=input_path(CFD_PT_FILE),
        cfd_ps=input_path(CFD_PS_FILE), out=str(OUTPUT_DIR), nperseg=NPERSEG,
        temperature_type=TEMPERATURE_TYPE,
        wt_start=-float('inf') if WT_START_S is None else WT_START_S,
        wt_end=float('inf') if WT_END_S is None else WT_END_S,
        cfd_start=-float('inf') if CFD_START_S is None else CFD_START_S,
        cfd_end=float('inf') if CFD_END_S is None else CFD_END_S,
        wt_temp_c=WT_TEMPERATURE_C, cfd_temp_c=CFD_TEMPERATURE_C,
        wt_offset_pa=WT_PRESSURE_OFFSET_PA, cfd_offset_pa=CFD_PRESSURE_OFFSET_PA)
    if args.nperseg < 8:
        raise ValueError('NPERSEG must be at least 8')
    if args.temperature_type not in ('static', 'total'):
        raise ValueError('TEMPERATURE_TYPE must be static or total')
    if bool(args.cfd_pt) != bool(args.cfd_ps) or not (args.wt or args.cfd_pt):
        raise ValueError('Specify WT_FILE and/or both CFD files')
    for case in ('wt', 'cfd'):
        if getattr(args, case + '_start') >= getattr(args, case + '_end'):
            raise ValueError(f'{case}: start time must be less than end time')
    for filename in (args.wt, args.cfd_pt, args.cfd_ps):
        if filename and not Path(filename).is_file():
            raise FileNotFoundError(f'Input file not found: {filename}. Edit USER SETTINGS.')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    if args.wt:
        wt = read_pressure(args.wt, 'wt')
        wt['Ps_ref'] = wt[PS].mean(axis=1)
        results['WT'] = spectrum_case('WT', wt, args, out)
    if args.cfd_pt:
        pt, ps = read_pressure(args.cfd_pt, 'pt'), read_pressure(args.cfd_ps, 'ps')
        if len(pt) != len(ps) or not np.allclose(pt.time, ps.time, rtol=0, atol=1e-10):
            raise ValueError('CFD Pt/Ps time arrays differ; align the exported monitors before analysis')
        pt['Ps_ref'] = ps.Ps_ref.to_numpy()
        results['CFD'] = spectrum_case('CFD', pt, args, out)
    rows = [row for result in results.values() for row in result['rows']]
    pd.DataFrame(rows).to_csv(out / 'probe_summary.csv', index=False)
    group_rows = []
    for name, res in results.items():
        for group, members in GROUPS.items():
            selected = [row for row in res['rows'] if row['probe'] in members]
            group_rows.append(dict(case=name, group=group,
                mean_speed_m_s=float(np.mean([r['mean_speed_m_s'] for r in selected])),
                mean_one_component_energy_m2_s2=float(np.mean([r['one_component_energy_m2_s2'] for r in selected]))))
    pd.DataFrame(group_rows).to_csv(out / 'group_summary.csv', index=False)
    metadata = {'assumptions': __doc__, 'cases': {name: res['metadata'] for name, res in results.items()}}
    (out / 'analysis_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    for domain in ['frequency', 'wavenumber']:
        plot_comparison(results, out, domain)
    print(pd.DataFrame(group_rows).to_string(index=False))
    print(f'Results saved in {out.resolve()}')
    if SHOW_PLOTS:
        plt.show()
    return results, pd.DataFrame(rows), pd.DataFrame(group_rows)


if __name__ == '__main__':
    RESULTS, PROBE_SUMMARY, GROUP_SUMMARY = main()
