# -*- coding: utf-8 -*-
"""Run this entire file with F5 in Spyder after editing USER SETTINGS.

Autocorrelation of 11 total-pressure probes and their instantaneous rack mean;
cross-correlation of all 55 unordered pairs, separately within WT and CFD.
No sample-to-sample pairing between WT and CFD, no equal-length truncation,
no pressure-to-velocity conversion, no inferred geometry or convection speed.

Definition: rho_mn(k) = sum p_m'(i)*p_n'(i+k) / ((N-|k|)*sigma_m*sigma_n).
Positive lag means n occurs later than m. Means and standard deviations are
computed over the whole selected record (ddof=0), as in the original script.
This overlap-corrected covariance estimator can exceed +/-1 at nonzero lag;
values are not clipped. N_overlap is not an independent effective sample size.
No Fisher confidence intervals or statistical equivalence verdict are used.
Delta_t = 2*first positive autocorrelation zero crossing is a descriptive
width, not a sample interval or an integral correlation time.

Dependencies: numpy, pandas, scipy, matplotlib. No plot_utils or seaborn needed.
"""
from pathlib import Path
from itertools import combinations
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import correlate

# ======================== USER SETTINGS ========================
DATA_DIR = Path(r"C:/Users/Tusas/Desktop/M2129")
WT_FILE = "WT_data.csv"       # Use None to skip WT
CFD_PT_FILE = "CFD_Pt_data.csv"
# CFD_Ps_data.csv is not required: this analysis uses total-pressure probes.
WT_TIME_WINDOW = (None, None)   # (start_s, end_s), inclusive; None = record edge
CFD_TIME_WINDOW = (None, None)
MAX_LAG_MS = 10.0              # REQUIRED: enter a positive value in milliseconds
# Example syntax only: MAX_LAG_MS = 10.0. Choose your own analysis range.
DELTA_ACCEPT = 0.05            # engineering band around WT; not a confidence band
MAKE_INDIVIDUAL_PLOTS = True
MAKE_CROSS_OVERVIEWS = True
SHOW_FIGURES = True           # show overview figures in Spyder's Plots pane
SHOW_INDIVIDUAL_FIGURES = False  # all individual PNGs are still saved
INCLUDE_RACK_MEAN_AUTO = True
OUTPUT_DIR = DATA_DIR / "time_space_correlation_results"
DPI = 180
TIME_UNIFORM_RTOL = 0.001     # numerical check: max dt deviation / median dt
# ===============================================================
PT_NAMES = ([f"Pt_{i}_u" for i in range(1, 6)] + ["Pt_center"]
            + [f"Pt_{i}_l" for i in range(1, 6)])


def read_source(path, window, label):
    if not path.is_file():
        raise FileNotFoundError(f"{label} CSV not found: {path}. Edit USER SETTINGS.")
    frame = pd.read_csv(path, sep=",", encoding="utf-8-sig")
    frame.columns = [str(c).replace('\u00a0', ' ').strip() for c in frame.columns]
    time_cols = [c for c in frame if re.search(r'time\s*\(\s*s\s*\)', c, re.I)]
    if len(time_cols) != 1:
        raise ValueError(f"{label}: expected one time column, found {time_cols}")
    time = pd.to_numeric(frame[time_cols[0]], errors='raise').to_numpy(float)
    if len(time) < 4 or not np.isfinite(time).all() or np.any(np.diff(time) <= 0):
        raise ValueError(f"{label}: time must be finite and strictly increasing, with >=4 samples.")
    if len(window) != 2:
        raise ValueError(f"{label}: window must be (start_s, end_s).")
    start, end = window
    for value in window:
        if value is not None and not np.isfinite(value):
            raise ValueError(f"{label}: window bounds must be finite or None.")
    if start is not None and end is not None and start > end:
        raise ValueError(f"{label}: window start exceeds end.")
    if ((start is not None and start < time[0]) or
            (end is not None and end > time[-1])):
        warnings.warn(f"{label}: requested window extends outside file; using available samples in that window.")
    mask = np.ones(len(time), dtype=bool)
    if start is not None:
        mask &= time >= start
    if end is not None:
        mask &= time <= end
    time = time[mask]
    if len(time) < 4:
        raise ValueError(f"{label}: fewer than four samples in selected window.")
    dt = float(np.median(np.diff(time)))
    deviation = float(np.max(np.abs(np.diff(time)-dt))/dt)
    if deviation > TIME_UNIFORM_RTOL:
        raise ValueError(f"{label}: time steps are nonuniform (relative deviation={deviation:g}); no resampling was applied.")
    data = {}
    for name in PT_NAMES:
        cols = [c for c in frame if c not in time_cols and
                re.search(r'(?<![A-Za-z0-9_])'+re.escape(name)+r'(?![A-Za-z0-9_])',c,re.I)]
        if len(cols) != 1:
            raise ValueError(f"{label}: expected one column for {name}, found {cols}")
        a = pd.to_numeric(frame.loc[mask,cols[0]], errors='raise').to_numpy(float)
        if not np.isfinite(a).all():
            raise ValueError(f"{label}/{name}: NaN or Inf; no filling or sample dropping was applied.")
        data[name] = a
    data = pd.DataFrame(data)
    if INCLUDE_RACK_MEAN_AUTO:
        data['Rack_mean'] = data[PT_NAMES].mean(axis=1)
    fs = 1.0/dt
    lag_samples = int(np.floor(MAX_LAG_MS*0.001*fs+1e-9))
    if lag_samples < 1:
        raise ValueError(f"{label}: MAX_LAG_MS is shorter than one sample ({dt*1000:g} ms).")
    if lag_samples > len(time)-2:
        raise ValueError(f"{label}: requested lag leaves fewer than two overlapping samples. Reduce MAX_LAG_MS or enlarge the time window.")
    if len(time)-lag_samples < len(time)*0.5:
        warnings.warn(f"{label}: fewer than half the samples overlap at the largest lag; tail estimates may be unstable.")
    meta = dict(source=label,n=len(time),start_s=time[0],end_s=time[-1],dt_s=dt,
                fs_Hz=fs,requested_max_lag_ms=MAX_LAG_MS,lag_samples=lag_samples,
                actual_max_lag_ms=1000*lag_samples/fs,min_overlap=len(time)-lag_samples,
                normalization='overlap_corrected_global_std',delta_accept=DELTA_ACCEPT)
    print(f"{label}: N={len(time):,}, fs={fs:g} Hz; window={time[0]:g}..{time[-1]:g} s; actual max lag={meta['actual_max_lag_ms']:g} ms")
    return data, meta


def correlation_curve(x, y, fs, lag_samples):
    x, y = np.asarray(x,float), np.asarray(y,float)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
        raise ValueError('Each within-source pair must contain equal-length 1D signals.')
    n = len(x)
    if not 1 <= lag_samples <= n-2:
        raise ValueError('Lag exceeds available record or is shorter than one sample.')
    k = np.arange(-lag_samples,lag_samples+1)
    overlap = n-np.abs(k)
    xm, ym = x-x.mean(), y-y.mean()
    sx, sy = np.std(xm), np.std(ym)
    if sx == 0 or sy == 0:
        rho = np.full(len(k),np.nan)
    else:
        # scipy correlate(y,x): positive lag means y follows x.
        full = correlate(ym,xm,mode='full',method='fft')
        cov = full[n-1-lag_samples:n+lag_samples]/overlap
        rho = cov/(sx*sy)
    return pd.DataFrame(dict(tau_s=k/fs,rho=rho,n_overlap=overlap))


def first_zero(curve):
    tau, rho = curve.tau_s.to_numpy(), curve.rho.to_numpy()
    positive = np.flatnonzero(tau >= 0)
    for i in positive[:-1]:
        a,b = rho[i:i+2]
        if np.isfinite(a) and np.isfinite(b) and a > 0 and b <= 0:
            return float(tau[i]+a/(a-b)*(tau[i+1]-tau[i]))
    return np.nan


def curve_summary(curve, source, kind, m, n):
    finite = np.isfinite(curve.rho.to_numpy())
    row = dict(source=source,kind=kind,probe_m=m,probe_n=n,
               status='ok' if finite.any() else 'undefined_constant_signal',
               rho_zero=np.nan,peak_abs_rho=np.nan,peak_abs_lag_s=np.nan,
               first_zero_s=np.nan,zero_width_s=np.nan)
    if finite.any():
        zero = int(np.argmin(np.abs(curve.tau_s.to_numpy())))
        peak = int(np.nanargmax(np.abs(curve.rho.to_numpy())))
        row.update(rho_zero=curve.rho.iloc[zero],peak_abs_rho=curve.rho.iloc[peak],
                   peak_abs_lag_s=curve.tau_s.iloc[peak])
        if kind == 'auto':
            z = first_zero(curve)
            row.update(first_zero_s=z,zero_width_s=2*z)
            if not np.isfinite(z):
                row['status'] = 'no_zero_crossing_in_selected_lags'
    return row


def draw(ax, curves, title, auto=False, small=False):
    for source,color in [('WT','tab:green'),('CFD','tab:blue')]:
        curve = curves[source]
        tau = 1000*curve.tau_s.to_numpy()
        rho = curve.rho.to_numpy()
        ax.plot(tau,rho,color=color,lw=1.3,label=source)
        if not np.isfinite(rho).any():
            ax.text(.02,.12 if source=='WT' else .04,f'{source}: constant signal; undefined',
                    transform=ax.transAxes,fontsize=6 if small else 9,color=color)
        if source == 'WT':
            ax.fill_between(tau,rho-DELTA_ACCEPT,rho+DELTA_ACCEPT,color=color,
                            alpha=.13,label=f'WT +/-{DELTA_ACCEPT:g} engineering band')
        if auto:
            z = first_zero(curve)
            if np.isfinite(z):
                ax.axvline(1000*z,color=color,ls=':',lw=.8)
                ax.axvline(-1000*z,color=color,ls=':',lw=.8)
                ax.text(.98,.96 if source=='WT' else .87,
                        f'{source}: 2*tau0={2000*z:.4g} ms',ha='right',va='top',
                        transform=ax.transAxes,color=color,fontsize=6 if small else 9)
    ax.axhline(0,color='.5',lw=.6)
    ax.axvline(0,color='.5',lw=.6)
    ax.set(title=title,xlabel='Lag [ms]',ylabel='Correlation')
    ax.grid(alpha=.25)
    ax.tick_params(labelsize=7 if small else 9)
    if not small:
        ax.legend(loc='best',fontsize=8)


def save_figure(fig,path,overview=False):
    fig.savefig(path,dpi=DPI,bbox_inches='tight')
    if not (SHOW_FIGURES and (overview or SHOW_INDIVIDUAL_FIGURES)):
        plt.close(fig)


def overview(items, path, title, auto=False):
    fig,axes = plt.subplots(4,3,figsize=(15,16))
    for ax,(name,curves) in zip(axes.ravel(),items):
        draw(ax,curves,name,auto=auto,small=True)
    for ax in axes.ravel()[len(items):]:
        ax.set_visible(False)
    handles,labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.965),ncol=3,fontsize=9)
    fig.suptitle(title,fontsize=14)
    fig.tight_layout(rect=(0,0,1,.94))
    save_figure(fig,path,overview=True)


def heatmaps(data, out):
    wt, cfd = data['WT'][PT_NAMES].corr(), data['CFD'][PT_NAMES].corr()
    delta = cfd-wt
    fig,axes = plt.subplots(1,3,figsize=(20,7))
    for ax,title,matrix in zip(axes,['WT','CFD','CFD - WT'],[wt,cfd,delta]):
        bound = 2 if title=='CFD - WT' else 1
        im = ax.imshow(np.ma.masked_invalid(matrix.to_numpy()),vmin=-bound,vmax=bound,cmap='coolwarm')
        ax.set_xticks(range(11)); ax.set_xticklabels(PT_NAMES,rotation=90,fontsize=7)
        ax.set_yticks(range(11)); ax.set_yticklabels(PT_NAMES,fontsize=7)
        ax.set_title(title)
        fig.colorbar(im,ax=ax,shrink=.7)
        matrix.to_csv(out/f'zero_lag_{title.replace(" - ","_minus_")}.csv')
    fig.suptitle('Zero-lag Pearson correlation within each source; gray/blank cells are undefined')
    fig.tight_layout(rect=(0,0,1,.95))
    save_figure(fig,out/'zero_lag_comparison.png',overview=True)
    return dict(WT=wt,CFD=cfd,difference=delta)


def run_analysis():
    if MAX_LAG_MS is None or not np.isscalar(MAX_LAG_MS) or not np.isfinite(MAX_LAG_MS) or MAX_LAG_MS <= 0:
        raise ValueError('Set MAX_LAG_MS in USER SETTINGS to your chosen positive lag in milliseconds.')
    if not np.isfinite(DELTA_ACCEPT) or DELTA_ACCEPT < 0:
        raise ValueError('DELTA_ACCEPT must be nonnegative and finite.')
    if SHOW_INDIVIDUAL_FIGURES:
        warnings.warn('Showing all individual figures can consume substantial memory; PNGs are saved regardless.')
    w,wm = read_source(DATA_DIR/WT_FILE,WT_TIME_WINDOW,'WT')
    c,cm = read_source(DATA_DIR/CFD_PT_FILE,CFD_TIME_WINDOW,'CFD')
    data, meta = dict(WT=w,CFD=c), dict(WT=wm,CFD=cm)
    for label,frame in data.items():
        for name in frame:
            if np.ptp(frame[name].to_numpy()) == 0:
                warnings.warn(f'{label}/{name}: constant signal; correlation is undefined and will be exported as NaN.')
    out = Path(OUTPUT_DIR)
    for sub in ['auto','cross','overview']:
        (out/sub).mkdir(parents=True,exist_ok=True)
    summaries=[]
    results=dict(auto={},cross={})
    pairs=list(combinations(PT_NAMES,2))
    assert len(pairs)==55
    jobs=[('auto',p,p) for p in w.columns]+[('cross',m,n) for m,n in pairs]
    for number,(kind,m,n) in enumerate(jobs,1):
        key=m if kind=='auto' else f'{m}__{n}'
        curves={}
        for label,frame in data.items():
            info=meta[label]
            curve=correlation_curve(frame[m].to_numpy(),frame[n].to_numpy(),info['fs_Hz'],info['lag_samples'])
            curves[label]=curve
            summaries.append(curve_summary(curve,label,kind,m,n))
        results[kind][key]=curves
        # Long format preserves each source's own lag grid and overlap counts.
        pd.concat([curve.assign(source=label) for label,curve in curves.items()],ignore_index=True).to_csv(
            out/kind/f'{kind}_{key}.csv',index=False)
        if MAKE_INDIVIDUAL_PLOTS:
            fig,ax=plt.subplots(figsize=(8,5))
            draw(ax,curves,('Autocorrelation: '+m) if kind=='auto' else f'Cross-correlation: {m} -> {n}',auto=kind=='auto')
            fig.tight_layout()
            save_figure(fig,out/kind/f'{kind}_{key}.png')
        print(f'[{number}/{len(jobs)}] {kind}: {key}')
    auto_items=list(results['auto'].items())
    overview(auto_items,out/'overview'/'auto_all_probes_and_rack_mean.png','Autocorrelation: WT vs CFD',auto=True)
    if MAKE_CROSS_OVERVIEWS:
        cross_items=list(results['cross'].items())
        for offset in range(0,len(cross_items),12):
            page=offset//12+1
            overview(cross_items[offset:offset+12],out/'overview'/f'cross_pairs_page_{page}.png',
                     f'Cross-correlation: all 55 pairs, page {page}/5; positive lag: second probe later')
    matrices=heatmaps(data,out)
    summary=pd.DataFrame(summaries)
    metadata=pd.DataFrame(list(meta.values()))
    summary.to_csv(out/'correlation_summary.csv',index=False)
    metadata.to_csv(out/'analysis_metadata.csv',index=False)
    (out/'analysis_notes.txt').write_text(__doc__+'\n\nSettings:\n'+
        f'WT window: {WT_TIME_WINDOW}\nCFD window: {CFD_TIME_WINDOW}\nMAX_LAG_MS: {MAX_LAG_MS}\n'+
        f'DELTA_ACCEPT: {DELTA_ACCEPT}\nWT file: {DATA_DIR/WT_FILE}\nCFD file: {DATA_DIR/CFD_PT_FILE}\n',encoding='utf-8')
    print(f'Done: {len(results["auto"])} auto channels and 55 cross pairs. Results: {out.resolve()}')
    print('N_overlap is overlap count, not independent sample size. No confidence interval or equivalence verdict.')
    if SHOW_FIGURES:
        plt.show()
    return results,summary,metadata,matrices


if __name__ == '__main__':
    RESULTS, CORRELATION_SUMMARY, ANALYSIS_METADATA, ZERO_LAG_MATRICES = run_analysis()