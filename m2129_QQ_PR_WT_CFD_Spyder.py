# -*- coding: utf-8 -*-
"""WT/CFD pressure-ratio Q-Q comparison; adapted from QQ_test_NEW_complete. Edit settings and run F5 in Spyder.

Dependencies: numpy, pandas, scipy, matplotlib.
Q-Q compares marginal distributions, not temporal correlation or spectra.
Regression and R² are descriptive. No inferential CI or equivalence verdict:
ordered quantiles are dependent, as may be the original pressure samples.
Rake_mean is the instantaneous arithmetic mean, not area weighted; it is
formed BEFORE quantiles. It is not the average of individual probe quantiles.
Raw-pressure differences include differences in operating conditions.
The CFD spatial static average and WT wall-pressure average are different
spatial measurements; their comparison requires physical justification.
"""
from pathlib import Path
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# ================= USER SETTINGS =================
DATA_DIR = Path(r"C:/Users/Tusas/Desktop/M2129")
WT_FILE = "WT_data.csv"       # Use None to skip WT
CFD_PT_FILE = "CFD_Pt_data.csv"
CFD_PS_FILE = "CFD_Ps_data.csv"
WT_TIME_WINDOW = (None, None)  # absolute start/end seconds, inclusive
CFD_TIME_WINDOW = (None, None)
# raw: Pa; normalized: pressure divided by supplied fixed reference;
# fluctuations: subtract each channel's own time mean (Pa).
# H0 is FREE-STREAM TOTAL PRESSURE, not temperature or enthalpy.
# Supply positive ABSOLUTE pressures in Pa, consistent with the input Pt.
# No estimated H0 is silently substituted. Replace None with your values.
H0_WT = 87100   # Pa, e.g. your measured WT reference total pressure
H0_CFD = 100556  # Pa, your CFD reference total pressure
COMPARISON_MODE = "normalized"  # PR = Pt / H0; keep normalized for this script
REFERENCE_PRESSURE_PA = {"WT": H0_WT, "CFD": H0_CFD}
MAKE_STATISTICS_SCATTER = True
MAKE_STATISTICS_CORRELATION = False  # optional: across the 11 physical probes
EXPORT_PR_SIGNALS = True
N_QUANTILES = 500  # capped at smaller sample count; no upsampling
MAKE_INDIVIDUAL_PLOTS = True
SHOW_FIGURES = True
COMMON_GRID_LIMITS = False  # True shares X ranges and Y ranges across panels
# Optional engineering tolerance; None disables it. No default equivalence claim.
ACCEPTANCE = 0.015  # original engineering band: +/-1.5%; None disables it
ACCEPTANCE_MODE = "relative"  # relative: fraction of abs(WT quantile)
OUTPUT_DIR = DATA_DIR / "QQ_PR_results"
DPI = 180
# =================================================
PT_NAMES = ([f"Pt_{i}_u" for i in range(1, 6)] + ["Pt_center"]
            + [f"Pt_{i}_l" for i in range(1, 6)])
PS_NAMES = ["Ps_1_u", "Ps_1_l", "Ps_1_left", "Ps_1_right"]


def read_csv(path):
    df = pd.read_csv(path, encoding="utf-8-sig", sep=",")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def time_mask(df, window):
    cols = [c for c in df if re.search(r"time\s*\(\s*s\s*\)", c, re.I)]
    if len(cols) != 1:
        raise ValueError(f"Expected one time column; found {cols}")
    t = pd.to_numeric(df[cols[0]], errors="raise").to_numpy(float)
    if len(t) < 2 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError("Time must be finite and strictly increasing.")
    mask = np.ones(len(t), dtype=bool)
    if window[0] is not None:
        mask &= t >= window[0]
    if window[1] is not None:
        mask &= t <= window[1]
    t = t[mask]
    if len(t) < 10:
        raise ValueError("Selected time window contains fewer than 10 samples.")
    dt = np.diff(t)
    if np.max(np.abs(dt-np.median(dt))) > .001*np.median(dt):
        raise ValueError("Nonuniform sampling: sample quantiles would not represent uniform time weighting.")
    return mask, dict(start_s=t[0], end_s=t[-1], n=len(t), fs_Hz=1/np.median(dt))


def extract(df, names, mask):
    out = {}
    for name in names:
        cols = [c for c in df if not re.search(r"time\s*\(\s*s\s*\)", c, re.I)
                and re.search(r"(?<![A-Za-z0-9_])"+re.escape(name)+r"(?![A-Za-z0-9_])", c, re.I)]
        if len(cols) != 1:
            raise ValueError(f"Expected one pressure column for {name}; found {cols}")
        out[name] = pd.to_numeric(df.loc[mask, cols[0]], errors="raise").to_numpy(float)
    data = pd.DataFrame(out)
    if not np.isfinite(data.to_numpy()).all():
        raise ValueError("Pressure contains NaN/Inf; no samples were silently discarded.")
    return data


def transform(a, case):
    a = np.asarray(a, dtype=float)
    if COMPARISON_MODE == "raw":
        return a.copy()
    if COMPARISON_MODE == "fluctuations":
        return a-a.mean()
    if COMPARISON_MODE == "normalized":
        ref = REFERENCE_PRESSURE_PA[case]
        if ref is None or not np.isfinite(ref) or ref <= 0:
            raise ValueError(f"Supply a positive fixed reference pressure for {case}.")
        return a/ref
    raise ValueError("COMPARISON_MODE must be raw, normalized, or fluctuations.")


def moments(a):
    varying = np.ptp(a) > 0
    return dict(n=len(a), min=np.min(a), max=np.max(a), mean=np.mean(a), std=np.std(a, ddof=1),
                skew=stats.skew(a, bias=False) if varying else np.nan,
                kurt_excess=stats.kurtosis(a, fisher=True, bias=False) if varying else np.nan)


def compare(x, y):
    n = min(int(N_QUANTILES), len(x), len(y))
    if n < 3:
        raise ValueError("At least three quantiles required.")
    p = (np.arange(n)+.5)/n
    xq, yq = np.quantile(x,p), np.quantile(y,p)
    if np.ptp(xq) == 0:
        slope, intercept, r2 = np.nan, np.nan, np.nan
    else:
        reg = stats.linregress(xq,yq)
        slope, intercept, r2 = reg.slope, reg.intercept, reg.rvalue**2
    within = np.nan
    if ACCEPTANCE is not None:
        if not np.isfinite(ACCEPTANCE) or ACCEPTANCE < 0:
            raise ValueError("ACCEPTANCE must be nonnegative and finite.")
        width = ACCEPTANCE*np.abs(xq) if ACCEPTANCE_MODE == "relative" else np.full(n,ACCEPTANCE)
        within = float(np.mean(np.abs(yq-xq) <= width)*100)
    return dict(p=p,xq=xq,yq=yq,slope=slope,intercept=intercept,r2=r2,
                quantile_rmse=np.sqrt(np.mean((yq-xq)**2)), within_band_pct=within, max_abs_quantile_error=float(np.max(np.abs(yq-xq))))


def limits(x,y):
    lo, hi = min(np.min(x),np.min(y)), max(np.max(x),np.max(y))
    pad = max((hi-lo)*.05, abs(lo)*1e-6, 1e-9)
    return lo-pad, hi+pad


def qq_limits(x, y):
    # WT and CFD may have very different operating pressures. Keep their
    # axis ranges separate so the distribution does not collapse to a dot.
    return limits(x, x), limits(y, y)


def draw(ax, name, x, y, q, bounds, unit, legend=False):
    xbounds, ybounds = bounds
    grid = np.linspace(*xbounds,200)
    ax.plot(q['xq'],q['yq'],'o',ms=3,color='tab:blue',alpha=.7,label='Q-Q quantiles',zorder=3)
    ax.plot(q['xq'][[0,-1]],q['yq'][[0,-1]],'^',color='tab:red',ms=5,
            label='Extreme plotted quantiles',zorder=4)
    overlap = (max(xbounds[0], ybounds[0]), min(xbounds[1], ybounds[1]))
    if overlap[0] <= overlap[1]:
        ax.plot(overlap,overlap,'b--',lw=1,label='Equality')
    else:
        ax.text(.03,.95,'Equality line outside displayed ranges',transform=ax.transAxes,
                va='top',fontsize=6.5,color='dimgray')
    if np.isfinite(q['slope']):
        # Fit shown only over observed WT quantiles, never extrapolated.
        gx = np.linspace(q['xq'].min(),q['xq'].max(),100)
        ax.plot(gx,q['intercept']+q['slope']*gx,color='gray',lw=1,label='Descriptive fit')
    if ACCEPTANCE is not None:
        width = ACCEPTANCE*np.abs(grid) if ACCEPTANCE_MODE == 'relative' else np.full(len(grid),ACCEPTANCE)
        ax.fill_between(grid,grid-width,grid+width,color='green',alpha=.12,label='Engineering tolerance')
    a,b=moments(x),moments(y)
    text = ('                 WT          CFD\n' + '\n'.join(
        f'{label:<10} {a[key]:.4g}    {b[key]:.4g}'
        for key,label in [('mean','Mean'),('std','Std'),('skew','Skew'),('kurt_excess','Kurt(ex)')])
        +f"\nN={len(x)} / {len(y)}; R²={q['r2']:.4g}")
    ax.text(.98,.02,text,transform=ax.transAxes,ha='right',va='bottom',fontsize=6.5,
            bbox=dict(facecolor='white',edgecolor='gray',alpha=.9))
    ax.set(xlim=xbounds,ylim=ybounds,title=name,xlabel=f'WT [{unit}]',ylabel=f'CFD [{unit}]')
    ax.set_aspect('auto')
    ax.ticklabel_format(style='plain',useOffset=False)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=.25)
    if legend:
        ax.legend(loc='upper left',bbox_to_anchor=(0,1.22),ncol=2,fontsize=7)


def save(fig,name):
    fig.savefig(OUTPUT_DIR/name,dpi=DPI,bbox_inches='tight')
    if not SHOW_FIGURES:
        plt.close(fig)


def statistics_scatter(summary):
    # Only physical probes are observations; Rake/static averages are excluded.
    frame=summary[summary.channel.isin(PT_NAMES)].copy()
    fig,axes=plt.subplots(2,3,figsize=(15,9))
    for ax,key,title in zip(axes.ravel(),['min','max','mean','std','skew','kurt_excess'],
                            ['Minimum PR','Maximum PR','Mean PR','PR standard deviation','Skewness','Excess kurtosis']):
        x=frame['WT_'+key].to_numpy(float);y=frame['CFD_'+key].to_numpy(float)
        valid=np.isfinite(x)&np.isfinite(y)
        if not valid.any():
            ax.text(.5,.5,'Undefined for constant signals',ha='center',transform=ax.transAxes)
            ax.set_title(title)
            continue
        xv,yv=x[valid],y[valid]
        ax.scatter(xv,yv,s=30,color='tab:blue')
        for a,b,label in zip(xv,yv,frame.channel.to_numpy()[valid]):
            ax.annotate(label,(a,b),xytext=(3,3),textcoords='offset points',fontsize=6)
        xb,yb=qq_limits(xv,yv)
        overlap=(max(xb[0],yb[0]),min(xb[1],yb[1]))
        if overlap[0]<=overlap[1]:ax.plot(overlap,overlap,'k--',lw=1)
        ax.set(xlim=xb,ylim=yb,xlabel='WT',ylabel='CFD',title=title)
        ax.ticklabel_format(style='plain',useOffset=False)
        ax.grid(alpha=.25)
    fig.suptitle('Statistics across 11 probes (each point is one probe)')
    fig.tight_layout(rect=(0,0,1,.96))
    save(fig,'PR_statistics_scatter.png')


def statistics_correlation(summary):
    frame=summary[summary.channel.isin(PT_NAMES)]
    columns=[case+'_'+key for case in ['WT','CFD']
             for key in ['min','max','mean','std','skew','kurt_excess']]
    corr=frame[columns].corr()
    corr.to_csv(OUTPUT_DIR/'PR_statistics_correlation.csv')
    fig,ax=plt.subplots(figsize=(12,10))
    im=ax.imshow(np.ma.masked_invalid(corr.to_numpy()),vmin=-1,vmax=1,cmap='coolwarm')
    ax.set_xticks(range(len(columns)));ax.set_xticklabels(columns,rotation=75,ha='right',fontsize=8)
    ax.set_yticks(range(len(columns)));ax.set_yticklabels(columns,fontsize=8)
    for i in range(len(columns)):
        for j in range(len(columns)):
            val=corr.iloc[i,j]
            if np.isfinite(val):ax.text(j,i,f'{val:.2f}',ha='center',va='center',fontsize=6)
    fig.colorbar(im,ax=ax,label='Pearson r across probes')
    ax.set_title('Descriptive correlation across 11 probes; no time-sample pairing')
    fig.tight_layout()
    save(fig,'PR_statistics_correlation.png')


def run_analysis():
    global REFERENCE_PRESSURE_PA
    REFERENCE_PRESSURE_PA = {"WT": H0_WT, "CFD": H0_CFD}
    if COMPARISON_MODE != "normalized":
        raise ValueError("This script calculates PR = Pt/H0. Keep COMPARISON_MODE='normalized'.")
    for case, h0 in REFERENCE_PRESSURE_PA.items():
        if h0 is None or not np.isscalar(h0) or not np.isfinite(h0) or h0 <= 0:
            raise ValueError(f"Enter H0_{case} as a positive absolute TOTAL PRESSURE in Pa in USER SETTINGS.")
    for filename in (WT_FILE, CFD_PT_FILE, CFD_PS_FILE):
        if not (DATA_DIR / filename).is_file():
            raise FileNotFoundError(f"CSV not found: {DATA_DIR / filename}. Edit DATA_DIR and filenames.")
    print(f"PR = Pt/H0; H0_WT={H0_WT:g} Pa; H0_CFD={H0_CFD:g} Pa")
    if ACCEPTANCE_MODE not in ['relative','absolute']:
        raise ValueError('ACCEPTANCE_MODE must be relative or absolute.')
    if ACCEPTANCE is not None and ACCEPTANCE_MODE == 'relative' and COMPARISON_MODE == 'fluctuations':
        warnings.warn('Relative bands collapse near zero; consider an absolute tolerance for fluctuations.')
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
    wt,pt,ps=[read_csv(DATA_DIR/f) for f in [WT_FILE,CFD_PT_FILE,CFD_PS_FILE]]
    wm,wi=time_mask(wt,WT_TIME_WINDOW)
    cm,ci=time_mask(pt,CFD_TIME_WINDOW)
    sm,si=time_mask(ps,CFD_TIME_WINDOW)
    w=extract(wt,PT_NAMES+PS_NAMES,wm)
    c=extract(pt,PT_NAMES,cm)
    w['Rake_mean']=w[PT_NAMES].mean(axis=1)
    c['Rake_mean']=c[PT_NAMES].mean(axis=1)
    wstatic=w[PS_NAMES].mean(axis=1).to_numpy()
    scols=[col for col in ps if 'ps_avg_aip_exp' in col.lower()]
    if len(scols)!=1:
        raise ValueError(f'Expected CFD static-pressure column; found {scols}')
    cstatic=pd.to_numeric(ps.loc[sm,scols[0]],errors='raise').to_numpy(float)
    if not np.isfinite(cstatic).all():
        raise ValueError('CFD static pressure contains NaN/Inf.')
    pairs={p:(w[p].to_numpy(),c[p].to_numpy()) for p in PT_NAMES+['Rake_mean']}
    pairs['Ps_reference']=(wstatic,cstatic)
    unit='P / H0 (dimensionless)'
    results={}
    rows=[]
    quantile_rows=[]
    for name,(x,y) in pairs.items():
        x,y=transform(x,'WT'),transform(y,'CFD')
        q=compare(x,y)
        results[name]=(x,y,q)
        row=dict(channel=name,mode=COMPARISON_MODE,slope=q['slope'],intercept=q['intercept'],
                 qq_r2=q['r2'],quantile_rmse=q['quantile_rmse'],within_band_pct=q['within_band_pct'],
                 max_abs_quantile_error=q['max_abs_quantile_error'],
                 H0_WT_Pa=H0_WT,H0_CFD_Pa=H0_CFD)
        for case,a in [('WT',x),('CFD',y)]:
            row.update({case+'_'+key:value for key,value in moments(a).items()})
        rows.append(row)
        quantile_rows.extend(dict(channel=name,probability=p,WT_quantile=a,CFD_quantile=b)
                             for p,a,b in zip(q['p'],q['xq'],q['yq']))
        if MAKE_INDIVIDUAL_PLOTS or name=='Ps_reference':
            fig,ax=plt.subplots(figsize=(7,7))
            draw(ax,name,x,y,q,qq_limits(q['xq'],q['yq']),unit,True)
            fig.suptitle('Pressure-ratio Q-Q comparison: P / H0')
            fig.tight_layout()
            save(fig,f'QQ_{name}.png')
    grid_names=PT_NAMES+['Rake_mean']
    allq=np.concatenate([results[p][2]['xq'] for p in grid_names])
    allc=np.concatenate([results[p][2]['yq'] for p in grid_names])
    bounds=qq_limits(allq,allc)
    fig,axs=plt.subplots(4,3,figsize=(15,19))
    for ax,name in zip(axs.ravel(),grid_names):
        x,y,q=results[name]
        draw(ax,name,x,y,q,bounds if COMMON_GRID_LIMITS else qq_limits(q['xq'],q['yq']),unit)
    fig.suptitle('WT vs CFD pressure ratios: Pt / H0\n11 probes and instantaneous Rake mean; independent axis ranges')
    fig.tight_layout(rect=(0,0,1,.96))
    save(fig,'QQ_all_probes_and_Rake_mean.png')
    summary=pd.DataFrame(rows)
    if EXPORT_PR_SIGNALS:
        # Keep each source's own time axis; Q-Q does not require time pairing.
        for frame, mask, values, source, h0 in [(wt,wm,w,'WT',H0_WT),(pt,cm,c,'CFD',H0_CFD)]:
            time_col=next(col for col in frame if re.search(r'time\s*\(\s*s\s*\)',col,re.I))
            normalized=values[PT_NAMES+['Rake_mean']].copy()/h0
            normalized.insert(0,'Time_s',pd.to_numeric(frame.loc[mask,time_col]).to_numpy(float))
            normalized.to_csv(OUTPUT_DIR/f'{source}_PR_signals.csv',index=False)
    if MAKE_STATISTICS_SCATTER:
        statistics_scatter(summary)
    if MAKE_STATISTICS_CORRELATION:
        statistics_correlation(summary)
    summary.to_csv(OUTPUT_DIR/'QQ_summary.csv',index=False)
    pd.DataFrame(quantile_rows).to_csv(OUTPUT_DIR/'QQ_quantiles.csv',index=False)
    metadata=pd.DataFrame([dict(source=s,**v) for s,v in [('WT',wi),('CFD_Pt',ci),('CFD_Ps',si)]])
    metadata['comparison_mode']=COMPARISON_MODE
    metadata['reference_pressure_Pa']=[REFERENCE_PRESSURE_PA['WT'],REFERENCE_PRESSURE_PA['CFD'],REFERENCE_PRESSURE_PA['CFD']]
    metadata['acceptance']=ACCEPTANCE
    metadata['acceptance_mode']=ACCEPTANCE_MODE
    metadata.to_csv(OUTPUT_DIR/'QQ_metadata.csv',index=False)
    print(summary[['channel','WT_mean','CFD_mean','qq_r2','within_band_pct']].to_string(index=False))
    print(f'Saved Q-Q figures and CSV tables: {OUTPUT_DIR.resolve()}')
    print('Descriptive distribution comparison; no equivalence verdict or independent-sample confidence interval.')
    if SHOW_FIGURES:
        plt.show()
    return results,summary,metadata


if __name__=='__main__':
    RESULTS,QQ_SUMMARY,ANALYSIS_METADATA=run_analysis()
