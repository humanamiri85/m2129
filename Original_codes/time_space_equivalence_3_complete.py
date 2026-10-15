# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 20:22:58 2025

@author: Tusas
"""

# time_space_equivalence.py
import re
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Set backend to non-interactive to prevent console plotting
import matplotlib.pyplot as plt
import seaborn as sns  # Add this import at the top
from scipy.interpolate import griddata
from plot_utils import plotfunction

# ---------- UHAD column names generated programmatically ----------
RINGS = range(1, 6)  # 1..5
POS = range(1, 9)  # 01..08

# Order: 101,201,301,401,501, then 102,202,...
UHAD_COLS: List[str] = [f"UHAD{k}{i:02d}" for i in POS for k in RINGS]

# -------- Small utilities --------
UHAD_RE = re.compile(r"^UHAD(?P<ring>[1-5])(?P<pos>\d{2})$")


def parse_uhad(name: str) -> Tuple[int, int]:
    """'UHAD101' -> (ring=1, pos=1)"""
    m = UHAD_RE.fullmatch(name)
    if not m:
        raise ValueError(f"Bad UHAD name: {name}")
    return int(m.group("ring")), int(m.group("pos"))


def same_rake_pairs() -> List[Tuple[str, str]]:
    """Pairs on the same rake at neighboring radial rings."""
    return [
        (f"UHAD{r}{p:02d}", f"UHAD{r+1}{p:02d}")
        for p in POS for r in RINGS if r < max(RINGS)
        if f"UHAD{r}{p:02d}" in UHAD_COLS and f"UHAD{r+1}{p:02d}" in UHAD_COLS
    ]


def same_ring_adjacent_pos_pairs(ring: int) -> List[Tuple[str, str]]:
    """Adjacent rake positions, including wrap from 08 to 01."""
    Pmax = max(POS)
    return [
        (f"UHAD{ring}{p:02d}", f"UHAD{ring}{((p % Pmax) + 1):02d}")
        for p in POS
        if f"UHAD{ring}{p:02d}" in UHAD_COLS
        and f"UHAD{ring}{((p % Pmax) + 1):02d}" in UHAD_COLS
    ]

# ---------- Core math ----------
def _corr_function(x: np.ndarray, y: np.ndarray, fs: float, max_lag_s: float
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Unbiased covariance at each lag, normalized by sqrt(Cxx(0) Cyy(0)).
    Returns tau [s], rho(tau), N_eff(tau)=N-|k|.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same length")
    N = x.size
    L = int(round(max_lag_s * fs))
    xm = x - x.mean()
    ym = y - y.mean()

    c_full = np.correlate(xm, ym, mode="full")
    center = N - 1
    k = np.arange(-L, L + 1)
    idx = slice(center - L, center + L + 1)
    c = c_full[idx]
    N_eff = N - np.abs(k)
    cov = c / N_eff

    sx = np.sqrt((xm @ xm) / N)
    sy = np.sqrt((ym @ ym) / N)
    denom = sx * sy if sx > 0 and sy > 0 else np.nan
    rho = cov / denom
    tau_s = k / fs
    return tau_s, rho, N_eff


from scipy.stats import norm


def fisher_ci(rho: np.ndarray, N_eff: np.ndarray, alpha: float = 0.05
              ) -> Tuple[np.ndarray, np.ndarray]:
    """(1-alpha) Fisher-z CI per lag using overlap N_eff = N-|k|."""
    r_clip = np.clip(rho, -0.999999, 0.999999)
    z = np.arctanh(r_clip)
    ok = N_eff > 3
    z_se = np.zeros_like(rho, float)
    z_se[ok] = 1.0 / np.sqrt(N_eff[ok] - 3.0)
    zcrit = norm.ppf(1 - alpha / 2.0)
    lo = np.tanh(z - zcrit * z_se); lo[~ok] = np.nan
    hi = np.tanh(z + zcrit * z_se); hi[~ok] = np.nan
    return lo, hi


def _delta_t_zero_from_autocorr(tau_s: np.ndarray,
                                rho: np.ndarray) -> Tuple[float, float]:
    if tau_s.ndim != 1 or rho.ndim != 1 or len(tau_s) != len(rho):
        raise ValueError("tau_s and rho must be 1D arrays of equal length")

    zero = int(np.argmin(np.abs(tau_s)))  # tau = 0 reference

    # Search to the right of tau=0 for first zero crossing
    for i in range(zero, len(rho) - 1):
        r0, r1 = rho[i], rho[i + 1]
        if not (np.isfinite(r0) and np.isfinite(r1)):
            continue
        if r0 == 0.0:
            tau0 = tau_s[i]
            break
        elif (r0 > 0.0 and r1 < 0.0) or (r0 >= 0.0 and r1 <= 0.0):
            # Linear interpolation at the zero crossing
            denom = (r0 - r1)
            if denom == 0.0:
                tau0 = tau_s[i + 1]
            else:
                alpha = r0 / denom
                tau0 = tau_s[i] + alpha * (tau_s[i + 1] - tau_s[i])
            break
    else:
        return np.nan, np.nan

    if tau0 > 0.0:
        return 2.0 * tau0, tau0
    return np.nan, np.nan


def _annotate_dt(ax,
                 tau_s: np.ndarray,
                 rho: np.ndarray,
                 color: str = "k"):
    """Draw the symmetric interval (-tau0, +tau0) and label Delta_t = 2*tau0."""
    Delta_t, tau0 = _delta_t_zero_from_autocorr(tau_s, rho)
    if not np.isfinite(Delta_t):
        return  # No zero crossing found inside the lag range

    # Place near top of plot
    y_top = np.nanmax(rho) if np.isfinite(np.nanmax(rho)) else 1.0
    y_lev = 0.70 * y_top

    tau0 = abs(tau0)
    ax.hlines(y=y_lev, xmin=-tau0, xmax=+tau0, colors=color,
              linestyles="--", linewidth=1.4)
    ax.vlines([-tau0, +tau0], ymin=y_lev*0.97, ymax=y_lev*1.03,
              colors=color, linewidth=1.1)

    x_text = 0.55 * tau0
    ax.annotate(rf"$\Delta t = 2\tau_0 \approx {Delta_t:.2f}\,\mathrm{{s}}$",
                xy=(x_text, y_lev + 0.10*y_top),
                ha="left", va="bottom", fontsize=10, color=color)


@dataclass
class CorrComparison:
    tau_s: np.ndarray
    rho_test: np.ndarray
    rho_cfd: np.ndarray
    ci_lo_cfd: np.ndarray
    ci_hi_cfd: np.ndarray
    acc_lo_test: np.ndarray
    acc_hi_test: np.ndarray


def corr_compare_series(x_test: np.ndarray, y_test: np.ndarray,
                        x_cfd: np.ndarray, y_cfd: np.ndarray,
                        fs: float, max_lag_s: float,
                        delta_accept: float = 0.05, ci_alpha: float = 0.05
                        ) -> CorrComparison:
    tau_s, rho_t, N_t = _corr_function(x_test, y_test, fs, max_lag_s)
    _, rho_c, N_c = _corr_function(x_cfd, y_cfd, fs, max_lag_s)
    ci_lo, ci_hi = fisher_ci(rho_c, N_c, alpha=ci_alpha)
    acc_lo = rho_t - delta_accept
    acc_hi = rho_t + delta_accept
    return CorrComparison(tau_s, rho_t, rho_c, ci_lo, ci_hi, acc_lo, acc_hi)

# ---------- Plotting ----------
# def _annotate_dt(ax, tau_s, rho, threshold=0.2):
#     pos = rho >= threshold
#     if not np.any(pos): return
#     zero = np.argmin(np.abs(tau_s))
#     L = R = zero
#     while L-1 >= 0 and pos[L-1]: L -= 1
#     while R+1 < len(pos) and pos[R+1]: R += 1
#     dt = tau_s[R] - tau_s[L]
#     ax.annotate(f"$\\Delta t \\approx {dt:.2f}\\,\\mathrm{{s}}$",
#                 xy=(0, rho[zero]),
#                 xytext=(tau_s[zero] + 0.0003, rho[zero] + 0.8),
#                 arrowprops=dict(arrowstyle='->', lw=1),
#                 fontsize=10)


def plot_corr(res: CorrComparison, title: str, ylabel: str,
              annotate_dt: bool = False, threshold: float = 0.2):
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.plot(res.tau_s, res.rho_test, "g-", lw=2, label="Test")
    ax.plot(res.tau_s, res.acc_lo_test, "k--", lw=1, label=r"$\pm\delta$ Acceptance")
    ax.plot(res.tau_s, res.acc_hi_test, "k--", lw=1)
    ax.plot(res.tau_s, res.rho_cfd, "b--", ms=4, label="CFD")
    ax.plot(res.tau_s, res.ci_lo_cfd, "b--", lw=1, label="95% CI (CFD)")
    ax.plot(res.tau_s, res.ci_hi_cfd, "b--", lw=1)
    ax.axhline(0.0, color="0.5", lw=0.8)
    ax.set_xlabel("Lag Time, $\tau$ (sec)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # Legend inside plot near top left
    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98),
              frameon=True, ncol=1, fontsize=9, borderaxespad=0.0)

    if annotate_dt:
        _annotate_dt(ax, res.tau_s, res.rho_test, color="k")

    return fig, ax

# -------- High-level class --------
class TimeSpaceEquivalence:
    def __init__(self, df_test: pd.DataFrame, df_cfd: pd.DataFrame, fs: float):
        self.fs = fs
        # Keep only UHAD_COLS, in this order
        self.test = df_test.filter(UHAD_COLS).copy()
        self.cfd = df_cfd.filter(UHAD_COLS).copy()
        # Align lengths and drop non-numeric quietly
        n = min(len(self.test), len(self.cfd))
        self.test = self.test.iloc[:n].apply(pd.to_numeric, errors="coerce").fillna(method="ffill")
        self.cfd = self.cfd.iloc[:n].apply(pd.to_numeric, errors="coerce").fillna(method="ffill")

    def auto(self, probe: str, max_lag_ms=2.0, delta_accept=0.05,
             ci_alpha=0.05, plot=True):
        x_t = self.test[probe].to_numpy()
        x_c = self.cfd[probe].to_numpy()
        res = corr_compare_series(x_t, x_t, x_c, x_c,
                                  self.fs, max_lag_ms/1000.0,
                                  delta_accept=delta_accept, ci_alpha=ci_alpha)
        if plot:
            fig, ax = plot_corr(res, f"Auto-Correlation / {probe}",
                                ylabel=r"Auto-Correlation, $\rho_{(m)}(\tau)$",
                                annotate_dt=True, threshold=0.2)
            # plt.show()
        return res

    def cross(self, probe_m: str, probe_n: str, max_lag_ms=2.0,
              delta_accept=0.05, ci_alpha=0.05, plot=True):
        xm_t = self.test[probe_m].to_numpy()
        xn_t = self.test[probe_n].to_numpy()
        xm_c = self.cfd[probe_m].to_numpy()
        xn_c = self.cfd[probe_n].to_numpy()
        res = corr_compare_series(xm_t, xn_t, xm_c, xn_c,
                                  self.fs, max_lag_ms/1000.0,
                                  delta_accept=delta_accept, ci_alpha=ci_alpha)
        if plot:
            fig, ax = plot_corr(res, f"Cross-Correlation / {probe_m} vs {probe_n}",
                                ylabel=r"Cross-Correlation, $\rho_{(m)}(\tau)$",
                                annotate_dt=False)
            # plt.show()
        return res

    def rho0_matrix(self, which: str = "test") -> pd.DataFrame:
        """Zero-lag correlation (Pearson) matrix for a quick 40x40 overview."""
        df = self.test if which.lower() == "test" else self.cfd
        return df.corr()


# =========================== Batch save utilities ===========================
def _ensure_dir(path: str):
    import os
    os.makedirs(path, exist_ok=True)


def _to_probe_name(val) -> str:
    """Convert numeric 101/101.0 or UHAD101 to standard UHAD<ring><pos:02d>."""
    s = str(val).strip().upper()
    if s.startswith("UHAD"):
        m = re.fullmatch(r"UHAD([1-5])([0-8]{2})", s)
        if not m or not (1 <= int(m.group(2)) <= 8):
            raise ValueError(f"Bad probe name: {val}")
        return f"UHAD{int(m.group(1))}{int(m.group(2)):02d}"
    else:
        # Numeric identifier such as 101
        if not re.fullmatch(r"([1-5])([0-8]{2})", s):
            raise ValueError(f"Bad probe code: {val}")
        ring, pos = int(s[0]), int(s[1:])
        if not (1 <= pos <= 8):
            raise ValueError(f"Bad pos in {val}")
        return f"UHAD{ring}{pos:02d}"


def _export_corr_csv(res: CorrComparison, csv_path: str):
    """Save correlation curves and confidence/acceptance limits."""
    df_out = pd.DataFrame({
        "tau_s": res.tau_s,
        "rho_test": res.rho_test,
        "rho_cfd": res.rho_cfd,
        "ci_lo_cfd": res.ci_lo_cfd,
        "ci_hi_cfd": res.ci_hi_cfd,
        "acc_lo_test": res.acc_lo_test,
        "acc_hi_test": res.acc_hi_test,
    })
    df_out.to_csv(csv_path, index=False)


class TimeSpaceEquivalence(TimeSpaceEquivalence):  # extend the class above
    def save_auto_all(self,
                      outdir: str,
                      probes: Optional[List[str]] = None,
                      max_lag_ms: float = 2.0,
                      delta_accept: float = 0.05,
                      ci_alpha: float = 0.05,
                      ylims: Optional[Tuple[float, float]] = None,
                      filetype: str = "png",
                      dpi: int = 300,
                      save_csv: bool = True):
        """Save auto-correlation plots (and optionally CSV curves) for all probes."""
        _ensure_dir(outdir)
        probes_list = probes or getattr(self, "probes", self.test.columns.tolist())

        for p in probes_list:
            if p not in self.test.columns:
                print(f"[SKIP] Missing probe in data: {p}")
                continue
            res = corr_compare_series(self.test[p].to_numpy(),
                                      self.test[p].to_numpy(),
                                      self.cfd[p].to_numpy(),
                                      self.cfd[p].to_numpy(),
                                      self.fs, max_lag_ms/1000.0,
                                      delta_accept=delta_accept, ci_alpha=ci_alpha)
            fig, ax = plot_corr(res,
                                title=f"Auto-Correlation / {p}",
                                ylabel=r"Auto-Correlation, $\rho_{(m)}(\tau)$",
                                annotate_dt=True)
            if ylims: ax.set_ylim(*ylims)
            fpath = os.path.join(outdir, f"auto_{p}.{filetype}")
            fig.savefig(fpath, dpi=dpi, bbox_inches="tight")
            plt.close(fig)

            if save_csv:
                cpath = os.path.join(outdir, f"auto_{p}.csv")
                _export_corr_csv(res, cpath)

        print(f"[DONE] Saved auto-corr figures to: {outdir}")


def _read_pairs_sheet(xlsx_path: str, sheet_name=None) -> Tuple[pd.DataFrame, str]:
    """Load an Excel sheet of probe pairs; if unspecified, take first suitable sheet."""
    obj = pd.read_excel(xlsx_path, sheet_name=sheet_name, dtype=str)

    def _cleanup(df: pd.DataFrame) -> pd.DataFrame:
        # Remove entirely empty rows/columns and strip cell text
        df2 = df.dropna(how="all").dropna(axis=1, how="all")
        df2 = df2.applymap(lambda v: v if pd.isna(v) else str(v).strip())
        return df2

    if isinstance(obj, dict):
        for name, df in obj.items():
            df2 = _cleanup(df)
            if df2.shape[1] >= 2 and not df2.empty:
                return df2, name
        raise ValueError("No non-empty sheet with at least two columns found in Excel.")
    else:
        df2 = _cleanup(obj)
        if df2.shape[1] < 2 or df2.empty:
            raise ValueError("Selected sheet is empty or has fewer than two columns.")
        return df2, (sheet_name if sheet_name is not None else 0)


def _to_probe_name(val) -> str:
    """Convert labels such as 101/101.0/UHAD101 to UHAD<ring><pos:02d>."""
    if pd.isna(val):
        raise ValueError("empty cell")
    s = str(val).strip().upper()
    s = s.replace('\u00a0', '')  # nbsp
    if s.endswith('.0'):  # Remove Excel decimal suffix
        s = s[:-2]
    if s.startswith("UHAD"):
        m = re.fullmatch(r"UHAD([1-5])([0-8]{2})", s)
        if not m: raise ValueError(f"Bad probe name: {val}")
        ring, pos = int(m.group(1)), int(m.group(2))
        if not (1 <= pos <= 8): raise ValueError(f"Bad pos: {pos}")
        return f"UHAD{ring}{pos:02d}"
    # short numeric ID
    m = re.fullmatch(r"([1-5])([0-8]{2})", s)
    if not m: raise ValueError(f"Bad probe code: {val}")
    ring, pos = int(m.group(1)), int(m.group(2))
    if not (1 <= pos <= 8): raise ValueError(f"Bad pos: {pos}")
    return f"UHAD{ring}{pos:02d}"


def _export_corr_csv(res: CorrComparison, csv_path: str):
    df_out = pd.DataFrame({
        "tau_s": res.tau_s,
        "rho_test": res.rho_test,
        "rho_cfd": res.rho_cfd,
        "ci_lo_cfd": res.ci_lo_cfd,
        "ci_hi_cfd": res.ci_hi_cfd,
        "acc_lo_test": res.acc_lo_test,
        "acc_hi_test": res.acc_hi_test,
    })
    df_out.to_csv(csv_path, index=False)


class TimeSpaceEquivalence(TimeSpaceEquivalence):  # extend the class
    def save_cross_from_excel(self,
                              xlsx_path: str,
                              outdir: str,
                              sheet_name: Optional[object] = 0,
                              col_m: int = 0,
                              col_n: int = 1,
                              max_lag_ms: float = 2.0,
                              delta_accept: float = 0.05,
                              ci_alpha: float = 0.05,
                              ylims: Optional[Tuple[float, float]] = None,
                              filetype: str = "png",
                              dpi: int = 300,
                              save_csv: bool = True):
        """Read Excel probe pairs by specified column indices and save plots/CSV."""
        _ensure_dir(outdir)

        # --- read pairs safely
        df_pairs, used_sheet = _read_pairs_sheet(xlsx_path, sheet_name=sheet_name)
        # Select only the two specified columns by index
        pair_vals = df_pairs.iloc[:, [col_m, col_n]].to_numpy()

        failures = []
        for idx, (v_m, v_n) in enumerate(pair_vals, start=1):
            try:
                m = _to_probe_name(v_m)
                n = _to_probe_name(v_n)
            except Exception as e:
                print(f"[SKIP row {idx}] bad names: {v_m}, {v_n} -> {e}")
                failures.append((idx, str(e)))
                continue

            if m not in self.test.columns or n not in self.test.columns:
                print(f"[SKIP row {idx}] missing in data: {m}, {n}")
                failures.append((idx, "missing columns"))
                continue

            res = corr_compare_series(self.test[m].to_numpy(),
                                      self.test[n].to_numpy(),
                                      self.cfd[m].to_numpy(),
                                      self.cfd[n].to_numpy(),
                                      self.fs, max_lag_ms/1000.0,
                                      delta_accept=delta_accept, ci_alpha=ci_alpha)

            fig, ax = plot_corr(res,
                                title=f"Cross-Correlation / {m} vs {n}",
                                ylabel=r"Cross-Correlation, $\rho_{(m)}(\tau)$",
                                annotate_dt=False)
            if ylims: ax.set_ylim(*ylims)
            safe = f"cross_{m}_{n}".replace("UHAD", "")
            fpath = os.path.join(outdir, f"{safe}.{filetype}")
            fig.savefig(fpath, dpi=dpi, bbox_inches="tight")
            plt.close(fig)

            if save_csv:
                cpath = os.path.join(outdir, f"{safe}.csv")
                _export_corr_csv(res, cpath)

        print(f"[DONE] Saved cross-corr figures to: {outdir} (sheet={used_sheet})")
        if failures:
            print(f"[INFO] Skipped {len(failures)} rows from Excel. See messages above.")


# ===================== نمونه استفاده =====================

# ---------- Example driver (edit paths & fs and run) ----------
if __name__ == "__main__":
    # 1) Load your files (replace with your paths). If you already have DataFrames, skip this part.
    # wdir = 'D:/Volcano_LES/Mach130_AoA0_AoS0_VREMAN'
    # wdir = '//vds/Bolumler/J13F0/VOLCANO/DATA/STATISTICS/Mach130_AoA0_AoS0_VREMAN/1994'
    # wdir = '//vds/Bolumler/J13F0/VOLCANO/DATA/STATISTICS/2SEC'

    wdir = 'D:/GE_PACKAGE_1_Ma130_AoA0_AoS5/4_SSEC/2014/'
    wt_runid = 'C1060Run0242T2014'
    cfd_runid = 'C1060Run0911T2014_CFD'

    os.chdir(wdir)
    # --- File Paths ---
    TEST_FILE = wt_runid + '.unsteady'
    CFD_FILE = cfd_runid + '.unsteady'
    wt_steady_file = wt_runid + '.steady'
    cfd_steady_file = cfd_runid + '.steady'

    coords_file = 'ProbeCoordinates_WT.xlsx'
    # --- Load Data ---
    # pressure_df = pd.read_csv(pressure_file, sep='\t')

    # TEST_FILE = 'test_timeseries.csv'
    # CFD_FILE = 'cfd_timeseries.csv'
    df_test = pd.read_csv(TEST_FILE, sep='\t')
    df_cfd = pd.read_csv(CFD_FILE, sep='\t')

    # 2) Sampling frequency [Hz] (set yours)
    fs = 25000.0

    tse = TimeSpaceEquivalence(df_test, df_cfd, fs)

    # Output directories
    OUT_ROOT = os.path.join(wdir, 'equivalence_outputs')
    AUTO_DIR = os.path.join(OUT_ROOT, 'auto')
    CROSS_DIR = os.path.join(OUT_ROOT, 'cross')

    # 1) Batch-save all auto-correlation plots (PNG + CSV)
    tse.save_auto_all(AUTO_DIR,
                      max_lag_ms=2.0,
                      delta_accept=0.05,
                      ci_alpha=0.05,
                      ylims=None,        # e.g. (1.1, -0.2)
                      filetype='png',
                      dpi=300,
                      save_csv=True)

    # 2) Read probe pairs from Excel and save cross-correlations
    PAIRS_XLSX = os.path.join(wdir, 'cross_pairs.xlsx')
    tse.save_cross_from_excel(PAIRS_XLSX,
                              outdir=CROSS_DIR,
                              sheet_name=None,   # use first suitable sheet
                              col_m=0, col_n=1,  # first two columns
                              max_lag_ms=2.0,
                              delta_accept=0.05,
                              ci_alpha=0.05,
                              ylims=None,
                              filetype='png',
                              dpi=300,
                              save_csv=True)

    # 3) Figure-7 style auto-correlation (example probe)
    tse.auto('UHAD101', max_lag_ms=2.0, delta_accept=0.05, ci_alpha=0.05)

    # 4) Figure-9 style cross-correlation (adjacent along same rake, pos=01: ring1 vs ring2)
    tse.cross('UHAD101', 'UHAD201', max_lag_ms=2.0, delta_accept=0.05, ci_alpha=0.05)

    # 5) Batch examples:
    #    a) All same-rake adjacent-ring pairs:
    for a, b in same_rake_pairs()[:4]:  # first few pairs as a demo
        tse.cross(a, b, max_lag_ms=2.0, delta_accept=0.05, ci_alpha=0.05)

    #    b) Same-ring adjacent-position pairs for ring 2:
    for a, b in same_ring_adjacent_pos_pairs(ring=2)[:4]:
        tse.cross(a, b, max_lag_ms=2.0, delta_accept=0.05, ci_alpha=0.05)

    # 6) Optional: zero-lag overview matrix (Test)
    # rho0 = tse.rho0_matrix('test')
    # print(rho0.round(3))

    # --- HEATMAP VISUALIZATION ---
    # plt.figure(figsize=(12, 10))
    # cmap='coolwarm'  # makes -1 blue, 0 white, and 1 red
    # sns.heatmap(rho0, annot=False, cmap='coolwarm', center=0,
    #             linewidths=0.1, linecolor='gray', cbar_kws={'label': 'Correlation Coefficient'})

    # plt.title('Zero-Lag Correlation Matrix (Test Data)', fontsize=15)
    # plt.xlabel('Probes')
    # plt.ylabel('Probes')

    # # Save as PNG
    # heatmap_path = os.path.join(OUT_ROOT, 'correlation_heatmap.png')
    # plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    # plt.close()
    # print(f'[DONE] Correlation heatmap saved to: {heatmap_path}')

    # # Keep your excel save if you still want the raw numbers
    # rho0_filename = os.path.join(OUT_ROOT, 'zero_lag_correlation_matrix.xlsx')
    # rho0.to_excel(rho0_filename)
    # print(f'[DONE] Correlation matrix saved to: {rho0_filename}')

    # # 1. Generate both matrices
    # rho0_test = tse.rho0_matrix('test')
    # rho0_cfd = tse.rho0_matrix('cfd')

    # # 2. Create a figure with two subplots (1 row, 2 columns)
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # # Plot Test Data on the left
    # sns.heatmap(rho0_test, annot=False, cmap='coolwarm', center=0,
    #             linewidths=0.1, linecolor='gray', ax=ax1,
    #             cbar_kws={'label': 'Correlation'})
    # ax1.set_title('Zero-Lag Correlation: TEST', fontsize=16)
    # ax1.set_xlabel('Probes')
    # ax1.set_ylabel('Probes')

    # # Plot CFD Data on the right
    # sns.heatmap(rho0_cfd, annot=False, cmap='coolwarm', center=0,
    #             linewidths=0.1, linecolor='gray', ax=ax2,
    #             cbar_kws={'label': 'Correlation'})
    # ax2.set_title('Zero-Lag Correlation: CFD', fontsize=16)
    # ax2.set_xlabel('Probes')
    # ax2.set_ylabel('Probes')

    # plt.tight_layout()

    # # 3. Save the comparison as a PNG
    # comparison_path = os.path.join(OUT_ROOT, 'correlation_comparison_test_vs_cfd.png')
    # plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    # plt.close()
    # print(f'[DONE] Comparison heatmap saved to: {comparison_path}')
