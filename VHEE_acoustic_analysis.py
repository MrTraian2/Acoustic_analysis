

# %% [0] SETUP --------------------------------------------------------------
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from scipy.optimize import curve_fit

# ---------------------------------------------------------------- paths ---
# Put the workbook in the same folder as this script, or give a full path.
DATA_FILE = Path("C:/Users/Traian/Desktop/CERN/Python code/Measurements_4.xlsx")
OUTPUT_DIR = Path("C:/Users/Traian/Desktop/CERN/Python code/figures")
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------- publication style ---
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 13.5,
    "axes.titleweight": "bold",
    "legend.fontsize": 10,
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "0.8",
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": ":",
    "lines.markersize": 6,
    "errorbar.capsize": 3,
})

COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def style_axes(ax):

    if ax.get_xscale() == "linear":
        ax.xaxis.set_minor_locator(AutoMinorLocator())
    if ax.get_yscale() == "linear":
        ax.yaxis.set_minor_locator(AutoMinorLocator())
    return ax


def save_fig(fig, name):
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{name}.png")
    fig.savefig(OUTPUT_DIR / f"{name}.pdf")
    print(f"  saved -> {OUTPUT_DIR / name}.png / .pdf")


# %% [1] DATA LOADING & "value ± error" PARSING ------------------------------

def parse_measurement(cell):

    if pd.isna(cell):
        return np.nan, np.nan
    if isinstance(cell, (int, float, np.integer, np.floating)):
        return float(cell), np.nan
    s = str(cell).strip().replace(",", ".")
    if "±" in s:
        v, e = s.split("±")
        return float(v.strip()), float(e.strip())
    try:
        return float(s), np.nan
    except ValueError:
        return np.nan, np.nan


def load_sheet(sheet_name, split_cols=()):

    df = pd.read_excel(DATA_FILE, sheet_name=sheet_name)
    for col in split_cols:
        vals, errs = zip(*df[col].map(parse_measurement))
        df[f"{col}_val"] = np.array(vals, dtype=float)
        df[f"{col}_err"] = np.array(errs, dtype=float)
    return df


print(f"Loading workbook: {DATA_FILE.resolve()}")
xls = pd.ExcelFile(DATA_FILE)
print("Sheets found:", xls.sheet_names)


# %% [2] FIT & ERROR-PROPAGATION HELPERS -------------------------------------

def linear(x, m, b):
    return m * x + b


def exp_decay(x, A0, lam, y0):
    """Simple exponential decay + offset, used for the depth (D) scans."""
    return A0 * np.exp(-x / lam) + y0


def power_law(x, A0, n):
    """Alternative model for the depth scans (geometric 1/D^n spreading)."""
    return A0 * np.power(x, -n)


def gaussian(x, A, mu, sigma, offset):
    return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) + offset


def quadratic(x, a, b, c):
    return a * x ** 2 + b * x + c


def fit_curve(func, x, y, yerr=None, p0=None, maxfev=20000):

    sigma = None
    if yerr is not None and np.any(np.isfinite(yerr)):
        fill = np.nanmean(yerr)
        sigma = np.where(np.isfinite(yerr), yerr, fill)
    popt, pcov = curve_fit(func, x, y, p0=p0, sigma=sigma,
                            absolute_sigma=sigma is not None,
                            method="trf", max_nfev=maxfev)
    perr = np.sqrt(np.diag(pcov))
    resid = y - func(x, *popt)
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return popt, perr, r2


def ratio_error(a, sa, b, sb):
    """Standard propagation for r = a / b (relative errors added in quadrature)."""
    r = a / b
    sa_ = np.nan_to_num(sa, nan=0.0)
    sb_ = np.nan_to_num(sb, nan=0.0)
    sr = np.abs(r) * np.sqrt((sa_ / a) ** 2 + (sb_ / b) ** 2)
    return r, sr


def product_error(a, sa, b, sb, k=1.0):
    """Standard propagation for v = k * a * b."""
    v = k * a * b
    sa_ = np.nan_to_num(sa, nan=0.0)
    sb_ = np.nan_to_num(sb, nan=0.0)
    sv = np.abs(v) * np.sqrt((sa_ / a) ** 2 + (sb_ / b) ** 2)
    return v, sv


# %% [3] SECTION A -- A vs Q, SINGLE BUNCH (attenuation scan) ---------------
print("\n=== Section A: A vs Q, single bunch ===")

df_sb = load_sheet("A vs. Q single bunch", split_cols=["A (mV)", "Q (nC)"])
df_sb = df_sb.sort_values("Q (nC)_val").reset_index(drop=True)

x, xerr = df_sb["Q (nC)_val"].values, df_sb["Q (nC)_err"].values
y, yerr = df_sb["A (mV)_val"].values, df_sb["A (mV)_err"].values

popt_sb, perr_sb, r2_sb = fit_curve(linear, x, y, yerr, p0=[70, 0])
m_sb, b_sb = popt_sb
dm_sb, db_sb = perr_sb
print(f"  A = ({m_sb:.2f} +/- {dm_sb:.2f}) * Q + ({b_sb:.2f} +/- {db_sb:.2f})   "
      f"R2 = {r2_sb:.4f}")

fig, ax = plt.subplots(figsize=(6, 4.5))
ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt='o', color=COLORS[0],
            ecolor='0.4', capsize=3, label="Data (single bunch, attenuator scan)")
xfit = np.linspace(0, x.max() * 1.08, 200)
ax.plot(xfit, linear(xfit, *popt_sb), '-', color=COLORS[1], lw=1.8,
        label=(fr"Linear fit: $A=({m_sb:.1f}\pm{dm_sb:.1f})\,Q"
               fr"+({b_sb:.1f}\pm{db_sb:.1f})$" + "\n" + fr"$R^2={r2_sb:.4f}$"))
ax.set_xlabel("Charge $Q$ (nC)")
ax.set_ylabel("Acoustic signal $A$ (mV)")
ax.set_title("Acoustic signal vs. charge (single bunch)")
ax.legend(loc="upper left")
style_axes(ax)
save_fig(fig, "A_section_A_single_bunch_A_vs_Q")
plt.show()

#%%
# --- A3: Improved Charge vs Attenuation fits (plateau -> drop models) -----
print("\n=== Section A3: improved Charge vs Attenuation fits ===")

# Use df_sb from Section A (already loaded). If not present, load it.
if "df_sb" not in globals():
    df_sb = load_sheet("A vs. Q single bunch", split_cols=["A (mV)", "Q (nC)"])
    df_sb = df_sb.sort_values("Q (nC)_val").reset_index(drop=True)

att = df_sb["Attenuation (%)"].astype(float).values
Q_val = df_sb["Q (nC)_val"].astype(float).values
Q_err = df_sb["Q (nC)_err"].astype(float)

def hill(x, Qmax, KA, n, Qmin):
    """Hill / saturation: Q = Qmin + (Qmax-Qmin) / (1 + (x/KA)^n)"""
    # guard against zero/negative KA
    KA = np.abs(KA) if KA != 0 else 1.0
    return Qmin + (Qmax - Qmin) / (1.0 + (x / KA) ** n)


ATT_MODEL = "hill"

# initial guesses and safe bounds per model
if ATT_MODEL == "hill":
    p0 = [np.nanmax(Q_val), 10.0, 2.0, np.nanmin(Q_val)]
    # try a fit
    try:
        popt, perr, r2 = fit_curve(hill, att, Q_val, Q_err, p0=p0)
        Qmax, KA, n, Qmin = popt
        print(f"Hill fit: Qmax={Qmax:.3g} +/- {perr[0]:.3g}, KA={KA:.3g} +/- {perr[1]:.3g}, n={n:.3g} +/- {perr[2]:.3g}, Qmin={Qmin:.3g} +/- {perr[3]:.3g}, R2={r2:.4f}")
        fit_fn = lambda x: hill(x, *popt)
        fit_label = fr"Hill fit: $Q_{{max}}={Qmax:.3g}, k={KA:.3g}, n={n:.2f}, Q_{{min}}={Qmin:.3g}$"
    except Exception as e:
        print("Hill fit failed:", e)
        popt = None


# Plot data + fit (if available)
fig, ax = plt.subplots(figsize=(6.4, 4.2))
ax.errorbar(att, Q_val, yerr=Q_err, fmt='o', color=COLORS[2], ecolor='0.4', capsize=3, label="Data (single bunch)")
if popt is not None:
    xfit = np.linspace(np.min(att)-2, np.max(att)+2, 300)
    ax.plot(xfit, fit_fn(xfit), '-', color='tab:red', lw=1.8, label=fit_label)
ax.set_xlabel("Attenuation (%)")
ax.set_ylabel("Charge $Q$ (nC)")
ax.set_title("Charge vs. Attenuation - single bunch")
ax.legend(loc="best")
style_axes(ax)
save_fig(fig, "A_section_A_charge_vs_attenuation_improved_fit")
plt.show()


# %% [4] SECTION B -- A vs Q, MULTI BUNCH ------------------------------------
print("\n=== Section B: A vs Q, multi bunch ===")

df_mb = load_sheet("A vs. Q multi bunch", split_cols=["A (mV)"])
df_mb["Q_val"] = df_mb["Q (nC)"].astype(float)
df_mb["Q_err"] = np.nan
df_mb = df_mb.sort_values("Q_val").reset_index(drop=True)

xm, xerrm = df_mb["Q_val"].values, df_mb["Q_err"].values
ym, yerrm = df_mb["A (mV)_val"].values, df_mb["A (mV)_err"].values

popt_mb, perr_mb, r2_mb = fit_curve(linear, xm, ym, yerrm, p0=[70, 0])
m_mb, b_mb = popt_mb
dm_mb, db_mb = perr_mb
print(f"  A = ({m_mb:.2f} +/- {dm_mb:.2f}) * Q + ({b_mb:.2f} +/- {db_mb:.2f})   "
      f"R2 = {r2_mb:.4f}")

fig, ax = plt.subplots(figsize=(6, 4.5))
ax.errorbar(xm, ym, yerr=yerrm, fmt='s', color=COLORS[2],
            ecolor='0.4', capsize=3, label="Data (multi-bunch)")
xfit = np.linspace(xm.min() * 0.97, xm.max() * 1.03, 200)
ax.plot(xfit, linear(xfit, *popt_mb), '-', color=COLORS[3], lw=1.8,
        label=(fr"Linear fit: $A=({m_mb:.0f}\pm{dm_mb:.0f})\,Q"
               fr"+({b_mb:.1f}\pm{db_mb:.1f})$" + "\n" + fr"$R^2={r2_mb:.4f}$"))
ax.set_xlabel("Charge $Q$ (nC)")
ax.set_ylabel("Acoustic signal $A$ (mV)")
ax.set_title("Acoustic signal vs. charge — multi bunch")
ax.legend(loc="upper left")
style_axes(ax)
save_fig(fig, "B_section_B_multi_bunch_A_vs_Q")
plt.show()


fig, ax = plt.subplots(figsize=(6.2, 4.7))
ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt='o', color=COLORS[0],
            ecolor='0.4', capsize=3, label="Single bunch")
ax.errorbar(xm, ym, yerr=yerrm, fmt='s', color=COLORS[2],
            ecolor='0.4', capsize=3, label="Multi bunch")
xfit_all = np.linspace(0, max(x.max(), xm.max()) * 1.05, 200)
ax.plot(xfit_all, linear(xfit_all, *popt_sb), '-', color=COLORS[1], lw=1.5,
        label="Single-bunch fit")
ax.plot(xfit_all, linear(xfit_all, *popt_mb), '--', color=COLORS[3], lw=1.5,
        label="Multi-bunch fit")
ax.set_xlabel("Charge $Q$ (nC)")
ax.set_ylabel("Acoustic signal $A$ (mV)")
ax.set_title("Acoustic signal vs. charge — regime comparison")
ax.legend(loc="upper left")
style_axes(ax)
save_fig(fig, "B2_single_vs_multi_bunch_overlay")
plt.show()


# %% [5] SECTION C -- A vs NUMBER OF BUNCHES -------------
print("\n=== Section C: A vs number of bunches (multi-charge comparison) ===")

# One entry per sheet: (sheet_name, legend_label, color, marker, model)
# model = "rise_decay"  -> series with a genuine decline after the peak
# model = "saturating"  -> series that plateau (treated as real physics,
#                          fit through every point including the flat tail)
MULTI_SERIES = [
    ("A vs. Q multi bunch",       "Q = 0.5 nC", COLORS[0], 'o', "rise_decay"),
    ("A vs. Q multi bunch 1.1nC", "Q = 1.1 nC", "#2ca02c", 's', "rise_decay"),
    ("A vs. Q multi bunch 1.4nC", "Q = 1.4 nC", "#9467bd", '^', "saturating"),
]


def rise_decay(x, A0, N0, N1, offset):
    return A0 * (1 - np.exp(-x / N0)) * np.exp(-x / N1) + offset


def saturating(x, A_max, N0, offset):
    return A_max * (1 - np.exp(-x / N0)) + offset


def safe_sigma(yerr):

    yerr = np.asarray(yerr, dtype=float)
    valid = np.isfinite(yerr) & (yerr > 0)
    if not np.any(valid):
        return None
    fill = np.mean(yerr[valid])
    return np.where(valid, yerr, fill)


multi_results = {}
fig, ax = plt.subplots(figsize=(7.2, 5.0))

for sheet, label, c, mk, model_name in MULTI_SERIES:
    if sheet not in xls.sheet_names:
        print(f"  Sheet not found: '{sheet}' -- skipping.")
        continue

    df = load_sheet(sheet, split_cols=["A (mV)"])
    bunch_col = "# Bunches" if "# Bunches" in df.columns else next(
        (c_ for c_ in df.columns if "bunch" in c_.lower()), None)
    if bunch_col is None:
        raise KeyError(f"No '# Bunches'-like column in sheet '{sheet}'. "
                        f"Columns: {df.columns.tolist()}")

    df = df.sort_values(bunch_col).reset_index(drop=True)
    n_b = df[bunch_col].values.astype(float)
    A_vals = df["A (mV)_val"].values
    A_errs = df["A (mV)_err"].values

    if "Q (nC)" in df.columns:
        qvals, _ = zip(*df["Q (nC)"].map(parse_measurement))
        print(f"  '{sheet}': mean Q = {np.nanmean(qvals):.3f} nC "
              f"(std {np.nanstd(qvals):.3f} nC)")

    mask = np.isfinite(n_b) & np.isfinite(A_vals)
    n_fit, A_fit, Aerr_fit = n_b[mask], A_vals[mask], A_errs[mask]

    ax.errorbar(n_fit, A_fit, yerr=Aerr_fit, fmt=mk, color=c, ecolor='0.6',
                capsize=3, ms=6, label=label, zorder=3)

    if len(n_fit) < 4:
        print(f"  Not enough points to fit '{sheet}' (need >=4).")
        continue

    sigma = safe_sigma(Aerr_fit)
    fit_func = rise_decay if model_name == "rise_decay" else saturating
    if model_name == "rise_decay":
        p0 = [A_fit.max(), 3.0, 60.0, A_fit.min()]
    else:

        i_half = np.argmax(A_fit >= 0.5 * A_fit.max())
        n0_guess = max(n_fit[i_half] / 1.5, 2.0)
        p0 = [A_fit.max() - A_fit.min(), n0_guess, 0.0]

    try:
        popt, pcov = curve_fit(fit_func, n_fit, A_fit, p0=p0, sigma=sigma,
                                absolute_sigma=sigma is not None,
                                method="trf", max_nfev=20000)
        perr = np.sqrt(np.diag(pcov))
        resid = A_fit - fit_func(n_fit, *popt)
        r2 = 1 - np.sum(resid ** 2) / np.sum((A_fit - A_fit.mean()) ** 2)
        multi_results[sheet] = (model_name, popt, perr, r2)

        nfit_x = np.linspace(n_fit.min(), n_fit.max(), 300)
        ax.plot(nfit_x, fit_func(nfit_x, *popt), '-', color=c, lw=1.6,
                 alpha=0.85, zorder=2, label='_nolegend_')

        if model_name == "rise_decay":
            print(f"  [{sheet}] rise_decay fit: A0={popt[0]:.1f}+/-{perr[0]:.1f} mV, "
                  f"N0={popt[1]:.1f}+/-{perr[1]:.1f}, N1={popt[2]:.0f}+/-{perr[2]:.0f}, "
                  f"R2={r2:.4f}")
        else:
            print(f"  [{sheet}] saturating fit: A_max={popt[0]:.1f}+/-{perr[0]:.1f} mV, "
                  f"N0={popt[1]:.1f}+/-{perr[1]:.1f}, R2={r2:.4f}")
    except RuntimeError as e:
        print(f"  Fit failed for '{sheet}': {e}")

ax.set_xlabel("Number of bunches $N$")
ax.set_ylabel("Acoustic signal $A$ (mV)")
ax.set_title("Acoustic signal vs. number of bunches")
ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
style_axes(ax)
save_fig(fig, "C_section_C_A_vs_bunches_three_charges")
plt.show()

if multi_results:
    print("\nSummary of fits:")
    for sheet, (model_name, popt, perr, r2) in multi_results.items():
        if model_name == "rise_decay":
            print(f"  {sheet} [rise_decay]: A0={popt[0]:.1f}+/-{perr[0]:.1f}, "
                  f"N0={popt[1]:.1f}+/-{perr[1]:.1f}, N1={popt[2]:.0f}+/-{perr[2]:.0f}, "
                  f"R2={r2:.4f}")
        else:
            print(f"  {sheet} [saturating]: A_max={popt[0]:.1f}+/-{perr[0]:.1f}, "
                  f"N0={popt[1]:.1f}+/-{perr[1]:.1f}, R2={r2:.4f}")


# %% [6] SECTION D -- VERTICAL BEAM POSITION & BEAM SIZE ---------------------

print("\n=== Section D: A vs vertical position, A vs beam area ===")
 
FWHM_ERR = 0.25  # mm, fixed assumption (given), applied to FWHM x and y alike
 
df_v = load_sheet("A vs. d vertical", split_cols=["A (mV)"])
df_v = df_v.sort_values("d at DETECTOR (mm)").reset_index(drop=True)
 
dv = df_v["d at DETECTOR (mm)"].values
Av, Averr = df_v["A (mV)_val"].values, df_v["A (mV)_err"].values
fwhm_x = df_v["FWHM x (mm)"].values
fwhm_y = df_v["FWHM y (mm)"].values
area = df_v["Area estimate (mm^2)"].values
# error propagation: Area = (pi/4) * FWHMx * FWHMy
_, area_err = product_error(fwhm_x, np.full_like(fwhm_x, FWHM_ERR),
                             fwhm_y, np.full_like(fwhm_y, FWHM_ERR),
                             k=np.pi / 4)
 
# -----------------------------------------------------------------------
# D1a -- Acoustic signal vs. vertical position: points only, no fit -----
# -----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 4.5))
ax.errorbar(dv, Av, yerr=Averr, fmt='o-', color=COLORS[0], ecolor='0.4',
            capsize=3, label="Data (points joined)")
ax.set_xlabel("Vertical position at detector, $d$ (mm)")
ax.set_ylabel("Acoustic signal $A$ (mV)")
ax.set_title("Acoustic signal vs. vertical beam position")
ax.legend(loc="best")
style_axes(ax)
save_fig(fig, "D1a_A_vs_vertical_position_points")
plt.show()
 
# -----------------------------------------------------------------------
# D1b -- Acoustic signal vs. vertical position: piecewise rise + drop ---
# The first 4 points trace a genuine, well-constrained exponential rise
# (fit below, R^2 computed on those 4 points alone). The final point is a
# sharp drop that only 2 points define -- there's nothing to statistically
# fit there, so it's drawn as the direct line it is, not dressed up as a
# fit. This is more honest than forcing one flexible curve through all 5
# points (which would just interpolate, see the parabola note above).
# -----------------------------------------------------------------------
i_peak = np.argmax(Av)  # index of the last "rise" point / observed peak
d_rise, A_rise, Aerr_rise = dv[:i_peak + 1], Av[:i_peak + 1], Averr[:i_peak + 1]
d_drop, A_drop = dv[i_peak:], Av[i_peak:]
 
 
def exp_rise(x, A0, tau, off):
    return A0 * np.exp((x - dv.min()) / tau) + off
 
 
popt_r, perr_r, r2_r = fit_curve(exp_rise, d_rise, A_rise, Aerr_rise,
                                  p0=[10, 3, A_rise.min()])
drop_slope = (A_drop[-1] - A_drop[0]) / (d_drop[-1] - d_drop[0])
print(f"  Rise segment (first {i_peak + 1} points): exponential fit "
      f"A0={popt_r[0]:.2f}+/-{perr_r[0]:.2f}, tau={popt_r[1]:.2f}+/-{perr_r[1]:.2f} mm, "
      f"R2={r2_r:.4f}")
print(f"  Drop segment (last 2 points): direct line, slope={drop_slope:.1f} mV/mm "
      f"(not a statistical fit -- only 2 points define it)")
 
fig, ax = plt.subplots(figsize=(6, 4.5))
ax.errorbar(dv, Av, yerr=Averr, fmt='o', color=COLORS[0], ecolor='0.4',
            capsize=3, zorder=3, label="Data")
d_rise_fit = np.linspace(d_rise.min(), d_rise.max(), 200)
ax.plot(d_rise_fit, exp_rise(d_rise_fit, *popt_r), '-', color=COLORS[1], lw=1.8,
        zorder=2, label=fr"Exponential rise fit (first {i_peak + 1} pts): $R^2={r2_r:.3f}$")
ax.plot(d_drop, A_drop, '--', color=COLORS[3], lw=1.8, zorder=2,
        label=fr"Drop: direct line, slope$={drop_slope:.0f}$ mV/mm")
ax.set_xlabel("Vertical position at detector, $d$ (mm)")
ax.set_ylabel("Acoustic signal $A$ (mV)")
ax.set_title("Acoustic signal vs. vertical position — rise/drop fit")
ax.legend(loc="best")
style_axes(ax)
save_fig(fig, "D1b_A_vs_vertical_position_fit")
plt.show()
 
# -----------------------------------------------------------------------
# D2a -- Acoustic signal vs. beam area: points only, no fit -------------
# -----------------------------------------------------------------------
order = np.argsort(area)
fig, ax = plt.subplots(figsize=(6, 4.5))
ax.errorbar(area[order], Av[order], xerr=area_err[order], yerr=Averr[order],
            fmt='o-', color=COLORS[2], ecolor='0.4', capsize=3,
            label="Data (points joined, sorted by area)")
ax.set_xlabel(r"Beam area, $\pi/4 \cdot \mathrm{FWHM}_x \mathrm{FWHM}_y$ (mm$^2$)")
ax.set_ylabel("Acoustic signal $A$ (mV)")
ax.set_title("Acoustic signal vs. beam area")
ax.legend(loc="best")
style_axes(ax)
save_fig(fig, "D2a_A_vs_area_points")
plt.show()
 
# -----------------------------------------------------------------------
# D2b -- Acoustic signal vs. beam area: linear fit -----------------------
# -----------------------------------------------------------------------
popt_a, perr_a, r2_a = fit_curve(linear, area, Av, Averr, p0=[-10, 300])
m_a, b_a = popt_a
dm_a, db_a = perr_a
print(f"  A vs area: A = ({m_a:.2f}+/-{dm_a:.2f})*Area + ({b_a:.1f}+/-{db_a:.1f}), "
      f"R2={r2_a:.4f} (larger beam spot -> lower peak signal, as expected "
      f"if the same charge is spread over a larger area)")
 
fig, ax = plt.subplots(figsize=(6, 4.5))
ax.errorbar(area, Av, xerr=area_err, yerr=Averr, fmt='o', color=COLORS[2],
            ecolor='0.4', capsize=3, label="Data")
xfit_a = np.linspace(area.min() * 0.9, area.max() * 1.05, 200)
ax.plot(xfit_a, linear(xfit_a, *popt_a), '-', color=COLORS[3], lw=1.8,
        label=(fr"Linear fit: $A=({m_a:.1f}\pm{dm_a:.1f})\,\mathrm{{Area}}"
               fr"+({b_a:.0f}\pm{db_a:.0f})$" + "\n" + fr"$R^2={r2_a:.3f}$"))
ax.set_xlabel(r"Beam area, $\pi/4 \cdot \mathrm{FWHM}_x \mathrm{FWHM}_y$ (mm$^2$)")
ax.set_ylabel("Acoustic signal $A$ (mV)")
ax.set_title("Acoustic signal vs. beam area — linear fit")
ax.legend(loc="best")
style_axes(ax)
save_fig(fig, "D2b_A_vs_area_fit")
plt.show()



# %% [6'] with DOSE

print("\n=== Section D: A vs vertical position, A vs beam area (with A & Dose fits) ===")

# Reuse the variables created in Section D above (df_v, dv, Av, Averr, area, area_err).
# Defensive fallback to load if they don't exist (keeps compatibility).
try:
    _ = df_v  # noqa: F821
except NameError:
    df_v = load_sheet("A vs. d vertical", split_cols=["A (mV)"])
    df_v = df_v.sort_values("d at DETECTOR (mm)").reset_index(drop=True)
    dv = df_v["d at DETECTOR (mm)"].values
    Av = df_v["A (mV)_val"].values
    Averr = df_v["A (mV)_err"].values
    fwhm_x = df_v["FWHM x (mm)"].values
    fwhm_y = df_v["FWHM y (mm)"].values
    _, area_err = product_error(fwhm_x, np.full_like(fwhm_x, FWHM_ERR),
                                fwhm_y, np.full_like(fwhm_y, FWHM_ERR),
                                k=np.pi / 4)
    area = df_v["Area estimate (mm^2)"].values

# Obtain per-row Q if present, otherwise fallback to Section A estimate
if "Q (nC)_val" in df_v.columns and "Q (nC)_err" in df_v.columns:
    Q_val = df_v["Q (nC)_val"].values
    Q_err = df_v["Q (nC)_err"].values
elif "Q (nC)" in df_v.columns:
    vals, errs = zip(*df_v["Q (nC)"].map(parse_measurement))
    Q_val = np.array(vals, dtype=float)
    Q_err = np.array(errs, dtype=float)
elif "Q_val" in df_v.columns:
    Q_val = df_v["Q_val"].astype(float).values
    Q_err = df_v.get("Q_err", np.full_like(Q_val, np.nan)).astype(float)
else:
    # Fallback estimate from Section A linear fit
    print("Note: per-point Q not found for vertical scan — estimating Q from A using Section A fit.")
    Q_val = (Av - b_sb) / m_sb
    Q_err = Averr / m_sb

# dose conversion
k_calib = globals().get("K_COEFF", 190.0)
dose = np.full_like(Av, np.nan)
dose_err = np.full_like(Av, np.nan)
mask_valid = np.isfinite(Q_val) & (Q_val != 0) & np.isfinite(area) & (area > 0)
if np.any(mask_valid):
    dose[mask_valid] = k_calib * Q_val[mask_valid] / area[mask_valid]
    Qerr_ = np.nan_to_num(Q_err[mask_valid], nan=0.0)
    area_err_ = np.nan_to_num(area_err[mask_valid], nan=0.0)
    dose_err[mask_valid] = np.abs(dose[mask_valid]) * np.sqrt(
        (Qerr_ / Q_val[mask_valid]) ** 2 + (area_err_ / area[mask_valid]) ** 2)

# ----------------- fit settings (toggle here) -----------------
FIT_MODE = "piecewise"   # "piecewise" or "global"
# MODEL choices: "rise_decay", "saturating", "quadratic"
MODEL = "rise_decay"
# If FIT_MODE=="global" and MODEL=="quadratic" this will fit a quadratic over all points.
# If you want to force fitting only the first N points in piecewise mode, set:
# MANUAL_RISE_N = 4
MANUAL_RISE_N = None
# --------------------------------------------------------------

# local model definitions (used only for Section D fits)
def rise_decay_pos(x, A0, x0, x1, offset):
    return A0 * (1 - np.exp(-x / x0)) * np.exp(-x / x1) + offset

def saturating_pos(x, Amax, x0, offset):
    return Amax * (1 - np.exp(-x / x0)) + offset

# choose the model function by name
def get_model_func(name):
    if name == "rise_decay":
        return rise_decay_pos, [np.max(Av) - np.min(Av), 2.0, 20.0, np.min(Av)]
    if name == "saturating":
        return saturating_pos, [np.max(Av) - np.min(Av), 2.0, np.min(Av)]
    if name == "quadratic":
        return quadratic, [ -0.1, 1.0, np.mean(Av) ]
    # fallback
    return rise_decay_pos, [np.max(Av) - np.min(Av), 2.0, 20.0, np.min(Av)]

# decide indices used for "rise" fit when in piecewise mode
i_peak = int(np.argmax(Av))  # last rise point index for A
if MANUAL_RISE_N is not None:
    i_use = max(1, MANUAL_RISE_N - 1)
else:
    i_use = i_peak

# ---------- Plot with fits ----------
fig, ax = plt.subplots(figsize=(6.2, 4.6))
colA = "tab:blue"
ax.errorbar(dv, Av, yerr=Averr, fmt='o', color=colA, ecolor='0.4', capsize=3, label="Acoustic A (mV)")
ax.set_xlabel("Vertical position at detector, $d$ (mm)")
ax.set_ylabel("Acoustic signal $A$ (mV)", color=colA)
ax.tick_params(axis='y', labelcolor=colA)
ax.set_title("Acoustic signal and Dose vs vertical position (fits)")

# Fit & plot A
if FIT_MODE == "piecewise":
    # fit rise segment up to i_use (inclusive)
    x_rise = dv[:i_use + 1].astype(float)
    y_rise = Av[:i_use + 1].astype(float)
    yerr_rise = Averr[:i_use + 1].astype(float)
    model_func, p0 = get_model_func(MODEL)
    try:
        poptA, perrA, r2A = fit_curve(model_func, x_rise, y_rise, yerr=yerr_rise, p0=p0)
        xfine = np.linspace(x_rise.min(), dv.max(), 250)
        ax.plot(xfine, model_func(xfine, *poptA), '-', color=colA, lw=1.6,
                label=(f"A fit ({MODEL}, rise): " + 
                       ", ".join([f"{v:.2g}±{e:.2g}" for v,e in zip(poptA, perrA)]) + f", R2={r2A:.3f}"))
        print(f"  A fit ({MODEL}, rise): popt={poptA}, perr={perrA}, R2={r2A:.4f}")
    except Exception as e:
        print("  A fit failed (piecewise):", e)
else:  # global
    model_func, p0 = get_model_func(MODEL)
    try:
        poptA, perrA, r2A = fit_curve(model_func, dv.astype(float), Av.astype(float), yerr=Averr, p0=p0)
        xfine = np.linspace(dv.min(), dv.max(), 250)
        ax.plot(xfine, model_func(xfine, *poptA), '-', color=colA, lw=1.6,
                label=(f"A fit ({MODEL}): " +
                       ", ".join([f"{v:.2g}±{e:.2g}" for v,e in zip(poptA, perrA)]) + f", R2={r2A:.3f}"))
        print(f"  A fit ({MODEL}, global): popt={poptA}, perr={perrA}, R2={r2A:.4f}")
    except Exception as e:
        print("  A fit failed (global):", e)

# Right axis: Dose, with its own fit
ax2 = ax.twinx()
colD = "tab:red"
if np.any(mask_valid):
    ax2.errorbar(dv[mask_valid], dose[mask_valid], yerr=dose_err[mask_valid],
                 fmt='s', color=colD, ecolor=colD, capsize=3, label="Dose (Gy)")
    # Fit dose using same strategy as A (use subset indices aligned with dv)
    D_dv = dv[mask_valid].astype(float)
    D_dose = dose[mask_valid].astype(float)
    D_dose_err = dose_err[mask_valid].astype(float)
    # For piecewise: use same rise index relative to full dv; convert to mask indices
    if FIT_MODE == "piecewise":
        # use full-data i_use to decide which mask entries are in the rise segment
        mask_rise = (np.arange(len(dv)) <= i_use) & mask_valid
        x_rise_d = dv[mask_rise].astype(float)
        y_rise_d = dose[mask_rise].astype(float)
        yerr_rise_d = dose_err[mask_rise].astype(float)
        model_func_d, p0d = get_model_func(MODEL)
        try:
            if len(x_rise_d) >= 3:
                poptD, perrD, r2D = fit_curve(model_func_d, x_rise_d, y_rise_d, yerr=yerr_rise_d, p0=p0d)
                xfine_d = np.linspace(x_rise_d.min(), dv[mask_valid].max(), 250)
                ax2.plot(xfine_d, model_func_d(xfine_d, *poptD), '--', color=colD, lw=1.6,
                         label=(f"Dose fit ({MODEL}, rise): " +
                                ", ".join([f"{v:.2g}±{e:.2g}" for v,e in zip(poptD, perrD)]) + f", R2={r2D:.3f}"))
                print(f"  Dose fit ({MODEL}, rise): popt={poptD}, perr={perrD}, R2={r2D:.4f}")
            else:
                print("  Not enough dose points in rise segment to fit.")
        except Exception as e:
            print("  Dose fit failed (piecewise):", e)
    else:
        model_func_d, p0d = get_model_func(MODEL)
        try:
            if len(D_dv) >= 3:
                poptD, perrD, r2D = fit_curve(model_func_d, D_dv, D_dose, yerr=D_dose_err, p0=p0d)
                xfine_d = np.linspace(D_dv.min(), D_dv.max(), 250)
                ax2.plot(xfine_d, model_func_d(xfine_d, *poptD), '--', color=colD, lw=1.6,
                         label=(f"Dose fit ({MODEL}): " +
                                ", ".join([f"{v:.2g}±{e:.2g}" for v,e in zip(poptD, perrD)]) + f", R2={r2D:.3f}"))
                print(f"  Dose fit ({MODEL}, global): popt={poptD}, perr={perrD}, R2={r2D:.4f}")
            else:
                print("  Not enough dose points to fit (need >=3).")
        except Exception as e:
            print("  Dose fit failed (global):", e)
else:
    ax2.plot([], [], ' ', label=f"Dose (Gy), k={k_calib:.0f} (no per-point area/Q)")

ax2.set_ylabel("Dose (Gy)", color=colD)
ax2.tick_params(axis='y', labelcolor=colD)
ax2.spines["right"].set_color(colD)

# Legend: combine both axes' handles and place outside
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0., fontsize=9)

style_axes(ax)
save_fig(fig, "D1a_A_and_Dose_vs_vertical_position_with_fits")
plt.show()

# %%
# NO-FIT plot (same data but without fitted curves) ----------------------------------------------
print("\n=== Section D: NO-FIT variant (A & Dose vs d, no model) ===")

fig, ax = plt.subplots(figsize=(6.2, 4.6))
colA = "tab:blue"
ax.errorbar(dv, Av, yerr=Averr, fmt='o', color=colA, ecolor='0.4', capsize=3, label="Acoustic A (mV)")
ax.set_xlabel("Vertical position at detector (mm)")
ax.set_ylabel("Acoustic signal $A$ (mV)", color=colA)
ax.tick_params(axis='y', labelcolor=colA)
ax.set_title("Detector response and dose across vertical positions")

ax2 = ax.twinx()
colD = "tab:red"
if np.any(mask_valid):
    ax2.errorbar(dv[mask_valid], dose[mask_valid], yerr=dose_err[mask_valid],
                 fmt='s', color=colD, ecolor=colD, capsize=3, label="Dose (Gy)")
else:
    ax2.plot([], [], ' ', label=f"Dose (Gy), k={k_calib:.0f} (no per-point area/Q)")
ax2.set_ylabel("Dose (Gy)", color=colD)
ax2.tick_params(axis='y', labelcolor=colD)
ax2.spines["right"].set_color(colD)

# legend outside
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="lower left")#, bbox_to_anchor=(1.02, 1.0), borderaxespad=0., fontsize=9)

style_axes(ax)
save_fig(fig, "D1a_A_and_Dose_vs_vertical_position_no_fit")
plt.show()

# %%

# ----------------- Replacement 3D block: simple labels + optional surface fit -----------
print("\n=== Section D: 3D scatter of Acoustic A vs Distance & Area (optional surface fit) ===")
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# Prepare arrays (use the same arrays as Section D)
dv_arr = np.array(dv, dtype=float)               # distance at detector (mm)
area_arr = np.array(area, dtype=float)           # beam area (mm^2)
A_arr = np.array(Av, dtype=float)                # acoustic signal (mV)

# Shift distances to be strictly positive for log-fit, but keep the plotted axis labeled simply.
dv_shift = dv_arr - np.nanmin(dv_arr) + 1.0  # shift so min becomes 1 mm; used only if fitting

mask3d = np.isfinite(A_arr) & (A_arr > 0) & np.isfinite(area_arr) & (area_arr > 0) & np.isfinite(dv_shift)
n_pts = np.sum(mask3d)
print(f"  3D: found {n_pts} valid points for (distance, area, A).")

# Toggle: do we fit and draw a surface? (default: False for small-N robustness)
DO_3D_FIT = False

fig = plt.figure(figsize=(8.0, 6.0))
ax3 = fig.add_subplot(111, projection='3d')

# scatter actual points
ax3.scatter(dv_arr[mask3d], area_arr[mask3d], A_arr[mask3d],
            color=COLORS[0], s=60, edgecolor='k', label="data")

ax3.set_xlabel("Distance (mm)")
ax3.set_ylabel("Beam area (mm$^2$)")
ax3.set_zlabel("Acoustic signal A (mV)")
ax3.set_title("3D: Acoustic signal vs Distance and Beam area")

if DO_3D_FIT:
    if n_pts < 3:
        print("  Not enough points to fit surface (need >=3). Skipping surface fit.")
    else:
        # Fit log(A) = a + b*log(distance_shift) + c*log(area)
        X = np.column_stack([np.ones(np.sum(mask3d)), np.log(dv_shift[mask3d]), np.log(area_arr[mask3d])])
        Y = np.log(A_arr[mask3d])
        coeffs, *_ = np.linalg.lstsq(X, Y, rcond=None)
        a0, b_exp, c_exp = coeffs
        Y_pred = X.dot(coeffs)
        ss_res = np.sum((Y - Y_pred)**2)
        ss_tot = np.sum((Y - np.mean(Y))**2)
        r2_log = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        print(f"  3D fit (log-space): log(A) = {a0:.3f} + {b_exp:.3f}*log(dist_shift) + {c_exp:.3f}*log(area)")
        print(f"    => A ≈ {np.exp(a0):.3g} * dist_shift^{b_exp:.3f} * area^{c_exp:.3f}, R2(log)={r2_log:.4f}")

        # Build grid for surface in the original (non-shifted-labelled) axes:
        d_grid = np.linspace(np.min(dv_arr[mask3d]), np.max(dv_arr[mask3d]), 40)
        a_grid = np.linspace(np.min(area_arr[mask3d]), np.max(area_arr[mask3d]), 40)
        Dg, Ag = np.meshgrid(d_grid, a_grid)
        # transform for fit (remember shift used in fit)
        Dg_shift = Dg - np.nanmin(dv_arr) + 1.0
        logA_surf = a0 + b_exp * np.log(Dg_shift) + c_exp * np.log(Ag)
        A_surf = np.exp(logA_surf)
        # plot the surface
        ax3.plot_surface(Dg, Ag, A_surf, cmap='viridis', alpha=0.45, linewidth=0, antialiased=True)

        # Add a small annotation with the fit result
        fit_text = f"A ≈ {np.exp(a0):.2g}·d^{b_exp:.2f}·area^{c_exp:.2f}\nR2_log={r2_log:.3f}"
        ax3.text2D(0.02, 0.95, fit_text, transform=ax3.transAxes, fontsize=9, verticalalignment='top',
                   bbox=dict(facecolor='white', alpha=0.6, edgecolor='0.8'))

ax3.legend(loc="upper left")
save_fig(fig, "D3D_A_vs_distance_area")
plt.show()


# %% [7] SECTION E -- DEPTH (LONGITUDINAL) SCAN, ENERGY COMPARISON ----------
print("\n=== Section E: A vs depth D, all beam energies ===")

# Auto-detect every "A vs. D <energy> MeV" sheet so a newly added energy
# is picked up automatically -- no code changes needed.
depth_pattern = re.compile(r"A\s*vs\.?\s*D\s*([\d.]+)\s*MeV", re.IGNORECASE)
depth_sheets = {}
for sn in xls.sheet_names:
    match = depth_pattern.match(sn.strip())
    if match:
        depth_sheets[float(match.group(1))] = sn
print("  Depth-scan sheets detected:", depth_sheets)

depth_data = {}
for energy, sn in sorted(depth_sheets.items()):
    df = load_sheet(sn, split_cols=["A (mV)", "Q (nC)"])
    df = df.sort_values("D (mm)").reset_index(drop=True)

    A_, Aerr_ = df["A (mV)_val"].values, df["A (mV)_err"].values
    Q_, Qerr_ = df["Q (nC)_val"].values, df["Q (nC)_err"].values

    # --- fill in the A/Q column (error-propagated) ---
    AQ, AQerr = ratio_error(A_, Aerr_, Q_, Qerr_)
    df["A/Q"] = AQ
    df["A/Q_err"] = AQerr

    depth_data[energy] = df
    print(f"    {energy:.0f} MeV: <Q> = {np.nanmean(Q_):.3f} +/- {np.nanstd(Q_):.3f} nC "
          f"(shot-to-shot spread; per-point errors are in the sheet)")

# --- E1: A vs D, linear y-axis, all energies on one plot -------------------
fig, ax = plt.subplots(figsize=(6.6, 5))
fit_results = {}
for i, (energy, df) in enumerate(sorted(depth_data.items())):
    D = df["D (mm)"].values.astype(float)
    A_, Aerr_ = df["A (mV)_val"].values, df["A (mV)_err"].values
    c = COLORS[i % len(COLORS)]
    ax.errorbar(D, A_, yerr=Aerr_, fmt='o', color=c, ecolor='0.5', capsize=3,
                label=f"{energy:.0f} MeV (data)")

    popt_d, perr_d, r2_d = fit_curve(exp_decay, D, A_, Aerr_,
                                      p0=[A_.max(), 40, 0])
    fit_results[energy] = (popt_d, perr_d, r2_d)
    Dfit = np.linspace(D.min(), D.max(), 300)
    ax.plot(Dfit, exp_decay(Dfit, *popt_d), '-', color=c, lw=1.6,
            label=fr"{energy:.0f} MeV fit: $\lambda={popt_d[1]:.1f}\pm{perr_d[1]:.1f}$ mm")
    print(f"  [{energy:.0f} MeV] exp. decay: A0={popt_d[0]:.1f}+/-{perr_d[0]:.1f} mV, "
          f"lambda={popt_d[1]:.1f}+/-{perr_d[1]:.1f} mm, y0={popt_d[2]:.1f}+/-{perr_d[2]:.1f} mV, "
          f"R2={r2_d:.4f}")

ax.set_xlabel("Water-layer depth $D$ (mm)")
ax.set_ylabel("Acoustic signal $A$ (mV)")
ax.set_title("Acoustic signal vs. depth — beam-energy comparison")
ax.legend(loc="upper right", ncol=1)
style_axes(ax)
save_fig(fig, "E1_section_E_A_vs_D_energy_comparison")
plt.show()

# --- E1b: same data, log-y axis (a pure exponential is a straight line) ----
fig, ax = plt.subplots(figsize=(6.6, 5))
for i, (energy, df) in enumerate(sorted(depth_data.items())):
    D = df["D (mm)"].values.astype(float)
    A_, Aerr_ = df["A (mV)_val"].values, df["A (mV)_err"].values
    c = COLORS[i % len(COLORS)]
    ax.errorbar(D, A_, yerr=Aerr_, fmt='o', color=c, ecolor='0.5', capsize=3,
                label=f"{energy:.0f} MeV")
    popt_d, _, _ = fit_results[energy]
    Dfit = np.linspace(D.min(), D.max(), 300)
    ax.plot(Dfit, exp_decay(Dfit, *popt_d), '-', color=c, lw=1.6)
ax.set_yscale("log")
ax.set_xlabel("Water-layer depth $D$ (mm)")
ax.set_ylabel("Acoustic signal $A$ (mV), log scale")
ax.set_title("Acoustic signal vs. depth — log scale")
ax.legend(loc="upper right")
style_axes(ax)
save_fig(fig, "E1b_section_E_A_vs_D_logscale")
plt.show()

# --- E2: normalised A/Q vs D, all energies (checks the decay is a real ----
#         beam-propagation effect and not a charge/shot-to-shot artefact) ---
fig, ax = plt.subplots(figsize=(6.6, 5))
for i, (energy, df) in enumerate(sorted(depth_data.items())):
    D = df["D (mm)"].values.astype(float)
    AQ, AQerr = df["A/Q"].values, df["A/Q_err"].values
    c = COLORS[i % len(COLORS)]
    ax.errorbar(D, AQ, yerr=AQerr, fmt='o-', color=c, ecolor='0.5', capsize=3,
                label=f"{energy:.0f} MeV")
ax.set_xlabel("Water-layer depth $D$ (mm)")
ax.set_ylabel(r"Normalised signal $A/Q$ (mV/nC)")
ax.set_title("Charge-normalised acoustic signal vs. depth")
ax.legend(loc="upper right")
style_axes(ax)
save_fig(fig, "E2_section_E_AoverQ_vs_D")
plt.show()

#%%
# --- E3 (final): A/Q vs D with FWHM on right axis, legend top-right, centered vertical labels ---
print("\n=== Section E3 (final): A/Q vs D with FWHM and centered max-FWHM labels ===")

def find_fwhm_col(df):
    for col in df.columns:
        if "fwhm" in col.lower() or "fwhm" in col.replace("µ","u").lower():
            return col
    return None

fig, ax1 = plt.subplots(figsize=(11.7, 4.2))  # wide for A4 landscape
ax2 = ax1.twinx()

# plot A/Q (left) and FWHM (right); keep original COLOR order (so energies keep familiar colours)
vertical_lines_x = []
vertical_lines_colors = []

for i, (energy, df) in enumerate(sorted(depth_data.items())):
    D = df["D (mm)"].values.astype(float)
    AQ = df.get("A/Q", df.get("A/Q_val", np.full_like(D, np.nan))).astype(float)
    AQerr = df.get("A/Q_err", df.get("A/Q_err", np.full_like(D, np.nan))).astype(float)
    c = COLORS[i % len(COLORS)]

    # A/Q left axis
    ax1.errorbar(D, AQ, yerr=AQerr, fmt='o-', color=c, ecolor='0.5', capsize=3,
                 lw=1.5, label=f"{energy:.0f} MeV — A/Q")

    # FWHM on right axis (open square, dashed)
    fwhm_col = find_fwhm_col(df)
    if fwhm_col is not None:
        raw = df[fwhm_col].values
        try:
            fwhm_vals = raw.astype(float)
        except Exception:
            vals, errs = zip(*[parse_measurement(x) for x in raw])
            fwhm_vals = np.array(vals, dtype=float)

        ax2.plot(D, fwhm_vals, 's--', color=c, markerfacecolor='none', lw=1.2, label=f"{energy:.0f} MeV — FWHM (µs)")

        # vertical dotted line at first D where FWHM reaches its maximum among finite values
        mask_finite = np.isfinite(fwhm_vals) & np.isfinite(D)
        if np.any(mask_finite):
            masked_indices = np.where(mask_finite)[0]
            idx_rel = int(np.nanargmax(fwhm_vals[mask_finite]))
            idx_orig = masked_indices[idx_rel]
            D_max_fwhm = float(D[idx_orig])
            ax1.axvline(D_max_fwhm, color=c, linestyle=':', lw=1.0, alpha=0.9)
            vertical_lines_x.append(D_max_fwhm)
            vertical_lines_colors.append(c)
    else:
        print(f"  [{energy:.0f} MeV] no FWHM column found; skipping FWHM and marker.")

# Labels
ax1.set_xlabel("Water-layer depth $D$ (mm)")
ax1.set_ylabel(r"Normalised signal $A/Q$ (mV/nC)")
ax2.set_ylabel("FWHM (µs)")

# Extend top of left y-axis to make space for an inside legend at top-right
ymin, ymax = ax1.get_ylim()
# add 10-14% headroom so legend does not cover data
headroom = 0.14
ax1.set_ylim(ymin, ymax + headroom * (ymax - ymin))

# Legend inside top-right (keeps default COLORS so energies remain recognisable)
# Combine handles from both axes; place in top-right inside axes
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
all_handles = h1 + h2
all_labels = l1 + l2
ax1.legend(all_handles, all_labels, loc="upper right", fontsize=9, framealpha=0.9, ncol=2)

# Centered "max FWHM" text on each vertical dotted line (no box)
# Use the *new* y-limits to compute vertical center
ymin_new, ymax_new = ax1.get_ylim()
y_center = ymin_new + 0.5 * (ymax_new - ymin_new)
for xval, col in zip(vertical_lines_x, vertical_lines_colors):
    ax1.text(xval, y_center, "max FWHM", color=col,
             rotation=90, va='center', ha='center', fontsize=8, alpha=0.9)

style_axes(ax1)
style_axes(ax2)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "E3_section_E_AoverQ_and_FWHM_vs_D_wide_final.png")
fig.savefig(OUTPUT_DIR / "E3_section_E_AoverQ_and_FWHM_vs_D_wide_final.pdf")
print(f"  saved -> {OUTPUT_DIR / 'E3_section_E_AoverQ_and_FWHM_vs_D_wide_final.png'} / .pdf")
plt.show()


# %% [8] SECTION F -- RECONSTRUCTED LONGITUDINAL BEAM PROFILE --------------

print("\n=== Section F: reconstructed longitudinal beam profile ===")

PEAK_DISAGREEMENT_THRESHOLD = 0.10  # us
 
profile_data = {}
for energy, df in sorted(depth_data.items()):
    D = df["D (mm)"].values.astype(float)
    A_, Aerr_ = df["A (mV)_val"].values, df["A (mV)_err"].values
    peak_t = df["Peak times (us)"].values.astype(float)
    acou_t = df["Acoustic Signal pos (us)"].values.astype(float)
    diff_t = df["Difference (us)"].values.astype(float)
    dist = df["Distance from top to beam (mm)"].values.astype(float)
 
    # data-driven timing-uncertainty proxy (see note above)
    t_err = np.abs(peak_t - acou_t)
 

    v_sound = np.sum(dist * diff_t) / np.sum(diff_t ** 2)  # mm/us
    dist_err = t_err * v_sound
 
    reliable = t_err < PEAK_DISAGREEMENT_THRESHOLD
    profile_data[energy] = dict(D=D, dist=dist, dist_err=dist_err,
                                 A=A_, Aerr=Aerr_, reliable=reliable)
 
    print(f"  {energy:.0f} MeV: recovered v_sound = {v_sound * 1000:.1f} m/s; "
          f"reconstructed depth = {dist[reliable].mean():.2f} +/- "
          f"{dist[reliable].std():.2f} mm over the reliable points")
    if np.any(~reliable):
        print(f"    flagged unreliable (peak-time disagreement > "
              f"{PEAK_DISAGREEMENT_THRESHOLD} us) at D = "
              f"{D[~reliable].astype(int).tolist()} mm")
 
# --- F1: diagnostic -- reconstructed depth vs. nominal sensor position D ---
# Shows directly where the reconstruction starts to break down (points
# should sit roughly flat/consistent if the beam keeps forming its acoustic
# pulse at the same physical depth regardless of where the sensor is).
fig, ax = plt.subplots(figsize=(6.6, 4.8))
for i, (energy, pd_) in enumerate(sorted(profile_data.items())):
    c = COLORS[i % len(COLORS)]
    rel = pd_["reliable"]
    ax.errorbar(pd_["D"][rel], pd_["dist"][rel], yerr=pd_["dist_err"][rel],
                fmt='o-', color=c, ecolor='0.5', capsize=3,
                label=f"{energy:.0f} MeV (reliable)")
    if np.any(~rel):
        ax.errorbar(pd_["D"][~rel], pd_["dist"][~rel], yerr=pd_["dist_err"][~rel],
                     fmt='x', color=c, ecolor='0.7', capsize=3, alpha=0.5,
                     label=f"{energy:.0f} MeV (flagged unreliable)")
ax.set_xlabel("Nominal sensor depth $D$ (mm)")
ax.set_ylabel("Reconstructed source depth (mm)")
ax.set_title("Reconstructed beam-formation depth vs. sensor position")
ax.legend(loc="best")
style_axes(ax)
save_fig(fig, "F1_reconstructed_depth_vs_D")
plt.show()
 
# --- F2: the longitudinal profile itself -- A vs reconstructed depth ------
fig, ax = plt.subplots(figsize=(6.6, 5))
for i, (energy, pd_) in enumerate(sorted(profile_data.items())):
    c = COLORS[i % len(COLORS)]
    rel = pd_["reliable"]
    ax.errorbar(pd_["dist"][rel], pd_["A"][rel], xerr=pd_["dist_err"][rel],
                yerr=pd_["Aerr"][rel], fmt='o-', color=c, ecolor='0.5',
                capsize=3, label=f"{energy:.0f} MeV")
    if np.any(~rel):
        ax.errorbar(pd_["dist"][~rel], pd_["A"][~rel], xerr=pd_["dist_err"][~rel],
                     yerr=pd_["Aerr"][~rel], fmt='x', color=c, ecolor='0.7',
                     capsize=3, alpha=0.4, label=f"{energy:.0f} MeV (unreliable)")
ax.set_xlabel("Reconstructed distance from water top to beam (mm)")
ax.set_ylabel("Acoustic signal $A$ (mV)")
ax.set_title("Reconstructed longitudinal beam profile")
ax.legend(loc="best")
style_axes(ax)
save_fig(fig, "F2_longitudinal_profile")
plt.show()


def plot_beam_profile_and_fit(profile_data,
                              beam_axis_from_bottom=46.0,
                              sensor_level=78.0,
                              water_level=93.0,
                              reflect=True,
                              drop_above_115_for_115MeV=True,
                              fname_profile="F3_beam_profile_simple",
                              fname_fit="F4_A_vs_dist_powerfit",
                              show_legend_outside=True):

    import numpy as np

    # ---------- Plot 1: beam cross-section longitudinal view ----------
    # Collect x-range across energies (use nominal D)
    all_D = np.hstack([np.array(p["D"], dtype=float) for p in profile_data.values()])
    all_D = all_D[np.isfinite(all_D)]
    if all_D.size == 0:
        print("No D coordinates found in profile_data. Aborting.")
        return
    max_D = float(np.nanmax(all_D))
    x_margin = max(5.0, max_D * 0.03)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))

    # Water band
    ax.fill_betweenx([0.0, water_level], -0.0, max_D + x_margin,
                     color="lightblue", alpha=0.12, zorder=0)

    # horizontal references
    ax.axhline(0.0, color="k", lw=1.0)                     # bottom
    ax.axhline(water_level, color="k", lw=1.0, ls="--")    # water level
    ax.axhline(sensor_level, color="k", lw=1.0, ls=":")    # sensor line
    ax.axhline(beam_axis_from_bottom, color="0.3", lw=1.0, ls="-.")  # beam axis

    # vertical entrance at x=0 (right)
    ax.axvline(0.0, color="k", lw=1.0, ls="--")

    for i, (energy, pdict) in enumerate(sorted(profile_data.items())):
        D = np.array(pdict["D"], dtype=float)
        dist = np.array(pdict["dist"], dtype=float)      # distance from sensor towards bottom
        A = np.array(pdict.get("A", np.full_like(D, np.nan)), dtype=float)
        # drop per-energy points if requested: specifically drop D>115 for energy==115
        keep = np.isfinite(D) & np.isfinite(dist)
        if drop_above_115_for_115MeV and float(energy) == 115.0:
            keep = keep & (D <= 115.0)
        if not np.any(keep):
            continue

        D = D[keep]
        dist = dist[keep]
        A = A[keep]

        # vertical position from tank bottom:
        y = sensor_level - dist   # sensor at 78 mm; dist measured downwards -> y lower = closer to bottom

        # order by D for plotting
        order = np.argsort(D)
        c = COLORS[i % len(COLORS)]
        ax.plot(D[order], y[order], '-o', color=c, markeredgecolor='k', lw=1.6, label=f"{int(energy)} MeV")

        # reflection about beam axis (vertical reflection)
        if reflect:
            y_ref = 2.0 * beam_axis_from_bottom - y
            ax.plot(D[order], y_ref[order], color=c, alpha=0.7)

    # axis labels and orientation
    ax.set_xlabel("Water layer (mm)")
    ax.set_ylabel("Vertical position from tank bottom (mm)")
    ax.set_title("Beam profile vs longitudinal distance")

    # invert x so 0 is on the right and larger D on the left; start at x=0 on the right exactly
    ax.set_xlim(max_D + x_margin, 0.0)
    ypad = (water_level - 0.0) * 0.06
    ax.set_ylim(-ypad, water_level + ypad)

    # place legend outside to the right where there will be no datapoints
    if show_legend_outside:
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.)
    else:
        ax.legend(loc="upper left")
    style_axes(ax)
    save_fig(fig, fname_profile)
    plt.show()

    # ---------- Plot 2: A vs dist with power-law fit per energy ----------
    fig, ax = plt.subplots(figsize=(7.0, 5.0))

    for i, (energy, pdict) in enumerate(sorted(profile_data.items())):
        D = np.array(pdict["D"], dtype=float)
        dist = np.array(pdict["dist"], dtype=float)
        A = np.array(pdict.get("A", np.full_like(D, np.nan)), dtype=float)
        Aerr = np.array(pdict.get("Aerr", np.full_like(A, np.nan)), dtype=float)

        # same per-energy filtering as above
        keep = np.isfinite(dist) & np.isfinite(A)
        if drop_above_115_for_115MeV and float(energy) == 115.0:
            keep = keep & (D <= 115.0)
        if np.sum(keep) == 0:
            continue

        xdata = dist[keep]    # this is the last-column distance
        ydata = A[keep]
        yerr = Aerr[keep] if np.any(np.isfinite(Aerr[keep])) else None

        # require at least 3 points to fit a power law robustly
        if np.sum(np.isfinite(xdata) & np.isfinite(ydata) & (ydata > 0)) < 3:
            ax.errorbar(xdata, ydata, yerr=yerr, fmt='o', color=COLORS[i % len(COLORS)], label=f"{int(energy)} MeV (data)")
            print(f"  [{energy:.0f} MeV] not enough points for fit; plotting data only.")
            continue

        # fit power law A = A0 * d^-n using existing power_law & fit_curve helpers
        mask = np.isfinite(xdata) & np.isfinite(ydata) & (xdata > 0) & (ydata > 0)
        xfit = xdata[mask]
        yfit = ydata[mask]
        try:
            p0 = [np.max(yfit) * (xfit.min() ** 1.0), 1.0]
            popt, perr, r2 = fit_curve(power_law, xfit, yfit, yerr=yerr[mask] if yerr is not None else None, p0=p0)
            A0, n = popt
            dA0, dn = perr
            # plot data and fit curve
            ax.errorbar(xdata, ydata, yerr=yerr, fmt='o', color=COLORS[i % len(COLORS)], alpha=0.85)
            Xfine = np.linspace(np.min(xfit)*0.95, np.max(xfit)*1.05, 250)
            ax.plot(Xfine, power_law(Xfine, *popt), '-', color=COLORS[i % len(COLORS)], lw=1.6,
                    label=(f"{int(energy)} MeV fit: A0={A0:.3g}, n={n:.3f}, R2={r2:.3f}"))
            print(f"  [{energy:.0f} MeV] fit: A0={A0:.3g} +/- {dA0:.3g}, n={n:.3f} +/- {dn:.3f}, R2={r2:.4f}")
        except Exception as e:
            ax.errorbar(xdata, ydata, yerr=yerr, fmt='o', color=COLORS[i % len(COLORS)], label=f"{int(energy)} MeV (data)")
            print(f"  [{energy:.0f} MeV] fit failed: {e}")

    ax.set_xlabel("Distance from sensor line (mm)")
    ax.set_ylabel("Acoustic signal A (mV)")
    ax.set_title("A vs distance from sensor (power-law fits)")
    # legend outside to the right
    if show_legend_outside:
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.)
    else:
        ax.legend(loc="upper right")
    style_axes(ax)
    save_fig(fig, fname_fit)
    plt.show()
    
    
plot_beam_profile_and_fit(profile_data,
                          beam_axis_from_bottom=46.0,
                          sensor_level=78.0,
                          water_level=93.0,
                          reflect=True,
                          drop_above_115_for_115MeV=True,
                          fname_profile="F3_beam_profile_simple",
                          fname_fit="F4_A_vs_dist_powerfit")


# %% [9] SECTION G -- OPTIONAL / NOT YET RELEVANT DIAGNOSTICS ---------------
# Kept ready to switch on; none of this runs by default.
 
# --- G1: attenuation-based cross-checks (redundant with A vs Q, but handy
#         for calibrating/validating the attenuator itself) ----------------
# fig, ax = plt.subplots(figsize=(6, 4.5))
# att = df_sb["Attenuation (%)"].values
# ax.errorbar(att, y, yerr=yerr, fmt='o', color=COLORS[0], ecolor='0.4', capsize=3)
# ax.set_xlabel("Attenuation (%)")
# ax.set_ylabel("Acoustic signal $A$ (mV)")
# ax.set_title("Acoustic signal vs. attenuator setting")
# style_axes(ax)
# save_fig(fig, "G1_A_vs_attenuation")
# plt.show()
 
# %% [G2] DOSE CONVERSION (charge -> dose, via k and a beam area) -----------
# Dose = k * charge_density = k * Q / Area.
#
# CAVEAT (read before using any number this produces):
# k = 190 Gy/(nC/mm^2) is NOT an independently validated coefficient for
# this setup -- see film_dose_and_charge_calibration.py. Treat every dose
# value from this section as provisional pending real YAG:Ce screen data.
#
# CAVEAT ON AREA: only the vertical-position scan ("A vs. d vertical")
# has a per-point measured beam area. None of the other sheets (single/
# multi-bunch, depth scans) recorded area for those specific shots. For
# those, NOMINAL_AREA_MM2 (the vertical scan's mean measured area) is used
# as an explicit stand-in -- NOT a per-shot measurement. Its spread is
# carried as a separate systematic band, not folded into the statistical
# error bars, so the two sources of uncertainty stay distinguishable.

""" 
K_COEFF = 190.0          # Gy/(nC/mm^2) -- provisional, see caveat above
K_RANGE = (170.0, 250.0)  # literature spread, shown as a systematic band
 
NOMINAL_AREA_MM2 = area.mean()       # from the vertical-position scan, Section D
NOMINAL_AREA_STD_MM2 = area.std()    # shot-to-shot spread in that scan
 
 
def charge_to_dose(Q_nC, Q_err_nC, area_mm2, area_err_mm2=0.0, k=K_COEFF):
    
    #Dose = k * Q / Area, with statistical error propagation from Q and
    #Area only. k's own uncertainty is NOT included here -- report the
    #K_RANGE band separately as a systematic, not a statistical, error.
    
    dose = k * Q_nC / area_mm2
    Q_err_ = np.nan_to_num(Q_err_nC, nan=0.0)
    area_err_ = np.nan_to_num(area_err_mm2, nan=0.0)
    dose_err = np.abs(dose) * np.sqrt((Q_err_ / Q_nC) ** 2 +
                                       (area_err_ / area_mm2) ** 2)
    return dose, dose_err
 
 
def dose_k_band(Q_nC, area_mm2, k_range=K_RANGE):
    #Dose at the low/high ends of the literature k range -- a systematic
    #band, separate from (and typically much larger than) the statistical
    #error from charge_to_dose().
    lo = k_range[0] * Q_nC / area_mm2
    hi = k_range[1] * Q_nC / area_mm2
    return lo, hi
 
 
print("\n=== Section G2: charge -> dose conversion ===")
print(f"  k = {K_COEFF} Gy/(nC/mm^2) (provisional -- see caveat)")
print(f"  Nominal area (from vertical scan, used where no per-shot area "
      f"exists): {NOMINAL_AREA_MM2:.2f} +/- {NOMINAL_AREA_STD_MM2:.2f} mm^2")
 
# --- Example: A vs Q (single bunch) with a secondary provisional dose ------
# axis. Reuses the Section A linear fit, since A ~ Q and Dose ~ Q (nominal
# area held constant) means Dose is an affine function of A.
dose_sb, dose_sb_err = charge_to_dose(x, xerr, NOMINAL_AREA_MM2)
dose_lo, dose_hi = dose_k_band(x, NOMINAL_AREA_MM2)
 
fig, ax = plt.subplots(figsize=(6.4, 4.7))
ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt='o', color=COLORS[0],
            ecolor='0.4', capsize=3, label="Data (single bunch)")
xfit = np.linspace(0, x.max() * 1.08, 200)
ax.plot(xfit, linear(xfit, *popt_sb), '-', color=COLORS[1], lw=1.8,
        label="Linear fit (Section A)")
ax.set_xlabel("Charge $Q$ (nC)")
ax.set_ylabel("Acoustic signal $A$ (mV)")
ax.set_title("Acoustic signal vs. charge, with provisional dose scale")
ax.legend(loc="upper left")
 
 
def A_to_dose(A_val):
    #A -> Q (via the Section A linear fit) -> Dose (via nominal area).
    Q_ = (A_val - b_sb) / m_sb
    return K_COEFF * Q_ / NOMINAL_AREA_MM2
 
 
def dose_to_A(dose_val):
    Q_ = dose_val * NOMINAL_AREA_MM2 / K_COEFF
    return m_sb * Q_ + b_sb
 
 
ax2 = ax.secondary_yaxis('right', functions=(A_to_dose, dose_to_A))
ax2.set_ylabel("Provisional dose (Gy), $k$=190")
style_axes(ax)
fig.subplots_adjust(right=0.86)
save_fig(fig, "G2_A_vs_Q_with_dose_axis")
plt.show()
 
print(f"  Example: at Q={x.max():.3f} nC, dose = {dose_sb[-1]:.2f} +/- "
      f"{dose_sb_err[-1]:.2f} Gy (statistical), systematic band "
      f"[{dose_lo[-1]:.2f}, {dose_hi[-1]:.2f}] Gy from k alone "
      f"(170-250 Gy/(nC/mm^2)), on top of the area/k caveats above.")
 
# ---------------------------------------------------------------------------
# To add the same dose scale to any other plot (e.g. the depth scans in
# Section E): compute dose, dose_err = charge_to_dose(Q, Qerr, NOMINAL_AREA_MM2)
# for that sheet's Q values, then either plot dose directly instead of A, or
# add a secondary_yaxis the same way as above using that sheet's own A-vs-Q
# fit parameters (or Q directly, if you'd rather skip the A intermediary).
# ---------------------------------------------------------------------------
 
# --- G3: temperature dependence (once temperature is logged per test) -----
# df_T = load_sheet("A vs. T", split_cols=["A (mV)"])
# fig, ax = plt.subplots(figsize=(6, 4.5))
# ax.errorbar(df_T["T (C)"], df_T["A (mV)_val"], yerr=df_T["A (mV)_err"],
#             fmt='o', color=COLORS[0], ecolor='0.4', capsize=3)
# ax.set_xlabel("Water temperature (°C)")
# ax.set_ylabel("Acoustic signal $A$ (mV)")
# ax.set_title("Acoustic signal vs. water temperature")
# style_axes(ax)
# save_fig(fig, "G3_A_vs_temperature")
# plt.show()

"""

# %% [10] SUMMARY OF FIT RESULTS ----------------------------------------------
print("\n=== Summary ===")

summary_rows = [
    {"Test": "A vs Q (single bunch)", "Model": "linear",
     "Param 1": f"m={m_sb:.2f}+/-{dm_sb:.2f} mV/nC",
     "Param 2": f"b={b_sb:.2f}+/-{db_sb:.2f} mV", "R2": round(r2_sb, 4)},
    {"Test": "A vs Q (multi bunch)", "Model": "linear",
     "Param 1": f"m={m_mb:.2f}+/-{dm_mb:.2f} mV/nC",
     "Param 2": f"b={b_mb:.2f}+/-{db_mb:.2f} mV", "R2": round(r2_mb, 4)},
]
for energy, (popt_d, perr_d, r2_d) in sorted(fit_results.items()):
    summary_rows.append({
        "Test": f"A vs D ({energy:.0f} MeV)", "Model": "exp. decay",
        "Param 1": f"lambda={popt_d[1]:.1f}+/-{perr_d[1]:.1f} mm",
        "Param 2": f"A0={popt_d[0]:.1f}+/-{perr_d[0]:.1f} mV",
        "R2": round(r2_d, 4),
    })

summary_df = pd.DataFrame(summary_rows)
print(summary_df.to_string(index=False))
summary_df.to_csv(OUTPUT_DIR / "fit_summary.csv", index=False)
print(f"\nFit summary written to {OUTPUT_DIR / 'fit_summary.csv'}")
print(f"All figures written to: {OUTPUT_DIR.resolve()}")
