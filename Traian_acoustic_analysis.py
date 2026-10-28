"""
==============================================================================
Traian_acoustic_analysis.py

Acoustic dosimetry of VHEE beams in water phantom


Reads the measurement workbook and produces fits and plots in the following order:

  A) Acoustic signal (A) vs. charge (Q)         - single-bunch attenuation scan
  B) Acoustic signal (A) vs. charge (Q)         - multi-bunch scan
  C) Acoustic signal (A) vs. number of bunches, charge held constant
  D) Acoustic signal (A) vs. vertical beam position, + beam area (FWHM ellipse)
  E) Acoustic signal (A) vs. longitudinal (depth) position, compared across
     beam energies, plus the normalised A/Q ratio vs. depth
  F) Longitudinal scans

Run the desired portion individually or altogether 

Author: Traian (CLEAR / John Adams Institute / Oxford)
==============================================================================
"""

# %% [0] SETUP --------------------------------------------------------------

import re                                       # regex utilities
from pathlib import Path                         # path objects
import numpy as np                               # numerical arrays
import pandas as pd                              # dataframes & Excel IO
import matplotlib.pyplot as plt                  # plotting
from matplotlib.ticker import AutoMinorLocator   # minor tick locator
from scipy.optimize import curve_fit             # fitting helper



DATA_FILE = Path("C:/Users/Traian/Desktop/CERN/Python code/Measurements_4.xlsx")  # input workbook

OUTPUT_DIR = Path("C:/Users/Traian/Desktop/CERN/Python code/figures")             # output folder

OUTPUT_DIR.mkdir(exist_ok=True)  # create output dir if absent


# global plotting style: publication-friendly defaults for fonts, ticks and grids
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
})  # update rcParams


# grab the default matplotlib color cycle for consistent use later (first use only)
COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]  # color list


# small helper: add minor ticks for linear axes (call after plotting)
def style_axes(ax):
    if ax.get_xscale() == "linear":
        ax.xaxis.set_minor_locator(AutoMinorLocator())  # add minor ticks on x
    if ax.get_yscale() == "linear":
        ax.yaxis.set_minor_locator(AutoMinorLocator())  # add minor ticks on y
    return ax  # return axis for chaining


# small helper: save a figure as PNG and PDF into OUTPUT_DIR and print confirmation
def save_fig(fig, name):
    fig.tight_layout()  # adjust layout
    fig.savefig(OUTPUT_DIR / f"{name}.png")  # save png
    fig.savefig(OUTPUT_DIR / f"{name}.pdf")  # save pdf
    print(f"  saved -> {OUTPUT_DIR / name}.png / .pdf")  # console confirmation


# %% [1] DATA LOADING & "value ± error" PARSING ------------------------------

# parse_measurement accepts numbers or strings like "38.8 ± 3.3" and returns (value, error)
def parse_measurement(cell):
    if pd.isna(cell):
        return np.nan, np.nan  # preserve missing values
    if isinstance(cell, (int, float, np.integer, np.floating)):
        return float(cell), np.nan  # numeric cell, no reported error
    s = str(cell).strip().replace(",", ".")  # normalize string and decimal comma
    if "±" in s:
        v, e = s.split("±")  # split into value and error components
        return float(v.strip()), float(e.strip())  # return numeric pair
    try:
        return float(s), np.nan  # single numeric string, no error provided
    except ValueError:
        return np.nan, np.nan  # non-numeric content -> NaNs


# load_sheet reads an Excel sheet and expands any split_cols of the "value ± error" form
def load_sheet(sheet_name, split_cols=()):
    df = pd.read_excel(DATA_FILE, sheet_name=sheet_name)  # read sheet into DataFrame
    for col in split_cols:
        vals, errs = zip(*df[col].map(parse_measurement)) # parse each cell into (value, err)
        df[f"{col}_val"] = np.array(vals, dtype=float)    # store parsed values
        df[f"{col}_err"] = np.array(errs, dtype=float)    # store parsed errors
    return df  # return DataFrame with added numeric columns


# show which workbook will be processed and the available sheets
print(f"Loading workbook: {DATA_FILE.resolve()}")  # print absolute path
xls = pd.ExcelFile(DATA_FILE)  # open workbook for sheet enumeration
print("Sheets found:", xls.sheet_names)  # list sheets


# %% [2] FIT & ERROR-PROPAGATION HELPERS -------------------------------------

# linear model
def linear(x, m, b):
    return m * x + b  # compute m*x + b

# exponential decay model with offset
def exp_decay(x, A0, lam, y0):
    return A0 * np.exp(-x / lam) + y0  # A0 * exp(-x/lam) + offset

# power-law model
def power_law(x, A0, n):
    return A0 * np.power(x, -n)  # A0 * x^-n

# gaussian peak plus offset model
def gaussian(x, A, mu, sigma, offset):
    return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) + offset  # gaussian

# quadratic polynomial
def quadratic(x, a, b, c):
    return a * x ** 2 + b * x + c  # ax^2 + bx + c


# fit_curve wraps scipy curve_fit, fills missing yerr with mean of reported errors,
# returns best-fit parameters, their 1-sigma errors and the R^2 goodness-of-fit.
def fit_curve(func, x, y, yerr=None, p0=None, maxfev=20000):
    sigma = None
    if yerr is not None and np.any(np.isfinite(yerr)):
        fill = np.nanmean(yerr)  # replace missing yerr with mean of provided ones
        sigma = np.where(np.isfinite(yerr), yerr, fill)  # sigma array for curve_fit
    popt, pcov = curve_fit(func, x, y, p0=p0, sigma=sigma,
                            absolute_sigma=sigma is not None,
                            method="trf", max_nfev=maxfev)  # perform fit
    perr = np.sqrt(np.diag(pcov))  # parameter 1-sigma errors
    resid = y - func(x, *popt)  # residuals
    ss_res = np.sum(resid ** 2)  # residual sum of squares
    ss_tot = np.sum((y - np.mean(y)) ** 2)  # total sum of squares
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan  # R^2 if defined
    return popt, perr, r2  # return parameters, errors, R^2


# ratio_error propagates uncertainty for r = a / b
def ratio_error(a, sa, b, sb):
    r = a / b  # nominal ratio
    sa_ = np.nan_to_num(sa, nan=0.0)  # safe error values (NaN -> 0)
    sb_ = np.nan_to_num(sb, nan=0.0)
    sr = np.abs(r) * np.sqrt((sa_ / a) ** 2 + (sb_ / b) ** 2)  # prop. of errors
    return r, sr  # return ratio and its std dev


# product_error propagates uncertainty for v = k * a * b
def product_error(a, sa, b, sb, k=1.0):
    v = k * a * b  # product value
    sa_ = np.nan_to_num(sa, nan=0.0)  # safe error arrays
    sb_ = np.nan_to_num(sb, nan=0.0)
    sv = np.abs(v) * np.sqrt((sa_ / a) ** 2 + (sb_ / b) ** 2)  # propagated error
    return v, sv  # return value and propagated uncertainty


# %% [3] SECTION A -- A vs Q, SINGLE BUNCH (attenuation scan) ---------------

# load the single-bunch attenuation sheet and expand A and Q "value ± error" columns
df_sb = load_sheet("A vs. Q single bunch", split_cols=["A (mV)", "Q (nC)"])

# sort rows by numeric Q value so plots and fits are ordered by increasing charge
df_sb = df_sb.sort_values("Q (nC)_val").reset_index(drop=True)

# numeric arrays for Q (value and error) and A (value and error) used throughout Section A
x, xerr = df_sb["Q (nC)_val"].values, df_sb["Q (nC)_err"].values
y, yerr = df_sb["A (mV)_val"].values, df_sb["A (mV)_err"].values

# fit a linear model A = m*Q + b to the single-bunch data using fit_curve (weights from yerr)
popt_sb, perr_sb, r2_sb = fit_curve(linear, x, y, yerr, p0=[70, 0])
m_sb, b_sb = popt_sb  # slope and intercept
dm_sb, db_sb = perr_sb  # 1-sigma uncertainties on slope and intercept
print(f"  A = ({m_sb:.2f} +/- {dm_sb:.2f}) * Q + ({b_sb:.2f} +/- {db_sb:.2f})   "
      f"R2 = {r2_sb:.4f}")  # print fit summary

# create figure: A vs Q with errorbars and linear fit line
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


# %%
# --- A3: Improved Charge vs Attenuation fits -----
# re-use df_sb if present, otherwise reload (keeps cell runnable independently)
if "df_sb" not in globals():
    df_sb = load_sheet("A vs. Q single bunch", split_cols=["A (mV)", "Q (nC)"])
    df_sb = df_sb.sort_values("Q (nC)_val").reset_index(drop=True)

# attenuation percent, numeric Q and its error taken from parsed columns
att = df_sb["Attenuation (%)"].astype(float).values
Q_val = df_sb["Q (nC)_val"].astype(float).values
Q_err = df_sb["Q (nC)_err"].astype(float)

# Saturation function: captures a low-att plateau, a knee at KA and asymptotic low value
def hill(x, Qmax, KA, n, Qmin):
    # use absolute KA to avoid sign issues; zero replaced with 1 to prevent division by zero
    KA = np.abs(KA) if KA != 0 else 1.0
    return Qmin + (Qmax - Qmin) / (1.0 + (x / KA) ** n)

ATT_MODEL = "hill"

# perform the hill fit with a reasonable initial guess and print parameters if successful
if ATT_MODEL == "hill":
    p0 = [np.nanmax(Q_val), 10.0, 2.0, np.nanmin(Q_val)]
    try:
        popt, perr, r2 = fit_curve(hill, att, Q_val, Q_err, p0=p0)
        Qmax, KA, n, Qmin = popt
        print(f"Hill fit: Qmax={Qmax:.3g} +/- {perr[0]:.3g}, KA={KA:.3g} +/- {perr[1]:.3g}, n={n:.3g} +/- {perr[2]:.3g}, Qmin={Qmin:.3g} +/- {perr[3]:.3g}, R2={r2:.4f}")
        fit_fn = lambda x: hill(x, *popt)
        fit_label = fr"Hill fit: $Q_{{max}}={Qmax:.3g}, k={KA:.3g}, n={n:.2f}, Q_{{min}}={Qmin:.3g}$"
    except Exception as e:
        print("Hill fit failed:", e)
        popt = None

# plot attenuation vs Q with the chosen fit overlay (if available)
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

# load multi-bunch sheet; parse A and use raw Q numeric values (no Q error provided)
df_mb = load_sheet("A vs. Q multi bunch", split_cols=["A (mV)"])
df_mb["Q_val"] = df_mb["Q (nC)"].astype(float)  # Q provided without ±error in this sheet
df_mb["Q_err"] = np.nan  # no per-point Q uncertainty present
df_mb = df_mb.sort_values("Q_val").reset_index(drop=True)  # sort by Q

# arrays for plotting and fitting
xm, xerrm = df_mb["Q_val"].values, df_mb["Q_err"].values
ym, yerrm = df_mb["A (mV)_val"].values, df_mb["A (mV)_err"].values

# linear fit A vs Q for multi-bunch regime (weighted by yerr if present)
popt_mb, perr_mb, r2_mb = fit_curve(linear, xm, ym, yerrm, p0=[70, 0])
m_mb, b_mb = popt_mb  # slope and intercept
dm_mb, db_mb = perr_mb
print(f"  A = ({m_mb:.2f} +/- {dm_mb:.2f}) * Q + ({b_mb:.2f} +/- {db_mb:.2f})   "
      f"R2 = {r2_mb:.4f}")  # print fit summary

# plot multi-bunch data and linear fit
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

# overlay single- and multi-bunch for comparison using previously computed fits
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

# multi-series comparison: list of sheets, legend label, color, marker and model to fit
MULTI_SERIES = [
    ("A vs. Q multi bunch",       "Q = 0.5 nC", COLORS[0], 'o', "rise_decay"),
    ("A vs. Q multi bunch 1.1nC", "Q = 1.1 nC", "#2ca02c", 's', "rise_decay"),
    ("A vs. Q multi bunch 1.4nC", "Q = 1.4 nC", "#9467bd", '^', "saturating"),
]

# rise-then-decay functional form used previously in Section C
def rise_decay(x, A0, N0, N1, offset):
    return A0 * (1 - np.exp(-x / N0)) * np.exp(-x / N1) + offset

# saturating function alternative
def saturating(x, A_max, N0, offset):
    return A_max * (1 - np.exp(-x / N0)) + offset

# helper to produce a safe sigma array or None if no valid yerr values provided
def safe_sigma(yerr):
    yerr = np.asarray(yerr, dtype=float)
    valid = np.isfinite(yerr) & (yerr > 0)
    if not np.any(valid):
        return None  # no valid per-point uncertainties
    fill = np.mean(yerr[valid])
    return np.where(valid, yerr, fill)  # replace NaNs by mean error

# prepare container for fit summaries and figure canvas
multi_results = {}
fig, ax = plt.subplots(figsize=(7.2, 5.0))

# loop through each configured series, load sheet, plot data and fit if enough points
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

    # print mean Q when present for diagnostic info
    if "Q (nC)" in df.columns:
        qvals, _ = zip(*df["Q (nC)"].map(parse_measurement))
        print(f"  '{sheet}': mean Q = {np.nanmean(qvals):.3f} nC "
              f"(std {np.nanstd(qvals):.3f} nC)")

    # mask finite entries and plot them with errorbars using the provided color/marker
    mask = np.isfinite(n_b) & np.isfinite(A_vals)
    n_fit, A_fit, Aerr_fit = n_b[mask], A_vals[mask], A_errs[mask]

    ax.errorbar(n_fit, A_fit, yerr=Aerr_fit, fmt=mk, color=c, ecolor='0.6',
                capsize=3, ms=6, label=label, zorder=3)

    # require minimum number of points to attempt a robust non-linear fit
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

# finalize the multi-series plot and save
ax.set_xlabel("Number of bunches $N$")
ax.set_ylabel("Acoustic signal $A$ (mV)")
ax.set_title("Acoustic signal vs. number of bunches")
ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
style_axes(ax)
save_fig(fig, "C_section_C_A_vs_bunches_three_charges")
plt.show()

# print summary of fits if any were successful
if multi_results:
    print("\nSummary of fits:")
    for sheet, (model_name, popt, perr, r2) in multi_results.items():
        if model_name == "rise_decay":
            print(f"  {sheet} [rise_decay]: A0={popt[0]:.1f}+/-{perr[0]:.1f}, "
                  f"N0={popt[1]:.1f}+/-{perr[1]:.1f}, N1={popt[2]:.1f}+/-{perr[2]:.1f}, "
                  f"R2={r2:.4f}")
        else:
            print(f"  {sheet} [saturating]: A_max={popt[0]:.1f}+/-{perr[0]:.1f}, "
                  f"N0={popt[1]:.1f}+/-{perr[1]:.1f}, R2={r2:.4f}")


# %% [6] SECTION D -- VERTICAL BEAM POSITION & BEAM SIZE ---------------------

# fixed assumption for per-FWHM measurement error (applied to FWHM x and y alike)
FWHM_ERR = 0.25  # mm, fixed assumption (given), applied to FWHM x and y alike

# load the vertical-scan sheet and parse the A column (value ± error)
df_v = load_sheet("A vs. d vertical", split_cols=["A (mV)"])

# ensure rows are sorted by detector position for plotting and fits
df_v = df_v.sort_values("d at DETECTOR (mm)").reset_index(drop=True)

# numeric arrays used in plotting and fits: detector position, A and its error
dv = df_v["d at DETECTOR (mm)"].values  # detector vertical positions
Av, Averr = df_v["A (mV)_val"].values, df_v["A (mV)_err"].values  # acoustic signal and error

# FWHM columns (beam size estimates) used to compute beam area
fwhm_x = df_v["FWHM x (mm)"].values  # FWHM in x
fwhm_y = df_v["FWHM y (mm)"].values  # FWHM in y
area = df_v["Area estimate (mm^2)"].values  # precomputed area estimate (may be present)

# error propagation for area = (pi/4) * FWHMx * FWHMy using product_error helper
_, area_err = product_error(fwhm_x, np.full_like(fwhm_x, FWHM_ERR),
                             fwhm_y, np.full_like(fwhm_y, FWHM_ERR),
                             k=np.pi / 4)  # area error per point

# -----------------------------------------------------------------------
# D1a -- Acoustic signal vs. vertical position: points only, no fit -----
fig, ax = plt.subplots(figsize=(6, 4.5))  # create figure and axes
ax.errorbar(dv, Av, yerr=Averr, fmt='o-', color=COLORS[0], ecolor='0.4',
            capsize=3, label="Data (points joined)")  # plot data with errorbars
ax.set_xlabel("Vertical position at detector, $d$ (mm)")  # x label
ax.set_ylabel("Acoustic signal $A$ (mV)")  # y label
ax.set_title("Acoustic signal vs. vertical beam position")  # title
ax.legend(loc="best")  # legend
style_axes(ax)  # add minor ticks
save_fig(fig, "D1a_A_vs_vertical_position_points")  # save fig
plt.show()  # display


# -----------------------------------------------------------------------
# D1b -- Acoustic signal vs. vertical position: piecewise rise + drop ---
i_peak = np.argmax(Av)  # index of the last "rise" point / observed peak
d_rise, A_rise, Aerr_rise = dv[:i_peak + 1], Av[:i_peak + 1], Averr[:i_peak + 1]  # rising segment
d_drop, A_drop = dv[i_peak:], Av[i_peak:]  # drop segment points


# small exponential rise model used only for the rising segment fit
def exp_rise(x, A0, tau, off):
    return A0 * np.exp((x - dv.min()) / tau) + off  # exponential rise anchored at dv.min()

# fit the rising segment and compute the drop slope between the final two points
popt_r, perr_r, r2_r = fit_curve(exp_rise, d_rise, A_rise, Aerr_rise,
                                  p0=[10, 3, A_rise.min()])  # fit rise segment
drop_slope = (A_drop[-1] - A_drop[0]) / (d_drop[-1] - d_drop[0])  # direct slope for drop (2 points)
print(f"  Rise segment (first {i_peak + 1} points): exponential fit "
      f"A0={popt_r[0]:.2f}+/-{perr_r[0]:.2f}, tau={popt_r[1]:.2f}+/-{perr_r[1]:.2f} mm, "
      f"R2={r2_r:.4f}")  # print rise fit summary
print(f"  Drop segment (last 2 points): direct line, slope={drop_slope:.1f} mV/mm "
      f"(not a statistical fit -- only 2 points define it)")  # print drop description

# plotting rising fit and direct drop line together with data
fig, ax = plt.subplots(figsize=(6, 4.5))
ax.errorbar(dv, Av, yerr=Averr, fmt='o', color=COLORS[0], ecolor='0.4',
            capsize=3, zorder=3, label="Data")  # data points
d_rise_fit = np.linspace(d_rise.min(), d_rise.max(), 200)  # fine x for fit curve
ax.plot(d_rise_fit, exp_rise(d_rise_fit, *popt_r), '-', color=COLORS[1], lw=1.8,
        zorder=2, label=fr"Exponential rise fit (first {i_peak + 1} pts): $R^2={r2_r:.3f}$")  # rising fit
ax.plot(d_drop, A_drop, '--', color=COLORS[3], lw=1.8, zorder=2,
        label=fr"Drop: direct line, slope$={drop_slope:.0f}$ mV/mm")  # drop as a direct line
ax.set_xlabel("Vertical position at detector, $d$ (mm)")  # x label
ax.set_ylabel("Acoustic signal $A$ (mV)")  # y label
ax.set_title("Acoustic signal vs. vertical position — rise/drop fit")  # title
ax.legend(loc="best")  # legend
style_axes(ax)  # minor ticks
save_fig(fig, "D1b_A_vs_vertical_position_fit")  # save
plt.show()  # display


# -----------------------------------------------------------------------
# D2a -- Acoustic signal vs. beam area: points only, no fit -------------
order = np.argsort(area)  # order indices by area for plotting
fig, ax = plt.subplots(figsize=(6, 4.5))
ax.errorbar(area[order], Av[order], xerr=area_err[order], yerr=Averr[order],
            fmt='o-', color=COLORS[2], ecolor='0.4', capsize=3,
            label="Data (points joined, sorted by area)")  # plot A vs area
ax.set_xlabel(r"Beam area, $\pi/4 \cdot \mathrm{FWHM}_x \mathrm{FWHM}_y$ (mm$^2$)")  # x label
ax.set_ylabel("Acoustic signal $A$ (mV)")  # y label
ax.set_title("Acoustic signal vs. beam area")  # title
ax.legend(loc="best")  # legend
style_axes(ax)  # minor ticks
save_fig(fig, "D2a_A_vs_area_points")  # save
plt.show()  # display


# -----------------------------------------------------------------------
# D2b -- Acoustic signal vs. beam area: linear fit -----------------------
popt_a, perr_a, r2_a = fit_curve(linear, area, Av, Averr, p0=[-10, 300])  # linear fit A vs area
m_a, b_a = popt_a  # slope and intercept
dm_a, db_a = perr_a  # parameter errors
print(f"  A vs area: A = ({m_a:.2f}+/-{dm_a:.2f})*Area + ({b_a:.1f}+/-{db_a:.1f}), "
      f"R2={r2_a:.4f} (larger beam spot -> lower peak signal, as expected "
      f"if the same charge is spread over a larger area)")  # print fit result and interpretation

# plotting A vs area with linear fit overlaid
fig, ax = plt.subplots(figsize=(6, 4.5))
ax.errorbar(area, Av, xerr=area_err, yerr=Averr, fmt='o', color=COLORS[2],
            ecolor='0.4', capsize=3, label="Data")  # data
xfit_a = np.linspace(area.min() * 0.9, area.max() * 1.05, 200)  # fit x range
ax.plot(xfit_a, linear(xfit_a, *popt_a), '-', color=COLORS[3], lw=1.8,
        label=(fr"Linear fit: $A=({m_a:.1f}\pm{dm_a:.1f})\,\mathrm{{Area}}"
               fr"+({b_a:.0f}\pm{db_a:.0f})$" + "\n" + fr"$R^2={r2_a:.3f}$"))  # fit line
ax.set_xlabel(r"Beam area, $\pi/4 \cdot \mathrm{FWHM}_x \mathrm{FWHM}_y$ (mm$^2$)")  # x label
ax.set_ylabel("Acoustic signal $A$ (mV)")  # y label
ax.set_title("Acoustic signal vs. beam area — linear fit")  # title
ax.legend(loc="best")  # legend
style_axes(ax)  # minor ticks
save_fig(fig, "D2b_A_vs_area_fit")  # save
plt.show()  # display


# %% [6'] with DOSE

# this section computes per-point dose where Q and area are available and fits A and dose
print("\n=== Section D: A vs vertical position, A vs beam area (with A & Dose fits) ===")

# attempt to reuse df_v and related arrays created above; fallback to loading if not present
try:
    _ = df_v  # verify df_v exists
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

# obtain per-row Q if recorded in the sheet; otherwise fallback to estimate from Section A fit
if "Q (nC)_val" in df_v.columns and "Q (nC)_err" in df_v.columns:
    Q_val = df_v["Q (nC)_val"].values  # parsed numeric Q
    Q_err = df_v["Q (nC)_err"].values  # parsed Q error
elif "Q (nC)" in df_v.columns:
    vals, errs = zip(*df_v["Q (nC)"].map(parse_measurement))  # parse textual Q column
    Q_val = np.array(vals, dtype=float)
    Q_err = np.array(errs, dtype=float)
elif "Q_val" in df_v.columns:
    Q_val = df_v["Q_val"].astype(float).values
    Q_err = df_v.get("Q_err", np.full_like(Q_val, np.nan)).astype(float)
else:
    # fallback: use Section A linear fit A = m_sb*Q + b_sb to estimate Q from measured A
    print("Note: per-point Q not found for vertical scan — estimating Q from A using Section A fit.")
    Q_val = (Av - b_sb) / m_sb  # invert Section A linear fit to estimate Q
    Q_err = Averr / m_sb  # propagate A error to Q error approximately

# dose conversion constant (Gy per nC per mm^2) default if not defined elsewhere
k_calib = globals().get("K_COEFF", 190.0)  # calibration constant

# prepare arrays for dose and its uncertainty; mask_valid marks rows where Q & area are usable
dose = np.full_like(Av, np.nan)  # initialize with NaN
dose_err = np.full_like(Av, np.nan)  # initialize with NaN
mask_valid = np.isfinite(Q_val) & (Q_val != 0) & np.isfinite(area) & (area > 0)  # valid entries
if np.any(mask_valid):
    dose[mask_valid] = k_calib * Q_val[mask_valid] / area[mask_valid]  # dose = k * Q / area
    Qerr_ = np.nan_to_num(Q_err[mask_valid], nan=0.0)  # safe Q error
    area_err_ = np.nan_to_num(area_err[mask_valid], nan=0.0)  # safe area error
    dose_err[mask_valid] = np.abs(dose[mask_valid]) * np.sqrt(
        (Qerr_ / Q_val[mask_valid]) ** 2 + (area_err_ / area[mask_valid]) ** 2
    )  # propagate errors for division

# ----------------- fit settings (toggle here) -----------------
# choose fitting mode: "piecewise" fits rise segment only, "global" fits all points
FIT_MODE = "piecewise"   # "piecewise" or "global"

# choose per-axis model for A(d) and Dose(d): "rise_decay", "saturating", or "quadratic"
MODEL = "rise_decay"

# optionally override automatic rising segment length (use None to auto-select)
MANUAL_RISE_N = None  # set integer to force using first N points for rise fit
# --------------------------------------------------------------

# local model definitions for Section D fitting variations
def rise_decay_pos(x, A0, x0, x1, offset):
    return A0 * (1 - np.exp(-x / x0)) * np.exp(-x / x1) + offset  # rise then decay shape

def saturating_pos(x, Amax, x0, offset):
    return Amax * (1 - np.exp(-x / x0)) + offset  # saturating rise to plateau

# model selector returns callable and default initial guess
def get_model_func(name):
    if name == "rise_decay":
        return rise_decay_pos, [np.max(Av) - np.min(Av), 2.0, 20.0, np.min(Av)]
    if name == "saturating":
        return saturating_pos, [np.max(Av) - np.min(Av), 2.0, np.min(Av)]
    if name == "quadratic":
        return quadratic, [ -0.1, 1.0, np.mean(Av) ]
    return rise_decay_pos, [np.max(Av) - np.min(Av), 2.0, 20.0, np.min(Av)]  # fallback


# determine rising-segment index automatically (last point before the drop)
i_peak = int(np.argmax(Av))  # index of observed peak in Av
if MANUAL_RISE_N is not None:
    i_use = max(1, MANUAL_RISE_N - 1)  # use manual override if provided
else:
    i_use = i_peak  # otherwise use the peak index


# ---------- Plot with fits ----------
fig, ax = plt.subplots(figsize=(6.2, 4.6))  # main figure for A and Dose
colA = "tab:blue"  # color for A traces
ax.errorbar(dv, Av, yerr=Averr, fmt='o', color=colA, ecolor='0.4', capsize=3, label="Acoustic A (mV)")  # A data
ax.set_xlabel("Vertical position at detector, $d$ (mm)")  # x label
ax.set_ylabel("Acoustic signal $A$ (mV)", color=colA)  # left y label
ax.tick_params(axis='y', labelcolor=colA)  # color left y ticks
ax.set_title("Acoustic signal and Dose vs vertical position (fits)")  # title

# Fit & plot A according to chosen FIT_MODE and MODEL
if FIT_MODE == "piecewise":
    x_rise = dv[:i_use + 1].astype(float)  # x values for rising fit
    y_rise = Av[:i_use + 1].astype(float)  # y values for rising fit
    yerr_rise = Averr[:i_use + 1].astype(float)  # y errors for rising fit
    model_func, p0 = get_model_func(MODEL)  # get model and initial guess
    try:
        poptA, perrA, r2A = fit_curve(model_func, x_rise, y_rise, yerr=yerr_rise, p0=p0)  # fit rising segment
        xfine = np.linspace(x_rise.min(), dv.max(), 250)  # fine x grid for plotting fit across full range
        ax.plot(xfine, model_func(xfine, *poptA), '-', color=colA, lw=1.6,
                label=(f"A fit ({MODEL}, rise): " + 
                       ", ".join([f"{v:.2g}±{e:.2g}" for v,e in zip(poptA, perrA)]) + f", R2={r2A:.3f}"))  # plot fit
        print(f"  A fit ({MODEL}, rise): popt={poptA}, perr={perrA}, R2={r2A:.4f}")  # print fit params
    except Exception as e:
        print("  A fit failed (piecewise):", e)  # report fit failure
else:
    model_func, p0 = get_model_func(MODEL)  # global fit across all dv
    try:
        poptA, perrA, r2A = fit_curve(model_func, dv.astype(float), Av.astype(float), yerr=Averr, p0=p0)  # fit all points
        xfine = np.linspace(dv.min(), dv.max(), 250)
        ax.plot(xfine, model_func(xfine, *poptA), '-', color=colA, lw=1.6,
                label=(f"A fit ({MODEL}): " +
                       ", ".join([f"{v:.2g}±{e:.2g}" for v,e in zip(poptA, perrA)]) + f", R2={r2A:.3f}"))
        print(f"  A fit ({MODEL}, global): popt={poptA}, perr={perrA}, R2={r2A:.4f}")  # print fit params
    except Exception as e:
        print("  A fit failed (global):", e)  # report fit failure

# Right axis: Dose plotted in red with its own fit mirroring A's approach
ax2 = ax.twinx()  # create twin y-axis for dose
colD = "tab:red"  # color for dose traces
if np.any(mask_valid):
    ax2.errorbar(dv[mask_valid], dose[mask_valid], yerr=dose_err[mask_valid],
                 fmt='s', color=colD, ecolor=colD, capsize=3, label="Dose (Gy)")  # dose points

    # prepare per-energy arrays for dose fitting
    D_dv = dv[mask_valid].astype(float)  # dv entries where dose is defined
    D_dose = dose[mask_valid].astype(float)  # dose values
    D_dose_err = dose_err[mask_valid].astype(float)  # dose errors

    if FIT_MODE == "piecewise":
        mask_rise = (np.arange(len(dv)) <= i_use) & mask_valid  # rising indices within valid mask
        x_rise_d = dv[mask_rise].astype(float)  # dose x for rising fit
        y_rise_d = dose[mask_rise].astype(float)  # dose y for rising fit
        yerr_rise_d = dose_err[mask_rise].astype(float)  # dose errors for rising fit
        model_func_d, p0d = get_model_func(MODEL)  # same model choices for dose
        try:
            if len(x_rise_d) >= 3:
                poptD, perrD, r2D = fit_curve(model_func_d, x_rise_d, y_rise_d, yerr=yerr_rise_d, p0=p0d)  # fit dose rising part
                xfine_d = np.linspace(x_rise_d.min(), dv[mask_valid].max(), 250)
                ax2.plot(xfine_d, model_func_d(xfine_d, *poptD), '--', color=colD, lw=1.6,
                         label=(f"Dose fit ({MODEL}, rise): " +
                                ", ".join([f"{v:.2g}±{e:.2g}" for v,e in zip(poptD, perrD)]) + f", R2={r2D:.3f}"))  # plot dose fit
                print(f"  Dose fit ({MODEL}, rise): popt={poptD}, perr={perrD}, R2={r2D:.4f}")  # print dose fit summary
            else:
                print("  Not enough dose points in rise segment to fit.")  # insufficient points
        except Exception as e:
            print("  Dose fit failed (piecewise):", e)  # report fit error
    else:
        model_func_d, p0d = get_model_func(MODEL)
        try:
            if len(D_dv) >= 3:
                poptD, perrD, r2D = fit_curve(model_func_d, D_dv, D_dose, yerr=D_dose_err, p0=p0d)  # global dose fit
                xfine_d = np.linspace(D_dv.min(), D_dv.max(), 250)
                ax2.plot(xfine_d, model_func_d(xfine_d, *poptD), '--', color=colD, lw=1.6,
                         label=(f"Dose fit ({MODEL}): " +
                                ", ".join([f"{v:.2g}±{e:.2g}" for v,e in zip(poptD, perrD)]) + f", R2={r2D:.3f}"))
                print(f"  Dose fit ({MODEL}, global): popt={poptD}, perr={perrD}, R2={r2D:.4f}")  # print dose fit summary
            else:
                print("  Not enough dose points to fit (need >=3).")  # insufficient points
        except Exception as e:
            print("  Dose fit failed (global):", e)  # report error
else:
    ax2.plot([], [], ' ', label=f"Dose (Gy), k={k_calib:.0f} (no per-point area/Q)")  # placeholder when no dose

ax2.set_ylabel("Dose (Gy)", color=colD)  # right-axis label
ax2.tick_params(axis='y', labelcolor=colD)  # color right ticks
ax2.spines["right"].set_color(colD)  # color right spine

# combine legends from both axes and place to the right of the main plot area
h1, l1 = ax.get_legend_handles_labels()  # left axis legend entries
h2, l2 = ax2.get_legend_handles_labels()  # right axis legend entries
ax.legend(h1 + h2, l1 + l2, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0., fontsize=9)  # combined legend

style_axes(ax)  # add minor ticks on both axes where linear
save_fig(fig, "D1a_A_and_Dose_vs_vertical_position_with_fits")  # save figure
plt.show()  # display


# %% NO-FIT plot (same data but without fitted curves) ------------------------
print("\n=== Section D: NO-FIT variant (A & Dose vs d, no model) ===")

fig, ax = plt.subplots(figsize=(6.2, 4.6))  # create figure
colA = "tab:blue"  # left axis color
ax.errorbar(dv, Av, yerr=Averr, fmt='o', color=colA, ecolor='0.4', capsize=3, label="Acoustic A (mV)")  # A data
ax.set_xlabel("Vertical position at detector (mm)")  # x label
ax.set_ylabel("Acoustic signal $A$ (mV)", color=colA)  # left y label
ax.tick_params(axis='y', labelcolor=colA)  # left tick color
ax.set_title("Detector response and dose across vertical positions")  # title

ax2 = ax.twinx()  # twin axis for dose
colD = "tab:red"  # right axis color
if np.any(mask_valid):
    ax2.errorbar(dv[mask_valid], dose[mask_valid], yerr=dose_err[mask_valid],
                 fmt='s', color=colD, ecolor=colD, capsize=3, label="Dose (Gy)")  # dose points
else:
    ax2.plot([], [], ' ', label=f"Dose (Gy), k={k_calib:.0f} (no per-point area/Q)")  # placeholder
ax2.set_ylabel("Dose (Gy)", color=colD)  # right y label
ax2.tick_params(axis='y', labelcolor=colD)  # right tick color
ax2.spines["right"].set_color(colD)  # color right spine

# combined legend placed below left to avoid overlapping data
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="lower left")  # legend
style_axes(ax)  # style axes
save_fig(fig, "D1a_A_and_Dose_vs_vertical_position_no_fit")  # save
plt.show()  # display


# %% 3D plot more appropriate?

# ----------------- Replacement 3D block: simple labels + optional surface fit -----------
print("\n=== Section D: 3D scatter of Acoustic A vs Distance & Area (optional surface fit) ===")
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (3D plotting toolkit import for backends)

# arrays reused from above: detector distance, area and acoustic amplitude
dv_arr = np.array(dv, dtype=float)               # distance at detector (mm)
area_arr = np.array(area, dtype=float)           # beam area (mm^2)
A_arr = np.array(Av, dtype=float)                # acoustic signal (mV)

# shift distances to strictly positive values for log-fit stability (used only in fits)
dv_shift = dv_arr - np.nanmin(dv_arr) + 1.0  # shift so the minimum is 1 mm for log-space fitting

# mask points suitable for 3D fit (positive A and area, finite shift)
mask3d = np.isfinite(A_arr) & (A_arr > 0) & np.isfinite(area_arr) & (area_arr > 0) & np.isfinite(dv_shift)
n_pts = np.sum(mask3d)  # number of valid points
print(f"  3D: found {n_pts} valid points for (distance, area, A).")  # diagnostic

# toggle to enable fitting the 3D surface (default is False for small datasets)
DO_3D_FIT = False  # set True to attempt log-linear surface fit

fig = plt.figure(figsize=(8.0, 6.0))  # create 3D figure
ax3 = fig.add_subplot(111, projection='3d')  # 3D axes

# scatter actual points in 3D
ax3.scatter(dv_arr[mask3d], area_arr[mask3d], A_arr[mask3d],
            color=COLORS[0], s=60, edgecolor='k', label="data")  # scatter points

# axis labels and title for 3D plot
ax3.set_xlabel("Distance (mm)")  # x label
ax3.set_ylabel("Beam area (mm$^2$)")  # y label
ax3.set_zlabel("Acoustic signal A (mV)")  # z label
ax3.set_title("3D: Acoustic signal vs Distance and Beam area")  # title

# optionally fit log(A) = a + b*log(dist_shift) + c*log(area) and plot surface
if DO_3D_FIT:
    if n_pts < 3:
        print("  Not enough points to fit surface (need >=3). Skipping surface fit.")  # insufficient points
    else:
        # design matrix for linear regression in log-space: [1, log(dist), log(area)]
        X = np.column_stack([np.ones(np.sum(mask3d)), np.log(dv_shift[mask3d]), np.log(area_arr[mask3d])])
        Y = np.log(A_arr[mask3d])  # response in log-space
        coeffs, *_ = np.linalg.lstsq(X, Y, rcond=None)  # least-squares solution
        a0, b_exp, c_exp = coeffs  # fitted coefficients
        Y_pred = X.dot(coeffs)  # predicted log(A)
        ss_res = np.sum((Y - Y_pred)**2)  # residual sum squares in log-space
        ss_tot = np.sum((Y - np.mean(Y))**2)
        r2_log = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan  # R^2 in log-space
        print(f"  3D fit (log-space): log(A) = {a0:.3f} + {b_exp:.3f}*log(dist_shift) + {c_exp:.3f}*log(area)")
        print(f"    => A ≈ {np.exp(a0):.3g} * dist_shift^{b_exp:.3f} * area^{c_exp:.3f}, R2(log)={r2_log:.4f}")

        # build grid for plotting the fitted surface back in linear A space
        d_grid = np.linspace(np.min(dv_arr[mask3d]), np.max(dv_arr[mask3d]), 40)
        a_grid = np.linspace(np.min(area_arr[mask3d]), np.max(area_arr[mask3d]), 40)
        Dg, Ag = np.meshgrid(d_grid, a_grid)
        Dg_shift = Dg - np.nanmin(dv_arr) + 1.0  # apply same shift used in fitting
        logA_surf = a0 + b_exp * np.log(Dg_shift) + c_exp * np.log(Ag)  # fitted log(A) surface
        A_surf = np.exp(logA_surf)  # convert back to linear A for plotting
        ax3.plot_surface(Dg, Ag, A_surf, cmap='viridis', alpha=0.45, linewidth=0, antialiased=True)  # plot surface

        # annotation with compact fit formula and R^2
        fit_text = f"A ≈ {np.exp(a0):.2g}·d^{b_exp:.2f}·area^{c_exp:.2f}\nR2_log={r2_log:.3f}"
        ax3.text2D(0.02, 0.95, fit_text, transform=ax3.transAxes, fontsize=9, verticalalignment='top',
                   bbox=dict(facecolor='white', alpha=0.6, edgecolor='0.8'))  # add text box

ax3.legend(loc="upper left")  # legend for 3D scatter
save_fig(fig, "D3D_A_vs_distance_area")  # save 3D figure
plt.show()  # display


# %% [7] SECTION E -- DEPTH (LONGITUDINAL) SCAN, ENERGY COMPARISON ----------

print("\n=== Section E: A vs depth D, all beam energies ===")

# detect all sheets named like "A vs D <energy> MeV" so new energies are auto-included
depth_pattern = re.compile(r"A\s*vs\.?\s*D\s*([\d.]+)\s*MeV", re.IGNORECASE)  # regex to extract energy
depth_sheets = {}
for sn in xls.sheet_names:
    match = depth_pattern.match(sn.strip())
    if match:
        depth_sheets[float(match.group(1))] = sn  # map energy value to sheet name
print("  Depth-scan sheets detected:", depth_sheets)  # print detected depth sheets

# read every detected depth sheet, parse A and Q and compute A/Q with propagated error
depth_data = {}
for energy, sn in sorted(depth_sheets.items()):
    df = load_sheet(sn, split_cols=["A (mV)", "Q (nC)"])  # load sheet with parsed A and Q
    df = df.sort_values("D (mm)").reset_index(drop=True)  # sort by depth D

    A_, Aerr_ = df["A (mV)_val"].values, df["A (mV)_err"].values  # A and error
    Q_, Qerr_ = df["Q (nC)_val"].values, df["Q (nC)_err"].values  # Q and error

    # compute A/Q and its propagated uncertainty using ratio_error helper
    AQ, AQerr = ratio_error(A_, Aerr_, Q_, Qerr_)
    df["A/Q"] = AQ  # store A/Q in DataFrame
    df["A/Q_err"] = AQerr  # store error

    depth_data[energy] = df  # add to depth_data dict keyed by energy
    print(f"    {energy:.0f} MeV: <Q> = {np.nanmean(Q_):.3f} +/- {np.nanstd(Q_):.3f} nC "
          f"(shot-to-shot spread; per-point errors are in the sheet)")  # info


# --- E1: A vs D, linear y-axis, all energies on one plot -------------------
fig, ax = plt.subplots(figsize=(6.6, 5))
fit_results = {}  # store per-energy fit results for later use
for i, (energy, df) in enumerate(sorted(depth_data.items())):
    D = df["D (mm)"].values.astype(float)  # depth positions
    A_, Aerr_ = df["A (mV)_val"].values, df["A (mV)_err"].values  # A and error
    c = COLORS[i % len(COLORS)]  # color per energy
    ax.errorbar(D, A_, yerr=Aerr_, fmt='o', color=c, ecolor='0.5', capsize=3,
                label=f"{energy:.0f} MeV (data)")  # plot data with errorbars

    popt_d, perr_d, r2_d = fit_curve(exp_decay, D, A_, Aerr_,
                                      p0=[A_.max(), 40, 0])  # fit exponential decay A(D)
    fit_results[energy] = (popt_d, perr_d, r2_d)  # store fit result
    Dfit = np.linspace(D.min(), D.max(), 300)
    ax.plot(Dfit, exp_decay(Dfit, *popt_d), '-', color=c, lw=1.6,
            label=fr"{energy:.0f} MeV fit: $\lambda={popt_d[1]:.1f}\pm{perr_d[1]:.1f}$ mm")  # plot exp fit
    print(f"  [{energy:.0f} MeV] exp. decay: A0={popt_d[0]:.1f}+/-{perr_d[0]:.1f} mV, "
          f"lambda={popt_d[1]:.1f}+/-{perr_d[1]:.1f} mm, y0={popt_d[2]:.1f}+/-{perr_d[2]:.1f} mV, "
          f"R2={r2_d:.4f}")  # print fit summary

ax.set_xlabel("Water-layer depth $D$ (mm)")  # x label
ax.set_ylabel("Acoustic signal $A$ (mV)")  # y label
ax.set_title("Acoustic signal vs. depth — beam-energy comparison")  # title
ax.legend(loc="upper right", ncol=1)  # legend
style_axes(ax)  # minor ticks
save_fig(fig, "E1_section_E_A_vs_D_energy_comparison")  # save
plt.show()  # display


# --- E1b: same data, log-y axis (a pure exponential is a straight line) ----
fig, ax = plt.subplots(figsize=(6.6, 5))
for i, (energy, df) in enumerate(sorted(depth_data.items())):
    D = df["D (mm)"].values.astype(float)
    A_, Aerr_ = df["A (mV)_val"].values, df["A (mV)_err"].values
    c = COLORS[i % len(COLORS)]
    ax.errorbar(D, A_, yerr=Aerr_, fmt='o', color=c, ecolor='0.5', capsize=3,
                label=f"{energy:.0f} MeV")  # plot data points
    popt_d, _, _ = fit_results[energy]  # reuse previously computed fit results
    Dfit = np.linspace(D.min(), D.max(), 300)
    ax.plot(Dfit, exp_decay(Dfit, *popt_d), '-', color=c, lw=1.6)  # plot fitted decay
ax.set_yscale("log")  # set log y-axis for straight-line exponential visualization
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
                label=f"{energy:.0f} MeV")  # plot A/Q with errorbars
ax.set_xlabel("Water-layer depth $D$ (mm)")
ax.set_ylabel(r"Normalised signal $A/Q$ (mV/nC)")
ax.set_title("Charge-normalised acoustic signal vs. depth")
ax.legend(loc="upper right")
style_axes(ax)
save_fig(fig, "E2_section_E_AoverQ_vs_D")
plt.show()


# %% [8] SECTION F -- RECONSTRUCTED LONGITUDINAL BEAM PROFILE --------------
print("\n=== Section F: reconstructed longitudinal beam profile ===")

# threshold for flagging peak-time disagreement between coarse and refined picks
PEAK_DISAGREEMENT_THRESHOLD = 0.10  # us

# reconstruct per-energy profile_data: store D, dist, dist_err, A, Aerr and a reliability flag
profile_data = {}
for energy, df in sorted(depth_data.items()):
    D = df["D (mm)"].values.astype(float)
    A_, Aerr_ = df["A (mV)_val"].values, df["A (mV)_err"].values
    peak_t = df["Peak times (us)"].values.astype(float)
    acou_t = df["Acoustic Signal pos (us)"].values.astype(float)
    diff_t = df["Difference (us)"].values.astype(float)
    dist = df["Distance from top to beam (mm)"].values.astype(float)

    # timing-uncertainty proxy from difference between coarse and refined peak picks
    t_err = np.abs(peak_t - acou_t)  # timing disagreement per point

    # recover an empirical speed-of-sound from the sheet's own time->distance pairing
    v_sound = np.sum(dist * diff_t) / np.sum(diff_t ** 2)  # mm per microsecond
    dist_err = t_err * v_sound  # convert time error to distance error

    reliable = t_err < PEAK_DISAGREEMENT_THRESHOLD  # flag reliable points
    profile_data[energy] = dict(D=D, dist=dist, dist_err=dist_err,
                                 A=A_, Aerr=Aerr_, reliable=reliable)  # store profile info

    print(f"  {energy:.0f} MeV: recovered v_sound = {v_sound * 1000:.1f} m/s; "
          f"reconstructed depth = {dist[reliable].mean():.2f} +/- "
          f"{dist[reliable].std():.2f} mm over the reliable points")  # print diagnostics
    if np.any(~reliable):
        print(f"    flagged unreliable (peak-time disagreement > "
              f"{PEAK_DISAGREEMENT_THRESHOLD} us) at D = "
              f"{D[~reliable].astype(int).tolist()} mm")  # list flagged depths


# --- F1: diagnostic -- reconstructed depth vs. nominal sensor position D ---
fig, ax = plt.subplots(figsize=(6.6, 4.8))
for i, (energy, pd_) in enumerate(sorted(profile_data.items())):
    c = COLORS[i % len(COLORS)]
    rel = pd_["reliable"]  # boolean mask of reliable points
    ax.errorbar(pd_["D"][rel], pd_["dist"][rel], yerr=pd_["dist_err"][rel],
                fmt='o-', color=c, ecolor='0.5', capsize=3,
                label=f"{energy:.0f} MeV (reliable)")  # reliable points
    if np.any(~rel):
        ax.errorbar(pd_["D"][~rel], pd_["dist"][~rel], yerr=pd_["dist_err"][~rel],
                     fmt='x', color=c, ecolor='0.7', capsize=3, alpha=0.5,
                     label=f"{energy:.0f} MeV (flagged unreliable)")  # flagged points
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
                capsize=3, label=f"{energy:.0f} MeV")  # reliable A vs dist points
    if np.any(~rel):
        ax.errorbar(pd_["dist"][~rel], pd_["A"][~rel], xerr=pd_["dist_err"][~rel],
                     yerr=pd_["Aerr"][~rel], fmt='x', color=c, ecolor='0.7',
                     capsize=3, alpha=0.4, label=f"{energy:.0f} MeV (unreliable)")  # flagged A points
ax.set_xlabel("Reconstructed distance from water top to beam (mm)")
ax.set_ylabel("Acoustic signal $A$ (mV)")
ax.set_title("Reconstructed longitudinal beam profile")
ax.legend(loc="best")
style_axes(ax)
save_fig(fig, "F2_longitudinal_profile")
plt.show()


# function to plot beam profile cross-section and per-energy power-law fits A vs distance
def plot_beam_profile_and_fit(profile_data,
                              beam_axis_from_bottom=46.0,
                              sensor_level=78.0,
                              water_level=93.0,
                              reflect=True,
                              drop_above_115_for_115MeV=True,
                              fname_profile="F3_beam_profile_simple",
                              fname_fit="F4_A_vs_dist_powerfit",
                              show_legend_outside=True):

    import numpy as np  # local import to ensure function is self-contained for plotting

    # ---------- Plot 1: beam cross-section longitudinal view ----------
    all_D = np.hstack([np.array(p["D"], dtype=float) for p in profile_data.values()])  # all D values
    all_D = all_D[np.isfinite(all_D)]  # finite filter
    if all_D.size == 0:
        print("No D coordinates found in profile_data. Aborting.")  # abort if no data
        return
    max_D = float(np.nanmax(all_D))  # maximal D used for plotting x-limits
    x_margin = max(5.0, max_D * 0.03)  # margin for plot edges

    fig, ax = plt.subplots(figsize=(8.5, 5.2))  # figure for beam cross-section

    # draw a light-blue band for water region for context
    ax.fill_betweenx([0.0, water_level], -0.0, max_D + x_margin,
                     color="lightblue", alpha=0.12, zorder=0)

    # horizontal reference lines for tank bottom, water level, sensor line and beam axis
    ax.axhline(0.0, color="k", lw=1.0)                     # bottom
    ax.axhline(water_level, color="k", lw=1.0, ls="--")    # water level
    ax.axhline(sensor_level, color="k", lw=1.0, ls=":")    # sensor line
    ax.axhline(beam_axis_from_bottom, color="0.3", lw=1.0, ls="-.")  # beam axis

    # vertical line for beam entrance at x=0 (right in plot orientation)
    ax.axvline(0.0, color="k", lw=1.0, ls="--")  # entrance marker

    # loop energies, plot reconstructed vertical positions vs nominal D
    for i, (energy, pdict) in enumerate(sorted(profile_data.items())):
        D = np.array(pdict["D"], dtype=float)
        dist = np.array(pdict["dist"], dtype=float)      # distance from sensor towards bottom
        A = np.array(pdict.get("A", np.full_like(D, np.nan)), dtype=float)  # amplitude array
        keep = np.isfinite(D) & np.isfinite(dist)  # finite-data mask
        if drop_above_115_for_115MeV and float(energy) == 115.0:
            keep = keep & (D <= 115.0)  # optional drop for 115 MeV above 115 mm
        if not np.any(keep):
            continue

        D = D[keep]  # filtered D
        dist = dist[keep]
        A = A[keep]

        # compute vertical position from tank bottom for plotting (sensor at sensor_level)
        y = sensor_level - dist   # vertical coordinate from tank bottom

        order = np.argsort(D)  # order points by D
        c = COLORS[i % len(COLORS)]  # per-energy color
        ax.plot(D[order], y[order], '-o', color=c, markeredgecolor='k', lw=1.6, label=f"{int(energy)} MeV")  # plot

        # optional reflection across beam axis to visualize symmetry
        if reflect:
            y_ref = 2.0 * beam_axis_from_bottom - y
            ax.plot(D[order], y_ref[order], color=c, alpha=0.7)  # reflected points

    # axis labels and orientation adjustments
    ax.set_xlabel("Water layer (mm)")  # x label
    ax.set_ylabel("Vertical position from tank bottom (mm)")  # y label
    ax.set_title("Beam profile vs longitudinal distance")  # title

    # invert x so entrance at the right (0 on right), larger D on left
    ax.set_xlim(max_D + x_margin, 0.0)  # invert x-axis
    ypad = (water_level - 0.0) * 0.06
    ax.set_ylim(-ypad, water_level + ypad)  # y limits with padding

    # legend placement: outside to the right by default to avoid overlapping datapoints
    if show_legend_outside:
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.)
    else:
        ax.legend(loc="upper left")
    style_axes(ax)
    save_fig(fig, fname_profile)  # save profile figure
    plt.show()

    # ---------- Plot 2: A vs dist with power-law fit per energy ----------
    fig, ax = plt.subplots(figsize=(7.0, 5.0))

    # iterate energies and fit A = A0 * dist^-n where possible
    for i, (energy, pdict) in enumerate(sorted(profile_data.items())):
        D = np.array(pdict["D"], dtype=float)
        dist = np.array(pdict["dist"], dtype=float)
        A = np.array(pdict.get("A", np.full_like(D, np.nan)), dtype=float)
        Aerr = np.array(pdict.get("Aerr", np.full_like(A, np.nan)), dtype=float)

        keep = np.isfinite(dist) & np.isfinite(A)  # mask finite entries
        if drop_above_115_for_115MeV and float(energy) == 115.0:
            keep = keep & (D <= 115.0)  # optional filtering
        if np.sum(keep) == 0:
            continue

        xdata = dist[keep]    # distance from sensor
        ydata = A[keep]  # acoustic amplitude
        yerr = Aerr[keep] if np.any(np.isfinite(Aerr[keep])) else None  # optional errors

        # require at least 3 valid positive points to fit a power law robustly
        if np.sum(np.isfinite(xdata) & np.isfinite(ydata) & (ydata > 0)) < 3:
            ax.errorbar(xdata, ydata, yerr=yerr, fmt='o', color=COLORS[i % len(COLORS)], label=f"{int(energy)} MeV (data)")  # plot raw data only
            print(f"  [{energy:.0f} MeV] not enough points for fit; plotting data only.")  # diagnostic
            continue

        # fit power-law A = A0 * dist^-n using power_law and fit_curve
        mask = np.isfinite(xdata) & np.isfinite(ydata) & (xdata > 0) & (ydata > 0)
        xfit = xdata[mask]
        yfit = ydata[mask]
        try:
            p0 = [np.max(yfit) * (xfit.min() ** 1.0), 1.0]  # initial guess for A0 and n
            popt, perr, r2 = fit_curve(power_law, xfit, yfit, yerr=yerr[mask] if yerr is not None else None, p0=p0)  # fit
            A0, n = popt  # fitted parameters
            dA0, dn = perr  # their errors
            ax.errorbar(xdata, ydata, yerr=yerr, fmt='o', color=COLORS[i % len(COLORS)], alpha=0.85)  # plot data
            Xfine = np.linspace(np.min(xfit)*0.95, np.max(xfit)*1.05, 250)
            ax.plot(Xfine, power_law(Xfine, *popt), '-', color=COLORS[i % len(COLORS)], lw=1.6,
                    label=(f"{int(energy)} MeV fit: A0={A0:.3g}, n={n:.3f}, R2={r2:.3f}"))  # plot fit
            print(f"  [{energy:.0f} MeV] fit: A0={A0:.3g} +/- {dA0:.3g}, n={n:.3f} +/- {dn:.3f}, R2={r2:.4f}")  # print fit result
        except Exception as e:
            ax.errorbar(xdata, ydata, yerr=yerr, fmt='o', color=COLORS[i % len(COLORS)], label=f"{int(energy)} MeV (data)")  # plot data on failure
            print(f"  [{energy:.0f} MeV] fit failed: {e}")  # report error

    ax.set_xlabel("Distance from sensor line (mm)")  # x label
    ax.set_ylabel("Acoustic signal A (mV)")  # y label
    ax.set_title("A vs distance from sensor (power-law fits)")  # title
    if show_legend_outside:
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.)  # legend outside by default
    else:
        ax.legend(loc="upper right")
    style_axes(ax)
    save_fig(fig, fname_fit)  # save fitted plot
    plt.show()


# call the above function with default geometry settings
plot_beam_profile_and_fit(profile_data,
                          beam_axis_from_bottom=46.0,
                          sensor_level=78.0,
                          water_level=93.0,
                          reflect=True,
                          drop_above_115_for_115MeV=True,
                          fname_profile="F3_beam_profile_simple",
                          fname_fit="F4_A_vs_dist_powerfit")



# %% [10] SUMMARY ----------------------------------------------
print("\n=== Summary ===")  # print header for fit summary

# assemble summary rows for the main fits performed earlier in the script
summary_rows = [
    {"Test": "A vs Q (single bunch)", "Model": "linear",
     "Param 1": f"m={m_sb:.2f}+/-{dm_sb:.2f} mV/nC",
     "Param 2": f"b={b_sb:.2f}+/-{db_sb:.2f} mV", "R2": round(r2_sb, 4)},
    {"Test": "A vs Q (multi bunch)", "Model": "linear",
     "Param 1": f"m={m_mb:.2f}+/-{dm_mb:.2f} mV/nC",
     "Param 2": f"b={b_mb:.2f}+/-{db_mb:.2f} mV", "R2": round(r2_mb, 4)},
]

# include per-energy exponential-decay fit summaries from Section E
for energy, (popt_d, perr_d, r2_d) in sorted(fit_results.items()):
    summary_rows.append({
        "Test": f"A vs D ({energy:.0f} MeV)", "Model": "exp. decay",
        "Param 1": f"lambda={popt_d[1]:.1f}+/-{perr_d[1]:.1f} mm",
        "Param 2": f"A0={popt_d[0]:.1f}+/-{perr_d[0]:.1f} mV",
        "R2": round(r2_d, 4),
    })

# create a DataFrame for the summary and print/save it
summary_df = pd.DataFrame(summary_rows)  # DataFrame of fit summaries
print(summary_df.to_string(index=False))  # print full table
summary_df.to_csv(OUTPUT_DIR / "fit_summary.csv", index=False)  # write CSV to output directory
print(f"\nFit summary written to {OUTPUT_DIR / 'fit_summary.csv'}")  # confirmation
print(f"All figures written to: {OUTPUT_DIR.resolve()}")  # final location reminder