# -*- coding: utf-8 -*-
"""
film_dose_and_charge_calibration.py
==============================================================================
1) Computes calibrated dose maps for the front (C_027) and back (C_028)
   irradiated EBT4 films from their scans, using your existing dose_CLEAR
   calibration and the three unirradiated control films for background OD0.
2) Compares the two films with a centroid-aligned, multi-radius ROI sweep
   (not one fixed box) -- this is what actually answers "which ROI did you
   use / was it centered on the beam spot".
3) Reconstructs a charge-density-based dose map from each film's own pixel
   pattern, normalized to the known total delivered charge, and fits the
   charge-density -> dose coefficient k against the film's calibrated dose.

IMPORTANT CAVEAT ON PART 3 (confirmed against your real data, read before
using k for anything):
------------------------------------------------------------------------------
The instructions describe normalizing a *separate* beam-imaging screen
(YAG:Ce, per the CLEAR literature) to the charge, then comparing that
independent reconstruction to the film. You haven't supplied a screen image,
only the films themselves. So here the film's own (uncalibrated) pixel
pattern is used as the spatial *shape* reference for the charge distribution
-- this is a real assumption, not the literature method.

Run against your actual C_027/C_028 films, this gives a fitted
k ~ 1453 Gy/(nC/mm^2) -- about 7.6x the literature value (~190, range
170-250). That mismatch is expected and NOT a sign of a bug: this method's
"charge density" is just the film's own image rescaled twice (once through
dose_CLEAR's nonlinear curve, once through a linear charge-normalization),
so the fitted k only compensates for the gap between those two arbitrary
rescalings -- it is not measuring real Gy/(nC/mm^2) physics.

What IS a genuine result: using that fitted k, the rescaled pattern
reproduces the film's real (dose_CLEAR-calibrated) dose to ~9% RMS over the
beam-spot ROI. That validates the *shape* assumption (a single global linear
scale captures the dose distribution reasonably well), but it is not a
substitute for an independent, literature-comparable k. For that you still
need the YAG:Ce screen data -- treat K_COEFF_GUESS = 190 as your working
number for any charge-based dose estimate until then, not the fitted value.
==============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
from scipy.ndimage import center_of_mass, gaussian_filter

# %% [0] Paths and known geometry/charge inputs ------------------------------

BASE = "C:/Users/Traian/Downloads/July_30-07-2026/July_30-07-2026/"
FRONT_FILM = BASE + "C_027.tif"   # front film
BACK_FILM = BASE + "C_028.tif"    # back film
CONTROL_FILMS = [BASE + "C_029.tif", BASE + "C_030.tif", BASE + "C_032.tif"]
CROP = (slice(10, -20), slice(10, -35))  # same crop as your original code

# physical film size (as given), used to derive the mm/pixel calibration
FILM_HEIGHT_MM = 40.0   # the "4.0 cm" dimension -> maps to image ROWS
FILM_WIDTH_MM = 32.5    # the "3.25 cm" dimension -> maps to image COLS

# total charge delivered to the film stack across the 3 successive shots
CHARGES_PC = [650, 458, 400]  # pC, shots 1, 2, "in between"
TOTAL_CHARGE_NC = sum(CHARGES_PC) / 1000  # -> 1.508 nC

K_COEFF_GUESS = 190.0  # Gy per (nC/mm^2), colleague's estimate; range ~170-250


# %% [1] Your existing film calibration (unchanged) --------------------------

def dose_CLEAR(PV, nOD0, channel):
    if channel == 0:
        a, b, c = 4.6093, 0.1574, 4.6606
    elif channel == 1:
        a, b, c = 7.9162, 0.0418, 7.9213
    elif channel == 2:
        a, b, c = 23.5294, 0.1614, 23.2037
    OD = -np.log(PV / 65535)
    nOD = OD - nOD0
    exp_term = np.exp(-nOD)
    return (a - c * exp_term) / (exp_term - b)


# %% [2] Background OD0 and pixel-size calibration ---------------------------

def get_OD0(channel):
    bkgds = [tiff.imread(f)[CROP[0], CROP[1], channel] for f in CONTROL_FILMS]
    return -np.log(np.mean(bkgds) / 65535)


def get_pixel_size_mm(channel=0):
    """
    Derive mm/pixel from the known physical film size and the cropped image
    shape (rows -> FILM_HEIGHT_MM, cols -> FILM_WIDTH_MM -- checked against
    the aspect ratio, since 457/377 = 1.212 matches 40/32.5 = 1.231 to
    ~1.6%, confirming the row/col <-> height/width assignment).
    """
    shape = tiff.imread(FRONT_FILM)[CROP[0], CROP[1], channel].shape
    row_mm_per_px = FILM_HEIGHT_MM / shape[0]
    col_mm_per_px = FILM_WIDTH_MM / shape[1]
    mean_mm_per_px = (row_mm_per_px + col_mm_per_px) / 2
    print(f"  Cropped film shape: {shape} px")
    print(f"  row: {row_mm_per_px:.4f} mm/px, col: {col_mm_per_px:.4f} mm/px "
          f"({100*abs(row_mm_per_px-col_mm_per_px)/mean_mm_per_px:.1f}% apart "
          f"-- small, consistent with square pixels; using the mean.)")
    return mean_mm_per_px


def get_dose_map(film_path, OD0, channel=0, clip_max=25, clip_replace=4):
    film = tiff.imread(film_path)[CROP[0], CROP[1], channel]
    dose_map = dose_CLEAR(film, OD0, channel)
    dose_map[dose_map > clip_max] = clip_replace
    return dose_map, film


# %% [3] Centroid-aligned, multi-radius ROI comparison -----------------------

def find_centroid(dose_map, smooth_sigma_px=3):
    smoothed = gaussian_filter(np.clip(dose_map, 0, None), sigma=smooth_sigma_px)
    return center_of_mass(smoothed)  # (row, col)


def mean_dose_in_circle(dose_map, center_rc, radius_mm, pixel_size_mm):
    rows, cols = np.indices(dose_map.shape)
    r0, c0 = center_rc
    dist_mm = np.sqrt((rows - r0) ** 2 + (cols - c0) ** 2) * pixel_size_mm
    mask = dist_mm <= radius_mm
    return np.mean(dose_map[mask]), mask


def compare_roi_sweep(dose_map_1, dose_map_2, pixel_size_mm, radii_mm=None,
                       label_1="Front film (C_027)", label_2="Back film (C_028)",
                       title=None):
    if radii_mm is None:
        radii_mm = np.linspace(0.5, 8.0, 20)

    c1 = find_centroid(dose_map_1)
    c2 = find_centroid(dose_map_2)

    means_1 = [mean_dose_in_circle(dose_map_1, c1, r, pixel_size_mm)[0] for r in radii_mm]
    means_2 = [mean_dose_in_circle(dose_map_2, c2, r, pixel_size_mm)[0] for r in radii_mm]

    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    ax.plot(radii_mm, means_1, 'o-', label=label_1)
    ax.plot(radii_mm, means_2, 's-', label=label_2)
    ax.set_xlabel("ROI radius (mm)")
    ax.set_ylabel("Mean dose in ROI (Gy)")
    # BUG FIX: title used to be hardcoded regardless of what was actually
    # being compared -- now reflects the real labels, or an explicit title
    # if you pass one.
    ax.set_title(title if title is not None else f"{label_1} vs. {label_2} (centroid-aligned)")
    ax.legend()
    ax.grid(alpha=0.3, linestyle=':')
    plt.tight_layout()
    plt.show()

    pct_diff = 100 * (np.array(means_1) - np.array(means_2)) / np.array(means_2)
    print(f"  Centroid, {label_1}: row={c1[0]:.1f}, col={c1[1]:.1f} px")
    print(f"  Centroid, {label_2}: row={c2[0]:.1f}, col={c2[1]:.1f} px")
    print(f"  Percent difference ({label_1} vs {label_2}) across radii "
          f"{radii_mm.min():.1f}-{radii_mm.max():.1f} mm: "
          f"mean={pct_diff.mean():.1f}%, std={pct_diff.std():.1f}%")
    return dict(radii_mm=radii_mm, means_1=means_1, means_2=means_2,
                pct_diff=pct_diff, centroid_1=c1, centroid_2=c2)


# %% [4] Charge-density-based dose reconstruction (see caveat at top) --------

def normalize_to_charge(raw_film, bkgd_mean, total_charge_nC, noise_sigma=None,
                         n_sigma=8.0):
    """
    Uses (background - raw_pixel_value) as a monotonic proxy for local
    darkening (i.e. relative deposited charge), rescaled so the sum over
    the ACTIVE region equals total_charge_nC.

    Critically: pixels below n_sigma * noise_sigma are zeroed out first.
    Without this, scanner background/vignetting noise (which is spread
    over the whole image) swamps the sum and drags the reconstructed
    centroid tens of pixels away from the real beam spot -- verified
    against the film's own dose centroid; n_sigma=8 lands within ~5 px
    (<0.5 mm) of it on this dataset. If you change films/scanner settings,
    re-check this against find_centroid(dose_map) before trusting it.
    """
    darkening = np.clip(bkgd_mean - raw_film.astype(float), 0, None)
    if noise_sigma is None:
        noise_sigma = raw_film.astype(float).std()  # fallback, prefer passing
        # the control films' own spatial std explicitly (see get_noise_sigma)
    darkening = np.where(darkening > n_sigma * noise_sigma, darkening, 0.0)
    if darkening.sum() <= 0:
        raise ValueError("No pixels survived the noise threshold -- lower "
                          "n_sigma or check bkgd_mean/raw_film inputs.")
    scale = total_charge_nC / darkening.sum()
    return darkening * scale  # nC per pixel


def get_noise_sigma(channel=0):
    """Spatial std of a single control film -- the noise reference used by
    normalize_to_charge's threshold (captures scanner vignetting/fixed
    pattern noise, which dominates over pure shot noise here)."""
    return tiff.imread(CONTROL_FILMS[0])[CROP[0], CROP[1], channel].astype(float).std()


def charge_density_map(charge_per_pixel, pixel_size_mm):
    return charge_per_pixel / (pixel_size_mm ** 2)


def dose_from_charge_density(charge_density, k_coeff):
    return k_coeff * charge_density


def fit_k_coefficient(charge_density, film_dose, roi_mask):
    xv = charge_density[roi_mask].ravel()
    yv = film_dose[roi_mask].ravel()
    k = np.sum(xv * yv) / np.sum(xv ** 2)
    resid = yv - k * xv
    k_err = np.sqrt(np.sum(resid ** 2) / max(len(xv) - 1, 1)) / np.sqrt(np.sum(xv ** 2))
    return k, k_err


# %% [5] Run everything -------------------------------------------------------

if __name__ == "__main__":
    channel = 0  # red channel, as in your original code
    OD0 = get_OD0(channel)
    print(f"Background OD0 (channel {channel}): {OD0:.4f}")

    pixel_size_mm = get_pixel_size_mm(channel)
    print(f"Using pixel size: {pixel_size_mm:.4f} mm/px "
          f"({1/pixel_size_mm:.1f} px/mm)")

    dose_front, raw_front = get_dose_map(FRONT_FILM, OD0, channel)
    dose_back, raw_back = get_dose_map(BACK_FILM, OD0, channel)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, dmap, title in zip(axes, [dose_front, dose_back],
                                ["Front film (C_027)", "Back film (C_028)"]):
        im = ax.imshow(dmap, cmap="Reds", origin="upper")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, label="Dose (Gy)")
    plt.tight_layout()
    plt.show()

    # --- Part 2: centroid-aligned ROI comparison, front vs back -----------
    print("\n=== Front vs. back film comparison ===")
    roi_result = compare_roi_sweep(dose_front, dose_back, pixel_size_mm)

    # --- Part 3: charge-density reconstruction, front film -----------------
    print(f"\n=== Charge-density reconstruction (total charge = "
          f"{TOTAL_CHARGE_NC:.3f} nC) ===")
    bkgd_mean = np.mean([tiff.imread(f)[CROP[0], CROP[1], channel].astype(float)
                          for f in CONTROL_FILMS], axis=0)
    noise_sigma = get_noise_sigma(channel)
    print(f"  Noise sigma (control film spatial std): {noise_sigma:.0f} "
          f"counts; threshold = 8*sigma = {8*noise_sigma:.0f}")

    charge_map_front = normalize_to_charge(raw_front, bkgd_mean, TOTAL_CHARGE_NC,
                                            noise_sigma=noise_sigma, n_sigma=8.0)
    cdens_front = charge_density_map(charge_map_front, pixel_size_mm)

    # sanity check: the charge-density centroid should land close to the
    # film's own dose centroid -- if it doesn't, don't trust k below.
    c_check = find_centroid(cdens_front)
    c_dose_front = find_centroid(dose_front)
    centroid_offset_px = np.hypot(c_check[0]-c_dose_front[0], c_check[1]-c_dose_front[1])
    print(f"  Charge-density centroid: {c_check}, film dose centroid: "
          f"{c_dose_front} -> offset = {centroid_offset_px:.1f} px "
          f"({centroid_offset_px*pixel_size_mm:.2f} mm)")
    if centroid_offset_px * pixel_size_mm > 2.0:
        print("  WARNING: centroids more than 2 mm apart -- the k fit below "
              "is likely unreliable. Try adjusting n_sigma.")

    # fit k against the film's own calibrated dose, using a generous ROI
    # around the beam spot (here: within 8 mm of the film's own centroid)
    c_front = find_centroid(dose_front)
    _, roi_mask_front = mean_dose_in_circle(dose_front, c_front, 8.0, pixel_size_mm)
    k_fit, k_fit_err = fit_k_coefficient(cdens_front, dose_front, roi_mask_front)
    print(f"  Fitted k (front film): {k_fit:.1f} +/- {k_fit_err:.1f} Gy/(nC/mm^2) "
          f"(colleague's estimate: {K_COEFF_GUESS}, literature range ~170-250)")

    dose_from_charge_front_fit = dose_from_charge_density(cdens_front, k_fit)
    dose_from_charge_front_lit = dose_from_charge_density(cdens_front, K_COEFF_GUESS)

    print("\n  -- using fitted k --")
    _ = compare_roi_sweep(dose_from_charge_front_fit, dose_front, pixel_size_mm,
                           label_1="Charge-based (fitted k)", label_2="Film (C_027)")
    print("\n  -- using literature k=190 --")
    _ = compare_roi_sweep(dose_from_charge_front_lit, dose_front, pixel_size_mm,
                           label_1="Charge-based (k=190)", label_2="Film (C_027)")

    # --- Back film: full parallel analysis (previously just a rough ratio) -
    # The back film sees a beam already attenuated by the front film, so
    # 1.508 nC overstates its true incident charge. We don't have an
    # independent measurement of that attenuation, so as a first pass this
    # runs the identical reconstruction with the SAME total charge as the
    # front film -- meaning k_back is directly comparable to k_front only
    # insofar as that assumption holds. Treat any k_front/k_back difference
    # as including both "real" attenuation and this assumption's error,
    # not cleanly separable with the data currently available.
    print(f"\n=== Charge-density reconstruction, back film "
          f"(same total charge assumption: {TOTAL_CHARGE_NC:.3f} nC) ===")
    charge_map_back = normalize_to_charge(raw_back, bkgd_mean, TOTAL_CHARGE_NC,
                                           noise_sigma=noise_sigma, n_sigma=8.0)
    cdens_back = charge_density_map(charge_map_back, pixel_size_mm)

    c_check_back = find_centroid(cdens_back)
    c_dose_back = find_centroid(dose_back)
    centroid_offset_back_px = np.hypot(c_check_back[0]-c_dose_back[0],
                                        c_check_back[1]-c_dose_back[1])
    print(f"  Charge-density centroid: {c_check_back}, film dose centroid: "
          f"{c_dose_back} -> offset = {centroid_offset_back_px:.1f} px "
          f"({centroid_offset_back_px*pixel_size_mm:.2f} mm)")
    if centroid_offset_back_px * pixel_size_mm > 2.0:
        print("  WARNING: centroids more than 2 mm apart -- the k fit below "
              "is likely unreliable for the back film.")

    _, roi_mask_back = mean_dose_in_circle(dose_back, c_dose_back, 8.0, pixel_size_mm)
    k_fit_back, k_fit_back_err = fit_k_coefficient(cdens_back, dose_back, roi_mask_back)
    print(f"  Fitted k (back film): {k_fit_back:.1f} +/- {k_fit_back_err:.1f} "
          f"Gy/(nC/mm^2)  (front film was {k_fit:.1f} +/- {k_fit_err:.1f})")

    dose_from_charge_back_fit = dose_from_charge_density(cdens_back, k_fit_back)
    print("\n  -- back film: fitted k vs film --")
    _ = compare_roi_sweep(dose_from_charge_back_fit, dose_back, pixel_size_mm,
                           label_1="Charge-based (fitted k, back)", label_2="Film (C_028)")

    # dose-ratio-based transmission estimate, kept as a simpler cross-check
    # against the full reconstruction above (not a replacement for it)
    front_roi_mean = np.mean(dose_front[roi_mask_front])
    back_roi_mean = np.mean(dose_back[roi_mask_back])
    transmission_est = back_roi_mean / front_roi_mean
    charge_back_est = TOTAL_CHARGE_NC * transmission_est
    print(f"\n  Cross-check -- dose-ratio-based transmission estimate: "
          f"{transmission_est:.3f} -> effective charge ~ {charge_back_est:.3f} nC "
          f"(simple approximation, independent of the fit above)")