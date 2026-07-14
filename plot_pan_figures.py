"""
plot_pan_figures.py
===================
Standalone script to produce publication-ready figures for
Physics of Atomic Nuclei (PAN / Ядерная физика).

Generates figures from CSV outputs of analyze_urqmd_kaons.py plus
NA61/SHINE HepData CSVs for multiple collision systems:
  - Ar+Sc  @ sqrt(s_NN) = 11.9 GeV  (ecm collision frame)
  - Xe+Xe  @ sqrt(s_NN) =  9.2 GeV  (ecm collision frame)
  - C+C    @ sqrt(s_NN) = 34.1 GeV  (target frame)
  - Xe+W   @ sqrt(s_NN) =  2.9 GeV  (target frame)

NA61/SHINE HEPdata units (confirmed from HEPdata keyword headers)
-----------------------------------------------------------------
Figure1a/b  ->  dn/dy          (per-event average rapidity spectra)
Figure2a/b  ->  dn/dpT         (per-event average transverse-momentum spectra)
Figure7     ->  d2n/dy dpT     (per-event average 2D K0S spectrum)

This means the dn_rap_overlay / dn_pt_overlay plots compare
UrQMD dn directly against the experimental data with NO rescaling.
The rap_overlay / pt_overlay plots still use UrQMD dN; the unit
mismatch (dN vs dn) is noted in the axis labels and must be
explained in the LaTeX caption.

Outputs per system (all as .eps + .png), e.g. for Ar+Sc:
  ArSc_11p9GeV_rap_overlay_unmod.{eps,png}        -- UrQMD dN/dy vs NA61/SHINE dn/dy
  ArSc_11p9GeV_rap_overlay_mod.{eps,png}
  ArSc_11p9GeV_pt_overlay_unmod.{eps,png}         -- UrQMD dN/dpT (linear) + R(pT) vs dn/dpT
  ArSc_11p9GeV_pt_overlay_mod.{eps,png}
  ArSc_11p9GeV_pan_y_species_unmod.{eps,png}      -- dN/dy per species (UrQMD only)
  ArSc_11p9GeV_pan_y_species_mod.{eps,png}
  ArSc_11p9GeV_pan_pt_species_unmod.{eps,png}     -- dN/dpT per species (UrQMD only)
  ArSc_11p9GeV_pan_pt_species_mod.{eps,png}
  ArSc_11p9GeV_pan_dn_y_species_unmod.{eps,png}   -- dn/dy per species (per-event)
  ArSc_11p9GeV_pan_dn_y_species_mod.{eps,png}
  ArSc_11p9GeV_pan_dn_pt_species_unmod.{eps,png}  -- dn/dpT per species (per-event)
  ArSc_11p9GeV_pan_dn_pt_species_mod.{eps,png}
  ArSc_11p9GeV_pan_ratio_y_unmod.{eps,png}
  ArSc_11p9GeV_pan_ratio_y_mod.{eps,png}
  ArSc_11p9GeV_dn_rap_overlay_unmod.{eps,png}     -- UrQMD dn/dy (single variant) vs NA61/SHINE dn/dy
  ArSc_11p9GeV_dn_rap_overlay_mod.{eps,png}
  ArSc_11p9GeV_dn_rap_overlay_ratio_y_unmod.{eps,png} -- TWO-PANEL: dn/dy overlay (top) + R_K(y) (bottom)
  ArSc_11p9GeV_dn_rap_overlay_ratio_y_mod.{eps,png}
  ArSc_11p9GeV_dn_pt_overlay_unmod.{eps,png}      -- UrQMD dn/dpT + R(pT) (single variant) vs dn/dpT
  ArSc_11p9GeV_dn_pt_overlay_mod.{eps,png}
  ArSc_11p9GeV_k0s_2d_combined_overlay.{eps,png}  -- K0S d2n/dydpT vs exp
  ArSc_11p9GeV_dn_rap_plain_unmod.{eps,png}       -- UrQMD dn/dy: (K++K-)/2 vs K0S only (no exp)
  ArSc_11p9GeV_dn_rap_plain_mod.{eps,png}
  ArSc_11p9GeV_dn_pt_plain_unmod.{eps,png}        -- UrQMD dn/dpT + R(pT): (K++K-)/2 vs K0S only (no exp)
  ArSc_11p9GeV_dn_pt_plain_mod.{eps,png}
  ArSc_11p9GeV_dn_rap_plain_ratio_y_unmod.{eps,png} -- TWO-PANEL: dn/dy (top) + R_K(y) (bottom), UrQMD only
  ArSc_11p9GeV_dn_rap_plain_ratio_y_mod.{eps,png}

NOTE: energy_str uses 'p' instead of '.' (e.g. 11p9GeV) to avoid
Windows treating the decimal as a file extension separator.

PAN journal style notes
-----------------------
* No in-plot titles -- system/energy/variant info goes in the LaTeX caption.
* Each dn_rap_overlay / dn_pt_overlay figure shows a single UrQMD variant
  (unmod or mod) overlaid with NA61/SHINE data -- one plot per variant.
* The rap_overlay / pt_overlay functions also show a single UrQMD variant (dN).
* PAN-only UrQMD plots show only species labels (K+, K-, K0S).
* Grayscale / black-and-white only.
* Curves distinguished by linestyle + marker shape.
* EPS vector output suitable for Yadernaya Fizika submission.
* pt_overlay top panel is LINEAR for both unmod and mod.

R(pT) bottom panel -- error note
---------------------------------
NA61/SHINE R(pT) errors are derived from the ratio of two Boltzmann
fits (one per species) with the uncertainty band obtained by propagating
the fit-parameter covariance matrices analytically.  This produces a
smooth shaded band, NOT discrete error bars.  Falls back to bin-by-bin
error bars if scipy is unavailable or fits fail.
UrQMD R(pT) uses raw bin-by-bin MC statistical errors (line only).

k0s_2d_combined_overlay panel note
-----------------------------------
The k0s_2d_combined_overlay plot shows K0S d^2n/dydpT as a function of pT
for each rapidity bin, overlaying:
  - unmod UrQMD (solid line, no markers)
  - mod  UrQMD  (dashed line, no markers)
  - NA61/SHINE experimental data (filled circles with stat+sys errors)
This uses hep_data/Figure7.csv which contains the 2D (y, pT) spectrum.

dn_rap_plain / dn_pt_plain note
---------------------------------
These plots show only UrQMD simulation results -- (K++K-)/2 and K0S --
with NO experimental data overlaid.  One figure per variant (unmod/mod).
dn_rap_plain: single panel, dn/dy vs y.
dn_pt_plain:  two-panel -- dn/dpT linear (top) + R(pT) = (K++K-)/2 / K0S (bottom).

dn_rap_plain_ratio_y note
---------------------------------
Two-panel combined figure (UrQMD only, no experimental data):
  top panel:    dn/dy vs y  -- (K++K-)/2 and K0S  (mirrors dn_rap_plain)
  bottom panel: R_K(y) = (K++K-)/2 / K0S vs y  (mirrors pan_ratio_y)
Layout mirrors dn_pt_plain: height_ratios [2.5, 1], sharex=True, hspace=0.05.

dn_rap_overlay_ratio_y note
---------------------------------
Two-panel combined figure WITH experimental data in the top panel:
  top panel:    dn/dy vs y -- UrQMD K0S + (K++K-)/2 overlaid with NA61/SHINE
  bottom panel: R_K(y) = UrQMD (K++K-)/2 / K0S vs y (UrQMD only, no exp R_K)
Layout mirrors dn_pt_plain / dn_rap_plain_ratio_y:
  height_ratios [2.5, 1], sharex=True, hspace=0.05.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

try:
    from scipy.optimize import curve_fit as _scipy_curve_fit
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.size": 11,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "legend.frameon": True,
    "legend.framealpha": 1.0,
    "legend.edgecolor": "black",
    "legend.fontsize": 9,
    "figure.dpi": 150,
})

# ---------------------------------------------------------------------------
# Style palette (pure black)
# ---------------------------------------------------------------------------

STYLE = {
    "urqmd_k0s":   dict(ls="-",    lw=1.4, marker="None", color="black"),
    "urqmd_kch":   dict(ls="--",   lw=1.4, marker="None", color="black"),
    "exp_k0s":     dict(ls="None", marker="o", ms=5,
                        mfc="black", mec="black", color="black",
                        capsize=3, elinewidth=0.8),
    "exp_kch":     dict(ls="None", marker="s", ms=5,
                        mfc="white", mec="black", color="black",
                        capsize=3, elinewidth=0.8),
    "ratio_urqmd": dict(ls="-",    lw=1.4, marker="None", color="black"),
    "ratio_exp":   dict(ls="None", marker="^", ms=5,
                        mfc="black", mec="black", color="black",
                        capsize=3, elinewidth=0.8),
    # PAN-only species
    "Kplus":  dict(ls="-",  lw=1.3, marker="o", ms=4,
                   mfc="white", mec="black", color="black"),
    "Kminus": dict(ls="--", lw=1.3, marker="s", ms=4,
                   mfc="black", mec="black", color="black"),
    "K0S":    dict(ls="-.", lw=1.3, marker="^", ms=4,
                   mfc="white", mec="black", color="black"),
    "ratio_y": dict(ls="-", lw=1.4, marker="o", ms=4,
                    mfc="black", mec="black", color="black"),
    # k0s_2d_combined_overlay styles
    "fig7_urqmd_unmod": dict(ls="-",  lw=1.4, marker="None", color="black"),
    "fig7_urqmd_mod":   dict(ls="--", lw=1.4, marker="None", color="black"),
    "fig7_exp":         dict(ls="None", marker="o", ms=4,
                             mfc="black", mec="black", color="black",
                             capsize=2, elinewidth=0.8),
}

SPECIES_LABEL = {
    "Kplus":  r"$K^+$",
    "Kminus": r"$K^-$",
    "K0S":    r"$K^0_S$",
}

# ---------------------------------------------------------------------------
# Collision system registry
# ---------------------------------------------------------------------------

@dataclass
class SystemDef:
    tag: str
    label: str
    energy_str: str
    energy_label: str
    unmod_dir: str
    mod_dir: str
    hep1a: str
    hep1b: str
    hep2a: str
    hep2b: str
    hep7: str = ""

    def out(self, outdir: Path, kind: str, mod: Optional[bool] = None) -> Path:
        if mod is None:
            return outdir / f"{self.tag}_{self.energy_str}_{kind}"
        suffix = "mod" if mod else "unmod"
        return outdir / f"{self.tag}_{self.energy_str}_{kind}_{suffix}"


ALL_SYSTEMS: List[SystemDef] = [
    SystemDef(
        tag="ArSc", label="Ar+Sc",
        energy_str="11p9GeV",
        energy_label=r"$\sqrt{s_{NN}}=11.9\,\mathrm{GeV}$",
        unmod_dir="ArSc-kaons-11.9GeV-ecm-collision/results",
        mod_dir="ArSc-kaons-11.9GeV-ecm-collision-modified-ud/results",
        hep1a="Figure1a.csv", hep1b="Figure1b.csv",
        hep2a="Figure2a.csv", hep2b="Figure2b.csv",
        hep7="Figure7.csv",
    ),
    SystemDef(
        tag="CC", label="C+C",
        energy_str="34p1GeV",
        energy_label=r"$\sqrt{s_{NN}}=34.1\,\mathrm{GeV}$",
        unmod_dir="CC-kaons-34.1GeV-ecm-target/results",
        mod_dir="CC-kaons-34.1GeV-ecm-target-modified-ud/results",
        hep1a="CC_Figure1a.csv", hep1b="CC_Figure1b.csv",
        hep2a="CC_Figure2a.csv", hep2b="CC_Figure2b.csv",
    ),
]

# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def _skip_comments_lines(path: Path) -> List[str]:
    lines = []
    with open(path, newline="") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            lines.append(s)
    return lines


def load_hepdata(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load a HEPdata CSV (x, y, err+, err-). Values are returned as-is (no rescaling)."""
    rows = _skip_comments_lines(path)
    reader = csv.reader(rows)
    next(reader, None)
    xs, ys, eup, edn = [], [], [], []
    for row in reader:
        if len(row) < 4:
            continue
        xs.append(float(row[0]))
        ys.append(float(row[1]))
        eup.append(abs(float(row[2])))
        edn.append(abs(float(row[3])))
    return np.array(xs), np.array(ys), np.array(eup), np.array(edn)


def load_urqmd_ydist_meancharged(path: Path):
    rows = _skip_comments_lines(path)
    reader = csv.reader(rows)
    next(reader, None)
    yc, kch, kch_e, k0s, k0s_e = [], [], [], [], []
    for row in reader:
        if len(row) < 7:
            continue
        yc.append(float(row[2]))
        kch.append(float(row[3]))
        kch_e.append(float(row[4]))
        k0s.append(float(row[5]))
        k0s_e.append(float(row[6]))
    return (np.array(yc), np.array(kch), np.array(kch_e),
            np.array(k0s), np.array(k0s_e))


def load_urqmd_y_distributions(path: Path) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    rows = _skip_comments_lines(path)
    reader = csv.reader(rows)
    next(reader, None)
    data: Dict[str, Tuple[List, List, List]] = {}
    for row in reader:
        if len(row) < 7:
            continue
        sp = row[0].strip()
        if sp not in data:
            data[sp] = ([], [], [])
        data[sp][0].append(float(row[3]))
        data[sp][1].append(float(row[5]))
        data[sp][2].append(float(row[6]))
    return {sp: (np.array(v[0]), np.array(v[1]), np.array(v[2]))
            for sp, v in data.items()}


def load_urqmd_pt_spectra(path: Path) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    rows = _skip_comments_lines(path)
    reader = csv.reader(rows)
    next(reader, None)
    data: Dict[str, Tuple[List, List, List]] = {}
    for row in reader:
        if len(row) < 7:
            continue
        sp = row[0].strip()
        if sp not in data:
            data[sp] = ([], [], [])
        data[sp][0].append(float(row[3]))
        data[sp][1].append(float(row[5]))
        data[sp][2].append(float(row[6]))
    return {sp: (np.array(v[0]), np.array(v[1]), np.array(v[2]))
            for sp, v in data.items()}


# Column layout is identical for dN and dn CSVs.
load_urqmd_dn_pt_spectra      = load_urqmd_pt_spectra
load_urqmd_dn_y_distributions = load_urqmd_y_distributions


def load_urqmd_ratio_pt(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = _skip_comments_lines(path)
    reader = csv.reader(rows)
    next(reader, None)
    pts, Rs, Res = [], [], []
    for row in reader:
        if len(row) < 8:
            continue
        pts.append(float(row[2]))
        Rs.append(float(row[6]) if row[6] else math.nan)
        Res.append(float(row[7]) if row[7] else math.nan)
    return np.array(pts), np.array(Rs), np.array(Res)


def load_urqmd_ratio_y(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = _skip_comments_lines(path)
    reader = csv.reader(rows)
    next(reader, None)
    yc, Rk, Rk_e = [], [], []
    for row in reader:
        if len(row) < 8:
            continue
        yc.append(float(row[2]))
        Rk.append(float(row[6]) if row[6] else math.nan)
        Rk_e.append(float(row[7]) if row[7] else math.nan)
    return np.array(yc), np.array(Rk), np.array(Rk_e)


def load_figure7(
    path: Path,
) -> Dict[Tuple[float, float], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Load Figure7.csv (HEPdata d^2n/dy dpT 2D K0S spectrum).
    Units are d^2n/dydpT (per-event, confirmed from HEPdata header).
    Total uncertainty = quadrature sum of stat + sys.
    """
    rows = _skip_comments_lines(path)
    reader = csv.reader(rows)
    next(reader, None)

    data: Dict[Tuple[float, float], Tuple[List, List, List, List]] = {}
    for row in reader:
        if len(row) < 11:
            continue
        try:
            y_lo  = float(row[1])
            y_hi  = float(row[2])
            pt    = float(row[3])
            val   = float(row[6])
            sp    = abs(float(row[7]))
            sm    = abs(float(row[8]))
            sysp  = abs(float(row[9]))
            sysm  = abs(float(row[10]))
        except (ValueError, IndexError):
            continue
        e_up = math.hypot(sp, sysp)
        e_dn = math.hypot(sm, sysm)
        key = (y_lo, y_hi)
        if key not in data:
            data[key] = ([], [], [], [])
        data[key][0].append(pt)
        data[key][1].append(val)
        data[key][2].append(e_up)
        data[key][3].append(e_dn)

    return {
        k: (np.array(v[0]), np.array(v[1]), np.array(v[2]), np.array(v[3]))
        for k, v in data.items()
    }


def _load_urqmd_k0s_pt_in_ybin(
    urqmd_dir: Path,
    y_lo: float,
    y_hi: float,
    use_dn: bool = True,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Build a per-event d^2n/dy dpT estimate for K0S.
    Scales dn/dpT (integrated over y) by the fraction of dn/dy that
    falls in the slice [y_lo, y_hi], then divides by the bin width.
    Returns (pt_centers, d2n_dydpt, err) or None.
    """
    pt_file = urqmd_dir / ("pt_spectra_dn.csv" if use_dn else "pt_spectra.csv")
    y_file  = urqmd_dir / ("y_distributions_dn.csv" if use_dn else "y_distributions.csv")

    if not pt_file.exists() or not y_file.exists():
        return None

    pt_data = load_urqmd_pt_spectra(pt_file)
    y_data  = load_urqmd_y_distributions(y_file)

    if "K0S" not in pt_data or "K0S" not in y_data:
        return None

    pt_centers, dn_dpt, dn_dpt_err = pt_data["K0S"]
    y_centers, dn_dy, dn_dy_err   = y_data["K0S"]

    if len(y_centers) == 0 or len(pt_centers) == 0:
        return None

    dy = y_centers[1] - y_centers[0] if len(y_centers) > 1 else 1.0
    mask = (y_centers >= y_lo - 1e-6) & (y_centers < y_hi + 1e-6)
    if not mask.any():
        return None

    dn_in_slice     = np.sum(dn_dy[mask] * dy)
    dn_in_slice_err = math.sqrt(np.sum((dn_dy_err[mask] * dy) ** 2))
    dn_total        = np.sum(dn_dy * dy)
    if dn_total <= 0:
        return None

    frac     = dn_in_slice / dn_total
    frac_err = dn_in_slice_err / dn_total

    dy_bin  = y_hi - y_lo
    d2n     = dn_dpt * frac / dy_bin
    d2n_err = np.sqrt(
        (dn_dpt_err * frac / dy_bin) ** 2
        + (dn_dpt    * frac_err / dy_bin) ** 2
    )
    return pt_centers, d2n, d2n_err


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def _save(fig, base_path: Path) -> None:
    fig.savefig(base_path.with_suffix(".eps"), format="eps", bbox_inches="tight")
    fig.savefig(base_path.with_suffix(".png"), dpi=200, bbox_inches="tight")


# ---------------------------------------------------------------------------
# Statistical-error band helper (black-and-white, PAN-compliant)
# ---------------------------------------------------------------------------
#
# The UrQMD modelling is drawn as a continuous line (solid for K0S, dashed for
# (K++K-)/2), so its bin-by-bin MC statistical error is shown as a HATCHED
# band hugging the line -- no solid fill and no color, so it stays PAN B&W and
# never merges with the solid-grey NA61/SHINE experimental band.  On a falling
# spectrum the fractional error is small near the peak and grows in the tail;
# the hatch texture keeps it legible without turning the curve into points.

def _stat_hatch_band(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray,
    *,
    hatch: str = "////",
    nsigma: float = 1.0,
    log_safe: bool = False,
    label: Optional[str] = None,
) -> None:
    """Shade a +/- nsigma statistical band with black hatching (no fill).

    Only finite points with a positive error contribute.  ``log_safe`` clips
    the lower edge to a small positive fraction of y so the band renders on a
    log-scaled axis.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    yerr = np.asarray(yerr, dtype=float)
    if x.size == 0 or yerr.size != y.size:
        return
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(yerr) & (yerr > 0)
    if not finite.any():
        return
    xf = x[finite]
    yf = y[finite]
    ef = nsigma * yerr[finite]
    lo = yf - ef
    hi = yf + ef
    if log_safe:
        floor = np.where(yf > 0, yf * 1e-3, 0.0)
        lo = np.maximum(lo, floor)
    ax.fill_between(
        xf, lo, hi,
        facecolor="none", edgecolor="black",
        hatch=hatch, linewidth=0.0, zorder=1.5,
        label=label,
    )


# ---------------------------------------------------------------------------
# Legend label helpers
# ---------------------------------------------------------------------------

def _urqmd_overlay_label(species_tex: str, modified: bool) -> str:
    prefix = r"UrQMD(3:1)" if modified else r"UrQMD"
    return rf"{prefix} {species_tex}"


def _urqmd_ratio_label(modified: bool) -> str:
    base = r"UrQMD(3:1)" if modified else r"UrQMD"
    return base


# ---------------------------------------------------------------------------
# Boltzmann fit helpers for R(pT) experimental band
# ---------------------------------------------------------------------------

_KAON_MASS = 0.4937  # GeV/c^2


def _boltzmann(pt: np.ndarray, A: float, T: float) -> np.ndarray:
    return A * pt * np.exp(-np.sqrt(pt**2 + _KAON_MASS**2) / T)


def _fit_boltzmann(
    x: np.ndarray, y: np.ndarray, sigma: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if not _SCIPY_OK or len(x) < 3:
        return None, None
    T0_guess = 0.20
    A0_guess = float(np.max(y)) / (_boltzmann(np.array([0.3]), 1.0, T0_guess)[0])
    for p0 in ([A0_guess, T0_guess], [A0_guess * 2, 0.15], [A0_guess * 0.5, 0.30]):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, pcov = _scipy_curve_fit(
                    _boltzmann, x, y, p0=p0, sigma=sigma,
                    absolute_sigma=True, bounds=([0.0, 0.05], [1e6, 1.0]),
                    maxfev=10000,
                )
            if np.all(np.isfinite(pcov)) and pcov[0, 0] > 0 and pcov[1, 1] > 0:
                return popt, pcov
        except Exception:
            continue
    return None, None


def _ratio_band_from_fits(
    hep2a_x, hep2a_y, hep2a_ep, hep2a_em,
    hep2b_x, hep2b_y, hep2b_ep, hep2b_em,
    pt_fine,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Fit Boltzmann to K0S and (K++K-)/2 independently, return R=kch/k0s
    with analytically propagated error band on pt_fine.
    """
    sig_k0s = 0.5 * (hep2a_ep + hep2a_em)
    sig_kch = 0.5 * (hep2b_ep + hep2b_em)
    popt_k0s, pcov_k0s = _fit_boltzmann(hep2a_x, hep2a_y, sig_k0s)
    popt_kch, pcov_kch = _fit_boltzmann(hep2b_x, hep2b_y, sig_kch)
    if popt_k0s is None or popt_kch is None:
        return None, None

    f_k0s = _boltzmann(pt_fine, *popt_k0s)
    f_kch = _boltzmann(pt_fine, *popt_kch)
    with np.errstate(invalid="ignore", divide="ignore"):
        R_fit = np.where(f_k0s > 0, f_kch / f_k0s, np.nan)

    E = np.sqrt(pt_fine**2 + _KAON_MASS**2)
    dR_dA_kch  =  R_fit / popt_kch[0]
    dR_dT_kch  =  R_fit * E / popt_kch[1]**2
    dR_dA_k0s  = -R_fit / popt_k0s[0]
    dR_dT_k0s  = -R_fit * E / popt_k0s[1]**2
    var_R = (
        dR_dA_kch**2 * pcov_kch[0, 0] + dR_dT_kch**2 * pcov_kch[1, 1]
        + 2 * dR_dA_kch * dR_dT_kch * pcov_kch[0, 1]
        + dR_dA_k0s**2 * pcov_k0s[0, 0] + dR_dT_k0s**2 * pcov_k0s[1, 1]
        + 2 * dR_dA_k0s * dR_dT_k0s * pcov_k0s[0, 1]
    )
    R_sigma = np.where(var_R >= 0, np.sqrt(np.abs(var_R)), np.nan)
    return R_fit, R_sigma


def _apply_ratio_band(
    ax, hep2a_x, hep2a_y, hep2a_ep, hep2a_em,
    hep2b_x, hep2b_y, hep2b_ep, hep2b_em,
) -> None:
    """Compute and draw the NA61/SHINE R(pT) band on ax. Falls back to error bars."""
    pt_lo   = max(min(hep2a_x.min(), hep2b_x.min()) - 0.05, 0.0)
    pt_hi   = max(hep2a_x.max(), hep2b_x.max()) + 0.1
    pt_fine = np.linspace(pt_lo, pt_hi, 300)

    R_fit, R_sigma = _ratio_band_from_fits(
        hep2a_x, hep2a_y, hep2a_ep, hep2a_em,
        hep2b_x, hep2b_y, hep2b_ep, hep2b_em,
        pt_fine,
    )
    if R_fit is not None:
        finite_b = np.isfinite(R_fit) & np.isfinite(R_sigma)
        if finite_b.any():
            ax.fill_between(
                pt_fine[finite_b],
                (R_fit - R_sigma)[finite_b],
                (R_fit + R_sigma)[finite_b],
                color="0.65",
                label=r"NA61/SHINE",
            )
    else:
        print("  [warn] Boltzmann fit failed -- falling back to bin-by-bin R(pT) errors")
        kch_on_k0s = np.interp(hep2a_x, hep2b_x, hep2b_y)
        kch_ep_i   = np.interp(hep2a_x, hep2b_x, hep2b_ep)
        kch_em_i   = np.interp(hep2a_x, hep2b_x, hep2b_em)
        with np.errstate(invalid="ignore", divide="ignore"):
            R_fb = np.where(hep2a_y > 0, kch_on_k0s / hep2a_y, np.nan)
            R_err_fb = np.abs(R_fb) * np.sqrt(
                ((0.5*(kch_ep_i + kch_em_i)) / np.where(kch_on_k0s > 0, kch_on_k0s, 1.0))**2 +
                ((0.5*(hep2a_ep + hep2a_em)) / np.where(hep2a_y > 0, hep2a_y, 1.0))**2
            )
        finite_e = np.isfinite(R_fb)
        if finite_e.any():
            ax.errorbar(hep2a_x[finite_e], R_fb[finite_e],
                        yerr=R_err_fb[finite_e],
                        label=r"NA61/SHINE", **STYLE["ratio_exp"])


# ---------------------------------------------------------------------------
# rap_overlay -- UrQMD dN/dy (single variant) vs NA61/SHINE dn/dy
# ---------------------------------------------------------------------------

def make_rap_overlay(
    urqmd_dir: Path,
    hep1a: Path,
    hep1b: Path,
    outpath: Path,
    modified: bool = False,
) -> None:
    """Single panel: UrQMD dN/dy (single variant) vs NA61/SHINE dn/dy. No title."""
    yc, kch_vals, kch_err, k0s_vals, k0s_err = load_urqmd_ydist_meancharged(
        urqmd_dir / "y_distributions_meancharged.csv"
    )
    hy_k0s_x, hy_k0s_y, hy_k0s_ep, hy_k0s_em = load_hepdata(hep1a)
    hy_kch_x, hy_kch_y, hy_kch_ep, hy_kch_em = load_hepdata(hep1b)

    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    ax.plot(yc, k0s_vals,
            label=_urqmd_overlay_label(r"$K^0_S$", modified),
            **STYLE["urqmd_k0s"])
    ax.plot(yc, kch_vals,
            label=_urqmd_overlay_label(r"$(K^+ {+} K^-)/2$", modified),
            **STYLE["urqmd_kch"])
    ax.errorbar(hy_k0s_x, hy_k0s_y, yerr=[hy_k0s_em, hy_k0s_ep],
                label=r"NA61/SHINE $K^0_S$ ($dn/dy$)", **STYLE["exp_k0s"])
    ax.errorbar(hy_kch_x, hy_kch_y, yerr=[hy_kch_em, hy_kch_ep],
                label=r"NA61/SHINE $(K^+ {+} K^-)/2$ ($dn/dy$)", **STYLE["exp_kch"])

    ax.set_xlabel(r"$y$")
    ax.set_ylabel(r"$dN/dy$")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.legend(loc="upper right")
    fig.tight_layout()
    _save(fig, outpath)
    plt.close(fig)
    print(f"  wrote {outpath.with_suffix('.eps')}")


# ---------------------------------------------------------------------------
# pt_overlay -- UrQMD dN/dpT (single variant) vs NA61/SHINE dn/dpT
# ---------------------------------------------------------------------------

def make_pt_overlay(
    urqmd_dir: Path,
    hep2a: Path,
    hep2b: Path,
    outpath: Path,
    modified: bool = False,
) -> None:
    """
    Two-panel: UrQMD dN/dpT linear (top) + R(pT) (bottom) vs NA61/SHINE dn/dpT.
    Single variant per figure.
    """
    pt_data = load_urqmd_pt_spectra(urqmd_dir / "pt_spectra.csv")
    pt_r, R_urqmd, R_urqmd_err = load_urqmd_ratio_pt(urqmd_dir / "ratio_pt.csv")

    k0s_pt, k0s_val, k0s_err = pt_data.get("K0S",    (np.array([]),)*3)
    kp_pt,  kp_val,  kp_err = pt_data.get("Kplus",  (np.array([]),)*3)
    km_pt,  km_val,  km_err = pt_data.get("Kminus", (np.array([]),)*3)

    if len(kp_val) == len(km_val) > 0:
        kch_pt  = kp_pt
        kch_val = 0.5 * (kp_val + km_val)
        kch_err = 0.5 * np.sqrt(kp_err**2 + km_err**2)
    else:
        kch_pt = kch_val = kch_err = np.array([])

    hep2a_x, hep2a_y, hep2a_ep, hep2a_em = load_hepdata(hep2a)
    hep2b_x, hep2b_y, hep2b_ep, hep2b_em = load_hepdata(hep2b)

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(6.5, 7.5),
        gridspec_kw={"height_ratios": [2.5, 1], "hspace": 0.05},
        sharex=True,
    )

    if len(k0s_val) > 0:
        ax_top.plot(k0s_pt, k0s_val,
                    label=_urqmd_overlay_label(r"$K^0_S$", modified),
                    **STYLE["urqmd_k0s"])
    if len(kch_val) > 0:
        ax_top.plot(kch_pt, kch_val,
                    label=_urqmd_overlay_label(r"$(K^+ {+} K^-)/2$", modified),
                    **STYLE["urqmd_kch"])
    ax_top.errorbar(hep2a_x, hep2a_y, yerr=[hep2a_em, hep2a_ep],
                    label=r"NA61/SHINE $K^0_S$ ($dn/dp_T$)", **STYLE["exp_k0s"])
    ax_top.errorbar(hep2b_x, hep2b_y, yerr=[hep2b_em, hep2b_ep],
                    label=r"NA61/SHINE $(K^+ {+} K^-)/2$ ($dn/dp_T$)", **STYLE["exp_kch"])

    ax_top.yaxis.set_minor_locator(AutoMinorLocator())
    ax_top.set_ylabel(r"$dN/dp_T$ $[(\mathrm{GeV}/c)^{-1}]$")
    ax_top.legend(loc="upper right", fontsize=8)
    ax_top.tick_params(labelbottom=False)

    ax_bot.axhline(1.0, ls=":", lw=0.9, color="black")
    finite_u = np.isfinite(R_urqmd)
    if finite_u.any():
        ax_bot.errorbar(pt_r[finite_u], R_urqmd[finite_u],
                        yerr=R_urqmd_err[finite_u],
                        label=_urqmd_ratio_label(modified),
                        capsize=1.5, elinewidth=0.7,
                        **STYLE["ratio_urqmd"])
    _apply_ratio_band(ax_bot,
                      hep2a_x, hep2a_y, hep2a_ep, hep2a_em,
                      hep2b_x, hep2b_y, hep2b_ep, hep2b_em)

    ax_bot.set_xlabel(r"$p_T\;[\mathrm{GeV}/c]$")
    ax_bot.set_ylabel(r"$R(p_T)$")
    ax_bot.xaxis.set_minor_locator(AutoMinorLocator())
    ax_bot.yaxis.set_minor_locator(AutoMinorLocator())
    ax_bot.legend(loc="upper right", fontsize=8)
    fig.align_ylabels([ax_top, ax_bot])
    fig.tight_layout()
    _save(fig, outpath)
    plt.close(fig)
    print(f"  wrote {outpath.with_suffix('.eps')}")


# ---------------------------------------------------------------------------
# PAN-style UrQMD-only plots  --  dN/dy  and  dN/dpT
# ---------------------------------------------------------------------------

def make_pan_y_species(urqmd_dir: Path, outpath: Path) -> None:
    """PAN-style dN/dy for K+, K-, K0S. No title."""
    data = load_urqmd_y_distributions(urqmd_dir / "y_distributions.csv")
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    for sp in ("Kplus", "Kminus", "K0S"):
        if sp not in data:
            continue
        yc, val, err = data[sp]
        ax.errorbar(yc, val, yerr=err,
                    label=SPECIES_LABEL.get(sp, sp), **STYLE.get(sp, {}))
    ax.set_xlabel(r"$y$")
    ax.set_ylabel(r"$dN/dy$")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.legend(loc="upper right")
    fig.tight_layout()
    _save(fig, outpath)
    plt.close(fig)
    print(f"  wrote {outpath.with_suffix('.eps')}")


def make_pan_pt_species(urqmd_dir: Path, outpath: Path) -> None:
    """PAN-style dN/dpT (log) for K+, K-, K0S. No title."""
    data = load_urqmd_pt_spectra(urqmd_dir / "pt_spectra.csv")
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    for sp in ("Kplus", "Kminus", "K0S"):
        if sp not in data:
            continue
        pt, val, err = data[sp]
        ax.errorbar(pt, val, yerr=err,
                    label=SPECIES_LABEL.get(sp, sp), **STYLE.get(sp, {}))
    ax.set_yscale("log")
    ax.set_xlabel(r"$p_T\;[\mathrm{GeV}/c]$")
    ax.set_ylabel(r"$dN/dp_T\;[(\mathrm{GeV}/c)^{-1}]$")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.legend(loc="upper right")
    fig.tight_layout()
    _save(fig, outpath)
    plt.close(fig)
    print(f"  wrote {outpath.with_suffix('.eps')}")


# ---------------------------------------------------------------------------
# PAN-style UrQMD-only plots  --  dn/dy  and  dn/dpT  (per-event)
# ---------------------------------------------------------------------------

def make_pan_dn_y_species(urqmd_dir: Path, outpath: Path) -> None:
    """PAN-style dn/dy (per-event) for K+, K-, K0S. No title."""
    csv_path = urqmd_dir / "y_distributions_dn.csv"
    if not csv_path.exists():
        print(f"  [warn] {csv_path} not found -- skipping dn/dy plot")
        return
    data = load_urqmd_dn_y_distributions(csv_path)
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    for sp in ("Kplus", "Kminus", "K0S"):
        if sp not in data:
            continue
        yc, val, err = data[sp]
        ax.errorbar(yc, val, yerr=err,
                    label=SPECIES_LABEL.get(sp, sp), **STYLE.get(sp, {}))
    ax.set_xlabel(r"$y$")
    ax.set_ylabel(r"$dn/dy$")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.legend(loc="upper right")
    fig.tight_layout()
    _save(fig, outpath)
    plt.close(fig)
    print(f"  wrote {outpath.with_suffix('.eps')}")


def make_pan_dn_pt_species(urqmd_dir: Path, outpath: Path) -> None:
    """PAN-style dn/dpT (log, per-event) for K+, K-, K0S. No title."""
    csv_path = urqmd_dir / "pt_spectra_dn.csv"
    if not csv_path.exists():
        print(f"  [warn] {csv_path} not found -- skipping dn/dpT plot")
        return
    data = load_urqmd_dn_pt_spectra(csv_path)
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    for sp in ("Kplus", "Kminus", "K0S"):
        if sp not in data:
            continue
        pt, val, err = data[sp]
        ax.errorbar(pt, val, yerr=err,
                    label=SPECIES_LABEL.get(sp, sp), **STYLE.get(sp, {}))
    ax.set_yscale("log")
    ax.set_xlabel(r"$p_T\;[\mathrm{GeV}/c]$")
    ax.set_ylabel(r"$dn/dp_T\;[(\mathrm{GeV}/c)^{-1}]$")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.legend(loc="upper right")
    fig.tight_layout()
    _save(fig, outpath)
    plt.close(fig)
    print(f"  wrote {outpath.with_suffix('.eps')}")


# ---------------------------------------------------------------------------
# PAN-style R_K(y)
# ---------------------------------------------------------------------------

def make_pan_ratio_y(urqmd_dir: Path, outpath: Path) -> None:
    """PAN-style R_K(y) = 0.5*(K++K-)/K0S vs y. No title."""
    yc, Rk, Rk_e = load_urqmd_ratio_y(urqmd_dir / "ratio_y.csv")
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    finite = np.isfinite(Rk)
    ax.axhline(1.0, ls=":", lw=0.9, color="black")
    if finite.any():
        ax.errorbar(yc[finite], Rk[finite], yerr=Rk_e[finite],
                    label=r"$R_K(y)$",
                    capsize=1.5, elinewidth=0.7, **STYLE["ratio_y"])
    ax.set_xlabel(r"$y$")
    ax.set_ylabel(r"$R_K(y)$")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.legend(loc="upper right")
    fig.tight_layout()
    _save(fig, outpath)
    plt.close(fig)
    print(f"  wrote {outpath.with_suffix('.eps')}")


# ---------------------------------------------------------------------------
# dn_rap_overlay -- UrQMD dn/dy (single variant) vs NA61/SHINE dn/dy
# Both sides are dn -- no rescaling needed.
# ---------------------------------------------------------------------------

def make_dn_rap_overlay(
    urqmd_dir: Path,
    hep1a: Path,
    hep1b: Path,
    outpath: Path,
    modified: bool = False,
) -> None:
    """
    Single panel: UrQMD dn/dy (single variant) vs NA61/SHINE dn/dy.
    HEPdata Figure1a/b are confirmed dn/dy; no rescaling needed.
    """
    csv_path = urqmd_dir / "y_distributions_dn.csv"
    if not csv_path.exists():
        print(f"  [warn] {csv_path} not found -- skipping dn_rap_overlay")
        return

    data = load_urqmd_dn_y_distributions(csv_path)
    hy_k0s_x, hy_k0s_y, hy_k0s_ep, hy_k0s_em = load_hepdata(hep1a)
    hy_kch_x, hy_kch_y, hy_kch_ep, hy_kch_em = load_hepdata(hep1b)

    fig, ax = plt.subplots(figsize=(6.5, 5.0))

    if "K0S" in data:
        yc, val, err = data["K0S"]
        ax.errorbar(yc, val, yerr=err,
                    label=_urqmd_overlay_label(r"$K^0_S$", modified),
                    ls="-", lw=1.4, marker="^", ms=4,
                    mfc="white", mec="black", color="black",
                    capsize=2, elinewidth=0.7)

    if "Kplus" in data and "Kminus" in data:
        yc_p, vp, ep = data["Kplus"]
        yc_m, vm, em = data["Kminus"]
        if len(yc_p) == len(yc_m):
            kch_val = 0.5 * (vp + vm)
            kch_err = 0.5 * np.sqrt(ep**2 + em**2)
            ax.errorbar(yc_p, kch_val, yerr=kch_err,
                        label=_urqmd_overlay_label(r"$(K^+ {+} K^-)/2$", modified),
                        ls="--", lw=1.4, marker="o", ms=4,
                        mfc="white", mec="black", color="black",
                        capsize=2, elinewidth=0.7)

    ax.errorbar(hy_k0s_x, hy_k0s_y, yerr=[hy_k0s_em, hy_k0s_ep],
                label=r"NA61/SHINE $K^0_S$", **STYLE["exp_k0s"])
    ax.errorbar(hy_kch_x, hy_kch_y, yerr=[hy_kch_em, hy_kch_ep],
                label=r"NA61/SHINE $(K^+ {+} K^-)/2$", **STYLE["exp_kch"])

    ax.set_xlabel(r"$y$")
    ax.set_ylabel(r"$dn/dy$")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    _save(fig, outpath)
    plt.close(fig)
    print(f"  wrote {outpath.with_suffix('.eps')}")


# ---------------------------------------------------------------------------
# dn_rap_overlay_ratio_y -- TWO-PANEL: dn/dy overlay (top) + R_K(y) (bottom)
# Top panel: UrQMD dn/dy + NA61/SHINE data.  Bottom: UrQMD R_K(y) only.
# ---------------------------------------------------------------------------

def make_dn_rap_overlay_with_ratio_y(
    urqmd_dir: Path,
    hep1a: Path,
    hep1b: Path,
    outpath: Path,
    modified: bool = False,
) -> None:
    """
    Two-panel combined figure:
      top panel:    dn/dy vs y -- UrQMD K0S + (K++K-)/2 overlaid with NA61/SHINE
      bottom panel: R_K(y) = UrQMD (K++K-)/2 / K0S vs y (UrQMD only)

    Layout identical to dn_pt_plain / dn_rap_plain_ratio_y:
      figsize (6.5, 7.5), height_ratios [2.5, 1], hspace 0.05, sharex=True.
    One figure per variant (unmod/mod).
    """
    csv_y = urqmd_dir / "y_distributions_dn.csv"
    csv_r = urqmd_dir / "ratio_y.csv"

    if not csv_y.exists():
        print(f"  [warn] {csv_y} not found -- skipping dn_rap_overlay_ratio_y")
        return
    if not csv_r.exists():
        print(f"  [warn] {csv_r} not found -- skipping dn_rap_overlay_ratio_y")
        return

    data = load_urqmd_dn_y_distributions(csv_y)
    hy_k0s_x, hy_k0s_y, hy_k0s_ep, hy_k0s_em = load_hepdata(hep1a)
    hy_kch_x, hy_kch_y, hy_kch_ep, hy_kch_em = load_hepdata(hep1b)
    yc_r, Rk, Rk_e = load_urqmd_ratio_y(csv_r)

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(6.5, 7.5),
        gridspec_kw={"height_ratios": [2.5, 1], "hspace": 0.05},
        sharex=True,
    )

    # --- top panel: dn/dy overlay ---
    if "K0S" in data:
        yc, val, err = data["K0S"]
        ax_top.errorbar(yc, val, yerr=err,
                        label=_urqmd_overlay_label(r"$K^0_S$", modified),
                        ls="-", lw=1.4, marker="^", ms=4,
                        mfc="white", mec="black", color="black",
                        capsize=2, elinewidth=0.7)

    if "Kplus" in data and "Kminus" in data:
        yc_p, vp, ep = data["Kplus"]
        yc_m, vm, em = data["Kminus"]
        if len(yc_p) == len(yc_m):
            kch_val = 0.5 * (vp + vm)
            kch_err = 0.5 * np.sqrt(ep**2 + em**2)
            ax_top.errorbar(yc_p, kch_val, yerr=kch_err,
                            label=_urqmd_overlay_label(r"$(K^+ {+} K^-)/2$", modified),
                            ls="--", lw=1.4, marker="o", ms=4,
                            mfc="white", mec="black", color="black",
                            capsize=2, elinewidth=0.7)

    ax_top.errorbar(hy_k0s_x, hy_k0s_y, yerr=[hy_k0s_em, hy_k0s_ep],
                    label=r"NA61/SHINE $K^0_S$", **STYLE["exp_k0s"])
    ax_top.errorbar(hy_kch_x, hy_kch_y, yerr=[hy_kch_em, hy_kch_ep],
                    label=r"NA61/SHINE $(K^+ {+} K^-)/2$", **STYLE["exp_kch"])

    ax_top.yaxis.set_minor_locator(AutoMinorLocator())
    ax_top.set_ylabel(r"$dn/dy$")
    ax_top.legend(loc="upper right", fontsize=8)
    ax_top.tick_params(labelbottom=False)

    # --- bottom panel: R_K(y) ---
    ax_bot.axhline(1.0, ls=":", lw=0.9, color="black")
    finite = np.isfinite(Rk)
    if finite.any():
        ax_bot.errorbar(yc_r[finite], Rk[finite], yerr=Rk_e[finite],
                        label=_urqmd_ratio_label(modified),
                        capsize=1.5, elinewidth=0.7, **STYLE["ratio_y"])

    ax_bot.set_xlabel(r"$y$")
    ax_bot.set_ylabel(r"$R_K(y)$")
    ax_bot.xaxis.set_minor_locator(AutoMinorLocator())
    ax_bot.yaxis.set_minor_locator(AutoMinorLocator())
    ax_bot.legend(loc="upper right", fontsize=8)

    fig.align_ylabels([ax_top, ax_bot])
    fig.tight_layout()
    _save(fig, outpath)
    plt.close(fig)
    print(f"  wrote {outpath.with_suffix('.eps')}")


# ---------------------------------------------------------------------------
# dn_pt_overlay -- UrQMD dn/dpT (single variant) vs NA61/SHINE dn/dpT
# Both sides are dn -- no rescaling needed.
# ---------------------------------------------------------------------------

def make_dn_pt_overlay(
    urqmd_dir: Path,
    hep2a: Path,
    hep2b: Path,
    outpath: Path,
    modified: bool = False,
) -> None:
    """
    Two-panel: UrQMD dn/dpT linear (top) + R(pT) (bottom) vs NA61/SHINE dn/dpT.
    Single variant per figure; no rescaling needed.
    """
    csv_pt = urqmd_dir / "pt_spectra_dn.csv"
    if not csv_pt.exists():
        print(f"  [warn] {csv_pt} not found -- skipping dn_pt_overlay")
        return

    data = load_urqmd_dn_pt_spectra(csv_pt)

    k0s = data.get("K0S",    (np.array([]),)*3)
    kp  = data.get("Kplus",  (np.array([]),)*3)
    km  = data.get("Kminus", (np.array([]),)*3)
    if len(kp[1]) == len(km[1]) > 0:
        kch = (kp[0], 0.5*(kp[1]+km[1]), 0.5*np.sqrt(kp[2]**2+km[2]**2))
    else:
        kch = (np.array([]),)*3

    with np.errstate(invalid="ignore", divide="ignore"):
        if len(kch[0]) > 0 and len(k0s[0]) > 0:
            k0s_i   = np.interp(kch[0], k0s[0], k0s[1], left=np.nan, right=np.nan)
            k0s_i_e = np.interp(kch[0], k0s[0], k0s[2], left=np.nan, right=np.nan)
            R_dn    = np.where(k0s_i > 0, kch[1] / k0s_i, np.nan)
            pt_r    = kch[0]
            kch_safe = np.where(kch[1] > 0, kch[1], np.nan)
            R_dn_err = np.abs(R_dn) * np.sqrt(
                (kch[2] / kch_safe) ** 2 + (k0s_i_e / np.where(k0s_i > 0, k0s_i, np.nan)) ** 2
            )
        else:
            pt_r = np.array([])
            R_dn = np.array([])
            R_dn_err = np.array([])

    hep2a_x, hep2a_y, hep2a_ep, hep2a_em = load_hepdata(hep2a)
    hep2b_x, hep2b_y, hep2b_ep, hep2b_em = load_hepdata(hep2b)

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(6.5, 7.5),
        gridspec_kw={"height_ratios": [2.5, 1], "hspace": 0.05},
        sharex=True,
    )

    if len(k0s[1]) > 0:
        _stat_hatch_band(ax_top, k0s[0], k0s[1], k0s[2])
        ax_top.plot(k0s[0], k0s[1],
                    label=_urqmd_overlay_label(r"$K^0_S$", modified),
                    **STYLE["urqmd_k0s"])
    if len(kch[1]) > 0:
        ax_top.plot(kch[0], kch[1],
                    label=_urqmd_overlay_label(r"$(K^+ {+} K^-)/2$", modified),
                    **STYLE["urqmd_kch"])
    ax_top.errorbar(hep2a_x, hep2a_y, yerr=[hep2a_em, hep2a_ep],
                    label=r"NA61/SHINE $K^0_S$", **STYLE["exp_k0s"])
    ax_top.errorbar(hep2b_x, hep2b_y, yerr=[hep2b_em, hep2b_ep],
                    label=r"NA61/SHINE $(K^+ {+} K^-)/2$", **STYLE["exp_kch"])

    ax_top.yaxis.set_minor_locator(AutoMinorLocator())
    ax_top.set_ylabel(r"$dn/dp_T\;[(\mathrm{GeV}/c)^{-1}]$")
    ax_top.legend(loc="upper right", fontsize=8)
    ax_top.tick_params(labelbottom=False)

    ax_bot.axhline(1.0, ls=":", lw=0.9, color="black")
    finite_r = np.isfinite(R_dn)
    if finite_r.any():
        ax_bot.errorbar(pt_r[finite_r], R_dn[finite_r],
                        yerr=R_dn_err[finite_r],
                        label=_urqmd_ratio_label(modified),
                        capsize=1.5, elinewidth=0.7,
                        **STYLE["ratio_urqmd"])
    _apply_ratio_band(ax_bot,
                      hep2a_x, hep2a_y, hep2a_ep, hep2a_em,
                      hep2b_x, hep2b_y, hep2b_ep, hep2b_em)

    ax_bot.set_xlabel(r"$p_T\;[\mathrm{GeV}/c]$")
    ax_bot.set_ylabel(r"$R(p_T)$")
    ax_bot.xaxis.set_minor_locator(AutoMinorLocator())
    ax_bot.yaxis.set_minor_locator(AutoMinorLocator())
    ax_bot.legend(loc="upper right", fontsize=8)
    fig.align_ylabels([ax_top, ax_bot])
    fig.tight_layout()
    _save(fig, outpath)
    plt.close(fig)
    print(f"  wrote {outpath.with_suffix('.eps')}")


# ---------------------------------------------------------------------------
# K0S d^2n/dydpT vs pT in rapidity slices
# ---------------------------------------------------------------------------

def make_k0s_2d_combined_overlay(
    unmod_dir: Path,
    mod_dir: Path,
    hep7_path: Path,
    outpath: Path,
) -> None:
    """
    Multi-panel: K0S d^2n/dy dpT vs pT per rapidity bin (from Figure7.csv).
    NA61/SHINE Figure7 is d^2n/dydpT (per-event, confirmed from HEPdata header).
    UrQMD values are derived from dn/dpT scaled by the y-slice fraction.
    Both unmod and mod UrQMD are overlaid here because each sub-panel
    contains only two theory lines + one exp dataset, which remains readable.
    """
    if not hep7_path.exists():
        print(f"  [warn] {hep7_path} not found -- skipping k0s_2d_combined_overlay plot")
        return
    exp_data = load_figure7(hep7_path)
    if not exp_data:
        print(f"  [warn] No data from {hep7_path} -- skipping k0s_2d_combined_overlay plot")
        return

    y_bins = sorted(exp_data.keys(), key=lambda k: k[0])
    n_bins = len(y_bins)
    n_cols = min(3, n_bins)
    n_rows = math.ceil(n_bins / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(n_cols * 4.0, n_rows * 3.5),
                              squeeze=False)

    for idx, (y_lo, y_hi) in enumerate(y_bins):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row][col]

        pt_exp, val_exp, ep_exp, em_exp = exp_data[(y_lo, y_hi)]
        ax.errorbar(pt_exp, val_exp, yerr=[em_exp, ep_exp],
                    label=r"NA61/SHINE", **STYLE["fig7_exp"])

        res_unmod = _load_urqmd_k0s_pt_in_ybin(unmod_dir, y_lo, y_hi, use_dn=True)
        if res_unmod is not None:
            pt_u, d2n_u, d2n_u_err = res_unmod
            finite_u = np.isfinite(d2n_u)
            if finite_u.any():
                ax.plot(pt_u[finite_u], d2n_u[finite_u],
                        label=r"UrQMD", **STYLE["fig7_urqmd_unmod"])

        res_mod = _load_urqmd_k0s_pt_in_ybin(mod_dir, y_lo, y_hi, use_dn=True)
        if res_mod is not None:
            pt_m, d2n_m, d2n_m_err = res_mod
            finite_m = np.isfinite(d2n_m)
            if finite_m.any():
                ax.plot(pt_m[finite_m], d2n_m[finite_m],
                        label=r"UrQMD(3:1)", **STYLE["fig7_urqmd_mod"])

        y_label = rf"${y_lo:.1f} < y < {y_hi:.1f}$"
        ax.text(0.97, 0.95, y_label, transform=ax.transAxes,
                ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))
        ax.set_yscale("log")
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        if row == n_rows - 1:
            ax.set_xlabel(r"$p_T\;[\mathrm{GeV}/c]$")
        if col == 0:
            ax.set_ylabel(r"$d^2n/dy\,dp_T\;[(\mathrm{GeV}/c)^{-1}]$")
        if idx == 0:
            ax.legend(loc="upper right", fontsize=8)

    for idx in range(n_bins, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    fig.tight_layout()
    _save(fig, outpath)
    plt.close(fig)
    print(f"  wrote {outpath.with_suffix('.eps')}")


# ---------------------------------------------------------------------------
# dn_rap_plain -- UrQMD dn/dy only: (K++K-)/2 vs K0S  (NO experimental data)
# ---------------------------------------------------------------------------

def make_dn_rap_plain(
    urqmd_dir: Path,
    outpath: Path,
    modified: bool = False,
) -> None:
    """
    Single panel: UrQMD dn/dy comparison of (K++K-)/2 and K0S.
    No experimental data -- UrQMD simulation only.
    One figure per variant (unmod/mod).
    """
    csv_path = urqmd_dir / "y_distributions_dn.csv"
    if not csv_path.exists():
        print(f"  [warn] {csv_path} not found -- skipping dn_rap_plain")
        return

    data = load_urqmd_dn_y_distributions(csv_path)

    fig, ax = plt.subplots(figsize=(6.5, 5.0))

    if "K0S" in data:
        yc, val, err = data["K0S"]
        ax.errorbar(yc, val, yerr=err,
                    label=_urqmd_overlay_label(r"$K^0_S$", modified),
                    ls="-", lw=1.4, marker="^", ms=4,
                    mfc="white", mec="black", color="black",
                    capsize=2, elinewidth=0.7)

    if "Kplus" in data and "Kminus" in data:
        yc_p, vp, ep = data["Kplus"]
        yc_m, vm, em = data["Kminus"]
        if len(yc_p) == len(yc_m):
            kch_val = 0.5 * (vp + vm)
            kch_err = 0.5 * np.sqrt(ep**2 + em**2)
            ax.errorbar(yc_p, kch_val, yerr=kch_err,
                        label=_urqmd_overlay_label(r"$(K^+ {+} K^-)/2$", modified),
                        ls="--", lw=1.4, marker="o", ms=4,
                        mfc="white", mec="black", color="black",
                        capsize=2, elinewidth=0.7)

    ax.set_xlabel(r"$y$")
    ax.set_ylabel(r"$dn/dy$")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    _save(fig, outpath)
    plt.close(fig)
    print(f"  wrote {outpath.with_suffix('.eps')}")


# ---------------------------------------------------------------------------
# dn_pt_plain -- UrQMD dn/dpT only: (K++K-)/2 vs K0S  (NO experimental data)
# ---------------------------------------------------------------------------

def make_dn_pt_plain(
    urqmd_dir: Path,
    outpath: Path,
    modified: bool = False,
) -> None:
    """
    Two-panel: UrQMD dn/dpT linear (top) + R(pT) = (K++K-)/2 / K0S (bottom).
    No experimental data -- UrQMD simulation only.
    One figure per variant (unmod/mod).
    """
    csv_pt = urqmd_dir / "pt_spectra_dn.csv"
    if not csv_pt.exists():
        print(f"  [warn] {csv_pt} not found -- skipping dn_pt_plain")
        return

    data = load_urqmd_dn_pt_spectra(csv_pt)

    k0s = data.get("K0S",    (np.array([]),)*3)
    kp  = data.get("Kplus",  (np.array([]),)*3)
    km  = data.get("Kminus", (np.array([]),)*3)
    if len(kp[1]) == len(km[1]) > 0:
        kch = (kp[0], 0.5*(kp[1]+km[1]), 0.5*np.sqrt(kp[2]**2+km[2]**2))
    else:
        kch = (np.array([]),)*3

    with np.errstate(invalid="ignore", divide="ignore"):
        if len(kch[0]) > 0 and len(k0s[0]) > 0:
            k0s_i   = np.interp(kch[0], k0s[0], k0s[1], left=np.nan, right=np.nan)
            k0s_i_e = np.interp(kch[0], k0s[0], k0s[2], left=np.nan, right=np.nan)
            R_dn    = np.where(k0s_i > 0, kch[1] / k0s_i, np.nan)
            pt_r    = kch[0]
            kch_safe = np.where(kch[1] > 0, kch[1], np.nan)
            R_dn_err = np.abs(R_dn) * np.sqrt(
                (kch[2] / kch_safe) ** 2 + (k0s_i_e / np.where(k0s_i > 0, k0s_i, np.nan)) ** 2
            )
        else:
            pt_r = np.array([])
            R_dn = np.array([])
            R_dn_err = np.array([])

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(6.5, 7.5),
        gridspec_kw={"height_ratios": [2.5, 1], "hspace": 0.05},
        sharex=True,
    )

    if len(k0s[1]) > 0:
        _stat_hatch_band(ax_top, k0s[0], k0s[1], k0s[2])
        ax_top.plot(k0s[0], k0s[1],
                    label=_urqmd_overlay_label(r"$K^0_S$", modified),
                    **STYLE["urqmd_k0s"])
    if len(kch[1]) > 0:
        ax_top.plot(kch[0], kch[1],
                    label=_urqmd_overlay_label(r"$(K^+ {+} K^-)/2$", modified),
                    **STYLE["urqmd_kch"])

    ax_top.yaxis.set_minor_locator(AutoMinorLocator())
    ax_top.set_ylabel(r"$dn/dp_T\;[(\mathrm{GeV}/c)^{-1}]$")
    ax_top.legend(loc="upper right", fontsize=8)
    ax_top.tick_params(labelbottom=False)

    ax_bot.axhline(1.0, ls=":", lw=0.9, color="black")
    finite_r = np.isfinite(R_dn)
    if finite_r.any():
        ax_bot.errorbar(pt_r[finite_r], R_dn[finite_r],
                        yerr=R_dn_err[finite_r],
                        label=_urqmd_ratio_label(modified),
                        capsize=1.5, elinewidth=0.7,
                        **STYLE["ratio_urqmd"])

    ax_bot.set_xlabel(r"$p_T\;[\mathrm{GeV}/c]$")
    ax_bot.set_ylabel(r"$R(p_T)$")
    ax_bot.xaxis.set_minor_locator(AutoMinorLocator())
    ax_bot.yaxis.set_minor_locator(AutoMinorLocator())
    ax_bot.legend(loc="upper right", fontsize=8)
    fig.align_ylabels([ax_top, ax_bot])
    fig.tight_layout()
    _save(fig, outpath)
    plt.close(fig)
    print(f"  wrote {outpath.with_suffix('.eps')}")


# ---------------------------------------------------------------------------
# dn_rap_plain_ratio_y -- TWO-PANEL: dn/dy (top) + R_K(y) (bottom)
# Mirrors dn_pt_plain layout.  UrQMD only, no experimental data.
# ---------------------------------------------------------------------------

def make_dn_rap_plain_with_ratio_y(
    urqmd_dir: Path,
    outpath: Path,
    modified: bool = False,
) -> None:
    """
    Two-panel combined figure (UrQMD only, no experimental data):
      top panel:    dn/dy vs y -- K0S (solid) and (K++K-)/2 (dashed)
      bottom panel: R_K(y) = (K++K-)/2 / K0S vs y with dotted reference at 1

    Layout mirrors make_dn_pt_plain:
      figsize (6.5, 7.5), height_ratios [2.5, 1], hspace 0.05, sharex=True.
    One figure per variant (unmod/mod).
    """
    csv_y = urqmd_dir / "y_distributions_dn.csv"
    csv_r = urqmd_dir / "ratio_y.csv"

    if not csv_y.exists():
        print(f"  [warn] {csv_y} not found -- skipping dn_rap_plain_ratio_y")
        return
    if not csv_r.exists():
        print(f"  [warn] {csv_r} not found -- skipping dn_rap_plain_ratio_y")
        return

    data = load_urqmd_dn_y_distributions(csv_y)
    yc_r, Rk, Rk_e = load_urqmd_ratio_y(csv_r)

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(6.5, 7.5),
        gridspec_kw={"height_ratios": [2.5, 1], "hspace": 0.05},
        sharex=True,
    )

    # --- top panel: dn/dy ---
    if "K0S" in data:
        yc, val, err = data["K0S"]
        ax_top.errorbar(yc, val, yerr=err,
                        label=_urqmd_overlay_label(r"$K^0_S$", modified),
                        ls="-", lw=1.4, marker="^", ms=4,
                        mfc="white", mec="black", color="black",
                        capsize=2, elinewidth=0.7)

    if "Kplus" in data and "Kminus" in data:
        yc_p, vp, ep = data["Kplus"]
        yc_m, vm, em = data["Kminus"]
        if len(yc_p) == len(yc_m):
            kch_val = 0.5 * (vp + vm)
            kch_err = 0.5 * np.sqrt(ep**2 + em**2)
            ax_top.errorbar(yc_p, kch_val, yerr=kch_err,
                            label=_urqmd_overlay_label(r"$(K^+ {+} K^-)/2$", modified),
                            ls="--", lw=1.4, marker="o", ms=4,
                            mfc="white", mec="black", color="black",
                            capsize=2, elinewidth=0.7)

    ax_top.yaxis.set_minor_locator(AutoMinorLocator())
    ax_top.set_ylabel(r"$dn/dy$")
    ax_top.legend(loc="upper right", fontsize=8)
    ax_top.tick_params(labelbottom=False)

    # --- bottom panel: R_K(y) ---
    ax_bot.axhline(1.0, ls=":", lw=0.9, color="black")
    finite = np.isfinite(Rk)
    if finite.any():
        ax_bot.errorbar(yc_r[finite], Rk[finite], yerr=Rk_e[finite],
                        label=_urqmd_ratio_label(modified),
                        capsize=1.5, elinewidth=0.7, **STYLE["ratio_y"])

    ax_bot.set_xlabel(r"$y$")
    ax_bot.set_ylabel(r"$R_K(y)$")
    ax_bot.xaxis.set_minor_locator(AutoMinorLocator())
    ax_bot.yaxis.set_minor_locator(AutoMinorLocator())
    ax_bot.legend(loc="upper right", fontsize=8)

    fig.align_ylabels([ax_top, ax_bot])
    fig.tight_layout()
    _save(fig, outpath)
    plt.close(fig)
    print(f"  wrote {outpath.with_suffix('.eps')}")


# ---------------------------------------------------------------------------
# Per-system driver
# ---------------------------------------------------------------------------

def process_system(
    sys_def: SystemDef,
    urqmd_base: Path,
    hep_dir: Path,
    outdir: Path,
    skip_missing_hep: bool = True,
) -> None:
    unmod_dir = urqmd_base / sys_def.unmod_dir
    mod_dir   = urqmd_base / sys_def.mod_dir

    missing_urqmd = [d for d in (unmod_dir, mod_dir) if not d.exists()]
    if missing_urqmd:
        for d in missing_urqmd:
            print(f"  [skip] UrQMD dir not found: {d}")
        return

    hep1a = hep_dir / sys_def.hep1a
    hep1b = hep_dir / sys_def.hep1b
    hep2a = hep_dir / sys_def.hep2a
    hep2b = hep_dir / sys_def.hep2b
    hep_ok = all(p.exists() for p in (hep1a, hep1b, hep2a, hep2b))
    hep7   = hep_dir / sys_def.hep7 if sys_def.hep7 else None

    if not hep_ok:
        for p in (hep1a, hep1b, hep2a, hep2b):
            if not p.exists():
                print(f"  [warn] HepData file not found: {p}")
        if not skip_missing_hep:
            return

    print(f"\n=== {sys_def.label} @ {sys_def.energy_str} ===")

    if hep_ok:
        print("  Rapidity overlay (UrQMD dN/dy vs NA61/SHINE dn/dy)...")
        make_rap_overlay(unmod_dir, hep1a, hep1b,
                         sys_def.out(outdir, "rap_overlay", mod=False), modified=False)
        make_rap_overlay(mod_dir,   hep1a, hep1b,
                         sys_def.out(outdir, "rap_overlay", mod=True),  modified=True)

        print("  pT overlay (UrQMD dN/dpT vs NA61/SHINE dn/dpT, linear+R)...")
        make_pt_overlay(unmod_dir, hep2a, hep2b,
                        sys_def.out(outdir, "pt_overlay", mod=False), modified=False)
        make_pt_overlay(mod_dir,   hep2a, hep2b,
                        sys_def.out(outdir, "pt_overlay", mod=True),  modified=True)

    print("  PAN-style UrQMD-only figures (dN)...")
    make_pan_y_species(unmod_dir,  sys_def.out(outdir, "pan_y_species",  mod=False))
    make_pan_y_species(mod_dir,    sys_def.out(outdir, "pan_y_species",  mod=True))
    make_pan_pt_species(unmod_dir, sys_def.out(outdir, "pan_pt_species", mod=False))
    make_pan_pt_species(mod_dir,   sys_def.out(outdir, "pan_pt_species", mod=True))

    print("  PAN-style UrQMD-only figures (dn per-event)...")
    make_pan_dn_y_species(unmod_dir,  sys_def.out(outdir, "pan_dn_y_species",  mod=False))
    make_pan_dn_y_species(mod_dir,    sys_def.out(outdir, "pan_dn_y_species",  mod=True))
    make_pan_dn_pt_species(unmod_dir, sys_def.out(outdir, "pan_dn_pt_species", mod=False))
    make_pan_dn_pt_species(mod_dir,   sys_def.out(outdir, "pan_dn_pt_species", mod=True))

    print("  PAN-style ratio R_K(y)...")
    make_pan_ratio_y(unmod_dir, sys_def.out(outdir, "pan_ratio_y", mod=False))
    make_pan_ratio_y(mod_dir,   sys_def.out(outdir, "pan_ratio_y", mod=True))

    if hep_ok:
        print("  dn/dy overlay (single variant per file)...")
        make_dn_rap_overlay(unmod_dir, hep1a, hep1b,
                            sys_def.out(outdir, "dn_rap_overlay", mod=False), modified=False)
        make_dn_rap_overlay(mod_dir,   hep1a, hep1b,
                            sys_def.out(outdir, "dn_rap_overlay", mod=True),  modified=True)

        print("  dn/dy overlay + R_K(y) combined (two-panel)...")
        make_dn_rap_overlay_with_ratio_y(
            unmod_dir, hep1a, hep1b,
            sys_def.out(outdir, "dn_rap_overlay_ratio_y", mod=False), modified=False)
        make_dn_rap_overlay_with_ratio_y(
            mod_dir,   hep1a, hep1b,
            sys_def.out(outdir, "dn_rap_overlay_ratio_y", mod=True),  modified=True)

        print("  dn/dpT overlay (single variant per file, linear+R)...")
        make_dn_pt_overlay(unmod_dir, hep2a, hep2b,
                           sys_def.out(outdir, "dn_pt_overlay", mod=False), modified=False)
        make_dn_pt_overlay(mod_dir,   hep2a, hep2b,
                           sys_def.out(outdir, "dn_pt_overlay", mod=True),  modified=True)

    if hep7 is not None and hep7.exists():
        print("  K0S d^2n/dydpT combined overlay (dn, unmod+mod vs NA61/SHINE)...")
        make_k0s_2d_combined_overlay(
            unmod_dir, mod_dir, hep7,
            sys_def.out(outdir, "k0s_2d_combined_overlay"),
        )
    elif hep7 is not None:
        print(f"  [warn] Fig7 HepData not found: {hep7} -- skipping k0s_2d_combined_overlay")

    print("  dn/dy plain UrQMD-only: (K++K-)/2 vs K0S...")
    make_dn_rap_plain(unmod_dir, sys_def.out(outdir, "dn_rap_plain", mod=False), modified=False)
    make_dn_rap_plain(mod_dir,   sys_def.out(outdir, "dn_rap_plain", mod=True),  modified=True)

    print("  dn/dpT plain UrQMD-only: (K++K-)/2 vs K0S + R(pT)...")
    make_dn_pt_plain(unmod_dir, sys_def.out(outdir, "dn_pt_plain", mod=False), modified=False)
    make_dn_pt_plain(mod_dir,   sys_def.out(outdir, "dn_pt_plain", mod=True),  modified=True)

    print("  dn/dy + R_K(y) combined plain UrQMD-only (two-panel)...")
    make_dn_rap_plain_with_ratio_y(
        unmod_dir, sys_def.out(outdir, "dn_rap_plain_ratio_y", mod=False), modified=False)
    make_dn_rap_plain_with_ratio_y(
        mod_dir,   sys_def.out(outdir, "dn_rap_plain_ratio_y", mod=True),  modified=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Generate PAN figures for kaon spectra (Ar+Sc, Xe+Xe, C+C, Xe+W)."
    )
    p.add_argument("--outdir",           default="pan_figures")
    p.add_argument("--hepdata-dir",      default="hep_data")
    p.add_argument("--urqmd-base",       default=".")
    p.add_argument("--systems", nargs="+",
                   choices=[s.tag for s in ALL_SYSTEMS], default=None,
                   help="Limit to specific system tags (default: all)")
    p.add_argument("--skip-missing-hep", action="store_true", default=True)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    outdir     = Path(args.outdir)
    hep_dir    = Path(args.hepdata_dir)
    urqmd_base = Path(args.urqmd_base)
    outdir.mkdir(parents=True, exist_ok=True)

    systems = ALL_SYSTEMS if not args.systems else \
              [s for s in ALL_SYSTEMS if s.tag in args.systems]

    for sys_def in systems:
        process_system(sys_def, urqmd_base, hep_dir, outdir,
                       skip_missing_hep=args.skip_missing_hep)

    print(f"\nDone. Files written to: {outdir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
