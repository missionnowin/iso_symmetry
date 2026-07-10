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

Outputs per system (all as .eps + .png), e.g. for Ar+Sc:
  ArSc_11p9GeV_rap_overlay_unmod.{eps,png}    -- dN/dy UrQMD vs NA61/SHINE
  ArSc_11p9GeV_rap_overlay_mod.{eps,png}
  ArSc_11p9GeV_pt_overlay_unmod.{eps,png}     -- dN/dpT (linear) + R(pT)
  ArSc_11p9GeV_pt_overlay_mod.{eps,png}
  ArSc_11p9GeV_pan_y_species_unmod.{eps,png}  -- dN/dy per species (UrQMD only)
  ArSc_11p9GeV_pan_y_species_mod.{eps,png}
  ArSc_11p9GeV_pan_pt_species_unmod.{eps,png} -- dN/dpT per species (UrQMD only)
  ArSc_11p9GeV_pan_pt_species_mod.{eps,png}
  ArSc_11p9GeV_pan_dn_y_species_unmod.{eps,png}  -- dn/dy per species (per-event)
  ArSc_11p9GeV_pan_dn_y_species_mod.{eps,png}
  ArSc_11p9GeV_pan_dn_pt_species_unmod.{eps,png} -- dn/dpT per species (per-event)
  ArSc_11p9GeV_pan_dn_pt_species_mod.{eps,png}
  ArSc_11p9GeV_pan_ratio_y_unmod.{eps,png}
  ArSc_11p9GeV_pan_ratio_y_mod.{eps,png}
  ArSc_11p9GeV_fig7_k0s_2d_unmod.{eps,png}   -- Fig7-style K0S d2n/dydpT vs exp
  ArSc_11p9GeV_fig7_k0s_2d_mod.{eps,png}

NOTE: energy_str uses 'p' instead of '.' (e.g. 11p9GeV) to avoid
Windows treating the decimal as a file extension separator.

PAN journal style notes
-----------------------
* No in-plot titles -- system/energy/variant info goes in the LaTeX caption.
* UrQMD vs UrQMD(3:1) labels appear only in overlay figs (rap_overlay/pt_overlay)
  where UrQMD curves are directly compared against NA61/SHINE data.
* PAN-only UrQMD plots show only species labels (K+, K-, K0S).
* Grayscale / black-and-white only.
* Curves distinguished by linestyle + marker shape.
* EPS vector output suitable for Yadernaya Fizika submission.
* pt_overlay top panel is LINEAR for both unmod and mod (matching the article
  Fig.2 presentation; log was dropped as it distorts comparison at high pT).

R(pT) bottom panel -- error note
---------------------------------
NA61/SHINE R(pT) errors match the article Fig.2: they are derived from
the ratio of two Boltzmann fits (one per species) with the uncertainty
band obtained by propagating the fit-parameter covariance matrices
analytically.  This produces a smooth shaded band, NOT discrete error bars.
UrQMD R(pT) uses raw bin-by-bin MC statistical errors (line only).

Fig.7 panel note
----------------
The Fig7-style plot shows K0S d^2n/dydpT as a function of pT for each
rapidity bin, overlaying:
  - unmod UrQMD (solid line, no markers)
  - mod  UrQMD  (dashed line, no markers)
  - NA61/SHINE experimental data (filled circles with stat+sys errors)
This uses hep_data/Figure7.csv which contains the 2D (y, pT) spectrum.
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
    # Fig7 styles
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
    tag: str          # e.g. "ArSc"
    label: str        # e.g. "Ar+Sc"  (for captions, not used on-plot)
    energy_str: str   # plain filename-safe energy, e.g. "11p9GeV"  (NO dot!)
    energy_label: str # LaTeX for captions, e.g. r"$\sqrt{s_{NN}}=11.9$ GeV"
    unmod_dir: str
    mod_dir: str
    hep1a: str
    hep1b: str
    hep2a: str
    hep2b: str
    hep7: str = ""    # Figure7 2D K0S spectrum (optional)

    def out(self, outdir: Path, kind: str, mod: bool) -> Path:
        """
        Build output path: {outdir}/{tag}_{energy_str}_{kind}_{mod|unmod}
        e.g. pan_figures/ArSc_11p9GeV_rap_overlay_mod
        """
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
        tag="XeXe", label="Xe+Xe",
        energy_str="9p2GeV",
        energy_label=r"$\sqrt{s_{NN}}=9.2\,\mathrm{GeV}$",
        unmod_dir="XeXe-kaons-9.2GeV-ecm-collision/results",
        mod_dir="XeXe-kaons-9.2GeV-ecm-collision-modified-ud/results",
        hep1a="XeXe_Figure1a.csv", hep1b="XeXe_Figure1b.csv",
        hep2a="XeXe_Figure2a.csv", hep2b="XeXe_Figure2b.csv",
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
    SystemDef(
        tag="XeW", label="Xe+W",
        energy_str="2p9GeV",
        energy_label=r"$\sqrt{s_{NN}}=2.9\,\mathrm{GeV}$",
        unmod_dir="XeW-kaons-2.9GeV-ecm-target/results",
        mod_dir="XeW-kaons-2.9GeV-ecm-target-modified-ud/results",
        hep1a="XeW_Figure1a.csv", hep1b="XeW_Figure1b.csv",
        hep2a="XeW_Figure2a.csv", hep2b="XeW_Figure2b.csv",
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


# load_urqmd_pt_spectra and load_urqmd_y_distributions work for both
# dN and dn CSV files since the column layout is identical (col 5 = value).
load_urqmd_dn_pt_spectra = load_urqmd_pt_spectra
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

    Returns a dict keyed by (y_lo, y_hi) tuples, each containing
    (pt_centers, values, err_up, err_dn) arrays.
    The total uncertainty is taken as quadrature sum of stat + sys.

    CSV columns (after skipping # comments):
      y, y_LOW, y_HIGH, pT, pT_LOW, pT_HIGH, d2n/dydpT, stat+, stat-, sys+, sys-
    """
    rows = _skip_comments_lines(path)
    reader = csv.reader(rows)
    next(reader, None)  # skip header

    data: Dict[Tuple[float, float], Tuple[List, List, List, List]] = {}
    for row in reader:
        if len(row) < 11:
            continue
        try:
            y_lo  = float(row[1])
            y_hi  = float(row[2])
            pt    = float(row[3])
            val   = float(row[6])
            sp    = abs(float(row[7]))   # stat+
            sm    = abs(float(row[8]))   # stat-
            sysp  = abs(float(row[9]))   # sys+
            sysm  = abs(float(row[10]))  # sys-
        except (ValueError, IndexError):
            continue
        # total uncertainty = quadrature sum
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
    Build a per-event d^2n/dy dpT estimate for K0S from the per-event
    dn/dpT CSV produced by analyze_urqmd_kaons.

    The analyzer accumulates pT spectra for all particles with
    |y_analysis| < ycut (integrated over y).  To compare with a Fig7
    rapidity slice [y_lo, y_hi], we scale by the fraction of the total
    y-distribution that falls in that bin, estimated from y_distributions_dn.csv.
    We then divide by the rapidity bin width (y_hi - y_lo) to get d^2n/dydpT.

    Returns (pt_centers, d2n_dydpt, d2n_dydpt_err) or None if files missing.
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

    # Fraction of the rapidity distribution in the slice [y_lo, y_hi]
    dy = y_centers[1] - y_centers[0] if len(y_centers) > 1 else 1.0
    mask = (y_centers >= y_lo - 1e-6) & (y_centers < y_hi + 1e-6)
    if not mask.any():
        return None

    # Integrate dn/dy over the slice -> dn in that y-slice
    dn_in_slice     = np.sum(dn_dy[mask] * dy)
    dn_in_slice_err = math.sqrt(np.sum((dn_dy_err[mask] * dy) ** 2))

    # Total dn (integral over all y)
    dn_total     = np.sum(dn_dy * dy)
    if dn_total <= 0:
        return None

    # Fraction in slice
    frac     = dn_in_slice / dn_total
    frac_err = dn_in_slice_err / dn_total  # ignoring denominator uncertainty

    # d^2n/dy dpT = (dn/dpT * frac) / (y_hi - y_lo)
    dy_bin = y_hi - y_lo
    d2n = dn_dpt * frac / dy_bin
    # error propagation (treat frac_err as correlated over pT)
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
# Legend label helpers (used ONLY in overlay rap_overlay / pt_overlay)
# ---------------------------------------------------------------------------

def _urqmd_overlay_label(species_tex: str, modified: bool) -> str:
    prefix = r"UrQMD(3:1)" if modified else r"UrQMD"
    return rf"{prefix} {species_tex}"


def _urqmd_ratio_label(modified: bool) -> str:
    base = r"UrQMD(3:1)" if modified else r"UrQMD"
    return base + r" (bin stat. err.)"

# ---------------------------------------------------------------------------
# Boltzmann fit helpers for R(pT) experimental band
# ---------------------------------------------------------------------------

# Kaon mass in GeV/c^2 (PDG average of K+/K- and K0S, ~493.7 MeV)
_KAON_MASS = 0.4937


def _boltzmann(pt: np.ndarray, A: float, T: float) -> np.ndarray:
    """
    Boltzmann pT spectrum (Eq. 4 in NA61/SHINE Nature Comm. 2025):
        f(pT) = A * pT * exp( -sqrt(pT^2 + m_K^2) / T )
    A  -- normalisation [same units as data]
    T  -- inverse slope parameter [GeV/c]
    """
    return A * pt * np.exp(-np.sqrt(pt**2 + _KAON_MASS**2) / T)


def _fit_boltzmann(
    x: np.ndarray,
    y: np.ndarray,
    sigma: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Fit _boltzmann to (x, y±sigma).  Returns (popt, pcov) or (None, None)
    on failure.  Uses two initial guesses to improve convergence.
    """
    if not _SCIPY_OK or len(x) < 3:
        return None, None

    # estimate T from the point closest to pT = 0.5 GeV as a start
    T0_guess = 0.20
    A0_guess = float(np.max(y)) / (_boltzmann(np.array([0.3]), 1.0, T0_guess)[0])

    for p0 in ([A0_guess, T0_guess], [A0_guess * 2, 0.15], [A0_guess * 0.5, 0.30]):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, pcov = _scipy_curve_fit(
                    _boltzmann, x, y,
                    p0=p0,
                    sigma=sigma,
                    absolute_sigma=True,
                    bounds=([0.0, 0.05], [1e6, 1.0]),
                    maxfev=10000,
                )
            # sanity check: covariance must be finite and positive-definite
            if np.all(np.isfinite(pcov)) and pcov[0, 0] > 0 and pcov[1, 1] > 0:
                return popt, pcov
        except Exception:
            continue
    return None, None


def _ratio_band_from_fits(
    hep2a_x: np.ndarray,
    hep2a_y: np.ndarray,
    hep2a_ep: np.ndarray,
    hep2a_em: np.ndarray,
    hep2b_x: np.ndarray,
    hep2b_y: np.ndarray,
    hep2b_ep: np.ndarray,
    hep2b_em: np.ndarray,
    pt_fine: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Fit Boltzmann to K0S (hep2a) and (K++K-)/2 (hep2b) independently.
    Return (R_fit, R_sigma) evaluated on pt_fine, or (None, None) if fits fail.

    Analytical error propagation for R = f_kch / f_k0s:

        partial R / partial A_kch  =  pT * exp(-E/T_kch) / f_k0s
                                    =  R / A_kch
        partial R / partial T_kch  =  R * E / T_kch^2          (E = sqrt(pT^2+m^2))
        partial R / partial A_k0s  = -R / A_k0s
        partial R / partial T_k0s  = -R * E / T_k0s^2

    sigma_R^2 = (dR/dA_kch)^2 * Var(A_kch)
              + (dR/dT_kch)^2 * Var(T_kch)
              + 2*(dR/dA_kch)*(dR/dT_kch)*Cov(A_kch,T_kch)
              + (dR/dA_k0s)^2 * Var(A_k0s)
              + (dR/dT_k0s)^2 * Var(T_k0s)
              + 2*(dR/dA_k0s)*(dR/dT_k0s)*Cov(A_k0s,T_k0s)
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

    E = np.sqrt(pt_fine**2 + _KAON_MASS**2)   # transverse energy

    # partial derivatives
    dR_dA_kch  =  R_fit / popt_kch[0]
    dR_dT_kch  =  R_fit * E / popt_kch[1]**2
    dR_dA_k0s  = -R_fit / popt_k0s[0]
    dR_dT_k0s  = -R_fit * E / popt_k0s[1]**2

    var_R = (
        dR_dA_kch**2 * pcov_kch[0, 0]
        + dR_dT_kch**2 * pcov_kch[1, 1]
        + 2 * dR_dA_kch * dR_dT_kch * pcov_kch[0, 1]
        + dR_dA_k0s**2 * pcov_k0s[0, 0]
        + dR_dT_k0s**2 * pcov_k0s[1, 1]
        + 2 * dR_dA_k0s * dR_dT_k0s * pcov_k0s[0, 1]
    )
    R_sigma = np.where(var_R >= 0, np.sqrt(np.abs(var_R)), np.nan)

    return R_fit, R_sigma

# ---------------------------------------------------------------------------
# rap_overlay -- dN/dy UrQMD vs NA61/SHINE
# ---------------------------------------------------------------------------

def make_rap_overlay(
    urqmd_dir: Path,
    hep1a: Path,
    hep1b: Path,
    outpath: Path,
    modified: bool = False,
) -> None:
    """Single panel dN/dy: UrQMD lines vs NA61/SHINE points. No title."""
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
                label=r"NA61/SHINE $K^0_S$", **STYLE["exp_k0s"])
    ax.errorbar(hy_kch_x, hy_kch_y, yerr=[hy_kch_em, hy_kch_ep],
                label=r"NA61/SHINE $(K^+ {+} K^-)/2$", **STYLE["exp_kch"])

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
# pt_overlay -- dN/dpT (linear) + R(pT) UrQMD vs NA61/SHINE
# ---------------------------------------------------------------------------

def make_pt_overlay(
    urqmd_dir: Path,
    hep2a: Path,
    hep2b: Path,
    outpath: Path,
    modified: bool = False,
) -> None:
    """
    Two-panel: dN/dpT linear scale (top) + R(pT) (bottom).
    Both unmodified and modified use linear top panel.

    NA61/SHINE R(pT) is shown as a smooth shaded band derived from
    independent Boltzmann fits to each spectrum, propagated analytically.
    Falls back to bin-by-bin error bars if scipy is unavailable or a fit
    fails. UrQMD R(pT) uses raw bin-by-bin MC statistical errors (line only).
    """
    pt_data = load_urqmd_pt_spectra(urqmd_dir / "pt_spectra.csv")
    pt_r, R_urqmd, _ = load_urqmd_ratio_pt(urqmd_dir / "ratio_pt.csv")

    k0s_pt, k0s_val, _e     = pt_data.get("K0S",    (np.array([]),)*3)
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

    # --- Build R(pT) experimental band via Boltzmann fits -------------------
    pt_lo = max(min(hep2a_x.min(), hep2b_x.min()) - 0.05, 0.0)
    pt_hi = max(hep2a_x.max(), hep2b_x.max()) + 0.1
    pt_fine = np.linspace(pt_lo, pt_hi, 300)

    R_fit, R_sigma = _ratio_band_from_fits(
        hep2a_x, hep2a_y, hep2a_ep, hep2a_em,
        hep2b_x, hep2b_y, hep2b_ep, hep2b_em,
        pt_fine,
    )

    use_fit_band = (R_fit is not None)
    if not use_fit_band:
        print("  [warn] Boltzmann fit failed -- falling back to bin-by-bin R(pT) errors")
        kch_on_k0s = np.interp(hep2a_x, hep2b_x, hep2b_y)
        kch_ep_i   = np.interp(hep2a_x, hep2b_x, hep2b_ep)
        kch_em_i   = np.interp(hep2a_x, hep2b_x, hep2b_em)
        with np.errstate(invalid="ignore", divide="ignore"):
            R_exp_fb = np.where(hep2a_y > 0, kch_on_k0s / hep2a_y, np.nan)
            R_exp_err_fb = np.abs(R_exp_fb) * np.sqrt(
                ((0.5*(kch_ep_i + kch_em_i)) / np.where(kch_on_k0s > 0, kch_on_k0s, 1.0))**2 +
                ((0.5*(hep2a_ep + hep2a_em)) / np.where(hep2a_y > 0, hep2a_y, 1.0))**2
            )

    # --- Figure layout -------------------------------------------------------
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(6.5, 7.5),
        gridspec_kw={"height_ratios": [2.5, 1], "hspace": 0.05},
        sharex=True,
    )

    # Top panel: dN/dpT spectra
    if len(k0s_val) > 0:
        ax_top.plot(k0s_pt, k0s_val,
                    label=_urqmd_overlay_label(r"$K^0_S$", modified),
                    **STYLE["urqmd_k0s"])
    if len(kch_val) > 0:
        ax_top.plot(kch_pt, kch_val,
                    label=_urqmd_overlay_label(r"$(K^+ {+} K^-)/2$", modified),
                    **STYLE["urqmd_kch"])
    ax_top.errorbar(hep2a_x, hep2a_y, yerr=[hep2a_em, hep2a_ep],
                    label=r"NA61/SHINE $K^0_S$", **STYLE["exp_k0s"])
    ax_top.errorbar(hep2b_x, hep2b_y, yerr=[hep2b_em, hep2b_ep],
                    label=r"NA61/SHINE $(K^+ {+} K^-)/2$", **STYLE["exp_kch"])

    ax_top.yaxis.set_minor_locator(AutoMinorLocator())
    ax_top.set_ylabel(r"$dN/dp_T\;[(\mathrm{GeV}/c)^{-1}]$")
    ax_top.legend(loc="upper right")
    ax_top.tick_params(labelbottom=False)

    # Bottom panel: R(pT)
    ax_bot.axhline(1.0, ls=":", lw=0.9, color="black")

    # UrQMD ratio -- bin-by-bin stat errors, shown as a line
    finite_u = np.isfinite(R_urqmd)
    if finite_u.any():
        ax_bot.plot(pt_r[finite_u], R_urqmd[finite_u],
                    label=_urqmd_ratio_label(modified),
                    **STYLE["ratio_urqmd"])

    # NA61/SHINE ratio -- shaded band only, plain label, no center line
    if use_fit_band:
        finite_b = np.isfinite(R_fit) & np.isfinite(R_sigma)
        if finite_b.any():
            ax_bot.fill_between(
                pt_fine[finite_b],
                (R_fit - R_sigma)[finite_b],
                (R_fit + R_sigma)[finite_b],
                alpha=0.35, color="black",
                label=r"NA61/SHINE",
            )
            # (no center line -- it was the spurious second black curve)
    else:
        finite_e = np.isfinite(R_exp_fb)
        if finite_e.any():
            ax_bot.errorbar(hep2a_x[finite_e], R_exp_fb[finite_e],
                            yerr=R_exp_err_fb[finite_e],
                            label=r"NA61/SHINE",
                            **STYLE["ratio_exp"])

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
# PAN-style R_K(y) -- isospin ratio
# ---------------------------------------------------------------------------

def make_pan_ratio_y(urqmd_dir: Path, outpath: Path) -> None:
    """PAN-style R_K(y) = 0.5*(K++K-)/K0S vs y. No title. No error bars."""
    yc, Rk, _Rk_e = load_urqmd_ratio_y(urqmd_dir / "ratio_y.csv")
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    finite = np.isfinite(Rk)
    ax.axhline(1.0, ls=":", lw=0.9, color="black")
    if finite.any():
        style = dict(STYLE["ratio_y"])
        ax.plot(yc[finite], Rk[finite],
                label=r"$R_K(y)$", **style)
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
# Fig7-style: K0S d^2n/dy dpT vs pT in rapidity slices
# Overlay: unmod UrQMD + mod UrQMD + NA61/SHINE exp data
# ---------------------------------------------------------------------------

def make_fig7_k0s_2d(
    unmod_dir: Path,
    mod_dir: Path,
    hep7_path: Path,
    outpath: Path,
) -> None:
    """
    Fig7-style multi-panel plot: K0S d^2n/dy dpT vs pT, one panel per
    rapidity bin from Figure7.csv.

    Each panel shows:
      - NA61/SHINE exp data points with total (stat ⊕ sys) uncertainties
      - UrQMD unmod (solid line)
      - UrQMD mod   (dashed line)

    The rapidity bin label is shown inside each panel.
    The figure matches the layout of Fig.7 in the NA61/SHINE paper
    (Methods Extended data section).
    """
    if not hep7_path.exists():
        print(f"  [warn] {hep7_path} not found -- skipping Fig7 plot")
        return

    exp_data = load_figure7(hep7_path)
    if not exp_data:
        print(f"  [warn] No data loaded from {hep7_path} -- skipping Fig7 plot")
        return

    # Sort rapidity bins by their lower edge
    y_bins = sorted(exp_data.keys(), key=lambda k: k[0])
    n_bins = len(y_bins)

    # Determine subplot grid: fill row-by-row
    n_cols = min(3, n_bins)
    n_rows = math.ceil(n_bins / n_cols)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 4.0, n_rows * 3.5),
        squeeze=False,
    )

    for idx, (y_lo, y_hi) in enumerate(y_bins):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row][col]

        pt_exp, val_exp, ep_exp, em_exp = exp_data[(y_lo, y_hi)]
        total_err_up = ep_exp
        total_err_dn = em_exp

        # Experimental data
        ax.errorbar(
            pt_exp, val_exp,
            yerr=[total_err_dn, total_err_up],
            label=r"NA61/SHINE",
            **STYLE["fig7_exp"],
        )

        # UrQMD unmod
        res_unmod = _load_urqmd_k0s_pt_in_ybin(unmod_dir, y_lo, y_hi, use_dn=True)
        if res_unmod is not None:
            pt_u, d2n_u, _ = res_unmod
            finite_u = np.isfinite(d2n_u) & (d2n_u > 0)
            if finite_u.any():
                ax.plot(
                    pt_u[finite_u], d2n_u[finite_u],
                    label=r"UrQMD",
                    **STYLE["fig7_urqmd_unmod"],
                )

        # UrQMD mod
        res_mod = _load_urqmd_k0s_pt_in_ybin(mod_dir, y_lo, y_hi, use_dn=True)
        if res_mod is not None:
            pt_m, d2n_m, _ = res_mod
            finite_m = np.isfinite(d2n_m) & (d2n_m > 0)
            if finite_m.any():
                ax.plot(
                    pt_m[finite_m], d2n_m[finite_m],
                    label=r"UrQMD(3:1)",
                    **STYLE["fig7_urqmd_mod"],
                )

        # y-bin label inside panel
        y_label = rf"${y_lo:.1f} < y < {y_hi:.1f}$"
        ax.text(0.97, 0.95, y_label, transform=ax.transAxes,
                ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))

        ax.set_yscale("log")
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())

        # Only show x-label on bottom row, y-label on left column
        if row == n_rows - 1:
            ax.set_xlabel(r"$p_T\;[\mathrm{GeV}/c]$")
        if col == 0:
            ax.set_ylabel(r"$d^2n/dy\,dp_T\;[(\mathrm{GeV}/c)^{-1}]$")

        if idx == 0:
            ax.legend(loc="upper right", fontsize=8)

    # Hide unused axes
    for idx in range(n_bins, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row][col].set_visible(False)

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

    hep7 = hep_dir / sys_def.hep7 if sys_def.hep7 else None

    if not hep_ok:
        for p in (hep1a, hep1b, hep2a, hep2b):
            if not p.exists():
                print(f"  [warn] HepData file not found: {p}")
        if not skip_missing_hep:
            return

    print(f"\n=== {sys_def.label} @ {sys_def.energy_str} ===")

    if hep_ok:
        print("  Rapidity overlay (dN/dy UrQMD vs NA61/SHINE)...")
        make_rap_overlay(unmod_dir, hep1a, hep1b,
                         sys_def.out(outdir, "rap_overlay", mod=False), modified=False)
        make_rap_overlay(mod_dir,   hep1a, hep1b,
                         sys_def.out(outdir, "rap_overlay", mod=True),  modified=True)

        print("  pT overlay (dN/dpT linear + R(pT) UrQMD vs NA61/SHINE)...")
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
    make_pan_ratio_y(unmod_dir,    sys_def.out(outdir, "pan_ratio_y",    mod=False))
    make_pan_ratio_y(mod_dir,      sys_def.out(outdir, "pan_ratio_y",    mod=True))

    if hep7 is not None and hep7.exists():
        print("  Fig7-style K0S d^2n/dydpT (unmod+mod UrQMD vs NA61/SHINE)...")
        make_fig7_k0s_2d(
            unmod_dir, mod_dir, hep7,
            sys_def.out(outdir, "fig7_k0s_2d", mod=False),
        )
    elif hep7 is not None:
        print(f"  [warn] Fig7 HepData file not found: {hep7} -- skipping Fig7 plot")

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
