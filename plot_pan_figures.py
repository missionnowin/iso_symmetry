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
  ArSc_11p9GeV_pan_y_species_unmod.{eps,png}
  ArSc_11p9GeV_pan_y_species_mod.{eps,png}
  ArSc_11p9GeV_pan_pt_species_unmod.{eps,png}
  ArSc_11p9GeV_pan_pt_species_mod.{eps,png}
  ArSc_11p9GeV_pan_ratio_y_unmod.{eps,png}
  ArSc_11p9GeV_pan_ratio_y_mod.{eps,png}

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
# PAN-style UrQMD-only plots
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

    print("  PAN-style UrQMD-only figures...")
    make_pan_y_species(unmod_dir,  sys_def.out(outdir, "pan_y_species",  mod=False))
    make_pan_y_species(mod_dir,    sys_def.out(outdir, "pan_y_species",  mod=True))
    make_pan_pt_species(unmod_dir, sys_def.out(outdir, "pan_pt_species", mod=False))
    make_pan_pt_species(mod_dir,   sys_def.out(outdir, "pan_pt_species", mod=True))
    make_pan_ratio_y(unmod_dir,    sys_def.out(outdir, "pan_ratio_y",    mod=False))
    make_pan_ratio_y(mod_dir,      sys_def.out(outdir, "pan_ratio_y",    mod=True))

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
