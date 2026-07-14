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
  ArSc_11p9GeV_dn_rap_overlay_ratio_y_unmod.{eps,png} -- TWO-PANEL: dn/dy overlay (top) + R_K(y) both (bottom)
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
Two-panel combined figure WITH experimental data in BOTH panels:
  top panel:    dn/dy vs y -- UrQMD K0S + (K++K-)/2 overlaid with NA61/SHINE
  bottom panel: R_K(y) = (K++K-)/2 / K0S vs y
                -- UrQMD line (from ratio_y.csv)
                -- NA61/SHINE bin-by-bin ratio computed from Figure1b/Figure1a,
                   interpolating (K++K-)/2 onto the K0S y-grid, with
                   quadrature-propagated error bars
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
    # experimental R_K(y) in rapidity-ratio panel
    "ratio_exp_y": dict(ls="None", marker="^", ms=5,
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


def load_