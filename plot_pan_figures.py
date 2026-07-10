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
  ArSc_11p9GeV_fig1_unmod.{eps,png}
  ArSc_11p9GeV_fig1_mod.{eps,png}
  ArSc_11p9GeV_fig2_unmod.{eps,png}
  ArSc_11p9GeV_fig2_mod.{eps,png}
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
* UrQMD vs UrQMD(3:1) labels appear only in overlay figs (fig1/fig2)
  where UrQMD curves are directly compared against NA61/SHINE data.
* PAN-only UrQMD plots show only species labels (K+, K-, K0S).
* Grayscale / black-and-white only.
* Curves distinguished by linestyle + marker shape.
* EPS vector output suitable for Yadernaya Fizika submission.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

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
        e.g. pan_figures/ArSc_11p9GeV_fig1_mod
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
# Legend label helpers (used ONLY in overlay fig1/fig2)
# ---------------------------------------------------------------------------

def _urqmd_overlay_label(species_tex: str, modified: bool) -> str:
    """
    Legend entry for UrQMD curves in overlay (NA61 comparison) figures.
    modified=True  -> "UrQMD(3:1) $K^0_S$"
    modified=False -> "UrQMD $K^0_S$"
    """
    prefix = r"UrQMD(3:1)" if modified else r"UrQMD"
    return rf"{prefix} {species_tex}"


def _urqmd_ratio_label(modified: bool) -> str:
    return r"UrQMD(3:1)" if modified else r"UrQMD"

# ---------------------------------------------------------------------------
# Article-style Figure 1 -- rapidity overlays (UrQMD vs NA61/SHINE)
# ---------------------------------------------------------------------------

def make_fig1(
    urqmd_dir: Path,
    hep1a: Path,
    hep1b: Path,
    outpath: Path,
    modified: bool = False,
) -> None:
    """
    Single panel: dN/dy for K0S and (K++K-)/2.
    UrQMD as lines labelled UrQMD / UrQMD(3:1);
    NA61/SHINE as points with error bars.
    No in-plot title -- system/energy go in LaTeX caption.
    """
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
# Article-style Figure 2 -- pT overlays + ratio (UrQMD vs NA61/SHINE)
# ---------------------------------------------------------------------------

def make_fig2(
    urqmd_dir: Path,
    hep2a: Path,
    hep2b: Path,
    outpath: Path,
    modified: bool = False,
    log_top: bool = True,
) -> None:
    """
    Two-panel: dN/dpT (top) + R(pT) (bottom).
    log_top=True  -> log y on top panel  (unmodified)
    log_top=False -> linear y on top panel (modified, matching paper)
    UrQMD(3:1) label only when modified=True.
    No in-plot title.
    """
    pt_data = load_urqmd_pt_spectra(urqmd_dir / "pt_spectra.csv")
    pt_r, R_urqmd, _ = load_urqmd_ratio_pt(urqmd_dir / "ratio_pt.csv")

    k0s_pt, k0s_val, _e  = pt_data.get("K0S",    (np.array([]),)*3)
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

    kch_on_k0s = np.interp(hep2a_x, hep2b_x, hep2b_y)
    kch_ep_i   = np.interp(hep2a_x, hep2b_x, hep2b_ep)
    kch_em_i   = np.interp(hep2a_x, hep2b_x, hep2b_em)
    with np.errstate(invalid="ignore", divide="ignore"):
        R_exp = np.where(hep2a_y > 0, kch_on_k0s / hep2a_y, np.nan)
        R_exp_err = np.abs(R_exp) * np.sqrt(
            ((0.5*(kch_ep_i + kch_em_i)) / np.where(kch_on_k0s > 0, kch_on_k0s, 1.0))**2 +
            ((0.5*(hep2a_ep + hep2a_em)) / np.where(hep2a_y > 0, hep2a_y, 1.0))**2
        )

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
                    label=r"NA61/SHINE $K^0_S$", **STYLE["exp_k0s"])
    ax_top.errorbar(hep2b_x, hep2b_y, yerr=[hep2b_em, hep2b_ep],
                    label=r"NA61/SHINE $(K^+ {+} K^-)/2$", **STYLE["exp_kch"])

    ax_top.set_yscale("log" if log_top else "linear")
    ax_top.yaxis.set_minor_locator(AutoMinorLocator())
    ax_top.set_ylabel(r"$dN/dp_T\;[(\mathrm{GeV}/c)^{-1}]$")
    ax_top.legend(loc="upper right")
    ax_top.tick_params(labelbottom=False)

    finite_u = np.isfinite(R_urqmd)
    finite_e = np.isfinite(R_exp)
    ax_bot.axhline(1.0, ls=":", lw=0.9, color="black")
    if finite_u.any():
        ax_bot.plot(pt_r[finite_u], R_urqmd[finite_u],
                    label=_urqmd_ratio_label(modified), **STYLE["ratio_urqmd"])
    if finite_e.any():
        ax_bot.errorbar(hep2a_x[finite_e], R_exp[finite_e],
                        yerr=R_exp_err[finite_e],
                        label="NA61/SHINE", **STYLE["ratio_exp"])

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
# PAN-style UrQMD-only plots -- species labels only, no UrQMD prefix
# ---------------------------------------------------------------------------

def make_pan_y_species(
    urqmd_dir: Path,
    outpath: Path,
) -> None:
    """PAN-style dN/dy for K+, K-, K0S. No title, no UrQMD label prefix."""
    data = load_urqmd_y_distributions(urqmd_dir / "y_distributions.csv")

    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    for sp in ("Kplus", "Kminus", "K0S"):
        if sp not in data:
            continue
        yc, val, err = data[sp]
        ax.errorbar(yc, val, yerr=err,
                    label=SPECIES_LABEL.get(sp, sp),
                    **STYLE.get(sp, {}))

    ax.set_xlabel(r"$y$")
    ax.set_ylabel(r"$dN/dy$")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.legend(loc="upper right")
    fig.tight_layout()

    _save(fig, outpath)
    plt.close(fig)
    print(f"  wrote {outpath.with_suffix('.eps')}")


def make_pan_pt_species(
    urqmd_dir: Path,
    outpath: Path,
) -> None:
    """PAN-style dN/dpT (log) for K+, K-, K0S. No title."""
    data = load_urqmd_pt_spectra(urqmd_dir / "pt_spectra.csv")

    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    for sp in ("Kplus", "Kminus", "K0S"):
        if sp not in data:
            continue
        pt, val, err = data[sp]
        ax.errorbar(pt, val, yerr=err,
                    label=SPECIES_LABEL.get(sp, sp),
                    **STYLE.get(sp, {}))

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


def make_pan_ratio_y(
    urqmd_dir: Path,
    outpath: Path,
) -> None:
    """PAN-style R_K(y) = 0.5*(K++K-)/K0S vs y. No title."""
    yc, Rk, Rk_e = load_urqmd_ratio_y(urqmd_dir / "ratio_y.csv")

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    finite = np.isfinite(Rk)
    ax.axhline(1.0, ls=":", lw=0.9, color="black")
    if finite.any():
        ax.errorbar(yc[finite], Rk[finite], yerr=Rk_e[finite],
                    label=r"$R_K(y)$", **STYLE["ratio_y"])

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
        print("  Fig 1 (rapidity overlays)...")
        make_fig1(unmod_dir, hep1a, hep1b,
                  sys_def.out(outdir, "fig1", mod=False), modified=False)
        make_fig1(mod_dir,   hep1a, hep1b,
                  sys_def.out(outdir, "fig1", mod=True),  modified=True)

        print("  Fig 2 (pT overlays + ratio)...")
        make_fig2(unmod_dir, hep2a, hep2b,
                  sys_def.out(outdir, "fig2", mod=False), modified=False, log_top=True)
        make_fig2(mod_dir,   hep2a, hep2b,
                  sys_def.out(outdir, "fig2", mod=True),  modified=True,  log_top=False)

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
