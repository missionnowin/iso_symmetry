"""
plot_pan_figures.py
===================
Standalone script to produce publication-ready figures for
Physics of Atomic Nuclei (PAN / Ядерная физика).

Generates 4 figures for Ar+Sc @ sqrt(s_NN) = 11.9 GeV:

  fig1_unmodified.eps  -- rapidity spectra: UrQMD (unmodified) + NA61/SHINE
  fig1_modified.eps    -- rapidity spectra: UrQMD (modified u/d) + NA61/SHINE
  fig2_unmodified.eps  -- pT spectra + ratio: UrQMD (unmodified) + NA61/SHINE
  fig2_modified.eps    -- pT spectra + ratio: UrQMD (modified u/d) + NA61/SHINE

Journal requirements
--------------------
* Black-and-white only (no colour); curves distinguished by linestyle + marker
* No info-boxes, no Nevt / system text on canvas
* Format: EPS (Encapsulated PostScript)
* Axes labels in LaTeX math mode

Directory layout expected (relative to this script)
----------------------------------------------------
  ArSc-kaons-11.9GeV-ecm-collision/results/
      y_distributions_meancharged.csv
      pt_spectra.csv
      ratio_pt.csv

  ArSc-kaons-11.9GeV-ecm-collision-modified-ud/results/
      (same files)

  HepData CSVs (from NA61/SHINE Nature Comm. article, HEPData record
  10.17182/hepdata.156978.v2):
      Figure1a.csv   -- K0S  dN/dy
      Figure1b.csv   -- (K+ + K-)/2  dN/dy
      Figure2a.csv   -- K0S  dN/dpT
      Figure2b.csv   -- (K+ + K-)/2  dN/dpT

Usage
-----
  python plot_pan_figures.py [--outdir OUTPUT_DIR]

  Optional flags:
    --outdir DIR        where to write EPS files (default: pan_figures)
    --hepdata-dir DIR   directory that contains Figure1a/1b/2a/2b.csv
                        (default: same directory as this script)
    --urqmd-base DIR    parent of the two ArSc result directories
                        (default: same directory as this script)
    --no-eps            skip EPS, write PNG only (useful if LaTeX not installed)
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

# ---------------------------------------------------------------------------
# Global style – strict black & white, suitable for print
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "text.usetex": False,          # set True if LaTeX is available on the system
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
# Marker / linestyle palette  (b/w only, 4 distinct series per plot)
# ---------------------------------------------------------------------------
#   series 0: K0S   UrQMD   -- solid line,  no marker
#   series 1: Kch   UrQMD   -- dashed line, no marker
#   series 2: K0S   NA61    -- filled circle,  no line
#   series 3: Kch   NA61    -- open square,    no line

STYLE = {
    "urqmd_k0s":  dict(ls="-",  lw=1.4, marker="None", color="black"),
    "urqmd_kch":  dict(ls="--", lw=1.4, marker="None", color="black"),
    "exp_k0s":    dict(ls="None", marker="o", ms=5, mfc="black",  mec="black",  color="black",  capsize=3, elinewidth=0.8),
    "exp_kch":    dict(ls="None", marker="s", ms=5, mfc="white",  mec="black",  color="black",  capsize=3, elinewidth=0.8),
    "ratio_urqmd": dict(ls="-",  lw=1.4, marker="None", color="black"),
    "ratio_exp":   dict(ls="None", marker="^", ms=5, mfc="black", mec="black", color="black", capsize=3, elinewidth=0.8),
}

# ---------------------------------------------------------------------------
# CSV loaders
# ---------------------------------------------------------------------------

def _skip_comments(path: Path) -> List[List[str]]:
    rows = []
    with open(path, newline="") as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("#") or not stripped:
                continue
            rows.append(stripped)
    return rows


def load_hepdata(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (x, y, err_up, err_dn) from a HepData-style CSV."""
    rows = _skip_comments(path)
    reader = csv.reader(rows)
    header = next(reader)
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
    """
    Load y_distributions_meancharged.csv produced by analyze_urqmd_kaons.py.
    Columns: y_lo, y_hi, y_center, KmeanCharged_value, KmeanCharged_err,
             TwoK0S_value, TwoK0S_err
    Returns (y_centers, kch_vals, kch_err, k0s_vals, k0s_err)
    """
    rows = _skip_comments(path)
    reader = csv.reader(rows)
    next(reader)  # header
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


def load_urqmd_pt_spectra(path: Path):
    """
    Load pt_spectra.csv.
    Columns: species, pt_lo, pt_hi, pt_center, counts, value, error
    Returns dict species -> (pt_centers, values, errors)
    """
    rows = _skip_comments(path)
    reader = csv.reader(rows)
    next(reader)
    data: dict = {}
    for row in reader:
        if len(row) < 7:
            continue
        sp = row[0].strip()
        pt = float(row[3])
        val = float(row[5])
        err = float(row[6])
        data.setdefault(sp, ([], [], []))
        data[sp][0].append(pt)
        data[sp][1].append(val)
        data[sp][2].append(err)
    return {sp: (np.array(v[0]), np.array(v[1]), np.array(v[2])) for sp, v in data.items()}


def load_urqmd_ratio_pt(path: Path):
    """
    Load ratio_pt.csv.
    Columns: pt_lo, pt_hi, pt_center, K+, K-, K0S, R, R_err
    Returns (pt_centers, R, R_err) — NaN where undefined.
    """
    rows = _skip_comments(path)
    reader = csv.reader(rows)
    next(reader)
    pts, Rs, Res = [], [], []
    for row in reader:
        if len(row) < 8:
            continue
        pts.append(float(row[2]))
        Rs.append(float(row[6]) if row[6] else math.nan)
        Res.append(float(row[7]) if row[7] else math.nan)
    return np.array(pts), np.array(Rs), np.array(Res)


# ---------------------------------------------------------------------------
# Figure 1 – rapidity spectra
# ---------------------------------------------------------------------------

def make_fig1(
    urqmd_dir: Path,
    hep1a: Path,
    hep1b: Path,
    outpath: Path,
    title_tag: str,
) -> None:
    """
    Single panel: dN/dy for K0S and (K++K-)/2.
    UrQMD as lines; NA61/SHINE as points with error bars.
    """
    yc, kch, kch_e, k0s_twox, k0s_twox_e = load_urqmd_ydist_meancharged(
        urqmd_dir / "y_distributions_meancharged.csv"
    )
    # y_distributions_meancharged stores 2*K0S; we want K0S = half of that
    k0s = k0s_twox / 2.0
    k0s_e = k0s_twox_e / 2.0

    hy_k0s_x, hy_k0s_y, hy_k0s_ep, hy_k0s_em = load_hepdata(hep1a)
    hy_kch_x, hy_kch_y, hy_kch_ep, hy_kch_em = load_hepdata(hep1b)

    fig, ax = plt.subplots(figsize=(6.5, 5.0))

    ax.plot(yc, k0s, label=r"UrQMD $K^0_S$", **STYLE["urqmd_k0s"])
    ax.plot(yc, kch, label=r"UrQMD $(K^++K^-)/2$", **STYLE["urqmd_kch"])
    ax.errorbar(hy_k0s_x, hy_k0s_y,
                yerr=[hy_k0s_em, hy_k0s_ep],
                label=r"NA61/SHINE $K^0_S$",
                **STYLE["exp_k0s"])
    ax.errorbar(hy_kch_x, hy_kch_y,
                yerr=[hy_kch_em, hy_kch_ep],
                label=r"NA61/SHINE $(K^++K^-)/2$",
                **STYLE["exp_kch"])

    ax.set_xlabel(r"$y$")
    ax.set_ylabel(r"$dN/dy$")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.legend(loc="upper right")
    fig.tight_layout()

    _save(fig, outpath)
    plt.close(fig)
    print(f"  wrote {outpath}")


# ---------------------------------------------------------------------------
# Figure 2 – pT spectra + ratio panel
# ---------------------------------------------------------------------------

def make_fig2(
    urqmd_dir: Path,
    hep2a: Path,
    hep2b: Path,
    outpath: Path,
    title_tag: str,
) -> None:
    """
    Two-panel figure:
      top:    dN/dpT for K0S and (K++K-)/2  (log y-scale)
      bottom: R(pT) = (K++K-)/2 / K0S
    """
    pt_data = load_urqmd_pt_spectra(urqmd_dir / "pt_spectra.csv")
    pt_r, R_urqmd, R_urqmd_e = load_urqmd_ratio_pt(urqmd_dir / "ratio_pt.csv")

    k0s_pt, k0s_val, k0s_err = pt_data.get("K0S", (np.array([]), np.array([]), np.array([])))
    kp_pt,  kp_val,  kp_err  = pt_data.get("Kplus",  (np.array([]), np.array([]), np.array([])))
    km_pt,  km_val,  km_err  = pt_data.get("Kminus", (np.array([]), np.array([]), np.array([])))

    # compute (K++K-)/2 from UrQMD pt_spectra
    if len(kp_val) == len(km_val) and len(kp_val) > 0:
        kch_val = 0.5 * (kp_val + km_val)
        kch_err = 0.5 * np.sqrt(kp_err**2 + km_err**2)
        kch_pt  = kp_pt
    else:
        kch_val = np.array([])
        kch_err = np.array([])
        kch_pt  = np.array([])

    hep2a_x, hep2a_y, hep2a_ep, hep2a_em = load_hepdata(hep2a)  # K0S
    hep2b_x, hep2b_y, hep2b_ep, hep2b_em = load_hepdata(hep2b)  # (K++K-)/2

    # ratio from HepData
    # interpolate on common pT grid = hep2a_x (K0S has fewer points)
    from numpy import interp
    kch_on_k0s_grid = interp(hep2a_x, hep2b_x, hep2b_y)
    kch_ep_interp   = interp(hep2a_x, hep2b_x, hep2b_ep)
    kch_em_interp   = interp(hep2a_x, hep2b_x, hep2b_em)
    with np.errstate(invalid="ignore", divide="ignore"):
        R_exp = np.where(hep2a_y > 0, kch_on_k0s_grid / hep2a_y, np.nan)
        R_exp_err = np.abs(R_exp) * np.sqrt(
            ((0.5*(kch_ep_interp + kch_em_interp)) / np.where(kch_on_k0s_grid > 0, kch_on_k0s_grid, 1.0))**2 +
            ((0.5*(hep2a_ep + hep2a_em)) / np.where(hep2a_y > 0, hep2a_y, 1.0))**2
        )

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(6.5, 7.5),
        gridspec_kw={"height_ratios": [2.5, 1], "hspace": 0.05},
        sharex=True,
    )

    # -- top panel
    if len(k0s_val) > 0:
        ax_top.plot(k0s_pt, k0s_val, label=r"UrQMD $K^0_S$", **STYLE["urqmd_k0s"])
    if len(kch_val) > 0:
        ax_top.plot(kch_pt, kch_val, label=r"UrQMD $(K^++K^-)/2$", **STYLE["urqmd_kch"])
    ax_top.errorbar(hep2a_x, hep2a_y,
                    yerr=[hep2a_em, hep2a_ep],
                    label=r"NA61/SHINE $K^0_S$",
                    **STYLE["exp_k0s"])
    ax_top.errorbar(hep2b_x, hep2b_y,
                    yerr=[hep2b_em, hep2b_ep],
                    label=r"NA61/SHINE $(K^++K^-)/2$",
                    **STYLE["exp_kch"])
    ax_top.set_yscale("log")
    ax_top.set_ylabel(r"$dN/dp_T\;[(\mathrm{GeV}/c)^{-1}]$")
    ax_top.yaxis.set_minor_locator(AutoMinorLocator())
    ax_top.legend(loc="upper right")
    ax_top.tick_params(labelbottom=False)

    # -- bottom panel: ratio
    finite_u = np.isfinite(R_urqmd)
    finite_e = np.isfinite(R_exp)
    ax_bot.axhline(1.0, ls=":", lw=0.9, color="black")
    if finite_u.any():
        ax_bot.plot(pt_r[finite_u], R_urqmd[finite_u],
                    label="UrQMD", **STYLE["ratio_urqmd"])
    if finite_e.any():
        ax_bot.errorbar(hep2a_x[finite_e], R_exp[finite_e],
                        yerr=R_exp_err[finite_e],
                        label="NA61/SHINE",
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
    print(f"  wrote {outpath}")


# ---------------------------------------------------------------------------
# Save helper – EPS + PNG
# ---------------------------------------------------------------------------

def _save(fig, base_path: Path) -> None:
    eps_path = base_path.with_suffix(".eps")
    png_path = base_path.with_suffix(".png")
    fig.savefig(eps_path, format="eps", bbox_inches="tight")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Generate PAN journal figures for Ar+Sc kaon spectra.")
    p.add_argument("--outdir",      default="pan_figures",  help="Output directory for EPS/PNG files")
    p.add_argument("--hepdata-dir", default=".",            help="Directory containing Figure1a/1b/2a/2b.csv")
    p.add_argument("--urqmd-base",  default=".",            help="Parent directory of the ArSc result folders")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    outdir    = Path(args.outdir)
    hep_dir   = Path(args.hepdata_dir)
    urqmd_base = Path(args.urqmd_base)

    outdir.mkdir(parents=True, exist_ok=True)

    unmod_dir = urqmd_base / "ArSc-kaons-11.9GeV-ecm-collision"        / "results"
    mod_dir   = urqmd_base / "ArSc-kaons-11.9GeV-ecm-collision-modified-ud" / "results"

    hep1a = hep_dir / "Figure1a.csv"
    hep1b = hep_dir / "Figure1b.csv"
    hep2a = hep_dir / "Figure2a.csv"
    hep2b = hep_dir / "Figure2b.csv"

    for fpath in (hep1a, hep1b, hep2a, hep2b):
        if not fpath.exists():
            print(f"[error] HepData file not found: {fpath}", file=sys.stderr)
            return 1
    for d in (unmod_dir, mod_dir):
        if not d.exists():
            print(f"[error] UrQMD results directory not found: {d}", file=sys.stderr)
            return 1

    print("Generating Fig. 1 (rapidity spectra)...")
    make_fig1(unmod_dir, hep1a, hep1b, outdir / "fig1_unmodified", title_tag="unmodified")
    make_fig1(mod_dir,   hep1a, hep1b, outdir / "fig1_modified",   title_tag="modified u/d")

    print("Generating Fig. 2 (pT spectra + ratio)...")
    make_fig2(unmod_dir, hep2a, hep2b, outdir / "fig2_unmodified", title_tag="unmodified")
    make_fig2(mod_dir,   hep2a, hep2b, outdir / "fig2_modified",   title_tag="modified u/d")

    print(f"\nDone. Files written to: {outdir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
