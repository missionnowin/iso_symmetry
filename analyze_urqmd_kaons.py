"""
analyze_urqmd_kaons.py


Standalone analyzer for UrQMD .f19 / OSCAR-like output. It computes kaon
yields, pT spectra, rapidity distributions, and


    R(pT) = 0.5 * (K+ + K-) / K0_S


inside a central rapidity slice |y_analysis| < ycut.


The script is intended for comparing several collision systems and energy
setups, including fixed-target and collider / CM-frame UrQMD outputs.


Key frame convention
--------------------
The script separates two ideas:


  1. --input-frame:
       Which frame UrQMD used when writing particle four-momenta.
       Allowed values are projectile, target, cm.


  2. --collision-mode:
       Which collision geometry is being represented.
       Allowed values are fixed-target, collider.


The analysis rapidity is always the rapidity relative to the NN
center-of-mass frame:


    y_analysis = y_input - y_cm_in_input_frame


For a fixed-target experiment, the target frame is the usual laboratory
frame. For a symmetric collider in its usual laboratory frame, the lab frame
is the NN center-of-mass frame, so use --input-frame cm.


Examples
--------
Fixed-target Xe124+W184 at sqrt(s_NN)=2.9 GeV, output in target/lab frame:


  python analyze_urqmd_kaons.py XeW_2p9.f19 -o out_XeW_2p9 \\
      --system "Xe124+W184" --beam-label "sqrt(s_NN)=2.9 GeV, fixed target" \\
      --collision-mode fixed-target --input-frame target --ecm-snn 2.9


Collider / CM-frame Xe124+Xe124 at sqrt(s_NN)=9.2 GeV:


  python analyze_urqmd_kaons.py XeXe_9p2.f19 -o out_XeXe_9p2 \\
      --system "Xe124+Xe124" --beam-label "sqrt(s_NN)=9.2 GeV, collider" \\
      --collision-mode collider --input-frame cm


Outputs
-------
In --outdir, default ./urqmd_kaon_out:


  pt_spectra.csv
  y_distributions.csv
  ratio_pt.csv
  summary.csv
  mean_kaon_y.csv
  pt_spectra.png
  y_distributions.png
  ratio_pt.png
  mean_kaon_y.png
"""


from __future__ import annotations


import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


import numpy as np


import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt



# ---------------------------------------------------------------------------
# Particle container & PDG codes
# ---------------------------------------------------------------------------


PDG_KPLUS = 321
PDG_KMINUS = -321
PDG_K0S = 310
PDG_K0 = 311
PDG_K0BAR = -311



@dataclass
class Particle:
    pdg: int
    px: float
    py: float
    pz: float
    E: float


    def pt(self) -> float:
        return math.hypot(self.px, self.py)


    def rapidity(self) -> float:
        """Return momentum rapidity y = 0.5 ln((E+pz)/(E-pz))."""
        denom = self.E - self.pz
        numer = self.E + self.pz
        if denom <= 0.0 or numer <= 0.0:
            return float("nan")
        return 0.5 * math.log(numer / denom)



# ---------------------------------------------------------------------------
# Tolerant OSCAR / f19 reader
# ---------------------------------------------------------------------------



def _try_parse_event_header(tokens: List[str]) -> Optional[int]:
    """Recognise an event header line and return n_particles, or None."""
    if not (2 <= len(tokens) <= 6):
        return None
    try:
        evt = int(tokens[0])
        n = int(tokens[1])
    except ValueError:
        return None
    if n < 0 or n > 100000 or evt < 0:
        return None
    return n



def _parse_particle_line(tokens: List[str], layout: str) -> Optional[Particle]:
    """Parse a particle line under the requested layout.


    Supported layouts:


      oscar1992a:
        pdg m x y z px py pz E t q rho


      oscar1997a:
        id pdg px py pz E m x y z t


      auto:
        try common variants and fall back to a loose heuristic.
    """
    try:
        if layout == "oscar1992a":
            if len(tokens) < 9:
                return None
            pdg = int(tokens[0])
            px = float(tokens[5])
            py = float(tokens[6])
            pz = float(tokens[7])
            E = float(tokens[8])
            return Particle(pdg, px, py, pz, E)


        if layout == "oscar1997a":
            if len(tokens) < 6:
                return None
            pdg = int(tokens[1])
            px = float(tokens[2])
            py = float(tokens[3])
            pz = float(tokens[4])
            E = float(tokens[5])
            return Particle(pdg, px, py, pz, E)


        if len(tokens) >= 12:
            try:
                pdg = int(tokens[0])
                m = float(tokens[1])
                px = float(tokens[5])
                py = float(tokens[6])
                pz = float(tokens[7])
                E = float(tokens[8])
                mass_shell = E * E - (px * px + py * py + pz * pz)
                if E > 0.0 and abs(mass_shell - m * m) < max(1.0, 5.0 * m * m + 1.0):
                    return Particle(pdg, px, py, pz, E)
            except ValueError:
                pass


        if len(tokens) >= 11:
            try:
                pdg = int(tokens[1])
                px = float(tokens[2])
                py = float(tokens[3])
                pz = float(tokens[4])
                E = float(tokens[5])
                return Particle(pdg, px, py, pz, E)
            except ValueError:
                pass


        if len(tokens) >= 9:
            pdg = int(tokens[0])
            px = float(tokens[5])
            py = float(tokens[6])
            pz = float(tokens[7])
            E = float(tokens[8])
            return Particle(pdg, px, py, pz, E)


        return None
    except (ValueError, IndexError):
        return None



def detect_layout(path: Path) -> str:
    """Sniff file header and choose a likely layout."""
    try:
        with open(path, "r", errors="replace") as f:
            head = [f.readline() for _ in range(8)]
    except OSError:
        return "auto"


    blob = " ".join(head).lower()
    if "osc1997a" in blob or "final_id_p_x" in blob:
        return "oscar1997a"
    if "oscar1992a" in blob or "final_p_x" in blob:
        return "oscar1992a"
    return "auto"



def stream_events(path: Path, layout: str) -> Iterable[List[Particle]]:
    """Yield events as lists of particles."""
    with open(path, "r", errors="replace") as f:
        pending: Optional[List[str]] = None


        while True:
            line = f.readline()
            if not line:
                return
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            tokens = stripped.split()
            n = _try_parse_event_header(tokens)
            if n is not None:
                pending = tokens
                break


        while pending is not None:
            n_particles = _try_parse_event_header(pending) or 0
            particles: List[Particle] = []
            read = 0


            while read < n_particles:
                line = f.readline()
                if not line:
                    break
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue


                tokens = stripped.split()
                particle = _parse_particle_line(tokens, layout)
                if particle is not None:
                    particles.append(particle)
                read += 1


            if particles:
                yield particles


            pending = None
            while True:
                line = f.readline()
                if not line:
                    break
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue


                tokens = stripped.split()
                n_next = _try_parse_event_header(tokens)
                if n_next is not None:
                    pending = tokens
                    break



# ---------------------------------------------------------------------------
# Beam / frame helpers
# ---------------------------------------------------------------------------



def ycm_fixed_target(
    beam_kinetic_agev: float,
    nucleon_mass_gev: float = 0.9382720813,
) -> float:
    """Return y_cm in lab frame for fixed-target NN collision.


    beam_kinetic_agev is projectile kinetic energy per nucleon.
    """
    m = nucleon_mass_gev
    e_beam = m + beam_kinetic_agev
    p_beam = math.sqrt(max(e_beam * e_beam - m * m, 0.0))
    e_tot = e_beam + m
    denom = e_tot - p_beam
    numer = e_tot + p_beam
    if denom <= 0.0 or numer <= 0.0:
        raise ValueError("Cannot compute fixed-target y_cm for this beam energy")
    return 0.5 * math.log(numer / denom)



def ycm_from_sqrts_fixed_target(
    sqrts_nn_gev: float,
    nucleon_mass_gev: float = 0.9382720813,
) -> float:
    """Return lab-frame y_cm for an equivalent fixed-target NN collision.


    For target nucleon at rest:


        s_NN = 2m^2 + 2m E_beam,total


    where E_beam,total is the projectile nucleon total energy.
    """
    m = nucleon_mass_gev
    if sqrts_nn_gev <= 2.0 * m:
        raise ValueError("sqrt(s_NN) must be larger than 2*m_N")


    s = sqrts_nn_gev * sqrts_nn_gev
    e_beam = (s - 2.0 * m * m) / (2.0 * m)
    p_beam = math.sqrt(max(e_beam * e_beam - m * m, 0.0))
    e_tot = e_beam + m
    denom = e_tot - p_beam
    numer = e_tot + p_beam
    if denom <= 0.0 or numer <= 0.0:
        raise ValueError("Cannot compute fixed-target y_cm for this sqrt(s_NN)")
    return 0.5 * math.log(numer / denom)



def ybeam_collider_from_sqrts(
    sqrts_nn_gev: float,
    nucleon_mass_gev: float = 0.9382720813,
) -> float:
    """Return beam rapidity in the CM frame for a symmetric NN collider."""
    m = nucleon_mass_gev
    if sqrts_nn_gev <= 2.0 * m:
        raise ValueError("sqrt(s_NN) must be larger than 2*m_N")
    e_beam_cm = 0.5 * sqrts_nn_gev
    gamma = e_beam_cm / m
    return math.acosh(gamma)



def resolve_y_shift(args: argparse.Namespace) -> float:
    """Return y_cm_in_input_frame.


    The analyzer then uses:


        y_analysis = y_input - y_shift


    where y_analysis is rapidity relative to the NN center-of-mass frame.
    """
    if args.input_frame == "cm":
        if args.y_shift is not None and abs(args.y_shift) > 1e-12:
            print("[warning] --input-frame cm: explicit --y-shift is ignored.", file=sys.stderr)
        return 0.0


    if args.y_shift is not None:
        return args.y_shift


    mode = args.collision_mode
    if mode == "fixed":
        mode = "fixed-target"


    if mode == "fixed-target":
        if args.beam_kinetic_agev is not None and args.ecm_snn is not None:
            raise ValueError("Use either --beam-kinetic-agev or --ecm-snn, not both")


        if args.beam_kinetic_agev is not None:
            ycm_target = ycm_fixed_target(args.beam_kinetic_agev)
        elif args.ecm_snn is not None:
            ycm_target = ycm_from_sqrts_fixed_target(args.ecm_snn)
        else:
            raise ValueError(
                "For fixed-target input in projectile/target frame, provide "
                "--ecm-snn, --beam-kinetic-agev, or explicit --y-shift"
            )


        if args.input_frame == "target":
            return ycm_target
        if args.input_frame == "projectile":
            return -ycm_target
        raise ValueError(f"Unsupported input frame: {args.input_frame}")


    if mode == "collider":
        if args.ecm_snn is None:
            raise ValueError(
                "For collider input in projectile/target frame, provide --ecm-snn "
                "or explicit --y-shift"
            )
        ybeam_cm = ybeam_collider_from_sqrts(args.ecm_snn)
        if args.input_frame == "projectile":
            return -ybeam_cm
        if args.input_frame == "target":
            return ybeam_cm
        raise ValueError(f"Unsupported input frame: {args.input_frame}")


    raise ValueError(f"Unsupported collision mode: {args.collision_mode}")



# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------



def make_edges(nbins: int, lo: float, hi: float) -> np.ndarray:
    if nbins <= 0:
        raise ValueError("Number of bins must be positive")
    if hi <= lo:
        raise ValueError("Histogram upper edge must be larger than lower edge")
    return np.linspace(lo, hi, nbins + 1)



def analyze(
    path: Path,
    layout: str,
    ycut: float,
    y_shift: float,
    pt_bins: int,
    pt_max: float,
    y_bins: int,
    y_min: float,
    y_max: float,
    include_k0: bool,
) -> dict:
    pt_edges = make_edges(pt_bins, 0.0, pt_max)
    y_edges = make_edges(y_bins, y_min, y_max)


    species = {
        "Kplus": (PDG_KPLUS,),
        "Kminus": (PDG_KMINUS,),
        "K0S": (PDG_K0S, PDG_K0, PDG_K0BAR) if include_k0 else (PDG_K0S,),
    }


    pt_counts = {name: np.zeros(pt_bins, dtype=np.int64) for name in species}
    y_counts = {name: np.zeros(y_bins, dtype=np.int64) for name in species}
    total_yield = {name: 0 for name in species}
    yield_in_window = {name: 0 for name in species}
    neutral_scale = 0.5 if include_k0 else 1.0


    n_events = 0
    n_particles_seen = 0


    for particles in stream_events(path, layout):
        n_events += 1
        for particle in particles:
            n_particles_seen += 1


            target = None
            for name, codes in species.items():
                if particle.pdg in codes:
                    target = name
                    break
            if target is None:
                continue


            total_yield[target] += 1
            y = particle.rapidity()
            if not math.isfinite(y):
                continue


            y_analysis = y - y_shift


            if y_min <= y_analysis < y_max:
                iy = int((y_analysis - y_min) / (y_max - y_min) * y_bins)
                if 0 <= iy < y_bins:
                    y_counts[target][iy] += 1


            if abs(y_analysis) < ycut:
                yield_in_window[target] += 1
                pt = particle.pt()
                if 0.0 <= pt < pt_max:
                    ip = int(pt / pt_max * pt_bins)
                    if 0 <= ip < pt_bins:
                        pt_counts[target][ip] += 1


    return {
        "n_events": n_events,
        "n_particles_seen": n_particles_seen,
        "pt_edges": pt_edges,
        "y_edges": y_edges,
        "pt_counts": pt_counts,
        "y_counts": y_counts,
        "total_yield": total_yield,
        "yield_in_window": yield_in_window,
        "ycut": ycut,
        "y_shift": y_shift,
        "include_k0": include_k0,
        "neutral_scale": neutral_scale,
    }



# ---------------------------------------------------------------------------
# Output: CSVs and plots
# ---------------------------------------------------------------------------



def write_csv(path: Path, header: List[str], rows: Iterable[List]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)



def save_pt_spectra(outdir: Path, res: dict, normalize: bool) -> None:
    edges = res["pt_edges"]
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    nev = max(res["n_events"], 1)
    rows = []


    fig, ax = plt.subplots(figsize=(7, 5))


    for name in ("Kplus", "Kminus", "K0S"):
        raw = res["pt_counts"][name].astype(float)
        scale = res.get("neutral_scale", 1.0) if name == "K0S" else 1.0
        counts = scale * raw
        err = scale * np.sqrt(np.maximum(raw, 0.0))


        if normalize:
            yvals = counts / (nev * widths)
            yerr = err / (nev * widths)
            ylabel = r"$dN/dp_T$ [(GeV/c)$^{-1}$]"
        else:
            yvals = counts
            yerr = err
            ylabel = "counts"


        ax.errorbar(
            centers,
            yvals,
            yerr=yerr,
            marker="o",
            ms=4,
            lw=1.2,
            capsize=2,
            label=_label(name),
        )


        for i, ctr in enumerate(centers):
            rows.append(
                [
                    name,
                    f"{edges[i]:.4f}",
                    f"{edges[i + 1]:.4f}",
                    f"{ctr:.4f}",
                    f"{counts[i]:.6e}",
                    f"{yvals[i]:.6e}",
                    f"{yerr[i]:.6e}",
                ]
            )


    y_label = _analysis_y_label(res)
    ax.set_xlabel(r"$p_T$ [GeV/c]")
    ax.set_ylabel(ylabel)
    ax.set_yscale("log")
    ax.set_title(f"Kaon $p_T$ spectra, $|{y_label}|<{res['ycut']:.2f}$")
    _add_plot_info(ax, res, loc="lower left")
    ax.legend()
    ax.grid(True, which="both", ls=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(outdir / "pt_spectra.png", dpi=140)
    plt.close(fig)


    write_csv(
        outdir / "pt_spectra.csv",
        ["species", "pt_lo", "pt_hi", "pt_center", "counts", "value", "error"],
        rows,
    )



def save_y_distributions(outdir: Path, res: dict, normalize: bool) -> None:
    edges = res["y_edges"]
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    nev = max(res["n_events"], 1)
    rows = []


    fig, ax = plt.subplots(figsize=(7, 5))


    for name in ("Kplus", "Kminus", "K0S"):
        raw = res["y_counts"][name].astype(float)
        scale = res.get("neutral_scale", 1.0) if name == "K0S" else 1.0
        counts = scale * raw
        err = scale * np.sqrt(np.maximum(raw, 0.0))


        if normalize:
            yvals = counts / (nev * widths)
            yerr = err / (nev * widths)
            ylabel = r"$dN/dy$"
        else:
            yvals = counts
            yerr = err
            ylabel = "counts"


        ax.errorbar(
            centers,
            yvals,
            yerr=yerr,
            marker="s",
            ms=4,
            lw=1.2,
            capsize=2,
            label=_label(name),
        )

        for i, ctr in enumerate(centers):
            rows.append(
                [
                    name,
                    f"{edges[i]:.4f}",
                    f"{edges[i + 1]:.4f}",
                    f"{ctr:.4f}",
                    f"{counts[i]:.6e}",
                    f"{yvals[i]:.6e}",
                    f"{yerr[i]:.6e}",
                ]
            )

    # ------------------------------------------------------------------
    # Extra curve: mean charged kaon  (K+ + K-) / 2
    # ------------------------------------------------------------------
    kp_raw = res["y_counts"]["Kplus"].astype(float)
    km_raw = res["y_counts"]["Kminus"].astype(float)
    mean_raw = 0.5 * (kp_raw + km_raw)
    mean_err_raw = 0.5 * np.sqrt(kp_raw + km_raw)   # Poisson sqrt on summed count

    if normalize:
        mean_vals = mean_raw / (nev * widths)
        mean_err  = mean_err_raw / (nev * widths)
    else:
        mean_vals = mean_raw
        mean_err  = mean_err_raw

    ax.errorbar(
        centers,
        mean_vals,
        yerr=mean_err,
        marker="D",
        ms=4,
        lw=1.4,
        capsize=2,
        ls="--",
        color="C3",
        label=r"$(K^+ + K^-)/2$",
    )

    for i, ctr in enumerate(centers):
        rows.append(
            [
                "KmeanCharged",
                f"{edges[i]:.4f}",
                f"{edges[i + 1]:.4f}",
                f"{ctr:.4f}",
                f"{mean_raw[i]:.6e}",
                f"{mean_vals[i]:.6e}",
                f"{mean_err[i]:.6e}",
            ]
        )
    # ------------------------------------------------------------------

    y_label = _analysis_y_label(res)
    ax.axvspan(
        -res["ycut"],
        res["ycut"],
        color="grey",
        alpha=0.10,
        label=rf"central window $\pm${res['ycut']:.2f}",
    )
    ax.set_xlabel(y_label)
    ax.set_ylabel(ylabel)
    ax.set_title("Kaon rapidity distributions")
    _add_plot_info(ax, res, loc="upper left")
    ax.legend()
    ax.grid(True, ls=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(outdir / "y_distributions.png", dpi=140)
    plt.close(fig)


    write_csv(
        outdir / "y_distributions.csv",
        ["species", "y_lo", "y_hi", "y_center", "counts", "value", "error"],
        rows,
    )



def save_ratio(outdir: Path, res: dict) -> None:
    edges = res["pt_edges"]
    centers = 0.5 * (edges[:-1] + edges[1:])


    kp = res["pt_counts"]["Kplus"].astype(float)
    km = res["pt_counts"]["Kminus"].astype(float)
    k0_raw = res["pt_counts"]["K0S"].astype(float)
    neutral_scale = res.get("neutral_scale", 1.0)
    k0 = neutral_scale * k0_raw


    num = 0.5 * (kp + km)
    num_var = 0.25 * (kp + km)
    den = k0
    den_var = neutral_scale * neutral_scale * k0_raw


    ratio = np.full_like(num, np.nan, dtype=float)
    err = np.full_like(num, np.nan, dtype=float)
    mask = den > 0


    ratio[mask] = num[mask] / den[mask]
    n_safe = np.where(num > 0.0, num, 1.0)
    d_safe = np.where(den > 0.0, den, 1.0)
    rel2 = (num_var / (n_safe * n_safe)) + (den_var / (d_safe * d_safe))
    err[mask] = np.abs(ratio[mask]) * np.sqrt(rel2[mask])


    rows = []
    for i, ctr in enumerate(centers):
        rows.append(
            [
                f"{edges[i]:.4f}",
                f"{edges[i + 1]:.4f}",
                f"{ctr:.4f}",
                int(kp[i]),
                int(km[i]),
                f"{k0[i]:.6e}",
                "" if np.isnan(ratio[i]) else f"{ratio[i]:.6e}",
                "" if np.isnan(err[i]) else f"{err[i]:.6e}",
            ]
        )


    write_csv(
        outdir / "ratio_pt.csv",
        ["pt_lo", "pt_hi", "pt_center", "K+", "K-", "K0S", "R", "R_err"],
        rows,
    )


    fig, ax = plt.subplots(figsize=(7, 5))
    finite = np.isfinite(ratio)
    ax.axhline(1.0, color="#555555", ls="--", lw=1.1)
    ax.fill_between(
        [edges[0], edges[-1]],
        0.95,
        1.05,
        color="#20808D",
        alpha=0.10,
        lw=0,
        label=r"$\pm$5% band",
    )


    if finite.any():
        ax.errorbar(
            centers[finite],
            ratio[finite],
            yerr=err[finite],
            marker="o",
            ms=4,
            lw=1.2,
            capsize=2,
            color="C3",
            label=r"$0.5(K^+ + K^-)/K^0_S$",
        )


    y_label = _analysis_y_label(res)
    ax.set_xlabel(r"$p_T$ [GeV/c]")
    ax.set_ylabel(r"$R(p_T) = \frac{1}{2}(K^+ + K^-)/K^0_S$")
    ax.set_title(f"Kaon ratio, $|{y_label}|<{res['ycut']:.2f}$")
    _add_plot_info(ax, res, loc="upper left")
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "ratio_pt.png", dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# NEW: dN/dy plot — K+, K-, K0S and (K+ + K-)/2 on a single panel
# ---------------------------------------------------------------------------


def save_mean_kaon_y(outdir: Path, res: dict, normalize: bool) -> None:
    """Plot dN/dy for K+, K-, K0S and the mean charged kaon (K+ + K-)/2.

    This dedicated panel makes the comparison between the mean charged kaon
    yield and the neutral K0S yield immediately visible.
    """
    edges = res["y_edges"]
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    nev = max(res["n_events"], 1)
    neutral_scale = res.get("neutral_scale", 1.0)

    # --- raw histograms ------------------------------------------------
    kp_raw = res["y_counts"]["Kplus"].astype(float)
    km_raw = res["y_counts"]["Kminus"].astype(float)
    k0_raw = res["y_counts"]["K0S"].astype(float)

    k0 = neutral_scale * k0_raw
    mean_raw = 0.5 * (kp_raw + km_raw)       # (K+ + K-)/2 in counts

    # Poisson uncertainties
    kp_err = np.sqrt(kp_raw)
    km_err = np.sqrt(km_raw)
    k0_err = neutral_scale * np.sqrt(k0_raw)
    mean_err_raw = 0.5 * np.sqrt(kp_raw + km_raw)

    if normalize:
        div = nev * widths
        kp_vals   = kp_raw  / div
        km_vals   = km_raw  / div
        k0_vals   = k0      / div
        mean_vals = mean_raw / div

        kp_yerr   = kp_err     / div
        km_yerr   = km_err     / div
        k0_yerr   = k0_err     / div
        mean_yerr = mean_err_raw / div

        ylabel = r"$dN/dy$"
    else:
        kp_vals = kp_raw;   kp_yerr   = kp_err
        km_vals = km_raw;   km_yerr   = km_err
        k0_vals = k0;       k0_yerr   = k0_err
        mean_vals = mean_raw; mean_yerr = mean_err_raw
        ylabel = "counts"

    # --- plot ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.errorbar(centers, kp_vals, yerr=kp_yerr,
                marker="^", ms=4, lw=1.2, capsize=2,
                color="C0", label=r"$K^+$")
    ax.errorbar(centers, km_vals, yerr=km_yerr,
                marker="v", ms=4, lw=1.2, capsize=2,
                color="C1", label=r"$K^-$")
    ax.errorbar(centers, k0_vals, yerr=k0_yerr,
                marker="s", ms=4, lw=1.2, capsize=2,
                color="C2", label=r"$K^0_S$")
    ax.errorbar(centers, mean_vals, yerr=mean_yerr,
                marker="D", ms=5, lw=1.6, capsize=2,
                ls="--", color="C3",
                label=r"$(K^+ + K^-)/2$")

    y_label = _analysis_y_label(res)
    ax.axvspan(
        -res["ycut"], res["ycut"],
        color="grey", alpha=0.10,
        label=rf"central window $\pm${res['ycut']:.2f}",
    )
    ax.set_xlabel(y_label)
    ax.set_ylabel(ylabel)
    ax.set_title(r"Kaon $dN/dy$ — charged mean vs. $K^0_S$")
    _add_plot_info(ax, res, loc="upper left")
    ax.legend(fontsize=9)
    ax.grid(True, ls=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(outdir / "mean_kaon_y.png", dpi=140)
    plt.close(fig)

    # --- CSV -----------------------------------------------------------
    rows = []
    for i, ctr in enumerate(centers):
        rows.append([
            f"{edges[i]:.4f}", f"{edges[i+1]:.4f}", f"{ctr:.4f}",
            f"{kp_vals[i]:.6e}", f"{kp_yerr[i]:.6e}",
            f"{km_vals[i]:.6e}", f"{km_yerr[i]:.6e}",
            f"{k0_vals[i]:.6e}", f"{k0_yerr[i]:.6e}",
            f"{mean_vals[i]:.6e}", f"{mean_yerr[i]:.6e}",
        ])
    write_csv(
        outdir / "mean_kaon_y.csv",
        ["y_lo", "y_hi", "y_center",
         "Kplus_value", "Kplus_err",
         "Kminus_value", "Kminus_err",
         "K0S_value", "K0S_err",
         "KmeanCharged_value", "KmeanCharged_err"],
        rows,
    )


def save_summary(outdir: Path, res: dict) -> None:
    nev = max(res["n_events"], 1)
    rows = [
        ["n_events", res["n_events"]],
        ["n_particles_seen", res["n_particles_seen"]],
        ["ycut", res["ycut"]],
        ["y_shift", res.get("y_shift", 0.0)],
        ["input_frame", res.get("input_frame", "")],
        ["collision_mode", res.get("collision_mode", "")],
        ["analysis_frame", res.get("analysis_frame", "cm")],
        ["ecm_snn_gev", res.get("ecm_snn_gev", "")],
        ["beam_kinetic_agev", res.get("beam_kinetic_agev", "")],
        ["collision_system", res.get("collision_system", "")],
        ["beam_label", res.get("beam_label", "")],
        ["neutral_scale_for_K0S", res.get("neutral_scale", 1.0)],
    ]


    for name in ("Kplus", "Kminus", "K0S"):
        scale = res.get("neutral_scale", 1.0) if name == "K0S" else 1.0
        rows.append([f"{name}_raw_total", res["total_yield"][name]])
        rows.append([f"{name}_total", scale * res["total_yield"][name]])
        rows.append([f"{name}_raw_in_window", res["yield_in_window"][name]])
        rows.append([f"{name}_in_window", scale * res["yield_in_window"][name]])
        rows.append(
            [
                f"{name}_per_event_in_window",
                scale * res["yield_in_window"][name] / nev,
            ]
        )


    kp = res["yield_in_window"]["Kplus"]
    km = res["yield_in_window"]["Kminus"]
    neutral_scale = res.get("neutral_scale", 1.0)
    k0_raw = res["yield_in_window"]["K0S"]
    k0 = neutral_scale * k0_raw


    if k0 > 0.0:
        R = 0.5 * (kp + km) / k0
        den_var = neutral_scale * neutral_scale * k0_raw
        sR = abs(R) * math.sqrt(
            0.25 * (kp + km) / max((0.5 * (kp + km)) ** 2, 1e-30)
            + den_var / max(k0 * k0, 1e-30)
        )
        rows.append(["R_integrated", R])
        rows.append(["R_integrated_err", sR])
    else:
        rows.append(["R_integrated", ""])
        rows.append(["R_integrated_err", ""])


    write_csv(outdir / "summary.csv", ["key", "value"], rows)



def _label(name: str) -> str:
    return {
        "Kplus": r"$K^+$",
        "Kminus": r"$K^-$",
        "K0S": r"$K^0_S$",
    }.get(name, name)



def _format_system(system: str) -> str:
    """Return compact mathtext labels for common systems."""
    s = system.strip().replace(" ", "")
    labels = {
        "12C+12C": r"$^{12}$C+$^{12}$C",
        "C12+C12": r"$^{12}$C+$^{12}$C",
        "C12C12": r"$^{12}$C+$^{12}$C",
        "Xe124+W184": r"$^{124}$Xe+$^{184}$W",
        "124Xe+184W": r"$^{124}$Xe+$^{184}$W",
        "Xe+W": r"Xe+W",
        "Xe124+Xe124": r"$^{124}$Xe+$^{124}$Xe",
        "124Xe+124Xe": r"$^{124}$Xe+$^{124}$Xe",
        "Xe+Xe": r"Xe+Xe",
        "d+d": r"d+d",
        "D+D": r"d+d",
    }
    return labels.get(s, system)



def _format_nev(n_events: int) -> str:
    if n_events >= 1000 and n_events % 1000 == 0:
        return f"{n_events // 1000}k"
    return f"{n_events:,}"



def _analysis_y_label(res: dict) -> str:
    if abs(float(res.get("y_shift", 0.0))) < 1e-12:
        return "y"
    return r"y-y_{cm}"



def _add_plot_info(ax, res: dict, loc: str = "lower left") -> None:
    system = _format_system(str(res.get("collision_system", ""))).strip()
    beam = str(res.get("beam_label", "")).strip()
    nev = _format_nev(int(res.get("n_events", 0)))
    parts = [part for part in (system, beam, f"$N_{{evt}}={nev}$") if part]
    text = ", ".join(parts)


    xy = {
        "lower left": (0.03, 0.05, "left", "bottom"),
        "upper left": (0.03, 0.95, "left", "top"),
        "lower right": (0.97, 0.05, "right", "bottom"),
        "upper right": (0.97, 0.95, "right", "top"),
    }.get(loc, (0.03, 0.05, "left", "bottom"))


    ax.text(
        xy[0],
        xy[1],
        text,
        transform=ax.transAxes,
        ha=xy[2],
        va=xy[3],
        fontsize=9,
        color="#444444",
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "none", "alpha": 0.65},
    )



# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------



def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze UrQMD .f19 / OSCAR-like output for kaon spectra and R(pT)."
    )
    parser.add_argument("input", help="Path to UrQMD .f19 / OSCAR file")
    parser.add_argument(
        "-o",
        "--outdir",
        default="urqmd_kaon_out",
        help="Output directory",
    )
    parser.add_argument(
        "--input-frame",
        choices=("projectile", "target", "cm"),
        default="target",
        help=(
            "Frame of UrQMD momentum output: projectile rest frame, target rest frame, "
            "or NN center-of-mass frame. For fixed-target lab output use target. "
            "For ordinary symmetric collider lab output use cm."
        ),
    )
    parser.add_argument(
        "--collision-mode",
        choices=("fixed-target", "fixed", "collider"),
        default="fixed-target",
        help="Collision geometry used to interpret projectile/target/cm frames.",
    )
    parser.add_argument(
        "--y-shift",
        type=float,
        default=None,
        help="Manual rapidity shift subtracted from y before cuts/histograms",
    )
    parser.add_argument(
        "--beam-kinetic-agev",
        type=float,
        default=None,
        help="Fixed-target kinetic beam energy per nucleon in AGeV; computes y_shift=y_cm(lab)",
    )
    parser.add_argument(
        "--ecm-snn",
        type=float,
        default=None,
        help="sqrt(s_NN) in GeV; for target/lab input computes equivalent fixed-target y_cm",
    )
    parser.add_argument(
        "--ycut",
        type=float,
        default=0.5,
        help="Central rapidity slice |y_analysis|<ycut",
    )
    parser.add_argument("--pt-bins", type=int, default=20)
    parser.add_argument(
        "--pt-max",
        type=float,
        default=2.0,
        help="Upper edge of pT histogram in GeV/c",
    )
    parser.add_argument("--y-bins", type=int, default=20)
    parser.add_argument("--y-min", type=float, default=-2.0)
    parser.add_argument("--y-max", type=float, default=2.0)
    parser.add_argument(
        "--layout",
        choices=("auto", "oscar1992a", "oscar1997a"),
        default="auto",
        help="OSCAR layout",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Plot raw counts instead of per-event densities",
    )
    parser.add_argument(
        "--include-k0",
        action="store_true",
        help="Use 0.5*(PDG 311 + -311) as K0S-equivalent together with true K0S=310",
    )
    parser.add_argument(
        "--system",
        default="",
        help='Collision-system label printed on plots, e.g. "Xe124+W184"',
    )
    parser.add_argument(
        "--beam-label",
        default="",
        help='Beam/energy label printed on plots, e.g. "sqrt(s_NN)=2.9 GeV, fixed target"',
    )
    return parser.parse_args(argv)



def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)


    in_path = Path(args.input)
    if not in_path.is_file():
        print(f"Input file not found: {in_path}", file=sys.stderr)
        return 2


    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)


    try:
        y_shift = resolve_y_shift(args)
    except ValueError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2


    layout = args.layout if args.layout != "auto" else detect_layout(in_path)


    print(f"[analyze_urqmd_kaons] file={in_path} layout={layout} outdir={outdir}")
    print(f"  collision_mode={args.collision_mode}")
    print(f"  input_frame={args.input_frame}")
    print(f"  analysis_frame=cm")
    print(f"  y_analysis = y_input - ({y_shift:.6f})")


    res = analyze(
        in_path,
        layout=layout,
        ycut=args.ycut,
        y_shift=y_shift,
        pt_bins=args.pt_bins,
        pt_max=args.pt_max,
        y_bins=args.y_bins,
        y_min=args.y_min,
        y_max=args.y_max,
        include_k0=args.include_k0,
    )


    res["collision_system"] = args.system
    res["beam_label"] = args.beam_label
    res["input_frame"] = args.input_frame
    res["collision_mode"] = args.collision_mode
    res["analysis_frame"] = "cm"
    res["ecm_snn_gev"] = args.ecm_snn if args.ecm_snn is not None else ""
    res["beam_kinetic_agev"] = (
        args.beam_kinetic_agev if args.beam_kinetic_agev is not None else ""
    )


    print(f"  events read: {res['n_events']}")
    print(f"  particles seen: {res['n_particles_seen']}")


    neutral_scale = res.get("neutral_scale", 1.0)
    k0s_total = neutral_scale * res["total_yield"]["K0S"]
    k0s_window = neutral_scale * res["yield_in_window"]["K0S"]


    print(
        f"  totals: K+={res['total_yield']['Kplus']} "
        f"K-={res['total_yield']['Kminus']} K0S_equiv={k0s_total:g}"
    )
    if args.include_k0:
        print(
            f"  neutral raw counts used for K0S_equiv: {res['total_yield']['K0S']} "
            f"(scale={neutral_scale:g})"
        )
    print(
        f"  in |y_analysis|<{args.ycut}: K+={res['yield_in_window']['Kplus']} "
        f"K-={res['yield_in_window']['Kminus']} K0S_equiv={k0s_window:g}"
    )


    normalize = not args.no_normalize
    save_pt_spectra(outdir, res, normalize=normalize)
    save_y_distributions(outdir, res, normalize=normalize)
    save_ratio(outdir, res)
    save_mean_kaon_y(outdir, res, normalize=normalize)
    save_summary(outdir, res)


    print(
        "  wrote: pt_spectra.{csv,png}, y_distributions.{csv,png}, "
        f"ratio_pt.{{csv,png}}, mean_kaon_y.{{csv,png}}, summary.csv -> {outdir}"
    )


    return 0



if __name__ == "__main__":
    sys.exit(main())
