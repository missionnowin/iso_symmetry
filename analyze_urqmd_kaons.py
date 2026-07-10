"""
analyze_urqmd_kaons.py

Standalone analyzer for UrQMD .f19 / OSCAR-like output. It computes kaon
yields, pT spectra, rapidity distributions, and

    R(pT) = 0.5 * (K+ + K-) / K0_S

inside a central rapidity slice |y_analysis| < ycut.

K0_S counting policy
--------------------
UrQMD writes hadrons in the STRONG basis: it stores K0 (PDG 311) and Kbar0
(PDG -311), not the weak eigenstates K0_S (310) / K0_L (130). The relation
to the experimentally observable K0_S (neglecting CP violation) is

    <K0_S> = 0.5 * (<K0> + <Kbar0>)

Therefore:

* Default (``--include-k0`` ON):
    K0_S_equiv = 0.5 * (N_{PDG 311} + N_{PDG -311})
    (strong eigenstates only, NO double counting with PDG 310)

* ``--no-include-k0`` (or strict mode): use only PDG 310 if the file
  happens to contain weak eigenstates already.

The script chooses the K0 counting policy via the ``--k0-mode`` flag,
which supersedes ``--include-k0`` for clarity:

    --k0-mode strong   : 0.5 * (N_{311} + N_{-311})   (correct for UrQMD .f19)
    --k0-mode weak     : N_{310}                      (only if file has K0_S)
    --k0-mode auto     : pick based on what the file contains (default)

Key frame convention
--------------------
y_analysis = y_input - y_cm_in_input_frame

Outputs
-------
In --outdir:

  pt_spectra.csv / .png                 (K+, K-, K0_S; dN/dpT total)
  pt_spectra_dn.csv / .png              (K+, K-, K0_S; dn/dpT per-event)
  y_distributions.csv / .png            (K+, K-, K0_S; dN/dy total)
  y_distributions_dn.csv / .png         (K+, K-, K0_S; dn/dy per-event)
  y_distributions_meancharged.csv/.png  ((K+ + K-)/2 vs 2*K0_S)
  ratio_pt.csv / .png
  ratio_y.csv / .png                    (R_K(y) isospin ratio vs rapidity)
  mean_kaon_y.csv / .png                (K+, K-, K0_S, (K+ + K-)/2 combined)
  summary.csv

Notation note
-------------
dN/dpT  -- total yield summed over all N_ev events, divided by bin width
dn/dpT  -- per-event yield: (1/N_ev) * dN/dpT
Same distinction applies to dN/dy vs dn/dy.
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

CHARGED_ACTIVITY_PDGS = {
    11, -11, 13, -13, 15, -15,
    211, -211, 321, -321, 2212, -2212,
    3112, -3112, 3222, -3222, 3312, -3312, 3334, -3334,
    1114, -1114, 2114, -2114, 2214, -2214, 2224, -2224,
    24, -24,
}

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
        denom = self.E - self.pz
        numer = self.E + self.pz
        if denom <= 0.0 or numer <= 0.0:
            return float("nan")
        return 0.5 * math.log(numer / denom)


# ---------------------------------------------------------------------------
# Tolerant OSCAR / f19 reader
# ---------------------------------------------------------------------------


def _try_parse_event_header(tokens: List[str]) -> Optional[int]:
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
# Beam / frame helpers (unchanged)
# ---------------------------------------------------------------------------


def ycm_fixed_target(beam_kinetic_agev: float, nucleon_mass_gev: float = 0.9382720813) -> float:
    m = nucleon_mass_gev
    e_beam = m + beam_kinetic_agev
    p_beam = math.sqrt(max(e_beam * e_beam - m * m, 0.0))
    e_tot = e_beam + m
    denom = e_tot - p_beam
    numer = e_tot + p_beam
    if denom <= 0.0 or numer <= 0.0:
        raise ValueError("Cannot compute fixed-target y_cm for this beam energy")
    return 0.5 * math.log(numer / denom)


def ycm_from_sqrts_fixed_target(sqrts_nn_gev: float, nucleon_mass_gev: float = 0.9382720813) -> float:
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


def ybeam_collider_from_sqrts(sqrts_nn_gev: float, nucleon_mass_gev: float = 0.9382720813) -> float:
    m = nucleon_mass_gev
    if sqrts_nn_gev <= 2.0 * m:
        raise ValueError("sqrt(s_NN) must be larger than 2*m_N")
    e_beam_cm = 0.5 * sqrts_nn_gev
    gamma = e_beam_cm / m
    return math.acosh(gamma)


def resolve_y_shift(args: argparse.Namespace) -> float:
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
                "For collider input in projectile/target frame, provide --ecm-snn or --y-shift"
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


def event_activity(particles: List[Particle], y_max_abs: Optional[float] = None) -> int:
    n = 0
    for p in particles:
        if p.pdg not in CHARGED_ACTIVITY_PDGS:
            continue
        if y_max_abs is not None:
            y = p.rapidity()
            if not math.isfinite(y) or abs(y) > y_max_abs:
                continue
        n += 1
    return n


def select_events_by_activity(
    path: Path,
    layout: str,
    top_fraction: Optional[float],
    activity_y_max: Optional[float],
) -> tuple[Optional[set[int]], dict]:
    if top_fraction is None:
        return None, {
            "enabled": False,
            "requested_fraction": None,
            "selected_events": None,
            "total_events_seen": None,
            "threshold_activity": None,
            "activity_y_max": activity_y_max,
        }

    if not (0.0 < top_fraction <= 1.0):
        raise ValueError("--centrality-top-fraction must be in the interval (0, 1]")

    activities = []
    idx = 0
    for particles in stream_events(path, layout):
        idx += 1
        activities.append((event_activity(particles, activity_y_max), idx))

    total = len(activities)
    if total == 0:
        return set(), {
            "enabled": True,
            "requested_fraction": top_fraction,
            "selected_events": 0,
            "total_events_seen": 0,
            "threshold_activity": None,
            "activity_y_max": activity_y_max,
        }

    keep = max(1, int(math.ceil(top_fraction * total)))
    activities.sort(reverse=True)
    selected = activities[:keep]
    selected_ids = {idx for _, idx in selected}
    threshold = selected[-1][0] if selected else None

    return selected_ids, {
        "enabled": True,
        "requested_fraction": top_fraction,
        "selected_events": len(selected_ids),
        "total_events_seen": total,
        "threshold_activity": threshold,
        "activity_y_max": activity_y_max,
    }


def _resolve_k0_mode(path: Path, layout: str, requested: str) -> tuple[str, dict]:
    if requested in ("strong", "weak"):
        info = {"requested": requested, "scanned_events": 0, "n_310": 0, "n_311": 0, "n_m311": 0}
        return requested, info

    n_310 = n_311 = n_m311 = 0
    scanned = 0
    for particles in stream_events(path, layout):
        scanned += 1
        for p in particles:
            if p.pdg == PDG_K0S:
                n_310 += 1
            elif p.pdg == PDG_K0:
                n_311 += 1
            elif p.pdg == PDG_K0BAR:
                n_m311 += 1
        if scanned >= 2000:
            break

    info = {"requested": "auto", "scanned_events": scanned, "n_310": n_310, "n_311": n_311, "n_m311": n_m311}
    has_strong = (n_311 + n_m311) > 0
    has_weak = n_310 > 0

    if has_strong and not has_weak:
        return "strong", info
    if has_weak and not has_strong:
        return "weak", info
    if has_strong and has_weak:
        print(
            f"[warning] file contains BOTH K0_S (PDG 310) and K0/Kbar0 (PDG +/-311) in the first {scanned} events. "
            f"Defaulting to 'strong' (0.5*(N_311 + N_-311)) to avoid double counting. "
            f"Use --k0-mode weak explicitly if you want N_310 instead.",
            file=sys.stderr,
        )
        return "strong", info
    return "strong", info


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
    k0_mode: str,
    selected_event_ids: Optional[set[int]] = None,
) -> dict:
    pt_edges = make_edges(pt_bins, 0.0, pt_max)
    y_edges = make_edges(y_bins, y_min, y_max)

    if k0_mode == "strong":
        k0s_codes = (PDG_K0, PDG_K0BAR)
        neutral_scale = 0.5
    elif k0_mode == "weak":
        k0s_codes = (PDG_K0S,)
        neutral_scale = 1.0
    else:
        raise ValueError(f"Unknown k0_mode: {k0_mode}")

    species = {
        "Kplus": (PDG_KPLUS,),
        "Kminus": (PDG_KMINUS,),
        "K0S": k0s_codes,
    }

    pt_counts = {name: np.zeros(pt_bins, dtype=np.int64) for name in species}
    y_counts = {name: np.zeros(y_bins, dtype=np.int64) for name in species}
    total_yield = {name: 0 for name in species}
    yield_in_window = {name: 0 for name in species}

    n_events = 0
    n_particles_seen = 0
    n_events_seen_total = 0

    for particles in stream_events(path, layout):
        n_events_seen_total += 1
        if selected_event_ids is not None and n_events_seen_total not in selected_event_ids:
            continue
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
        "n_events_seen_total": n_events_seen_total,
        "n_particles_seen": n_particles_seen,
        "pt_edges": pt_edges,
        "y_edges": y_edges,
        "pt_counts": pt_counts,
        "y_counts": y_counts,
        "total_yield": total_yield,
        "yield_in_window": yield_in_window,
        "ycut": ycut,
        "y_shift": y_shift,
        "k0_mode": k0_mode,
        "k0s_codes": k0s_codes,
        "neutral_scale": neutral_scale,
    }


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def write_csv(path: Path, header: List[str], rows: Iterable[List]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)


def _label(name: str) -> str:
    return {"Kplus": r"$K^+$", "Kminus": r"$K^-$", "K0S": r"$K^0_S$"}.get(name, name)


def _format_nev(n_events: int) -> str:
    if n_events >= 1000 and n_events % 1000 == 0:
        return f"{n_events // 1000}k"
    return f"{n_events:,}"


def _analysis_y_label(res: dict) -> str:
    #if abs(float(res.get("y_shift", 0.0))) < 1e-12:
     #   return "y"
    #return r"y-y_{cm}"
    return "y"


def _add_plot_info(ax, res: dict, loc: str = "lower left") -> None:
    system = str(res.get("collision_system", "")).strip()
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
        xy[0], xy[1], text,
        transform=ax.transAxes,
        ha=xy[2], va=xy[3],
        fontsize=9, color="#444444",
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "none", "alpha": 0.65},
    )


# ---------------------------------------------------------------------------
# pT spectra -- dN/dpT (total, existing behaviour)
# ---------------------------------------------------------------------------


def save_pt_spectra(outdir: Path, res: dict, normalize: bool) -> None:
    edges = res["pt_edges"]
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    nev = max(res["n_events"], 1)
    rows = []
    fig, ax = plt.subplots(figsize=(7, 5))
    ylabel = "counts"
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
        ax.errorbar(centers, yvals, yerr=yerr, marker="o", ms=4, lw=1.2, capsize=2, label=_label(name))
        for i, ctr in enumerate(centers):
            rows.append([name, f"{edges[i]:.4f}", f"{edges[i + 1]:.4f}", f"{ctr:.4f}", f"{counts[i]:.6e}", f"{yvals[i]:.6e}", f"{yerr[i]:.6e}"])
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
    write_csv(outdir / "pt_spectra.csv", ["species", "pt_lo", "pt_hi", "pt_center", "counts", "value", "error"], rows)


# ---------------------------------------------------------------------------
# pT spectra -- dn/dpT  (per-event, NEW)
# ---------------------------------------------------------------------------


def save_pt_spectra_dn(outdir: Path, res: dict) -> None:
    """Write pt_spectra_dn.csv and pt_spectra_dn.png with per-event dn/dpT.

    dn/dpT = (1/N_ev) * dN/dpT
           = counts / (N_ev * delta_pT)

    The CSV columns are identical to pt_spectra.csv but 'value' and 'error'
    now hold dn/dpT and its statistical uncertainty instead of dN/dpT.
    A header comment line records N_ev so the normalisation is unambiguous.
    """
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
        err_counts = scale * np.sqrt(np.maximum(raw, 0.0))
        # per-event normalisation
        dn = counts / (nev * widths)
        dn_err = err_counts / (nev * widths)
        ax.errorbar(centers, dn, yerr=dn_err, marker="o", ms=4, lw=1.2, capsize=2, label=_label(name))
        for i, ctr in enumerate(centers):
            rows.append([
                name,
                f"{edges[i]:.4f}", f"{edges[i + 1]:.4f}", f"{ctr:.4f}",
                f"{counts[i]:.6e}",
                f"{dn[i]:.6e}",
                f"{dn_err[i]:.6e}",
            ])

    y_label = _analysis_y_label(res)
    ax.set_xlabel(r"$p_T$ [GeV/c]")
    ax.set_ylabel(r"$dn/dp_T$ [(GeV/c)$^{-1}$]")
    ax.set_yscale("log")
    ax.set_title(f"Kaon $p_T$ spectra (per event), $|{y_label}|<{res['ycut']:.2f}$")
    _add_plot_info(ax, res, loc="lower left")
    ax.legend()
    ax.grid(True, which="both", ls=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(outdir / "pt_spectra_dn.png", dpi=140)
    plt.close(fig)

    # write CSV with a leading comment so the normalisation is self-documenting
    csv_path = outdir / "pt_spectra_dn.csv"
    with open(csv_path, "w", newline="") as f:
        f.write(f"# dn/dpT per-event spectrum; N_ev={nev}\n")
        writer = csv.writer(f)
        writer.writerow(["species", "pt_lo", "pt_hi", "pt_center", "counts", "dn_dpt", "dn_dpt_err"])
        for row in rows:
            writer.writerow(row)


# ---------------------------------------------------------------------------
# y distributions  --  dN/dy (total, existing behaviour)
# ---------------------------------------------------------------------------


def save_y_distributions(outdir: Path, res: dict, normalize: bool) -> None:
    """Three-species dN/dy plot: K+, K-, K0_S only.

    The mean-charged curve has been moved to its own dedicated panel
    (save_y_distributions_meancharged) so this plot shows ONLY the three
    primary species.
    """
    edges = res["y_edges"]
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    nev = max(res["n_events"], 1)
    rows = []
    fig, ax = plt.subplots(figsize=(7, 5))
    ylabel = "counts"
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
        ax.errorbar(centers, yvals, yerr=yerr, marker="s", ms=4, lw=1.2, capsize=2, label=_label(name))
        for i, ctr in enumerate(centers):
            rows.append([name, f"{edges[i]:.4f}", f"{edges[i + 1]:.4f}", f"{ctr:.4f}", f"{counts[i]:.6e}", f"{yvals[i]:.6e}", f"{yerr[i]:.6e}"])
    y_label = _analysis_y_label(res)
    ax.axvspan(-res["ycut"], res["ycut"], color="grey", alpha=0.10, label=rf"central window $\pm${res['ycut']:.2f}")
    ax.set_xlabel(y_label)
    ax.set_ylabel(ylabel)
    ax.set_title(r"Kaon rapidity distributions: $K^+$, $K^-$, $K^0_S$")
    _add_plot_info(ax, res, loc="upper left")
    ax.legend()
    ax.grid(True, ls=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(outdir / "y_distributions.png", dpi=140)
    plt.close(fig)
    write_csv(outdir / "y_distributions.csv", ["species", "y_lo", "y_hi", "y_center", "counts", "value", "error"], rows)


# ---------------------------------------------------------------------------
# y distributions -- dn/dy  (per-event, NEW)
# ---------------------------------------------------------------------------


def save_y_distributions_dn(outdir: Path, res: dict) -> None:
    """Write y_distributions_dn.csv and y_distributions_dn.png with per-event dn/dy.

    dn/dy = (1/N_ev) * dN/dy
          = counts / (N_ev * delta_y)

    A header comment line records N_ev so the normalisation is unambiguous.
    """
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
        err_counts = scale * np.sqrt(np.maximum(raw, 0.0))
        # per-event normalisation
        dn = counts / (nev * widths)
        dn_err = err_counts / (nev * widths)
        ax.errorbar(centers, dn, yerr=dn_err, marker="s", ms=4, lw=1.2, capsize=2, label=_label(name))
        for i, ctr in enumerate(centers):
            rows.append([
                name,
                f"{edges[i]:.4f}", f"{edges[i + 1]:.4f}", f"{ctr:.4f}",
                f"{counts[i]:.6e}",
                f"{dn[i]:.6e}",
                f"{dn_err[i]:.6e}",
            ])

    y_label = _analysis_y_label(res)
    ax.axvspan(-res["ycut"], res["ycut"], color="grey", alpha=0.10,
               label=rf"central window $\pm${res['ycut']:.2f}")
    ax.set_xlabel(y_label)
    ax.set_ylabel(r"$dn/dy$")
    ax.set_title(r"Kaon rapidity distributions (per event): $K^+$, $K^-$, $K^0_S$")
    _add_plot_info(ax, res, loc="upper left")
    ax.legend()
    ax.grid(True, ls=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(outdir / "y_distributions_dn.png", dpi=140)
    plt.close(fig)

    csv_path = outdir / "y_distributions_dn.csv"
    with open(csv_path, "w", newline="") as f:
        f.write(f"# dn/dy per-event spectrum; N_ev={nev}\n")
        writer = csv.writer(f)
        writer.writerow(["species", "y_lo", "y_hi", "y_center", "counts", "dn_dy", "dn_dy_err"])
        for row in rows:
            writer.writerow(row)


# ---------------------------------------------------------------------------
# y distributions  --  (K+ + K-)/2  vs  2 * K0_S   (unchanged)
# ---------------------------------------------------------------------------


def save_y_distributions_meancharged(outdir: Path, res: dict, normalize: bool) -> None:
    """Dedicated panel comparing  (K+ + K-)/2  to  2 * K0_S  in dN/dy.

    The isospin-symmetry test of Eq. (2) reads

        R_K = (<K+> + <K->) / (2 <K0_S>) = 1

    Plotting (K+ + K-)/2 and 2*K0_S together makes the test direct: under
    exact isospin symmetry the two curves must coincide.  Equivalently this
    is the same as plotting (K+ + K-)/2 against <K0_S> after multiplying by 2,
    or (K+ + K-) against 2*K0_S -- we use the half/double convention here.
    """
    edges = res["y_edges"]
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    nev = max(res["n_events"], 1)
    neutral_scale = res.get("neutral_scale", 1.0)
    kp_raw = res["y_counts"]["Kplus"].astype(float)
    km_raw = res["y_counts"]["Kminus"].astype(float)
    k0_raw = res["y_counts"]["K0S"].astype(float)
    k0s_counts = neutral_scale * k0_raw
    k0s_err = neutral_scale * np.sqrt(np.maximum(k0_raw, 0.0))
    mean_raw = 0.5 * (kp_raw + km_raw)
    mean_err_raw = 0.5 * np.sqrt(kp_raw + km_raw)
    if normalize:
        div = nev * widths
        mean_vals = mean_raw / div
        mean_yerr = mean_err_raw / div
        k0s_vals = k0s_counts / div
        k0s_yerr = k0s_err / div
        ylabel = r"$dN/dy$"
    else:
        mean_vals = mean_raw
        mean_yerr = mean_err_raw
        k0s_vals = k0s_counts
        k0s_yerr = k0s_err
        ylabel = "counts"
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(centers, mean_vals, yerr=mean_yerr, marker="D", ms=5, lw=1.6, capsize=2, ls="-", color="C3", label=r"$(K^+ + K^-)/2$")
    ax.errorbar(centers, k0s_vals, yerr=k0s_yerr, marker="o", ms=5, lw=1.6, capsize=2, ls="--", color="C2", label=r"$K^0_S$")
    y_label = _analysis_y_label(res)
    ax.axvspan(-res["ycut"], res["ycut"], color="grey", alpha=0.10, label=rf"central window $\pm${res['ycut']:.2f}")
    ax.set_xlabel(y_label)
    ax.set_ylabel(ylabel)
    ax.set_title(r"$(K^+ + K^-)/2$ vs $K^0_S$")
    _add_plot_info(ax, res, loc="upper left")
    ax.legend()
    ax.grid(True, ls=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(outdir / "y_distributions_meancharged.png", dpi=140)
    plt.close(fig)
    rows = []
    for i, ctr in enumerate(centers):
        rows.append([f"{edges[i]:.4f}", f"{edges[i + 1]:.4f}", f"{ctr:.4f}", f"{mean_vals[i]:.6e}", f"{mean_yerr[i]:.6e}", f"{k0s_vals[i]:.6e}", f"{k0s_yerr[i]:.6e}"])
    write_csv(outdir / "y_distributions_meancharged.csv", ["y_lo", "y_hi", "y_center", "KmeanCharged_value", "KmeanCharged_err", "TwoK0S_value", "TwoK0S_err"], rows)


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
        rows.append([f"{edges[i]:.4f}", f"{edges[i + 1]:.4f}", f"{ctr:.4f}", int(kp[i]), int(km[i]), f"{k0[i]:.6e}", "" if np.isnan(ratio[i]) else f"{ratio[i]:.6e}", "" if np.isnan(err[i]) else f"{err[i]:.6e}"])
    write_csv(outdir / "ratio_pt.csv", ["pt_lo", "pt_hi", "pt_center", "K+", "K-", "K0S", "R", "R_err"], rows)
    fig, ax = plt.subplots(figsize=(7, 5))
    finite = np.isfinite(ratio)
    ax.axhline(1.0, color="#555555", ls="--", lw=1.1)
    ax.fill_between([edges[0], edges[-1]], 0.95, 1.05, color="#20808D", alpha=0.10, lw=0, label=r"$\pm$5% band")
    if finite.any():
        ax.errorbar(centers[finite], ratio[finite], yerr=err[finite], marker="o", ms=4, lw=1.2, capsize=2, color="C3", label=r"$0.5(K^+ + K^-)/K^0_S$")
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
# R_K(y)  --  isospin ratio (K+ + K-)/2  /  K0_S  vs rapidity
# ---------------------------------------------------------------------------


def save_ratio_y(outdir: Path, res: dict) -> None:
    """Plot and save R_K(y) = 0.5*(K+ + K-) / K0_S as a function of rapidity.

    Under exact isospin symmetry R_K = 1 everywhere.
    Deviations reveal where charge symmetry is broken across rapidity.
    Complements save_ratio() which shows the same observable vs p_T.
    """
    edges = res["y_edges"]
    centers = 0.5 * (edges[:-1] + edges[1:])
    neutral_scale = res.get("neutral_scale", 1.0)

    kp      = res["y_counts"]["Kplus"].astype(float)
    km      = res["y_counts"]["Kminus"].astype(float)
    k0_raw  = res["y_counts"]["K0S"].astype(float)
    k0      = neutral_scale * k0_raw

    num     = 0.5 * (kp + km)
    num_var = 0.25 * (kp + km)
    den_var = neutral_scale * neutral_scale * k0_raw

    ratio = np.full_like(num, np.nan, dtype=float)
    err   = np.full_like(num, np.nan, dtype=float)
    mask  = k0 > 0
    ratio[mask] = num[mask] / k0[mask]
    n_safe = np.where(num > 0.0, num, 1.0)
    d_safe = np.where(k0  > 0.0, k0,  1.0)
    rel2   = (num_var / (n_safe * n_safe)) + (den_var / (d_safe * d_safe))
    err[mask] = np.abs(ratio[mask]) * np.sqrt(rel2[mask])

    rows = []
    for i, ctr in enumerate(centers):
        rows.append([
            f"{edges[i]:.4f}", f"{edges[i + 1]:.4f}", f"{ctr:.4f}",
            int(kp[i]), int(km[i]), f"{k0[i]:.6e}",
            "" if np.isnan(ratio[i]) else f"{ratio[i]:.6e}",
            "" if np.isnan(err[i])   else f"{err[i]:.6e}",
        ])
    write_csv(
        outdir / "ratio_y.csv",
        ["y_lo", "y_hi", "y_center", "K+", "K-", "K0S", "R_K", "R_K_err"],
        rows,
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.axhline(1.0, color="#555555", ls="--", lw=1.1, label="isospin symmetry ($R_K=1$)")
    finite = np.isfinite(ratio)
    if finite.any():
        ax.errorbar(
            centers[finite], ratio[finite], yerr=err[finite],
            marker="s", ms=4, lw=1.2, capsize=2, color="C1",
            label=r"$R_K = \frac{1}{2}(K^+ + K^-)/K^0_S$",
        )
    ax.axvspan(
        -res["ycut"], res["ycut"],
        color="grey", alpha=0.10,
        label=rf"central window $\pm${res['ycut']:.2f}",
    )
    y_label = _analysis_y_label(res)
    ax.set_xlabel(y_label)
    ax.set_ylabel(r"$R_K(y)$")
    ax.set_title(r"Isospin ratio $R_K(y) = \frac{1}{2}(K^+ + K^-)\,/\,K^0_S$")
    _add_plot_info(ax, res, loc="lower right")
    ax.legend(fontsize=8)
    ax.grid(True, ls=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(outdir / "ratio_y.png", dpi=140)
    plt.close(fig)

def save_summary(outdir: Path, res: dict) -> None:
    nev = max(res["n_events"], 1)
    rows = [
        ["n_events", res["n_events"]],
        ["n_events_seen_total", res.get("n_events_seen_total", res["n_events"])],
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
        ["k0_mode", res.get("k0_mode", "")],
        ["k0s_codes", str(res.get("k0s_codes", ""))],
        ["neutral_scale_for_K0S", res.get("neutral_scale", 1.0)],
        ["centrality_top_fraction", res.get("centrality_top_fraction", "")],
        ["centrality_activity_ymax", res.get("centrality_activity_ymax", "")],
        ["centrality_selected_events", res.get("centrality_selected_events", "")],
        ["centrality_total_events_seen", res.get("centrality_total_events_seen", "")],
        ["centrality_threshold_activity", res.get("centrality_threshold_activity", "")],
    ]
    for name in ("Kplus", "Kminus", "K0S"):
        scale = res.get("neutral_scale", 1.0) if name == "K0S" else 1.0
        rows.append([f"{name}_raw_total", res["total_yield"][name]])
        rows.append([f"{name}_total", scale * res["total_yield"][name]])
        rows.append([f"{name}_raw_in_window", res["yield_in_window"][name]])
        rows.append([f"{name}_in_window", scale * res["yield_in_window"][name]])
        rows.append([f"{name}_per_event_in_window", scale * res["yield_in_window"][name] / nev])
    kp = res["yield_in_window"]["Kplus"]
    km = res["yield_in_window"]["Kminus"]
    neutral_scale = res.get("neutral_scale", 1.0)
    k0_raw = res["yield_in_window"]["K0S"]
    k0 = neutral_scale * k0_raw
    if k0 > 0.0:
        R = 0.5 * (kp + km) / k0
        den_var = neutral_scale * neutral_scale * k0_raw
        sR = abs(R) * math.sqrt(0.25 * (kp + km) / max((0.5 * (kp + km)) ** 2, 1e-30) + den_var / max(k0 * k0, 1e-30))
        rows.append(["R_integrated", R])
        rows.append(["R_integrated_err", sR])
    else:
        rows.append(["R_integrated", ""])
        rows.append(["R_integrated_err", ""])
    write_csv(outdir / "summary.csv", ["key", "value"], rows)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze UrQMD .f19 / OSCAR-like output for kaon spectra and R(pT).")
    parser.add_argument("input", help="Path to UrQMD .f19 / OSCAR file")
    parser.add_argument("-o", "--outdir", default="urqmd_kaon_out")
    parser.add_argument("--input-frame", choices=("projectile", "target", "cm"), default="target")
    parser.add_argument("--collision-mode", choices=("fixed-target", "fixed", "collider"), default="fixed-target")
    parser.add_argument("--y-shift", type=float, default=None)
    parser.add_argument("--beam-kinetic-agev", type=float, default=None)
    parser.add_argument("--ecm-snn", type=float, default=None)
    parser.add_argument("--ycut", type=float, default=0.5)
    parser.add_argument("--pt-bins", type=int, default=20)
    parser.add_argument("--pt-max", type=float, default=2.0)
    parser.add_argument("--y-bins", type=int, default=20)
    parser.add_argument("--y-min", type=float, default=-2.0)
    parser.add_argument("--y-max", type=float, default=2.0)
    parser.add_argument("--layout", choices=("auto", "oscar1992a", "oscar1997a"), default="auto")
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--k0-mode", choices=("auto", "strong", "weak"), default="auto")
    parser.add_argument("--include-k0", action="store_true", help="DEPRECATED. Equivalent to --k0-mode strong.")
    parser.add_argument("--system", default="")
    parser.add_argument("--beam-label", default="")
    parser.add_argument("--centrality-top-fraction", type=float, default=None, help="Keep only the top fraction of events ranked by charged-particle activity, e.g. 0.10 for top 10% most active events.")
    parser.add_argument("--centrality-activity-ymax", type=float, default=None, help="Optional |y| acceptance used when defining event activity. If omitted, all charged particles are counted.")
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
    requested_k0_mode = args.k0_mode
    if args.include_k0 and args.k0_mode == "auto":
        requested_k0_mode = "strong"
    k0_mode, k0_info = _resolve_k0_mode(in_path, layout, requested_k0_mode)
    selected_event_ids, centrality_info = select_events_by_activity(in_path, layout, args.centrality_top_fraction, args.centrality_activity_ymax)
    print(f"[analyze_urqmd_kaons] file={in_path} layout={layout} outdir={outdir}")
    print(f" collision_mode={args.collision_mode}")
    print(f" input_frame={args.input_frame}")
    print(f" analysis_frame=cm")
    print(f" y_analysis = y_input - ({y_shift:.6f})")
    print(f" k0_mode={k0_mode} (requested={requested_k0_mode})")
    if k0_info.get("scanned_events"):
        print(f" k0_prescan: events={k0_info['scanned_events']} N_310={k0_info['n_310']} N_311={k0_info['n_311']} N_-311={k0_info['n_m311']}")
    if centrality_info.get("enabled"):
        print(f" centrality_filter=top_activity_fraction {centrality_info['requested_fraction']:.6f}; selected={centrality_info['selected_events']}/{centrality_info['total_events_seen']}; threshold_activity={centrality_info['threshold_activity']}; activity_|y|<={centrality_info['activity_y_max']}")
    res = analyze(in_path, layout=layout, ycut=args.ycut, y_shift=y_shift, pt_bins=args.pt_bins, pt_max=args.pt_max, y_bins=args.y_bins, y_min=args.y_min, y_max=args.y_max, k0_mode=k0_mode, selected_event_ids=selected_event_ids)
    res["collision_system"] = args.system
    res["beam_label"] = args.beam_label
    res["input_frame"] = args.input_frame
    res["collision_mode"] = args.collision_mode
    res["analysis_frame"] = "cm"
    res["ecm_snn_gev"] = args.ecm_snn if args.ecm_snn is not None else ""
    res["beam_kinetic_agev"] = args.beam_kinetic_agev if args.beam_kinetic_agev is not None else ""
    res["centrality_top_fraction"] = args.centrality_top_fraction if args.centrality_top_fraction is not None else ""
    res["centrality_activity_ymax"] = args.centrality_activity_ymax if args.centrality_activity_ymax is not None else ""
    res["centrality_selected_events"] = centrality_info.get("selected_events", "")
    res["centrality_total_events_seen"] = centrality_info.get("total_events_seen", "")
    res["centrality_threshold_activity"] = centrality_info.get("threshold_activity", "")
    print(f" events accepted: {res['n_events']}")
    print(f" events seen total: {res['n_events_seen_total']}")
    print(f" particles seen: {res['n_particles_seen']}")
    neutral_scale = res.get("neutral_scale", 1.0)
    k0s_total = neutral_scale * res["total_yield"]["K0S"]
    k0s_window = neutral_scale * res["yield_in_window"]["K0S"]
    print(f" totals: K+={res['total_yield']['Kplus']} K-={res['total_yield']['Kminus']} K0S_equiv={k0s_total:g}")
    print(f" neutral raw counts used for K0S_equiv: {res['total_yield']['K0S']} (scale={neutral_scale:g}, codes={res['k0s_codes']})")
    print(f" in |y_analysis|<{args.ycut}: K+={res['yield_in_window']['Kplus']} K-={res['yield_in_window']['Kminus']} K0S_equiv={k0s_window:g}")
    normalize = not args.no_normalize
    save_pt_spectra(outdir, res, normalize=normalize)
    save_pt_spectra_dn(outdir, res)
    save_y_distributions(outdir, res, normalize=normalize)
    save_y_distributions_dn(outdir, res)
    save_y_distributions_meancharged(outdir, res, normalize=normalize)
    save_ratio(outdir, res)
    save_ratio_y(outdir, res)
    save_summary(outdir, res)
    print(" wrote: pt_spectra.{csv,png}, pt_spectra_dn.{csv,png}, y_distributions.{csv,png}, y_distributions_dn.{csv,png}, y_distributions_meancharged.{csv,png}, ratio_pt.{csv,png}, ratio_y.{csv,png}, summary.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
