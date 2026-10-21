"""Reproduce every number and figure: python -m oregonator.report [--reuse]

Writes results/summary.json, results/sweep.csv and figures/*.{pdf,png}.
The parameter sweep and the bisection for the cycle edges are the slow part
(10–30 minutes on 8 cores); --reuse takes them from results/.
"""
from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .analysis import hopf_frequency, hopf_points, linear_period, transversality
from .digitize import FIGURES, digitize_all
from .model import (
    SLIDE_PARAMS,
    SLIDE_RATES,
    SLIDE_U0,
    Params,
    char_coeffs,
    char_coeffs_origin,
    eigenvalues,
    equilibrium,
    from_rate_constants,
)
from .simulate import attractor, simulate

ROOT = Path(__file__).resolve().parents[2]
FIG_DIR, RES_DIR, PIC_DIR = ROOT / "figures", ROOT / "results", ROOT / "original" / "pic"

# What the original slides state.
CLAIMED = {
    "unstable_interval": [0.25, 0.5],
    "E2_at_f_0.1": [0.798, 0.2, 0.798],
    "period_at_f_0.4": 6822.44,
}

COLORS = {"x": "#2a78d6", "y": "#eb6834", "z": "#1baf7a"}
INK, INK_2, MUTED, GRID = "#0b0b0b", "#52514e", "#a3a29c", "#e4e3df"
LABELS = {"x": r"HBrO$_2$ ($x$)", "y": r"Br$^-$ ($y$)", "z": r"Ce(IV) ($z$)"}
DASHES = {"x": "-", "y": "-", "z": "--"}

SWEEP_F = np.round(np.arange(0.05, 1.5001, 0.025), 4)
FIT_F = np.round(np.concatenate([np.arange(0.05, 1.5, 0.05), np.arange(1.5, 8.001, 0.1)]), 3)


def _style():
    plt.rcParams.update({
        "font.size": 11, "axes.titlesize": 11, "axes.labelsize": 11, "legend.fontsize": 9.5,
        "axes.edgecolor": INK_2, "axes.labelcolor": INK, "xtick.color": INK_2, "ytick.color": INK_2,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.8,
        "legend.frameon": False, "savefig.bbox": "tight", "savefig.dpi": 200,
    })


def _save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{name}.{ext}")
    plt.close(fig)


# ---------------------------------------------------------------- computations

def _fit_job(args):
    name, f = args
    spec = FIGURES[name]
    curves = digitize_all(PIC_DIR)[name]
    t, u = simulate(SLIDE_PARAMS.with_f(f), SLIDE_U0, spec["t_max"], 7001)
    log_u = np.log10(u)
    err = np.concatenate([np.abs(np.interp(pts[:, 0], t, log_u["xyz".index(k)]) - pts[:, 1])
                          for k, pts in curves.items()])
    return name, float(f), float(np.median(err))


def fit_original_figures(ex) -> dict:
    """Median |log10 error| between each original figure and simulations over a grid of f."""
    jobs = [(name, f) for name in FIGURES for f in FIT_F]
    rows = list(ex.map(_fit_job, jobs))
    out = {}
    for name, spec in FIGURES.items():
        errs = {f: e for n, f, e in rows if n == name}
        best = min(errs, key=errs.get)
        out[name] = {"f_label": spec["f_label"], "error_at_label": errs[spec["f_label"]],
                     "f_best": best, "error_at_best": errs[best],
                     "errors": {str(f): e for f, e in errs.items()}}
    return out


def _sweep_job(args):
    f, start = args
    p = SLIDE_PARAMS.with_f(f)
    u0 = equilibrium(p) * (1 + 1e-3) if start == "near E2*" else SLIDE_U0
    a = attractor(p, u0, t_end=100.0, transient=70.0, n=200001)
    return {"f": f, "start": start, "oscillates": a.oscillates, "log_x_min": a.log_x_min,
            "log_x_max": a.log_x_max, "period": a.period}


def _large_cycle(p, u0):
    a = attractor(p, u0, t_end=100.0, transient=70.0, n=200001)
    return a.log_x_max - a.log_x_min > 1.0


def cycle_edge(inside: float, outside: float, tol: float = 1e-3) -> float:
    """Bisect for the last f where the relaxation cycle survives, starting on the cycle at f = inside."""
    u_cycle = simulate(SLIDE_PARAMS.with_f(inside), SLIDE_U0, 50.0, 3)[1][:, -1]
    while abs(outside - inside) > tol:
        mid = (inside + outside) / 2
        inside, outside = (mid, outside) if _large_cycle(SLIDE_PARAMS.with_f(mid), u_cycle) else (inside, mid)
    return (inside + outside) / 2


def hopf_summary(p: Params) -> list[dict]:
    rows = []
    for f_h in hopf_points(p):
        ph = p.with_f(f_h)
        rows.append({"f": f_h, "omega": hopf_frequency(ph), "linear_period": linear_period(ph),
                     "dRe_df": transversality(p, f_h), "minus_A": -char_coeffs(ph)[0]})
    return rows


def compute(ex, cache: dict | None = None) -> tuple[dict, list[dict]]:
    """Everything in summary.json. With a cache, the sweep, cycle edges and period are reused."""
    p = SLIDE_PARAMS
    hopf = hopf_summary(p)
    f_lo, f_hi = hopf[0]["f"], hopf[1]["f"]

    # The slides' interval 1/4 < f < 1/2 is the q -> 0 limit at fixed eps and delta.
    sens = {f"q={q:g}": [round(f, 4) for f in hopf_points(Params(p.eps, p.delta, q), f_max=3, n=20000)]
            for q in (8e-4, 8e-5, 8e-6, 8e-7, 8e-8, 8e-10)}

    if cache:
        sweep = cache["sweep"]
        cycle_lo, cycle_hi = cache["summary"]["relaxation_cycle_interval_from_cycle"]
        period_04 = cache["summary"]["period_at_f_0.4"]
    else:
        sweep = list(ex.map(_sweep_job, [(f, s) for s in ("slides' u0", "near E2*") for f in SWEEP_F]))
        cycle_hi, cycle_lo = ex.map(cycle_edge, [1.16, 0.26], [1.4, 0.2])
        period_04 = attractor(p.with_f(0.4), SLIDE_U0, 100.0, 70.0, 200001).period
    figure_fit = fit_original_figures(ex)

    e2 = {str(f): equilibrium(p.with_f(f)).tolist() for f in (0.1, 0.4, 1.0, 5.0)}
    summary = {
        "params": vars(p) | {"u0": list(SLIDE_U0)},
        "claimed_in_slides": CLAIMED,
        "q_from_rate_constants": from_rate_constants(**SLIDE_RATES, A=1.0, B=1.0, f=0.0).q,
        "E1_char_coeffs_A_B_C": list(char_coeffs_origin(p)),
        "E2": e2,
        "E2_char_coeffs_at_f_0.4": list(char_coeffs(p.with_f(0.4))),
        "E2_eigenvalues": {str(f): [complex(v).__repr__() for v in eigenvalues(p.with_f(f))]
                           for f in (0.1, 0.4, 1.0, 5.0)},
        "hopf": hopf,
        "unstable_interval": [f_lo, f_hi],
        "relaxation_cycle_interval_from_cycle": [cycle_lo, cycle_hi],
        "hopf_interval_sensitivity": sens,
        "period_at_f_0.4": period_04,
        "original_figures": figure_fit,
    }
    return summary, sweep


# --------------------------------------------------------------------- figures

def _log_series(ax, f, t_max, n=6001):
    t, u = simulate(SLIDE_PARAMS.with_f(f), SLIDE_U0, t_max, n)
    for i, k in enumerate("xyz"):
        ax.plot(t, np.log10(u[i]), DASHES[k], color=COLORS[k], label=LABELS[k])
    ax.set_xlim(0, t_max)
    ax.set_ylim(-7, 3)
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(r"$\log_{10}$ konsentrasi (tak berdimensi)")


def fig_timeseries(f, t_max, name):
    fig, ax = plt.subplots(figsize=(6, 4.2))
    _log_series(ax, f, t_max)
    ax.set_title(f"$f = {f:g}$")
    ax.legend(loc="lower right", ncol=3)
    _save(fig, name)


def fig_original_check(figure_fit):
    curves = digitize_all(PIC_DIR)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.4), sharey=True)
    for col, (name, spec) in enumerate(FIGURES.items()):
        fit = figure_fit[name]
        for row, f in enumerate((fit["f_label"], fit["f_best"])):
            ax = axes[row, col]
            for k, pts in curves[name].items():
                ax.plot(pts[:, 0], pts[:, 1], ".", ms=2.5, color=MUTED, zorder=1,
                        label="gambar asli (didigitasi)" if k == "x" else None)
            _log_series(ax, f, spec["t_max"])
            err = fit["error_at_label"] if row == 0 else fit["error_at_best"]
            kind = "label di slide" if row == 0 else "paling cocok"
            ax.set_title(f"{name} vs simulasi $f = {f:g}$ ({kind}); galat median {err:.3f}")
            if row == 0:
                ax.set_xlabel("")
            if col == 1:
                ax.set_ylabel("")
    axes[0, 0].legend(loc="lower right", ncol=2)
    fig.tight_layout()
    _save(fig, "original_vs_simulation")


def fig_stability(summary):
    fs = np.linspace(0.01, 2.0, 2000)
    re = np.array([eigenvalues(SLIDE_PARAMS.with_f(f)).real.max() for f in fs])
    f_lo, f_hi = summary["unstable_interval"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axvspan(f_lo, f_hi, color=COLORS["x"], alpha=0.10, lw=0, label=f"tidak stabil: {f_lo:.4f} < $f$ < {f_hi:.4f}")
    ax.axvspan(*CLAIMED["unstable_interval"], facecolor="none", edgecolor=INK_2, hatch="///", lw=0,
               label=r"klaim slide: $\frac{1}{4} < f < \frac{1}{2}$")
    ax.plot(fs, re, color=COLORS["x"])
    ax.axhline(0, color=INK_2, lw=0.9)
    ax.set_yscale("symlog", linthresh=1)
    ax.set_xlim(0, 2)
    ax.set_xlabel("$f$")
    ax.set_ylabel(r"$\max\ \mathrm{Re}\,\lambda$ di $E_2^*$")
    for f_h in (f_lo, f_hi):
        ax.axvline(f_h, color=INK_2, lw=0.9, ls=":")
    ax.legend(loc="upper right")
    _save(fig, "stability")


def fig_bifurcation(summary, sweep):
    f_lo, f_hi = summary["unstable_interval"]
    c_lo, c_hi = summary["relaxation_cycle_interval_from_cycle"]
    fs = np.linspace(0.05, 1.5, 1500)
    log_xs = np.log10([equilibrium(SLIDE_PARAMS.with_f(f))[0] for f in fs])
    unstable = (fs > f_lo) & (fs < f_hi)
    fig, (ax, axp) = plt.subplots(2, 1, figsize=(7, 6.4), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    if c_hi > f_hi:
        ax.axvspan(f_hi, c_hi, color=COLORS["y"], alpha=0.10, lw=0, label="bistabil (siklus + $E_2^*$ stabil)")
    ax.plot(np.where(unstable, np.nan, fs), log_xs, color=INK, label="$E_2^*$ stabil")
    ax.plot(np.where(unstable, fs, np.nan), log_xs, color=INK, ls="--", label="$E_2^*$ tidak stabil")
    styles = {"slides' u0": dict(marker="o", color=COLORS["x"], mfc=COLORS["x"], label="siklus, $u_0$ slide"),
              "near E2*": dict(marker="o", color=COLORS["y"], mfc="none", label=r"siklus, $u_0 \approx E_2^*$")}
    for start, st in styles.items():
        rows = [r for r in sweep if r["start"] == start and r["oscillates"]]
        f = [r["f"] for r in rows]
        ax.plot(f, [r["log_x_max"] for r in rows], ls="none", ms=5 if start == "near E2*" else 3.5, **st)
        st = st | {"label": None}
        ax.plot(f, [r["log_x_min"] for r in rows], ls="none", ms=5 if start == "near E2*" else 3.5, **st)
        if start == "slides' u0":
            axp.plot(f, [r["period"] for r in rows], ls="none", ms=3.5, **st)
    for a in (ax, axp):
        for f_h in (f_lo, f_hi):
            a.axvline(f_h, color=INK_2, lw=0.9, ls=":")
    ax.set_ylabel(r"$\log_{10} x$ (min / maks)")
    ax.legend(loc="center right", fontsize=8.5)
    axp.set_ylabel(r"periode ($\tau$)")
    axp.set_xlabel("$f$")
    axp.set_xlim(0.05, 1.5)
    _save(fig, "bifurcation")


def write_sweep(sweep):
    with open(RES_DIR / "sweep.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(sweep[0]))
        w.writeheader()
        w.writerows(sweep)


def read_cache() -> dict:
    with open(RES_DIR / "sweep.csv") as fh:
        sweep = [{"f": float(r["f"]), "start": r["start"], "oscillates": r["oscillates"] == "True",
                  "log_x_min": float(r["log_x_min"]), "log_x_max": float(r["log_x_max"]),
                  "period": float(r["period"])} for r in csv.DictReader(fh)]
    return {"sweep": sweep, "summary": json.loads((RES_DIR / "summary.json").read_text())}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reuse", action="store_true",
                        help="reuse the sweep, cycle edges and period in results/ (the slow part)")
    args = parser.parse_args()
    FIG_DIR.mkdir(exist_ok=True)
    RES_DIR.mkdir(exist_ok=True)
    _style()
    with ProcessPoolExecutor() as ex:
        summary, sweep = compute(ex, read_cache() if args.reuse else None)
    (RES_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_sweep(sweep)
    fig_timeseries(0.1, 7, "timeseries_f0.1")
    fig_timeseries(0.4, 10, "timeseries_f0.4")
    fig_timeseries(1.0, 10, "timeseries_f1.0")
    fig_original_check(summary["original_figures"])
    fig_stability(summary)
    fig_bifurcation(summary, sweep)
    print(json.dumps({k: summary[k] for k in ("unstable_interval", "relaxation_cycle_interval_from_cycle",
                                              "period_at_f_0.4", "hopf_interval_sensitivity")}, indent=2))
    for name, fit in summary["original_figures"].items():
        print(name, {k: v for k, v in fit.items() if k != "errors"})


if __name__ == "__main__":
    main()
