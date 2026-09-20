"""Validation experiment: simulated vs. analytic Wigner negativity under pure loss.

Reproduces, from a mixed-state Fock simulation, the closed form

    int |W| = 4 eta exp(-(1 - 1/(2 eta))) - 1     for eta >= 1/2
    int |W| = 1                                    for eta <= 1/2

for a single photon after a pure-loss channel of transmissivity eta, and with it the
sharp threshold: Wigner negativity of a lossy single photon vanishes at exactly
eta = 1/2.

Run:
    python experiments/00_validation/run_validation.py

Writes results/00_validation/ (JSON metrics + figure). That directory is gitignored;
the run is reproducible from this script.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ngphotonic.backends.reference import (  # noqa: E402
    fock_dm,
    mean_photon_number,
    purity,
    to_dm,
    vacuum_dm,
)
from ngphotonic.metrics.negativity import (  # noqa: E402
    analytic_lossy_fock1_abs_integral,
    grid_diagnostics,
    integrate_abs_wigner,
    negative_volume,
    wigner_log_negativity,
)
from ngphotonic.metrics.wigner import phase_space_grid, wigner  # noqa: E402
from ngphotonic.noise.loss import apply_loss  # noqa: E402

CUTOFF = 30
GRID_LIMIT = 6.0
GRID_POINTS = 401
OUT_DIR = ROOT / "results" / "00_validation"
# The headline figure is a tracked deliverable; the JSON dump is not.
FIGURE_PATH = ROOT / "figures" / "00_validation_loss_threshold.png"


def loss_sweep() -> dict:
    """Sweep transmissivity and compare simulated negativity to the closed form."""
    x, p, X, P = phase_space_grid(limit=GRID_LIMIT, points=GRID_POINTS)
    etas = np.linspace(0.0, 1.0, 51)

    rows = []
    for eta in etas:
        rho = apply_loss(fock_dm(1, CUTOFF), float(eta))
        W = wigner(rho, X, P)
        simulated = integrate_abs_wigner(W, x, p)
        analytic = analytic_lossy_fock1_abs_integral(float(eta))
        rows.append(
            {
                "eta": float(eta),
                "abs_integral_simulated": simulated,
                "abs_integral_analytic": analytic,
                "abs_error": abs(simulated - analytic),
                "w_log": wigner_log_negativity(W, x, p),
                "negative_volume": negative_volume(W, x, p),
                "purity": purity(rho),
                "mean_photon_number": mean_photon_number(rho),
            }
        )
    return {"rows": rows, "max_abs_error": max(r["abs_error"] for r in rows)}


def cutoff_convergence() -> list[dict]:
    """Cutoff convergence |M_{c+dc} - M_c| for the headline metric at eta = 0.8."""
    x, p, X, P = phase_space_grid(limit=GRID_LIMIT, points=GRID_POINTS)
    target = analytic_lossy_fock1_abs_integral(0.8)

    out, previous = [], None
    for cutoff in (5, 10, 15, 20, 30, 40):
        value = integrate_abs_wigner(wigner(apply_loss(fock_dm(1, cutoff), 0.8), X, P), x, p)
        out.append(
            {
                "cutoff": cutoff,
                "abs_integral": value,
                "error_vs_analytic": abs(value - target),
                "delta_vs_previous": None if previous is None else abs(value - previous),
            }
        )
        previous = value
    return out


def grid_convergence() -> list[dict]:
    """Quadrature convergence for the single photon as the grid is refined."""
    target = 4.0 * np.exp(-0.5) - 1.0
    out = []
    for limit in (4.0, 5.0, 6.0, 7.0):
        for points in (201, 401, 801):
            x, p, X, P = phase_space_grid(limit=limit, points=points)
            W = wigner(fock_dm(1, CUTOFF), X, P)
            diag = grid_diagnostics(W, x, p)
            out.append(
                {
                    "limit": limit,
                    "points": points,
                    "abs_integral": integrate_abs_wigner(W, x, p),
                    "error": abs(integrate_abs_wigner(W, x, p) - target),
                    "norm_error": diag["norm_error"],
                    "edge_mass": diag["edge_mass"],
                }
            )
    return out


def locate_threshold(tolerance: float = 1e-4) -> dict:
    """Bisect for the transmissivity at which Wigner negativity switches on.

    Reported separately from the sweep because the sweep's 0.02 spacing cannot resolve
    a threshold: it can only say which sampled point first shows negativity. Bisection
    gives a real bracket.

    The resolution floor is set by quadrature error. Near eta = 1/2 the negative lobe
    is shallow -- W_log ~ 1e-4 at eta = 0.505 -- so a threshold located to better than
    about +/-0.005 would be reporting numerical noise, and ``tolerance`` is chosen to
    stay above it.
    """
    x, p, X, P = phase_space_grid(limit=GRID_LIMIT, points=GRID_POINTS)

    def is_negative(eta: float) -> bool:
        rho = apply_loss(fock_dm(1, CUTOFF), eta)
        return wigner_log_negativity(wigner(rho, X, P), x, p) > tolerance

    low, high = 0.4, 0.7  # known negative-free / known negative
    assert not is_negative(low) and is_negative(high)
    for _ in range(24):
        mid = 0.5 * (low + high)
        if is_negative(mid):
            high = mid
        else:
            low = mid
    return {
        "threshold_lower": low,
        "threshold_upper": high,
        "threshold_estimate": 0.5 * (low + high),
        "analytic": 0.5,
        "tolerance_on_w_log": tolerance,
        "error_vs_analytic": abs(0.5 * (low + high) - 0.5),
    }


def gaussian_null_check() -> list[dict]:
    """Gaussian states must report zero negativity, or every later advantage is noise."""
    from ngphotonic.backends.reference import coherent_ket, squeezed_ket, thermal_dm

    x, p, X, P = phase_space_grid(limit=GRID_LIMIT, points=GRID_POINTS)
    states = {
        "vacuum": vacuum_dm(CUTOFF),
        "coherent_1.2": to_dm(coherent_ket(1.2, CUTOFF)),
        "squeezed_r0.6_cutoff60": to_dm(squeezed_ket(0.6, 0.0, 60)),
        "thermal_nbar0.9": thermal_dm(0.9, CUTOFF),
    }
    return [
        {"state": name, "w_log": wigner_log_negativity(wigner(rho, X, P), x, p)}
        for name, rho in states.items()
    ]


def make_figure(sweep: dict, convergence: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = sweep["rows"]
    etas = [r["eta"] for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    ax = axes[0]
    ax.plot(etas, [r["abs_integral_analytic"] for r in rows], "-", lw=3, alpha=0.35,
            color="k", label="analytic")
    ax.plot(etas, [r["abs_integral_simulated"] for r in rows], "o", ms=3.5,
            color="C0", label="simulated")
    ax.axvline(0.5, ls="--", color="C3", lw=1.2)
    ax.annotate(r"$\eta = 1/2$", xy=(0.5, 1.30), xytext=(0.17, 1.33),
                color="C3", fontsize=10,
                arrowprops=dict(arrowstyle="->", color="C3", lw=1.0))
    ax.set_xlabel(r"transmissivity $\eta$")
    ax.set_ylabel(r"$\int |W|\,dx\,dp$")
    ax.set_title("Lossy single photon:\nsimulation reproduces closed form")
    ax.legend(frameon=False)

    ax = axes[1]
    ax.semilogy(etas, [max(r["abs_error"], 1e-18) for r in rows], "-o", ms=3, color="C2")
    ax.set_xlabel(r"transmissivity $\eta$")
    ax.set_ylabel("|simulated - analytic|")
    ax.set_title(f"Residual (max {sweep['max_abs_error']:.1e})\n"
                 "exact below the threshold")

    ax = axes[2]
    ax.semilogy([c["cutoff"] for c in convergence],
                [max(c["error_vs_analytic"], 1e-18) for c in convergence], "-o", color="C4")
    ax.set_xlabel("Fock cutoff")
    ax.set_ylabel(r"error at $\eta = 0.8$")
    ax.set_title("Cutoff convergence\n(a photon needs only 2 levels)")

    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle("V2 validation: simulated Wigner negativity under pure loss", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    print("Running loss sweep ...")
    sweep = loss_sweep()
    print("Running cutoff convergence ...")
    convergence = cutoff_convergence()
    print("Running grid convergence ...")
    grids = grid_convergence()
    print("Running Gaussian null check ...")
    nulls = gaussian_null_check()

    print("Bisecting for the negativity threshold ...")
    threshold = locate_threshold()
    measured_threshold = threshold["threshold_estimate"]
    worst_gaussian = max(n["w_log"] for n in nulls)

    report = {
        "configuration": {
            "cutoff": CUTOFF,
            "grid_limit": GRID_LIMIT,
            "grid_points": GRID_POINTS,
            "backend": "reference (pure NumPy mixed-state Fock)",
        },
        "loss_sweep": sweep,
        "cutoff_convergence": convergence,
        "grid_convergence": grids,
        "gaussian_null_check": nulls,
        "threshold_bisection": threshold,
        "summary": {
            "max_abs_error_vs_analytic": sweep["max_abs_error"],
            "negativity_threshold_measured": measured_threshold,
            "negativity_threshold_error": threshold["error_vs_analytic"],
            "negativity_threshold_analytic": 0.5,
            "worst_gaussian_w_log": worst_gaussian,
            "runtime_seconds": time.perf_counter() - started,
        },
    }

    (OUT_DIR / "validation_report.json").write_text(json.dumps(report, indent=2))
    make_figure(sweep, convergence, OUT_DIR / "validation_loss_threshold.png")
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    make_figure(sweep, convergence, FIGURE_PATH)

    print("\n" + "=" * 68)
    print("V2 VALIDATION SUMMARY")
    print("=" * 68)
    print(f"  max |simulated - analytic|      {sweep['max_abs_error']:.3e}")
    print(f"  negativity threshold, measured  eta = {measured_threshold:.6f} "
          f"(error {threshold['error_vs_analytic']:.1e})")
    print("  negativity threshold, analytic  eta = 0.5")
    print(f"  worst Gaussian-state W_log      {worst_gaussian:.3e}  (must be ~0)")
    print(f"  runtime                         {report['summary']['runtime_seconds']:.1f}s")
    print("=" * 68)
    print(f"\nWrote {OUT_DIR}")

    ok = (
        sweep["max_abs_error"] < 1e-3
        and worst_gaussian < 1e-6
        and threshold["error_vs_analytic"] < 0.01
    )
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
