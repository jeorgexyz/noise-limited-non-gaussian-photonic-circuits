"""Experiment 01: non-Gaussian resource decay under pure loss.

RESEARCH_PLAN.md programme step 2. For each resource state, sweep transmissivity and
locate the critical ``eta*`` at which Wigner logarithmic negativity falls to a survival
threshold ``epsilon``.

Three things are reported that a bare threshold number would hide:

1. **eta* depends on epsilon**, and not weakly. The collapse point is a property of the
   resource together with the amount of surviving negativity required, not of the
   resource alone. The sweep over epsilon is reported so that the dependence is
   explicit rather than fixed by an undocumented constant.

2. **Each eta* carries a bracket**, set by the metric's noise floor via
   ``resolution_floor``. Additional digits beyond that width resolve quadrature error.

3. **Cutoff convergence per resource**. The cat states in particular require enough
   Fock headroom that truncation does not contribute to the measured negativity.

The single photon is included as a control: its threshold is known analytically to be
exactly ``eta = 1/2`` in the limit ``epsilon -> 0``.

Run:
    python experiments/01_loss_threshold/run.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ngphotonic.analysis.thresholds import critical_parameter, resolution_floor  # noqa: E402
from ngphotonic.backends.reference import (  # noqa: E402
    apply_unitary,
    cat_ket,
    fock_dm,
    kerr_unitary,
    squeezed_ket,
    tail_weight,
    to_dm,
)
from ngphotonic.metrics.negativity import wigner_log_negativity  # noqa: E402
from ngphotonic.metrics.wigner import phase_space_grid, wigner  # noqa: E402
from ngphotonic.noise.loss import apply_loss  # noqa: E402

# Cutoff 30 / 201 points chosen by measurement, not caution: eta* agrees to five
# decimals with cutoff 40 and 301 points, while running ~12x faster. The W_log
# quadrature error at this grid is 1.3e-04, which sets METRIC_NOISE below.
CUTOFF = 30
GRID = dict(limit=6.0, points=201)
EPSILON = 1e-3
EPSILONS = [1e-4, 1e-3, 1e-2, 5e-2, 1e-1]
METRIC_NOISE = 1e-4  # quadrature floor on W_log for this grid

OUT_DIR = ROOT / "results" / "01_loss_threshold"
FIGURE_PATH = ROOT / "figures" / "01_loss_threshold.png"


def resources(cutoff: int = CUTOFF) -> dict[str, np.ndarray]:
    """The non-Gaussian resources under test, all at the same cutoff."""
    return {
        "Fock |1>": fock_dm(1, cutoff),
        "Fock |2>": fock_dm(2, cutoff),
        "Fock |3>": fock_dm(3, cutoff),
        "cat a=1.0": to_dm(cat_ket(1.0, "even", cutoff)),
        "cat a=1.5": to_dm(cat_ket(1.5, "even", cutoff)),
        "cat a=2.0": to_dm(cat_ket(2.0, "even", cutoff)),
        "Kerr(0.6) sq(0.6)": apply_unitary(
            to_dm(squeezed_ket(0.6, 0.0, cutoff)), kerr_unitary(0.6, cutoff)
        ),
    }


def _grid():
    return phase_space_grid(**GRID)


def negativity_after_loss(rho: np.ndarray, eta: float, grid) -> float:
    x, p, X, P = grid
    return wigner_log_negativity(wigner(apply_loss(rho, eta), X, P), x, p)


def sweep_transmissivity(rho: np.ndarray, grid, points: int = 81) -> list[dict]:
    etas = np.linspace(0.05, 1.0, points)
    return [
        {"eta": float(eta), "w_log": negativity_after_loss(rho, float(eta), grid)}
        for eta in etas
    ]


def locate_eta_star(rho: np.ndarray, grid, epsilon: float) -> dict:
    """Bisect for eta*, and report how finely it can honestly be located."""
    threshold = critical_parameter(
        metric=lambda eta: negativity_after_loss(rho, eta, grid),
        lower=0.05,
        upper=1.0,
        epsilon=epsilon,
        iterations=14,
    )
    record = {
        "epsilon": epsilon,
        "eta_star": threshold.estimate,
        "bracket_lower": threshold.lower,
        "bracket_upper": threshold.upper,
        "bisection_half_width": threshold.half_width,
        "resolved": threshold.resolved,
    }
    if threshold.resolved:
        record["resolution_floor"] = resolution_floor(
            metric=lambda eta: negativity_after_loss(rho, eta, grid),
            point=threshold.estimate,
            delta=0.01,
            noise=METRIC_NOISE,
        )
    return record


def cutoff_convergence(name: str, grid) -> list[dict]:
    """Check that each resource's threshold is not an artefact of truncation."""
    out = []
    for cutoff in (20, 30, 40, 50):
        rho = resources(cutoff)[name]
        out.append(
            {
                "cutoff": cutoff,
                "tail_weight": tail_weight(rho),
                "w_log_lossless": negativity_after_loss(rho, 1.0, grid),
                "eta_star": locate_eta_star(rho, grid, EPSILON)["eta_star"],
            }
        )
    return out


def make_figure(sweeps: dict, thresholds: dict, epsilon_scan: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(sweeps)))

    ax = axes[0]
    for (name, rows), color in zip(sweeps.items(), colors):
        ax.plot([r["eta"] for r in rows], [r["w_log"] for r in rows], label=name, color=color)
    ax.axhline(EPSILON, ls=":", color="k", lw=1)
    ax.axvline(0.5, ls="--", color="C3", lw=1)
    ax.set_xlabel(r"transmissivity $\eta$")
    ax.set_ylabel(r"$W_{\log}$")
    ax.set_title("Resource decay under pure loss\n(dashed: analytic Fock-1 threshold)")
    ax.legend(fontsize=7, frameon=False)

    ax = axes[1]
    names = list(thresholds)
    values = [thresholds[n]["eta_star"] for n in names]
    errors = [max(thresholds[n]["bisection_half_width"],
                  thresholds[n].get("resolution_floor", 0.0)) for n in names]
    ax.barh(range(len(names)), values, xerr=errors, color=colors, capsize=3)
    ax.axvline(0.5, ls="--", color="C3", lw=1)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel(r"$\eta^*$")
    ax.set_xlim(0.4, 0.75)
    ax.set_title(rf"Critical transmissivity ($\epsilon$ = {EPSILON:g})" "\nbars: honest resolution")

    ax = axes[2]
    for (name, rows), color in zip(epsilon_scan.items(), colors):
        good = [r for r in rows if r["resolved"]]
        ax.semilogx([r["epsilon"] for r in good], [r["eta_star"] for r in good],
                    "-o", ms=3, label=name, color=color)
    ax.set_xlabel(r"survival threshold $\epsilon$")
    ax.set_ylabel(r"$\eta^*$")
    ax.set_title(r"$\eta^*$ is a function of $\epsilon$" "\nnot of the resource alone")

    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle("Experiment 01: non-Gaussian resource decay under pure loss", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    grid = _grid()
    states = resources()

    print("Sweeping transmissivity ...")
    sweeps = {name: sweep_transmissivity(rho, grid) for name, rho in states.items()}

    print("Locating eta* ...")
    thresholds = {name: locate_eta_star(rho, grid, EPSILON) for name, rho in states.items()}

    print("Scanning epsilon dependence ...")
    epsilon_scan = {
        name: [locate_eta_star(rho, grid, eps) for eps in EPSILONS]
        for name, rho in states.items()
    }

    print("Checking cutoff convergence ...")
    convergence = {name: cutoff_convergence(name, grid) for name in states}

    report = {
        "configuration": {
            "cutoff": CUTOFF,
            "grid": GRID,
            "epsilon": EPSILON,
            "metric_noise_floor": METRIC_NOISE,
        },
        "sweeps": sweeps,
        "thresholds": thresholds,
        "epsilon_scan": epsilon_scan,
        "cutoff_convergence": convergence,
        "runtime_seconds": time.perf_counter() - started,
    }
    (OUT_DIR / "loss_threshold_report.json").write_text(json.dumps(report, indent=2))
    make_figure(sweeps, thresholds, epsilon_scan, FIGURE_PATH)

    control = thresholds["Fock |1>"]
    print("\n" + "=" * 74)
    print("EXPERIMENT 01: resource decay under pure loss")
    print("=" * 74)
    print(f"  {'resource':20s} {'W_log(eta=1)':>13} {'eta*':>8} {'+/-':>8} {'tail':>9}")
    for name, rho in states.items():
        t = thresholds[name]
        width = max(t["bisection_half_width"], t.get("resolution_floor", 0.0))
        print(f"  {name:20s} {sweeps[name][-1]['w_log']:13.4f} "
              f"{t['eta_star']:8.4f} {width:8.4f} {tail_weight(rho):9.1e}")

    print("\n  control: Fock |1> analytic threshold is eta = 0.5 as epsilon -> 0")
    print(f"           measured at epsilon = {EPSILON:g}: eta* = {control['eta_star']:.4f}")
    span = [epsilon_scan['Fock |1>'][i]['eta_star'] for i in (0, -1)]
    print(f"           and it moves {span[0]:.3f} -> {span[1]:.3f} across the epsilon scan,")
    print("           which is why eta* is reported with epsilon, never alone.")
    print(f"\n  runtime {report['runtime_seconds']:.1f}s")
    print("=" * 74)
    print(f"Wrote {OUT_DIR} and {FIGURE_PATH.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
