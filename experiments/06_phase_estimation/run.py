"""Experiment 06: advantage on phase estimation, against the same matched baseline.

The second operational task of RESEARCH_PLAN.md section 9, run on the *same* resources
and the *same* baseline construction as experiment 05, so that the two tasks can be
compared directly.

Task: estimate a phase ``theta`` imprinted by ``U(theta) = exp(-i theta n)``, scored by
the classical Fisher information under homodyne detection. Probe and baseline both pass
through the same loss channel.

The question this answers is whether the decoupling found in experiment 05 -- resource
survival collapsing at a different point from task usefulness -- is a property of that
one task or of the resources. It is not a property of the task: the ordering of the
resources changes completely, and a resource that wins decisively at state preparation
is useless here.

Two structural facts, both verified numerically in the test suite, make the comparison
clean:

- **A Fock state is invariant under phase rotation.** ``exp(-i theta n)|n><n|exp(i theta
  n) = |n><n|`` exactly, so a Fock probe carries no phase information under any
  measurement and its Fisher information is identically zero. Fock |1>, the strongest
  resource in experiment 05, is therefore the weakest possible probe here.
- **The Gaussian family is closed under phase rotation**, since
  ``U(theta) D(alpha) S(r, phi) = D(alpha e^{-i theta}) S(r, phi - 2 theta)``. Optimizing
  over the family at fixed ``theta`` therefore covers every orientation, so the baseline
  is evaluated at ``theta = 0`` without loss of generality. The non-Gaussian probes are
  fixed states and are scanned over ``theta`` explicitly.

Phase rotation also commutes exactly with pure loss, so the order of the two is
immaterial.

Run:
    python experiments/06_phase_estimation/run.py
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
    apply_unitary,
    cat_ket,
    fock_ket,
    mean_photon_number,
    to_dm,
)
from ngphotonic.metrics.homodyne import (  # noqa: E402
    homodyne_fisher_information,
    phase_rotation,
    quadrature_grid,
    quantum_fisher_information_pure,
)
from ngphotonic.noise.loss import apply_loss  # noqa: E402
from ngphotonic.optimization.gaussian_baseline import (  # noqa: E402
    default_shell_starts,
    optimize_gaussian_baseline,
)

# Cutoff 60, not 30: the baseline optimizer drives this task to high squeezing
# (r ~ 1.2), where cutoff 30 truncates and *underestimates* F_G* by up to 20%.
# Re-optimizing at increasing cutoff gave 54.27 (30) -> 66.54 (50) -> 67.96 (80) at
# n_bar = 2.2, so 60 sits inside the converged region.
CUTOFF = 60
QUAD_GRID = dict(limit=8.0, points=801)
ETAS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
THETA_SCAN = np.linspace(0.0, np.pi, 33)

OUT_DIR = ROOT / "results" / "06_phase_estimation"
FIGURE_PATH = ROOT / "figures" / "06_phase_estimation.png"


def probes(cutoff: int = CUTOFF) -> dict[str, np.ndarray]:
    """The same resources as experiment 05, so the two tasks are comparable."""
    return {
        "Fock |1>": fock_ket(1, cutoff),
        "Fock |2>": fock_ket(2, cutoff),
        "cat a=1.5": cat_ket(1.5, "even", cutoff),
    }


def fisher_of_state(rho0: np.ndarray, theta: float, x: np.ndarray) -> float:
    cutoff = rho0.shape[0]
    return homodyne_fisher_information(
        lambda th: apply_unitary(rho0, phase_rotation(th, cutoff)), theta, x
    )


def best_fisher_over_theta(rho0: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    """Scan ``theta`` for a fixed probe and return the best value and where it occurs."""
    values = [fisher_of_state(rho0, float(theta), x) for theta in THETA_SCAN]
    index = int(np.argmax(values))
    return float(values[index]), float(THETA_SCAN[index])


def sweep_probe(name: str, ket: np.ndarray, x: np.ndarray) -> list[dict]:
    budget = mean_photon_number(to_dm(ket))
    rows = []
    warm = None

    for eta in ETAS:
        noisy = apply_loss(to_dm(ket), eta)
        f_ng, theta_star = best_fisher_over_theta(noisy, x)

        # Shell coordinates: n_bar equals the budget identically. The free
        # parameterization is unsafe here, since Fisher information grows
        # monotonically with energy and the optimizer will pay the soft penalty to
        # overspend the budget by ~15%, breaking the matching.
        starts = list(default_shell_starts(extra=3))
        if warm is not None:
            starts.insert(0, warm)

        # The family is closed under phase rotation, so evaluating at theta = 0 loses
        # no generality: the optimizer reaches every orientation through alpha and phi.
        baseline = optimize_gaussian_baseline(
            score=lambda rho, e=eta: fisher_of_state(apply_loss(rho, e), 0.0, x),
            cutoff=CUTOFF,
            n_budget=budget,
            starts=starts,
        )
        warm = None  # shell coordinates are not the params vector; restart fresh

        rows.append(
            {
                "eta": eta,
                "fisher_ng": f_ng,
                "fisher_gaussian": baseline.score,
                "advantage": f_ng - baseline.score,
                "theta_star_ng": theta_star,
                "baseline_params": list(baseline.params.to_vector()),
                "baseline_alpha_abs": float(abs(baseline.params.alpha)),
                "baseline_r": baseline.params.r,
                "baseline_n_bar": baseline.mean_photon_number,
                "n_budget": budget,
            }
        )
        print(f"    eta={eta:4.2f}  F_NG={f_ng:9.4f}  F_G*={baseline.score:9.4f}  "
              f"A={f_ng - baseline.score:+10.4f}  (baseline r={baseline.params.r:.3f})")
    return rows


def make_figure(sweeps: dict, prep: dict | None, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = 3 if prep else 2
    fig, axes = plt.subplots(1, panels, figsize=(5.5 * panels, 4.8))
    colors = plt.cm.viridis(np.linspace(0, 0.75, len(sweeps)))

    ax = axes[0]
    for (name, rows), color in zip(sweeps.items(), colors):
        etas = [r["eta"] for r in rows]
        ax.plot(etas, [r["fisher_ng"] for r in rows], "-o", ms=3, color=color,
                label=f"{name} — probe")
        ax.plot(etas, [r["fisher_gaussian"] for r in rows], "--s", ms=3, color=color,
                alpha=0.65, label=f"{name} — Gaussian*")
    ax.set_xlabel(r"transmissivity $\eta$")
    ax.set_ylabel("Fisher information")
    ax.set_title("Phase estimation:\nprobe vs optimized matched Gaussian")
    ax.legend(fontsize=6.5, frameon=False)

    ax = axes[1]
    for (name, rows), color in zip(sweeps.items(), colors):
        ax.plot([r["eta"] for r in rows], [r["advantage"] for r in rows], "-o", ms=3,
                color=color, label=name)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlabel(r"transmissivity $\eta$")
    ax.set_ylabel(r"$A_F = F_{\rm NG} - F^*_{\rm G}$")
    ax.set_title("Advantage is negative everywhere\n(the Gaussian baseline wins this task)")
    ax.legend(fontsize=7, frameon=False)

    if prep:
        ax = axes[2]
        width = 0.35
        names = list(sweeps)
        positions = np.arange(len(names))
        prep_vals = [prep[n] for n in names]
        phase_vals = [sweeps[n][0]["advantage"] for n in names]
        ax.barh(positions - width / 2, prep_vals, width, label="preparation ($A$)",
                color="C0")
        ax.barh(positions + width / 2, phase_vals, width, label="phase est. ($A_F$)",
                color="C3")
        ax.axvline(0.0, color="k", lw=0.8)
        ax.set_yticks(positions)
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xscale("symlog", linthresh=0.1)
        ax.set_xlabel("advantage at $\\eta = 1$ (symlog)")
        ax.set_title("Same resources, opposite verdicts")
        ax.legend(fontsize=7.5, frameon=False)

    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle("Experiment 06: phase estimation against the same matched Gaussian baseline",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def load_preparation_advantages() -> dict | None:
    """Advantage at eta = 1 from experiment 05, for the side-by-side comparison."""
    path = ROOT / "results" / "05_gaussian_baseline" / "gaussian_baseline_report.json"
    if not path.exists():
        return None
    report = json.loads(path.read_text())
    out = {}
    for name, rows in report["sweeps"].items():
        lossless = [r for r in rows if r["eta"] == 1.0]
        if lossless:
            out[name] = lossless[0]["advantage"]
    return out or None


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    x = quadrature_grid(**QUAD_GRID)

    sweeps = {}
    qfi = {}
    for name, ket in probes().items():
        print(f"  {name} ...")
        qfi[name] = quantum_fisher_information_pure(ket)
        sweeps[name] = sweep_probe(name, ket, x)

    prep = load_preparation_advantages()

    report = {
        "configuration": {
            "cutoff": CUTOFF,
            "quadrature_grid": QUAD_GRID,
            "task": "phase estimation, classical Fisher information under homodyne",
            "baseline_family": "displaced squeezed vacuum, energy-matched",
        },
        "quantum_fisher_information_lossless": qfi,
        "sweeps": sweeps,
        "preparation_advantage_at_eta_1": prep,
        "runtime_seconds": time.perf_counter() - started,
    }
    (OUT_DIR / "phase_estimation_report.json").write_text(json.dumps(report, indent=2))
    make_figure(sweeps, prep, FIGURE_PATH)

    print("\n" + "=" * 78)
    print("EXPERIMENT 06: phase estimation vs the matched Gaussian baseline")
    print("=" * 78)
    print(f"  {'resource':12s} {'QFI(eta=1)':>11} {'F_NG(1)':>9} {'F_G*(1)':>9} "
          f"{'A_F(1)':>10} {'A_prep(1)':>10}")
    for name, rows in sweeps.items():
        first = rows[0]
        prep_value = prep.get(name) if prep else float("nan")
        print(f"  {name:12s} {qfi[name]:11.4f} {first['fisher_ng']:9.4f} "
              f"{first['fisher_gaussian']:9.4f} {first['advantage']:+10.4f} "
              f"{prep_value:+10.4f}")
    worst = min(min(r["advantage"] for r in rows) for rows in sweeps.values())
    any_positive = any(r["advantage"] > 0 for rows in sweeps.values() for r in rows)
    print(f"\n  advantage positive anywhere: {any_positive}")
    print(f"  most negative advantage    : {worst:+.4f}")
    print(f"  runtime {report['runtime_seconds']:.0f}s")
    print("=" * 78)
    print(f"Wrote {OUT_DIR} and {FIGURE_PATH.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
