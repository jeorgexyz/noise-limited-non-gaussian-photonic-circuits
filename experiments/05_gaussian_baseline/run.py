"""Experiment 05: advantage over an optimized, resource-matched Gaussian baseline.

RESEARCH_PLAN.md programme step 8, and the first measurement of

    ``A(nu) = S_NG(nu) - S_G*(nu)``

where ``S_G*`` is the best score found over energy-matched Gaussian states subjected to
the same channel, rather than an arbitrary Gaussian comparison.

Task: target-state preparation, scored by fidelity to the target. The non-Gaussian
circuit prepares the target and passes it through loss. The Gaussian baseline is
optimized over the four-parameter displaced-squeezed family, constrained to the same
mean photon number, and passed through the same loss. Optimizing the *input* parameters
to maximise the *post-channel* score is the physically meaningful comparison: the
experimenter chooses what to send, not what arrives.

The result this experiment exists to test is whether resource survival and task
usefulness collapse at the same point. They do not. Wigner negativity of a lossy single
photon vanishes abruptly at eta = 1/2, while the task advantage decays smoothly and
remains positive well below it, so there is a regime with

    ``W_log = 0``  and  ``A > 0``

-- the resource is dead by the resource measure and still useful by the task measure.
RESEARCH_PLAN.md section 2 anticipated the opposite pairing (``W_log > 0, A <= 0``);
finding this direction as well means the two measures decouple both ways.

Interpretation caveat, stated because it bounds the claim: the target is a state the
non-Gaussian circuit holds exactly, so a positive ``A`` at high transmissivity is
expected by construction. What is not by construction is *how far down in eta* the
advantage persists, and how that compares with the negativity threshold. The ratio
``S_NG / S_G*`` is reported alongside ``A`` because the absolute difference necessarily
vanishes as both scores go to zero.

Run:
    python experiments/05_gaussian_baseline/run.py
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
    cat_ket,
    fock_dm,
    fock_ket,
    mean_photon_number,
    to_dm,
)
from ngphotonic.baselines.gaussian import is_gaussian_channel  # noqa: E402
from ngphotonic.metrics.negativity import wigner_log_negativity  # noqa: E402
from ngphotonic.metrics.operational import target_state_fidelity  # noqa: E402
from ngphotonic.metrics.wigner import phase_space_grid, wigner  # noqa: E402
from ngphotonic.noise.loss import apply_loss  # noqa: E402
from ngphotonic.analysis.thresholds import critical_parameter  # noqa: E402
from ngphotonic.optimization.gaussian_baseline import (  # noqa: E402
    default_shell_starts,
    optimize_gaussian_baseline,
)

CUTOFF = 30
GRID = dict(limit=6.0, points=201)
ETAS = [1.0, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1]
EPSILON = 0.01
SIGMA_PHI = 0.0  # pure loss only, so the baseline stays a Gaussian channel

OUT_DIR = ROOT / "results" / "05_gaussian_baseline"
FIGURE_PATH = ROOT / "figures" / "05_gaussian_baseline.png"


def targets(cutoff: int = CUTOFF) -> dict[str, np.ndarray]:
    return {
        "Fock |1>": fock_ket(1, cutoff),
        "Fock |2>": fock_ket(2, cutoff),
        "cat a=1.5": cat_ket(1.5, "even", cutoff),
    }


def non_gaussian_source(name: str, cutoff: int = CUTOFF) -> np.ndarray:
    """The non-Gaussian circuit prepares the target exactly, before noise."""
    if name == "Fock |1>":
        return fock_dm(1, cutoff)
    if name == "Fock |2>":
        return fock_dm(2, cutoff)
    if name == "cat a=1.5":
        return to_dm(cat_ket(1.5, cutoff=cutoff))
    raise ValueError(f"Unknown source {name!r}.")


def sweep_target(name: str, target: np.ndarray, grid) -> list[dict]:
    """Sweep transmissivity, optimizing the Gaussian baseline at each point.

    Each point is optimized from the same fixed start set rather than warm-started, so
    no point inherits a branch from its neighbour.
    """
    x, p, X, P = grid
    source = non_gaussian_source(name)
    budget = mean_photon_number(source)

    rows = []
    for eta in ETAS:
        noisy = apply_loss(source, eta)
        s_ng = target_state_fidelity(noisy, target)

        # Shell coordinates, so n_bar matches the budget identically. On this task the
        # free parameterization happened to respect the budget anyway and the two agree
        # to 1e-6, but the shell form removes the possibility.
        starts = list(default_shell_starts(extra=3))

        baseline = optimize_gaussian_baseline(
            score=lambda rho, e=eta: target_state_fidelity(apply_loss(rho, e), target),
            cutoff=CUTOFF,
            n_budget=budget,
            starts=starts,
        )
        rows.append(
            {
                "eta": eta,
                "s_ng": s_ng,
                "s_gaussian": baseline.score,
                "advantage": s_ng - baseline.score,
                "ratio": s_ng / baseline.score if baseline.score > 1e-12 else float("inf"),
                "w_log_ng": wigner_log_negativity(wigner(noisy, X, P), x, p),
                "baseline_alpha_abs": float(abs(baseline.params.alpha)),
                "baseline_r": baseline.params.r,
                "baseline_n_bar": baseline.mean_photon_number,
                "baseline_converged_fraction": baseline.converged_fraction,
                "n_budget": budget,
            }
        )
        print(f"    eta={eta:4.2f}  S_NG={s_ng:.4f}  S_G*={baseline.score:.4f}  "
              f"A={s_ng - baseline.score:+.4f}  W_log={rows[-1]['w_log_ng']:.4f}")
    return rows


def _bracket_crossing(rows: list[dict], key: str, level: float) -> tuple[float, float] | None:
    """Adjacent sweep points straddling ``level``, as a bracket on the crossing.

    Used for the advantage threshold, where each evaluation costs a full baseline
    optimization and bisecting to high precision is not worth the compute. Reporting
    the bracket keeps the resolution explicit instead of implying a bisected value.
    """
    ordered = sorted(rows, key=lambda r: r["eta"])
    for low, high in zip(ordered, ordered[1:]):
        if low[key] <= level < high[key]:
            return (low["eta"], high["eta"])
    return None


def decoupling_summary(name: str, rows: list[dict], target: np.ndarray, grid) -> dict:
    """Locate where negativity dies and where the task advantage does.

    The negativity threshold is bisected directly, since evaluating it costs only a
    Wigner transform. The advantage threshold is reported as the bracket between
    adjacent sweep points, because each evaluation there requires re-optimizing the
    Gaussian baseline. Neither is read off the sweep grid as though it were measured.
    """
    x, p, X, P = grid
    source = non_gaussian_source(name)

    negativity_threshold = critical_parameter(
        metric=lambda eta: wigner_log_negativity(
            wigner(apply_loss(source, eta), X, P), x, p
        ),
        lower=0.05,
        upper=1.0,
        epsilon=1e-9,
        iterations=16,
    )

    advantage_bracket = _bracket_crossing(rows, "advantage", EPSILON)
    # Where the optimized Gaussian baseline overtakes the non-Gaussian circuit. This
    # direction is robust: S_G* is a lower bound on the true Gaussian optimum, so a
    # measured A < 0 can only understate how far ahead the Gaussian actually is.
    parity_bracket = _bracket_crossing(rows, "advantage", 0.0)
    gaussian_wins = [r["eta"] for r in rows if r["advantage"] < 0.0]
    dead_but_useful = [
        r for r in rows
        if r["eta"] < negativity_threshold.estimate and r["advantage"] > EPSILON
    ]

    return {
        "negativity_threshold": negativity_threshold.estimate,
        "negativity_threshold_bracket": [negativity_threshold.lower, negativity_threshold.upper],
        "advantage_threshold_bracket": advantage_bracket,
        "parity_bracket": parity_bracket,
        "gaussian_wins_below": max(gaussian_wins) if gaussian_wins else None,
        "min_advantage": min(r["advantage"] for r in rows),
        "decoupled_etas": [r["eta"] for r in dead_but_useful],
        "max_advantage_without_negativity": max(
            (r["advantage"] for r in dead_but_useful), default=0.0
        ),
        "epsilon": EPSILON,
    }


def make_figure(sweeps: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))
    colors = plt.cm.viridis(np.linspace(0, 0.75, len(sweeps)))

    ax = axes[0]
    for (name, rows), color in zip(sweeps.items(), colors):
        etas = [r["eta"] for r in rows]
        ax.plot(etas, [r["s_ng"] for r in rows], "-o", ms=3, color=color, label=f"{name} — NG")
        ax.plot(etas, [r["s_gaussian"] for r in rows], "--s", ms=3, color=color, alpha=0.65,
                label=f"{name} — Gaussian*")
    ax.set_xlabel(r"transmissivity $\eta$")
    ax.set_ylabel("fidelity to target")
    ax.set_title("Task score:\nnon-Gaussian vs optimized matched Gaussian")
    ax.legend(fontsize=6.5, frameon=False)

    ax = axes[1]
    for (name, rows), color in zip(sweeps.items(), colors):
        ax.plot([r["eta"] for r in rows], [r["advantage"] for r in rows], "-o", ms=3,
                color=color, label=name)
    ax.axhline(EPSILON, ls=":", color="k", lw=1)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.axvline(0.5, ls="--", color="C3", lw=1)
    ax.set_xlabel(r"transmissivity $\eta$")
    ax.set_ylabel(r"$A = S_{\rm NG} - S^*_{\rm G}$")
    ax.set_title("Advantage decays smoothly\n(red: Fock-1 negativity threshold)")
    ax.legend(fontsize=7, frameon=False)

    ax = axes[2]
    rows = sweeps["Fock |1>"]
    etas = [r["eta"] for r in rows]
    ax.plot(etas, [r["w_log_ng"] for r in rows], "-o", ms=3, color="C3", label=r"$W_{\log}$")
    ax.plot(etas, [r["advantage"] for r in rows], "-o", ms=3, color="C0", label="$A$")
    ax.axvline(0.5, ls="--", color="C3", lw=1)
    shade = [r["eta"] for r in rows if r["w_log_ng"] <= 1e-9 and r["advantage"] > EPSILON]
    if shade:
        ax.axvspan(min(shade), max(shade), color="C0", alpha=0.10)
        ax.text(0.5 * (min(shade) + max(shade)), 0.30,
                "$W_{\\log}=0$\nbut $A>0$", ha="center", fontsize=8, color="C0")
    ax.set_xlabel(r"transmissivity $\eta$")
    ax.set_ylabel("value")
    ax.set_title("Fock |1>: resource measure and\ntask measure collapse at different points")
    ax.legend(fontsize=8, frameon=False)

    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle("Experiment 05: advantage over an optimized resource-matched Gaussian baseline",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    grid = phase_space_grid(**GRID)

    sweeps = {}
    for name, target in targets().items():
        print(f"  {name} ...")
        sweeps[name] = sweep_target(name, target, grid)

    summaries = {
        name: decoupling_summary(name, rows, targets()[name], grid)
        for name, rows in sweeps.items()
    }

    report = {
        "configuration": {
            "cutoff": CUTOFF,
            "grid": GRID,
            "epsilon": EPSILON,
            "sigma_phi": SIGMA_PHI,
            "baseline_is_gaussian_channel": is_gaussian_channel(SIGMA_PHI),
            "task": "target-state preparation (fidelity)",
            "baseline_family": "displaced squeezed vacuum, energy-matched",
        },
        "sweeps": sweeps,
        "decoupling": summaries,
        "runtime_seconds": time.perf_counter() - started,
    }
    (OUT_DIR / "gaussian_baseline_report.json").write_text(json.dumps(report, indent=2))
    make_figure(sweeps, FIGURE_PATH)

    print("\n" + "=" * 78)
    print("EXPERIMENT 05: advantage over an optimized matched Gaussian baseline")
    print("=" * 78)
    for name, summary in summaries.items():
        bracket = summary["advantage_threshold_bracket"]
        print(f"  {name}")
        print(f"    negativity threshold (bisected)   : eta = {summary['negativity_threshold']:.4f}")
        if bracket:
            print(f"    advantage threshold (A > {EPSILON:g})     : eta in [{bracket[0]:.2f}, {bracket[1]:.2f}]")
        else:
            print(f"    advantage threshold (A > {EPSILON:g})     : not crossed in the swept range")
        if summary["gaussian_wins_below"] is not None:
            print(f"    Gaussian baseline wins (A < 0)    : eta <= {summary['gaussian_wins_below']:.2f}"
                  f"   (min A = {summary['min_advantage']:+.4f})")
        if summary["decoupled_etas"]:
            print(f"    decoupled regime (W_log = 0, A > {EPSILON:g}): "
                  f"eta in [{min(summary['decoupled_etas'])}, {max(summary['decoupled_etas'])}], "
                  f"max A = {summary['max_advantage_without_negativity']:.4f}")
        else:
            print("    decoupled regime: none found")
    print(f"\n  runtime {report['runtime_seconds']:.0f}s")
    print("=" * 78)
    print(f"Wrote {OUT_DIR} and {FIGURE_PATH.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
