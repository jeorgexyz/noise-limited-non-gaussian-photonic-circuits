"""Experiment 02: certified Kerr depth collapse and a discrete (D, eta) map.

Run from any working directory:
    python experiments/02_depth_collapse/run.py
    python experiments/02_depth_collapse/run.py --smoke
    python experiments/02_depth_collapse/run.py --publish

The default run checks cutoff/grid/optimizer convergence, writes all numerical data
under results/, and returns nonzero if a check fails. --publish additionally copies
the checked report and figures to tracked publication paths. --smoke exercises the
same pipeline on a short, explicitly horizon-limited sweep. It cannot be published.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
import time
import warnings
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ngphotonic.backends.reference import squeezed_ket, to_dm
from ngphotonic.circuits.templates import LayerSpec, apply_layer
from ngphotonic.metrics.negativity import grid_diagnostics, wigner_log_negativity
from ngphotonic.metrics.operational import target_state_fidelity
from ngphotonic.metrics.wigner import phase_space_grid, wigner
from ngphotonic.sweeps.depth import (
    DepthConfig,
    depth_summary,
    evaluate_depth,
    sweep_depth,
    vacuum_advantage_bound,
)


def validate_sweep(sweep: dict, config: DepthConfig, settings: dict) -> dict:
    """Re-evolve every depth at higher cutoff; re-optimize every possible boundary.

    Higher-cutoff NG fidelity is checked at EVERY depth. Scores within 0.002 of any
    reported epsilon, and the peak and final viable depths, are reoptimized with more
    starts and a different seed. A separate same-cutoff rerun isolates start/seed
    sensitivity from cutoff error. Wigner resolution and extent are checked separately.
    """
    high = replace(config, cutoff=settings["validation_cutoff"], extra_starts=11, seed=1)
    spec = LayerSpec(**sweep["layer"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        source = squeezed_ket(high.squeezing, 0, high.cutoff)
        source_low = squeezed_ket(config.squeezing, 0, config.cutoff)
    rho = to_dm(source)
    rows = sweep["rows"]
    selected = {0, int(np.argmax([r["advantage"] for r in rows])),
                int(np.argmax([r["w_log"] for r in rows]))}
    for eps in (0.005, 0.01, 0.02):
        viable = [r["depth"] for r in rows if r["advantage"] > eps]
        if viable:
            last = max(viable)
            selected.update(d for d in (last - 1, last, last + 1) if 0 <= d < len(rows))
        selected.update(r["depth"] for r in rows if abs(r["advantage"] - eps) < 0.002)
    selected.add(len(rows) - 1)
    comparisons = []
    max_ng_error = 0.0
    checked_advantages = [r["advantage"] for r in rows]
    fine_grid = phase_space_grid(limit=config.grid_limit,
                                 points=settings["validation_grid_points"])
    # Keep approximately the same spacing while increasing the grid extent.
    extent_points = round((settings["validation_grid_points"] - 1)
                              * settings["validation_grid_limit"] / config.grid_limit) + 1
    extent_grid = phase_space_grid(limit=settings["validation_grid_limit"], points=extent_points)
    base_grid = phase_space_grid(limit=config.grid_limit, points=config.grid_points)
    rho_low = to_dm(source_low)
    for depth, row in enumerate(rows):
        if depth:
            rho = apply_layer(rho, spec)
            rho_low = apply_layer(rho_low, spec)
        target = np.exp(1j * depth * spec.kerr_xi * np.arange(high.cutoff)**2) * source
        ng_error = abs(target_state_fidelity(rho, target) - row["s_ng"])
        max_ng_error = max(max_ng_error, ng_error)
        if depth not in selected:
            continue
        checked = evaluate_depth(rho, target, spec, depth, high, measure_wigner=False)
        target_low = np.exp(1j * depth * spec.kerr_xi * np.arange(config.cutoff)**2)
        target_low *= source_low
        reseeded = evaluate_depth(rho_low, target_low, spec, depth,
                                  replace(config, extra_starts=11, seed=1), measure_wigner=False)
        checked_advantages[depth] = checked["advantage"]
        w_values = []
        w_diagnostics = []
        for x, p, X, P in (base_grid, fine_grid, extent_grid):
            W = wigner(rho, X, P)
            w_values.append(wigner_log_negativity(W, x, p))
            w_diagnostics.append(grid_diagnostics(W, x, p))
        comparisons.append({
            "depth": depth,
            "ng_cutoff_error": ng_error,
            "baseline_cutoff_error": abs(checked["s_gaussian"] - reseeded["s_gaussian"]),
            "baseline_seed_error": abs(reseeded["s_gaussian"] - row["s_gaussian"]),
            "advantage_error": abs(checked["advantage"] - row["advantage"]),
            "wigner_cutoff_error": abs(w_values[0] - row["w_log"]),
            "wigner_grid_error": abs(w_values[1] - w_values[0]),
            "wigner_extent_error": abs(w_values[2] - w_values[1]),
            "wigner_norm_error": max(d["norm_error"] for d in w_diagnostics),
            "checked_advantage": checked["advantage"],
            "checked_baseline_energy_error": checked["baseline_energy_error"],
            "checked_baseline_converged_fraction": checked["baseline_converged_fraction"],
        })
    checked_summary = depth_summary(
        checked_advantages, config.epsilon,
        tail_bound=vacuum_advantage_bound(config.budget, spec.eta, len(rows)),
    )
    maxima = {key: max(c[key] for c in comparisons) for key in (
        "baseline_cutoff_error", "baseline_seed_error", "advantage_error", "wigner_cutoff_error",
        "wigner_grid_error", "wigner_extent_error", "wigner_norm_error",
    )}
    boundary_stable = (checked_summary["viable_intervals"] == sweep["collapse"]["viable_intervals"])
    passed = (
        max_ng_error < settings["score_tolerance"]
        and max(maxima[k] for k in ("baseline_cutoff_error", "baseline_seed_error",
                                    "advantage_error"))
        < settings["score_tolerance"]
        and max(maxima[k] for k in ("wigner_cutoff_error", "wigner_grid_error",
                                   "wigner_extent_error", "wigner_norm_error"))
        < settings["wigner_tolerance"]
        and max(r["trace_error"] for r in rows) < 1e-10
        and min(r["minimum_eigenvalue"] for r in rows) > -1e-10
        and max(r["baseline_energy_error"] for r in rows) < settings["score_tolerance"]
        and boundary_stable
        and sweep["collapse"]["boundary_margin"] > 2 * maxima["advantage_error"]
    )
    return {
        "passed": bool(passed), "cutoff": high.cutoff, "extra_starts": high.extra_starts,
        "seed": high.seed, "max_ng_cutoff_error_all_depths": max_ng_error,
        "max_errors": maxima, "viable_intervals_stable": boundary_stable,
        "checked_depths": comparisons,
        "scope": "NG cutoff at all depths; baseline and Wigner checks at listed depths",
    }


def make_figure(report: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    sweeps = report["sweeps"]
    epsilon = report["configuration"]["epsilon"]
    sigmas = sorted({s["layer"]["sigma_phi"] for s in sweeps})
    etas = sorted({s["layer"]["eta"] for s in sweeps})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.4))
    colors = plt.cm.viridis(np.linspace(0.12, 0.85, len(etas)))
    display_depth = min(40, max(len(s["rows"]) - 1 for s in sweeps))
    for sweep in sweeps:
        eta, sigma = sweep["layer"]["eta"], sweep["layer"]["sigma_phi"]
        color = colors[etas.index(eta)]
        rows = sweep["rows"][:display_depth + 1]
        style = "-" if sigma == 0 else "--"
        label = rf"$\eta={eta:g},\ \sigma={sigma:g}$"
        axes[0, 0].plot([r["depth"] for r in rows], [r["advantage"] for r in rows],
                        style, color=color, label=label, lw=1.4)
        axes[0, 1].plot([r["depth"] for r in rows], [r["w_log"] for r in rows],
                        style, color=color, lw=1.4)
    axes[0, 0].axhline(epsilon, color="black", ls=":", lw=1, label=rf"$\epsilon={epsilon:g}$")
    axes[0, 0].axhline(0, color="grey", lw=.6)
    axes[0, 0].set(title="Task advantage can disappear and revive", ylabel=r"$A=S_{NG}-S_G^*$")
    axes[0, 0].legend(fontsize=7, ncol=2, frameon=False)
    axes[0, 1].set(title="Resource survival is a different observable", ylabel=r"$W_{\log}$")
    # strict=False: the axes row is longer than the sigma list, and the extra
    # axis is handled separately below. Truncation here is intended.
    for ax, sigma in zip(axes[1], sigmas, strict=False):
        matrix = np.array([[r["advantage"] for r in s["rows"][:display_depth + 1]]
                           for s in sweeps if s["layer"]["sigma_phi"] == sigma])
        vmax = max(float(np.max(np.abs(matrix))), epsilon)
        plot = ax.imshow(matrix, origin="lower", aspect="auto", interpolation="nearest",
                         extent=(-.5, display_depth + .5, -.5, len(etas) - .5),
                         cmap="RdBu", norm=TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax))
        ax.set_yticks(range(len(etas)), [f"{e:g}" for e in etas])
        for i, sweep in enumerate(s for s in sweeps if s["layer"]["sigma_phi"] == sigma):
            last = sweep["collapse"]["last_viable_depth_observed"]
            if last is not None and last <= display_depth:
                ax.scatter(last, i, marker="|", color="black", s=160)
                ax.annotate(f" D*={last}", (last, i), xytext=(4, 5), textcoords="offset points",
                            fontsize=8, color="black",
                            bbox={"facecolor": "white", "edgecolor": "none", "alpha": .7, "pad": 1})
        ax.set(title=rf"Discrete collapse map: $\sigma={sigma:g}$", ylabel=r"per-layer $\eta$")
        fig.colorbar(plot, ax=ax, label="advantage", fraction=.045)
    if len(sigmas) == 1:
        axes[1, 1].set_visible(False)
    for ax in axes.flat:
        ax.set_xlabel("depth D")
    horizons = sorted({s["collapse"]["searched_through"] for s in sweeps})
    fig.suptitle("Kerr depth collapse | squeezed input r=0.6, Kerr strength 0.2", fontsize=13)
    fig.text(.5, .005,
             f"Shown through D={display_depth}; all integer depths searched "
             f"through {horizons}. "
             "Baseline: Gaussian input preparation, identical noise, no intermediate pumping.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .025, 1, .97))
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "02_depth_collapse")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    if args.smoke and args.publish:
        parser.error("A smoke run cannot publish headline artifacts.")
    settings = json.loads(args.config.read_text(encoding="utf-8"))
    config = DepthConfig(**{k: settings[k] for k in DepthConfig.__dataclass_fields__})
    if args.smoke:
        config = replace(config, grid_points=81, extra_starts=0)
        settings.update(etas=[0.9], sigmas=[0.1], validation_grid_points=161,
                        wigner_tolerance=0.002)
        args.output_dir /= "smoke"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    sweeps = []
    for sigma in settings["sigmas"]:
        for eta in settings["etas"]:
            print(f"eta={eta}, sigma={sigma}", flush=True)
            def progress(row):
                if row["depth"] % 20 == 0:
                    print(f"  D={row['depth']:3d} A={row['advantage']:+.6f}", flush=True)
            sweep = sweep_depth(config, eta, sigma, max_depth=6 if args.smoke else None,
                                progress=progress)
            print(f"  {sweep['collapse']}", flush=True)
            print("  Checking cutoff, starts, and Wigner grid...", flush=True)
            sweep["validation"] = validate_sweep(sweep, config, settings)
            print(f"  validation passed: {sweep['validation']['passed']}", flush=True)
            sweeps.append(sweep)
            (args.output_dir / "checkpoint.json").write_text(
                json.dumps({"configuration": asdict(config), "settings": settings,
                            "sweeps": sweeps}, indent=2, allow_nan=False), encoding="utf-8")
    try:
        git = ["git", "-c", f"safe.directory={ROOT.as_posix()}"]
        commit = subprocess.check_output([*git, "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = bool(subprocess.check_output([*git, "status", "--porcelain"], cwd=ROOT, text=True))
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    report = {
        "schema_version": 1, "experiment": "02_depth_collapse", "smoke": args.smoke,
        "configuration": asdict(config), "settings": settings,
        "protocol": {
            "input": "squeezed vacuum S(r)|0>",
            "layer_order": ["Kerr", "pure loss", "phase diffusion"],
            "target": "Kerr(D xi) S(r)|0>, noiseless output at each depth",
            "baseline": "pure displaced squeezed input, exact analytic input-energy shell; "
                        "D matched noise exposures, no Kerr, no intermediate pumping",
            "score": "fidelity to the common pure target",
            "bound_direction": "best-found Gaussian score is a lower bound; "
                           "advantage is an upper bound",
            "future_certificate": "A(D) <= 2 sqrt(sinh(r)^2 eta^D) using the "
                              "squeezed-input witness",
            "negativity_threshold": "not used to define operational D*",
            "kerr_convention_source": "https://docs.piquasso.com/instructions/gates.html#piquasso.instructions.gates.Kerr",
        },
        "provenance": {
            "utc": datetime.now(UTC).isoformat(), "git_commit": commit,
            "working_tree_dirty": dirty, "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {p: importlib.metadata.version(p)
                     for p in ("numpy", "scipy", "matplotlib")},
            "source_sha256": {
                str(p.relative_to(ROOT).as_posix()): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted((ROOT / "src").rglob("*.py"))
            } | {"experiment_script": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 "config": hashlib.sha256(args.config.read_bytes()).hexdigest()},
        },
        "sweeps": sweeps,
        "validation_passed": all(s["validation"]["passed"] for s in sweeps),
        "runtime_seconds": time.perf_counter() - started,
    }
    report_path = args.output_dir / "depth_collapse_report.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    figure_path = args.output_dir / "02_depth_collapse.png"
    make_figure(report, figure_path)
    if not report["validation_passed"]:
        print(f"Validation failed; inspect {report_path}. Publication artifacts were not updated.")
        return 1
    if args.publish:
        for path in (ROOT / "figures" / figure_path.name,
                 ROOT / "docs" / "assets" / figure_path.name):
            shutil.copyfile(figure_path, path)
        shutil.copyfile(report_path, Path(__file__).with_name("report.json"))
    print(f"Wrote {report_path} ({report['runtime_seconds']:.1f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
