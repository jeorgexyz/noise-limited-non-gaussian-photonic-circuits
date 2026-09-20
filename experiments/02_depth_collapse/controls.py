"""Hold total Kerr strength, loss and dephasing fixed while changing layer count.

Run after run.py:
    python experiments/02_depth_collapse/controls.py

This distinguishes ordering effects from simply increasing the total gate/noise dose.
The target, energy budget, and Gaussian baseline are identical at every depth.
"""

from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import warnings

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ngphotonic.backends.reference import squeezed_ket, to_dm  # noqa: E402
from ngphotonic.circuits.templates import LayerSpec, run_layered  # noqa: E402
from ngphotonic.sweeps.depth import DepthConfig, evaluate_depth  # noqa: E402


def fixed_total_control(config: DepthConfig) -> dict:
    totals = {"eta": .5, "sigma_phi": .2, "kerr_xi": 1.2}
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        source = squeezed_ket(config.squeezing, 0, config.cutoff)
    target = np.exp(1j * totals["kerr_xi"] * np.arange(config.cutoff)**2) * source
    for depth in (1, 2, 4, 8, 16, 32):
        spec = LayerSpec(eta=totals["eta"]**(1/depth),
                         sigma_phi=totals["sigma_phi"]/np.sqrt(depth),
                         kerr_xi=totals["kerr_xi"]/depth)
        rho = run_layered(to_dm(source), spec, depth)
        row = evaluate_depth(rho, target, spec, depth, config)
        row["layer"] = asdict(spec)
        rows.append(row)
    return {"total_parameters": totals, "rows": rows}


def main() -> int:
    config = DepthConfig()
    control = fixed_total_control(config)
    high = fixed_total_control(replace(config, cutoff=44, extra_starts=11, seed=1))
    errors = {key: max(abs(a[key]-b[key]) for a, b in zip(control["rows"], high["rows"]))
              for key in ("s_ng", "s_gaussian", "advantage", "lumped_trace_distance")}
    passed = max(errors.values()) < 1e-5
    report = {
        "configuration": asdict(config), "control": control,
        "protocol": "Fixed total Kerr, transmissivity, phase variance, input energy and pure target; "
                    "only their subdivision into interleaved layers changes.",
        "validation": {"passed": passed, "higher_cutoff": 44, "extra_starts": 11,
                       "seed": 1, "max_errors": errors},
    }
    out = ROOT / "results" / "02_depth_collapse"
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "fixed_totals_report.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    if not passed:
        print("Control convergence check failed.")
        return 1

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = control["rows"]
    depths = [r["depth"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(depths, [r["s_ng"] for r in rows], "o-", label="interleaved Kerr + noise")
    axes[0].axhline(rows[0]["s_lumped"], ls="--", color="C1", label="one lumped channel")
    axes[0].axhline(rows[0]["s_gaussian"], ls=":", color="black", label="matched Gaussian baseline")
    axes[0].set(xlabel="number of layers", ylabel="fidelity to the same target",
                title="Same total resources, different outcome")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].plot(depths, [r["lumped_trace_distance"] for r in rows], "o-", color="C2")
    axes[1].set(xlabel="number of layers", ylabel="trace distance from lumped output",
                title="Layer ordering has a measurable effect")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks(depths, [str(d) for d in depths])
        ax.grid(alpha=.2)
    fig.suptitle(r"Fixed totals: $\eta=0.5$, $\sigma=0.2$, $\xi=1.2$, input $r=0.6$")
    fig.tight_layout()
    for path in (out / "02_depth_controls.png", ROOT / "figures" / "02_depth_controls.png",
                 ROOT / "docs" / "assets" / "02_depth_controls.png"):
        fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    Path(__file__).with_name("controls_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    for row in rows:
        print(f"D={row['depth']:2d}: S_NG={row['s_ng']:.6f} S_G={row['s_gaussian']:.6f} "
              f"distance={row['lumped_trace_distance']:.6f}")
    print(f"Validation passed; max score/distance change {max(errors.values()):.3g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
