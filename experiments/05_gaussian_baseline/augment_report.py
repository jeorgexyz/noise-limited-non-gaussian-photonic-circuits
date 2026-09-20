"""Add the Gaussian-parity crossing to an existing experiment 05 report.

Recomputes the summary fields that are derivable from the saved sweep rows, so a
reporting change does not require repeating the ~10 minute optimization sweep. Run
after ``run.py``; running ``run.py`` alone also produces these fields.

The crossing added here is where ``A`` changes sign, i.e. where the optimized
energy-matched Gaussian baseline overtakes the non-Gaussian circuit. That direction is
robust to optimizer quality: ``S_G*`` is a lower bound on the true Gaussian optimum, so
a measured ``A < 0`` can only understate how far ahead the Gaussian actually is.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "results" / "05_gaussian_baseline" / "gaussian_baseline_report.json"


def bracket_crossing(rows, key, level):
    ordered = sorted(rows, key=lambda r: r["eta"])
    for low, high in zip(ordered, ordered[1:]):
        if low[key] <= level < high[key]:
            return [low["eta"], high["eta"]]
    return None


def main() -> int:
    if not REPORT.exists():
        print(f"No report at {REPORT}; run experiments/05_gaussian_baseline/run.py first.")
        return 1

    report = json.loads(REPORT.read_text())
    for name, rows in report["sweeps"].items():
        summary = report["decoupling"][name]
        losing = [r["eta"] for r in rows if r["advantage"] < 0.0]
        summary["parity_bracket"] = bracket_crossing(rows, "advantage", 0.0)
        summary["gaussian_wins_below"] = max(losing) if losing else None
        summary["min_advantage"] = min(r["advantage"] for r in rows)

    REPORT.write_text(json.dumps(report, indent=2))

    print("=" * 78)
    print("EXPERIMENT 05 (augmented): where the Gaussian baseline overtakes")
    print("=" * 78)
    for name, summary in report["decoupling"].items():
        print(f"  {name}")
        print(f"    negativity threshold             : eta = {summary['negativity_threshold']:.4f}")
        if summary["gaussian_wins_below"] is not None:
            print(f"    Gaussian baseline wins (A < 0)   : eta <= {summary['gaussian_wins_below']:.2f}"
                  f"  (min A = {summary['min_advantage']:+.4f})")
        else:
            print(f"    Gaussian baseline never wins     : min A = {summary['min_advantage']:+.4f}")
        if summary["decoupled_etas"]:
            print(f"    decoupled (W_log = 0, A > {summary['epsilon']:g})    : "
                  f"eta in [{min(summary['decoupled_etas'])}, {max(summary['decoupled_etas'])}], "
                  f"max A = {summary['max_advantage_without_negativity']:.4f}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
