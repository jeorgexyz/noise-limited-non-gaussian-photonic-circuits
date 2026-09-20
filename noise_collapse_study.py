"""
Noise-Limited Photonic Circuit Depth & Non-Gaussianity Collapse

Investigates how realistic noise mechanisms (loss, thermal noise, imperfect squeezing)
progressively destroy non-Gaussian quantum advantage in photonic circuits.

This is the CV quantum optics analogue of noise-adaptive depth thresholds
and coherence collapse studies in discrete-variable quantum computing.

Author: Research implementation
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Dict
import json

# NOTE: This framework is designed to integrate with MrMustard
# Install MrMustard in your local environment with: pip install mrmustard
# For now, we'll create the analysis framework and helper functions


@dataclass
class NoiseParameters:
    """Physical noise parameters for photonic circuits"""
    loss_per_layer: float = 0.01  # Photon loss probability per layer
    thermal_photons: float = 0.001  # Thermal occupation number
    squeezing_imperfection: float = 0.95  # Squeezing efficiency (1.0 = perfect)
    detector_efficiency: float = 0.98  # Photodetector quantum efficiency
    
    def to_dict(self) -> Dict:
        return {
            'loss_per_layer': self.loss_per_layer,
            'thermal_photons': self.thermal_photons,
            'squeezing_imperfection': self.squeezing_imperfection,
            'detector_efficiency': self.detector_efficiency
        }


@dataclass
class CircuitMetrics:
    """Metrics tracking non-Gaussianity and quantum advantage"""
    depth: int
    wigner_negativity: float  # Volume of negative Wigner function
    fock_purity: float  # Purity in Fock basis
    photon_number_variance: float  # Non-classical photon statistics
    classical_simulability: float  # Estimated classical simulation cost
    
    def to_dict(self) -> Dict:
        return {
            'depth': self.depth,
            'wigner_negativity': self.wigner_negativity,
            'fock_purity': self.fock_purity,
            'photon_number_variance': self.photon_number_variance,
            'classical_simulability': self.classical_simulability
        }


class WignerNegativityTracker:
    """
    Track Wigner negativity evolution through noisy photonic circuits
    
    Wigner negativity is a key non-classical resource:
    - Positive Wigner function → classical phase space representation exists
    - Negative Wigner function → genuinely quantum, required for advantage
    """
    
    def __init__(self, noise_params: NoiseParameters):
        self.noise_params = noise_params
        self.history: List[CircuitMetrics] = []
        
    def estimate_wigner_negativity(self, 
                                   depth: int,
                                   initial_negativity: float = 1.0) -> float:
        """
        Estimate Wigner negativity decay with circuit depth
        
        Model: Negativity decays exponentially with effective noise per layer
        N(d) = N_0 * exp(-γ * d)
        where γ depends on loss, thermal noise, and imperfections
        """
        # Effective decoherence rate combining all noise sources
        gamma = (
            self.noise_params.loss_per_layer * 2.0 +  # Loss affects both quadratures
            self.noise_params.thermal_photons * 1.5 +  # Thermal noise mixing
            (1 - self.noise_params.squeezing_imperfection) * 0.5  # Squeezing imperfection
        )
        
        negativity = initial_negativity * np.exp(-gamma * depth)
        
        # Threshold: below this, numerical noise dominates
        negativity_threshold = 1e-6
        return max(0.0, negativity if negativity > negativity_threshold else 0.0)
    
    def estimate_fock_purity(self, depth: int, initial_purity: float = 0.9) -> float:
        """
        Estimate state purity in Fock basis
        Pure states have purity = 1, mixed states < 1
        """
        # Mixing increases with depth and thermal noise
        mixing_rate = (
            self.noise_params.loss_per_layer +
            self.noise_params.thermal_photons * 2.0
        )
        
        purity = initial_purity * np.exp(-mixing_rate * depth * 0.8)
        return max(0.1, min(1.0, purity))
    
    def estimate_photon_variance(self, depth: int, initial_var: float = 2.0) -> float:
        """
        Estimate photon number variance (non-classical statistics indicator)
        
        For coherent states: Var(n) = mean(n) (Poissonian)
        For Fock states: Var(n) = 0 (sub-Poissonian)
        For thermal states: Var(n) > mean(n) (super-Poissonian)
        """
        # Noise drives toward thermal statistics
        thermal_variance_growth = self.noise_params.thermal_photons * depth * 0.3
        loss_variance_reduction = self.noise_params.loss_per_layer * depth * 0.2
        
        variance = initial_var + thermal_variance_growth - loss_variance_reduction
        return max(0.5, variance)
    
    def estimate_classical_simulability(self, depth: int) -> float:
        """
        Estimate difficulty of classical simulation
        
        Returns a score (0 to 1):
        - 0: Easily simulable (Gaussian state)
        - 1: Hard to simulate (highly non-Gaussian)
        
        Based on required Fock cutoff and Wigner negativity
        """
        negativity = self.estimate_wigner_negativity(depth)
        purity = self.estimate_fock_purity(depth)
        
        # Simulability decreases with negativity and purity
        hardness = negativity * purity * 0.5 + negativity * 0.5
        
        return min(1.0, max(0.0, hardness))
    
    def simulate_depth_sweep(self, max_depth: int = 50) -> List[CircuitMetrics]:
        """
        Simulate circuit depth sweep and track all metrics
        """
        self.history = []
        
        for depth in range(max_depth + 1):
            metrics = CircuitMetrics(
                depth=depth,
                wigner_negativity=self.estimate_wigner_negativity(depth),
                fock_purity=self.estimate_fock_purity(depth),
                photon_number_variance=self.estimate_photon_variance(depth),
                classical_simulability=self.estimate_classical_simulability(depth)
            )
            self.history.append(metrics)
        
        return self.history
    
    def find_collapse_depth(self, threshold: float = 0.01) -> int:
        """
        Find the depth where Wigner negativity collapses below threshold
        (i.e., the "last viable non-Gaussian layer")
        """
        if not self.history:
            self.simulate_depth_sweep()
        
        for metrics in self.history:
            if metrics.wigner_negativity < threshold:
                return metrics.depth
        
        return len(self.history)


class NoiseComparison:
    """
    Compare different noise regimes to identify critical transitions
    """
    
    def __init__(self):
        self.scenarios: Dict[str, Tuple[NoiseParameters, WignerNegativityTracker]] = {}
    
    def add_scenario(self, name: str, noise_params: NoiseParameters):
        """Add a noise scenario to compare"""
        tracker = WignerNegativityTracker(noise_params)
        self.scenarios[name] = (noise_params, tracker)
    
    def run_comparison(self, max_depth: int = 50):
        """Run depth sweep for all scenarios"""
        for name, (params, tracker) in self.scenarios.items():
            print(f"Running scenario: {name}")
            tracker.simulate_depth_sweep(max_depth)
    
    def plot_comparison(self, save_path: str = None):
        """Generate comprehensive comparison plots"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Non-Gaussianity Collapse vs Circuit Depth', fontsize=16, fontweight='bold')
        
        # Plot 1: Wigner Negativity
        ax1 = axes[0, 0]
        for name, (params, tracker) in self.scenarios.items():
            depths = [m.depth for m in tracker.history]
            negativities = [m.wigner_negativity for m in tracker.history]
            ax1.semilogy(depths, negativities, marker='o', label=name, linewidth=2, markersize=4)
        
        ax1.axhline(y=0.01, color='red', linestyle='--', alpha=0.5, label='Threshold')
        ax1.set_xlabel('Circuit Depth', fontsize=12)
        ax1.set_ylabel('Wigner Negativity', fontsize=12)
        ax1.set_title('Wigner Negativity Decay', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Fock Purity
        ax2 = axes[0, 1]
        for name, (params, tracker) in self.scenarios.items():
            depths = [m.depth for m in tracker.history]
            purities = [m.fock_purity for m in tracker.history]
            ax2.plot(depths, purities, marker='s', label=name, linewidth=2, markersize=4)
        
        ax2.set_xlabel('Circuit Depth', fontsize=12)
        ax2.set_ylabel('Fock Purity', fontsize=12)
        ax2.set_title('State Purity Evolution', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Classical Simulability
        ax3 = axes[1, 0]
        for name, (params, tracker) in self.scenarios.items():
            depths = [m.depth for m in tracker.history]
            simulability = [1 - m.classical_simulability for m in tracker.history]  # Invert for clarity
            ax3.plot(depths, simulability, marker='^', label=name, linewidth=2, markersize=4)
        
        ax3.set_xlabel('Circuit Depth', fontsize=12)
        ax3.set_ylabel('Classical Simulability', fontsize=12)
        ax3.set_title('Quantum Advantage Loss', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Photon Number Variance
        ax4 = axes[1, 1]
        for name, (params, tracker) in self.scenarios.items():
            depths = [m.depth for m in tracker.history]
            variances = [m.photon_number_variance for m in tracker.history]
            ax4.plot(depths, variances, marker='d', label=name, linewidth=2, markersize=4)
        
        ax4.set_xlabel('Circuit Depth', fontsize=12)
        ax4.set_ylabel('Photon Number Variance', fontsize=12)
        ax4.set_title('Non-Classical Statistics', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        
        return fig
    
    def generate_report(self, save_path: str = None) -> Dict:
        """Generate summary report of findings"""
        report = {
            'scenarios': {},
            'comparative_analysis': {}
        }
        
        collapse_depths = {}
        
        for name, (params, tracker) in self.scenarios.items():
            collapse_depth = tracker.find_collapse_depth(threshold=0.01)
            collapse_depths[name] = collapse_depth
            
            report['scenarios'][name] = {
                'noise_parameters': params.to_dict(),
                'collapse_depth': collapse_depth,
                'final_negativity': tracker.history[-1].wigner_negativity,
                'final_purity': tracker.history[-1].fock_purity,
                'final_simulability': tracker.history[-1].classical_simulability
            }
        
        # Comparative analysis
        min_depth = min(collapse_depths.values())
        max_depth = max(collapse_depths.values())
        
        report['comparative_analysis'] = {
            'min_collapse_depth': min_depth,
            'max_collapse_depth': max_depth,
            'depth_range': max_depth - min_depth,
            'most_robust_scenario': min(collapse_depths, key=collapse_depths.get),
            'least_robust_scenario': max(collapse_depths, key=collapse_depths.get)
        }
        
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"Saved report to {save_path}")
        
        return report


def create_standard_scenarios() -> NoiseComparison:
    """
    Create standard noise scenarios for comparison
    """
    comparison = NoiseComparison()
    
    # Scenario 1: Ideal (minimal noise)
    comparison.add_scenario(
        "Near-Ideal",
        NoiseParameters(
            loss_per_layer=0.001,
            thermal_photons=0.0001,
            squeezing_imperfection=0.99,
            detector_efficiency=0.99
        )
    )
    
    # Scenario 2: Laboratory-grade (realistic current technology)
    comparison.add_scenario(
        "Lab-Grade",
        NoiseParameters(
            loss_per_layer=0.01,
            thermal_photons=0.001,
            squeezing_imperfection=0.95,
            detector_efficiency=0.98
        )
    )
    
    # Scenario 3: Moderate loss
    comparison.add_scenario(
        "Moderate Loss",
        NoiseParameters(
            loss_per_layer=0.05,
            thermal_photons=0.001,
            squeezing_imperfection=0.95,
            detector_efficiency=0.95
        )
    )
    
    # Scenario 4: High thermal noise (room temperature components)
    comparison.add_scenario(
        "High Thermal",
        NoiseParameters(
            loss_per_layer=0.01,
            thermal_photons=0.01,
            squeezing_imperfection=0.95,
            detector_efficiency=0.98
        )
    )
    
    # Scenario 5: Poor squeezing
    comparison.add_scenario(
        "Poor Squeezing",
        NoiseParameters(
            loss_per_layer=0.01,
            thermal_photons=0.001,
            squeezing_imperfection=0.80,
            detector_efficiency=0.98
        )
    )
    
    # Scenario 6: Worst-case (all noise sources high)
    comparison.add_scenario(
        "Worst-Case",
        NoiseParameters(
            loss_per_layer=0.1,
            thermal_photons=0.02,
            squeezing_imperfection=0.75,
            detector_efficiency=0.90
        )
    )
    
    return comparison


def main():
    """
    Main analysis pipeline
    """
    print("=" * 70)
    print("Noise-Limited Photonic Circuit Depth & Non-Gaussianity Collapse")
    print("=" * 70)
    print()
    
    # Create and run scenarios
    comparison = create_standard_scenarios()
    print("Running depth sweep for all scenarios...")
    comparison.run_comparison(max_depth=50)
    print()
    
    # Generate report
    print("Generating analysis report...")
    report = comparison.generate_report(save_path='/home/claude/noise_collapse_report.json')
    print()
    
    # Print key findings
    print("=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)
    print()
    
    for scenario_name, data in report['scenarios'].items():
        print(f"{scenario_name}:")
        print(f"  Collapse Depth: {data['collapse_depth']} layers")
        print(f"  Final Negativity: {data['final_negativity']:.6f}")
        print(f"  Final Purity: {data['final_purity']:.4f}")
        print(f"  Classical Simulability: {data['final_simulability']:.4f}")
        print()
    
    print("Comparative Analysis:")
    comp = report['comparative_analysis']
    print(f"  Depth Range: {comp['min_collapse_depth']} - {comp['max_collapse_depth']} layers")
    print(f"  Most Robust: {comp['most_robust_scenario']}")
    print(f"  Least Robust: {comp['least_robust_scenario']}")
    print()
    
    # Generate plots
    print("Generating comparison plots...")
    fig = comparison.plot_comparison(save_path='/home/claude/noise_collapse_plots.png')
    plt.show()
    
    print()
    print("=" * 70)
    print("Analysis complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
