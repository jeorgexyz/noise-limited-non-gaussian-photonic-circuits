"""
Experimental Analysis: Parameter Sweeps and Sensitivity Studies

This script performs detailed parameter sweeps to identify:
1. Most critical noise sources
2. Operating regime boundaries  
3. Platform-specific design constraints
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from noise_collapse_study import NoiseParameters, WignerNegativityTracker
import json


class ParameterSweepAnalysis:
    """
    Perform 2D parameter sweeps to map out collapse boundaries
    """
    
    def __init__(self):
        self.results = {}
        
    def sweep_loss_vs_thermal(self, 
                              loss_range: tuple = (0.001, 0.1, 20),
                              thermal_range: tuple = (0.0001, 0.02, 20),
                              target_depth: int = 30):
        """
        2D sweep: photon loss vs thermal noise
        
        Goal: Identify which noise source is more destructive
        """
        loss_values = np.linspace(*loss_range)
        thermal_values = np.linspace(*thermal_range)
        
        negativity_map = np.zeros((len(thermal_values), len(loss_values)))
        collapse_map = np.zeros((len(thermal_values), len(loss_values)))
        
        print("Running loss vs thermal sweep...")
        print(f"Grid: {len(loss_values)} × {len(thermal_values)} = {len(loss_values)*len(thermal_values)} points")
        
        for i, thermal in enumerate(thermal_values):
            for j, loss in enumerate(loss_values):
                # Create noise parameters
                params = NoiseParameters(
                    loss_per_layer=loss,
                    thermal_photons=thermal,
                    squeezing_imperfection=0.95,
                    detector_efficiency=0.98
                )
                
                # Run simulation
                tracker = WignerNegativityTracker(params)
                tracker.simulate_depth_sweep(max_depth=50)
                
                # Store results
                negativity_map[i, j] = tracker.history[target_depth].wigner_negativity
                collapse_map[i, j] = tracker.find_collapse_depth(threshold=0.01)
                
            print(f"  Progress: {(i+1)/len(thermal_values)*100:.1f}%", end='\r')
        
        print("\nSweep complete!")
        
        self.results['loss_vs_thermal'] = {
            'loss_values': loss_values.tolist(),
            'thermal_values': thermal_values.tolist(),
            'negativity_map': negativity_map.tolist(),
            'collapse_map': collapse_map.tolist(),
            'target_depth': target_depth
        }
        
        return loss_values, thermal_values, negativity_map, collapse_map
    
    def sweep_depth_vs_loss(self,
                           depth_range: tuple = (0, 50, 51),
                           loss_range: tuple = (0.001, 0.1, 10)):
        """
        2D sweep: circuit depth vs photon loss
        
        Goal: Visualize how collapse depth changes with loss
        """
        depths = np.linspace(depth_range[0], depth_range[1], depth_range[2])
        loss_values = np.linspace(*loss_range)
        
        negativity_map = np.zeros((len(loss_values), len(depths)))
        
        print("Running depth vs loss sweep...")
        
        for i, loss in enumerate(loss_values):
            params = NoiseParameters(
                loss_per_layer=loss,
                thermal_photons=0.001,
                squeezing_imperfection=0.95,
                detector_efficiency=0.98
            )
            
            tracker = WignerNegativityTracker(params)
            tracker.simulate_depth_sweep(max_depth=int(depths[-1]))
            
            for j, depth in enumerate(depths):
                depth_idx = int(depth)
                negativity_map[i, j] = tracker.history[depth_idx].wigner_negativity
            
            print(f"  Progress: {(i+1)/len(loss_values)*100:.1f}%", end='\r')
        
        print("\nSweep complete!")
        
        self.results['depth_vs_loss'] = {
            'depths': depths.tolist(),
            'loss_values': loss_values.tolist(),
            'negativity_map': negativity_map.tolist()
        }
        
        return depths, loss_values, negativity_map
    
    def plot_2d_sweep_results(self, save_prefix: str = None):
        """
        Generate publication-quality 2D sweep plots
        """
        fig = plt.figure(figsize=(16, 12))
        
        # Plot 1: Loss vs Thermal - Negativity at target depth
        if 'loss_vs_thermal' in self.results:
            ax1 = plt.subplot(2, 2, 1)
            data = self.results['loss_vs_thermal']
            loss_vals = np.array(data['loss_values'])
            thermal_vals = np.array(data['thermal_values'])
            neg_map = np.array(data['negativity_map'])
            
            im1 = ax1.contourf(loss_vals, thermal_vals, neg_map, 
                              levels=20, cmap='viridis')
            ax1.contour(loss_vals, thermal_vals, neg_map, 
                       levels=[0.01], colors='red', linewidths=2, 
                       linestyles='--')
            
            ax1.set_xlabel('Loss per Layer', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Thermal Photons', fontsize=12, fontweight='bold')
            ax1.set_title(f'Wigner Negativity (depth={data["target_depth"]})', 
                         fontsize=13, fontweight='bold')
            plt.colorbar(im1, ax=ax1, label='Negativity')
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Loss vs Thermal - Collapse depth
            ax2 = plt.subplot(2, 2, 2)
            collapse_map = np.array(data['collapse_map'])
            
            im2 = ax2.contourf(loss_vals, thermal_vals, collapse_map, 
                              levels=20, cmap='RdYlGn')
            ax2.set_xlabel('Loss per Layer', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Thermal Photons', fontsize=12, fontweight='bold')
            ax2.set_title('Collapse Depth', fontsize=13, fontweight='bold')
            plt.colorbar(im2, ax=ax2, label='Layers')
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Depth vs Loss - Full evolution
        if 'depth_vs_loss' in self.results:
            ax3 = plt.subplot(2, 2, 3)
            data = self.results['depth_vs_loss']
            depths = np.array(data['depths'])
            loss_vals = np.array(data['loss_values'])
            neg_map = np.array(data['negativity_map'])
            
            im3 = ax3.contourf(depths, loss_vals, neg_map, 
                              levels=20, cmap='plasma')
            ax3.contour(depths, loss_vals, neg_map, 
                       levels=[0.01], colors='cyan', linewidths=2, 
                       linestyles='--')
            
            ax3.set_xlabel('Circuit Depth', fontsize=12, fontweight='bold')
            ax3.set_ylabel('Loss per Layer', fontsize=12, fontweight='bold')
            ax3.set_title('Negativity Evolution', fontsize=13, fontweight='bold')
            plt.colorbar(im3, ax=ax3, label='Negativity')
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: Collapse boundary
            ax4 = plt.subplot(2, 2, 4)
            
            # Extract collapse boundary
            collapse_depths = []
            for i, loss in enumerate(loss_vals):
                # Find where negativity crosses threshold
                neg_curve = neg_map[i, :]
                crossing_idx = np.where(neg_curve < 0.01)[0]
                if len(crossing_idx) > 0:
                    collapse_depths.append(depths[crossing_idx[0]])
                else:
                    collapse_depths.append(depths[-1])
            
            ax4.plot(loss_vals, collapse_depths, 'o-', 
                    linewidth=3, markersize=8, color='darkred')
            ax4.fill_between(loss_vals, 0, collapse_depths, 
                            alpha=0.3, color='red')
            
            ax4.set_xlabel('Loss per Layer', fontsize=12, fontweight='bold')
            ax4.set_ylabel('Collapse Depth', fontsize=12, fontweight='bold')
            ax4.set_title('Viable Operating Region', fontsize=13, fontweight='bold')
            ax4.grid(True, alpha=0.3)
            
            # Add text annotation
            ax4.text(0.95, 0.95, 'Quantum\nAdvantage',
                    transform=ax4.transAxes,
                    fontsize=11, fontweight='bold',
                    verticalalignment='top',
                    horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
            
            ax4.text(0.95, 0.05, 'Classical\nRegime',
                    transform=ax4.transAxes,
                    fontsize=11, fontweight='bold',
                    verticalalignment='bottom',
                    horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
        
        plt.tight_layout()
        
        if save_prefix:
            filename = f"{save_prefix}_parameter_sweeps.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved parameter sweep plots to {filename}")
        
        return fig
    
    def save_results(self, filename: str):
        """Save all sweep results to JSON"""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Saved sweep results to {filename}")


class PlatformComparison:
    """
    Compare different photonic platforms
    """
    
    PLATFORMS = {
        'silicon_nitride': {
            'name': 'Silicon Nitride (SiN)',
            'loss_per_cm': 0.1,  # dB
            'typical_length_cm': 5.0,
            'thermal_photons': 0.0001,  # Cryogenic
            'squeezing_efficiency': 0.95,
            'detector_efficiency': 0.95
        },
        'silicon_photonics': {
            'name': 'Silicon Photonics (SOI)',
            'loss_per_cm': 2.0,  # dB
            'typical_length_cm': 2.0,
            'thermal_photons': 0.001,  # Room temp possible
            'squeezing_efficiency': 0.90,
            'detector_efficiency': 0.85
        },
        'fiber': {
            'name': 'Fiber Optics',
            'loss_per_cm': 0.0002,  # dB/cm = 0.2 dB/km
            'typical_length_cm': 1000.0,  # 10m
            'thermal_photons': 0.0001,
            'squeezing_efficiency': 0.97,
            'detector_efficiency': 0.98
        },
        'free_space': {
            'name': 'Free Space',
            'loss_per_cm': 0.001,  # dB (atmospheric)
            'typical_length_cm': 100.0,  # 1m
            'thermal_photons': 0.01,  # Room temp
            'squeezing_efficiency': 0.92,
            'detector_efficiency': 0.95
        }
    }
    
    @staticmethod
    def convert_db_to_loss_probability(loss_db: float) -> float:
        """Convert dB loss to probability"""
        transmission = 10**(-loss_db / 10)
        return 1 - transmission
    
    def compare_platforms(self, max_depth: int = 30):
        """
        Compare collapse depths across different platforms
        """
        results = {}
        
        print("=" * 70)
        print("Platform Comparison Analysis")
        print("=" * 70)
        print()
        
        for platform_id, config in self.PLATFORMS.items():
            print(f"Analyzing {config['name']}...")
            
            # Calculate effective loss per layer
            total_loss_db = config['loss_per_cm'] * config['typical_length_cm']
            loss_per_layer = self.convert_db_to_loss_probability(total_loss_db / max_depth)
            
            # Create noise parameters
            params = NoiseParameters(
                loss_per_layer=loss_per_layer,
                thermal_photons=config['thermal_photons'],
                squeezing_imperfection=config['squeezing_efficiency'],
                detector_efficiency=config['detector_efficiency']
            )
            
            # Run simulation
            tracker = WignerNegativityTracker(params)
            tracker.simulate_depth_sweep(max_depth=max_depth)
            
            # Store results
            collapse_depth = tracker.find_collapse_depth()
            
            results[platform_id] = {
                'name': config['name'],
                'collapse_depth': collapse_depth,
                'final_negativity': tracker.history[-1].wigner_negativity,
                'final_purity': tracker.history[-1].fock_purity,
                'tracker': tracker
            }
            
            print(f"  Collapse depth: {collapse_depth} layers")
            print(f"  Final negativity: {results[platform_id]['final_negativity']:.6f}")
            print()
        
        return results
    
    def plot_platform_comparison(self, results: dict, save_path: str = None):
        """
        Visualize platform comparison
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Negativity evolution
        ax1 = axes[0]
        for platform_id, data in results.items():
            tracker = data['tracker']
            depths = [m.depth for m in tracker.history]
            negativities = [m.wigner_negativity for m in tracker.history]
            
            ax1.semilogy(depths, negativities, marker='o', 
                        label=data['name'], linewidth=2, markersize=4)
        
        ax1.axhline(y=0.01, color='red', linestyle='--', 
                   alpha=0.5, label='Threshold')
        ax1.set_xlabel('Circuit Depth', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Wigner Negativity', fontsize=12, fontweight='bold')
        ax1.set_title('Platform Comparison: Negativity Decay', 
                     fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Collapse depth bar chart
        ax2 = axes[1]
        platforms = [data['name'] for data in results.values()]
        collapse_depths = [data['collapse_depth'] for data in results.values()]
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(platforms)))
        
        bars = ax2.barh(platforms, collapse_depths, color=colors, edgecolor='black')
        ax2.set_xlabel('Collapse Depth (layers)', fontsize=12, fontweight='bold')
        ax2.set_title('Maximum Viable Circuit Depth', 
                     fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, depth) in enumerate(zip(bars, collapse_depths)):
            ax2.text(depth + 1, i, f'{depth}', 
                    va='center', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved platform comparison to {save_path}")
        
        return fig


def main():
    """
    Run experimental analysis suite
    """
    print("=" * 70)
    print("Experimental Analysis: Parameter Sweeps & Platform Comparison")
    print("=" * 70)
    print()
    
    # Parameter sweeps
    sweep = ParameterSweepAnalysis()
    
    print("Experiment 1: Loss vs Thermal Noise Sweep")
    print("-" * 70)
    sweep.sweep_loss_vs_thermal(
        loss_range=(0.001, 0.1, 15),
        thermal_range=(0.0001, 0.02, 15),
        target_depth=30
    )
    print()
    
    print("Experiment 2: Depth vs Loss Sweep")
    print("-" * 70)
    sweep.sweep_depth_vs_loss(
        depth_range=(0, 50, 51),
        loss_range=(0.001, 0.1, 12)
    )
    print()
    
    # Generate plots
    print("Generating parameter sweep visualizations...")
    sweep.plot_2d_sweep_results(save_prefix='/home/claude/experimental')
    sweep.save_results('/home/claude/sweep_results.json')
    print()
    
    # Platform comparison
    print("Experiment 3: Platform Comparison")
    print("-" * 70)
    platform_comp = PlatformComparison()
    platform_results = platform_comp.compare_platforms(max_depth=30)
    
    print("Generating platform comparison plots...")
    platform_comp.plot_platform_comparison(
        platform_results,
        save_path='/home/claude/platform_comparison.png'
    )
    print()
    
    print("=" * 70)
    print("Experimental analysis complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
