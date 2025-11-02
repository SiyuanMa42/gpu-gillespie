#!/usr/bin/env python3
"""
Basic usage example for GPU-Gillespie package
Demonstrates simple dimerization model simulation
"""

import numpy as np
import matplotlib.pyplot as plt
from gpu_gillespie import DimerizationModel

def main():
    print("GPU-Gillespie Basic Usage Example")
    print("=" * 40)
    
    # Create a simple dimerization model
    # 2A ⇌ A₂
    print("\n1. Creating dimerization model...")
    model = DimerizationModel(
        k_dimerization=0.1,    # Rate of dimerization
        k_dissociation=0.01,   # Rate of dissociation
        initial_A=1000,        # Starting with 1000 monomers
        initial_A2=0           # No dimers initially
    )
    
    print(f"   Model: 2A ⇌ A₂")
    print(f"   k_dimerization: {model.k_dimerization}")
    print(f"   k_dissociation: {model.k_dissociation}")
    print(f"   Initial A: 1000, Initial A₂: 0")
    
    # Run GPU-accelerated simulation
    print("\n2. Running GPU simulation...")
    results = model.run_simulation(
        time_span=(0, 100),        # Simulate for 100 time units
        n_timepoints=101,          # Output 101 time points
        n_trajectories=10000       # Run 10,000 parallel trajectories
    )
    
    # Display performance metrics
    print(f"   ✓ Simulation completed!")
    print(f"   Execution time: {results['performance_stats']['execution_time']:.3f} seconds")
    print(f"   Speedup vs CPU: {results['performance_stats']['speedup_factor']:.1f}x")
    print(f"   Trajectories per second: {10000/results['performance_stats']['execution_time']:.0f}")
    
    # Analyze results
    print("\n3. Analyzing results...")
    analysis = model.simulator.analyze_results(results)
    
    # Print equilibrium analysis
    print(f"   Equilibrium constant (theoretical): {model.get_equilibrium_constant():.3f}")
    
    for species in ['A', 'A2']:
        final_stats = analysis['final_distributions'][species]
        print(f"   {species} final count - Mean: {final_stats['mean']:.1f}, Std: {final_stats['std']:.1f}")
        print(f"   {species} extinction fraction: {final_stats['extinct_fraction']:.3f}")
    
    # Create visualization
    print("\n4. Creating visualization...")
    fig = model.simulator.plot_trajectories(
        results,
        n_sample_trajectories=20,
        show_plot=False
    )
    
    # Add equilibrium line
    ax = fig.axes[0]
    ax.axhline(y=500, color='red', linestyle='--', alpha=0.7, 
               label='Theoretical equilibrium (A)')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('dimerization_example.png', dpi=300, bbox_inches='tight')
    print("   ✓ Plot saved as 'dimerization_example.png'")
    
    # Export data
    print("\n5. Exporting data...")
    analyzer = model.simulator.SimulationAnalyzer(results)
    analyzer.export_to_csv('dimerization_results.csv', include_trajectories=False)
    print("   ✓ Data exported to 'dimerization_results.csv'")
    
    print("\n" + "=" * 40)
    print("Example completed successfully!")
    print("Check the generated files for results and visualization.")

if __name__ == "__main__":
    main()