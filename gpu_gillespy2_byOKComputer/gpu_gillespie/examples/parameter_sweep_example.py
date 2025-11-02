#!/usr/bin/env python3
"""
Parameter sweep example
Demonstrates how to perform parameter sensitivity analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from gpu_gillespie import EnzymeKineticsModel, ParallelSimulator

def main():
    print("GPU-Gillespie Parameter Sweep Example")
    print("=" * 45)
    
    # Create enzyme kinetics model
    print("\n1. Setting up enzyme kinetics model...")
    base_model = EnzymeKineticsModel(
        k_binding=1.0,
        k_dissociation=0.1,
        k_cat=0.5,
        initial_E=100,
        initial_S=1000,
        initial_ES=0,
        initial_P=0
    )
    
    print(f"   Base Michaelis constant (Km): {base_model.get_michaelis_constant():.3f}")
    print(f"   Base maximum velocity (Vmax): {base_model.get_max_velocity(100):.1f}")
    
    # Create parallel simulator
    print("\n2. Setting up parallel simulator...")
    parallel_sim = ParallelSimulator(base_model.simulator)
    
    # Define parameter sweep
    print("\n3. Running parameter sweep...")
    
    # Sweep k_cat (catalytic rate constant)
    k_cat_values = np.logspace(-1, 1, 20)  # From 0.1 to 10
    
    sweep_results = parallel_sim.parameter_sweep(
        parameter_name='k_cat',
        parameter_values=k_cat_values,
        fixed_parameters={
            'k_binding': 1.0,
            'k_dissociation': 0.1
        },
        time_span=(0, 100),
        n_timepoints=101,
        n_trajectories_per_point=2000,
        n_workers=4
    )
    
    print(f"   ✓ Parameter sweep completed!")
    print(f"   Tested {len(k_cat_values)} values of k_cat")
    print(f"   Range: {k_cat_values[0]:.3f} to {k_cat_values[-1]:.3f}")
    
    # Export results
    print("\n4. Exporting results...")
    df = parallel_sim.export_to_dataframe(sweep_results, analysis_type='parameter_sweep')
    df.to_csv('enzyme_kcat_sweep.csv', index=False)
    print("   ✓ Results exported to 'enzyme_kcat_sweep.csv'")
    
    # Create visualization
    print("\n5. Creating visualization...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Final product vs k_cat
    ax = axes[0, 0]
    
    # Extract data from sweep results
    k_cat_vals = []
    product_means = []
    product_stds = []
    
    for result in sweep_results['sweep_results']:
        k_cat = result['parameter_value']
        analysis = result['analysis_results']
        
        if 'P' in analysis['final_distributions']:
            product_stats = analysis['final_distributions']['P']
            k_cat_vals.append(k_cat)
            product_means.append(product_stats['mean'])
            product_stds.append(product_stats['std'])
    
    k_cat_vals = np.array(k_cat_vals)
    product_means = np.array(product_means)
    product_stds = np.array(product_stds)
    
    ax.errorbar(k_cat_vals, product_means, yerr=product_stds, 
                fmt='o-', capsize=3, capthick=1, markersize=4)
    ax.set_xlabel('k_cat (catalytic rate constant)')
    ax.set_ylabel('Final Product Count')
    ax.set_title('Product Formation vs Catalytic Rate')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: ES complex vs k_cat
    ax = axes[0, 1]
    
    es_means = []
    es_stds = []
    
    for result in sweep_results['sweep_results']:
        analysis = result['analysis_results']
        if 'ES' in analysis['final_distributions']:
            es_stats = analysis['final_distributions']['ES']
            es_means.append(es_stats['mean'])
            es_stds.append(es_stats['std'])
    
    es_means = np.array(es_means)
    es_stds = np.array(es_stds)
    
    ax.errorbar(k_cat_vals, es_means, yerr=es_stds, 
                fmt='s-', capsize=3, capthick=1, markersize=4, color='orange')
    ax.set_xlabel('k_cat (catalytic rate constant)')
    ax.set_ylabel('Final ES Complex Count')
    ax.set_title('ES Complex vs Catalytic Rate')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Substrate depletion vs k_cat
    ax = axes[1, 0]
    
    s_means = []
    s_stds = []
    
    for result in sweep_results['sweep_results']:
        analysis = result['analysis_results']
        if 'S' in analysis['final_distributions']:
            s_stats = analysis['final_distributions']['S']
            s_means.append(s_stats['mean'])
            s_stds.append(s_stats['std'])
    
    s_means = np.array(s_means)
    s_stds = np.array(s_stds)
    
    ax.errorbar(k_cat_vals, s_means, yerr=s_stds, 
                fmt='^-', capsize=3, capthick=1, markersize=4, color='green')
    ax.set_xlabel('k_cat (catalytic rate constant)')
    ax.set_ylabel('Final Substrate Count')
    ax.set_title('Substrate Depletion vs Catalytic Rate')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Reaction velocity analysis
    ax = axes[1, 1]
    
    # Calculate initial velocity approximation
    initial_velocity = k_cat_vals * 100  # Approximate: V ≈ k_cat * [E_total]
    
    ax.plot(k_cat_vals, initial_velocity, 'r-', linewidth=2, label='Theoretical Vmax')
    
    # Calculate observed velocity from simulation
    observed_velocity = (1000 - s_means) / 100  # (S_initial - S_final) / time
    ax.scatter(k_cat_vals, observed_velocity, alpha=0.7, s=30, 
              label='Observed velocity', color='blue')
    
    ax.set_xlabel('k_cat (catalytic rate constant)')
    ax.set_ylabel('Reaction Velocity')
    ax.set_title('Michaelis-Menten Kinetics')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('enzyme_kcat_sweep.png', dpi=300, bbox_inches='tight')
    print("   ✓ Visualization saved as 'enzyme_kcat_sweep.png'")
    
    # Advanced analysis
    print("\n6. Advanced analysis...")
    
    # Find optimal k_cat value
    optimal_idx = np.argmax(product_means)
    optimal_k_cat = k_cat_vals[optimal_idx]
    max_product = product_means[optimal_idx]
    
    print(f"   Optimal k_cat for product formation: {optimal_k_cat:.3f}")
    print(f"   Maximum product formed: {max_product:.1f} molecules")
    
    # Calculate catalytic efficiency
    catalytic_efficiency = product_means / k_cat_vals
    max_efficiency_idx = np.argmax(catalytic_efficiency)
    
    print(f"   Maximum catalytic efficiency at k_cat: {k_cat_vals[max_efficiency_idx]:.3f}")
    print(f"   Efficiency: {catalytic_efficiency[max_efficiency_idx]:.1f} products per k_cat")
    
    # Sensitivity analysis
    sensitivity = np.gradient(product_means, k_cat_vals)
    max_sensitivity_idx = np.argmax(np.abs(sensitivity))
    
    print(f"   Maximum sensitivity at k_cat: {k_cat_vals[max_sensitivity_idx]:.3f}")
    print(f"   Sensitivity: {sensitivity[max_sensitivity_idx]:.1f} products per k_cat unit")
    
    print("\n" + "=" * 45)
    print("Parameter sweep analysis completed!")
    print("Results saved in CSV and PNG files.")

if __name__ == "__main__":
    main()