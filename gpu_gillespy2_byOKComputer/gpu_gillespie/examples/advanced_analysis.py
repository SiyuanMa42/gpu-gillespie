#!/usr/bin/env python3
"""
Advanced analysis example
Demonstrates comprehensive statistical analysis and visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from gpu_gillespie import GeneExpressionModel, SimulationAnalyzer

def main():
    print("GPU-Gillespie Advanced Analysis Example")
    print("=" * 45)
    
    # Create gene expression model
    print("\n1. Setting up gene expression model...")
    model = GeneExpressionModel(
        k_transcription=0.1,
        k_translation=0.2,
        k_mrna_degradation=0.05,
        k_protein_degradation=0.01,
        initial_gene=1,
        initial_mrna=0,
        initial_protein=0
    )
    
    print(f"   Theoretical mRNA steady state: {model.get_steady_state_mrna():.1f}")
    print(f"   Theoretical protein steady state: {model.get_steady_state_protein():.1f}")
    
    # Run simulation
    print("\n2. Running simulation...")
    results = model.run_simulation(
        time_span=(0, 200),
        n_timepoints=201,
        n_trajectories=5000
    )
    
    print(f"   ✓ Simulation completed!")
    print(f"   Execution time: {results['performance_stats']['execution_time']:.2f} seconds")
    print(f"   Speedup: {results['performance_stats']['speedup_factor']:.1f}x")
    
    # Advanced analysis
    print("\n3. Performing advanced analysis...")
    analyzer = SimulationAnalyzer(results)
    stats = analyzer.calculate_statistics()
    
    # Create comprehensive visualization
    print("\n4. Creating comprehensive visualization...")
    fig = analyzer.plot_trajectory_overview(
        n_sample_trajectories=50,
        figsize=(16, 12)
    )
    plt.savefig('gene_expression_overview.png', dpi=300, bbox_inches='tight')
    print("   ✓ Overview plot saved")
    
    # Correlation analysis
    print("\n5. Correlation analysis...")
    corr_fig = analyzer.plot_correlation_analysis(figsize=(14, 10))
    plt.savefig('gene_expression_correlations.png', dpi=300, bbox_inches='tight')
    print("   ✓ Correlation analysis saved")
    
    # Phase space analysis
    print("\n6. Phase space analysis...")
    phase_fig = analyzer.plot_phase_space(
        'mRNA', 'Protein',
        figsize=(12, 8)
    )
    plt.savefig('gene_expression_phasespace.png', dpi=300, bbox_inches='tight')
    print("   ✓ Phase space analysis saved")
    
    # Statistical analysis
    print("\n7. Statistical analysis...")
    
    # Print key statistics
    for species in results['species_names']:
        if species in stats['species_stats']:
            species_stats = stats['species_stats'][species]
            
            print(f"\n   {species} Statistics:")
            print(f"     Final mean: {species_stats['mean_trajectory'][-1]:.2f}")
            print(f"     Final std: {species_stats['std_trajectory'][-1]:.2f}")
            print(f"     CV at steady state: {species_stats['coefficient_of_variation'][-1]:.3f}")
            print(f"     Initial-final correlation: {species_stats['initial_final_correlation']:.3f}")
        
        if species in stats['distribution_analysis']:
            dist_stats = stats['distribution_analysis'][species]
            print(f"     Distribution normality: {'Normal' if dist_stats['is_normal'] else 'Non-normal'}")
            print(f"     Skewness: {dist_stats['skewness']:.3f}")
            print(f"     Kurtosis: {dist_stats['kurtosis']:.3f}")
    
    # Cross-correlation analysis
    print("\n8. Cross-correlation analysis...")
    temporal_corrs = stats['temporal_analysis']['cross_correlations']
    
    if 'mRNA' in temporal_corrs and 'Protein' in temporal_corrs['mRNA']:
        mrna_protein_corr = temporal_corrs['mRNA']['Protein']
        max_corr_time = results['time_points'][np.argmax(np.abs(mrna_protein_corr))]
        max_corr_value = np.max(np.abs(mrna_protein_corr))
        
        print(f"   Maximum mRNA-Protein correlation: {max_corr_value:.3f}")
        print(f"   Occurs at time: {max_corr_time:.1f}")
        
        # Plot temporal correlation
        plt.figure(figsize=(10, 6))
        plt.plot(results['time_points'], mrna_protein_corr, linewidth=2)
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        plt.xlabel('Time')
        plt.ylabel('Correlation Coefficient')
        plt.title('Temporal Cross-Correlation: mRNA vs Protein')
        plt.grid(True, alpha=0.3)
        plt.savefig('temporal_correlation.png', dpi=300, bbox_inches='tight')
        print("   ✓ Temporal correlation plot saved")
    
    # Noise analysis
    print("\n9. Noise analysis...")
    
    # Calculate noise levels (coefficient of variation)
    noise_analysis = {}
    for i, species in enumerate(results['species_names']):
        final_counts = results['species_trajectories'][:, -1, i]
        mean_count = np.mean(final_counts)
        std_count = np.std(final_counts)
        noise_level = std_count / mean_count if mean_count > 0 else 0
        
        noise_analysis[species] = {
            'mean': mean_count,
            'std': std_count,
            'noise_level': noise_level,
            'fano_factor': std_count**2 / mean_count if mean_count > 0 else 0
        }
        
        print(f"   {species} noise level (CV): {noise_level:.3f}")
        print(f"   {species} Fano factor: {noise_analysis[species]['fano_factor']:.3f}")
    
    # Extinction analysis
    print("\n10. Extinction analysis...")
    
    for species in ['mRNA', 'Protein']:
        try:
            extinction_analysis = analyzer.detect_extinction_events(species, threshold=0)
            
            print(f"\n   {species} Extinction Analysis:")
            print(f"     Extinction fraction: {extinction_analysis['extinction_fraction']:.3f}")
            
            if extinction_analysis['survival_times']:
                print(f"     Mean survival time: {extinction_analysis['mean_survival_time']:.1f}")
                print(f"     Median survival time: {extinction_analysis['median_survival_time']:.1f}")
        
        except Exception as e:
            print(f"   Could not analyze {species} extinction: {e}")
    
    # Export comprehensive results
    print("\n11. Exporting comprehensive results...")
    
    # Export trajectory data
    analyzer.export_to_csv('gene_expression_trajectories.csv', include_trajectories=True)
    
    # Export statistics
    stats_df = []
    for species in results['species_names']:
        if species in stats['species_stats']:
            species_stats = stats['species_stats'][species]
            for t_idx, time_point in enumerate(results['time_points']):
                stats_df.append({
                    'time': time_point,
                    'species': species,
                    'mean': species_stats['mean_trajectory'][t_idx],
                    'std': species_stats['std_trajectory'][t_idx],
                    'median': species_stats['median_trajectory'][t_idx],
                    'q25': species_stats['q25_trajectory'][t_idx],
                    'q75': species_stats['q75_trajectory'][t_idx],
                    'cv': species_stats['coefficient_of_variation'][t_idx]
                })
    
    stats_df = pd.DataFrame(stats_df)
    stats_df.to_csv('gene_expression_statistics.csv', index=False)
    
    # Export noise analysis
    noise_df = pd.DataFrame(noise_analysis).T
    noise_df.to_csv('gene_expression_noise_analysis.csv')
    
    print("   ✓ All results exported to CSV files")
    
    print("\n" + "=" * 45)
    print("Advanced analysis completed!")
    print("Check generated files for detailed results.")

if __name__ == "__main__":
    main()