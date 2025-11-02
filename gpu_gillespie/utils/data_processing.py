"""
Data processing and visualization utilities for simulation results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

class SimulationAnalyzer:
    """
    Comprehensive analysis and visualization tools for simulation results
    """
    
    def __init__(self, results: Dict):
        """
        Initialize analyzer with simulation results
        
        Parameters:
        -----------
        results : Dict
            Simulation results from GPUGillespieSimulator
        """
        self.results = results
        self.trajectories = results['species_trajectories']
        self.time_points = results['time_points']
        self.species_names = results['species_names']
        self.n_trajectories = results['n_trajectories']
        self.n_species = len(self.species_names)
        self.n_timepoints = len(self.time_points)
    
    def calculate_statistics(self) -> Dict:
        """
        Calculate comprehensive statistics for all species
        
        Returns:
        --------
        Dict containing statistical analysis
        """
        stats_dict = {
            'species_stats': {},
            'correlation_analysis': {},
            'temporal_analysis': {},
            'distribution_analysis': {}
        }
        
        # Per-species statistics
        for i, species in enumerate(self.species_names):
            species_data = self.trajectories[:, :, i]
            
            stats_dict['species_stats'][species] = {
                'mean_trajectory': np.mean(species_data, axis=0),
                'std_trajectory': np.std(species_data, axis=0),
                'median_trajectory': np.median(species_data, axis=0),
                'q25_trajectory': np.percentile(species_data, 25, axis=0),
                'q75_trajectory': np.percentile(species_data, 75, axis=0),
                'min_trajectory': np.min(species_data, axis=0),
                'max_trajectory': np.max(species_data, axis=0),
                'coefficient_of_variation': (
                    np.std(species_data, axis=0) / 
                    (np.mean(species_data, axis=0) + 1e-10)
                ),
                'final_distribution': species_data[:, -1],
                'initial_final_correlation': np.corrcoef(
                    species_data[:, 0], species_data[:, -1]
                )[0, 1]
            }
        
        # Correlation analysis between species
        final_states = self.trajectories[:, -1, :]  # Final time point
        correlation_matrix = np.corrcoef(final_states.T)
        stats_dict['correlation_analysis']['final_state_correlations'] = correlation_matrix
        
        # Temporal correlation analysis
        temporal_corrs = {}
        for i, species1 in enumerate(self.species_names):
            temporal_corrs[species1] = {}
            for j, species2 in enumerate(self.species_names):
                if i != j:
                    # Calculate temporal cross-correlation
                    corr_coeffs = []
                    for t in range(self.n_timepoints):
                        corr = np.corrcoef(
                            self.trajectories[:, t, i],
                            self.trajectories[:, t, j]
                        )[0, 1]
                        corr_coeffs.append(corr)
                    temporal_corrs[species1][species2] = np.array(corr_coeffs)
        
        stats_dict['temporal_analysis']['cross_correlations'] = temporal_corrs
        
        # Distribution analysis for final states
        for i, species in enumerate(self.species_names):
            final_counts = self.trajectories[:, -1, i]
            
            # Test for normality
            shapiro_stat, shapiro_p = stats.shapiro(final_counts)
            
            # Calculate moments
            moments = {
                'mean': np.mean(final_counts),
                'variance': np.var(final_counts),
                'skewness': stats.skew(final_counts),
                'kurtosis': stats.kurtosis(final_counts),
                'shapiro_statistic': shapiro_stat,
                'shapiro_pvalue': shapiro_p,
                'is_normal': shapiro_p > 0.05
            }
            
            stats_dict['distribution_analysis'][species] = moments
        
        return stats_dict
    
    def plot_trajectory_overview(self, 
                               n_sample_trajectories: int = 10,
                               figsize: Tuple[int, int] = (15, 10),
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Create comprehensive trajectory overview plot
        
        Parameters:
        -----------
        n_sample_trajectories : int
            Number of individual trajectories to show
        figsize : Tuple[int, int]
            Figure size
        save_path : str, optional
            Path to save the plot
            
        Returns:
        --------
        matplotlib Figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        colors = plt.cm.Set3(np.linspace(0, 1, self.n_species))
        
        # Plot 1: Individual trajectories
        ax = axes[0]
        sample_indices = np.random.choice(
            self.n_trajectories, 
            min(n_sample_trajectories, self.n_trajectories), 
            replace=False
        )
        
        for i, species in enumerate(self.species_names):
            for idx in sample_indices:
                ax.plot(self.time_points, self.trajectories[idx, :, i], 
                       color=colors[i], alpha=0.3, linewidth=0.5)
            
            # Add mean trajectory
            mean_traj = np.mean(self.trajectories[:, :, i], axis=0)
            ax.plot(self.time_points, mean_traj, color=colors[i], 
                   linewidth=2, label=f'{species} (mean)')
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Species Count')
        ax.set_title('Sample Trajectories with Mean')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Mean trajectories with confidence intervals
        ax = axes[1]
        for i, species in enumerate(self.species_names):
            mean_traj = np.mean(self.trajectories[:, :, i], axis=0)
            std_traj = np.std(self.trajectories[:, :, i], axis=0)
            
            ax.plot(self.time_points, mean_traj, color=colors[i], 
                   linewidth=2, label=species)
            ax.fill_between(self.time_points, 
                           mean_traj - std_traj, 
                           mean_traj + std_traj,
                           color=colors[i], alpha=0.2)
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Species Count')
        ax.set_title('Mean Trajectories ± 1 Std')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Coefficient of variation over time
        ax = axes[2]
        for i, species in enumerate(self.species_names):
            mean_traj = np.mean(self.trajectories[:, :, i], axis=0)
            std_traj = np.std(self.trajectories[:, :, i], axis=0)
            cv = std_traj / (mean_traj + 1e-10)
            
            ax.plot(self.time_points, cv, color=colors[i], 
                   linewidth=2, label=species)
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Coefficient of Variation')
        ax.set_title('Temporal Variability (CV)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Final state distributions
        ax = axes[3]
        final_states = self.trajectories[:, -1, :]
        
        violin_data = []
        violin_labels = []
        for i, species in enumerate(self.species_names):
            violin_data.append(final_states[:, i])
            violin_labels.append(species)
        
        parts = ax.violinplot(violin_data, positions=range(self.n_species), 
                             showmeans=True, showmedians=True)
        
        # Color violin plots
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.7)
        
        ax.set_xticks(range(self.n_species))
        ax.set_xticklabels(violin_labels)
        ax.set_ylabel('Final Count')
        ax.set_title('Final State Distributions')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_correlation_analysis(self, figsize: Tuple[int, int] = (12, 8),
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        Create correlation analysis plots
        
        Parameters:
        -----------
        figsize : Tuple[int, int]
            Figure size
        save_path : str, optional
            Path to save the plot
            
        Returns:
        --------
        matplotlib Figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        # Calculate statistics if not already done
        stats = self.calculate_statistics()
        
        # Plot 1: Correlation heatmap
        ax = axes[0]
        corr_matrix = stats['correlation_analysis']['final_state_correlations']
        
        im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_xticks(range(self.n_species))
        ax.set_yticks(range(self.n_species))
        ax.set_xticklabels(self.species_names, rotation=45)
        ax.set_yticklabels(self.species_names)
        
        # Add correlation values
        for i in range(self.n_species):
            for j in range(self.n_species):
                text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                              ha="center", va="center", color="black")
        
        ax.set_title('Final State Correlations')
        plt.colorbar(im, ax=ax)
        
        # Plot 2: Scatter plot matrix
        ax = axes[1]
        if self.n_species >= 2:
            # Show correlation between first two species
            species1_data = self.trajectories[:, -1, 0]
            species2_data = self.trajectories[:, -1, 1]
            
            ax.scatter(species1_data, species2_data, alpha=0.6, s=10)
            
            # Add correlation coefficient
            corr_coef = np.corrcoef(species1_data, species2_data)[0, 1]
            ax.set_xlabel(f'{self.species_names[0]} Final Count')
            ax.set_ylabel(f'{self.species_names[1]} Final Count')
            ax.set_title(f'Correlation: r = {corr_coef:.3f}')
            ax.grid(True, alpha=0.3)
        
        # Plot 3: Temporal correlations
        ax = axes[2]
        if self.n_species >= 2:
            temporal_corrs = stats['temporal_analysis']['cross_correlations']
            if self.species_names[0] in temporal_corrs:
                if self.species_names[1] in temporal_corrs[self.species_names[0]]:
                    corr_series = temporal_corrs[self.species_names[0]][self.species_names[1]]
                    ax.plot(self.time_points, corr_series, linewidth=2)
                    ax.set_xlabel('Time')
                    ax.set_ylabel('Correlation Coefficient')
                    ax.set_title(f'Temporal Correlation: {self.species_names[0]} vs {self.species_names[1]}')
                    ax.grid(True, alpha=0.3)
        
        # Plot 4: Distribution comparison
        ax = axes[3]
        final_states = self.trajectories[:, -1, :]
        
        for i, species in enumerate(self.species_names):
            ax.hist(final_states[:, i], bins=30, alpha=0.6, 
                   label=species, density=True)
        
        ax.set_xlabel('Final Count')
        ax.set_ylabel('Density')
        ax.set_title('Final State Distributions')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_phase_space(self, species_x: str, species_y: str,
                        time_points: Optional[List[int]] = None,
                        figsize: Tuple[int, int] = (10, 8),
                        save_path: Optional[str] = None) -> plt.Figure:
        """
        Create phase space plot for two species
        
        Parameters:
        -----------
        species_x : str
            Species for x-axis
        species_y : str
            Species for y-axis
        time_points : List[int], optional
            Specific time points to plot (default: all)
        figsize : Tuple[int, int]
            Figure size
        save_path : str, optional
            Path to save the plot
            
        Returns:
        --------
        matplotlib Figure object
        """
        if species_x not in self.species_names or species_y not in self.species_names:
            raise ValueError("Species not found in simulation results")
        
        x_idx = self.species_names.index(species_x)
        y_idx = self.species_names.index(species_y)
        
        if time_points is None:
            time_points = list(range(self.n_timepoints))
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Plot 1: Phase space trajectories
        ax = axes[0]
        
        # Plot sample trajectories
        sample_indices = np.random.choice(
            self.n_trajectories, 
            min(20, self.n_trajectories), 
            replace=False
        )
        
        for idx in sample_indices:
            x_vals = self.trajectories[idx, time_points, x_idx]
            y_vals = self.trajectories[idx, time_points, y_idx]
            ax.plot(x_vals, y_vals, alpha=0.6, linewidth=1)
        
        ax.set_xlabel(f'{species_x} Count')
        ax.set_ylabel(f'{species_y} Count')
        ax.set_title(f'Phase Space: {species_x} vs {species_y}')
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Time-colored phase space
        ax = axes[1]
        
        # Use a single trajectory colored by time
        traj_idx = 0
        x_vals = self.trajectories[traj_idx, :, x_idx]
        y_vals = self.trajectories[traj_idx, :, y_idx]
        
        # Create scatter plot colored by time
        scatter = ax.scatter(x_vals, y_vals, c=self.time_points, 
                           cmap='viridis', s=20, alpha=0.7)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Time')
        
        ax.set_xlabel(f'{species_x} Count')
        ax.set_ylabel(f'{species_y} Count')
        ax.set_title(f'Time-Colored Phase Space (Trajectory {traj_idx})')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def export_to_csv(self, filename: str, include_trajectories: bool = True):
        """
        Export simulation data to CSV format
        
        Parameters:
        -----------
        filename : str
            Output filename
        include_trajectories : bool
            Whether to include all trajectory data
        """
        if include_trajectories:
            # Export trajectory data
            rows = []
            for traj_idx in range(self.n_trajectories):
                for time_idx, time_point in enumerate(self.time_points):
                    row = {
                        'trajectory': traj_idx,
                        'time': time_point
                    }
                    for i, species in enumerate(self.species_names):
                        row[species] = self.trajectories[traj_idx, time_idx, i]
                    rows.append(row)
            
            df = pd.DataFrame(rows)
            df.to_csv(filename, index=False)
        else:
            # Export summary statistics only
            stats = self.calculate_statistics()
            rows = []
            
            for time_idx, time_point in enumerate(self.time_points):
                row = {'time': time_point}
                for species in self.species_names:
                    species_stats = stats['species_stats'][species]
                    row[f'{species}_mean'] = species_stats['mean_trajectory'][time_idx]
                    row[f'{species}_std'] = species_stats['std_trajectory'][time_idx]
                    row[f'{species}_median'] = species_stats['median_trajectory'][time_idx]
                    row[f'{species}_q25'] = species_stats['q25_trajectory'][time_idx]
                    row[f'{species}_q75'] = species_stats['q75_trajectory'][time_idx]
                rows.append(row)
            
            df = pd.DataFrame(rows)
            df.to_csv(filename, index=False)
    
    def detect_extinction_events(self, species: str, threshold: int = 0) -> Dict:
        """
        Detect and analyze extinction events for a species
        
        Parameters:
        -----------
        species : str
            Species to analyze
        threshold : int
            Count threshold for extinction (default: 0)
            
        Returns:
        --------
        Dict containing extinction analysis
        """
        if species not in self.species_names:
            raise ValueError(f"Species {species} not found")
        
        species_idx = self.species_names.index(species)
        species_data = self.trajectories[:, :, species_idx]
        
        extinction_analysis = {
            'species': species,
            'threshold': threshold,
            'extinction_events': [],
            'survival_times': [],
            'extinction_fraction': 0.0
        }
        
        for traj_idx in range(self.n_trajectories):
            trajectory = species_data[traj_idx, :]
            
            # Find when extinction occurs
            extinction_indices = np.where(trajectory <= threshold)[0]
            
            if len(extinction_indices) > 0:
                extinction_time = self.time_points[extinction_indices[0]]
                extinction_analysis['extinction_events'].append({
                    'trajectory': traj_idx,
                    'extinction_time': extinction_time,
                    'extinction_index': extinction_indices[0]
                })
                extinction_analysis['survival_times'].append(extinction_time)
            else:
                # Never went extinct
                extinction_analysis['survival_times'].append(self.time_points[-1])
        
        extinction_analysis['extinction_fraction'] = (
            len(extinction_analysis['extinction_events']) / self.n_trajectories
        )
        
        if extinction_analysis['survival_times']:
            extinction_analysis['mean_survival_time'] = np.mean(
                extinction_analysis['survival_times']
            )
            extinction_analysis['median_survival_time'] = np.median(
                extinction_analysis['survival_times']
            )
        
        return extinction_analysis