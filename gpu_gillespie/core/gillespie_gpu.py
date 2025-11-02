"""
GPU-accelerated Gillespie simulator main class
Provides high-level interface for stochastic simulation
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple, Union
from .cuda_kernels import launch_gillespie_simulation
from ..utils.performance_metrics import PerformanceMonitor

class GPUGillespieSimulator:
    """
    High-level interface for GPU-accelerated Gillespie stochastic simulation
    """
    
    def __init__(self, 
                 species_names: List[str],
                 reaction_names: List[str],
                 stoichiometry: np.ndarray,
                 rate_constants: Union[List[float], np.ndarray],
                 initial_conditions: Dict[str, int] = None):
        """
        Initialize the GPU Gillespie simulator
        
        Parameters:
        -----------
        species_names : List[str]
            Names of chemical species
        reaction_names : List[str] 
            Names of reactions
        stoichiometry : np.ndarray
            Stoichiometry matrix [n_reactions, n_species]
        rate_constants : List[float] or np.ndarray
            Reaction rate constants
        initial_conditions : Dict[str, int], optional
            Initial species counts (default: all zeros)
        """
        self.species_names = species_names
        self.reaction_names = reaction_names
        self.stoichiometry = np.array(stoichiometry, dtype=np.int32)
        self.rate_constants = np.array(rate_constants, dtype=np.float32)
        self.n_species = len(species_names)
        self.n_reactions = len(reaction_names)
        
        # Set initial conditions
        if initial_conditions is None:
            self.initial_conditions = np.zeros(self.n_species, dtype=np.int32)
        else:
            self.initial_conditions = np.array([
                initial_conditions.get(name, 0) for name in species_names
            ], dtype=np.int32)
            
        # Validate inputs
        if self.stoichiometry.shape != (self.n_reactions, self.n_species):
            raise ValueError(f"Stoichiometry matrix shape {self.stoichiometry.shape} "
                           f"doesn't match ({self.n_reactions}, {self.n_species})")
        
        if len(self.rate_constants) != self.n_reactions:
            raise ValueError(f"Number of rate constants {len(self.rate_constants)} "
                           f"doesn't match number of reactions {self.n_reactions}")
        
        self.performance_monitor = PerformanceMonitor()
    
    def run_simulation(self, 
                      time_span: Union[List[float], Tuple[float, float]],
                      n_timepoints: int = 101,
                      n_trajectories: int = 1000,
                      use_tau_leaping: bool = False,
                      seed: Optional[int] = None) -> Dict:
        """
        Run GPU-accelerated Gillespie simulation
        
        Parameters:
        -----------
        time_span : List[float] or Tuple[float, float]
            Time span [t_start, t_end] or list of time points
        n_timepoints : int
            Number of time points for output
        n_trajectories : int
            Number of parallel trajectories to simulate
        use_tau_leaping : bool
            Use tau-leaping approximation for faster simulation
        seed : int, optional
            Random seed for reproducibility
            
        Returns:
        --------
        Dict containing simulation results and metadata
        """
        # Prepare time points
        if isinstance(time_span, (list, tuple)) and len(time_span) == 2:
            time_points = np.linspace(time_span[0], time_span[1], n_timepoints)
        else:
            time_points = np.array(time_span)
            n_timepoints = len(time_points)
        
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        # Start performance monitoring
        self.performance_monitor.start_monitoring()
        
        try:
            # Launch GPU simulation
            results = launch_gillespie_simulation(
                initial_species=self.initial_conditions,
                stoichiometry=self.stoichiometry,
                rate_constants=self.rate_constants,
                time_points=time_points,
                n_trajectories=n_trajectories,
                use_tau_leaping=use_tau_leaping
            )
            
            # Stop performance monitoring
            performance_stats = self.performance_monitor.stop_monitoring()
            
            # Package results
            simulation_results = {
                'time_points': time_points,
                'species_trajectories': results,
                'species_names': self.species_names,
                'reaction_names': self.reaction_names,
                'n_trajectories': n_trajectories,
                'n_species': self.n_species,
                'n_reactions': self.n_reactions,
                'use_tau_leaping': use_tau_leaping,
                'performance_stats': performance_stats,
                'initial_conditions': self.initial_conditions,
                'rate_constants': self.rate_constants
            }
            
            return simulation_results
            
        except Exception as e:
            self.performance_monitor.stop_monitoring()
            raise RuntimeError(f"GPU simulation failed: {str(e)}")
    
    def analyze_results(self, results: Dict) -> Dict:
        """
        Analyze simulation results
        
        Parameters:
        -----------
        results : Dict
            Simulation results from run_simulation()
            
        Returns:
        --------
        Dict containing analysis results
        """
        trajectories = results['species_trajectories']
        time_points = results['time_points']
        
        analysis = {
            'mean_trajectories': {},
            'std_trajectories': {},
            'final_distributions': {},
            'extinction_times': {},
            'equilibrium_analysis': {}
        }
        
        for i, species_name in enumerate(results['species_names']):
            # Calculate mean and standard deviation
            mean_traj = np.mean(trajectories[:, :, i], axis=0)
            std_traj = np.std(trajectories[:, :, i], axis=0)
            
            analysis['mean_trajectories'][species_name] = mean_traj
            analysis['std_trajectories'][species_name] = std_traj
            
            # Final distribution analysis
            final_counts = trajectories[:, -1, i]
            analysis['final_distributions'][species_name] = {
                'mean': np.mean(final_counts),
                'std': np.std(final_counts),
                'min': np.min(final_counts),
                'max': np.max(final_counts),
                'extinct_fraction': np.sum(final_counts == 0) / len(final_counts)
            }
            
            # Extinction time analysis
            extinction_times = []
            for traj in range(trajectories.shape[0]):
                zero_indices = np.where(trajectories[traj, :, i] == 0)[0]
                if len(zero_indices) > 0:
                    extinction_times.append(time_points[zero_indices[0]])
            
            if extinction_times:
                analysis['extinction_times'][species_name] = {
                    'mean': np.mean(extinction_times),
                    'std': np.std(extinction_times),
                    'fraction_extinct': len(extinction_times) / trajectories.shape[0]
                }
        
        return analysis
    
    def plot_trajectories(self, 
                         results: Dict, 
                         species_subset: List[str] = None,
                         n_sample_trajectories: int = 10,
                         save_path: Optional[str] = None,
                         show_plot: bool = True) -> plt.Figure:
        """
        Plot simulation trajectories
        
        Parameters:
        -----------
        results : Dict
            Simulation results
        species_subset : List[str], optional
            Subset of species to plot
        n_sample_trajectories : int
            Number of sample trajectories to show
        save_path : str, optional
            Path to save the plot
        show_plot : bool
            Whether to display the plot
            
        Returns:
        --------
        matplotlib Figure object
        """
        if species_subset is None:
            species_subset = results['species_names']
        
        # Create figure with subplots
        n_species_to_plot = len(species_subset)
        fig, axes = plt.subplots(n_species_to_plot, 1, 
                               figsize=(12, 4 * n_species_to_plot))
        
        if n_species_to_plot == 1:
            axes = [axes]
        
        time_points = results['time_points']
        trajectories = results['species_trajectories']
        
        colors = plt.cm.viridis(np.linspace(0, 1, n_sample_trajectories))
        
        for i, species_name in enumerate(species_subset):
            species_idx = results['species_names'].index(species_name)
            
            # Plot sample trajectories
            sample_indices = np.random.choice(
                trajectories.shape[0], 
                min(n_sample_trajectories, trajectories.shape[0]), 
                replace=False
            )
            
            for j, traj_idx in enumerate(sample_indices):
                axes[i].plot(time_points, trajectories[traj_idx, :, species_idx],
                           color=colors[j], alpha=0.6, linewidth=1)
            
            # Plot mean trajectory
            mean_traj = np.mean(trajectories[:, :, species_idx], axis=0)
            axes[i].plot(time_points, mean_traj, 'k-', linewidth=3, 
                        label='Mean', alpha=0.8)
            
            # Add confidence interval
            std_traj = np.std(trajectories[:, :, species_idx], axis=0)
            axes[i].fill_between(time_points, 
                               mean_traj - std_traj, 
                               mean_traj + std_traj,
                               alpha=0.3, color='gray', label='±1 std')
            
            axes[i].set_xlabel('Time')
            axes[i].set_ylabel(f'Count ({species_name})')
            axes[i].set_title(f'Trajectories for {species_name}')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show_plot:
            plt.show()
        
        return fig
    
    def get_performance_summary(self, results: Dict) -> str:
        """
        Generate performance summary from simulation results
        
        Parameters:
        -----------
        results : Dict
            Simulation results
            
        Returns:
        --------
        str: Formatted performance summary
        """
        stats = results['performance_stats']
        
        summary = f"""
GPU-Gillespie Performance Summary
================================

Simulation Parameters:
- Number of trajectories: {results['n_trajectories']:,}
- Number of species: {results['n_species']}
- Number of reactions: {results['n_reactions']}
- Time points: {len(results['time_points'])}
- Algorithm: {'Tau-leaping' if results['use_tau_leaping'] else 'Exact Gillespie'}

Performance Metrics:
- Total execution time: {stats['execution_time']:.3f} seconds
- Trajectories per second: {results['n_trajectories']/stats['execution_time']:.1f}
- Memory usage: {stats['memory_usage_mb']:.1f} MB
- GPU utilization: {stats['gpu_utilization']:.1f}%

Speedup Analysis:
- Estimated CPU time: {stats['estimated_cpu_time']:.1f} seconds
- Achieved speedup: {stats['speedup_factor']:.1f}x
        """
        
        return summary