#!/usr/bin/env python3
"""
Performance comparison example
Compares GPU vs CPU performance across different trajectory counts
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from gpu_gillespie import DimerizationModel, BenchmarkSuite
from gpu_gillespie.utils.performance_metrics import PerformanceMonitor

def cpu_gillespie_simulation(initial_A, initial_A2, k_dimerization, k_dissociation, 
                           time_span, n_timepoints, n_trajectories):
    """
    Simple CPU implementation for comparison
    Note: This is a basic implementation for demonstration
    """
    from scipy.stats import expon
    
    dt = (time_span[1] - time_span[0]) / (n_timepoints - 1)
    time_points = np.linspace(time_span[0], time_span[1], n_timepoints)
    
    all_trajectories = []
    
    for _ in range(n_trajectories):
        # Initialize
        A = initial_A
        A2 = initial_A2
        t = 0.0
        
        trajectory = np.zeros((n_timepoints, 2))
        time_idx = 0
        
        while t < time_span[1] and time_idx < n_timepoints:
            # Store current state
            if t >= time_points[time_idx]:
                trajectory[time_idx] = [A, A2]
                time_idx += 1
            
            # Calculate propensities
            prop_dimer = k_dimerization * A * (A - 1) / 2
            prop_dissoc = k_dissociation * A2
            total_prop = prop_dimer + prop_dissoc
            
            if total_prop <= 0:
                # Fill remaining time points
                while time_idx < n_timepoints:
                    trajectory[time_idx] = [A, A2]
                    time_idx += 1
                break
            
            # Time to next reaction
            tau = expon.rvs(scale=1/total_prop)
            t += tau
            
            # Which reaction occurs
            if np.random.random() < prop_dimer / total_prop:
                # Dimerization
                A -= 2
                A2 += 1
            else:
                # Dissociation
                A += 2
                A2 -= 1
        
        all_trajectories.append(trajectory)
    
    return np.array(all_trajectories)

def main():
    print("GPU-Gillespie Performance Comparison")
    print("=" * 50)
    
    # Test configurations
    trajectory_counts = [100, 500, 1000, 2000, 5000, 10000]
    results = []
    
    print(f"{'Trajectories':>12} {'CPU Time':>10} {'GPU Time':>10} {'Speedup':>10}")
    print("-" * 50)
    
    for n_traj in trajectory_counts:
        print(f"{n_traj:>12,}", end=" ", flush=True)
        
        # CPU simulation
        start_time = time.time()
        cpu_results = cpu_gillespie_simulation(
            initial_A=1000,
            initial_A2=0,
            k_dimerization=0.1,
            k_dissociation=0.01,
            time_span=(0, 50),
            n_timepoints=51,
            n_trajectories=n_traj
        )
        cpu_time = time.time() - start_time
        print(f"{cpu_time:>9.3f}s", end=" ", flush=True)
        
        # GPU simulation
        model = DimerizationModel(
            k_dimerization=0.1,
            k_dissociation=0.01,
            initial_A=1000,
            initial_A2=0
        )
        
        gpu_results = model.run_simulation(
            time_span=(0, 50),
            n_timepoints=51,
            n_trajectories=n_traj
        )
        gpu_time = gpu_results['performance_stats']['execution_time']
        print(f"{gpu_time:>9.3f}s", end=" ", flush=True)
        
        # Calculate speedup
        speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')
        print(f"{speedup:>9.1f}x")
        
        # Store results
        results.append({
            'n_trajectories': n_traj,
            'cpu_time': cpu_time,
            'gpu_time': gpu_time,
            'speedup': speedup,
            'throughput': n_traj / gpu_time if gpu_time > 0 else 0
        })
    
    # Create performance comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Extract data
    n_traj = [r['n_trajectories'] for r in results]
    cpu_times = [r['cpu_time'] for r in results]
    gpu_times = [r['gpu_time'] for r in results]
    speedups = [r['speedup'] for r in results]
    throughputs = [r['throughput'] for r in results]
    
    # Plot 1: Execution time comparison
    ax = axes[0, 0]
    ax.loglog(n_traj, cpu_times, 'o-', label='CPU', linewidth=2, markersize=8)
    ax.loglog(n_traj, gpu_times, 's-', label='GPU', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Trajectories')
    ax.set_ylabel('Execution Time (seconds)')
    ax.set_title('Execution Time Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Speedup factor
    ax = axes[0, 1]
    ax.semilogx(n_traj, speedups, 'o-', linewidth=2, markersize=8, color='green')
    ax.set_xlabel('Number of Trajectories')
    ax.set_ylabel('Speedup Factor')
    ax.set_title('GPU Speedup vs Trajectory Count')
    ax.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(np.log10(n_traj), speedups, 1)
    p = np.poly1d(z)
    ax.semilogx(n_traj, p(np.log10(n_traj)), "--", alpha=0.8, color='red', 
                label=f'Trend: {z[0]:.1f}*log10(n) + {z[1]:.1f}')
    ax.legend()
    
    # Plot 3: Throughput analysis
    ax = axes[1, 0]
    ax.semilogx(n_traj, throughputs, 'o-', linewidth=2, markersize=8, color='orange')
    ax.set_xlabel('Number of Trajectories')
    ax.set_ylabel('Trajectories per Second')
    ax.set_title('GPU Throughput')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Efficiency analysis
    ax = axes[1, 1]
    ideal_speedup = np.array(n_traj) / n_traj[0] * speedups[0]  # Linear scaling
    efficiency = np.array(speedups) / ideal_speedup
    ax.semilogx(n_traj, efficiency, 'o-', linewidth=2, markersize=8, color='purple')
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Ideal scaling')
    ax.set_xlabel('Number of Trajectories')
    ax.set_ylabel('Parallel Efficiency')
    ax.set_title('GPU Parallel Efficiency')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
    
    # Print summary statistics
    print("\n" + "=" * 50)
    print("PERFORMANCE SUMMARY")
    print("=" * 50)
    
    max_speedup = max(speedups)
    max_speedup_idx = speedups.index(max_speedup)
    
    print(f"Maximum speedup: {max_speedup:.1f}x (@ {n_traj[max_speedup_idx]:,} trajectories)")
    print(f"Average speedup: {np.mean(speedups):.1f}x")
    print(f"Peak throughput: {max(throughputs):.0f} trajectories/second")
    
    # Calculate efficiency metrics
    linear_scaling_efficiency = efficiency[-1] if len(efficiency) > 0 else 0
    print(f"Parallel efficiency at {n_traj[-1]:,} trajectories: {linear_scaling_efficiency:.2f}")
    
    # Estimate break-even point
    if len(speedups) > 1:
        breakeven_idx = next((i for i, s in enumerate(speedups) if s > 1.0), None)
        if breakeven_idx is not None:
            print(f"GPU becomes faster than CPU at: {n_traj[breakeven_idx]:,} trajectories")
        else:
            print("GPU is faster than CPU at all tested trajectory counts")
    
    print(f"\nVisualization saved as 'performance_comparison.png'")

if __name__ == "__main__":
    main()