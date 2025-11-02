#!/usr/bin/env python3
"""
Command-line interface for GPU-Gillespie package
"""

import argparse
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

from .models.basic_models import (
    DimerizationModel, 
    EnzymeKineticsModel, 
    GeneExpressionModel,
    ToggleSwitchModel
)
from .utils.performance_metrics import BenchmarkSuite

def create_parser():
    """Create command-line argument parser"""
    parser = argparse.ArgumentParser(
        description='GPU-Gillespie: High-performance stochastic simulation with GPU acceleration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run dimerization model with 1000 trajectories
    gpu-gillespie dimerization --trajectories 1000 --time 100
    
    # Run enzyme kinetics with parameter sweep
    gpu-gillespie enzyme --sweep k_cat --values 0.1,1.0,10.0
    
    # Run performance benchmark
    gpu-gillespie benchmark --model dimerization --trajectories 100,1000,10000
        """
    )
    
    # Global arguments
    parser.add_argument(
        '--version', 
        action='version', 
        version='GPU-Gillespie 1.0.1'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Dimerization model command
    dimer_parser = subparsers.add_parser('dimerization', help='Run dimerization model')
    dimer_parser.add_argument('--kd', type=float, default=0.1, help='Dimerization rate')
    dimer_parser.add_argument('--ku', type=float, default=0.01, help='Dissociation rate')
    dimer_parser.add_argument('--a0', type=int, default=1000, help='Initial A count')
    dimer_parser.add_argument('--a20', type=int, default=0, help='Initial A2 count')
    dimer_parser.add_argument('-t', '--time', type=float, default=100, help='Simulation time')
    dimer_parser.add_argument('-n', '--trajectories', type=int, default=1000, help='Number of trajectories')
    dimer_parser.add_argument('--output', type=str, help='Output file path')
    
    # Enzyme kinetics command
    enzyme_parser = subparsers.add_parser('enzyme', help='Run enzyme kinetics model')
    enzyme_parser.add_argument('--kb', type=float, default=1.0, help='Binding rate')
    enzyme_parser.add_argument('--kd', type=float, default=0.1, help='Dissociation rate')
    enzyme_parser.add_argument('--kc', type=float, default=0.5, help='Catalytic rate')
    enzyme_parser.add_argument('--e0', type=int, default=100, help='Initial enzyme count')
    enzyme_parser.add_argument('--s0', type=int, default=1000, help='Initial substrate count')
    enzyme_parser.add_argument('-t', '--time', type=float, default=100, help='Simulation time')
    enzyme_parser.add_argument('-n', '--trajectories', type=int, default=1000, help='Number of trajectories')
    enzyme_parser.add_argument('--sweep', type=str, help='Parameter to sweep')
    enzyme_parser.add_argument('--values', type=str, help='Comma-separated values for sweep')
    enzyme_parser.add_argument('--output', type=str, help='Output file path')
    
    # Gene expression command
    gene_parser = subparsers.add_parser('gene', help='Run gene expression model')
    gene_parser.add_argument('--ktx', type=float, default=0.1, help='Transcription rate')
    gene_parser.add_argument('--ktl', type=float, default=0.2, help='Translation rate')
    gene_parser.add_argument('--kmd', type=float, default=0.05, help='mRNA degradation rate')
    gene_parser.add_argument('--kpd', type=float, default=0.01, help='Protein degradation rate')
    gene_parser.add_argument('-t', '--time', type=float, default=200, help='Simulation time')
    gene_parser.add_argument('-n', '--trajectories', type=int, default=1000, help='Number of trajectories')
    gene_parser.add_argument('--output', type=str, help='Output file path')
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', help='Run performance benchmark')
    benchmark_parser.add_argument('--model', type=str, choices=['dimerization', 'enzyme', 'gene'], 
                                  default='dimerization', help='Model to benchmark')
    benchmark_parser.add_argument('--trajectories', type=str, default='100,1000,10000', 
                                  help='Comma-separated trajectory counts')
    benchmark_parser.add_argument('--runs', type=int, default=3, help='Number of runs per test')
    benchmark_parser.add_argument('--output', type=str, help='Output file path')
    
    return parser

def run_dimerization_model(args):
    """Run dimerization model simulation"""
    print(f"Running dimerization model with {args.trajectories} trajectories...")
    
    model = DimerizationModel(
        k_dimerization=args.kd,
        k_dissociation=args.ku,
        initial_A=args.a0,
        initial_A2=args.a20
    )
    
    start_time = time.time()
    results = model.run_simulation(
        time_span=(0, args.time),
        n_trajectories=args.trajectories
    )
    execution_time = time.time() - start_time
    
    print(f"Simulation completed in {execution_time:.3f} seconds")
    print(f"GPU speedup: {results['performance_stats']['speedup_factor']:.1f}x")
    
    # Save results if output specified
    if args.output:
        output_data = {
            'model': 'dimerization',
            'parameters': {
                'k_dimerization': args.kd,
                'k_dissociation': args.ku,
                'initial_A': args.a0,
                'initial_A2': args.a20
            },
            'simulation_settings': {
                'time_span': args.time,
                'n_trajectories': args.trajectories
            },
            'performance_stats': results['performance_stats'],
            'equilibrium_constant': model.get_equilibrium_constant()
        }
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"Results saved to {args.output}")
    
    return results

def run_enzyme_model(args):
    """Run enzyme kinetics model simulation"""
    if args.sweep and args.values:
        return run_parameter_sweep(args)
    
    print(f"Running enzyme kinetics model with {args.trajectories} trajectories...")
    
    model = EnzymeKineticsModel(
        k_binding=args.kb,
        k_dissociation=args.kd,
        k_cat=args.kc,
        initial_E=args.e0,
        initial_S=args.s0
    )
    
    start_time = time.time()
    results = model.run_simulation(
        time_span=(0, args.time),
        n_trajectories=args.trajectories
    )
    execution_time = time.time() - start_time
    
    print(f"Simulation completed in {execution_time:.3f} seconds")
    print(f"GPU speedup: {results['performance_stats']['speedup_factor']:.1f}x")
    print(f"Michaelis constant (Km): {model.get_michaelis_constant():.3f}")
    print(f"Maximum velocity (Vmax): {model.get_max_velocity(args.e0):.1f}")
    
    # Save results if output specified
    if args.output:
        output_data = {
            'model': 'enzyme_kinetics',
            'parameters': {
                'k_binding': args.kb,
                'k_dissociation': args.kd,
                'k_cat': args.kc,
                'initial_E': args.e0,
                'initial_S': args.s0
            },
            'simulation_settings': {
                'time_span': args.time,
                'n_trajectories': args.trajectories
            },
            'performance_stats': results['performance_stats'],
            'michaelis_constant': model.get_michaelis_constant(),
            'max_velocity': model.get_max_velocity(args.e0)
        }
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"Results saved to {args.output}")
    
    return results

def run_parameter_sweep(args):
    """Run parameter sweep analysis"""
    from .core.parallel_simulator import ParallelSimulator
    
    values = [float(x) for x in args.values.split(',')]
    print(f"Running parameter sweep for {args.sweep} with values: {values}")
    
    model = EnzymeKineticsModel(
        k_binding=args.kb,
        k_dissociation=args.kd,
        k_cat=args.kc,
        initial_E=args.e0,
        initial_S=args.s0
    )
    
    parallel_sim = ParallelSimulator(model.simulator)
    
    start_time = time.time()
    results = parallel_sim.parameter_sweep(
        parameter_name=args.sweep,
        parameter_values=values,
        fixed_parameters={
            'k_binding': args.kb,
            'k_dissociation': args.kd
        },
        time_span=(0, args.time),
        n_trajectories_per_point=args.trajectories
    )
    execution_time = time.time() - start_time
    
    print(f"Parameter sweep completed in {execution_time:.3f} seconds")
    print(f"Tested {len(values)} parameter values")
    
    # Save results if output specified
    if args.output:
        df = parallel_sim.export_to_dataframe(results, analysis_type='parameter_sweep')
        df.to_csv(args.output, index=False)
        print(f"Sweep results saved to {args.output}")
    
    return results

def run_gene_model(args):
    """Run gene expression model simulation"""
    print(f"Running gene expression model with {args.trajectories} trajectories...")
    
    model = GeneExpressionModel(
        k_transcription=args.ktx,
        k_translation=args.ktl,
        k_mrna_degradation=args.kmd,
        k_protein_degradation=args.kpd
    )
    
    start_time = time.time()
    results = model.run_simulation(
        time_span=(0, args.time),
        n_trajectories=args.trajectories
    )
    execution_time = time.time() - start_time
    
    print(f"Simulation completed in {execution_time:.3f} seconds")
    print(f"GPU speedup: {results['performance_stats']['speedup_factor']:.1f}x")
    print(f"Theoretical mRNA steady state: {model.get_steady_state_mrna():.1f}")
    print(f"Theoretical protein steady state: {model.get_steady_state_protein():.1f}")
    
    # Save results if output specified
    if args.output:
        output_data = {
            'model': 'gene_expression',
            'parameters': {
                'k_transcription': args.ktx,
                'k_translation': args.ktl,
                'k_mrna_degradation': args.kmd,
                'k_protein_degradation': args.kpd
            },
            'simulation_settings': {
                'time_span': args.time,
                'n_trajectories': args.trajectories
            },
            'performance_stats': results['performance_stats'],
            'mrna_steady_state': model.get_steady_state_mrna(),
            'protein_steady_state': model.get_steady_state_protein()
        }
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"Results saved to {args.output}")
    
    return results

def run_benchmark(args):
    """Run performance benchmark"""
    print("Running performance benchmark...")
    
    # Parse trajectory counts
    trajectory_counts = [int(x) for x in args.trajectories.split(',')]
    
    # Create model
    if args.model == 'dimerization':
        model = DimerizationModel()
    elif args.model == 'enzyme':
        model = EnzymeKineticsModel()
    elif args.model == 'gene':
        model = GeneExpressionModel()
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    # Run benchmark
    benchmark = BenchmarkSuite()
    results = benchmark.run_benchmark(
        simulator=model.simulator,
        test_name=f"{args.model}_benchmark",
        n_trajectories_list=trajectory_counts,
        n_runs=args.runs
    )
    
    # Print results
    print("\nBenchmark Results:")
    print("=" * 50)
    for config in results['configurations']:
        print(f"Trajectories: {config['n_trajectories']:,}")
        print(f"  Avg Time: {config['avg_execution_time']:.3f}s")
        print(f"  Throughput: {config['trajectories_per_second']:.0f} traj/s")
        print(f"  Speedup: {config['avg_speedup']:.1f}x")
        print()
    
    print("Summary Statistics:")
    print(f"  Average Speedup: {results['summary_stats']['avg_speedup']:.1f}x")
    print(f"  Maximum Speedup: {results['summary_stats']['max_speedup']:.1f}x")
    print(f"  Peak Throughput: {results['summary_stats']['max_throughput']:.0f} traj/s")
    print(f"  Scalability Score: {results['summary_stats']['scalability_score']:.3f}")
    
    # Save results if output specified
    if args.output:
        benchmark.export_to_csv(args.output)
        print(f"\nBenchmark results saved to {args.output}")
    
    return results

def main():
    """Main CLI entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'dimerization':
            run_dimerization_model(args)
        elif args.command == 'enzyme':
            run_enzyme_model(args)
        elif args.command == 'gene':
            run_gene_model(args)
        elif args.command == 'benchmark':
            run_benchmark(args)
        else:
            print(f"Unknown command: {args.command}")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()