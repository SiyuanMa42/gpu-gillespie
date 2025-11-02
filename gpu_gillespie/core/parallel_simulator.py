"""
Advanced parallel simulation capabilities
Including parameter sweeps and ensemble simulations
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Union, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from .gillespie_gpu import GPUGillespieSimulator
from ..utils.performance_metrics import PerformanceMonitor

class ParallelSimulator:
    """
    Advanced parallel simulation manager for parameter sweeps and ensemble studies
    """
    
    def __init__(self, base_simulator: GPUGillespieSimulator):
        """
        Initialize parallel simulator
        
        Parameters:
        -----------
        base_simulator : GPUGillespieSimulator
            Base simulator instance to use for parallel runs
        """
        self.base_simulator = base_simulator
        self.performance_monitor = PerformanceMonitor()
    
    def parameter_sweep(self,
                       parameter_name: str,
                       parameter_values: List[float],
                       fixed_parameters: Dict[str, float] = None,
                       time_span: Tuple[float, float] = (0, 100),
                       n_timepoints: int = 101,
                       n_trajectories_per_point: int = 1000,
                       n_workers: int = 4) -> Dict:
        """
        Perform parameter sweep analysis
        
        Parameters:
        -----------
        parameter_name : str
            Name of parameter to sweep (must be a rate constant)
        parameter_values : List[float]
            Values of parameter to test
        fixed_parameters : Dict[str, float], optional
            Other parameters to keep fixed
        time_span : Tuple[float, float]
            Time span for simulation
        n_timepoints : int
            Number of time points
        n_trajectories_per_point : int
            Number of trajectories per parameter value
        n_workers : int
            Number of parallel workers
            
        Returns:
        --------
        Dict containing sweep results
        """
        if fixed_parameters is None:
            fixed_parameters = {}
        
        # Validate parameter name
        if parameter_name not in self.base_simulator.reaction_names:
            raise ValueError(f"Parameter {parameter_name} not found in reaction names")
        
        param_idx = self.base_simulator.reaction_names.index(parameter_name)
        
        self.performance_monitor.start_monitoring()
        
        results = {
            'parameter_name': parameter_name,
            'parameter_values': parameter_values,
            'sweep_results': [],
            'performance_stats': {}
        }
        
        # Prepare tasks for parallel execution
        tasks = []
        for param_value in parameter_values:
            task = {
                'parameter_value': param_value,
                'param_idx': param_idx,
                'fixed_parameters': fixed_parameters,
                'time_span': time_span,
                'n_timepoints': n_timepoints,
                'n_trajectories': n_trajectories_per_point
            }
            tasks.append(task)
        
        # Execute parameter sweep in parallel
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            future_to_task = {
                executor.submit(self._run_single_parameter_set, task): task 
                for task in tasks
            }
            
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    task_result = future.result()
                    results['sweep_results'].append(task_result)
                except Exception as e:
                    print(f"Error processing parameter value {task['parameter_value']}: {e}")
        
        # Sort results by parameter value
        results['sweep_results'].sort(key=lambda x: x['parameter_value'])
        
        # Add performance statistics
        results['performance_stats'] = self.performance_monitor.stop_monitoring()
        
        return results
    
    def _run_single_parameter_set(self, task: Dict) -> Dict:
        """
        Run simulation for a single parameter set
        """
        parameter_value = task['parameter_value']
        param_idx = task['param_idx']
        fixed_parameters = task['fixed_parameters']
        
        # Create modified rate constants
        modified_rates = self.base_simulator.rate_constants.copy()
        modified_rates[param_idx] = parameter_value
        
        # Apply fixed parameters
        for param_name, param_val in fixed_parameters.items():
            if param_name in self.base_simulator.reaction_names:
                idx = self.base_simulator.reaction_names.index(param_name)
                modified_rates[idx] = param_val
        
        # Create temporary simulator with modified parameters
        temp_simulator = GPUGillespieSimulator(
            species_names=self.base_simulator.species_names,
            reaction_names=self.base_simulator.reaction_names,
            stoichiometry=self.base_simulator.stoichiometry,
            rate_constants=modified_rates,
            initial_conditions=dict(zip(
                self.base_simulator.species_names, 
                self.base_simulator.initial_conditions
            ))
        )
        
        # Run simulation
        sim_results = temp_simulator.run_simulation(
            time_span=task['time_span'],
            n_timepoints=task['n_timepoints'],
            n_trajectories=task['n_trajectories']
        )
        
        # Analyze results
        analysis = temp_simulator.analyze_results(sim_results)
        
        return {
            'parameter_value': parameter_value,
            'simulation_results': sim_results,
            'analysis_results': analysis
        }
    
    def ensemble_simulation(self,
                           parameter_sets: List[Dict[str, float]],
                           time_span: Tuple[float, float] = (0, 100),
                           n_timepoints: int = 101,
                           n_trajectories_per_set: int = 1000,
                           n_workers: int = 4) -> Dict:
        """
        Run ensemble simulation with multiple parameter sets
        
        Parameters:
        -----------
        parameter_sets : List[Dict[str, float]]
            List of parameter dictionaries
        time_span : Tuple[float, float]
            Time span for simulation
        n_timepoints : int
            Number of time points
        n_trajectories_per_set : int
            Number of trajectories per parameter set
        n_workers : int
            Number of parallel workers
            
        Returns:
        --------
        Dict containing ensemble results
        """
        self.performance_monitor.start_monitoring()
        
        results = {
            'parameter_sets': parameter_sets,
            'ensemble_results': [],
            'performance_stats': {}
        }
        
        # Prepare tasks
        tasks = []
        for i, param_set in enumerate(parameter_sets):
            task = {
                'set_id': i,
                'parameters': param_set,
                'time_span': time_span,
                'n_timepoints': n_timepoints,
                'n_trajectories': n_trajectories_per_set
            }
            tasks.append(task)
        
        # Execute ensemble simulation
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            future_to_task = {
                executor.submit(self._run_ensemble_member, task): task 
                for task in tasks
            }
            
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    task_result = future.result()
                    results['ensemble_results'].append(task_result)
                except Exception as e:
                    print(f"Error processing ensemble member {task['set_id']}: {e}")
        
        # Sort results by set_id
        results['ensemble_results'].sort(key=lambda x: x['set_id'])
        results['performance_stats'] = self.performance_monitor.stop_monitoring()
        
        return results
    
    def _run_ensemble_member(self, task: Dict) -> Dict:
        """
        Run single ensemble member simulation
        """
        parameters = task['parameters']
        
        # Create modified rate constants
        modified_rates = self.base_simulator.rate_constants.copy()
        
        for param_name, param_val in parameters.items():
            if param_name in self.base_simulator.reaction_names:
                idx = self.base_simulator.reaction_names.index(param_name)
                modified_rates[idx] = param_val
        
        # Create temporary simulator
        temp_simulator = GPUGillespieSimulator(
            species_names=self.base_simulator.species_names,
            reaction_names=self.base_simulator.reaction_names,
            stoichiometry=self.base_simulator.stoichiometry,
            rate_constants=modified_rates,
            initial_conditions=dict(zip(
                self.base_simulator.species_names, 
                self.base_simulator.initial_conditions
            ))
        )
        
        # Run simulation
        sim_results = temp_simulator.run_simulation(
            time_span=task['time_span'],
            n_timepoints=task['n_timepoints'],
            n_trajectories=task['n_trajectories']
        )
        
        # Analyze results
        analysis = temp_simulator.analyze_results(sim_results)
        
        return {
            'set_id': task['set_id'],
            'parameters': parameters,
            'simulation_results': sim_results,
            'analysis_results': analysis
        }
    
    def sensitivity_analysis(self,
                           parameter_names: List[str],
                           base_values: Dict[str, float],
                           perturbation_factors: List[float] = [0.1, 0.5, 2.0, 10.0],
                           time_span: Tuple[float, float] = (0, 100),
                           n_timepoints: int = 101,
                           n_trajectories: int = 1000,
                           n_workers: int = 4) -> Dict:
        """
        Perform sensitivity analysis on model parameters
        
        Parameters:
        -----------
        parameter_names : List[str]
            Names of parameters to analyze
        base_values : Dict[str, float]
            Base values for parameters
        perturbation_factors : List[float]
            Factors to multiply base values by
        time_span : Tuple[float, float]
            Time span for simulation
        n_timepoints : int
            Number of time points
        n_trajectories : int
            Number of trajectories per simulation
        n_workers : int
            Number of parallel workers
            
        Returns:
        --------
        Dict containing sensitivity analysis results
        """
        self.performance_monitor.start_monitoring()
        
        results = {
            'parameter_names': parameter_names,
            'base_values': base_values,
            'perturbation_factors': perturbation_factors,
            'sensitivity_results': {},
            'performance_stats': {}
        }
        
        # Run base case simulation
        base_simulator = self._create_simulator_with_parameters(base_values)
        base_results = base_simulator.run_simulation(
            time_span=time_span,
            n_timepoints=n_timepoints,
            n_trajectories=n_trajectories
        )
        base_analysis = base_simulator.analyze_results(base_results)
        
        results['base_results'] = {
            'simulation': base_results,
            'analysis': base_analysis
        }
        
        # Perform sensitivity analysis for each parameter
        for param_name in parameter_names:
            param_results = []
            
            tasks = []
            for factor in perturbation_factors:
                perturbed_params = base_values.copy()
                perturbed_params[param_name] = base_values[param_name] * factor
                
                task = {
                    'parameter_name': param_name,
                    'factor': factor,
                    'parameters': perturbed_params,
                    'time_span': time_span,
                    'n_timepoints': n_timepoints,
                    'n_trajectories': n_trajectories
                }
                tasks.append(task)
            
            # Execute sensitivity tasks
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                future_to_task = {
                    executor.submit(self._run_sensitivity_case, task): task 
                    for task in tasks
                }
                
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    try:
                        task_result = future.result()
                        param_results.append(task_result)
                    except Exception as e:
                        print(f"Error in sensitivity analysis for {task['parameter_name']} "
                              f"with factor {task['factor']}: {e}")
            
            # Sort results by factor
            param_results.sort(key=lambda x: x['factor'])
            results['sensitivity_results'][param_name] = param_results
        
        results['performance_stats'] = self.performance_monitor.stop_monitoring()
        
        return results
    
    def _run_sensitivity_case(self, task: Dict) -> Dict:
        """
        Run single sensitivity analysis case
        """
        simulator = self._create_simulator_with_parameters(task['parameters'])
        
        sim_results = simulator.run_simulation(
            time_span=task['time_span'],
            n_timepoints=task['n_timepoints'],
            n_trajectories=task['n_trajectories']
        )
        
        analysis = simulator.analyze_results(sim_results)
        
        return {
            'factor': task['factor'],
            'simulation_results': sim_results,
            'analysis_results': analysis
        }
    
    def _create_simulator_with_parameters(self, parameters: Dict[str, float]) -> GPUGillespieSimulator:
        """
        Create simulator with given parameters
        """
        modified_rates = self.base_simulator.rate_constants.copy()
        
        for param_name, param_val in parameters.items():
            if param_name in self.base_simulator.reaction_names:
                idx = self.base_simulator.reaction_names.index(param_name)
                modified_rates[idx] = param_val
        
        return GPUGillespieSimulator(
            species_names=self.base_simulator.species_names,
            reaction_names=self.base_simulator.reaction_names,
            stoichiometry=self.base_simulator.stoichiometry,
            rate_constants=modified_rates,
            initial_conditions=dict(zip(
                self.base_simulator.species_names, 
                self.base_simulator.initial_conditions
            ))
        )
    
    def export_to_dataframe(self, results: Dict, analysis_type: str = 'parameter_sweep') -> pd.DataFrame:
        """
        Export results to pandas DataFrame for further analysis
        
        Parameters:
        -----------
        results : Dict
            Results from parameter_sweep, ensemble_simulation, or sensitivity_analysis
        analysis_type : str
            Type of analysis ('parameter_sweep', 'ensemble', 'sensitivity')
            
        Returns:
        --------
        pd.DataFrame containing results
        """
        if analysis_type == 'parameter_sweep':
            return self._export_parameter_sweep_to_df(results)
        elif analysis_type == 'ensemble':
            return self._export_ensemble_to_df(results)
        elif analysis_type == 'sensitivity':
            return self._export_sensitivity_to_df(results)
        else:
            raise ValueError(f"Unknown analysis type: {analysis_type}")
    
    def _export_parameter_sweep_to_df(self, results: Dict) -> pd.DataFrame:
        """Export parameter sweep results to DataFrame"""
        rows = []
        
        for sweep_result in results['sweep_results']:
            param_value = sweep_result['parameter_value']
            analysis = sweep_result['analysis_results']
            
            for species in results['parameter_name']:
                if species in analysis['final_distributions']:
                    dist = analysis['final_distributions'][species]
                    rows.append({
                        'parameter_name': results['parameter_name'],
                        'parameter_value': param_value,
                        'species': species,
                        'final_mean': dist['mean'],
                        'final_std': dist['std'],
                        'final_min': dist['min'],
                        'final_max': dist['max'],
                        'extinct_fraction': dist['extinct_fraction']
                    })
        
        return pd.DataFrame(rows)
    
    def _export_ensemble_to_df(self, results: Dict) -> pd.DataFrame:
        """Export ensemble simulation results to DataFrame"""
        rows = []
        
        for ensemble_result in results['ensemble_results']:
            set_id = ensemble_result['set_id']
            parameters = ensemble_result['parameters']
            analysis = ensemble_result['analysis_results']
            
            for species in self.base_simulator.species_names:
                if species in analysis['final_distributions']:
                    dist = analysis['final_distributions'][species]
                    row = {
                        'set_id': set_id,
                        'species': species,
                        'final_mean': dist['mean'],
                        'final_std': dist['std'],
                        'extinct_fraction': dist['extinct_fraction']
                    }
                    row.update(parameters)
                    rows.append(row)
        
        return pd.DataFrame(rows)
    
    def _export_sensitivity_to_df(self, results: Dict) -> pd.DataFrame:
        """Export sensitivity analysis results to DataFrame"""
        rows = []
        
        for param_name, param_results in results['sensitivity_results'].items():
            for result in param_results:
                factor = result['factor']
                analysis = result['analysis_results']
                
                for species in self.base_simulator.species_names:
                    if species in analysis['final_distributions']:
                        dist = analysis['final_distributions'][species]
                        rows.append({
                            'parameter': param_name,
                            'base_value': results['base_values'][param_name],
                            'perturbation_factor': factor,
                            'perturbed_value': results['base_values'][param_name] * factor,
                            'species': species,
                            'final_mean': dist['mean'],
                            'final_std': dist['std'],
                            'extinct_fraction': dist['extinct_fraction']
                        })
        
        return pd.DataFrame(rows)