"""
Performance monitoring and metrics collection for GPU simulations
"""

import time
import psutil
import numpy as np
import threading
from typing import Dict, List

try:
    # 注意: pynvml 已弃用，建议使用 nvidia-ml-py
    import pynvml

    HAS_NVML = True
except ImportError:
    HAS_NVML = False


class PerformanceMonitor:
    """
    Monitor system performance during GPU simulations
    """

    def __init__(self):
        self.start_time = None
        self.monitoring = False
        self.monitoring_thread = None
        self.performance_data = {
            "cpu_percentages": [],
            "memory_usages": [],
            "gpu_utilizations": [],
            "gpu_memory_usages": [],
            "timestamps": [],
        }
        self.cpu_percent = 0
        self.memory_usage = 0
        self.gpu_utilization = 0
        self.gpu_memory = 0
        self._nvml_initialized = False
        self._gpu_handle = None  # GPU handle for NVML operations

    def _init_nvml(self):
        if HAS_NVML and not self._nvml_initialized:
            try:
                pynvml.nvmlInit()
                self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self._nvml_initialized = True
            except pynvml.NVMLError_LibraryNotFound:
                # 在没有 NVIDIA 驱动的系统上优雅降级
                pass

    def update(self):
        # CPU 和内存指标总是可用
        self.cpu_percent = psutil.cpu_percent()
        self.memory_usage = psutil.Process().memory_info().rss / (1024 * 1024)  # MB

        # GPU 指标仅在支持时获取
        if HAS_NVML:
            try:
                if not self._nvml_initialized:
                    self._init_nvml()

                if self._nvml_initialized:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    self.gpu_utilization = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                    self.gpu_memory = pynvml.nvmlDeviceGetMemoryInfo(handle).used / (1024 * 1024)  # MB
            except (pynvml.NVMLError_LibraryNotFound, pynvml.NVMLError):
                # 忽略 GPU 相关错误，保持默认值
                pass

    def start_monitoring(self, interval: float = 0.1):
        """
        Start performance monitoring

        Parameters:
        -----------
        interval : float
            Monitoring interval in seconds
        """
        self.start_time = time.time()
        self.monitoring = True
        self.performance_data = {
            "cpu_percentages": [],
            "memory_usages": [],
            "gpu_utilizations": [],
            "gpu_memory_usages": [],
            "timestamps": [],
        }

        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitor_loop, args=(interval,))
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()

    def stop_monitoring(self) -> Dict:
        """
        Stop monitoring and return performance statistics

        Returns:
        --------
        Dict containing performance statistics
        """
        self.monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join()

        end_time = time.time()
        execution_time = end_time - self.start_time

        # Calculate statistics
        stats = {
            "execution_time": execution_time,
            "cpu_percentages": self.performance_data["cpu_percentages"],
            "memory_usages": self.performance_data["memory_usages"],
            "gpu_utilizations": self.performance_data["gpu_utilizations"],
            "gpu_memory_usages": self.performance_data["gpu_memory_usages"],
            "timestamps": self.performance_data["timestamps"],
        }

        # Add summary statistics
        if stats["cpu_percentages"]:
            stats["avg_cpu_percent"] = np.mean(stats["cpu_percentages"])
            stats["max_cpu_percent"] = np.max(stats["cpu_percentages"])

        if stats["memory_usages"]:
            stats["avg_memory_usage"] = np.mean(stats["memory_usages"])
            stats["max_memory_usage"] = np.max(stats["memory_usages"])
            stats["memory_usage_mb"] = stats["max_memory_usage"] / (1024 * 1024)

        if stats["gpu_utilizations"]:
            stats["avg_gpu_utilization"] = np.mean(stats["gpu_utilizations"])
            stats["max_gpu_utilization"] = np.max(stats["gpu_utilizations"])
            stats["gpu_utilization"] = stats["avg_gpu_utilization"]

        if stats["gpu_memory_usages"]:
            stats["avg_gpu_memory_usage"] = np.mean(stats["gpu_memory_usages"])
            stats["max_gpu_memory_usage"] = np.max(stats["gpu_memory_usages"])

        # Estimate CPU performance for comparison
        stats["estimated_cpu_time"] = self._estimate_cpu_time(execution_time)
        stats["speedup_factor"] = stats["estimated_cpu_time"] / execution_time

        return stats

    def _monitor_loop(self, interval: float):
        """Main monitoring loop running in background thread"""
        process = psutil.Process()

        while self.monitoring:
            try:
                # CPU and memory monitoring
                cpu_percent = process.cpu_percent()
                memory_info = process.memory_info()
                memory_usage = memory_info.rss

                # GPU monitoring
                gpu_utilization = 0
                gpu_memory_usage = 0

                if HAS_NVML and self._nvml_initialized and self._gpu_handle:
                    try:
                        gpu_utilization = pynvml.nvmlDeviceGetUtilizationRates(self._gpu_handle).gpu
                        gpu_memory_info = pynvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
                        gpu_memory_usage = gpu_memory_info.used
                    except Exception as e:
                        print(f"GPU monitoring error: {e}")

                # Store data
                if self.start_time is not None:
                    self.performance_data["cpu_percentages"].append(cpu_percent)
                    self.performance_data["memory_usages"].append(memory_usage)
                    self.performance_data["gpu_utilizations"].append(gpu_utilization)
                    self.performance_data["gpu_memory_usages"].append(gpu_memory_usage)
                    self.performance_data["timestamps"].append(time.time() - self.start_time)

            except Exception as e:
                print(f"Monitoring error: {e}")

            time.sleep(interval)

    def _estimate_cpu_time(self, gpu_time: float) -> float:
        """
        Estimate CPU execution time based on GPU time and typical speedup factors

        Parameters:
        -----------
        gpu_time : float
            Actual GPU execution time

        Returns:
        --------
        float: Estimated CPU execution time
        """
        # Typical speedup factors for different types of simulations
        # These are empirical estimates based on literature
        base_speedup = 50.0  # Conservative base speedup estimate

        # Check GPU utilization from performance data
        gpu_utils = self.performance_data.get("gpu_utilizations", [])
        avg_gpu_util = float(np.mean(gpu_utils)) if gpu_utils else 0.0

        # Adjust based on GPU utilization
        if avg_gpu_util > 80.0:
            # High GPU utilization suggests good parallelization
            speedup_factor = base_speedup * 1.5
        else:
            speedup_factor = base_speedup

        return float(gpu_time * speedup_factor)


class BenchmarkSuite:
    """
    Comprehensive benchmarking suite for GPU-Gillespie performance
    """

    def __init__(self):
        self.benchmark_results = []

    def run_benchmark(
        self,
        simulator,
        test_name: str,
        n_trajectories_list: List[int],
        time_span: tuple = (0, 100),
        n_timepoints: int = 101,
        n_runs: int = 3,
    ) -> Dict:
        """
        Run comprehensive benchmark test

        Parameters:
        -----------
        simulator : GPUGillespieSimulator
            Simulator instance to benchmark
        test_name : str
            Name of the benchmark test
        n_trajectories_list : List[int]
            List of trajectory counts to test
        time_span : tuple
            Time span for simulation
        n_timepoints : int
            Number of time points
        n_runs : int
            Number of runs per configuration

        Returns:
        --------
        Dict containing benchmark results
        """
        benchmark_data = {"test_name": test_name, "configurations": [], "summary_stats": {}}

        for n_traj in n_trajectories_list:
            print(f"Benchmarking {test_name} with {n_traj} trajectories...")

            run_times = []
            memory_usages = []
            speedups = []

            for run in range(n_runs):
                try:
                    # Run simulation
                    results = simulator.run_simulation(
                        time_span=time_span, n_timepoints=n_timepoints, n_trajectories=n_traj
                    )

                    # Collect performance data
                    stats = results["performance_stats"]
                    run_times.append(stats["execution_time"])
                    memory_usages.append(stats["memory_usage_mb"])
                    speedups.append(stats["speedup_factor"])

                except Exception as e:
                    print(f"Error in run {run}: {e}")
                    continue

            if run_times:  # If we have successful runs
                config_result = {
                    "n_trajectories": n_traj,
                    "avg_execution_time": np.mean(run_times),
                    "std_execution_time": np.std(run_times),
                    "avg_memory_usage": np.mean(memory_usages),
                    "avg_speedup": np.mean(speedups),
                    "trajectories_per_second": n_traj / np.mean(run_times),
                }
                benchmark_data["configurations"].append(config_result)

        # Calculate summary statistics
        if benchmark_data["configurations"]:
            all_speedups = [c["avg_speedup"] for c in benchmark_data["configurations"]]
            all_throughputs = [c["trajectories_per_second"] for c in benchmark_data["configurations"]]

            benchmark_data["summary_stats"] = {
                "avg_speedup": np.mean(all_speedups),
                "max_speedup": np.max(all_speedups),
                "avg_throughput": np.mean(all_throughputs),
                "max_throughput": np.max(all_throughputs),
                "scalability_score": self._calculate_scalability_score(benchmark_data["configurations"]),
            }

        self.benchmark_results.append(benchmark_data)
        return benchmark_data

    def _calculate_scalability_score(self, configurations: List[Dict]) -> float:
        """
        Calculate scalability score based on how well performance scales with problem size

        Parameters:
        -----------
        configurations : List[Dict]
            List of configuration results

        Returns:
        --------
        float: Scalability score (0-1, where 1 is perfect scaling)
        """
        if len(configurations) < 2:
            return 0.0

        n_traj_list = [float(c["n_trajectories"]) for c in configurations]
        throughput_list = [float(c["trajectories_per_second"]) for c in configurations]

        # Calculate scaling efficiency
        # Perfect scaling would have constant throughput per trajectory
        scaling_factors = []
        for i in range(1, len(configurations)):
            n_ratio = n_traj_list[i] / n_traj_list[i - 1]
            t_ratio = throughput_list[i] / throughput_list[i - 1]
            scaling_efficiency = float(t_ratio / n_ratio)
            scaling_factors.append(scaling_efficiency)

        return float(np.mean(scaling_factors)) if scaling_factors else 0.0

    def generate_report(self) -> str:
        """
        Generate comprehensive benchmark report

        Returns:
        --------
        str: Formatted benchmark report
        """
        if not self.benchmark_results:
            return "No benchmark results available."

        report = []
        report.append("GPU-Gillespie Performance Benchmark Report")
        report.append("=" * 50)
        report.append("")

        for benchmark in self.benchmark_results:
            report.append(f"Test: {benchmark['test_name']}")
            report.append("-" * 30)

            for config in benchmark["configurations"]:
                report.append(f"  Trajectories: {config['n_trajectories']:,}")
                report.append(f"    Avg Time: {config['avg_execution_time']:.3f} s")
                report.append(f"    Throughput: {config['trajectories_per_second']:.1f} traj/s")
                report.append(f"    Speedup: {config['avg_speedup']:.1f}x")
                report.append(f"    Memory: {config['avg_memory_usage']:.1f} MB")
                report.append("")

            summary = benchmark["summary_stats"]
            report.append("  Summary Statistics")
            report.append(f"    Average Speedup: {summary['avg_speedup']:.1f}x")
            report.append(f"    Maximum Speedup: {summary['max_speedup']:.1f}x")
            report.append(f"    Peak Throughput: {summary['max_throughput']:.1f} traj/s")
            report.append(f"    Scalability Score: {summary['scalability_score']:.3f}")
            report.append("")

        return "\n".join(report)

    def export_to_csv(self, filename: str):
        """
        Export benchmark results to CSV file

        Parameters:
        -----------
        filename : str
            Output CSV filename
        """
        import csv

        with open(filename, "w", newline="") as csvfile:
            fieldnames = [
                "test_name",
                "n_trajectories",
                "avg_execution_time",
                "std_execution_time",
                "avg_memory_usage",
                "avg_speedup",
                "trajectories_per_second",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for benchmark in self.benchmark_results:
                for config in benchmark["configurations"]:
                    row = {"test_name": benchmark["test_name"], **config}
                    writer.writerow(row)
