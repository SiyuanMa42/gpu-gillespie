"""
CUDA kernels for GPU-accelerated Gillespie algorithm
Optimized for parallel execution of multiple trajectories
"""

import numpy as np
from numba import cuda
from numba.cuda import random
import math

try:
    from numba import cuda

    HAS_NUMBA_CUDA = True
except Exception:
    cuda = None
    HAS_NUMBA_CUDA = False


@cuda.jit
def gillespie_kernel(
    species_array,  # Species counts [n_trajectories, n_species]
    reactions_array,  # Reaction propensities [n_trajectories, n_reactions]
    stoichiometry,  # Stoichiometry matrix [n_reactions, n_species]
    rate_constants,  # Rate constants [n_reactions]
    time_points,  # Output time points [n_timepoints]
    results,  # Output results [n_trajectories, n_timepoints, n_species]
    max_time,  # Maximum simulation time
    n_trajectories,  # Number of parallel trajectories
    n_species,  # Number of species
    n_reactions,  # Number of reactions
    n_timepoints,  # Number of output time points
):
    """
    CUDA kernel for parallel Gillespie stochastic simulation
    Each thread handles one trajectory
    """
    # Get thread index (trajectory index)
    traj_idx = cuda.grid(1)

    if traj_idx >= n_trajectories:
        return

    # Initialize random state for this trajectory
    rng_states = cuda.local.array(1, dtype=random.uint64)
    rng_states[0] = random.init_xoroshiro128p_states(traj_idx, 0)

    # Initialize species counts for this trajectory
    species_local = cuda.local.array(10, dtype=np.int32)  # Max 10 species
    for i in range(n_species):
        species_local[i] = species_array[traj_idx, i]

    current_time = 0.0
    time_idx = 0

    # Main simulation loop
    while current_time < max_time and time_idx < n_timepoints:
        # Calculate reaction propensities
        total_propensity = 0.0
        for r in range(n_reactions):
            propensity = 1.0
            for s in range(n_species):
                if stoichiometry[r, s] < 0:  # Reactant
                    # Simple mass action kinetics
                    propensity *= species_local[s] ** abs(stoichiometry[r, s])
            propensity *= rate_constants[r]
            reactions_array[traj_idx, r] = propensity
            total_propensity += propensity

        if total_propensity <= 0.0:
            # No reactions possible, fill remaining time points
            while time_idx < n_timepoints and current_time < max_time:
                for s in range(n_species):
                    results[traj_idx, time_idx, s] = species_local[s]
                time_idx += 1
                current_time = max_time
            break

        # Generate random numbers
        u1 = random.xoroshiro128p_uniform_float32(rng_states, 0)
        u2 = random.xoroshiro128p_uniform_float32(rng_states, 0)

        # Time to next reaction
        tau = -math.log(u1) / total_propensity
        current_time += tau

        # Determine which reaction occurs
        threshold = u2 * total_propensity
        cumulative = 0.0
        reaction_idx = 0

        for r in range(n_reactions):
            cumulative += reactions_array[traj_idx, r]
            if cumulative >= threshold:
                reaction_idx = r
                break

        # Update species counts
        for s in range(n_species):
            species_local[s] += stoichiometry[reaction_idx, s]

        # Store results at appropriate time points
        while time_idx < n_timepoints and current_time >= time_points[time_idx]:
            for s in range(n_species):
                results[traj_idx, time_idx, s] = species_local[s]
            time_idx += 1


@cuda.jit
def tau_leaping_kernel(
    species_array,
    reactions_array,
    stoichiometry,
    rate_constants,
    time_points,
    results,
    max_time,
    n_trajectories,
    n_species,
    n_reactions,
    n_timepoints,
    epsilon=0.03,  # Leap size control parameter
):
    """
    CUDA kernel for tau-leaping approximation
    Faster for large systems with many reactions
    """
    traj_idx = cuda.grid(1)

    if traj_idx >= n_trajectories:
        return

    # Initialize random state
    rng_states = cuda.local.array(1, dtype=random.uint64)
    rng_states[0] = random.init_xoroshiro128p_states(traj_idx, 0)

    # Local species array
    species_local = cuda.local.array(10, dtype=np.int32)
    for i in range(n_species):
        species_local[i] = species_array[traj_idx, i]

    current_time = 0.0
    time_idx = 0

    while current_time < max_time and time_idx < n_timepoints:
        # Calculate propensities
        total_propensity = 0.0
        for r in range(n_reactions):
            propensity = 1.0
            for s in range(n_species):
                if stoichiometry[r, s] < 0:
                    propensity *= species_local[s] ** abs(stoichiometry[r, s])
            propensity *= rate_constants[r]
            reactions_array[traj_idx, r] = propensity
            total_propensity += propensity

        if total_propensity <= 0.0:
            break

        # Calculate leap time tau
        tau = epsilon / total_propensity

        # Limit tau to not exceed output time points
        if time_idx < n_timepoints - 1:
            tau = min(tau, time_points[time_idx + 1] - current_time)

        # Sample number of reactions using Poisson distribution
        for r in range(n_reactions):
            lambda_param = reactions_array[traj_idx, r] * tau
            # Approximate Poisson sampling
            if lambda_param > 0:
                k = 0
                p = 1.0
                while p >= math.exp(-lambda_param):
                    u = random.xoroshiro128p_uniform_float32(rng_states, 0)
                    p *= u
                    k += 1
                k -= 1

                # Update species based on reaction count
                for s in range(n_species):
                    species_local[s] += k * stoichiometry[r, s]

        current_time += tau

        # Store results
        while time_idx < n_timepoints and current_time >= time_points[time_idx]:
            for s in range(n_species):
                results[traj_idx, time_idx, s] = species_local[s]
            time_idx += 1


def launch_gillespie_simulation(
    initial_species,  # Initial species counts [n_species]
    stoichiometry,  # Stoichiometry matrix [n_reactions, n_species]
    rate_constants,  # Rate constants [n_reactions]
    time_points,  # Time points for output
    n_trajectories=1000,  # Number of parallel trajectories
    use_tau_leaping=False,  # Use tau-leaping approximation
):
    """
    Launch GPU-accelerated Gillespie simulation
    """
    if not HAS_NUMBA_CUDA:
        raise RuntimeError(
            "Numba CUDA bindings are not available in this environment. "
            "Set NUMBA_CUDA_USE_NVIDIA_BINDING=0 for ctypes fallback in CI "
            "or install the CUDA bindings (e.g. `pip install cuda-python` / "
            "`pip install numba-cuda[cuXY]`)."
        )
    n_species = len(initial_species)
    n_reactions = len(rate_constants)
    n_timepoints = len(time_points)
    max_time = time_points[-1]

    # Convert to numpy arrays
    initial_species = np.array(initial_species, dtype=np.int32)
    stoichiometry = np.array(stoichiometry, dtype=np.int32)
    rate_constants = np.array(rate_constants, dtype=np.float32)
    time_points = np.array(time_points, dtype=np.float32)

    # Allocate device arrays
    d_species = cuda.to_device(np.tile(initial_species, (n_trajectories, 1)).astype(np.int32))
    d_reactions = cuda.device_array((n_trajectories, n_reactions), dtype=np.float32)
    d_stoichiometry = cuda.to_device(stoichiometry)
    d_rate_constants = cuda.to_device(rate_constants)
    d_time_points = cuda.to_device(time_points)
    d_results = cuda.device_array((n_trajectories, n_timepoints, n_species), dtype=np.int32)

    # Configure kernel launch parameters
    threads_per_block = 128
    blocks_per_grid = (n_trajectories + threads_per_block - 1) // threads_per_block

    # Launch kernel
    if use_tau_leaping:
        tau_leaping_kernel[blocks_per_grid, threads_per_block](
            d_species,
            d_reactions,
            d_stoichiometry,
            d_rate_constants,
            d_time_points,
            d_results,
            max_time,
            n_trajectories,
            n_species,
            n_reactions,
            n_timepoints,
        )
    else:
        gillespie_kernel[blocks_per_grid, threads_per_block](
            d_species,
            d_reactions,
            d_stoichiometry,
            d_rate_constants,
            d_time_points,
            d_results,
            max_time,
            n_trajectories,
            n_species,
            n_reactions,
            n_timepoints,
        )

    # Copy results back to host
    results = d_results.copy_to_host()

    return results
