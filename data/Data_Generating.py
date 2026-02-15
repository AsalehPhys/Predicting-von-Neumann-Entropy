"""
Rydberg Atom Quantum Simulation Data Generator

Generates quantum many-body simulation data for 2D rectangular arrays of Rydberg atoms
by exactly diagonalizing the Rydberg blockade Hamiltonian across random parameter configurations.


Computational Approach:
    1. Sample random parameter configurations (Ny, Δ/Ω, R_b/a)
    2. For each configuration:
        a. Construct sparse Hamiltonian matrix using NetKet
        b. Compute ground state via sparse eigensolver 
        c. Calculate von Neumann entanglement entropy for random bipartition
        d. Compute classical mutual information for comparison in later evaluating of the model
    3. Store full quantum state (probabilities + indices) for downstream ML tasks
    4. Parallelize across CPU cores with memory-aware batching

Output Format:
    Parquet file containing one row per parameter configuration with fields:
        - Ny, Delta_over_Omega, Rb_over_a: Hamiltonian parameters
        - Energy: Ground state energy (in units of Ω)
        - All_Indices, All_Probabilities: Full quantum state representation
        - Von_Neumann_Entropy: Entanglement entropy of random bipartition
        - Classical_MI: Classical mutual information of same bipartition
        - N_A, Subsystem_Mask: Bipartition specification

"""

import logging
import os
import random
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import netket as nk
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from netket.operator.spin import sigmam, sigmap, sigmaz
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

from data.config_generation import (
    BATCH_SIZES,
    DELTA_OVER_OMEGA_RANGE,
    LOG_FILENAME,
    NUM_POINTS,
    NX,
    NY_RANGE,
    OUTPUT_FILENAME,
    R_CUT,
    RANDOM_SEED,
    RB_OVER_A_RANGE,
)

# Force JAX to use CPU backend (required for NetKet sparse operations)
# GPU backend would cause memory issues with dense state representations
os.environ["JAX_PLATFORM_NAME"] = "cpu"


def setup_logging() -> logging.Logger:
    """
    Configures dual-channel logging (file + console) with timestamped formatting.
    
    Creates a rotating log file that captures full execution history while streaming
    INFO-level messages to console for real-time monitoring.
    
    Returns:
        logging.Logger: Configured logger instance for module-level usage.
    
    Log Levels:
        - DEBUG: Detailed computational steps (not enabled by default)
        - INFO: Progress updates, batch completions, timing statistics
        - WARNING: Non-critical issues (e.g., empty batches)
        - ERROR: Critical failures requiring intervention
        - EXCEPTION: Full tracebacks for worker process failures
    
    Note:
        Log file is overwritten on each run (mode='w') to maintain clean execution history.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Avoid duplicate handlers when module is reimported
    if logger.handlers:
        return logger
    
    # File handler with full history
    file_handler = logging.FileHandler(LOG_FILENAME, mode='w')
    file_handler.setLevel(logging.INFO)
    
    # Console handler for real-time monitoring
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # Unified formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def reshape_for_subsystem(
    psi: np.ndarray,
    A_indices: List[int],
    N: int
) -> np.ndarray:
    """
    Reshapes full wavefunction into bipartite tensor for subsystem entropy calculation.
    
    Transforms the wavefunction from a flat array of length 2^N into a matrix of shape
    (2^N_A, 2^N_B) where rows correspond to subsystem A states and columns to subsystem B states.
    
    Args:
        psi (np.ndarray): Full wavefunction array of shape (2^N,) with complex amplitudes.
        A_indices (List[int]): List of site indices belonging to subsystem A (0-based).
        N (int): Total number of lattice sites in the system.
    
    Returns:
        np.ndarray: Reshaped wavefunction array of shape (2^N_A, 2^N_B) where:
            - N_A = len(A_indices)
            - N_B = N - N_A
    
    Algorithm:
        For each computational basis state |i⟩ (represented as integer i):
            1. Decompose i into binary representation across N sites
            2. Extract bits corresponding to subsystem A → integer index i_A
            3. Extract bits corresponding to subsystem B → integer index i_B
            4. Assign psi[i] to psi_reshaped[i_A, i_B]
    
    """
    A_indices = sorted(A_indices)
    B_indices = [i for i in range(N) if i not in A_indices]
    N_A = len(A_indices)
    N_B = N - N_A

    psi_reshaped = np.zeros((2**N_A, 2**N_B), dtype=psi.dtype)

    # Precompute bit position mappings for efficiency
    A_pos_map = {spin: pos for pos, spin in enumerate(A_indices)}
    B_pos_map = {spin: pos for pos, spin in enumerate(B_indices)}

    # Iterate over all computational basis states
    for i in range(2**N):
        i_bin = i
        i_A = 0
        i_B = 0
        for spin in range(N):
            bit = i_bin & 1
            i_bin >>= 1
            if spin in A_pos_map:
                i_A |= (bit << A_pos_map[spin])
            else:
                i_B |= (bit << B_pos_map[spin])

        psi_reshaped[i_A, i_B] = psi[i]

    return psi_reshaped


def calculate_classical_MI(
    psi_gs: np.ndarray,
    A_indices: List[int],
    N: int
) -> float:
    """
    Computes classical mutual information between subsystems A and B.
    
    Args:
        psi_gs (np.ndarray): Ground state wavefunction of shape (2^N,).
        A_indices (List[int]): Site indices belonging to subsystem A.
        N (int): Total number of lattice sites.
    
    Returns:
        float: Classical mutual information in natural units (nats).
    
    """
    B_indices = [i for i in range(N) if i not in A_indices]
    
    # Joint probability distribution from wavefunction
    p_joint = np.abs(psi_gs) ** 2
    
    # Initialize marginal distributions
    p_A = np.zeros(2**len(A_indices))
    p_B = np.zeros(2**len(B_indices))
    
    # Compute marginals by summing over complementary subsystem
    for state_idx in range(2**N):
        state_bin = format(state_idx, f'0{N}b')
        A_state = int(''.join(state_bin[i] for i in A_indices), 2)
        B_state = int(''.join(state_bin[i] for i in B_indices), 2)
        p_A[A_state] += p_joint[state_idx]
        p_B[B_state] += p_joint[state_idx]
    
    # Compute mutual information
    MI = 0.0
    for state_idx in range(2**N):
        if p_joint[state_idx] > 1e-12:  # Skip negligible probabilities
            state_bin = format(state_idx, f'0{N}b')
            A_state = int(''.join(state_bin[i] for i in A_indices), 2)
            B_state = int(''.join(state_bin[i] for i in B_indices), 2)
            # Add epsilon to denominator for numerical stability
            MI += p_joint[state_idx] * np.log(
                p_joint[state_idx] / (p_A[A_state] * p_B[B_state] + 1e-12)
            )
    
    return float(MI)


def calculate_properties_cpu_batch(
    params_batch: List[Tuple[int, float, float]]
) -> List[Dict[str, Any]]:
    """
    Computes quantum properties for a batch of Hamiltonian parameter configurations.
    
    For each parameter set (Ny, Δ/Ω, R_b/a), constructs the Rydberg Hamiltonian,
    computes its ground state via exact diagonalization, and extracts entanglement
    properties for a random bipartition.
    
    Args:
        params_batch (List[Tuple[int, float, float]]): List of parameter tuples:
            - Ny (int): Number of columns in 2×Ny lattice
            - Delta_over_Omega (float): Dimensionless detuning parameter
            - Rb_over_a (float): Blockade radius in lattice units
    
    Returns:
        List[Dict[str, Any]]: List of result dictionaries containing:
            - 'Ny', 'Delta_over_Omega', 'Rb_over_a': Input parameters
            - 'Energy': Ground state energy (units of Ω)
            - 'All_Indices': List of state indices (0 to 2^N-1)
            - 'All_Probabilities': Corresponding probabilities |ψ_i|²
            - 'Von_Neumann_Entropy': Entanglement entropy of random bipartition (nats)
            - 'Classical_MI': Classical mutual information for same bipartition (nats)
            - 'N_A': Size of subsystem A
            - 'Subsystem_Mask': Binary string indicating subsystem membership
    
    
    Computational Notes:
        - Uses sparse matrix representation (CSR format) for Hamiltonian
        - Employs ARPACK eigensolver (eigsh) for ground state computation
        - Batch processing enables efficient CPU core utilization

    """
    logger = logging.getLogger(__name__)
    batch_results: List[Dict[str, Any]] = []

    if not params_batch:
        logger.warning("Received empty parameter batch. Skipping processing.")
        return batch_results

    # Validate batch consistency (all same Ny for efficient batching)
    Ny_values = [params[0] for params in params_batch]
    if len(set(Ny_values)) != 1:
        logger.warning(f"Inconsistent Ny values in batch: {set(Ny_values)}. Processing individually.")
        # Fallback: process each parameter separately
        for params in params_batch:
            single_result = calculate_properties_cpu_batch([params])
            if single_result:
                batch_results.extend(single_result)
        return batch_results

    Ny = Ny_values[0]
    N = NX * Ny

    # Define Hilbert space for this system size
    hi = nk.hilbert.Spin(s=1/2, N=N)

    # Precompute lattice positions for all parameters in batch (same geometry)
    positions = np.array([
        (col * 1.0, row * 2.0)  # Anisotropic spacing: 1 unit horizontally, 2 units vertically
        for row in range(NX) 
        for col in range(Ny)
    ], dtype=np.float64)

    # Precompute distance matrix and interaction pairs
    dx = positions[:, 0][:, np.newaxis] - positions[:, 0]
    dy = positions[:, 1][:, np.newaxis] - positions[:, 1]
    r_squared = dx.astype(float)**2 + dy.astype(float)**2
    np.fill_diagonal(r_squared, np.inf)  # Avoid self-interaction
    r = np.sqrt(r_squared)
    
    # Identify interacting pairs within cutoff radius
    interaction_mask = (r <= R_CUT)
    interaction_mask = np.triu(interaction_mask, k=1)  # Upper triangle only (i<j)
    valid_pairs = np.argwhere(interaction_mask)
    
    # Precompute interaction strengths for all pairs (same for all params in batch)
    if len(valid_pairs) > 0:
        V_ij_base = 1.0 / (4 * r_squared[interaction_mask] ** 3)
    else:
        V_ij_base = np.array([])

    # Construct Hamiltonians for each parameter set
    H_list: List[csr_matrix] = []
    for params in params_batch:
        _, Delta_over_Omega, Rb_over_a = params
        
        # Scale interaction strengths by (Rb/a)^6 factor
        if len(valid_pairs) > 0:
            V_ij = (Rb_over_a ** 6) * V_ij_base
        else:
            V_ij = np.array([])

        # Build Hamiltonian using NetKet operators
        H = nk.operator.LocalOperator(hi)

        # Precompute spin operators for efficiency
        sigmap_ops = [sigmap(hi, i) for i in range(N)]
        sigmam_ops = [sigmam(hi, i) for i in range(N)]
        sigmaz_ops = [sigmaz(hi, i) for i in range(N)]

        # Rabi drive term: (Ω/2) Σ σˣ = (Ω/4) Σ (σ⁺ + σ⁻) → Ω=1 ⇒ 1/2 Σ (σ⁺ + σ⁻)
        for i in range(N):
            H += 0.5 * (sigmap_ops[i] + sigmam_ops[i])
            # Detuning term: -Δ nᵢ = -(Δ/2)(σᶻ + 1) → -Δ/2 σᶻ - Δ/2
            H += (Delta_over_Omega / 2) * (sigmaz_ops[i] - 1.0)

        # Rydberg interaction term: Vᵢⱼ nᵢ nⱼ = Vᵢⱼ/4 (σᶻᵢ+1)(σᶻⱼ+1)
        for (i, j), V in zip(valid_pairs, V_ij):
            H += V * (sigmaz_ops[i] - 1.0) * (sigmaz_ops[j] - 1.0)

        # Convert to sparse matrix format for eigensolver
        sp_h = H.to_sparse()
        sp_h_cpu = csr_matrix(sp_h)
        H_list.append(sp_h_cpu)

    # Compute ground states via sparse eigensolver
    eig_vals: List[Optional[np.ndarray]] = []
    eig_vecs: List[Optional[np.ndarray]] = []
    
    for H in H_list:
        try:
            # Compute smallest algebraic eigenvalue (ground state)
            val, vec = eigsh(H, k=1, which="SA", tol=1e-8, maxiter=1000)
            eig_vals.append(val)
            eig_vecs.append(vec)
        except Exception as e:
            logger.exception(f"Eigenvalue solver failed: {str(e)}")
            eig_vals.append(None)
            eig_vecs.append(None)

    # Process results and compute entanglement properties
    for idx, (params, E_gs, psi_gs) in enumerate(zip(params_batch, eig_vals, eig_vecs)):
        if E_gs is None or psi_gs is None:
            logger.warning(f"Skipping parameter set {params} due to eigensolver failure")
            continue

        Ny_current, Delta_over_Omega, Rb_over_a = params

        # Normalize wavefunction (eigsh may return unnormalized vectors)
        psi_gs = psi_gs[:, 0].astype(np.complex128)
        norm = np.linalg.norm(psi_gs)
        if norm < 1e-12:
            logger.warning(f"Vanishing wavefunction norm for params {params}. Skipping.")
            continue
        psi_gs /= norm

        # Compute measurement probabilities
        probabilities = np.abs(psi_gs) ** 2
        indices = list(range(len(probabilities)))

        # Generate random bipartition (subsystem A)
        A_size = random.randint(1, N - 1)
        A_indices = random.sample(range(N), A_size)
        A_mask = np.zeros(N, dtype=int)
        A_mask[A_indices] = 1
        A_mask_str = ''.join(str(x) for x in A_mask)

        # Compute von Neumann entanglement entropy
        try:
            psi_reshaped = reshape_for_subsystem(psi_gs, A_indices, N)
            U, s, Vh = np.linalg.svd(psi_reshaped, full_matrices=False)
            s_squared = s**2
            s_squared_normalized = s_squared / np.sum(s_squared)
            # Add epsilon to avoid log(0)
            entropy = -np.sum(s_squared_normalized * np.log(s_squared_normalized + 1e-12))
        except Exception as e:
            logger.exception(f"SVD failed for params {params}: {str(e)}")
            continue

        # Compute classical mutual information for comparison
        try:
            classical_MI = calculate_classical_MI(psi_gs, A_indices, N)
        except Exception as e:
            logger.exception(f"Classical MI calculation failed: {str(e)}")
            classical_MI = 0.0

        batch_results.append({
            'Ny': Ny_current,
            'Delta_over_Omega': Delta_over_Omega,
            'Rb_over_a': Rb_over_a,
            'Energy': float(E_gs[0]),
            'All_Indices': indices,
            'All_Probabilities': probabilities.tolist(),
            'Von_Neumann_Entropy': float(entropy),
            'Classical_MI': float(classical_MI),
            'N_A': A_size,
            'Subsystem_Mask': A_mask_str
        })

    return batch_results


def generate_parameter_samples(
    num_points: int,
    ny_range: Tuple[int, int],
    delta_range: Tuple[float, float],
    rb_range: Tuple[float, float],
    seed: int = 42
) -> Dict[int, List[Tuple[int, float, float]]]:
    """
    Generates random parameter samples grouped by system size (Ny).
    
    Creates uniformly distributed samples across the parameter space while ensuring
    reproducibility through fixed random seed. Grouping by Ny enables memory-aware batching.
    
    Args:
        num_points (int): Total number of parameter configurations to generate.
        ny_range (Tuple[int, int]): Inclusive range of Ny values (columns in lattice).
        delta_range (Tuple[float, float]): Range for Δ/Ω parameter sampling.
        rb_range (Tuple[float, float]): Range for R_b/a parameter sampling.
        seed (int): Random seed for reproducible sampling (default: 42).
    
    Returns:
        Dict[int, List[Tuple[int, float, float]]]: Dictionary mapping Ny values to
            lists of parameter tuples (Ny, Δ/Ω, R_b/a).


    """
    random.seed(seed)
    np.random.seed(seed)
    
    parameter_list = [
        (
            np.random.randint(ny_range[0], ny_range[1] + 1),
            np.random.uniform(delta_range[0], delta_range[1]),
            np.random.uniform(rb_range[0], rb_range[1])
        )
        for _ in range(num_points)
    ]
    
    # Group by Ny for efficient batching
    grouped_params = defaultdict(list)
    for params in parameter_list:
        Ny = params[0]
        grouped_params[Ny].append(params)
    
    return grouped_params


def create_parquet_schema() -> pa.Schema:
    """
    Defines Apache Parquet schema for Rydberg simulation data storage.
    
    Returns:
        pyarrow.Schema: Schema object defining column names and data types.
    
    Schema Details:
        - Integer fields: Ny, N_A (64-bit integers)
        - Float fields: All continuous parameters and computed quantities (64-bit floats)
        - List fields: All_Indices (int64), All_Probabilities (float64)
        - String field: Subsystem_Mask (binary string representation of bipartition)

    """
    return pa.schema([
        ('Ny', pa.int64()),
        ('Delta_over_Omega', pa.float64()),
        ('Rb_over_a', pa.float64()),
        ('Energy', pa.float64()),
        ('All_Indices', pa.list_(pa.int64())),
        ('All_Probabilities', pa.list_(pa.float64())),
        ('Von_Neumann_Entropy', pa.float64()),
        ('Classical_MI', pa.float64()),
        ('N_A', pa.int64()),
        ('Subsystem_Mask', pa.string())
    ])


def main() -> None:
    """

    Execution Workflow:
        1. Configure logging system (file + console output)
        2. Generate random parameter samples across Hamiltonian parameter space
        3. Group parameters by system size (Ny) for memory-aware batching
        4. Determine batch sizes based on system complexity (smaller batches for larger N)
        5. Parallelize computation across available CPU cores using ProcessPoolExecutor
        6. Stream results directly to Parquet file to minimize memory footprint
        7. Report comprehensive timing statistics and completion metrics
    
    
    Memory Management:
        - Batch sizing prevents OOM errors for large systems (N=12)
        - Direct streaming to Parquet avoids accumulating full dataset in memory
        - Worker processes release memory immediately after batch completion
    
    Output Validation:
        Generated Parquet file contains exactly one row per successfully processed
        parameter configuration with all quantum state information preserved for
        downstream machine learning tasks 
    
    Raises:
        Exception: Any unhandled exception during computation (logged with traceback).
    """
    logger = setup_logging()
    logger.info("Starting Rydberg atom simulation data generation pipeline")
    logger.info(f"Configuration: Nx={NX}, Ny range={NY_RANGE}, "
                f"Δ/Ω range={DELTA_OVER_OMEGA_RANGE}, R_b/a range={RB_OVER_A_RANGE}")
    logger.info(f"Total parameter samples: {NUM_POINTS}, Random seed: {RANDOM_SEED}")
    
    # Generate and group parameter samples
    grouped_params = generate_parameter_samples(
        num_points=NUM_POINTS,
        ny_range=NY_RANGE,
        delta_range=DELTA_OVER_OMEGA_RANGE,
        rb_range=RB_OVER_A_RANGE,
        seed=RANDOM_SEED
    )
    
    # Create parameter batches with size adapted to system complexity
    all_param_batches: List[List[Tuple[int, float, float]]] = []
    total_samples = 0
    
    for Ny, params_list in grouped_params.items():
        N = NX * Ny
        batch_size = BATCH_SIZES.get(N, 5)  # Fallback to conservative size
        
        # Split into batches
        for i in range(0, len(params_list), batch_size):
            batch = params_list[i:i + batch_size]
            all_param_batches.append(batch)
            total_samples += len(batch)
        
        logger.info(f"Ny={Ny} (N={N} sites): {len(params_list)} samples → "
                   f"{(len(params_list) + batch_size - 1)//batch_size} batches "
                   f"(batch size={batch_size})")
    
    logger.info(f"Total batches to process: {len(all_param_batches)} "
               f"(representing {total_samples} parameter configurations)")
    
    # Initialize Parquet writer with explicit schema
    schema = create_parquet_schema()
    writer = pq.ParquetWriter(OUTPUT_FILENAME, schema)
    
    # Parallel processing with progress tracking
    start_time_all = time.time()
    completed_batches = 0
    successful_samples = 0
    
    with ProcessPoolExecutor() as executor:
        # Submit all batches for processing
        futures = [
            executor.submit(calculate_properties_cpu_batch, batch) 
            for batch in all_param_batches
        ]
        
        # Process results as they complete
        for future in as_completed(futures):
            completed_batches += 1
            
            try:
                batch_results = future.result()
            except Exception as e:
                logger.exception(f"Worker process failed for batch {completed_batches}: {str(e)}")
                continue
            
            # Write successful results to Parquet file
            if batch_results:
                df_batch = pd.DataFrame(batch_results)
                table = pa.Table.from_pandas(df_batch, schema=schema, preserve_index=False)
                writer.write_table(table)
                successful_samples += len(batch_results)
            
            # Progress logging every 10% of batches
            if completed_batches % max(1, len(all_param_batches)//10) == 0:
                elapsed = time.time() - start_time_all
                rate = completed_batches / elapsed if elapsed > 0 else 0
                logger.info(f"Progress: {completed_batches}/{len(all_param_batches)} batches "
                           f"({completed_batches/len(all_param_batches)*100:.1f}%) | "
                           f"Successful samples: {successful_samples} | "
                           f"Rate: {rate:.2f} batches/sec")
    
    writer.close()
    total_time = time.time() - start_time_all
    
    # Final statistics
    logger.info("=" * 70)
    logger.info("DATA GENERATION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Output file: {OUTPUT_FILENAME}")
    logger.info(f"Total batches processed: {completed_batches}/{len(all_param_batches)}")
    logger.info(f"Successful samples: {successful_samples} / {total_samples} "
               f"({successful_samples/total_samples*100:.2f}%)")
    logger.info(f"Total execution time: {total_time:.2f} seconds "
               f"({total_time/60:.2f} minutes)")
    logger.info(f"Average throughput: {successful_samples/total_time:.2f} samples/second")
    logger.info(f"Disk usage: {os.path.getsize(OUTPUT_FILENAME)/1024/1024:.2f} MB")
    logger.info("=" * 70)
    
    # Quick validation check
    try:
        test_df = pd.read_parquet(OUTPUT_FILENAME)
        logger.info(f"Validation: Successfully read {len(test_df)} rows from output file")
        logger.info(f"Column check: {list(test_df.columns)}")
    except Exception as e:
        logger.error(f"Validation failed: {str(e)}")


if __name__ == "__main__":
    main()