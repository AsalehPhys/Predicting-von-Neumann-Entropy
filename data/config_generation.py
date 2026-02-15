"""
Configuration module for Rydberg atom quantum simulation data generation.

Defines physical parameters, sampling strategies, and computational settings
for generating quantum many-body simulation data of Rydberg atom arrays
in rectangular 2×Ny geometries. Parameters are chosen to cover experimentally
relevant regimes of the Rydberg Hamiltonian.
"""

from typing import Dict, Tuple

# Physical system constants
NX: int = 2  # Fixed number of rows in rectangular lattice geometry
R_CUT: float = 6.0  # Interaction cutoff radius (in units of lattice spacing)
                    # Beyond this distance, van der Waals interactions are neglected

# Hamiltonian parameter sampling ranges
NY_RANGE: Tuple[int, int] = (1, 6)  # Range of columns (system sizes: 2×1 to 2×6 sites)
DELTA_OVER_OMEGA_RANGE: Tuple[float, float] = (0.0, 4.0)  # Detuning-to-Rabi frequency ratio
RB_OVER_A_RANGE: Tuple[float, float] = (0.1, 5.0)  # Blockade radius to lattice spacing ratio

# Dataset generation parameters
NUM_POINTS: int = 10000  # Total number of random parameter configurations to sample
RANDOM_SEED: int = 42  # Seed for reproducible random sampling of parameters

# Computational batching strategy (memory-aware)
# Keys correspond to total system size N = Nx * Ny
BATCH_SIZES: Dict[int, int] = {
    2: 10000,   # 2×1 system (2 sites) - very fast diagonalization
    4: 10000,   # 2×2 system (4 sites)
    6: 500,     # 2×3 system (6 sites)
    8: 500,     # 2×4 system (8 sites)
    10: 50,    # 2×5 system (10 sites)
    12: 50     # 2×6 system (12 sites) - largest system in our range
}

# Output configuration
OUTPUT_FILENAME: str = 'Rydberg_data.parquet'  # Parquet output file for efficient storage
LOG_FILENAME: str = 'rydberg_system.log'        # Log file for execution tracking