"""
Configuration module for Rydberg atom graph dataset processing pipeline.

Contains all hyperparameters, file paths, and processing constants required
for converting raw Rydberg atom simulation data into PyTorch Geometric graphs.
"""

from typing import Dict

CONFIG: Dict = {
    'data_paths': [
        'Rydbergsize2.parquet',
        'Rydbergsize1-5.parquet',
        'Rydberg0.5M.parquet',
    ],
    'processed_dir': './processed',
    'processed_file_name': 'data.pt',
    'random_seed': 42,
    'chunk_size': 400,          # Number of rows to process in a single chunk
    'use_gpu': False,           # Whether to use GPU for tensor operations during processing
    'save_interval': 150,       # Save intermediate results every N chunks
    'num_workers': None,        # Will be set dynamically at runtime (see set_dynamic_config)
    'batch_size': 5,            # Number of chunks to process in parallel batches
    'timeout': 600,             # Timeout (seconds) for processing individual chunks
    'max_points_per_ny': {      # Maximum samples to retain per system size (Ny dimension)
        1: 30000,
        2: 60000,
        3: 100000,
        4: 200000,
        5: 350000,
        6: 500000,
    }
}


def set_dynamic_config() -> None:
    """
    Sets dynamic configuration values that depend on runtime environment.
    
    Currently sets:
        - num_workers: min(available CPUs - 1, 40) to avoid overloading the system
    
    This function should be called once at application startup before CONFIG is used.
    """
    import multiprocessing as mp
    CONFIG['num_workers'] = min(mp.cpu_count() - 1, 40)