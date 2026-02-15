import os
from typing import Any, Dict

# Validate critical paths at import time
PROCESSED_DIR = './processed_experimentalrung1_6'
if not os.path.exists(PROCESSED_DIR):
    raise FileNotFoundError(
        f"Processed data directory not found: {PROCESSED_DIR}\n"
        "Run data generation pipeline first to create this directory."
    )

# Core training configuration with type annotations
CONFIG: Dict[str, Any] = {
    # Data configuration
    'processed_dir': PROCESSED_DIR,
    'processed_file_name': 'data.pt',
    'batch_size': 256,          # Per-GPU batch size (adjust based on GPU memory)
    
    # Optimization hyperparameters
    'learning_rate': 1.5e-4,    # AdamW base learning rate
    'weight_decay': 1e-4,       # L2 regularization strength
    'grad_clip': 1.0,           # Gradient norm clipping threshold
    
    # Model architecture
    'hidden_channels': 512,     # Base dimensionality for all hidden layers
    'num_layers': 6,            # Number of message passing layers
    'dropout_p': 0.4,           # Dropout probability for regularization
    
    # Training protocol
    'num_epochs': 500,          # Maximum training epochs
    'random_seed': 42,          # Global random seed for reproducibility
    
    # Checkpointing
    'best_model_path': 'best_model_rung1_6.pth',  # Path for best validation model
    
    # Learning rate schedule (CosineAnnealingWarmRestarts)
    'scheduler_T0': 50,         # Initial cycle length in epochs
    'scheduler_Tmult': 2,       # Cycle length multiplier
    'scheduler_eta_min': 1e-6,  # Minimum learning rate
}

# Feature schema definitions (for model input validation)
FEATURE_SCHEMA = {
    'node_features': {
        'position_x': 0,
        'position_y': 1,
        'rydberg_prob': 2,
        'subsystem_mask': 3,
        'total_dim': 4
    },
    'edge_features': {
        'angle': 0,
        'connected_corr': 1,
        'normalized_dist': 2,
        'total_dim': 3
    },
    'graph_features': ['system_size', 'nA', 'nB']
}

def validate_config() -> None:
    """
    Validates configuration values against physical and computational constraints.
    
    Raises:
        ValueError: If any configuration parameter violates domain constraints.
        FileNotFoundError: If required data directories don't exist.
    
    Validation Rules:
        - Batch size must be positive integer
        - Learning rate must be in (0, 1)
        - Dropout probability must be in [0, 1)
        - Hidden channels must be power of 2 (for efficient GPU memory alignment)
        - Model path must have .pth extension
    """
    if CONFIG['batch_size'] <= 0:
        raise ValueError(f"Invalid batch_size: {CONFIG['batch_size']} (must be > 0)")
    if not (0 < CONFIG['learning_rate'] < 1):
        raise ValueError(f"Invalid learning_rate: {CONFIG['learning_rate']} (must be in (0,1))")
    if not (0 <= CONFIG['dropout_p'] < 1):
        raise ValueError(f"Invalid dropout_p: {CONFIG['dropout_p']} (must be in [0,1))")
    if CONFIG['hidden_channels'] & (CONFIG['hidden_channels'] - 1) != 0:
        raise ValueError(f"hidden_channels={CONFIG['hidden_channels']} should be power of 2 for optimal GPU performance")
    if not CONFIG['best_model_path'].endswith('.pth'):
        raise ValueError(f"Model path must end with .pth extension: {CONFIG['best_model_path']}")

# Execute validation at import time
validate_config()