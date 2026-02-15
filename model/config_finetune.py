"""
Fine-tuning configuration for transferring Rydberg GNN from small (N=2-12) 
to larger systems (N=14-16).

Contains hyperparameters specifically tuned for stable transfer learning:
  - Reduced learning rate to prevent catastrophic forgetting
  - Smaller batch size to accommodate larger graph memory footprint
  - Tighter gradient clipping for training stability on larger systems
"""

from typing import Any, Dict

FINETUNE_CONFIG: Dict[str, Any] = {
    'pretrained_model_path': 'best_model_rung1_6.pth',      # Path to base model trained on N=2-12
    'processed_dir_larger': './processed_experimentalrung7-8_10k',  # Larger system data (N=14-16)
    'processed_file_name': 'data.pt',
    'batch_size': 128,          # Reduced from 256 due to larger graph sizes
    'learning_rate': 0.5e-4,    # Conservative LR for stable transfer learning
    'weight_decay': 1.5e-4,
    'num_epochs': 200,
    'patience': 50,             # Early stopping patience (epochs without improvement)
    'finetuned_model_path': 'finetuned_model.pth',
    'dropout_p': 0.3,           # Slightly reduced dropout for better signal retention
    'grad_clip': 0.5,           # Tighter clipping for stability on larger systems
    'random_seed': 42
}