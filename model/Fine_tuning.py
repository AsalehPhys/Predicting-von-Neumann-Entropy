"""
Fine-tuning pipeline for Rydberg entanglement entropy GNN on larger systems.

Transfers knowledge from a model pretrained on small systems (N=2-12) to 
larger systems.

Workflow:
    1. Load pretrained weights from small-system training
    2. Load larger-system dataset
    3. Train with reduced learning rate to avoid catastrophic forgetting
    4. Early stopping based on validation loss
    5. Save best model checkpoint

Note:
    This script assumes single-GPU execution (no distributed training).
    For multi-GPU fine-tuning, integrate with train.py's DDP setup.
"""

import logging

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader

from model.config_finetune import FINETUNE_CONFIG
from model.model import ExperimentalGNN, PhysicalScaleAwareLoss
from model.Training import RydbergDataset


def setup_logging() -> None:
    """
    Configures console logging with timestamped format for training progress tracking.
    
    Sets log level to INFO to capture epoch metrics and checkpoint events.
    Format: YYYY-MM-DD HH:MM:SS [LEVEL] message
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def fine_tune_model() -> None:
    """
    Executes fine-tuning of pretrained Rydberg GNN on larger system sizes.
    
    Steps:
        1. Initialize logging and device (GPU if available)
        2. Load pretrained model architecture and weights
        3. Load larger-system dataset and split 80/20 train/val
        4. Configure optimizer with reduced learning rate for stable transfer
        5. Train for up to num_epochs with early stopping (patience=50)
        6. Save best model based on validation loss
    
    """
    setup_logging()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    # Load pretrained model architecture and weights
    model = ExperimentalGNN(
        hidden_channels=512,
        dropout_p=FINETUNE_CONFIG['dropout_p']
    ).to(device)
    
    pretrained_state_dict = torch.load(
        FINETUNE_CONFIG['pretrained_model_path'], 
        map_location=device
    )
    model.load_state_dict(pretrained_state_dict)
    logging.info(f"Loaded pretrained model from {FINETUNE_CONFIG['pretrained_model_path']}")

    # Load larger-system dataset 
    dataset = RydbergDataset(root=FINETUNE_CONFIG['processed_dir_larger'])
    logging.info(f"Loaded dataset with {len(dataset)} samples")
    
    # 80/20 train/validation split with fixed seed for reproducibility
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(FINETUNE_CONFIG['random_seed'])
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=FINETUNE_CONFIG['batch_size'], 
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=FINETUNE_CONFIG['batch_size'], 
        shuffle=False
    )

    # Loss and optimizer with conservative learning rate for transfer learning
    criterion = PhysicalScaleAwareLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=FINETUNE_CONFIG['learning_rate'],
        weight_decay=FINETUNE_CONFIG['weight_decay']
    )

    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=20,      # Shorter cycle for faster adaptation
        T_mult=2,
        eta_min=1e-7
    )

    # Training loop with early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(FINETUNE_CONFIG['num_epochs']):
        # Training phase
        model.train()
        total_train_loss = 0
        train_mae = 0
        total_train_samples = 0
        
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
        
            pred_s = model(data)
            targets = data.y.squeeze().to(device)
            system_size = data.system_size.squeeze(-1).to(device)
            subsystem_size = data.nA.squeeze(-1).to(device)
            
            loss = criterion(pred_s, targets, system_size, subsystem_size)
            loss.backward()
            
            if FINETUNE_CONFIG['grad_clip'] is not None:
                nn.utils.clip_grad_norm_(model.parameters(), FINETUNE_CONFIG['grad_clip'])
            
            optimizer.step()
            
            mae = torch.abs(pred_s - targets).sum().item()
            train_mae += mae
            total_train_samples += data.num_graphs
            total_train_loss += loss.item() * data.num_graphs

        avg_train_loss = total_train_loss / len(train_dataset)
        avg_train_mae = train_mae / total_train_samples
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        val_mae = 0
        total_val_samples = 0
        
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                pred_s = model(data)
                targets = data.y.squeeze().to(device)
                system_size = data.system_size.squeeze(-1).to(device)
                subsystem_size = data.nA.squeeze(-1).to(device)
                
                loss = criterion(pred_s, targets, system_size, subsystem_size)
                total_val_loss += loss.item() * data.num_graphs
                
                mae = torch.abs(pred_s - targets).sum().item()
                val_mae += mae
                total_val_samples += data.num_graphs

        avg_val_loss = total_val_loss / len(val_dataset)
        avg_val_mae = val_mae / total_val_samples
        scheduler.step()

        # Logging
        logging.info(f'Epoch {epoch+1}/{FINETUNE_CONFIG["num_epochs"]}:')
        logging.info(f'  Training Loss: {avg_train_loss:.6f}')
        logging.info(f'  Training MAE: {avg_train_mae:.6f}')
        logging.info(f'  Validation Loss: {avg_val_loss:.6f}')
        logging.info(f'  Validation MAE: {avg_val_mae:.6f}')

        # Early stopping and checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), FINETUNE_CONFIG['finetuned_model_path'])
            logging.info(f'  Saved new best model (val_loss={best_val_loss:.6f})')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= FINETUNE_CONFIG['patience']:
                logging.info('Early stopping triggered')
                break

    logging.info('Fine-tuning completed')
    logging.info(f'Best validation loss: {best_val_loss:.6f}')


if __name__ == "__main__":
    fine_tune_model()