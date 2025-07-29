
"""
Training script for multi-tissue WSI damage scoring.

Includes cross-validation, early stopping, learning rate scheduling, and comprehensive logging.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import yaml
import numpy as np
import logging
from sklearn.model_selection import KFold, train_test_split
from typing import Dict, Tuple, List, Optional
import time
import os
from pathlib import Path

from src.data.preprocess import load_test_data, preprocess_images
from src.models.cnn import build_model
from src.evaluate import evaluate_model, calculate_comprehensive_metrics

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TISSUE_TYPES = ["lung", "kidney", "heart", "liver", "bowel"]

def one_hot_tissue(tissue: str) -> torch.Tensor:
    """Convert tissue type to one-hot encoding."""
    if tissue not in TISSUE_TYPES:
        raise ValueError(f"Unknown tissue type: {tissue}")
    
    idx = TISSUE_TYPES.index(tissue)
    arr = torch.zeros(len(TISSUE_TYPES), dtype=torch.float32)
    arr[idx] = 1.0
    return arr

class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.patience_counter = 0
        
    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            
        return self.patience_counter >= self.patience

def create_data_loaders(train_data: torch.Tensor, train_labels: torch.Tensor, 
                       tissue_labels: List[str], batch_size: int = 32, 
                       validation_split: float = 0.2) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation data loaders."""
    
    # Convert tissue labels to one-hot
    tissue_onehot = torch.stack([one_hot_tissue(t) for t in tissue_labels])
    
    # Split data
    indices = list(range(len(train_data)))
    train_idx, val_idx = train_test_split(indices, test_size=validation_split, 
                                         random_state=42, stratify=train_labels)
    
    # Create datasets
    train_dataset = TensorDataset(
        train_data[train_idx], 
        train_labels[train_idx], 
        tissue_onehot[train_idx]
    )
    
    val_dataset = TensorDataset(
        train_data[val_idx], 
        train_labels[val_idx], 
        tissue_onehot[val_idx]
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

def train_epoch(model: nn.Module, train_loader: DataLoader, criterion: nn.Module,
               optimizer: optim.Optimizer, device: torch.device) -> Tuple[float, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    for batch_idx, (inputs, targets, tissue_onehot) in enumerate(train_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        tissue_onehot = tissue_onehot.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs, tissue_onehot)
        loss = criterion(outputs, targets.long())
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_samples += targets.size(0)
        correct_predictions += (predicted == targets).sum().item()
    
    avg_loss = total_loss / len(train_loader)
    accuracy = correct_predictions / total_samples
    
    return avg_loss, accuracy

def validate_epoch(model: nn.Module, val_loader: DataLoader, criterion: nn.Module,
                  device: torch.device) -> Tuple[float, float]:
    """Validate for one epoch."""
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    with torch.no_grad():
        for inputs, targets, tissue_onehot in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            tissue_onehot = tissue_onehot.to(device)
            
            outputs = model(inputs, tissue_onehot)
            loss = criterion(outputs, targets.long())
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += targets.size(0)
            correct_predictions += (predicted == targets).sum().item()
    
    avg_loss = total_loss / len(val_loader)
    accuracy = correct_predictions / total_samples
    
    return avg_loss, accuracy

def train_model(train_data: torch.Tensor, train_labels: torch.Tensor, 
               tissue_labels: List[str], model: nn.Module, config: Dict,
               device: torch.device, model_save_path: str = 'best_model.pth') -> Dict:
    """
    Train model with comprehensive monitoring and early stopping.
    
    Args:
        train_data: Training images
        train_labels: Training labels (damage scores 0-9)
        tissue_labels: Tissue type labels
        model: Model to train
        config: Configuration dictionary
        device: Training device
        model_save_path: Path to save best model
    
    Returns:
        Training history dictionary
    """
    model.to(device)
    
    # Training parameters
    epochs = config['training']['epochs']
    batch_size = config['training']['batch_size']
    learning_rate = config['training']['learning_rate']
    validation_split = config['training']['validation_split']
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        train_data, train_labels, tissue_labels, batch_size, validation_split
    )
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=10)
    
    # TensorBoard logging
    if config['logging']['use_tensorboard']:
        log_dir = Path(config['logging']['log_dir'])
        log_dir.mkdir(exist_ok=True)
        writer = SummaryWriter(log_dir)
    else:
        writer = None
    
    # Training history
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': [],
        'learning_rates': []
    }
    
    best_val_loss = float('inf')
    
    logger.info(f"Starting training for {epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # Training phase
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validation phase
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)
        history['learning_rates'].append(current_lr)
        
        # TensorBoard logging
        if writer:
            writer.add_scalar('Loss/Train', train_loss, epoch)
            writer.add_scalar('Loss/Validation', val_loss, epoch)
            writer.add_scalar('Accuracy/Train', train_acc, epoch)
            writer.add_scalar('Accuracy/Validation', val_acc, epoch)
            writer.add_scalar('Learning_Rate', current_lr, epoch)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': config
            }, model_save_path)
            logger.info(f"New best model saved at epoch {epoch + 1}")
        
        epoch_time = time.time() - epoch_start
        
        logger.info(
            f"Epoch {epoch + 1}/{epochs} "
            f"- Train Loss: {train_loss:.4f} "
            f"- Train Acc: {train_acc:.4f} "
            f"- Val Loss: {val_loss:.4f} "
            f"- Val Acc: {val_acc:.4f} "
            f"- LR: {current_lr:.6f} "
            f"- Time: {epoch_time:.1f}s"
        )
        
        # Early stopping check
        if early_stopping(val_loss):
            logger.info(f"Early stopping triggered at epoch {epoch + 1}")
            break
    
    total_time = time.time() - start_time
    logger.info(f"Training completed in {total_time:.1f} seconds")
    
    if writer:
        writer.close()
    
    return history

def cross_validate_model(train_data: torch.Tensor, train_labels: torch.Tensor,
                        tissue_labels: List[str], config: Dict, 
                        device: torch.device, k_folds: int = 5) -> List[Dict]:
    """
    Perform k-fold cross-validation.
    
    Args:
        train_data: Training images
        train_labels: Training labels
        tissue_labels: Tissue type labels
        config: Configuration dictionary
        device: Training device
        k_folds: Number of folds
    
    Returns:
        List of evaluation results for each fold
    """
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_data)):
        logger.info(f"Training fold {fold + 1}/{k_folds}")
        
        # Split data for this fold
        fold_train_data = train_data[train_idx]
        fold_train_labels = train_labels[train_idx]
        fold_train_tissues = [tissue_labels[i] for i in train_idx]
        
        fold_val_data = train_data[val_idx]
        fold_val_labels = train_labels[val_idx]
        fold_val_tissues = [tissue_labels[i] for i in val_idx]
        
        # Create and train model
        model = build_model(
            input_shape=tuple(config["model"]["input_shape"]),
            num_tissues=config["model"]["num_tissues"],
            num_classes=config["model"]["num_classes"]
        )
        
        # Train model for this fold
        fold_config = config.copy()
        fold_config['training']['epochs'] = min(config['training']['epochs'], 20)  # Reduce epochs for CV
        
        train_model(fold_train_data, fold_train_labels, fold_train_tissues, 
                   model, fold_config, device, f'fold_{fold + 1}_model.pth')
        
        # Evaluate on validation set
        results = evaluate_model(model, fold_val_data, fold_val_labels, 
                               fold_val_tissues, device=device)
        
        fold_results.append(results)
        
        logger.info(f"Fold {fold + 1} - Accuracy: {results['accuracy']:.3f}, "
                   f"MAE: {results['mae']:.3f}")
    
    return fold_results

def main():
    """Main training function."""
    # Load configuration
    with open("src/config.yaml") as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load and preprocess data
    # Note: Replace with actual data loading
    train_data, train_labels = load_test_data('path/to/training/data')
    tissue_labels = ["lung"] * len(train_labels)  # Replace with actual tissue labels
    
    # For demonstration, create synthetic multi-tissue data
    logger.info("Creating synthetic multi-tissue data for demonstration...")
    num_samples = len(train_labels)
    tissue_labels = np.random.choice(TISSUE_TYPES, num_samples).tolist()
    
    # Apply preprocessing if reference images are available
    if all(os.path.exists(path) for path in config["preprocessing"]["reference_images"].values()):
        target_image = np.random.rand(224, 224, 3) * 255  # Placeholder
        train_data = preprocess_images(train_data, target_image, 
                                     config["preprocessing"]["normalizer_method"])
    
    # Convert to tensors
    train_data = torch.tensor(train_data, dtype=torch.float32)
    if train_data.dim() == 4 and train_data.shape[-1] == 3:  # NHWC to NCHW
        train_data = train_data.permute(0, 3, 1, 2)
    
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    
    # Create model
    model = build_model(
        input_shape=tuple(config["model"]["input_shape"]),
        num_tissues=config["model"]["num_tissues"],
        num_classes=config["model"]["num_classes"]
    )
    
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Perform cross-validation if enabled
    if config['training'].get('cross_validation', False):
        logger.info("Performing cross-validation...")
        cv_results = cross_validate_model(
            train_data, train_labels, tissue_labels, config, device,
            config['training'].get('k_folds', 5)
        )
        
        # Average results across folds
        avg_accuracy = np.mean([r['accuracy'] for r in cv_results])
        avg_mae = np.mean([r['mae'] for r in cv_results])
        
        logger.info(f"Cross-validation results:")
        logger.info(f"Average Accuracy: {avg_accuracy:.3f} ± {np.std([r['accuracy'] for r in cv_results]):.3f}")
        logger.info(f"Average MAE: {avg_mae:.3f} ± {np.std([r['mae'] for r in cv_results]):.3f}")
    
    # Train final model on all data
    logger.info("Training final model on all data...")
    history = train_model(train_data, train_labels, tissue_labels, model, config, device)
    
    logger.info("Training completed successfully!")
    
    return model, history

if __name__ == "__main__":
    model, history = main()
    tissue_onehot = torch.stack([one_hot_tissue(t) for t in tissue_labels])

    model = build_model(input_shape=tuple(config["model"]["input_shape"]), num_tissues=config["model"]["num_tissues"], num_classes=config["model"]["num_classes"])
    train_model(train_data, train_labels, tissue_onehot, model, epochs=config["training"]["epochs"], batch_size=config["training"]["batch_size"], device=device, model_path='path/to/save/model.pth')

if __name__ == "__main__":
    main()