"""
Memory-optimized data processing utilities for resource-constrained systems.

Includes batch size optimization, memory monitoring, and efficient data loading.
"""

import torch
import psutil
import gc
from typing import Tuple, Optional, Iterator
import numpy as np


class MemoryOptimizer:
    """Utilities for optimizing memory usage on low-spec systems."""
    
    @staticmethod
    def get_optimal_batch_size(model: torch.nn.Module, input_shape: Tuple[int, ...], 
                              target_memory_gb: float = 4.0) -> int:
        """
        Determine optimal batch size based on available memory.
        
        Args:
            model: PyTorch model
            input_shape: Input tensor shape (C, H, W)
            target_memory_gb: Target memory usage in GB
            
        Returns:
            Optimal batch size
        """
        # Get system memory
        ram_gb = psutil.virtual_memory().total / (1024**3)
        available_gb = min(ram_gb * 0.6, target_memory_gb)  # Use 60% of RAM max
        
        # Calculate memory per sample (rough estimate)
        # Input: batch_size * C * H * W * 4 bytes (float32)
        input_memory_per_sample = np.prod(input_shape) * 4 / (1024**3)
        
        # Model parameters
        param_count = sum(p.numel() for p in model.parameters())
        model_memory = param_count * 4 / (1024**3)
        
        # Estimate forward pass memory (2-3x input for activations)
        forward_memory_per_sample = input_memory_per_sample * 3
        
        # Calculate batch size
        memory_per_sample = input_memory_per_sample + forward_memory_per_sample
        available_for_batch = available_gb - model_memory - 1.0  # 1GB buffer
        
        if available_for_batch <= 0:
            return 1
        
        optimal_batch_size = max(1, int(available_for_batch / memory_per_sample))
        
        # Cap at reasonable limits
        return min(optimal_batch_size, 32)
    
    @staticmethod
    def optimize_for_system() -> dict:
        """
        Get optimization settings based on current system specs.
        
        Returns:
            Dictionary with optimization settings
        """
        ram_gb = psutil.virtual_memory().total / (1024**3)
        cpu_count = psutil.cpu_count() or 4
        gpu_available = torch.cuda.is_available()
        
        if gpu_available:
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        else:
            gpu_memory_gb = 0
        
        # Determine settings based on system specs
        if ram_gb < 8:
            # Very low memory system
            settings = {
                'model_type': 'lightweight',
                'batch_size': 1,
                'num_workers': 1,
                'pin_memory': False,
                'patch_size': 224,
                'use_mixed_precision': False,
                'gradient_accumulation': 4
            }
        elif ram_gb < 16:
            # Low memory system (8-16GB)
            settings = {
                'model_type': 'lightweight',
                'batch_size': 2,
                'num_workers': min(2, cpu_count // 2),
                'pin_memory': gpu_available,
                'patch_size': 224,
                'use_mixed_precision': gpu_available,
                'gradient_accumulation': 2
            }
        else:
            # Normal system (16GB+)
            settings = {
                'model_type': 'enhanced',
                'batch_size': 8,
                'num_workers': min(4, cpu_count // 2),
                'pin_memory': gpu_available,
                'patch_size': 224,
                'use_mixed_precision': gpu_available,
                'gradient_accumulation': 1
            }
        
        # Adjust for GPU memory if available
        if gpu_available and gpu_memory_gb < 6:
            settings['batch_size'] = min(settings['batch_size'], 2)
            settings['use_mixed_precision'] = True
        
        return settings
    
    @staticmethod
    def clear_memory():
        """Clear GPU and system memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    @staticmethod
    def get_memory_usage() -> dict:
        """Get current memory usage statistics."""
        stats = {
            'ram_used_gb': psutil.virtual_memory().used / (1024**3),
            'ram_available_gb': psutil.virtual_memory().available / (1024**3),
            'ram_percent': psutil.virtual_memory().percent
        }
        
        if torch.cuda.is_available():
            stats['gpu_allocated_gb'] = torch.cuda.memory_allocated() / (1024**3)
            stats['gpu_reserved_gb'] = torch.cuda.memory_reserved() / (1024**3)
        
        return stats


class EfficientDataLoader:
    """Memory-efficient data loader for WSI processing."""
    
    def __init__(self, batch_size: Optional[int] = None, num_workers: Optional[int] = None, 
                 pin_memory: Optional[bool] = None):
        # Auto-configure if not specified
        if batch_size is None or num_workers is None or pin_memory is None:
            settings = MemoryOptimizer.optimize_for_system()
            self.batch_size = batch_size or settings['batch_size']
            self.num_workers = num_workers or settings['num_workers']
            self.pin_memory = pin_memory or settings['pin_memory']
        else:
            self.batch_size = batch_size
            self.num_workers = num_workers
            self.pin_memory = pin_memory
    
    def create_dataloader(self, dataset, shuffle: bool = True) -> torch.utils.data.DataLoader:
        """Create optimized DataLoader."""
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            persistent_workers=self.num_workers > 0
        )


def process_wsi_efficiently(wsi_patches, model, device, progress_callback=None):
    """
    Process WSI patches efficiently with memory monitoring.
    
    Args:
        wsi_patches: List or generator of patches
        model: PyTorch model
        device: Processing device
        progress_callback: Optional progress callback function
        
    Returns:
        List of predictions
    """
    model.eval()
    predictions = []
    
    # Get optimal batch size
    optimizer = MemoryOptimizer()
    settings = optimizer.optimize_for_system()
    batch_size = settings['batch_size']
    
    print(f"Processing with batch size: {batch_size}")
    
    # Process in batches
    for i in range(0, len(wsi_patches), batch_size):
        batch_patches = wsi_patches[i:i + batch_size]
        
        # Convert to tensor if needed
        if not isinstance(batch_patches, torch.Tensor):
            batch_patches = torch.stack([torch.from_numpy(p) if isinstance(p, np.ndarray) 
                                       else p for p in batch_patches])
        
        batch_patches = batch_patches.to(device)
        
        # Forward pass
        with torch.no_grad():
            if settings['use_mixed_precision']:
                with torch.cuda.amp.autocast():
                    batch_pred = model(batch_patches)
            else:
                batch_pred = model(batch_patches)
            
            predictions.extend(batch_pred.cpu().numpy())
        
        # Memory cleanup
        del batch_patches
        if i % (batch_size * 5) == 0:  # Clean every 5 batches
            optimizer.clear_memory()
        
        # Progress callback
        if progress_callback:
            progress = (i + batch_size) / len(wsi_patches)
            progress_callback(min(progress, 1.0))
    
    return predictions


def print_memory_requirements():
    """Print memory requirements for different configurations."""
    print("Memory Requirements by Configuration:")
    print("=" * 50)
    
    configs = [
        ("Lightweight (8GB system)", "lightweight", 2),
        ("Standard (16GB system)", "enhanced", 8),
        ("High-performance (32GB system)", "enhanced", 16)
    ]
    
    for name, model_type, batch_size in configs:
        if model_type == "lightweight":
            params = 5_000_000  # ~5M parameters
        else:
            params = 51_000_000  # ~51M parameters
        
        model_memory = params * 4 / (1024**2)  # MB
        batch_memory = batch_size * 3 * 224 * 224 * 4 / (1024**2)  # MB
        total_memory = model_memory + batch_memory + 500  # +500MB overhead
        
        print(f"{name}:")
        print(f"  Model: {model_memory:.0f}MB")
        print(f"  Batch: {batch_memory:.0f}MB")
        print(f"  Total: {total_memory:.0f}MB")
        print()


if __name__ == "__main__":
    print_memory_requirements()
    
    # Test optimization
    settings = MemoryOptimizer.optimize_for_system()
    print("Recommended settings for this system:")
    for key, value in settings.items():
        print(f"  {key}: {value}")
