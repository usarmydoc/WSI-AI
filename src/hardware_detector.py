"""
Automatic hardware detection and model selection for WSI AI system.

This module automatically detects system capabilities and selects the optimal
model configuration without user intervention.
"""

import psutil
import platform
import torch
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class HardwareDetector:
    """Automatically detect system hardware and select optimal configuration."""
    
    def __init__(self):
        self.system_info = self._detect_hardware()
        self.config = self._determine_optimal_config()
    
    def _detect_hardware(self) -> Dict:
        """Detect system hardware specifications."""
        info = {
            'os': platform.system(),
            'os_version': platform.release(),
            'cpu_cores': psutil.cpu_count() or 4,
            'ram_gb': psutil.virtual_memory().total / (1024**3),
            'gpu_available': torch.cuda.is_available(),
            'gpu_memory_gb': 0,
            'gpu_name': 'None'
        }
        
        # GPU detection
        if info['gpu_available']:
            try:
                info['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                info['gpu_name'] = torch.cuda.get_device_name(0)
            except Exception as e:
                logger.warning(f"GPU detection failed: {e}")
                info['gpu_available'] = False
        
        return info
    
    def _determine_optimal_config(self) -> Dict:
        """Determine optimal configuration based on hardware."""
        config = {
            'model_type': 'enhanced',
            'batch_size': 8,
            'num_workers': 2,
            'use_gpu': False,
            'use_mixed_precision': False,
            'gradient_accumulation': 1,
            'pin_memory': False,
            'performance_tier': 'standard'
        }
        
        ram_gb = self.system_info['ram_gb']
        cpu_cores = self.system_info['cpu_cores']
        gpu_available = self.system_info['gpu_available']
        gpu_memory_gb = self.system_info['gpu_memory_gb']
        
        # Tier 1: Low-spec systems (< 12GB RAM)
        if ram_gb < 12:
            config.update({
                'model_type': 'lightweight',
                'batch_size': 1,
                'num_workers': 1,
                'use_gpu': False,
                'use_mixed_precision': False,
                'gradient_accumulation': 4,
                'pin_memory': False,
                'performance_tier': 'low_spec'
            })
            
        # Tier 2: Medium-spec systems (12-16GB RAM)
        elif ram_gb < 16:
            config.update({
                'model_type': 'lightweight',
                'batch_size': 2,
                'num_workers': min(2, cpu_cores // 2),
                'use_gpu': gpu_available and gpu_memory_gb >= 4,
                'use_mixed_precision': gpu_available,
                'gradient_accumulation': 2,
                'pin_memory': gpu_available,
                'performance_tier': 'medium_spec'
            })
            
        # Tier 3: High-spec systems (16+ GB RAM)
        else:
            config.update({
                'model_type': 'enhanced',
                'batch_size': 4 if ram_gb < 24 else 8,
                'num_workers': min(4, cpu_cores // 2),
                'use_gpu': gpu_available,
                'use_mixed_precision': gpu_available and gpu_memory_gb < 8,
                'gradient_accumulation': 1,
                'pin_memory': gpu_available,
                'performance_tier': 'high_spec'
            })
        
        # GPU-specific adjustments
        if gpu_available:
            if gpu_memory_gb < 4:
                # Very limited GPU memory
                config['batch_size'] = min(config['batch_size'], 1)
                config['use_mixed_precision'] = True
            elif gpu_memory_gb < 6:
                # Limited GPU memory
                config['batch_size'] = min(config['batch_size'], 2)
                config['use_mixed_precision'] = True
            elif gpu_memory_gb >= 8:
                # Ample GPU memory
                config['batch_size'] = min(config['batch_size'] * 2, 16)
        
        return config
    
    def get_model_config(self) -> Dict:
        """Get configuration for model creation."""
        return {
            'model_type': self.config['model_type'],
            'input_shape': (3, 224, 224),
            'num_tissues': 5,
            'num_classes': 10
        }
    
    def get_training_config(self) -> Dict:
        """Get configuration for training/inference."""
        return {
            'batch_size': self.config['batch_size'],
            'num_workers': self.config['num_workers'],
            'pin_memory': self.config['pin_memory'],
            'use_mixed_precision': self.config['use_mixed_precision'],
            'gradient_accumulation': self.config['gradient_accumulation']
        }
    
    def get_device(self) -> torch.device:
        """Get optimal device for computation."""
        if self.config['use_gpu'] and torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')
    
    def print_system_info(self):
        """Print detected system information."""
        info = self.system_info
        config = self.config
        
        print("🔍 Hardware Detection Results:")
        print("=" * 40)
        print(f"OS: {info['os']} {info['os_version']}")
        print(f"CPU Cores: {info['cpu_cores']}")
        print(f"RAM: {info['ram_gb']:.1f} GB")
        
        if info['gpu_available']:
            print(f"GPU: {info['gpu_name']} ({info['gpu_memory_gb']:.1f} GB)")
        else:
            print("GPU: Not available")
        
        print(f"\n⚙️ Optimal Configuration:")
        print(f"Performance Tier: {config['performance_tier'].upper()}")
        print(f"Model Type: {config['model_type']}")
        print(f"Batch Size: {config['batch_size']}")
        print(f"Device: {'GPU' if config['use_gpu'] else 'CPU'}")
        print(f"Mixed Precision: {config['use_mixed_precision']}")
        
        # Performance estimates
        if config['performance_tier'] == 'low_spec':
            print(f"\n📊 Expected Performance:")
            print(f"Processing Time: 15-25 min per WSI")
            print(f"Memory Usage: 3-5 GB peak")
        elif config['performance_tier'] == 'medium_spec':
            print(f"\n📊 Expected Performance:")
            print(f"Processing Time: 8-15 min per WSI")
            print(f"Memory Usage: 6-10 GB peak")
        else:
            print(f"\n📊 Expected Performance:")
            print(f"Processing Time: 3-8 min per WSI")
            print(f"Memory Usage: 8-16 GB peak")
    
    def is_compatible(self) -> Tuple[bool, str]:
        """Check if system meets minimum requirements."""
        ram_gb = self.system_info['ram_gb']
        cpu_cores = self.system_info['cpu_cores']
        
        if ram_gb < 6:
            return False, f"Insufficient RAM: {ram_gb:.1f}GB (minimum 6GB required)"
        
        if cpu_cores < 2:
            return False, f"Insufficient CPU cores: {cpu_cores} (minimum 2 required)"
        
        return True, "System meets minimum requirements"


# Global hardware detector instance
_hardware_detector = None

def get_hardware_detector() -> HardwareDetector:
    """Get global hardware detector instance."""
    global _hardware_detector
    if _hardware_detector is None:
        _hardware_detector = HardwareDetector()
    return _hardware_detector


def auto_build_model():
    """Automatically build optimal model for current system."""
    detector = get_hardware_detector()
    config = detector.get_model_config()
    
    # Import here to avoid circular imports
    from .models.cnn import build_model
    
    model = build_model(**config)
    
    # Print selection info
    param_count = sum(p.numel() for p in model.parameters())
    print(f"🤖 Auto-selected {config['model_type']} model ({param_count:,} parameters)")
    
    return model


def auto_configure_processing():
    """Get automatic processing configuration."""
    detector = get_hardware_detector()
    return {
        'device': detector.get_device(),
        'training_config': detector.get_training_config(),
        'model_config': detector.get_model_config()
    }


if __name__ == "__main__":
    # Test hardware detection
    detector = HardwareDetector()
    detector.print_system_info()
    
    compatible, message = detector.is_compatible()
    print(f"\n✅ Compatibility: {message}")
