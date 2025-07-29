"""
Simple interface for automatic WSI AI configuration.

This module provides easy-to-use functions for automatic hardware detection
and optimal model configuration.
"""

from .models.cnn import build_model
from .memory_optimizer import MemoryOptimizer
import torch
import psutil


class WSIAIConfig:
    """Simple configuration class for WSI AI system."""
    
    def __init__(self):
        """Initialize with automatic hardware detection."""
        self.ram_gb = psutil.virtual_memory().total / (1024**3)
        self.gpu_available = torch.cuda.is_available()
        self.cpu_cores = psutil.cpu_count() or 4
        
        if self.gpu_available:
            self.gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            self.gpu_name = torch.cuda.get_device_name(0)
        else:
            self.gpu_memory_gb = 0
            self.gpu_name = "None"
        
        # Auto-configure
        self._configure()
    
    def _configure(self):
        """Configure settings based on hardware."""
        # Model selection
        if self.ram_gb < 12:
            self.model_type = 'lightweight'
            self.batch_size = 1
            self.performance_level = 'basic'
        elif self.ram_gb < 20:
            if self.gpu_available and self.gpu_memory_gb >= 4:
                self.model_type = 'enhanced'
                self.batch_size = 4
                self.performance_level = 'good'
            else:
                self.model_type = 'lightweight'
                self.batch_size = 2
                self.performance_level = 'basic'
        else:
            self.model_type = 'enhanced'
            self.batch_size = 8
            self.performance_level = 'excellent'
        
        # Device selection
        self.device = 'cuda' if self.gpu_available else 'cpu'
        
        # Other settings
        self.num_workers = min(4, self.cpu_cores // 2)
        self.use_mixed_precision = self.gpu_available and self.gpu_memory_gb < 8
    
    def create_model(self):
        """Create optimally configured model."""
        return build_model(model_type=self.model_type)
    
    def get_device(self):
        """Get optimal torch device."""
        return torch.device(self.device)
    
    def print_config(self):
        """Print current configuration."""
        print("WSI AI Configuration:")
        print(f"  System: {self.ram_gb:.1f}GB RAM, {self.cpu_cores} cores")
        if self.gpu_available:
            print(f"  GPU: {self.gpu_name} ({self.gpu_memory_gb:.1f}GB)")
        else:
            print(f"  GPU: Not available")
        print(f"  Model: {self.model_type}")
        print(f"  Batch Size: {self.batch_size}")
        print(f"  Device: {self.device}")
        print(f"  Performance: {self.performance_level}")


def auto_configure():
    """Create automatic configuration for current system."""
    return WSIAIConfig()


def quick_setup():
    """Quick setup function that returns model and config."""
    config = auto_configure()
    model = config.create_model()
    device = config.get_device()
    
    print("[INFO] Quick setup complete!")
    config.print_config()
    
    return model, device, config


# Convenience functions
def get_optimal_model():
    """Get optimally configured model for current system."""
    return build_model(model_type='auto')


def get_optimal_batch_size():
    """Get optimal batch size for current system."""
    try:
        optimizer = MemoryOptimizer()
        settings = optimizer.optimize_for_system()
        return settings['batch_size']
    except:
        # Fallback
        ram_gb = psutil.virtual_memory().total / (1024**3)
        if ram_gb < 12:
            return 1
        elif ram_gb < 16:
            return 2
        else:
            return 4


def get_system_info():
    """Get current system information."""
    ram_gb = psutil.virtual_memory().total / (1024**3)
    gpu_available = torch.cuda.is_available()
    
    info = {
        'ram_gb': ram_gb,
        'cpu_cores': psutil.cpu_count() or 4,
        'gpu_available': gpu_available,
        'recommended_model': 'lightweight' if ram_gb < 16 else 'enhanced',
        'recommended_batch_size': get_optimal_batch_size()
    }
    
    if gpu_available:
        info['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        info['gpu_name'] = torch.cuda.get_device_name(0)
    
    return info


if __name__ == "__main__":
    # Demo the configuration
    config = auto_configure()
    config.print_config()
    
    print("\nQuick setup demo:")
    model, device, config = quick_setup()
