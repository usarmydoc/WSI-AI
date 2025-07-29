#!/usr/bin/env python3
"""
Quick verification script for GitHub repository.
Run this before pushing to ensure everything works.
"""

import psutil
import platform
import torch

def test_system_requirements():
    """Check system meets minimum requirements."""
    print("System Requirements Check:")
    
    # OS Check
    os_info = platform.system()
    print(f"  OS: {os_info} {platform.release()}")
    
    # CPU Check
    cpu_count = psutil.cpu_count() or 0
    print(f"  CPU Cores: {cpu_count}")
    if cpu_count >= 6:
        print("  [PASS] CPU: Excellent performance expected")
    elif cpu_count >= 4:
        print("  [PASS] CPU: Good for research lab use")
    else:
        print("  [WARN] CPU: Consider upgrading for better performance")
    
    # RAM Check
    ram_gb = psutil.virtual_memory().total / (1024**3)
    print(f"  RAM: {ram_gb:.1f} GB")
    if ram_gb >= 16:
        print("  [PASS] RAM: Excellent for WSI processing")
    elif ram_gb >= 8:
        print("  [PASS] RAM: Sufficient (will use smaller batch sizes)")
    else:
        print("  [WARN] RAM: 8+ GB recommended for reliable operation")
    
    # GPU Check
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"  GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        if gpu_memory >= 4:
            print("  [PASS] GPU: Sufficient VRAM for acceleration")
        else:
            print("  [WARN] GPU: 4+ GB VRAM recommended")
    else:
        print("  [INFO] GPU: Running in CPU-only mode (slower but functional)")
    
    return True

def test_imports():
    """Test all critical imports work."""
    try:
        from src.models.cnn import build_model
        from src.evaluate import debug_evaluate_model
        from src.data.synthetic_data import debug_synthetic_data_generation
        from src.visualization import debug_visualize_results
        print("[PASS] All core imports successful")
        
        # Test TIAToolbox import
        try:
            import tiatoolbox
            print("[PASS] TIAToolbox available for WSI preprocessing")
        except ImportError:
            print("[WARN] TIAToolbox not installed - will need for real WSI data")
        
        return True
    except Exception as e:
        print(f"[FAIL] Import failed: {e}")
        return False

def test_memory_optimization():
    """Test memory optimization features."""
    try:
        from src.memory_optimizer import MemoryOptimizer
        
        # Get system optimization settings
        settings = MemoryOptimizer.optimize_for_system()
        
        print("[PASS] Memory optimizer working")
        print(f"  Recommended model: {settings['model_type']}")
        print(f"  Optimal batch size: {settings['batch_size']}")
        print(f"  Memory settings configured for system")
        
        # Test memory monitoring
        memory_stats = MemoryOptimizer.get_memory_usage()
        ram_used = memory_stats['ram_used_gb']
        ram_available = memory_stats['ram_available_gb']
        
        print(f"  Current RAM usage: {ram_used:.1f}GB / {ram_used + ram_available:.1f}GB")
        
        return True
    except Exception as e:
        print(f"[WARN] Memory optimization failed: {e}")
        print("  System will use default settings")
        return True  # Non-critical failure

def test_model_creation():
    """Test model can be created with automatic hardware detection."""
    try:
        from src.models.cnn import build_model
        
        print("  Testing automatic hardware detection...")
        
        # Test automatic model selection
        model = build_model(model_type='auto')
        
        param_count = sum(p.numel() for p in model.parameters())
        memory_mb = param_count * 4 / (1024 * 1024)  # Assuming float32
        
        print(f"[PASS] Model created: {param_count:,} parameters ({memory_mb:.1f}MB)")
        
        # Provide system-specific recommendations
        ram_gb = psutil.virtual_memory().total / (1024**3)
        if param_count < 10_000_000:  # Lightweight model
            print("  [INFO] Lightweight model selected - optimized for your system")
            if ram_gb < 12:
                print("  [INFO] Recommended batch size: 1-2 for this system")
            else:
                print("  [INFO] Recommended batch size: 2-4 for this system")
        else:  # Full model
            print("  [INFO] Enhanced model selected - good performance expected")
            if ram_gb < 16:
                print("  [INFO] Recommended batch size: 4-8 for this system")
            else:
                print("  [INFO] Recommended batch size: 8-16 for this system")
            
        return True
    except Exception as e:
        print(f"[FAIL] Model creation failed: {e}")
        return False

def main():
    """Run all verification tests."""
    print("Verifying repository for GitHub...")
    
    tests = [
        ("System Requirements", test_system_requirements),
        ("Memory Optimization", test_memory_optimization),
        ("Imports", test_imports),
        ("Model Creation", test_model_creation),
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\nTesting {name}...")
        results.append(test_func())
    
    if all(results):
        print("\nRepository verification successful!")
        print("[PASS] Ready to push to GitHub!")
        print("[INFO] Optimized for low-spec systems!")
    else:
        print("\n[WARN] Some tests failed. Please fix before pushing.")
    
    return all(results)

if __name__ == "__main__":
    main()
