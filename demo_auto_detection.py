#!/usr/bin/env python3
"""
Example script demonstrating automatic hardware detection and model selection.

This script shows how the WSI AI system automatically adapts to your hardware.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

def main():
    """Demonstrate automatic hardware detection."""
    print("🔬 WSI AI - Automatic Hardware Detection Demo")
    print("=" * 50)
    
    # Basic hardware detection
    try:
        from src.models.cnn import build_model
        
        print("🔍 Detecting your system hardware...")
        
        # This will automatically detect hardware and select optimal model
        model = build_model(model_type='auto')
        
        param_count = sum(p.numel() for p in model.parameters())
        memory_mb = param_count * 4 / (1024 * 1024)
        
        print(f"\n🤖 Selected Model:")
        print(f"  Parameters: {param_count:,}")
        print(f"  Memory: {memory_mb:.1f} MB")
        
        # Show what this means for performance
        if param_count < 10_000_000:
            print(f"\n⚡ Performance Profile: OPTIMIZED")
            print(f"  • Lightweight model selected")
            print(f"  • Lower memory usage")
            print(f"  • Good performance on standard lab computers")
            print(f"  • Compatible with Dell OptiPlex systems")
        else:
            print(f"\n⚡ Performance Profile: HIGH-PERFORMANCE")
            print(f"  • Enhanced model selected")
            print(f"  • Maximum accuracy")
            print(f"  • Suitable for high-spec workstations")
        
    except ImportError as e:
        print(f"❌ Could not import model: {e}")
        return
    
    # Advanced hardware detection (if available)
    try:
        from src.hardware_detector import HardwareDetector
        
        print(f"\n🔧 Detailed Hardware Analysis:")
        detector = HardwareDetector()
        detector.print_system_info()
        
        # Check compatibility
        compatible, message = detector.is_compatible()
        if compatible:
            print(f"\n✅ System Status: {message}")
        else:
            print(f"\n❌ System Status: {message}")
            
    except ImportError:
        print(f"\n⚠️ Advanced hardware detection not available")
        print(f"  Basic auto-detection is still working")
    
    # Memory optimization demo
    try:
        from src.memory_optimizer import MemoryOptimizer
        
        print(f"\n💾 Memory Optimization:")
        settings = MemoryOptimizer.optimize_for_system()
        
        print(f"  Recommended batch size: {settings['batch_size']}")
        print(f"  Number of workers: {settings['num_workers']}")
        print(f"  Mixed precision: {settings['use_mixed_precision']}")
        print(f"  Performance tier: {settings['performance_tier']}")
        
    except ImportError:
        print(f"\n⚠️ Memory optimizer not available")
    
    print(f"\n🎯 Summary:")
    print(f"  • System automatically detected and configured")
    print(f"  • No manual configuration required")
    print(f"  • Optimized for your specific hardware")
    print(f"  • Ready for WSI processing!")


if __name__ == "__main__":
    main()
