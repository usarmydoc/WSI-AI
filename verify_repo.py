#!/usr/bin/env python3
"""
Quick verification script for GitHub repository.
Run this before pushing to ensure everything works.
"""

def test_imports():
    """Test all critical imports work."""
    try:
        from src.models.cnn import build_model
        from src.evaluate import debug_evaluate_model
        from src.data.synthetic_data import debug_synthetic_data_generation
        from src.visualization import debug_visualize_results
        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_model_creation():
    """Test model can be created."""
    try:
        from src.models.cnn import build_model
        model = build_model(input_shape=(3, 224, 224), num_tissues=5, num_classes=10)
        param_count = sum(p.numel() for p in model.parameters())
        print(f"✅ Model created: {param_count:,} parameters")
        return True
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False

def main():
    """Run all verification tests."""
    print("🧪 Verifying repository for GitHub...")
    
    tests = [
        ("Imports", test_imports),
        ("Model Creation", test_model_creation),
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\n📋 Testing {name}...")
        results.append(test_func())
    
    if all(results):
        print("\n🎉 Repository verification successful!")
        print("✅ Ready to push to GitHub!")
    else:
        print("\n⚠️ Some tests failed. Please fix before pushing.")
    
    return all(results)

if __name__ == "__main__":
    main()
