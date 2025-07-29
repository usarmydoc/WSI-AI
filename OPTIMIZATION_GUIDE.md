# Optimized Usage Guide for Low-Spec Systems

## Quick Start for Dell OptiPlex & Lab Desktops

### **Automatic Optimization**
The system automatically detects your hardware and adjusts settings:

```python
from src.models.cnn import build_model
from src.memory_optimizer import MemoryOptimizer

# Auto-select model based on system specs
model = build_model(low_memory=True)  # Automatically chooses lightweight for <16GB systems

# Get optimized settings
settings = MemoryOptimizer.optimize_for_system()
print(f"Recommended batch size: {settings['batch_size']}")
```

### **Memory-Optimized Processing**
```python
from src.memory_optimizer import process_wsi_efficiently

# Process WSI with automatic memory management
predictions = process_wsi_efficiently(
    wsi_patches=patches,
    model=model,
    device=device
)
```

## Performance by System Type

### **8GB RAM System (Dell OptiPlex 7070)**
- **Model**: Lightweight CNN (~5M parameters)
- **Batch Size**: 1-2 patches
- **Processing Time**: 15-25 min per WSI
- **Memory Usage**: ~3-4GB peak

### **16GB RAM System (Dell OptiPlex 7080+)**
- **Model**: Lightweight CNN (~5M parameters)  
- **Batch Size**: 4-8 patches
- **Processing Time**: 8-15 min per WSI
- **Memory Usage**: ~6-8GB peak

### **16GB RAM + GPU (RTX 3050)**
- **Model**: Standard CNN (~51M parameters)
- **Batch Size**: 4-8 patches
- **Processing Time**: 3-8 min per WSI
- **Memory Usage**: ~8-12GB peak

## Manual Configuration

### **For Very Limited Systems**
```python
# Force smallest configuration
model = build_model(model_type='lightweight')
settings = {
    'batch_size': 1,
    'num_workers': 1,
    'pin_memory': False
}
```

### **For Systems with Some GPU**
```python
# Use mixed precision for efficiency
settings = {
    'batch_size': 4,
    'use_mixed_precision': True,
    'gradient_accumulation': 2
}
```

## Monitoring and Optimization

### **Check Memory Usage**
```python
from src.memory_optimizer import MemoryOptimizer

# Monitor memory during processing
stats = MemoryOptimizer.get_memory_usage()
print(f"RAM: {stats['ram_used_gb']:.1f}GB used")

# Clear memory if needed
MemoryOptimizer.clear_memory()
```

### **Batch Size Optimization**
```python
# Calculate optimal batch size for your model
optimal_batch = MemoryOptimizer.get_optimal_batch_size(
    model=model,
    input_shape=(3, 224, 224),
    target_memory_gb=4.0  # Adjust based on available RAM
)
```

## Tips for Best Performance

### **Hardware Optimization**
1. **Upgrade RAM**: 8GB → 16GB provides significant improvement
2. **Add SSD**: Faster I/O for WSI files
3. **Add GPU**: Even GTX 1650 provides 3-5x speedup
4. **Close other apps**: Free up memory during processing

### **Software Optimization**
1. **Process smaller batches**: Reduces memory usage
2. **Use CPU threads**: Set `num_workers` to CPU cores / 2
3. **Enable mixed precision**: Saves GPU memory
4. **Clear memory regularly**: Prevents memory leaks

### **Dataset Management**
1. **External storage**: Use USB/network drives for data
2. **Process sequentially**: One WSI at a time for small systems
3. **Cache results**: Save processed patches to avoid recomputation
4. **Compress data**: Use efficient formats

## Troubleshooting

### **Out of Memory Errors**
```python
# Reduce batch size
settings['batch_size'] = 1

# Use gradient accumulation instead
settings['gradient_accumulation'] = 4

# Clear memory more frequently
MemoryOptimizer.clear_memory()
```

### **Slow Processing**
```python
# Check if GPU is being used
print(f"CUDA available: {torch.cuda.is_available()}")

# Use appropriate number of workers
settings['num_workers'] = min(2, psutil.cpu_count() // 2)

# Enable optimizations
settings['use_mixed_precision'] = True
```

### **System Freezing**
- Reduce batch size to 1
- Close other applications
- Add swap space if available
- Consider cloud processing for large datasets

## Getting Help

If you encounter issues:
1. Run `python verify_repo.py` to check system compatibility
2. Check the memory optimization recommendations
3. Consider upgrading hardware if budget allows
4. Use cloud computing for large-scale processing
