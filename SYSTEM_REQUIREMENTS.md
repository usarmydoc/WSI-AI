# System Requirements for WSI AI

## Minimum Requirements

### **Operating System**
- **Windows 10/11** (64-bit)
- **Linux** (Ubuntu 18.04+ recommended)
- **macOS** (10.15+ with Intel or Apple Silicon)

### **Hardware Requirements**

#### **CPU**
- **Minimum**: Intel i5-8th gen / AMD Ryzen 5 (4+ cores)
- **Recommended**: Intel i7-10th gen / AMD Ryzen 7 (6+ cores)
- **Dell OptiPlex Compatible**: 7070, 7080, 7090 series and newer

#### **RAM**
- **Minimum**: 8 GB
- **Recommended**: 16+ GB
- **Note**: Can run with 8GB but will process smaller batches

#### **Storage**
- **Minimum**: 50 GB free space
- **Recommended**: 250+ GB (SSD if available)
- **Note**: HDD acceptable for development, SSD preferred

#### **GPU (Optional but Recommended)**
- **Budget**: NVIDIA GTX 1050 Ti / RTX 3050 (4+ GB VRAM)
- **Recommended**: NVIDIA GTX 1660 / RTX 3060 (6+ GB VRAM)
- **Note**: Can run CPU-only mode on integrated graphics

## 🔧 Software Dependencies

### **Python Environment**
- **Python**: 3.8 - 3.12 (tested on 3.12)
- **PyTorch**: 1.12+ (with CUDA support for GPU)
- **TIAToolbox**: Latest version for WSI preprocessing

### **Key Libraries**
```
torch>=1.12.0
torchvision>=0.13.0
tiatoolbox
opencv-python
matplotlib
numpy
pillow
scikit-learn
```

## 🏥 Deployment Scenarios

### **Research Lab Desktop (Dell OptiPlex)**
- **Model**: OptiPlex 7070/7080/7090 or newer
- **CPU**: Intel i5-8th gen or newer
- **RAM**: 8-16 GB (upgrade to 16GB recommended)
- **GPU**: Add NVIDIA GTX 1650/RTX 3050 if budget allows
- **Storage**: 256GB SSD (external drive for datasets)
- **Use Case**: Development, small-scale analysis

### **Enhanced Research Workstation**
- **CPU**: Intel i7-10th gen or newer
- **RAM**: 16-32 GB
- **GPU**: NVIDIA GTX 1660/RTX 3060
- **Storage**: 512GB SSD + 1TB HDD
- **Use Case**: Regular WSI processing, model training

### **High-Performance Setup**
- **CPU**: Intel i9/AMD Ryzen 9
- **RAM**: 32+ GB
- **GPU**: RTX 3080+ (for heavy workloads)
- **Storage**: 1TB+ NVMe SSD
- **Use Case**: Large datasets, production deployment

## Performance Estimates

### **Model Inference** (per WSI)
- **CPU Only (i5-8th gen)**: 10-20 minutes
- **CPU + GPU (GTX 1650)**: 3-8 minutes  
- **CPU + GPU (RTX 3060)**: 1-3 minutes

### **TIAToolbox Preprocessing** (per WSI)
- **Standard Desktop**: 3-15 minutes (depends on WSI size)
- **With SSD**: 2-8 minutes
- **Quality Filtering**: 1-3 minutes

### **Memory Usage (Optimized for 8GB Systems)**
- **Model**: ~800MB-1.5GB RAM
- **Small Batch (4 patches)**: 2-4 GB RAM total
- **WSI Processing**: 1-3 GB RAM per file
- **Background OS**: ~2-3 GB RAM

### **Storage Requirements**
- **Program**: ~2-5 GB
- **Single WSI**: 100MB - 2GB each
- **Small Dataset**: 10-50 GB
- **Working Space**: 20-100 GB recommended

## 🔍 Tested Configurations

### **Development Environment (Current)**
- **OS**: Windows 11
- **CPU**: [Your current CPU]
- **RAM**: [Your current RAM]
- **Python**: 3.12
- **Status**: [VERIFIED] Verified working

### **Cloud Options**
- **Google Colab Pro**: T4/A100 GPU, good for development
- **AWS EC2**: p3.2xlarge or better for production
- **Azure**: NC6s_v3 or better

## Quick Compatibility Check

Run this command to check your system:
```bash
python verify_repo.py
```

The script will test:
- [CHECK] Python environment
- [CHECK] Core dependencies
- [CHECK] Model creation (tests GPU/CPU)
- [CHECK] Memory requirements

## Performance Tips for Lab Desktop Systems

### **For 8GB RAM Systems**
1. **Reduce batch size** to 2-4 patches at once
2. **Process WSI sequentially** rather than in parallel
3. **Use patch caching** to avoid reprocessing
4. **Close other applications** during processing

### **For Systems without Dedicated GPU**
1. **Enable CPU optimization** in PyTorch
2. **Use smaller patch sizes** (224x224 instead of 512x512)
3. **Process during off-hours** for better performance
4. **Consider cloud processing** for large datasets

### **Dell OptiPlex Optimization**
1. **Upgrade to 16GB RAM** if possible (~$50-100)
2. **Add SSD** for faster I/O
3. **Install GTX 1650/RTX 3050** for GPU acceleration
4. **Ensure adequate cooling** during long processing runs

### **Storage Strategy**
- **System Drive**: Keep OS and program on SSD
- **Data Drive**: External HDD/SSD for WSI datasets
- **Network Storage**: Use lab NAS for large datasets
- **Cloud Backup**: Store processed results in cloud
