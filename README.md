# Medical-Grade AI for Multi-Tissue WSI Damage Scoring

A comprehensive deep learning system for scoring tissue damage (0-9) in Whole Slide Images (WSIs) across lung, kidney, heart, liver, and bowel tissues.

## 🏥 Medical Research Grade Features

### Core Capabilities
- **Multi-tissue damage scoring**: 0-9 scale for lung, kidney, heart, liver, and bowel
- **Uncertainty quantification**: MC Dropout for prediction confidence
- **Explainable AI**: Grad-CAM heatmaps for clinical interpretation
- **Medical-grade validation**: Clinical metrics including Cohen's Kappa
- **Comprehensive evaluation**: Per-tissue and severity-based analysis

### Technical Architecture
- **Advanced CNN**: ResNet-inspired with attention mechanisms
- **Stain normalization**: Macenko and Vahadane methods
- **Quality filtering**: Automatic tissue/background separation
- **Cross-validation**: K-fold validation for robust evaluation
- **Early stopping**: Prevents overfitting with patience monitoring

## 📁 Project Structure

```
lung AI/
├── src/
│   ├── models/
│   │   └── cnn.py                  # Enhanced CNN architectures
│   ├── data/
│   │   ├── dataset.py              # PyTorch dataset classes
│   │   ├── preprocess.py           # WSI preprocessing utilities
│   │   └── tia_utils.py            # TIAToolbox integration
│   ├── tests/
│   │   └── test_model.py           # Unit tests
│   ├── config.yaml                 # Experiment configuration
│   ├── train.py                    # Medical-grade training pipeline
│   ├── evaluate.py                 # Comprehensive evaluation
│   ├── inference.py                # Batch inference pipeline
│   ├── visualization.py            # Clinical visualization tools
│   └── utils.py                    # Utility functions
├── model_evaluation_debug.ipynb    # Testing and development notebook
├── requirements.txt                # Enhanced dependencies
└── README.md                       # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+ (tested with 3.12)
- CUDA-capable GPU (recommended)
- Git for version control

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/medical-wsi-ai.git
cd medical-wsi-ai

# Install dependencies
pip install -r requirements.txt

# For GPU support (optional but recommended)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Configuration

Edit `src/config.yaml` to customize:

```yaml
model:
  input_shape: [3, 224, 224]
  num_tissues: 5
  num_classes: 10
  
training:
  epochs: 50
  batch_size: 32
  learning_rate: 0.001
  cross_validation: true
  k_folds: 5

preprocessing:
  stain_normalization: true
  normalizer_method: "Macenko"
```

### Training

```python
from src.train import main
model, history = main()
```

### Inference

```python
from src.inference import run_inference

slides_info = [
    {"wsi_path": "path/to/slide.svs", "tissue": "lung"},
    {"wsi_path": "path/to/slide2.svs", "tissue": "kidney"}
]

results = run_inference("best_model.pth", slides_info, "src/config.yaml")
```

### Evaluation

```python
from src.evaluate import evaluate_model, generate_clinical_report

results = evaluate_model(model, test_data, test_labels, tissue_labels)
report = generate_clinical_report(results)
print(report)
```

## 📊 Model Architecture

### MedicalGradeCNN Features

1. **Residual Connections**: Better gradient flow for deeper networks
2. **Attention Mechanisms**: Focus on relevant tissue regions
3. **Multi-scale Features**: Capture details at different resolutions
4. **Tissue-aware Processing**: Embedding tissue type information
5. **Uncertainty Estimation**: MC Dropout for confidence intervals

### Input/Output Specification

- **Input**: 
  - Images: `(batch_size, 3, 224, 224)` RGB patches
  - Tissue: `(batch_size, 5)` one-hot encoded tissue type
- **Output**: 
  - Logits: `(batch_size, 10)` damage scores 0-9
  - Uncertainty: `(batch_size, 1)` confidence measure (optional)

## 🔬 Medical Validation

### Clinical Metrics

- **Accuracy**: Overall prediction accuracy
- **Cohen's Kappa**: Inter-rater agreement equivalent
- **Mean Absolute Error**: Average prediction error
- **Sensitivity/Specificity**: Per-severity analysis
- **Uncertainty Calibration**: Confidence vs. accuracy correlation

### Quality Assurance

- **Patch Filtering**: Remove background and artifacts
- **Stain Normalization**: Consistent color representation
- **Cross-validation**: Robust performance estimation
- **Error Analysis**: Clinical significance assessment

## 📈 Evaluation and Visualization

### Comprehensive Metrics

```python
from src.evaluate import plot_evaluation_results
from src.visualization import create_clinical_dashboard

# Generate evaluation plots
plot_evaluation_results(results, save_path="evaluation.png")

# Create clinical dashboard
create_clinical_dashboard(results, save_path="dashboard.png")
```

### Explainable AI

```python
from src.visualization import GradCAM

# Generate attention maps
grad_cam = GradCAM(model, target_layer_name='conv3')
cam = grad_cam.generate_cam(image_tensor, tissue_onehot)
overlayed = grad_cam.visualize_cam(original_image, cam)
```

## 🧪 Testing and Development

### Jupyter Notebook

The `model_evaluation_debug.ipynb` provides:

1. **Synthetic Data Generation**: Create test WSIs
2. **Pipeline Testing**: End-to-end validation
3. **Visualization Examples**: Result interpretation
4. **Performance Analysis**: Comprehensive metrics

### Unit Tests

```bash
# Run all tests
pytest src/tests/

# Run specific test
python src/tests/test_model.py
```

## 🏗️ Research Workflow

### 1. Literature Review
- Search PubMed, Google Scholar for WSI damage assessment
- Focus on non-cancerous tissue damage methodologies
- Document findings in notebook

### 2. Data Preparation
- **Real Data**: Acquire annotated WSI datasets
- **Synthetic Data**: Use provided generator for testing
- **Preprocessing**: Apply stain normalization and quality filtering

### 3. Model Development
- **Architecture**: Choose between enhanced or medical-grade CNN
- **Training**: Use cross-validation for robust evaluation
- **Validation**: Apply clinical metrics and uncertainty quantification

### 4. Clinical Validation
- **Performance**: Achieve >85% accuracy and >0.6 Cohen's Kappa
- **Safety**: Minimize large errors (>3 damage score difference)
- **Interpretability**: Provide Grad-CAM explanations

### 5. Publication Preparation
- **Reproducibility**: Document all hyperparameters and methods
- **Validation**: Multi-site testing and expert review
- **Ethics**: Ensure patient privacy and clinical safety

## 📋 Clinical Recommendations

### Performance Thresholds

- **Minimum Accuracy**: 85% for clinical deployment
- **Cohen's Kappa**: ≥0.6 for acceptable agreement
- **Large Error Rate**: ≤5% for clinical safety
- **Uncertainty Calibration**: Well-calibrated confidence intervals

### Tissue-Specific Considerations

- **Lung**: Often well-defined structures, good for training
- **Kidney**: Complex architecture, may need specialized preprocessing
- **Heart**: Muscle fiber patterns, attention mechanisms helpful
- **Liver**: Uniform appearance, stain normalization critical
- **Bowel**: Variable morphology, extensive validation needed

## 🔧 Configuration Options

### Model Selection

```python
# Backward compatible model
model = build_model(model_type='enhanced')

# Medical-grade model with attention
model = build_model(model_type='medical_grade')
```

### Training Modes

```python
# Standard training
config['training']['cross_validation'] = False

# Cross-validation mode
config['training']['cross_validation'] = True
config['training']['k_folds'] = 5
```

### Preprocessing Options

```python
# Stain normalization methods
config['preprocessing']['normalizer_method'] = 'Macenko'  # or 'Vahadane'

# Quality filtering
config['preprocessing']['min_tissue_ratio'] = 0.7
```

## 🚨 Important Notes

### Clinical Use Disclaimer

This AI system is designed for research purposes. Clinical deployment requires:

1. **Regulatory Approval**: FDA/CE marking for medical devices
2. **Clinical Validation**: Multi-site prospective studies
3. **Expert Review**: Pathologist validation of results
4. **Quality Assurance**: Ongoing monitoring and calibration

### Data Requirements

- **Minimum Dataset**: 10,000+ annotated patches per tissue type
- **Quality Standards**: Expert pathologist annotations
- **Diversity**: Multiple institutions, patient demographics
- **Privacy**: De-identified data following HIPAA guidelines

### Performance Monitoring

- **Calibration**: Regular uncertainty vs. accuracy assessment
- **Drift Detection**: Monitor performance degradation over time
- **Error Analysis**: Investigate large prediction errors
- **Feedback Loop**: Incorporate expert corrections

## 📖 References and Resources

### Key Publications
- TIAToolbox: Comprehensive toolbox for computational pathology
- Macenko et al.: Stain normalization methods for histopathology
- Grad-CAM: Visual explanations for deep networks

### Datasets
- TCGA: Cancer genome atlas (for transfer learning)
- PathAI: Commercial pathology datasets
- Kaggle Competitions: Histopathology challenges

### Tools and Libraries
- **TIAToolbox**: WSI processing and analysis
- **OpenSlide**: WSI file format support
- **HistomicsTK**: Image analysis algorithms
- **PyTorch**: Deep learning framework

## 📞 Support and Contributing

### Issues and Questions
- Create GitHub issues for bugs or feature requests
- Use discussion forums for methodology questions
- Contact medical AI experts for clinical validation

### Contributing
1. Fork the repository
2. Create feature branch
3. Add comprehensive tests
4. Update documentation
5. Submit pull request

### Citation

If you use this code in your research, please cite:

```bibtex
@software{medical_wsi_ai,
  title={Medical-Grade AI for Multi-Tissue WSI Damage Scoring},
  author={[Your Name]},
  year={2025},
  url={https://github.com/your-repo/medical-wsi-ai}
}
```

---

**⚠️ Remember**: This is research software. Always validate results with medical experts before any clinical application.
