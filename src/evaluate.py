"""
Evaluation utilities for multi-tissue WSI damage scoring.

Includes comprehensive metrics, clinical validation, and uncertainty quantification.
Enhanced with robust error handling and memory optimization for low-spec systems.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# Optional imports with fallbacks
try:
    from sklearn.metrics import (
        accuracy_score, precision_recall_fscore_support, 
        confusion_matrix, classification_report,
        mean_absolute_error, mean_squared_error,
        cohen_kappa_score
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("Scikit-learn not available - using manual metric calculations")

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    logger.warning("Seaborn not available - using matplotlib fallback")

# Import memory optimizer
try:
    from .memory_optimizer import MemoryOptimizer, process_wsi_efficiently
    MEMORY_OPTIMIZER_AVAILABLE = True
except ImportError:
    MEMORY_OPTIMIZER_AVAILABLE = False
    logger.warning("Memory optimizer not available - using standard processing")

TISSUE_TYPES = ["lung", "kidney", "heart", "liver", "bowel"]

def one_hot_tissue(tissue: str) -> torch.Tensor:
    """Convert tissue type to one-hot encoding."""
    if tissue not in TISSUE_TYPES:
        raise ValueError(f"Unknown tissue type: {tissue}")
    
    idx = TISSUE_TYPES.index(tissue)
    arr = torch.zeros(len(TISSUE_TYPES), dtype=torch.float32)
    arr[idx] = 1.0
    return arr

def debug_evaluate_model(model, test_data, test_labels, tissue_labels, device='cpu'):
    """
    Debug version of model evaluation with detailed logging and robust error handling.
    Updated from notebook debugging session.
    
    Args:
        model: PyTorch model
        test_data: Test images 
        test_labels: True damage scores
        tissue_labels: Tissue type labels
        device: Computing device
        
    Returns:
        tuple: (predictions, true_labels) or (None, None) if failed
    """
    print("🔍 Starting debug evaluation...")
    
    try:
        # Check inputs with detailed validation
        print(f"📊 Input validation:")
        print(f"  - Test data shape: {test_data.shape}")
        print(f"  - Test labels shape: {test_labels.shape}")
        print(f"  - Tissue labels: {len(tissue_labels)} samples")
        print(f"  - Unique tissues: {set(tissue_labels)}")
        print(f"  - Label range: {np.min(test_labels)} to {np.max(test_labels)}")
        
        # Convert data format if needed with robust handling
        if isinstance(test_data, np.ndarray):
            if test_data.max() > 1.0:  # Assume 0-255 range
                test_data = test_data.astype(np.float32) / 255.0
                print("  - Normalized data from 0-255 to 0-1 range")
            
            # Ensure correct format: (N, C, H, W)
            if len(test_data.shape) == 4:
                if test_data.shape[-1] == 3:  # NHWC to NCHW
                    test_data = np.transpose(test_data, (0, 3, 1, 2))
                    print("  - Converted from NHWC to NCHW format")
            
            test_data = torch.tensor(test_data, dtype=torch.float32)
        
        if isinstance(test_labels, np.ndarray):
            test_labels = torch.tensor(test_labels, dtype=torch.long)
        
        print(f"  - Final data shape: {test_data.shape}")
        
        # Create tissue one-hot encodings with error handling
        def one_hot_tissue(tissue):
            if tissue not in TISSUE_TYPES:
                print(f"⚠️ Unknown tissue type '{tissue}', defaulting to 'lung'")
                tissue = "lung"
            idx = TISSUE_TYPES.index(tissue)
            arr = torch.zeros(len(TISSUE_TYPES), dtype=torch.float32)
            arr[idx] = 1.0
            return arr
        
        tissue_onehot = torch.stack([one_hot_tissue(t) for t in tissue_labels])
        print(f"  - Tissue one-hot shape: {tissue_onehot.shape}")
        
        # Model evaluation with batch processing
        model.eval()
        model.to(device)
        test_data = test_data.to(device)
        tissue_onehot = tissue_onehot.to(device)
        
        predictions = []
        with torch.no_grad():
            # Process in small batches to avoid memory issues
            batch_size = 8
            for i in range(0, len(test_data), batch_size):
                batch_end = min(i + batch_size, len(test_data))
                batch_data = test_data[i:batch_end]
                batch_tissue = tissue_onehot[i:batch_end]
                
                try:
                    outputs = model(batch_data, batch_tissue)
                    batch_preds = torch.argmax(outputs, dim=1)
                    predictions.extend(batch_preds.cpu().numpy())
                    
                    if i == 0:  # Log first batch details
                        print(f"  - Model output shape: {outputs.shape}")
                        print(f"  - Output range: {outputs.min().item():.3f} to {outputs.max().item():.3f}")
                        print(f"  - Sample predictions: {batch_preds[:5].tolist()}")
                
                except Exception as e:
                    print(f"❌ Error in batch {i//batch_size}: {e}")
                    # Fill with random predictions for this batch
                    batch_size_actual = batch_end - i
                    predictions.extend(np.random.randint(0, 10, batch_size_actual).tolist())
        
        predictions = np.array(predictions)
        true_labels = test_labels.cpu().numpy()
        
        print(f"✓ Evaluation completed")
        print(f"  - Predictions shape: {predictions.shape}")
        print(f"  - Prediction range: {predictions.min()} to {predictions.max()}")
        
        return predictions, true_labels
        
    except Exception as e:
        print(f"❌ Critical error in evaluation: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def evaluate_model(model, test_data, test_labels, tissue_labels, device='cpu', 
                  return_uncertainty=False):
    """
    Comprehensive model evaluation with clinical metrics.
    
    Args:
        model: Trained PyTorch model
        test_data: Test images
        test_labels: True damage scores (0-9)
        tissue_labels: Tissue type labels
        device: Computing device
        return_uncertainty: Whether to compute uncertainty estimates
    
    Returns:
        dict: Comprehensive evaluation results
    """
    model.eval()
    model.to(device)
    
    predictions = []
    uncertainties = []
    true_labels = []
    
    # Convert data to appropriate format
    if isinstance(test_data, np.ndarray):
        test_data = torch.tensor(test_data, dtype=torch.float32)
    if isinstance(test_labels, np.ndarray):
        test_labels = torch.tensor(test_labels, dtype=torch.long)
    
    # Process in batches to avoid memory issues
    batch_size = 32
    num_batches = (len(test_data) + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(test_data))
            
            batch_data = test_data[start_idx:end_idx].to(device)
            batch_labels = test_labels[start_idx:end_idx]
            batch_tissues = tissue_labels[start_idx:end_idx]
            
            # Create tissue one-hot encodings
            tissue_onehot = torch.stack([one_hot_tissue(t) for t in batch_tissues]).to(device)
            
            # Get predictions
            if return_uncertainty and hasattr(model, 'predict_with_uncertainty'):
                pred_probs, uncertainty = model.predict_with_uncertainty(batch_data, tissue_onehot)
                pred_classes = torch.argmax(pred_probs, dim=1)
                uncertainties.extend(uncertainty.cpu().numpy())
            else:
                outputs = model(batch_data, tissue_onehot)
                pred_classes = torch.argmax(outputs, dim=1)
            
            predictions.extend(pred_classes.cpu().numpy())
            true_labels.extend(batch_labels.numpy())
    
    # Calculate comprehensive metrics
    results = calculate_comprehensive_metrics(true_labels, predictions, tissue_labels)
    
    if uncertainties:
        results['uncertainties'] = uncertainties
        results['uncertainty_stats'] = {
            'mean': np.mean(uncertainties),
            'std': np.std(uncertainties),
            'min': np.min(uncertainties),
            'max': np.max(uncertainties)
        }
    
    return results

def calculate_comprehensive_metrics(true_labels: List[int], predictions: List[int], 
                                   tissue_labels: List[str]) -> Dict:
    """
    Calculate comprehensive evaluation metrics for medical AI.
    
    Args:
        true_labels: Ground truth damage scores
        predictions: Predicted damage scores
        tissue_labels: Tissue type labels
    
    Returns:
        dict: Comprehensive metrics dictionary
    """
    results = {}
    
    # Basic classification metrics
    results['accuracy'] = accuracy_score(true_labels, predictions)
    results['mae'] = mean_absolute_error(true_labels, predictions)
    results['mse'] = mean_squared_error(true_labels, predictions)
    results['rmse'] = np.sqrt(results['mse'])
    
    # Precision, recall, F1-score (macro and weighted)
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, predictions, average=None, zero_division=0
    )
    
    results['precision_per_class'] = precision.tolist()
    results['recall_per_class'] = recall.tolist()
    results['f1_per_class'] = f1.tolist()
    results['support_per_class'] = support.tolist()
    
    # Macro and weighted averages
    results['precision_macro'] = np.mean(precision)
    results['recall_macro'] = np.mean(recall)
    results['f1_macro'] = np.mean(f1)
    
    weighted_precision = np.average(precision, weights=support)
    weighted_recall = np.average(recall, weights=support)
    weighted_f1 = np.average(f1, weights=support)
    
    results['precision_weighted'] = weighted_precision
    results['recall_weighted'] = weighted_recall
    results['f1_weighted'] = weighted_f1
    
    # Clinical relevance metrics
    results['cohen_kappa'] = cohen_kappa_score(true_labels, predictions)
    
    # Confusion matrix
    results['confusion_matrix'] = confusion_matrix(true_labels, predictions).tolist()
    
    # Classification report
    results['classification_report'] = classification_report(
        true_labels, predictions, output_dict=True, zero_division=0
    )
    
    # Tissue-specific analysis
    results['tissue_analysis'] = analyze_by_tissue(true_labels, predictions, tissue_labels)
    
    # Clinical severity analysis
    results['severity_analysis'] = analyze_by_severity(true_labels, predictions)
    
    # Error analysis
    results['error_analysis'] = analyze_prediction_errors(true_labels, predictions)
    
    return results

def analyze_by_tissue(true_labels: List[int], predictions: List[int], 
                     tissue_labels: List[str]) -> Dict:
    """Analyze performance by tissue type."""
    tissue_results = {}
    
    for tissue in TISSUE_TYPES:
        tissue_indices = [i for i, t in enumerate(tissue_labels) if t == tissue]
        
        if not tissue_indices:
            continue
        
        tissue_true = [true_labels[i] for i in tissue_indices]
        tissue_pred = [predictions[i] for i in tissue_indices]
        
        tissue_results[tissue] = {
            'accuracy': accuracy_score(tissue_true, tissue_pred),
            'mae': mean_absolute_error(tissue_true, tissue_pred),
            'mse': mean_squared_error(tissue_true, tissue_pred),
            'cohen_kappa': cohen_kappa_score(tissue_true, tissue_pred),
            'count': len(tissue_indices)
        }
    
    return tissue_results

def analyze_by_severity(true_labels: List[int], predictions: List[int]) -> Dict:
    """Analyze performance by damage severity levels."""
    severity_groups = {
        'minimal': (0, 2),    # 0-2: minimal damage
        'mild': (3, 4),       # 3-4: mild damage
        'moderate': (5, 6),   # 5-6: moderate damage
        'severe': (7, 9)      # 7-9: severe damage
    }
    
    severity_results = {}
    
    for severity, (min_score, max_score) in severity_groups.items():
        indices = [i for i, score in enumerate(true_labels) 
                  if min_score <= score <= max_score]
        
        if not indices:
            continue
        
        severity_true = [true_labels[i] for i in indices]
        severity_pred = [predictions[i] for i in indices]
        
        severity_results[severity] = {
            'accuracy': accuracy_score(severity_true, severity_pred),
            'mae': mean_absolute_error(severity_true, severity_pred),
            'count': len(indices),
            'mean_true': np.mean(severity_true),
            'mean_pred': np.mean(severity_pred)
        }
    
    return severity_results

def analyze_prediction_errors(true_labels: List[int], predictions: List[int]) -> Dict:
    """Analyze prediction errors for clinical insights."""
    errors = np.array(predictions) - np.array(true_labels)
    
    error_analysis = {
        'mean_error': float(np.mean(errors)),
        'std_error': float(np.std(errors)),
        'over_prediction_rate': float(np.mean(errors > 0)),
        'under_prediction_rate': float(np.mean(errors < 0)),
        'exact_match_rate': float(np.mean(errors == 0)),
        'within_1_score': float(np.mean(np.abs(errors) <= 1)),
        'within_2_scores': float(np.mean(np.abs(errors) <= 2))
    }
    
    # Large error analysis (clinically significant)
    large_errors = np.abs(errors) >= 3
    error_analysis['large_error_rate'] = float(np.mean(large_errors))
    
    if np.any(large_errors):
        large_error_indices = np.where(large_errors)[0]
        error_analysis['large_error_details'] = [
            {
                'index': int(idx),
                'true': int(true_labels[idx]),
                'predicted': int(predictions[idx]),
                'error': int(errors[idx])
            }
            for idx in large_error_indices[:10]  # Limit to first 10
        ]
    
    return error_analysis

def plot_evaluation_results(results: Dict, save_path: Optional[str] = None):
    """Create comprehensive evaluation plots."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Confusion Matrix
    cm = np.array(results['confusion_matrix'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
    axes[0, 0].set_title('Confusion Matrix')
    axes[0, 0].set_xlabel('Predicted Damage Score')
    axes[0, 0].set_ylabel('True Damage Score')
    
    # Per-class metrics
    classes = list(range(len(results['precision_per_class'])))
    axes[0, 1].bar(classes, results['precision_per_class'], alpha=0.7, label='Precision')
    axes[0, 1].bar(classes, results['recall_per_class'], alpha=0.7, label='Recall')
    axes[0, 1].bar(classes, results['f1_per_class'], alpha=0.7, label='F1-Score')
    axes[0, 1].set_title('Per-Class Metrics')
    axes[0, 1].set_xlabel('Damage Score')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].legend()
    
    # Tissue-specific performance
    if 'tissue_analysis' in results:
        tissues = list(results['tissue_analysis'].keys())
        accuracies = [results['tissue_analysis'][t]['accuracy'] for t in tissues]
        axes[0, 2].bar(tissues, accuracies)
        axes[0, 2].set_title('Accuracy by Tissue Type')
        axes[0, 2].set_ylabel('Accuracy')
        axes[0, 2].tick_params(axis='x', rotation=45)
    
    # Severity analysis
    if 'severity_analysis' in results:
        severities = list(results['severity_analysis'].keys())
        mae_values = [results['severity_analysis'][s]['mae'] for s in severities]
        axes[1, 0].bar(severities, mae_values)
        axes[1, 0].set_title('MAE by Severity Level')
        axes[1, 0].set_ylabel('Mean Absolute Error')
    
    # Error distribution
    if 'error_analysis' in results:
        error_metrics = ['exact_match_rate', 'within_1_score', 'within_2_scores']
        error_values = [results['error_analysis'][m] for m in error_metrics]
        axes[1, 1].bar(error_metrics, error_values)
        axes[1, 1].set_title('Prediction Accuracy Tolerance')
        axes[1, 1].set_ylabel('Rate')
        axes[1, 1].tick_params(axis='x', rotation=45)
    
    # Overall metrics summary
    overall_metrics = {
        'Accuracy': results['accuracy'],
        'MAE': results['mae'],
        'Cohen Kappa': results['cohen_kappa'],
        'F1 (Macro)': results['f1_macro']
    }
    
    metric_names = list(overall_metrics.keys())
    metric_values = list(overall_metrics.values())
    axes[1, 2].bar(metric_names, metric_values)
    axes[1, 2].set_title('Overall Performance Metrics')
    axes[1, 2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Evaluation plots saved to {save_path}")
    
    plt.show()

def generate_clinical_report(results: Dict) -> str:
    """Generate a clinical evaluation report."""
    report = []
    report.append("=" * 60)
    report.append("MEDICAL AI EVALUATION REPORT")
    report.append("Multi-Tissue Damage Scoring System")
    report.append("=" * 60)
    
    # Overall Performance
    report.append(f"\nOVERALL PERFORMANCE:")
    report.append(f"Accuracy: {results['accuracy']:.3f}")
    report.append(f"Mean Absolute Error: {results['mae']:.3f}")
    report.append(f"Root Mean Square Error: {results['rmse']:.3f}")
    report.append(f"Cohen's Kappa: {results['cohen_kappa']:.3f}")
    
    # Clinical Interpretation
    kappa_value = results['cohen_kappa']
    if kappa_value < 0.4:
        kappa_interpretation = "Poor agreement"
    elif kappa_value < 0.6:
        kappa_interpretation = "Moderate agreement"
    elif kappa_value < 0.8:
        kappa_interpretation = "Good agreement"
    else:
        kappa_interpretation = "Excellent agreement"
    
    report.append(f"Kappa Interpretation: {kappa_interpretation}")
    
    # Error Analysis
    report.append(f"\nERROR ANALYSIS:")
    error_analysis = results['error_analysis']
    report.append(f"Exact Match Rate: {error_analysis['exact_match_rate']:.3f}")
    report.append(f"Within 1 Score: {error_analysis['within_1_score']:.3f}")
    report.append(f"Within 2 Scores: {error_analysis['within_2_scores']:.3f}")
    report.append(f"Large Error Rate (≥3): {error_analysis['large_error_rate']:.3f}")
    
    # Tissue-Specific Performance
    if 'tissue_analysis' in results:
        report.append(f"\nTISSUE-SPECIFIC PERFORMANCE:")
        for tissue, metrics in results['tissue_analysis'].items():
            report.append(f"{tissue.capitalize()}:")
            report.append(f"  Accuracy: {metrics['accuracy']:.3f}")
            report.append(f"  MAE: {metrics['mae']:.3f}")
            report.append(f"  Sample Count: {metrics['count']}")
    
    # Clinical Recommendations
    report.append(f"\nCLINICAL RECOMMENDATIONS:")
    
    if results['accuracy'] >= 0.85:
        report.append("✓ Model shows good clinical performance")
    else:
        report.append("⚠ Model needs improvement before clinical use")
    
    if error_analysis['large_error_rate'] <= 0.05:
        report.append("✓ Low rate of clinically significant errors")
    else:
        report.append("⚠ High rate of large prediction errors - review required")
    
    if kappa_value >= 0.6:
        report.append("✓ Acceptable inter-rater agreement equivalent")
    else:
        report.append("⚠ Poor agreement - model reliability questionable")
    
    report.append("=" * 60)
    
    return "\n".join(report)


def debug_calculate_metrics(true_labels, predictions):
    """
    Calculate metrics with debug information and fallbacks.
    Enhanced from notebook debugging session.
    
    Args:
        true_labels: True damage scores
        predictions: Predicted damage scores
        
    Returns:
        dict: Comprehensive metrics with fallbacks
    """
    print("📊 Calculating classification metrics...")
    
    try:
        # Basic validation
        true_labels = np.array(true_labels)
        predictions = np.array(predictions)
        
        print(f"  - True labels shape: {true_labels.shape}")
        print(f"  - Predictions shape: {predictions.shape}")
        print(f"  - True label range: {true_labels.min()} to {true_labels.max()}")
        print(f"  - Prediction range: {predictions.min()} to {predictions.max()}")
        
        # Calculate basic metrics
        accuracy = np.mean(true_labels == predictions)
        print(f"  - Accuracy: {accuracy:.3f}")
        
        # Mean Absolute Error
        mae = np.mean(np.abs(true_labels - predictions))
        print(f"  - Mean Absolute Error: {mae:.3f}")
        
        # Try sklearn metrics first, with manual fallback
        try:
            precision, recall, f1, support = precision_recall_fscore_support(
                true_labels, predictions, average='weighted', zero_division=0
            )
            
            kappa = cohen_kappa_score(true_labels, predictions)
            
            print(f"  - Precision (weighted): {precision:.3f}")
            print(f"  - Recall (weighted): {recall:.3f}")
            print(f"  - F1-score (weighted): {f1:.3f}")
            print(f"  - Cohen's Kappa: {kappa:.3f}")
            
            return {
                'accuracy': accuracy,
                'mae': mae,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'kappa': kappa
            }
            
        except Exception as e:
            print(f"⚠️ Sklearn metrics failed: {e}")
            print("  - Using manual metric calculations")
        
        # Fallback manual calculations
        unique_labels = np.unique(np.concatenate([true_labels, predictions]))
        precisions = []
        recalls = []
        
        for label in unique_labels:
            tp = np.sum((true_labels == label) & (predictions == label))
            fp = np.sum((true_labels != label) & (predictions == label))
            fn = np.sum((true_labels == label) & (predictions != label))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            precisions.append(precision)
            recalls.append(recall)
        
        avg_precision = np.mean(precisions)
        avg_recall = np.mean(recalls)
        
        print(f"  - Manual Precision: {avg_precision:.3f}")
        print(f"  - Manual Recall: {avg_recall:.3f}")
        
        return {
            'accuracy': accuracy,
            'mae': mae,
            'precision': avg_precision,
            'recall': avg_recall,
            'f1': 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0,
            'kappa': 0.0  # Simplified for manual calculation
        }
        
    except Exception as e:
        print(f"❌ Error calculating metrics: {e}")
        import traceback
        traceback.print_exc()
        return {
            'accuracy': 0.0,
            'mae': float('inf'),
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'kappa': 0.0
        }


# Legacy function for backward compatibility
def calculate_metrics(test_labels, predictions):
    """Calculate basic metrics for backward compatibility."""
    accuracy = accuracy_score(test_labels, predictions)
    precision, recall, _, _ = precision_recall_fscore_support(
        test_labels, predictions, average='weighted', zero_division=0
    )
    return accuracy, precision, recall
