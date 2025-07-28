"""
Medical-grade visualization utilities for WSI damage scoring analysis.

Includes Grad-CAM, attention maps, uncertainty visualization, and clinical reporting.
Enhanced with debug visualization functions from notebook.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
from PIL import Image
import logging

logger = logging.getLogger(__name__)

# Optional imports with fallbacks
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("OpenCV not available - some features will be limited")

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    logger.warning("Seaborn not available - using matplotlib fallback")

try:
    from sklearn.metrics import confusion_matrix
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("Scikit-learn not available - using manual implementations")


def debug_visualize_results(true_labels, predictions, tissue_labels=None):
    """
    Create visualizations with fallbacks for missing dependencies.
    Enhanced from notebook debugging session.
    
    Args:
        true_labels: Array of true damage scores
        predictions: Array of predicted damage scores  
        tissue_labels: Optional list of tissue types
    """
    print("📊 Creating debug visualizations...")
    
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Confusion Matrix
        try:
            if SKLEARN_AVAILABLE:
                cm = confusion_matrix(true_labels, predictions)
            else:
                # Manual confusion matrix
                labels = np.unique(np.concatenate([true_labels, predictions]))
                cm = np.zeros((len(labels), len(labels)), dtype=int)
                for i, true_label in enumerate(labels):
                    for j, pred_label in enumerate(labels):
                        cm[i, j] = np.sum((true_labels == true_label) & (predictions == pred_label))
            
            # Plot confusion matrix
            if SEABORN_AVAILABLE:
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
            else:
                im = axes[0, 0].imshow(cm, cmap='Blues')
                # Add text annotations
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        axes[0, 0].text(j, i, str(cm[i, j]), ha='center', va='center')
                plt.colorbar(im, ax=axes[0, 0])
            
            axes[0, 0].set_xlabel('Predicted')
            axes[0, 0].set_ylabel('True')
            axes[0, 0].set_title('Confusion Matrix: Damage Scores')
            
        except Exception as e:
            axes[0, 0].text(0.5, 0.5, f'Confusion Matrix\nError: {str(e)[:50]}...', 
                           ha='center', va='center', transform=axes[0, 0].transAxes)
            axes[0, 0].set_title('Confusion Matrix (Error)')
        
        # 2. Prediction Distribution
        try:
            axes[0, 1].hist(predictions, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, 1].set_xlabel('Predicted Damage Score')
            axes[0, 1].set_ylabel('Count')
            axes[0, 1].set_title('Distribution of Predicted Damage Scores')
            axes[0, 1].set_xticks(range(10))
            
        except Exception as e:
            axes[0, 1].text(0.5, 0.5, f'Prediction Distribution\nError: {str(e)[:50]}...', 
                           ha='center', va='center', transform=axes[0, 1].transAxes)
        
        # 3. True vs Predicted Scatter
        try:
            axes[1, 0].scatter(true_labels, predictions, alpha=0.6, c='red')
            axes[1, 0].plot([0, 9], [0, 9], 'k--', alpha=0.75, zorder=0)  # Perfect prediction line
            axes[1, 0].set_xlabel('True Damage Score')
            axes[1, 0].set_ylabel('Predicted Damage Score')
            axes[1, 0].set_title('True vs Predicted Scores')
            axes[1, 0].grid(True, alpha=0.3)
            
        except Exception as e:
            axes[1, 0].text(0.5, 0.5, f'Scatter Plot\nError: {str(e)[:50]}...', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
        
        # 4. Error Distribution
        try:
            errors = predictions - true_labels
            axes[1, 1].hist(errors, bins=15, alpha=0.7, color='orange', edgecolor='black')
            axes[1, 1].axvline(x=0, color='red', linestyle='--', alpha=0.8)
            axes[1, 1].set_xlabel('Prediction Error (Pred - True)')
            axes[1, 1].set_ylabel('Count')
            axes[1, 1].set_title(f'Error Distribution (Mean: {np.mean(errors):.2f})')
            
        except Exception as e:
            axes[1, 1].text(0.5, 0.5, f'Error Distribution\nError: {str(e)[:50]}...', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
        
        plt.tight_layout()
        plt.show()
        
        # Additional analysis
        print(f"\n📊 Detailed Analysis:")
        
        # Accuracy by damage level
        unique_true = np.unique(true_labels)
        print(f"  Accuracy by damage level:")
        for damage_level in unique_true:
            mask = true_labels == damage_level
            if np.sum(mask) > 0:
                acc = np.mean(predictions[mask] == damage_level)
                count = np.sum(mask)
                print(f"    Level {damage_level}: {acc:.2f} ({count} samples)")
        
        # Error analysis
        errors = predictions - true_labels
        print(f"  Error Analysis:")
        print(f"    Mean error: {np.mean(errors):.3f}")
        print(f"    Std error: {np.std(errors):.3f}")
        print(f"    Max error: {np.max(np.abs(errors))}")
        print(f"    Exact matches: {np.sum(errors == 0)}/{len(errors)} ({np.mean(errors == 0):.1%})")
        print(f"    Within ±1: {np.sum(np.abs(errors) <= 1)}/{len(errors)} ({np.mean(np.abs(errors) <= 1):.1%})")
        
        if tissue_labels is not None:
            print(f"  Performance by tissue:")
            for tissue in set(tissue_labels):
                mask = np.array(tissue_labels) == tissue
                if np.sum(mask) > 0:
                    tissue_acc = np.mean(predictions[mask] == true_labels[mask])
                    tissue_mae = np.mean(np.abs(predictions[mask] - true_labels[mask]))
                    count = np.sum(mask)
                    print(f"    {tissue}: Acc={tissue_acc:.2f}, MAE={tissue_mae:.2f} ({count} samples)")
        
    except Exception as e:
        print(f"❌ Error in visualization: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback simple plot
        try:
            plt.figure(figsize=(8, 4))
            plt.subplot(1, 2, 1)
            plt.hist(true_labels, alpha=0.7, label='True', bins=10)
            plt.hist(predictions, alpha=0.7, label='Predicted', bins=10)
            plt.xlabel('Damage Score')
            plt.ylabel('Count')
            plt.title('Score Distribution')
            plt.legend()
            
            plt.subplot(1, 2, 2)
            plt.scatter(true_labels, predictions, alpha=0.6)
            plt.xlabel('True')
            plt.ylabel('Predicted')
            plt.title('True vs Predicted')
            
            plt.tight_layout()
            plt.show()
            
        except Exception as fallback_error:
            print(f"❌ Even fallback visualization failed: {fallback_error}")


def plot_damage_scores(scores: List[float], tissue_types: List[str], 
                      save_path: Optional[str] = None, title: str = "Tissue Damage Scores"):
    """
    Create a comprehensive visualization of damage scores across tissues.
    
    Args:
        scores: List of damage scores (0-9)
        tissue_types: List of corresponding tissue types
        save_path: Optional path to save the plot
        title: Plot title
    """
    plt.figure(figsize=(12, 8))
    
    # Create subplot layout
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Bar plot of scores by tissue
    unique_tissues = list(set(tissue_types))
    tissue_scores = {tissue: [] for tissue in unique_tissues}
    
    for score, tissue in zip(scores, tissue_types):
        tissue_scores[tissue].append(score)
    
    mean_scores = [np.mean(tissue_scores[tissue]) for tissue in unique_tissues]
    std_scores = [np.std(tissue_scores[tissue]) for tissue in unique_tissues]
    
    bars = axes[0, 0].bar(unique_tissues, mean_scores, yerr=std_scores, 
                         capsize=5, alpha=0.7, color='skyblue')
    axes[0, 0].set_title('Mean Damage Scores by Tissue Type')
    axes[0, 0].set_ylabel('Damage Score (0-9)')
    axes[0, 0].set_ylim(0, 9)
    
    # Add value labels on bars
    for i, (bar, score) in enumerate(zip(bars, mean_scores)):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                       f'{score:.1f}', ha='center', va='bottom')
    
    # 2. Distribution of damage scores
    axes[0, 1].hist(scores, bins=10, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[0, 1].set_title('Distribution of All Damage Scores')
    axes[0, 1].set_xlabel('Damage Score')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_xticks(range(10))
    
    # 3. Box plot by tissue type
    tissue_data = [tissue_scores[tissue] for tissue in unique_tissues if tissue_scores[tissue]]
    if tissue_data:
        axes[1, 0].boxplot(tissue_data, labels=unique_tissues)
        axes[1, 0].set_title('Damage Score Distribution by Tissue')
        axes[1, 0].set_ylabel('Damage Score')
        axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 4. Severity classification
    severity_counts = {
        'Minimal (0-2)': sum(1 for s in scores if 0 <= s <= 2),
        'Mild (3-4)': sum(1 for s in scores if 3 <= s <= 4),
        'Moderate (5-6)': sum(1 for s in scores if 5 <= s <= 6),
        'Severe (7-9)': sum(1 for s in scores if 7 <= s <= 9)
    }
    
    colors = ['green', 'yellow', 'orange', 'red']
    axes[1, 1].pie(severity_counts.values(), labels=severity_counts.keys(), 
                   autopct='%1.1f%%', colors=colors)
    axes[1, 1].set_title('Damage Severity Distribution')
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Damage score plot saved to {save_path}")
    
    plt.show()

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for medical image interpretation.
    """
    
    def __init__(self, model, target_layer_name: str):
        self.model = model
        self.target_layer_name = target_layer_name
        self.gradients = None
        self.activations = None
        self.hooks = []
        
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks."""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        # Find target layer
        for name, module in self.model.named_modules():
            if name == self.target_layer_name:
                self.hooks.append(module.register_forward_hook(forward_hook))
                self.hooks.append(module.register_backward_hook(backward_hook))
                break
    
    def generate_cam(self, input_tensor: torch.Tensor, tissue_onehot: torch.Tensor, 
                    class_idx: Optional[int] = None) -> np.ndarray:
        """
        Generate Class Activation Map.
        
        Args:
            input_tensor: Input image tensor
            tissue_onehot: Tissue type one-hot encoding
            class_idx: Target class index (if None, uses predicted class)
        
        Returns:
            CAM heatmap as numpy array
        """
        self.model.eval()
        
        # Forward pass
        input_tensor.requires_grad_()
        output = self.model(input_tensor, tissue_onehot)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        output[0, class_idx].backward()
        
        # Generate CAM
        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=[1, 2])  # [C]
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # Apply ReLU and normalize
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.numpy()
    
    def visualize_cam(self, original_image: np.ndarray, cam: np.ndarray, 
                     alpha: float = 0.4) -> np.ndarray:
        """
        Overlay CAM on original image.
        
        Args:
            original_image: Original input image
            cam: Class activation map
            alpha: Transparency of overlay
        
        Returns:
            Overlayed image
        """
        # Resize CAM to match image size
        if cam.shape != original_image.shape[:2]:
            cam = cv2.resize(cam, (original_image.shape[1], original_image.shape[0]))
        
        # Convert CAM to heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Ensure original image is in correct format
        if original_image.max() <= 1.0:
            original_image = (original_image * 255).astype(np.uint8)
        
        # Overlay heatmap on original image
        overlayed = cv2.addWeighted(original_image, 1 - alpha, heatmap, alpha, 0)
        
        return overlayed
    
    def cleanup(self):
        """Remove hooks."""
        for hook in self.hooks:
            hook.remove()

# Legacy function for backward compatibility
def grad_cam(model, image_tensor, tissue_onehot, target_class=None):
    """
    Generate Grad-CAM heatmap for a given image and tissue type.
    Legacy function for backward compatibility.
    """
    # Use the new GradCAM class
    grad_cam_generator = GradCAM(model, 'conv3')  # Assuming conv3 as target layer
    cam = grad_cam_generator.generate_cam(image_tensor, tissue_onehot, target_class)
    grad_cam_generator.cleanup()
    return cam
