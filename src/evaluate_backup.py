"""
Model evaluation utilities for histopathology image analysis.

Includes functions for evaluating a PyTorch model and calculating classification metrics.

"""

import torch
import numpy as np
from src.data.preprocess import load_test_data


def evaluate_model(model, test_data, test_labels, tissue_labels, device: 'str | torch.device' = 'cpu'):
    """
    Evaluate a PyTorch model on test data.

    Args:
        model (torch.nn.Module): Trained PyTorch model.
        test_data (np.ndarray or torch.Tensor): Test images.
        test_labels (np.ndarray or torch.Tensor): True labels.
        tissue_labels (list): List of tissue types for each sample.
        device (str or torch.device): Device to run evaluation.

    Returns:
        np.ndarray: Predicted scores.
    """
    TISSUE_TYPES = ["lung", "kidney", "heart", "liver", "bowel"]
    def one_hot_tissue(tissue):
        idx = TISSUE_TYPES.index(tissue)
        arr = torch.zeros(len(TISSUE_TYPES), dtype=torch.float32)
        arr[idx] = 1.0
        return arr
    model.eval()
    test_data = torch.tensor(test_data, dtype=torch.float32).to(device)
    tissue_onehot = torch.stack([one_hot_tissue(t) for t in tissue_labels]).to(device)
    with torch.no_grad():
        outputs = model(test_data, tissue_onehot)
        scores = torch.argmax(outputs, dim=1).cpu().numpy().flatten()
    return scores

def calculate_metrics(true_labels, predicted_scores, threshold=0.5):
    """
    Calculate accuracy, precision, and recall for binary classification.

    Args:
        true_labels (np.ndarray): Ground truth labels.
        predicted_scores (np.ndarray): Model predicted scores.
        threshold (float): Threshold for positive class.

    Returns:
        tuple: (accuracy, precision, recall)
    """
    preds = predicted_scores > threshold
    true_labels = np.array(true_labels).astype(bool)
    accuracy = (true_labels == preds).mean()
    precision = (true_labels & preds).sum() / (preds.sum() + 1e-8)
    recall = (true_labels & preds).sum() / (true_labels.sum() + 1e-8)
    return accuracy, precision, recall


def main():
    import yaml
    from src.models.cnn import build_model
    with open("src/config.yaml") as f:
        config = yaml.safe_load(f)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(input_shape=tuple(config["model"]["input_shape"]), num_tissues=config["model"]["num_tissues"], num_classes=config["model"]["num_classes"])
    model.load_state_dict(torch.load('path_to_your_trained_model.pth', map_location=device))
    model.to(device)

    test_data, test_labels = load_test_data('path_to_your_test_data')
    tissue_labels = ["lung"] * len(test_labels)  # Replace with actual tissue labels
    damage_scores = evaluate_model(model, test_data, test_labels, tissue_labels, device=device)
    accuracy, precision, recall = calculate_metrics(test_labels, damage_scores)

    print(f'Accuracy: {accuracy:.2f}')
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')

if __name__ == "__main__":
    main()