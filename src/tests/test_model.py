"""
Basic unit test for MultiTissueDamageCNN model output shape and tissue input.
"""
import torch
from src.models.cnn import build_model

def test_model_output_shape():
    model = build_model(input_shape=(3,224,224), num_tissues=5, num_classes=10)
    x = torch.randn(2, 3, 224, 224)
    tissue_onehot = torch.eye(5)[[0,1]]
    out = model(x, tissue_onehot)
    assert out.shape == (2, 10), f"Expected output shape (2,10), got {out.shape}"
    print("test_model_output_shape passed.")

if __name__ == "__main__":
    test_model_output_shape()
