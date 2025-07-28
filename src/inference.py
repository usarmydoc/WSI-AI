"""
Batch inference script for multi-tissue WSI damage scoring.

Loads a trained model, processes slides, aggregates patch predictions, and exports results.
"""
import torch
import numpy as np
import yaml
import csv
from src.models.cnn import build_model
from src.data.dataset import WSIPatchDataset
from src.data.preprocess import extract_patches_grid, preprocess_images
from src.utils import load_model
from PIL import Image

TISSUE_TYPES = ["lung", "kidney", "heart", "liver", "bowel"]

def one_hot_tissue(tissue):
    idx = TISSUE_TYPES.index(tissue)
    arr = np.zeros(len(TISSUE_TYPES), dtype=np.float32)
    arr[idx] = 1.0
    return arr

def aggregate_patch_scores(scores, method="mean"):
    if method == "mean":
        return float(np.mean(scores))
    elif method == "median":
        return float(np.median(scores))
    elif method == "max":
        return float(np.max(scores))
    else:
        raise ValueError(f"Unknown aggregation method: {method}")

def run_inference(model_path, slides_info, config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    model = build_model(
        input_shape=tuple(config["model"]["input_shape"]),
        num_classes=config["model"]["num_classes"]
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    results = []
    for slide in slides_info:
        wsi_path = slide["wsi_path"]
        tissue = slide["tissue"]
        reference_image_path = config["preprocessing"]["reference_images"][tissue]
        # Load reference image for stain normalization
        reference_image = np.array(Image.open(reference_image_path))
        patches = extract_patches_grid(wsi_path, patch_size=tuple(config["model"]["patch_size"]), stride=tuple(config["model"]["stride"]))
        patches = preprocess_images(patches, reference_image, normalizer_method=config["preprocessing"]["normalizer_method"])
        tissue_onehot = torch.tensor(one_hot_tissue(tissue)).unsqueeze(0)
        patch_scores = []
        for patch in patches:
            patch_tensor = torch.tensor(patch, dtype=torch.float32).permute(2,0,1).unsqueeze(0)
            with torch.no_grad():
                output = model(patch_tensor, tissue_onehot)
                score = torch.argmax(output, dim=1).item()
                patch_scores.append(score)
        slide_score = aggregate_patch_scores(patch_scores, method=config["model"]["aggregate_method"])
        results.append({"wsi_path": wsi_path, "tissue": tissue, "damage_score": slide_score})
    # Export results
    with open(config["inference"]["output_csv"], "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["wsi_path", "tissue", "damage_score"])
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    print(f"Inference complete. Results saved to {config['inference']['output_csv']}")

if __name__ == "__main__":
    # Example usage
    slides_info = [
        {"wsi_path": "path/to/slide1.svs", "tissue": "lung"},
        {"wsi_path": "path/to/slide2.svs", "tissue": "kidney"},
    ]
    run_inference("path/to/model.pth", slides_info, "src/config.yaml")
