"""
Utility script to prepare dependencies for Industrial Anomaly Detection:
1. Cache ResNet-18 pretrained weights locally
2. Download MVTec AD 'bottle' category dataset from Hugging Face
"""

import sys
from pathlib import Path
from torchvision.models import resnet18, ResNet18_Weights
from huggingface_hub import snapshot_download


def download_all():
    print("=" * 60)
    print("Step 1: Downloading & caching pretrained ResNet-18 weights...")
    print("=" * 60)
    _ = resnet18(weights=ResNet18_Weights.DEFAULT)
    print("[OK] ResNet-18 weights ready.\n")

    print("=" * 60)
    print("Step 2: Downloading MVTec AD 'bottle' dataset (~60MB)...")
    print("=" * 60)
    data_dir = Path("data/mvtec_anomaly_detection")
    data_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id="foersben/mvtec-ad",
        repo_type="dataset",
        allow_patterns="bottle/**",
        local_dir=str(data_dir),
    )
    
    bottle_dir = data_dir / "bottle"
    train_good = bottle_dir / "train" / "good"
    if train_good.exists():
        num_images = len(list(train_good.glob("*.png")))
        print(f"[OK] MVTec 'bottle' dataset ready with {num_images} normal training images.")
    else:
        print("[WARNING] Dataset directory created. Please verify structure.")


if __name__ == "__main__":
    download_all()

