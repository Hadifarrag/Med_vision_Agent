"""
Step 1 EDA: run this BEFORE writing any training code.

Checks:
1. Class counts per split (catch imbalance early)
2. A visual sample grid (build intuition for what "normal" vs "pneumonia"
   actually look like -- you'll need this intuition later to judge whether
   Grad-CAM heatmaps look sane)
3. A basic filename-overlap check between splits (catch potential leakage)
"""
from pathlib import Path
from collections import defaultdict
import random

import matplotlib.pyplot as plt
from PIL import Image

DATA_DIR = Path(__file__).parent.parent / "data" / "chest_xray"
SPLITS = ["train", "val", "test"]
CLASSES = ["NORMAL", "PNEUMONIA"]


def class_counts():
    print("=" * 50)
    print("CLASS COUNTS PER SPLIT")
    print("=" * 50)
    counts = defaultdict(dict)
    for split in SPLITS:
        for cls in CLASSES:
            folder = DATA_DIR / split / cls
            n = len(list(folder.glob("*.jpeg"))) if folder.exists() else 0
            counts[split][cls] = n
        total = sum(counts[split].values())
        if total > 0:
            pneumonia_pct = 100 * counts[split]["PNEUMONIA"] / total
            print(f"{split:6s}  NORMAL={counts[split]['NORMAL']:5d}  "
                  f"PNEUMONIA={counts[split]['PNEUMONIA']:5d}  "
                  f"({pneumonia_pct:.1f}% pneumonia)")
        else:
            print(f"{split:6s}  -- folder not found, run download_data.py first --")
    return counts


def sample_grid(n_per_class=4):
    print("\nSaving sample image grid to notebooks/sample_grid.png ...")
    fig, axes = plt.subplots(2, n_per_class, figsize=(3 * n_per_class, 6))
    for row, cls in enumerate(CLASSES):
        folder = DATA_DIR / "train" / cls
        files = list(folder.glob("*.jpeg"))
        if not files:
            continue
        sample = random.sample(files, min(n_per_class, len(files)))
        for col, f in enumerate(sample):
            img = Image.open(f).convert("L")
            axes[row, col].imshow(img, cmap="gray")
            axes[row, col].set_title(cls, fontsize=9)
            axes[row, col].axis("off")
    plt.tight_layout()
    out_path = Path(__file__).parent / "sample_grid.png"
    plt.savefig(out_path, dpi=120)
    print(f"Saved: {out_path}")


def leakage_check():
    """Flag exact filename overlap between splits (a cheap first check --
    doesn't catch near-duplicate/same-patient leakage, but catches the
    obvious case for free)."""
    print("\n" + "=" * 50)
    print("FILENAME OVERLAP CHECK (cheap leakage sanity check)")
    print("=" * 50)
    split_files = {}
    for split in SPLITS:
        files = set()
        for cls in CLASSES:
            folder = DATA_DIR / split / cls
            if folder.exists():
                files.update(f.name for f in folder.glob("*.jpeg"))
        split_files[split] = files

    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    for a, b in pairs:
        overlap = split_files.get(a, set()) & split_files.get(b, set())
        print(f"{a} ∩ {b}: {len(overlap)} overlapping filenames")
    print("\nNote: this only catches identical filenames. This dataset is "
          "known to have some near-duplicate images across patients/splits -- "
          "worth a manual spot-check later, don't fully trust test accuracy "
          "at face value.")


if __name__ == "__main__":
    class_counts()
    sample_grid()
    leakage_check()
