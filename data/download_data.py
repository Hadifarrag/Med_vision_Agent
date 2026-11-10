"""
Downloads the Chest X-Ray Pneumonia dataset (Kermany et al.) from Kaggle.

Requires a Kaggle API token at ~/.kaggle/kaggle.json.
Get one from: https://www.kaggle.com/settings -> API -> Create New Token
"""
import shutil
from pathlib import Path
import kagglehub

DEST = Path(__file__).parent / "chest_xray"


def main():
    print("Downloading dataset via kagglehub (cached after first run)...")
    path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
    print(f"Downloaded to cache: {path}")

    src = Path(path)
    # kagglehub caches outside the project; copy the relevant subfolder in
    if not DEST.exists():
        # The dataset zip usually extracts to a nested chest_xray/ folder
        inner = next(src.glob("**/chest_xray"), src)
        shutil.copytree(inner, DEST)
        print(f"Copied dataset into {DEST}")
    else:
        print(f"{DEST} already exists, skipping copy.")

    for split in ["train", "val", "test"]:
        split_dir = DEST / split
        if split_dir.exists():
            n_normal = len(list((split_dir / "NORMAL").glob("*.jpeg")))
            n_pneumonia = len(list((split_dir / "PNEUMONIA").glob("*.jpeg")))
            print(f"{split}: NORMAL={n_normal}  PNEUMONIA={n_pneumonia}")
        else:
            print(f"WARNING: expected split folder not found: {split_dir}")


if __name__ == "__main__":
    main()
