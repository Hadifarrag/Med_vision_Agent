from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class ChestXrayDataset(Dataset):
    def __init__(self,root_dir,transform=None):
        self.root_dir=Path(root_dir)
        self.transform=transform

        class_to_idx={"NORMAL":0,"PNEUMONIA":1}

        self.samples = []
        for class_name, label in class_to_idx.items():
            class_dir = self.root_dir / class_name
            if class_dir.exists():
                for filepath in class_dir.glob("*.jpeg"):
                    self.samples.append((filepath, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self,idx):
        filepath, label = self.samples[idx]

        image = Image.open(filepath).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)
        return image ,label
    @staticmethod
    def get_transform(split : str):
        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]
        if split == "train":
            return T.Compose([
            T.Resize((224, 224)),
            T.RandomRotation(degrees=7),
            T.ToTensor(),
            T.Normalize(mean=imagenet_mean, std=imagenet_std),])
        elif split in ["val", "test"]:
            return T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=imagenet_mean, std=imagenet_std),
            ])
        else:
            raise ValueError(f"Invalid split '{split}'. Expected 'train', 'val', or 'test'.")
if __name__=="__main__":
    print("=" * 50)
    print("DATASET SANITY CHECK")
    print("=" * 50)

    train_dir = Path(__file__).parent / "chest_xray" / "train"

    if not train_dir.exists():
        train_dir = Path(__file__).parent.parent / "data" / "chest_xray" / "train"

    print(f"Loading dataset from: {train_dir}")

    train_transform = ChestXrayDataset.get_transform("train")
    dataset = ChestXrayDataset(root_dir=train_dir, transform=train_transform)

    total_count = len(dataset)
    print(f"Total samples loaded: {total_count}")

    if total_count > 0:
        sample_img, sample_label = dataset[0]
        
        print("\n--- Sample [0] Inspection ---")
        print(f"Image Tensor Shape : {sample_img.shape} (Expected: torch.Size([3, 224, 224]))")
        print(f"Image Tensor Type  : {sample_img.dtype}")
        print(f"Label              : {sample_label} (0 = NORMAL, 1 = PNEUMONIA)")
        print(f"Value Range        : min={sample_img.min():.2f}, max={sample_img.max():.2f}")
        
        assert sample_img.shape == torch.Size([3, 224, 224]), "ERROR: Unexpected image shape!"
        assert sample_label in [0, 1], "ERROR: Label out of expected binary range!"
        print("\n✅ All sanity checks passed successfully!")
    else:
        print("\n❌ WARNING: No images found. Check your dataset path.")