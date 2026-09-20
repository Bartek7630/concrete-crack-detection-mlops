import hashlib
import json
import os
import random
import shutil
from pathlib import Path

import kagglehub
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

SAMPLE_PER_CLASS = 15000
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
RANDOM_SEED = 42
classes = ["Positive", "Negative"]

TARGET_DIR = Path("data/processed")
REF_DIR = Path("data/reference")


def download_data() -> str:
    path = kagglehub.dataset_download("arunrk7/surface-crack-detection")
    return path


random.seed(RANDOM_SEED)


def find_and_delete_duplicates(path_to_raw_data: str) -> None:
    for cls in classes:
        class_folder = Path(path_to_raw_data) / cls
        seen_hashes = {}
        n_deleted = 0

        for file_path in class_folder.glob("*.jpg"):
            with open(file_path, "rb") as f:
                file_hash = hashlib.md5(f.read()).hexdigest()

            if file_hash not in seen_hashes:
                seen_hashes[file_hash] = file_path
            else:
                os.remove(file_path)
                n_deleted += 1

        print(f"{cls}: deleted {n_deleted} duplicates, left {len(seen_hashes)} unique files")


def get_max_balanced_sample_size(path_to_raw_data: str) -> int:
    counts = []
    for cls in classes:
        class_folder = Path(path_to_raw_data) / cls
        n_images = len(list(class_folder.glob("*.jpg")))
        counts.append(n_images)
    return min(counts)


def split_data(path_to_raw_data: str) -> None:
    if TARGET_DIR.exists():
        shutil.rmtree(TARGET_DIR)
    if REF_DIR.exists():
        shutil.rmtree(REF_DIR)

    max_available = get_max_balanced_sample_size(path_to_raw_data)
    sample_size = min(SAMPLE_PER_CLASS, max_available)

    print(
        f"Requested {SAMPLE_PER_CLASS} per class, {max_available} available -> using {sample_size}"
    )

    for split in ["train", "val", "test"]:
        for cls in classes:
            (TARGET_DIR / split / cls).mkdir(parents=True, exist_ok=True)

    for cls in classes:
        (REF_DIR / cls).mkdir(parents=True, exist_ok=True)

    for cls in classes:
        class_folder = Path(path_to_raw_data) / cls
        all_images = list(class_folder.glob("*.jpg"))

        assert len(all_images) > 0, f"No images found in {class_folder}"

        random.shuffle(all_images)

        selected = all_images[:sample_size]

        n_train = int(len(selected) * SPLIT_RATIOS["train"])
        n_val = int(len(selected) * SPLIT_RATIOS["val"])

        splits_data = {
            "train": selected[:n_train],
            "val": selected[n_train : n_train + n_val],
            "test": selected[n_train + n_val :],
        }

        for split, files in splits_data.items():
            for file in files:
                shutil.copy(file, TARGET_DIR / split / cls / file.name)

        for file in splits_data["train"][:500]:
            shutil.copy(file, REF_DIR / cls / file.name)


train_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

val_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def flip_label(y: int) -> int:
    return 1 - y


def upload_data():
    train_dataset = datasets.ImageFolder(
        root=str(TARGET_DIR / "train"),
        transform=train_transforms,
        target_transform=flip_label,
    )

    val_dataset = datasets.ImageFolder(
        root=str(TARGET_DIR / "val"),
        transform=val_transforms,
        target_transform=flip_label,
    )

    test_dataset = datasets.ImageFolder(
        root=str(TARGET_DIR / "test"),
        transform=val_transforms,
        target_transform=flip_label,
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=1)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=1)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=1)

    class_mapping = {"Positive": 0, "Negative": 1}

    print("Class mapping:", class_mapping)

    with open(TARGET_DIR / "class_mapping.json", "w") as f:
        json.dump(class_mapping, f)

    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")

    return list((train_loader, val_loader, test_loader)), class_mapping


if __name__ == "__main__":
    data_path = download_data()
    find_and_delete_duplicates(data_path)
    split_data(data_path)
    upload_data()
