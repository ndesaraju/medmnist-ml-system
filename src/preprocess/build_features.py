from torch.utils.data import DataLoader
from torchvision import transforms
from src.common.config import load_config
from medmnist import INFO
import medmnist
import os
import numpy as np
import random


def get_transforms(train=True):
    """
    Define image transformations for training and evaluation. Includes 
    data augmentation for training and basic preprocessing for evaluation.

    Args:
        train (bool, optional): Whether to apply training augmentations.
            Defaults to True.

    Returns:
        torchvision.transforms.Compose: Transformation pipeline.
    """
    if train:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
        ])
    return transforms.ToTensor()


def get_dataloaders(config):
    """
    Create PyTorch DataLoaders for train, validation, and test splits.

    Args:
        config (dict): Configuration dictionary containing dataset paths,
            batch size, and dataset identifier.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]:
            - Training DataLoader
            - Validation DataLoader
            - Test DataLoader

    Workflow:
        1. Load dataset class dynamically using MedMNIST metadata
        2. Initialize datasets for train/val/test splits
        3. Apply appropriate transformations
        4. Wrap datasets in DataLoaders with configured batch size and shuffling for training
    """
    info = INFO[config["data"]["data_flag"]]
    DataClass = getattr(medmnist, info["python_class"])

    train_dataset = DataClass(
        split="train",
        root=config["data"]["raw_dir"],
        transform=get_transforms(True),
    )

    val_dataset = DataClass(
        split="val",
        root=config["data"]["raw_dir"],
        transform=get_transforms(False),
    )

    test_dataset = DataClass(
        split="test",
        root=config["data"]["raw_dir"],
        transform=get_transforms(False),
    )


    return (
        DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True),
        DataLoader(val_dataset, batch_size=config["training"]["batch_size"]),
        DataLoader(test_dataset, batch_size=config["training"]["batch_size"]),
    )


def build_features():
    """
    Validate and inspect the data preprocessing pipeline.

    Performs sanity checks on dataset loading and transformation by:
        - Initializing DataLoaders
        - Verifying dataset sizes
        - Inspecting sample shapes
        - Confirming expected batch dimensions

    Returns:
        None

    Raises:
        ValueError: If dataset length cannot be determined.

    Notes:
        - Intended as a debugging and validation step.
        - Does not modify or persist data.
        - Helps ensure consistency before training.
    """
    config = load_config("config.yaml")
    batch_size = config["training"]["batch_size"]

    train_loader, val_loader, test_loader = get_dataloaders(config)

    print("Validating data pipeline...")

    for name, loader in zip(
        ["train", "val", "test"],
        [train_loader, val_loader, test_loader],
    ):
        dataset = getattr(loader, "dataset", None)
        ds_len = None
        try:
            ds_len = len(dataset) if dataset is not None else None
        except Exception:
            ds_len = None

        if ds_len is not None:
            # dataset[0] returns (image, label) for these medmnist datasets
            x, y = dataset[0]
            sample_shape = tuple(x.shape)
            expected_bs = min(batch_size, ds_len) if ds_len is not None else batch_size
            expected_batch_shape = (expected_bs,) + sample_shape
            print(f"{name}: dataset_len={ds_len}, sample_shape={sample_shape}, expected_batch_shape={expected_batch_shape}")
        else:
            raise ValueError(f"Could not determine dataset length for {name} dataset. Please check the data pipeline.")
        
    print("Preprocessing step complete.\n")


if __name__ == "__main__":
    build_features()