import torch

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def get_dataloaders(
    data_dir="./data",
    image_size=32,
    batch_size=128,
    val_ratio=0.1,
    num_workers=4
):
    # =========================================================
    # 1. Image preprocessing
    #
    # CIFAR-10:
    # original image: [3, 32, 32]
    #
    # ToTensor:
    # pixel range [0,255] -> [0,1]
    #
    # Normalize:
    # [0,1] -> approximately [-1,1]
    # =========================================================
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5)
        )
    ])

    # =========================================================
    # 2. Load CIFAR-10 training dataset
    # =========================================================
    full_train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        transform=transform,
        download=True
    )

    # =========================================================
    # 3. Split training / validation dataset
    # =========================================================
    val_size = int(
        len(full_train_dataset) * val_ratio
    )

    train_size = (
        len(full_train_dataset) - val_size
    )

    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # =========================================================
    # 4. DataLoader
    # =========================================================
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    return train_dataloader, val_dataloader