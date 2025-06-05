import os

# import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class LFWFaceDataset(Dataset):
    """
    A custom dataset class for the LFW (Labeled Faces in the Wild) dataset.

    This class handles the loading, preprocessing, and transformation of LFW face images
    for use in PyTorch models.
    """

    def __init__(self, images: np.ndarray, targets: np.ndarray, transform=None) -> None:
        """
        Initializes the LFWFaceDataset.

        :param images: Array of images in (N, H, W, C) format.
        :type images: np.ndarray
        :param targets: Array of labels.
        :type targets: np.ndarray
        :param transform: Optional transform to apply to the images.
        :type transform: callable, optional
        """
        self.images = images
        self.targets = targets
        self.transform = transform

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.

        :return: The length of the dataset.
        :rtype: int
        """
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """
        Retrieves a sample from the dataset at the given index.

        :param index: The index of the sample to retrieve.
        :type index: int
        :return: A tuple containing the transformed image (as a tensor) and its label.
        :rtype: tuple[torch.Tensor, int]
        """
        img = self.images[index]
        label = self.targets[index]

        # Convert to PIL Image
        img = Image.fromarray(img.astype("uint8"), "RGB")

        # Pad to square and resize to 160x160
        img = self.pad_and_resize(img)

        if self.transform:
            img = self.transform(img)

        return img, label

    def pad_and_resize(self, img: Image.Image) -> Image.Image:
        """
        Pads the image to a square and resizes it to 160x160 pixels.

        :param img: The input PIL Image.
        :type img: PIL.Image.Image
        :return: The padded and resized PIL Image.
        :rtype: PIL.Image.Image
        """
        width, height = img.size

        # Calculate padding
        max_dim = max(width, height)
        pad_width = (max_dim - width) // 2
        pad_height = (max_dim - height) // 2

        # Create new square image with black padding
        new_img = Image.new("RGB", (max_dim, max_dim), (0, 0, 0))
        new_img.paste(img, (pad_width, pad_height))

        # Resize to target size
        return new_img.resize((160, 160), Image.BILINEAR)


def get_lfw_dataloaders(
    batch_size: int = 32, data_dir: str = "./data", min_faces: int = 100, resize: float | None = None
) -> tuple[DataLoader, DataLoader, DataLoader, list]:
    """
    Downloads, preprocesses, and splits the LFW dataset into train, validation, and test DataLoaders.

    :param batch_size: The batch size for the DataLoaders.
    :type batch_size: int, optional
    :param data_dir: The directory to store the dataset.
    :type data_dir: str, optional
    :param min_faces: The minimum number of faces per person to include in the dataset.
    :type min_faces: int, optional
    :param resize: Ratio to resize each face picture.
    :type resize: float, optional
    :return: A tuple containing the train, validation, and test DataLoaders.
    :rtype: tuple[DataLoader, DataLoader, DataLoader, list]
    """
    # Ensure data directory exists
    os.makedirs(data_dir, exist_ok=True)

    # Load LFW dataset with specified parameters
    lfw_dataset = fetch_lfw_people(
        data_home=data_dir, min_faces_per_person=min_faces, color=True, download_if_missing=True, resize=resize
    )

    # Extract data and targets
    X = lfw_dataset.images
    y = lfw_dataset.target
    target_names = lfw_dataset.target_names

    # Print dataset information
    print(f"Loaded LFW dataset with {X.shape[0]} images")
    print(f"Image shape: {X.shape[1:]} (original size)")
    print(f"Number of classes: {len(lfw_dataset.target_names)}")

    # Split into train, validation, and test sets
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.15, stratify=y_temp, random_state=42)

    print(f"Train size: {len(X_train)}")
    print(f"Validation size: {len(X_val)}")
    print(f"Test size: {len(X_test)}")

    # Shared normalization
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # Train transformations with augmentation
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.RandomApply(
                [
                    transforms.ColorJitter(brightness=0.2, contrast=0.2),
                ],
                p=0.5,
            ),
            transforms.ToTensor(),
            norm,
        ]
    )

    # Validation/test transformations
    val_transform = transforms.Compose([transforms.ToTensor(), norm])

    # Create datasets
    train_dataset = LFWFaceDataset(X_train, y_train, transform=train_transform)
    val_dataset = LFWFaceDataset(X_val, y_val, transform=val_transform)
    test_dataset = LFWFaceDataset(X_test, y_test, transform=val_transform)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader, test_loader, target_names


if __name__ == "__main__":
    get_lfw_dataloaders()
