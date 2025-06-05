import numpy as np
import pytest
import torch
from PIL import Image
from torchvision import transforms

from src.data.data_loader import LFWFaceDataset, get_lfw_dataloaders

# Mock dataset parameters
IMG_SHAPE = (62, 47, 3)
NUM_CLASSES = 5
NUM_SAMPLES = 100
BATCH_SIZE = 16


@pytest.fixture
def mock_lfw_dataset():
    """Create a mock LFW dataset with random images and labels"""
    images = np.random.randint(0, 255, size=(NUM_SAMPLES, *IMG_SHAPE), dtype=np.uint8)
    targets = np.random.randint(0, NUM_CLASSES, size=(NUM_SAMPLES,))
    return images, targets


@pytest.fixture
def mock_lfw_dataset_small():
    """Small dataset for edge case testing"""
    images = np.random.randint(0, 255, size=(3, 10, 10, 3), dtype=np.uint8)
    targets = np.array([0, 1, 0])
    return images, targets


def test_dataset_init(mock_lfw_dataset):
    """Test dataset initialization"""
    images, targets = mock_lfw_dataset
    dataset = LFWFaceDataset(images, targets)

    assert len(dataset) == NUM_SAMPLES
    assert dataset.images.shape == (NUM_SAMPLES, *IMG_SHAPE)
    assert dataset.targets.shape == (NUM_SAMPLES,)
    assert dataset.transform is None


def test_dataset_getitem(mock_lfw_dataset):
    """Test single item retrieval"""
    images, targets = mock_lfw_dataset
    dataset = LFWFaceDataset(images, targets)

    for idx in [0, 10, -1]:
        img_tensor, label = dataset[idx]

        assert isinstance(img_tensor, Image.Image)
        assert img_tensor.size == (160, 160)
        assert label == targets[idx]


def test_transforms(mock_lfw_dataset):
    """Test transformation pipelines"""
    images, targets = mock_lfw_dataset

    # With augmentation
    train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
    dataset = LFWFaceDataset(images, targets, train_transform)
    img_tensor, _ = dataset[0]
    assert isinstance(img_tensor, torch.Tensor)
    assert img_tensor.shape == (3, 160, 160)

    # Without augmentation
    dataset = LFWFaceDataset(images, targets, transforms.ToTensor())
    img_tensor, _ = dataset[0]
    assert isinstance(img_tensor, torch.Tensor)


def test_edge_cases(mock_lfw_dataset_small):
    """Test dataset edge cases"""
    images, targets = mock_lfw_dataset_small
    dataset = LFWFaceDataset(images, targets)

    # Test negative index
    img, label = dataset[-1]
    assert label == targets[-1]

    # Test index out of range
    with pytest.raises(IndexError):
        _ = dataset[len(dataset)]


def test_dataloader_transforms(tmp_path):
    """Test transform differences between train/val"""
    train_loader, val_loader, _, _ = get_lfw_dataloaders(data_dir=str(tmp_path / "data"), min_faces=20, resize=0.1)

    # Train should have augmentations
    assert any(isinstance(t, transforms.RandomHorizontalFlip) for t in train_loader.dataset.transform.transforms)

    # Validation should be simple
    val_transforms = val_loader.dataset.transform.transforms
    assert len(val_transforms) == 2  # Only ToTensor and Normalize
    assert isinstance(val_transforms[0], transforms.ToTensor)
