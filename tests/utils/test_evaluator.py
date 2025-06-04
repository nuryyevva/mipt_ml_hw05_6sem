import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.utils.evaluator import ModelEvaluator


# Mock configuration class
class MockConfig:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 32


# Realistic mock model that matches face recognition input dimensions
class FaceRecognitionMock(torch.nn.Module):
    def __init__(self, num_classes=5, embedding_size=128):
        super().__init__()
        # Convolutional layers to handle image input
        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # Classifier head
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(32 * 40 * 40, 256),  # 160x160 -> 80x80 -> 40x40
            torch.nn.ReLU(),
            torch.nn.Linear(256, num_classes),
        )
        # Embedding extractor
        self.embedding_layer = torch.nn.Linear(32 * 40 * 40, embedding_size)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        return self.classifier(x)

    def get_embeddings(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        return self.embedding_layer(x)


@pytest.fixture
def test_data():
    """Create realistic test dataset matching face recognition input"""
    # Create random image data (100 samples, 3 channels, 160x160 pixels)
    features = torch.randn(100, 3, 160, 160)
    labels = torch.randint(0, 5, (100,))  # 5 classes
    return TensorDataset(features, labels)


@pytest.fixture
def test_loader(test_data):
    """Create test data loader with realistic batch size"""
    return DataLoader(test_data, batch_size=10)


@pytest.fixture
def mock_model():
    """Create a mock model that matches face recognition input dimensions"""
    return FaceRecognitionMock()


@pytest.fixture
def mock_config():
    """Create mock configuration"""
    return MockConfig()


def test_evaluator_initialization(mock_model, test_loader, mock_config):
    """Test that evaluator initializes correctly"""
    evaluator = ModelEvaluator(mock_model, test_loader, mock_config)

    # Check model is on correct device
    assert next(mock_model.parameters()).device == mock_config.DEVICE

    # Check model in eval mode
    assert not mock_model.training


def test_evaluation_output(mock_model, test_loader, mock_config):
    """Test evaluation returns expected outputs"""
    evaluator = ModelEvaluator(mock_model, test_loader, mock_config)
    results = evaluator.evaluate()

    # Check output types and keys
    assert isinstance(results, dict)
    assert "accuracy" in results
    assert "embeddings" in results
    assert "labels" in results

    # Check accuracy is within valid range
    assert 0 <= results["accuracy"] <= 100

    # Check embeddings shape (100 samples, 128-dim embeddings)
    assert results["embeddings"].shape == (100, 128)

    # Check labels shape
    assert results["labels"].shape == (100,)


def test_evaluation_accuracy(mock_config):
    """Test accuracy calculation is correct with predictable model"""
    # Create small predictable dataset (10 samples)
    features = torch.randn(10, 3, 160, 160)
    labels = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])  # 10 samples

    # Create model that always predicts class 0
    class AlwaysZeroModel(FaceRecognitionMock):
        def forward(self, x):
            return torch.zeros((x.size(0), 5))  # Predict all zeros

    # Create data loader
    dataset = TensorDataset(features, labels)
    loader = DataLoader(dataset, batch_size=5)

    # Evaluate
    model = AlwaysZeroModel()
    evaluator = ModelEvaluator(model, loader, mock_config)
    results = evaluator.evaluate()

    # Should have 20% accuracy (2 correct out of 10)
    # (Only class 0 predictions will match the first and sixth samples)
    assert results["accuracy"] == 20.0


def test_embedding_collection(mock_model, test_loader, mock_config):
    """Test embeddings are collected correctly"""
    evaluator = ModelEvaluator(mock_model, test_loader, mock_config)
    results = evaluator.evaluate()

    # Check embeddings have expected size
    embeddings = results["embeddings"]
    assert embeddings.dim() == 2
    assert embeddings.size(0) == 100  # 100 samples
    assert embeddings.size(1) == 128  # 128-dim embeddings

    # Check labels match dataset size
    assert results["labels"].size(0) == 100
