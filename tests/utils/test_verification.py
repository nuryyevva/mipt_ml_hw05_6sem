from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

from src.utils.verification import FaceVerifier


@pytest.fixture
def test_config():
    """Fixture providing test configuration"""

    class TestConfig:
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        NUM_CLASSES = 10
        VERIFICATION_THRESHOLD = 0.85

    return TestConfig()


@pytest.fixture
def mock_model():
    """Fixture providing mock model"""
    model = MagicMock()
    model.backbone.return_value = torch.randn(1, 512)
    return model


@pytest.fixture
def face_verifier(test_config, mock_model):
    """Fixture providing FaceVerifier instance with mock model"""
    with patch("torch.load", return_value={}):
        with patch("src.model.face_recognition.FaceRecognitionModel", return_value=mock_model):
            verifier = FaceVerifier("dummy_path", test_config)
    return verifier


def test_initialization(face_verifier, test_config):
    """Test FaceVerifier initialization"""
    assert face_verifier.device == test_config.DEVICE
    assert face_verifier.num_classes == test_config.NUM_CLASSES
    assert face_verifier.config == test_config


def test_preprocess_image(face_verifier, tmp_path):
    """Test image preprocessing"""
    img_path = tmp_path / "test.jpg"
    Image.new("RGB", (100, 150), color="red").save(img_path)

    tensor = face_verifier.preprocess_image(str(img_path))
    assert tensor.shape == (1, 3, 160, 160)
    assert tensor.dtype == torch.float32


def test_get_embedding(face_verifier, tmp_path):
    """Test embedding extraction"""
    img_path = tmp_path / "test.jpg"
    Image.new("RGB", (160, 160), color="blue").save(img_path)

    embedding = face_verifier.get_embedding(str(img_path))
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (1, 512)


def test_verification(face_verifier, tmp_path):
    """Test face verification"""
    anchor_path = tmp_path / "anchor.jpg"
    test_path1 = tmp_path / "test1.jpg"
    test_path2 = tmp_path / "test2.jpg"

    Image.new("RGB", (160, 160), color="red").save(anchor_path)
    Image.new("RGB", (160, 160), color="red").save(test_path1)
    Image.new("RGB", (160, 160), color="blue").save(test_path2)

    results = face_verifier.verify(str(anchor_path), [str(test_path1), str(test_path2)])

    assert len(results) == 2
    assert all(isinstance(r, dict) for r in results)
    assert all("path" in r and "similarity" in r and "is_same" in r for r in results)


def test_evaluation_accuracy(face_verifier):
    """Test verification accuracy calculation"""
    test_results = [
        {"path": "1.jpg", "similarity": 0.9, "is_same": True},
        {"path": "2.jpg", "similarity": 0.8, "is_same": False},
        {"path": "3.jpg", "similarity": 0.7, "is_same": True},
    ]
    expected_labels = [True, False, False]

    accuracy = face_verifier.evaluate_verification(test_results, expected_labels)
    assert accuracy == pytest.approx(2 / 3)  # 2 correct out of 3


def test_empty_test_paths(face_verifier, tmp_path):
    """Test verification with empty test paths"""
    anchor_path = tmp_path / "anchor.jpg"
    Image.new("RGB", (160, 160), color="red").save(anchor_path)

    results = face_verifier.verify(str(anchor_path), [])
    assert len(results) == 0


def test_model_eval_mode(face_verifier):
    """Test that model is in eval mode"""
    assert not face_verifier.model.training
