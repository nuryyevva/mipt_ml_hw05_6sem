from unittest.mock import patch

import pytest
import torch
from torch import nn

from src.model.face_recognition import FaceRecognitionModel


@pytest.fixture
def test_model():
    """Fixture providing a test model with 10 classes"""
    return FaceRecognitionModel(num_classes=10)


def test_initialization(test_model):
    """Test model initialization and architecture"""
    assert isinstance(test_model, nn.Module)
    assert isinstance(test_model.backbone, nn.Module)
    assert isinstance(test_model.classifier, nn.Sequential)

    classifier_layers = list(test_model.classifier.children())
    assert isinstance(classifier_layers[0], nn.Linear)
    assert isinstance(classifier_layers[1], nn.BatchNorm1d)
    assert isinstance(classifier_layers[2], nn.ReLU)
    assert isinstance(classifier_layers[3], nn.Dropout)
    assert isinstance(classifier_layers[4], nn.Linear)


def test_parameter_freezing(test_model):
    """Test that backbone parameters are frozen except last layers"""
    for name, param in test_model.backbone.named_parameters():
        if "last_linear" in name or "last_bn" in name:
            assert param.requires_grad
        else:
            assert not param.requires_grad

    for param in test_model.classifier.parameters():
        assert param.requires_grad


def test_forward_pass(test_model):
    """Test forward pass with classification output"""
    x = torch.randn(2, 3, 160, 160)

    output = test_model(x)
    assert output.shape == (2, 10)


def test_forward_pass_with_embeddings(test_model):
    """Test forward pass with embeddings output"""
    x = torch.randn(2, 3, 160, 160)

    embeddings = test_model(x, return_embeddings=True)
    assert embeddings.shape == (2, 512)


def test_get_embeddings(test_model):
    """Test get_embeddings shortcut method"""
    x = torch.randn(2, 3, 160, 160)

    with patch.object(test_model, "forward") as mock_forward:
        mock_forward.return_value = torch.randn(2, 512)
        embeddings = test_model.get_embeddings(x)

        mock_forward.assert_called_once_with(x, return_embeddings=True)
        assert embeddings.shape == (2, 512)


def test_classifier_output(test_model):
    """Test classifier output dimensions with different num_classes"""
    model = FaceRecognitionModel(num_classes=100)
    x = torch.randn(2, 3, 160, 160)

    output = model(x)
    assert output.shape == (2, 100)


def test_backbone_output(test_model):
    """Test backbone output before classifier"""
    x = torch.randn(2, 3, 160, 160)

    embeddings = test_model.backbone(x)
    assert embeddings.shape == (2, 512)

    output = test_model.classifier(embeddings)
    assert output.shape == (2, 10)


def test_dropout_behavior(test_model):
    """Test dropout behavior during train/eval modes"""
    x = torch.randn(2, 3, 160, 160)

    test_model.eval()
    embeddings1 = test_model.backbone(x)
    output1 = test_model.classifier(embeddings1)

    embeddings2 = test_model.backbone(x)
    output2 = test_model.classifier(embeddings2)

    assert torch.allclose(output1, output2)

    test_model.train()
    output3 = test_model(x)
    output4 = test_model(x)
    assert not torch.allclose(output3, output4)
