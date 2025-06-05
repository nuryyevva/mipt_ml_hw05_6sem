import pytest
import torch.nn as nn
from torch import optim

from src.utils.optimizer import get_optimizer


class TestModel(nn.Module):
    """Test model with backbone and classifier layers"""

    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(10, 20), nn.ReLU())
        self.classifier = nn.Linear(20, 2)

    def forward(self, x):
        x = self.backbone(x)
        return self.classifier(x)


@pytest.fixture
def test_config():
    """A fixture with a test configuration"""

    class TestConfig:
        LEARNING_RATE = 0.001
        WEIGHT_DECAY = 1e-5

    return TestConfig()


def test_optimizer_creation(test_config):
    """A test for creating an optimizer with the correct parameter groups"""
    model = TestModel()
    optimizer = get_optimizer(model, test_config)

    assert isinstance(optimizer, optim.Adam)
    assert len(optimizer.param_groups) == 2


def test_learning_rates(test_config):
    """Learning rate installation test"""
    model = TestModel()
    optimizer = get_optimizer(model, test_config)

    assert optimizer.param_groups[0]["lr"] == pytest.approx(test_config.LEARNING_RATE * 0.1)
    assert optimizer.param_groups[1]["lr"] == pytest.approx(test_config.LEARNING_RATE)


def test_parameter_groups(test_config):
    """The test of the correctness of the distribution of parameters by groups"""
    model = TestModel()
    optimizer = get_optimizer(model, test_config)

    assert len(optimizer.param_groups[0]["params"]) == 2
    assert len(optimizer.param_groups[1]["params"]) == 2


def test_weight_decay(test_config):
    """Weight decay application test"""
    model = TestModel()
    optimizer = get_optimizer(model, test_config)

    assert optimizer.param_groups[0]["weight_decay"] == test_config.WEIGHT_DECAY
    assert optimizer.param_groups[1]["weight_decay"] == test_config.WEIGHT_DECAY


def test_frozen_parameters(test_config):
    """The test verifies that frozen parameters do not fall into groups."""
    model = TestModel()

    for param in model.classifier.parameters():
        param.requires_grad = False

    optimizer = get_optimizer(model, test_config)

    assert len(optimizer.param_groups) == 2
    assert len(optimizer.param_groups[0]["params"]) == 2
    assert len(optimizer.param_groups[1]["params"]) == 0


def test_empty_model(test_config):
    """Test with a model without parameters"""
    model = nn.Module()
    optimizer = get_optimizer(model, test_config)

    assert len(optimizer.param_groups) == 2
    assert len(optimizer.param_groups[0]["params"]) == 0
    assert len(optimizer.param_groups[1]["params"]) == 0
