import pytest
import torch
from torch import nn, optim

from src.utils.scheduler import get_scheduler


class TestConfig:
    """Test configuration stub"""

    LR_PATIENCE = 3


@pytest.fixture
def test_optimizer():
    """Fixture providing a test optimizer"""
    model = nn.Linear(10, 2)
    return optim.Adam(model.parameters(), lr=0.001)


@pytest.fixture
def test_config():
    """Fixture providing a test configuration"""
    return TestConfig()


def test_scheduler_creation(test_optimizer, test_config):
    """
    Test that scheduler is created with correct type and attached to optimizer
    """
    scheduler = get_scheduler(test_optimizer, test_config)

    assert isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau)
    assert scheduler.optimizer == test_optimizer


def test_scheduler_parameters(test_optimizer, test_config):
    """
    Test that scheduler has correct parameter values
    """
    scheduler = get_scheduler(test_optimizer, test_config)

    assert scheduler.mode == "max"
    assert scheduler.factor == 0.5
    assert scheduler.patience == test_config.LR_PATIENCE
    assert scheduler.min_lrs[0] == 1e-6
    assert scheduler.verbose is True


def test_scheduler_with_custom_patience(test_optimizer):
    """
    Test scheduler with custom patience value
    """
    custom_config = TestConfig()
    custom_config.LR_PATIENCE = 5

    scheduler = get_scheduler(test_optimizer, custom_config)
    assert scheduler.patience == 5


def test_scheduler_with_multiple_param_groups(test_optimizer, test_config):
    """
    Test scheduler works with optimizer having multiple parameter groups
    """
    test_optimizer.add_param_group({"params": [nn.Parameter(torch.randn(10, 2))], "lr": 0.01})

    scheduler = get_scheduler(test_optimizer, test_config)
    assert len(scheduler.min_lrs) == 2


def test_scheduler_step_operation(test_optimizer, test_config):
    """
    Test that scheduler step works correctly with 'max' mode
    """
    scheduler = get_scheduler(test_optimizer, test_config)

    initial_lr = test_optimizer.param_groups[0]["lr"]

    scheduler.step(0.9)
    for _ in range(test_config.LR_PATIENCE + 1):
        scheduler.step(0.1)

    assert test_optimizer.param_groups[0]["lr"] == pytest.approx(initial_lr * 0.5)
