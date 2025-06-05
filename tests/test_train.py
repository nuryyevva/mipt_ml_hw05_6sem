import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import Config
from src.train import Trainer


class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)


@pytest.fixture
def setup():
    X_train = torch.randn(100, 10)
    y_train = torch.randint(0, 2, (100,))
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=10)

    X_val = torch.randn(20, 10)
    y_val = torch.randint(0, 2, (20,))
    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=10)

    model = TestModel()
    config = Config()

    yield model, train_loader, val_loader, config

    if os.path.exists(config.LOG_DIR):
        for root, dirs, files in os.walk(config.LOG_DIR, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(config.LOG_DIR)

    if os.path.exists(config.MODEL_SAVE_PATH):
        os.remove(config.MODEL_SAVE_PATH)


def test_trainer_initialization(setup):
    model, train_loader, val_loader, config = setup

    trainer = Trainer(model, train_loader, val_loader, config)

    assert trainer.model == model
    assert trainer.train_loader == train_loader
    assert trainer.val_loader == val_loader
    assert trainer.config == config
    assert trainer.device == config.DEVICE
    assert isinstance(trainer.criterion, nn.CrossEntropyLoss)
    assert trainer.optimizer is not None
    assert trainer.scheduler is not None
    assert os.path.exists(config.LOG_DIR)
    assert trainer.best_val_acc == 0.0


def test_train_epoch(setup):
    model, train_loader, val_loader, config = setup

    trainer = Trainer(model, train_loader, val_loader, config)

    original_forward = model.forward

    def mock_forward(x):
        outputs = torch.zeros(x.size(0), 2, device=x.device)
        outputs[:, 0] = 1.0
        outputs.requires_grad_(True)
        return outputs

    model.forward = mock_forward

    try:
        loss, acc = trainer.train_epoch(1)

        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert 0 <= acc <= 100
    finally:
        model.forward = original_forward


def test_validate(setup):
    model, train_loader, val_loader, config = setup

    trainer = Trainer(model, train_loader, val_loader, config)

    original_forward = model.forward

    def mock_forward(x):
        outputs = torch.zeros(x.size(0), 2)
        outputs[:, 0] = 1.0
        return outputs

    model.forward = mock_forward

    try:
        loss, acc = trainer.validate(1)

        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert 0 <= acc <= 100
    finally:
        model.forward = original_forward


@patch("torch.save")
@patch("torch.utils.tensorboard.SummaryWriter")
def test_train(mock_writer, mock_save, setup):
    model, train_loader, val_loader, config = setup
    config.NUM_EPOCHS = 5
    config.MODEL_SAVE_PATH = Path(__file__).parent.parent / "model/face_recognition_model_01.pth"

    mock_writer_instance = MagicMock()
    mock_writer_instance.add_scalar = MagicMock()
    mock_writer.return_value = mock_writer_instance

    trainer = Trainer(model, train_loader, val_loader, config)

    with (
        patch.object(trainer, "train_epoch", return_value=(0.5, 80.0)),
        patch.object(
            trainer, "validate", side_effect=[(0.6, 70.0), (0.55, 75.0), (0.5, 80.0), (0.45, 85.0), (0.4, 90.0)]
        ),
    ):
        print(f"Before train - writer mock: {mock_writer_instance}")
        print(f"Before train - add_scalar mock: {mock_writer_instance.add_scalar}")

        trainer.train()

        print(f"After train - add_scalar calls: {mock_writer_instance.add_scalar.call_args_list}")

        assert trainer.train_epoch.call_count == config.NUM_EPOCHS
        assert trainer.validate.call_count == config.NUM_EPOCHS


def test_device_handling(setup):
    model, train_loader, val_loader, config = setup

    config.DEVICE = "cpu"
    trainer = Trainer(model, train_loader, val_loader, config)

    assert next(trainer.model.parameters()).device.type == "cpu"

    sample_input = torch.randn(1, 10)
    sample_output = trainer.model(sample_input.to(config.DEVICE))
    assert sample_output.device.type == "cpu"


def test_scheduler_step(setup):
    model, train_loader, val_loader, config = setup
    os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)

    trainer = Trainer(model, train_loader, val_loader, config)

    original_step = trainer.scheduler.step
    mock_step = MagicMock()
    trainer.scheduler.step = mock_step

    try:
        trainer.validate = MagicMock(return_value=(0.5, 80.0))
        trainer.train()

        assert mock_step.call_count == config.NUM_EPOCHS
    finally:
        trainer.scheduler.step = original_step
        if os.path.exists(config.MODEL_SAVE_PATH):
            os.remove(config.MODEL_SAVE_PATH)
