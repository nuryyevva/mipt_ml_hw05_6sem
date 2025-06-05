import os
from datetime import datetime
from typing import Tuple, TypeVar

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.utils.optimizer import get_optimizer
from src.utils.scheduler import get_scheduler

# Type variable for model
Model = TypeVar("Model", bound=nn.Module)


class Trainer:
    """Handles model training and validation process.

    :param model: The neural network model to train
    :param train_loader: DataLoader for training data
    :param val_loader: DataLoader for validation data
    :param config: Configuration object with training parameters
    """

    def __init__(self, model: Model, train_loader: DataLoader, val_loader: DataLoader, config: object) -> None:
        """Initialize trainer with model, data loaders and configuration."""
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = config.DEVICE
        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = get_optimizer(model, config)
        self.scheduler = get_scheduler(self.optimizer, config)

        # Setup logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.join(config.LOG_DIR, f"run_{timestamp}")
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)

        self.best_val_acc = 0.0

    def train_epoch(self) -> Tuple[float, float]:
        """Train model for one epoch.

        :returns: Tuple containing:
            - epoch_loss: Average loss for the epoch
            - epoch_acc: Accuracy for the epoch (percentage)
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc="Training")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({"loss": running_loss / (total / labels.size(0)), "acc": 100.0 * correct / total})

        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100.0 * correct / total
        return epoch_loss, epoch_acc

    def validate(self) -> Tuple[float, float]:
        """Validate model on validation set.

        :returns: Tuple containing:
            - val_loss: Average loss on validation set
            - val_acc: Accuracy on validation set (percentage)
        """
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in tqdm(self.val_loader, desc="Validation"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_loss /= len(self.val_loader)
        val_acc = 100.0 * correct / total
        return val_loss, val_acc

    def train(self) -> None:
        """Run full training process including multiple epochs.

        Performs:
        - Training for specified number of epochs
        - Validation after each epoch
        - Logging metrics to TensorBoard
        - Model saving when validation accuracy improves
        """
        for epoch in range(self.config.NUM_EPOCHS):
            print(f"\nEpoch {epoch + 1}/{self.config.NUM_EPOCHS}")

            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()

            # Log metrics
            self.writer.add_scalar("Loss/train", train_loss, epoch)
            self.writer.add_scalar("Accuracy/train", train_acc, epoch)
            self.writer.add_scalar("Loss/val", val_loss, epoch)
            self.writer.add_scalar("Accuracy/val", val_acc, epoch)

            # Update scheduler
            self.scheduler.step(val_acc)

            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                torch.save(self.model.state_dict(), self.config.MODEL_SAVE_PATH)
                print(f"Saved new best model with val acc: {val_acc:.2f}%")

            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        self.writer.close()
        print("Training completed!")
