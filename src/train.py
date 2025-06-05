import os
from datetime import datetime
from typing import Tuple, TypeVar

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import wandb
from src.utils.logger import setup_logger
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
        self.logger = setup_logger("Trainer")

        # Initialize wandb if enabled
        if config.USE_WANDB:
            wandb.init(
                project=config.WANDB_PROJECT,
                entity=config.WANDB_ENTITY,
                name=config.WANDB_RUN_NAME or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                tags=config.WANDB_TAGS,
                config=config.__dict__,
            )
            wandb.watch(self.model, log="all", log_freq=10)
            self.logger.info(f"Initialized WandB run: {wandb.run.url}")

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = get_optimizer(model, config)
        self.scheduler = get_scheduler(self.optimizer, config)

        # Setup TensorBoard
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.join(config.LOG_DIR, f"run_{timestamp}")
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)
        self.logger.info(f"TensorBoard logs at: {self.log_dir}")

        self.best_val_acc = 0.0

    def train_epoch(self, epoch) -> Tuple[float, float]:
        """Train model for one epoch.

        :returns: Tuple containing:
            - epoch_loss: Average loss for the epoch
            - epoch_acc: Accuracy for the epoch (percentage)
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.config.NUM_EPOCHS} [Train]")
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

            current_loss = running_loss / (total / labels.size(0))
            current_acc = 100.0 * correct / total
            pbar.set_postfix({"loss": f"{current_loss:.4f}", "acc": f"{current_acc:.2f}%"})

        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100.0 * correct / total
        return epoch_loss, epoch_acc

    def validate(self, epoch) -> Tuple[float, float]:
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
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch + 1}/{self.config.NUM_EPOCHS} [Val]")
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                current_loss = val_loss / (total / labels.size(0))
                current_acc = 100.0 * correct / total
                pbar.set_postfix({"loss": f"{current_loss:.4f}", "acc": f"{current_acc:.2f}%"})

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
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate(epoch)

            # Log epoch metrics to TensorBoard
            self.writer.add_scalar("Loss/train", train_loss, epoch)
            self.writer.add_scalar("Accuracy/train", train_acc, epoch)
            self.writer.add_scalar("Loss/val", val_loss, epoch)
            self.writer.add_scalar("Accuracy/val", val_acc, epoch)

            # Log epoch metrics to WandB
            if self.config.USE_WANDB:
                wandb.log(
                    {
                        "epoch": epoch,
                        "train/loss": train_loss,
                        "train/acc": train_acc,
                        "val/loss": val_loss,
                        "val/acc": val_acc,
                        "lr": self.optimizer.param_groups[0]["lr"],
                    }
                )

            # Update scheduler
            self.scheduler.step(val_acc)

            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                torch.save(self.model.state_dict(), self.config.MODEL_SAVE_PATH)
                self.logger.info(f"Saved new best model with val acc: {val_acc:.2f}%")

                # Save to WandB
                if self.config.USE_WANDB:
                    artifact = wandb.Artifact(f"model-epoch{epoch}", type="model")
                    artifact.add_file(self.config.MODEL_SAVE_PATH)
                    wandb.log_artifact(artifact)

            print(f"\nEpoch {epoch + 1}/{self.config.NUM_EPOCHS}")
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        # Finalize
        self.writer.close()
        if self.config.USE_WANDB:
            wandb.finish()
        self.logger.info("Training completed!")
