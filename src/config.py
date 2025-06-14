import torch

from src.utils.device import setup_device


class Config:
    # Data configuration
    DATA_DIR = "./data"
    MIN_FACES_PER_PERSON = 100
    RESIZE_RATIO = None  # Original size

    # Model configuration
    EMBEDDING_SIZE = 512
    DROPOUT_RATE = 0.1

    # Training configuration
    BATCH_SIZE = 16
    NUM_EPOCHS = 30
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-5
    MOMENTUM = 0.9
    NUM_CLASSES = 5

    # Device configuration
    DEVICE = torch.device(setup_device())

    # Paths
    MODEL_SAVE_PATH = "./model/face_recognition_model_06.pth"
    LOG_DIR = "./logs"

    # Verification threshold
    VERIFICATION_THRESHOLD = 0.987

    # Learning rate scheduler settings
    LR_PATIENCE = 3  # Number of epochs to wait before reducing LR

    # WandB Configuration
    USE_WANDB = True
    WANDB_PROJECT = "face-recognition"
    WANDB_ENTITY = None  # WandB username
    WANDB_RUN_NAME = "hw05_face_recognition_model"  # Auto-generated if None
    WANDB_TAGS = ["hw05", "face-recognition"]
