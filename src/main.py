import os

import torch

from config import Config
from data.data_loader import get_lfw_dataloaders
from src.model.face_recognition import FaceRecognitionModel
from src.train import Trainer
from src.utils.evaluator import ModelEvaluator


def main():
    # Load configuration
    config = Config()

    # Prepare data
    train_loader, val_loader, test_loader, target_names = get_lfw_dataloaders(
        batch_size=config.BATCH_SIZE,
        data_dir=config.DATA_DIR,
        min_faces=config.MIN_FACES_PER_PERSON,
        resize=config.RESIZE_RATIO,
    )

    print(f"Number of classes: {len(target_names)}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    num_classes = len(target_names)

    # Initialize model
    model = FaceRecognitionModel(num_classes=num_classes)

    # Train model
    if not os.path.exists(config.MODEL_SAVE_PATH):
        print("Starting training...")
        trainer = Trainer(model, train_loader, val_loader, config)
        trainer.train()
    else:
        print("Loading pre-trained model...")
        model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE))

    # Evaluate model
    evaluator = ModelEvaluator(model, test_loader, config)
    eval_results = evaluator.evaluate()


if __name__ == "__main__":
    main()
