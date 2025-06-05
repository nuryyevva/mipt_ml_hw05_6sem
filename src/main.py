import os

import torch

import wandb
from config import Config
from data.data_loader import get_lfw_dataloaders
from src.model.face_recognition import FaceRecognitionModel
from src.train import Trainer
from src.utils.evaluator import ModelEvaluator
from src.utils.logger import setup_logger


def main() -> None:
    """Main function to execute the face recognition pipeline.

    The pipeline includes:
    - Loading configuration
    - Preparing data
    - Initializing the model
    - Training or loading a pre-trained model
    - Evaluating the model

    :returns: None
    """
    logger = setup_logger("Main")

    # Load configuration
    config = Config()
    logger.info("Configuration loaded")

    # Prepare data
    train_loader, val_loader, test_loader, target_names = get_lfw_dataloaders(
        batch_size=config.BATCH_SIZE,
        data_dir=config.DATA_DIR,
        min_faces=config.MIN_FACES_PER_PERSON,
        resize=config.RESIZE_RATIO,
    )

    logger.info(f"Number of classes: {len(target_names)}")
    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Validation samples: {len(val_loader.dataset)}")
    logger.info(f"Test samples: {len(test_loader.dataset)}")

    num_classes = len(target_names)

    # Initialize model
    model = FaceRecognitionModel(num_classes=num_classes)
    logger.info("Model initialized")

    # Train model
    if not os.path.exists(config.MODEL_SAVE_PATH):
        logger.info("Starting training...")
        trainer = Trainer(model, train_loader, val_loader, config)
        trainer.train()
    else:
        logger.info("Loading pre-trained model...")
        model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE))

    # Evaluate model
    logger.info("Starting evaluation...")
    evaluator = ModelEvaluator(model, test_loader, config)
    evaluator.evaluate()

    # Log test metrics to WandB
    if config.USE_WANDB:
        wandb.init(
            project=config.WANDB_PROJECT,
            entity=config.WANDB_ENTITY,
            name=f"{config.WANDB_RUN_NAME}-eval" if config.WANDB_RUN_NAME else None,
            tags=config.WANDB_TAGS + ["evaluation"],
            config=config.__dict__,
        )

        wandb.log({"test/acc": eval_results["accuracy"], "test/num_samples": len(test_loader.dataset)})

        # Confusion matrix
        wandb.log(
            {
                "conf_mat": wandb.plot.confusion_matrix(
                    y_true=eval_results["labels"].numpy(),
                    preds=eval_results["predictions"].numpy(),
                    class_names=target_names,
                )
            }
        )

        wandb.finish()


if __name__ == "__main__":
    main()
