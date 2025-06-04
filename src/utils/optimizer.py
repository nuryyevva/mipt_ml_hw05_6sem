from torch import nn, optim

from src.config import Config


def get_optimizer(model: nn.Module, config: Config) -> optim.Adam:
    """
    Create optimizer with different learning rates for backbone and classifier.

    :params model: PyTorch model containing backbone and classifier parameters.
                  Parameters are identified by having "backbone" or "classifier" in their name.
    :params config: Configuration object containing:
                   - LEARNING_RATE: Base learning rate (used for classifier)
                   - WEIGHT_DECAY: Weight decay coefficient for L2 regularization

    :return: Configured Adam optimizer with separate learning rates for different parts of the model.
    """
    backbone_params = []
    classifier_params = []

    for name, param in model.named_parameters():
        if "backbone" in name and param.requires_grad:
            backbone_params.append(param)
        elif "classifier" in name and param.requires_grad:
            classifier_params.append(param)

    optimizer = optim.Adam(
        [
            {"params": backbone_params, "lr": config.LEARNING_RATE * 0.1},
            {"params": classifier_params, "lr": config.LEARNING_RATE},
        ],
        weight_decay=config.WEIGHT_DECAY,
    )
    return optimizer
