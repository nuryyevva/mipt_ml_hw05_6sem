from torch import optim


def get_optimizer(model, config):
    """Create optimizer with different learning rates for backbone and classifier"""
    # Different learning rates for backbone and classifier
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
