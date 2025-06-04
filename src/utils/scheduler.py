from torch import optim


def get_scheduler(optimizer, config):
    """Create learning rate scheduler"""
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=config.LR_PATIENCE, verbose=True, min_lr=1e-6
    )
