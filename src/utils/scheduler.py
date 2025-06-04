from torch import optim

from src.config import Config


def get_scheduler(optimizer: optim.Optimizer, config: Config) -> optim.lr_scheduler.ReduceLROnPlateau:
    """
    Create learning rate scheduler (ReduceLROnPlateau) for optimizer.

    :params optimizer: Optimizer instance to attach scheduler to
    :type optimizer: torch.optim.Optimizer

    :params config: Configuration object containing training parameters
    :type config: Config
        - LR_PATIENCE: int - Number of epochs to wait before reducing LR when no improvement

    :return: Configured ReduceLROnPlateau learning rate scheduler
    :rtype: torch.optim.lr_scheduler.ReduceLROnPlateau

    :note:
        - Mode: 'max' (monitors maximum metric value)
        - Reduction factor: 0.5 (cuts LR in half)
        - Minimum LR: 1e-6 (lower bound)
        - Uses config.LR_PATIENCE for patience setting
        - Verbose: True (prints update messages)
    """
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=config.LR_PATIENCE, verbose=True, min_lr=1e-6
    )
