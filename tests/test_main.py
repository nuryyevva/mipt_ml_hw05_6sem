from unittest.mock import MagicMock, patch

import pytest
from torch.utils.data import DataLoader

from src.config import Config
from src.main import main


@pytest.fixture
def mock_dataloaders():
    """A fixture for moking the return values of get_lfw_dataloaders."""
    train_loader = MagicMock(spec=DataLoader)
    train_loader.dataset = MagicMock()
    train_loader.dataset.__len__.return_value = 100

    val_loader = MagicMock(spec=DataLoader)
    val_loader.dataset = MagicMock()
    val_loader.dataset.__len__.return_value = 20

    test_loader = MagicMock(spec=DataLoader)
    test_loader.dataset = MagicMock()
    test_loader.dataset.__len__.return_value = 30

    target_names = ["person1", "person2", "person3"]

    return train_loader, val_loader, test_loader, target_names


@pytest.fixture
def mock_lfw_dataset():
    """A fixture for mopping the fetch_lfw_people return value."""
    mock = MagicMock()
    mock.data = MagicMock()
    mock.target = MagicMock()
    mock.target_names = ["person1", "person2", "person3"]
    mock.images = MagicMock()
    mock.DESCR = ""
    return mock


def test_main_with_training(mock_dataloaders, mock_lfw_dataset):
    """The main() test is when the model needs to be trained (there is no model file)."""
    train_loader, val_loader, test_loader, target_names = mock_dataloaders

    with (
        patch("src.main.Config", return_value=Config),
        patch("src.main.get_lfw_dataloaders", return_value=(train_loader, val_loader, test_loader, target_names)),
        patch("src.main.FaceRecognitionModel") as mock_model,
        patch("src.main.Trainer") as mock_trainer,
        patch("src.main.ModelEvaluator") as mock_evaluator,
        patch("os.path.exists", return_value=False),
        patch("os.makedirs"),
        patch("torch.save"),
        patch("torch.utils.tensorboard.SummaryWriter"),
        patch("sklearn.datasets.fetch_lfw_people", return_value=mock_lfw_dataset),
    ):
        # Configure mock model
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance

        main()

        mock_model.assert_called_once_with(num_classes=len(target_names))
        mock_trainer.assert_called_once()
        mock_trainer.return_value.train.assert_called_once()
        mock_evaluator.assert_called_once()
        mock_evaluator.return_value.evaluate.assert_called_once()


def test_main_with_pretrained(mock_dataloaders, mock_lfw_dataset):
    """The main() test is used when a pre-trained model is used (the model file exists)."""
    train_loader, val_loader, test_loader, target_names = mock_dataloaders

    with (
        patch("src.main.Config", return_value=Config),
        patch("src.main.get_lfw_dataloaders", return_value=(train_loader, val_loader, test_loader, target_names)),
        patch("src.main.FaceRecognitionModel") as mock_model,
        patch("src.main.Trainer"),
        patch("src.main.ModelEvaluator") as mock_evaluator,
        patch("os.path.exists", return_value=True),
        patch("torch.load") as mock_load,
        patch("os.makedirs"),
        patch("torch.utils.tensorboard.SummaryWriter"),
        patch("sklearn.datasets.fetch_lfw_people", return_value=mock_lfw_dataset),
    ):
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance

        # Create a proper mock state dict
        mock_state_dict = MagicMock()
        mock_load.return_value = mock_state_dict

        main()

        mock_model.assert_called_once_with(num_classes=len(target_names))
        mock_load.assert_called_once_with(Config.MODEL_SAVE_PATH, map_location=Config.DEVICE)
        mock_model_instance.load_state_dict.assert_called_once_with(mock_state_dict)
        mock_evaluator.assert_called_once()
        mock_evaluator.return_value.evaluate.assert_called_once()


def test_main_prints_correct_info(mock_dataloaders, mock_lfw_dataset, capsys):
    """The test is that main() correctly outputs information about the data."""
    train_loader, val_loader, test_loader, target_names = mock_dataloaders
    train_loader.dataset.__len__.return_value = 100
    val_loader.dataset.__len__.return_value = 20
    test_loader.dataset.__len__.return_value = 30

    with (
        patch("src.main.Config", return_value=Config),
        patch("src.main.get_lfw_dataloaders", return_value=(train_loader, val_loader, test_loader, target_names)),
        patch("src.main.FaceRecognitionModel"),
        patch("src.main.Trainer"),
        patch("src.main.ModelEvaluator"),
        patch("os.path.exists", return_value=False),
        patch("os.makedirs"),
        patch("torch.save"),
        patch("torch.utils.tensorboard.SummaryWriter"),
        patch("sklearn.datasets.fetch_lfw_people", return_value=mock_lfw_dataset),
    ):
        main()
        captured = capsys.readouterr()

        assert f"Number of classes: {len(target_names)}" in captured.out
        assert "Train samples: 100" in captured.out
        assert "Validation samples: 20" in captured.out
        assert "Test samples: 30" in captured.out
        assert "Starting training..." in captured.out


def test_main_with_existing_model_prints_message(mock_dataloaders, mock_lfw_dataset, capsys):
    """The test is that main() outputs a message when an existing model is loaded."""
    with (
        patch("src.main.Config", return_value=Config),
        patch("src.main.get_lfw_dataloaders", return_value=mock_dataloaders),
        patch("src.main.FaceRecognitionModel"),
        patch("src.main.Trainer"),
        patch("src.main.ModelEvaluator"),
        patch("os.path.exists", return_value=True),
        patch("torch.load", return_value=MagicMock()),
        patch("os.makedirs"),
        patch("torch.utils.tensorboard.SummaryWriter"),
        patch("sklearn.datasets.fetch_lfw_people", return_value=mock_lfw_dataset),
    ):
        main()
        captured = capsys.readouterr()

        assert "Loading pre-trained model..." in captured.out
