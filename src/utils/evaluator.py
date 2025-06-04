import torch
from tqdm import tqdm

from src.config import Config


class ModelEvaluator:
    """Evaluates a trained face recognition model on a test dataset.

    This class handles:
    - Moving the model to the appropriate device
    - Running inference on the test dataset
    - Calculating accuracy metrics
    - Collecting embeddings for further analysis

    :param model: Trained face recognition model
    :type model: torch.nn.Module
    :param test_loader: DataLoader for test dataset
    :type test_loader: torch.utils.data.DataLoader
    :param config: Configuration object containing settings
    :type config: Config
    """

    def __init__(self, model: torch.nn.Module, test_loader: torch.utils.data.DataLoader, config: Config) -> None:
        """Initializes the evaluator with model, data loader, and configuration."""
        self.model = model
        self.test_loader = test_loader
        self.config = config
        self.device = config.DEVICE

        # Ensure model is on the correct device and in evaluation mode
        self.model.to(self.device)
        self.model.eval()

    def evaluate(self) -> dict[str, torch.Tensor]:
        """Evaluates the model on the test dataset.

        Performs the following steps:
        1. Iterates through all batches in the test loader
        2. Computes model predictions
        3. Calculates classification accuracy
        4. Collects embeddings for potential verification tasks

        :return: Dictionary containing:
            - accuracy: Test accuracy percentage
            - embeddings: Tensor of all embeddings (num_samples, embedding_size)
            - labels: Tensor of corresponding labels
        :rtype: Dict[str, torch.Tensor]
        """
        correct = 0
        total = 0
        all_embeddings = []
        all_labels = []

        # Use inference mode for efficiency
        with torch.no_grad():
            # Wrap loader in progress bar for better UX
            for inputs, labels in tqdm(self.test_loader, desc="Evaluating"):
                # Move data to the same device as model
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # Forward pass - get class predictions
                outputs = self.model(inputs)

                # Get predicted classes (indices of max log-probability)
                _, predicted = torch.max(outputs, 1)

                # Update accuracy metrics
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Get embeddings for potential verification analysis
                # Note: Assumes model has get_embeddings() method
                embeddings = self.model.get_embeddings(inputs)
                all_embeddings.append(embeddings.cpu())  # Move to CPU to save GPU memory
                all_labels.append(labels.cpu())

        # Calculate final accuracy
        accuracy = 100.0 * correct / total
        print(f"Test Accuracy: {accuracy:.2f}%")

        return {
            "accuracy": accuracy,
            "embeddings": torch.cat(all_embeddings, dim=0),
            "labels": torch.cat(all_labels, dim=0),
        }
