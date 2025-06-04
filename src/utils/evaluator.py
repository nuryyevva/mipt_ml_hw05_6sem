import torch
from tqdm import tqdm


class ModelEvaluator:
    def __init__(self, model, test_loader, config):
        self.model = model
        self.test_loader = test_loader
        self.config = config
        self.device = config.DEVICE
        self.model.to(self.device)
        self.model.eval()

    def evaluate(self):
        correct = 0
        total = 0
        all_embeddings = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in tqdm(self.test_loader, desc="Evaluating"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Get embeddings for verification analysis
                embeddings = self.model.get_embeddings(inputs)
                all_embeddings.append(embeddings.cpu())
                all_labels.append(labels.cpu())

        accuracy = 100.0 * correct / total
        print(f"Test Accuracy: {accuracy:.2f}%")

        return {
            "accuracy": accuracy,
            "embeddings": torch.cat(all_embeddings, dim=0),
            "labels": torch.cat(all_labels, dim=0),
        }
