from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import transforms

from src.config import Config
from src.model.face_recognition import FaceRecognitionModel


class FaceVerifier:
    """Face verification system using cosine similarity of embeddings."""

    def __init__(self, model_path: str, config: Config) -> None:
        """
        Initialize face verifier with model and configuration.

        :params model_path: Path to saved model weights
        :type model_path: str

        :params config: Configuration object containing:
            - DEVICE: torch.device - Device for model execution
            - NUM_CLASSES: Optional[int] - Number of classes in classifier
            - VERIFICATION_THRESHOLD: float - Default similarity threshold
        :type config: Config
        """
        self.config = config
        self.device = config.DEVICE
        self.num_classes = config.NUM_CLASSES
        self.model = self.load_model(model_path)
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        )

    def load_model(self, model_path: str) -> FaceRecognitionModel:
        """
        Load model weights with handling for classifier mismatch.

        :params model_path: Path to saved model weights
        :type model_path: str

        :return: Initialized face recognition model
        :rtype: FaceRecognitionModel

        :note:
            - Handles classifier size mismatch by loading backbone only
            - Uses strict=False for state dict loading
            - Moves model to configured device
        """
        # Create model with the correct number of classes
        model = FaceRecognitionModel(num_classes=self.num_classes or 1)

        # Load state dict, ignoring classifier mismatch
        state_dict = torch.load(model_path, map_location=self.device)

        # Remove classifier weights if num_classes doesn't match
        if self.num_classes is not None:
            # Get classifier keys
            classifier_keys = [k for k in state_dict.keys() if k.startswith("classifier")]
            current_classifier_keys = [k for k in model.state_dict().keys() if k.startswith("classifier")]

            # Check if classifier sizes match
            if (
                len(classifier_keys) > 0
                and state_dict[classifier_keys[0]].shape != model.state_dict()[current_classifier_keys[0]].shape
            ):
                print("Warning: Classifier size mismatch. Using backbone only for embeddings.")
                for key in classifier_keys:
                    del state_dict[key]

        # Load state dict with strict=False to ignore mismatched classifier
        model.load_state_dict(state_dict, strict=False)
        model.to(self.device)
        model.eval()
        return model

    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """
        Preprocess image for model input.

        :params image_path: Path to input image
        :type image_path: str

        :return: Preprocessed image tensor
        :rtype: torch.Tensor

        :note:
            - Performs padding to square aspect ratio
            - Resizes to 160x160 pixels
            - Applies ImageNet normalization
        """
        img = Image.open(image_path).convert("RGB")
        width, height = img.size
        max_dim = max(width, height)
        pad_width = (max_dim - width) // 2
        pad_height = (max_dim - height) // 2
        new_img = Image.new("RGB", (max_dim, max_dim), (0, 0, 0))
        new_img.paste(img, (pad_width, pad_height))
        img = new_img.resize((160, 160), Image.BILINEAR)
        return self.transform(img).unsqueeze(0).to(self.device)

    def get_embedding(self, image_path: str) -> np.ndarray:
        """
        Extract face embedding from image.

        :params image_path: Path to input image
        :type image_path: str

        :return: Face embedding vector
        :rtype: numpy.ndarray

        :note:
            - Uses model backbone only
            - Returns CPU numpy array
        """
        img_tensor = self.preprocess_image(image_path)
        with torch.no_grad():
            # Directly use backbone for embeddings
            embedding = self.model.backbone(img_tensor)
        return embedding.cpu().numpy()

    def verify(self, anchor_path: str, test_paths: List[str], threshold: Optional[float] = None) -> List[Dict]:
        """
        Verify face similarity against anchor image.

        :params anchor_path: Path to anchor/reference image
        :type anchor_path: str

        :params test_paths: List of paths to test images
        :type test_paths: List[str]

        :params threshold: Optional similarity threshold (uses config default if None)
        :type threshold: Optional[float]

        :return: List of verification results with:
            - path: str - Test image path
            - similarity: float - Cosine similarity score
            - is_same: bool - Verification decision
                :rtype: List[Dict]
        """
        if threshold is None:
            threshold = self.config.VERIFICATION_THRESHOLD

        anchor_embed = self.get_embedding(anchor_path)
        results = []

        for test_path in test_paths:
            test_embed = self.get_embedding(test_path)
            similarity = cosine_similarity(anchor_embed, test_embed)[0][0]
            results.append({"path": test_path, "similarity": similarity, "is_same": similarity > threshold})

        return results

    def evaluate_verification(self, results: List[Dict], expected_labels: List[bool]) -> float:
        """
        Calculate verification accuracy.

        :params results: Verification results from verify() method
        :type results: List[Dict]

        :params expected_labels: Ground truth labels (True/False for match)
        :type expected_labels: List[bool]

        :return: Accuracy score between 0 and 1
        :rtype: float
        """
        correct = 0
        total = len(results)

        for result, expected in zip(results, expected_labels):
            if result["is_same"] == expected:
                correct += 1

        return correct / total
