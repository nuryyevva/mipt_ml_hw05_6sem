from typing import Tuple, Union

import torch.nn as nn
from facenet_pytorch import InceptionResnetV1
from torch import Tensor


class FaceRecognitionModel(nn.Module):
    """
    Face recognition model with frozen backbone and trainable classifier head.
    Combines InceptionResnetV1 backbone with customizable classifier.
    """

    def __init__(self, num_classes: int, embedding_size: int = 512, dropout_rate: float = 0.5) -> None:
        """
        Initialize face recognition model.

        :params num_classes: Number of output classes for classification
        :type num_classes: int

        :params embedding_size: Size of face embedding vector (default: 512)
        :type embedding_size: int

        :params dropout_rate: Dropout probability for classifier (default: 0.5)
        :type dropout_rate: float

        :note:
            - Backbone weights are frozen except last linear and batch norm layers
            - Classifier consists of FC -> BN -> ReLU -> Dropout -> FC
        """
        super().__init__()
        self.backbone = InceptionResnetV1(pretrained=None, classify=False)

        # Freeze backbone layers
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze last block
        for param in self.backbone.last_linear.parameters():
            param.requires_grad = True
        for param in self.backbone.last_bn.parameters():
            param.requires_grad = True

        self.classifier = nn.Sequential(
            nn.Linear(embedding_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: Tensor, return_embeddings: bool = False) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Forward pass of the model.

        :params x: Input image tensor
        :type x: torch.Tensor

        :params return_embeddings: Whether to return embeddings or class scores
        :type return_embeddings: bool

        :return: Either embeddings or class predictions
        :rtype: torch.Tensor or tuple(torch.Tensor, torch.Tensor)
        """
        embeddings = self.backbone(x)
        if return_embeddings:
            return embeddings
        return self.classifier(embeddings)

    def get_embeddings(self, x: Tensor) -> Tensor:
        """
        Extract face embeddings from input images.

        :params x: Input image tensor
        :type x: torch.Tensor

        :return: Face embedding vectors
        :rtype: torch.Tensor

        :note: Shortcut for forward(x, return_embeddings=True)
        """
        return self.forward(x, return_embeddings=True)
