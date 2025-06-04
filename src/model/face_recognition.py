import torch.nn as nn
from facenet_pytorch import InceptionResnetV1


class FaceRecognitionModel(nn.Module):
    def __init__(self, num_classes, embedding_size=512, dropout_rate=0.5):
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

    def forward(self, x, return_embeddings=False):
        embeddings = self.backbone(x)
        if return_embeddings:
            return embeddings
        return self.classifier(embeddings)

    def get_embeddings(self, x):
        return self.forward(x, return_embeddings=True)
