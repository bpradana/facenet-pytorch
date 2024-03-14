import torch.nn as nn
from torch.nn import functional as F
from .utils_ghostnet import ghostnet


class GhostNetTriplet(nn.Module):
    """Constructs a GhostNet model for FaceNet training using triplet loss.

    Args:
        embedding_dimension (int): Required dimension of the resulting embedding layer that is outputted by the model.
                                    using triplet loss. Defaults to 512.
        pretrained (bool): If True, returns a model pre-trained on the ImageNet dataset from a PyTorch repository.
                            Defaults to False.
    """

    def __init__(self, embedding_dimension=512, pretrained=False):
        super(GhostNetTriplet, self).__init__()
        self.model = ghostnet(pretrained=pretrained)

        # Output embedding
        self.model.classifier = nn.Linear(
            embedding_dimension,
            embedding_dimension,
            bias=False,
        )

    def forward(self, images):
        """Forward pass to output the embedding vector (feature vector) after l2-normalization."""
        embedding = self.model(images)
        # From: https://github.com/timesler/facenet-pytorch/blob/master/models/inception_resnet_v1.py#L301
        embedding = F.normalize(embedding, p=2, dim=1)

        return embedding
