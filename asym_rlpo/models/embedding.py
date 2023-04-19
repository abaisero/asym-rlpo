import torch
import torch.nn as nn

from asym_rlpo.models.model import FeatureModel


class EmbeddingModel(FeatureModel):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        padding_idx: int | None = None
    ):
        super().__init__()
        self.embeddings = nn.Embedding(
            num_embeddings,
            embedding_dim,
            padding_idx=padding_idx,
        )

    @property
    def dim(self):
        return self.embeddings.embedding_dim

    def forward(self, inputs):
        inputs = torch.as_tensor(inputs)
        return self.embeddings(inputs)

    def zeros_like(self, device: torch.device | None = None):
        features = self.embeddings(torch.tensor(0))
        return torch.zeros_like(features, device=device)
