from typing import Optional

import torch
import torch.nn as nn

from .base import Representation


class EmbeddingRepresentation(Representation, nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        padding_idx: Optional[int] = None
    ):
        super().__init__()
        self.embeddings = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )

    @property
    def dim(self):
        return self.embeddings.embedding_dim

    def forward(self, inputs):
        inputs = torch.as_tensor(inputs)
        return self.embeddings(inputs)
