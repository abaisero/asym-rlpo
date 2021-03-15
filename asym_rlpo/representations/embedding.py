import torch
import torch.nn as nn

from .base import Representation


class EmbeddingRepresentation(Representation, nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)

    @property
    def dim(self):
        return self.embeddings.embedding_dim

    def __call__(self, inputs):
        inputs = torch.as_tensor(inputs)
        return self.embeddings(inputs)
