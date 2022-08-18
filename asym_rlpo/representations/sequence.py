import abc
from typing import Generic, Optional, Tuple, TypeVar

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Representation
from .mlp import MLPRepresentation

H = TypeVar('H')


class SequenceRepresentation(Generic[H], Representation):
    @abc.abstractmethod
    def forward(
        self,
        input: torch.Tensor,
        hidden=Optional[H],
    ) -> Tuple[torch.Tensor, H]:
        assert False


class RNNSequenceRepresentation(SequenceRepresentation[torch.Tensor]):
    def __init__(self, in_features: int, out_features: int, **kwargs):
        super().__init__()
        self.rnn = nn.RNN(
            in_features,
            out_features,
            batch_first=True,
            **kwargs,
        )

    @property
    def dim(self) -> int:
        return self.rnn.hidden_size

    def forward(
        self,
        input: torch.Tensor,
        hidden=Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.rnn(input, hidden)


class GRUSequenceRepresentation(SequenceRepresentation[torch.Tensor]):
    def __init__(self, in_features: int, out_features: int, **kwargs):
        super().__init__()
        self.gru = nn.GRU(
            in_features,
            out_features,
            batch_first=True,
            **kwargs,
        )

    @property
    def dim(self) -> int:
        return self.gru.hidden_size

    def forward(
        self,
        input: torch.Tensor,
        hidden=Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.gru(input, hidden)


class AttentionSequenceRepresentation(SequenceRepresentation[torch.Tensor]):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int,
        **kwargs,
    ):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            in_features,
            num_heads,
            batch_first=True,
            **kwargs,
        )
        self.mlp = MLPRepresentation([in_features, out_features])

    @property
    def dim(self) -> int:
        return self.mlp.dim

    @staticmethod
    def _make_causal_mask(rows: int, columns: int) -> torch.Tensor:
        diagonal = 1 + columns - rows
        return torch.ones(rows, columns, dtype=torch.bool).triu(
            diagonal=diagonal
        )

    def forward(
        self,
        input: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        source = input if hidden is None else torch.cat([hidden, input], dim=1)

        L = input.size(-2)
        S = source.size(-2)
        mask = self._make_causal_mask(L, S)

        output, _ = self.attention(
            input,
            source,
            source,
            attn_mask=mask,
            need_weights=False,
        )
        output = self.mlp(F.relu(output))

        return output, source


def make_sequence_model(
    name: str,
    in_features: int,
    out_features: int,
    **kwargs,
) -> SequenceRepresentation:
    """sequence model factory"""

    if name == 'rnn':
        kwargs.setdefault('nonlinearity', 'relu')
        return RNNSequenceRepresentation(
            in_features,
            out_features,
            **kwargs,
        )

    if name == 'gru':
        return GRUSequenceRepresentation(in_features, out_features, **kwargs)

    if name == 'attention':
        num_heads = kwargs.pop('num_heads', 1)
        return AttentionSequenceRepresentation(
            in_features,
            out_features,
            num_heads,
            **kwargs,
        )

    raise ValueError(f'invalid sequence model name {name}')
