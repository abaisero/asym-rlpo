import abc
from typing import Generic, TypeAlias, TypeVar

import torch
import torch.nn as nn
import torch.nn.functional as F

from asym_rlpo.models.mlp import MLP_Model
from asym_rlpo.models.model import Model
from asym_rlpo.types import Features

HiddenType = TypeVar('HiddenType')


class SequenceModel(Model, Generic[HiddenType]):
    @abc.abstractmethod
    def forward(
        self,
        input: Features,
        *,
        hidden: HiddenType | None = None,
    ) -> tuple[Features, HiddenType]:
        assert False


RNN_Hidden: TypeAlias = torch.Tensor


class RNN_SequenceModel(SequenceModel[RNN_Hidden]):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.rnn = nn.RNN(
            in_features,
            out_features,
            batch_first=True,
            nonlinearity='relu',
        )

    @property
    def dim(self) -> int:
        return self.rnn.hidden_size

    def forward(
        self,
        input: Features,
        *,
        hidden: RNN_Hidden | None = None,
    ) -> tuple[Features, RNN_Hidden]:
        return self.rnn(input, hidden)


GRU_Hidden: TypeAlias = torch.Tensor


class GRU_SequenceModel(SequenceModel[GRU_Hidden]):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.gru = nn.GRU(
            in_features,
            out_features,
            batch_first=True,
        )

    @property
    def dim(self) -> int:
        return self.gru.hidden_size

    def forward(
        self,
        input: Features,
        *,
        hidden: GRU_Hidden | None = None,
    ) -> tuple[Features, GRU_Hidden]:
        return self.gru(input, hidden)


Attention_Hidden: TypeAlias = torch.Tensor


class Attention_SequenceModel(SequenceModel[Attention_Hidden]):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int,
    ):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            in_features,
            num_heads,
            batch_first=True,
        )
        self.mlp = MLP_Model([in_features, out_features], ['relu'])

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
        input: Features,
        *,
        hidden: Attention_Hidden | None = None,
    ) -> tuple[Features, Attention_Hidden]:
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
    *,
    attention_num_heads: int | None = None,
) -> SequenceModel:
    """sequence model factory"""

    if name == 'rnn':
        return RNN_SequenceModel(in_features, out_features)

    if name == 'gru':
        return GRU_SequenceModel(in_features, out_features)

    if name == 'attention':
        if attention_num_heads is None:
            raise ValueError(
                f'{attention_num_heads=} required for attention model'
            )

        if attention_num_heads <= 0:
            raise ValueError(
                f'{attention_num_heads=} must be positive for attention model'
            )

        return Attention_SequenceModel(
            in_features,
            out_features,
            attention_num_heads,
        )

    raise ValueError(f'invalid sequence model name {name}')
