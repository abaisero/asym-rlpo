from __future__ import annotations

import torch.nn as nn

from .base import Representation


class HistoryRepresentation(Representation):
    def __init__(
        self,
        interaction_model: Representation,
        rnn: nn.Module,
    ):
        # crease a separate SequenceModule thingie which will encompass various
        # possibilities;  it just receives a sequence as input and returns a
        # sequence as output..
        super().__init__()
        self.interaction_model = interaction_model
        self.rnn = rnn

    @staticmethod
    def make_rnn(
        interaction_model: Representation,
        *,
        hidden_size: int,
        num_layers: int = 1,
        nonlinearity: str = 'relu',
    ) -> HistoryRepresentation:
        rnn = nn.RNN(
            input_size=interaction_model.dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            nonlinearity=nonlinearity,
            batch_first=True,
        )
        return HistoryRepresentation(interaction_model, rnn)

    @staticmethod
    def make_gru(
        interaction_model: Representation,
        *,
        hidden_size: int,
        num_layers: int = 1,
    ) -> HistoryRepresentation:
        gru = nn.GRU(
            input_size=interaction_model.dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        return HistoryRepresentation(interaction_model, gru)

    @property
    def dim(self):
        return self.rnn.hidden_size

    def forward(self, inputs, *, hidden=None):
        return self.rnn(inputs, hidden)
