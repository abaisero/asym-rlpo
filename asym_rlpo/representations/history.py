import torch.nn as nn

from .base import Representation


class RNNHistoryRepresentation(Representation):
    def __init__(
        self,
        interaction_model: Representation,
        *,
        hidden_size: int,
        num_layers: int = 1,
        nonlinearity: str = 'relu',
    ):
        super().__init__()
        self.interaction_model = interaction_model
        self.rnn = nn.RNN(
            input_size=self.interaction_model.dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            nonlinearity=nonlinearity,
            batch_first=True,
        )

    @property
    def dim(self):
        return self.rnn.hidden_size

    def forward(self, inputs, *, hidden=None):
        return self.rnn(inputs, hidden)


class GRUHistoryRepresentation(Representation):
    def __init__(
        self,
        interaction_model: Representation,
        *,
        hidden_size: int,
        num_layers: int = 1,
    ):
        super().__init__()
        self.interaction_model = interaction_model
        self.gru = nn.GRU(
            input_size=self.interaction_model.dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

    @property
    def dim(self):
        return self.gru.hidden_size

    def forward(self, inputs, *, hidden=None):
        return self.gru(inputs, hidden)
