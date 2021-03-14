import torch
import torch.nn as nn

from .base import Representation


class RNNHistoryRepresentation(Representation, torch.Module):
    def __init__(
        self,
        action_representation: Representation,
        observation_representation: Representation,
        *,
        hidden_size: int,
        num_layers: int = 1,
        nonlinearity: str = 'relu',
    ):
        super().__init__()
        self.action_representation = action_representation
        self.observation_representation = observation_representation
        input_size = (
            self.action_representation.dim + self.observation_representation.dim
        )
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            nonlinearity=nonlinearity,
            batch_first=True,
        )

    @property
    def dim(self):
        return self.rnn.hidden_size

    def __call__(self, actions, observations, *, hidden=None):
        action_features = self.action_representation(actions)
        observation_features = self.observation_representation(observations)
        inputs = torch.cat([action_features, observation_features], dim=-1)
        return self.rnn(inputs, hidden)
