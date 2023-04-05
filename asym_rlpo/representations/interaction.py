from typing import Optional

import torch

from asym_rlpo.data import TorchObservation

from .base import Representation


class InteractionRepresentation(Representation):
    def __init__(
        self,
        action_model: Representation,
        observation_model: Representation,
    ):
        super().__init__()
        self.action_model = action_model
        self.observation_model = observation_model

    @property
    def dim(self):
        return self.action_model.dim + self.observation_model.dim

    # NOTE:  the actionless case is not due to there being an empty action
    # representation, but due to the first timestep where only an observation
    # is available
    def forward(
        self,
        *,
        actions: Optional[torch.Tensor],
        observations: TorchObservation,
    ):
        observation_features = self.observation_model(observations)
        action_features = (
            self._default_action_features(observation_features)
            if actions is None
            else self.action_model(actions)
        )
        return torch.cat([action_features, observation_features], dim=-1)

    def episode_features(
        self, actions: torch.Tensor, observations: TorchObservation
    ):
        """syncs actions and observations, and applies interaction models"""
        action_features = self.action_model(actions)
        action_features = action_features.roll(1, 0)
        action_features[0, :] = 0.0
        observation_features = self.observation_model(observations)

        return torch.cat([action_features, observation_features], dim=-1)

    def _default_action_features(self, observation_features):
        batch_shape = observation_features.shape[:-1]
        shape = batch_shape + (self.action_model.dim,)
        return torch.zeros(shape, device=observation_features.device)
