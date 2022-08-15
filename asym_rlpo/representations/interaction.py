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

        return (
            self._forward_actionless(observations=observations)
            if actions is None
            else self._forward_actionful(
                actions=actions, observations=observations
            )
        )

    def _forward_actionful(
        self,
        *,
        actions: torch.Tensor,
        observations: TorchObservation,
    ):
        action_features = self.action_model(actions)
        observation_features = self.observation_model(observations)
        return torch.cat([action_features, observation_features], dim=-1)

    def _forward_actionless(self, *, observations: TorchObservation):
        observation_features = self.observation_model(observations)
        batch_shape = observation_features.shape[:-1]
        shape = batch_shape + (self.action_model.dim,)
        action_features = torch.zeros(shape, device=observation_features.device)
        return torch.cat([action_features, observation_features], dim=-1)
