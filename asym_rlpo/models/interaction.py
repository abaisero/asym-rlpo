from __future__ import annotations

import torch

from asym_rlpo.data import Episode, TorchObservation
from asym_rlpo.models.model import FeatureModel
from asym_rlpo.types import Features


class InteractionModel(FeatureModel):
    def __init__(
        self,
        action_model: FeatureModel,
        observation_model: FeatureModel,
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
        actions: torch.Tensor | None,
        observations: TorchObservation,
    ):
        observation_features = self.observation_model(observations)
        action_features = (
            self._default_action_features(
                observation_features.shape[:-1],
                observation_features.device,
            )
            if actions is None
            else self.action_model(actions)
        )
        return torch.cat([action_features, observation_features], dim=-1)

    def _default_action_features(self, batch_shape, device):
        return self.action_model.zeros_like(device).expand(*batch_shape, -1)

    def episodic(self, episode: Episode) -> Features:
        """syncs actions and observations, and applies interaction models"""
        action_features = self.action_model(episode.actions)
        action_features = action_features.roll(1, 0)
        action_features[0, :] = 0.0
        observation_features = self.observation_model(episode.observations)

        return torch.cat([action_features, observation_features], dim=-1)

    def zeros_like(self, device: torch.device | None = None):
        return torch.cat(
            [
                self.action_model.zeros_like(device),
                self.observation_model.zeros_like(device),
            ]
        )
