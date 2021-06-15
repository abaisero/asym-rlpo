import torch
import torch.nn as nn

from asym_rlpo.data import Episode
from asym_rlpo.features import compute_history_features

from .base import A2C_Base


class A2C(A2C_Base):
    model_keys = [
        'action_model',
        'observation_model',
        'history_model',
        'policy_model',
        'vh_model',
    ]

    @staticmethod
    def compute_v_values(
        models: nn.ModuleDict, episode: Episode
    ) -> torch.Tensor:

        history_features = compute_history_features(
            models.action_model,
            models.observation_model,
            models.history_model,
            episode.actions,
            episode.observations,
        )
        vh_values = models.vh_model(history_features).squeeze(-1)
        return vh_values
