import torch
import torch.nn as nn

from asym_rlpo.data import Episode
from asym_rlpo.features import compute_history_features

from .base import A2C_Base


class AsymA2C(A2C_Base):
    model_keys = [
        'state_model',
        'action_model',
        'observation_model',
        'history_model',
        'policy_model',
        'vhs_model',
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
        state_features = models.state_model(episode.states)
        inputs = torch.cat([history_features, state_features], dim=-1)
        vhs_values = models.vhs_model(inputs).squeeze(-1)
        return vhs_values
