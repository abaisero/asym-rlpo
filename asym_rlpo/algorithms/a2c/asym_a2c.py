import torch
import torch.nn as nn

from asym_rlpo.data import Episode
from asym_rlpo.features import compute_history_features

from .base import A2C_Base


class AsymA2C(A2C_Base):
    model_keys = [
        # actor
        'action_model',
        'observation_model',
        'history_model',
        'policy_model',
        # critic
        'critic_state_model',
        'critic_action_model',
        'critic_observation_model',
        'critic_history_model',
        'vhs_model',
    ]

    @staticmethod
    def compute_v_values(
        models: nn.ModuleDict, episode: Episode
    ) -> torch.Tensor:

        history_features = compute_history_features(
            models.critic_action_model,
            models.critic_observation_model,
            models.critic_history_model,
            episode.actions,
            episode.observations,
        )
        state_features = models.critic_state_model(episode.states)
        inputs = torch.cat([history_features, state_features], dim=-1)
        vhs_values = models.vhs_model(inputs).squeeze(-1)
        return vhs_values
