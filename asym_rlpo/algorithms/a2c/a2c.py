import torch
import torch.nn as nn

from asym_rlpo.data import Episode

from .base import PO_A2C_ABC


class A2C(PO_A2C_ABC):
    model_keys = [
        # actor
        'action_model',
        'observation_model',
        'history_model',
        'policy_model',
        # critic
        'critic_action_model',
        'critic_observation_model',
        'critic_history_model',
        'vh_model',
    ]

    def compute_v_values(
        self, models: nn.ModuleDict, episode: Episode
    ) -> torch.Tensor:

        history_features = self.compute_history_features(
            models.critic_action_model,
            models.critic_observation_model,
            models.critic_history_model,
            episode.actions,
            episode.observations,
        )
        vh_values = models.vh_model(history_features).squeeze(-1)
        return vh_values
