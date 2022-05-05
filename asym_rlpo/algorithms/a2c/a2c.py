import torch
import torch.nn as nn

from asym_rlpo.data import Episode

from .base import A2C_ABC


class A2C(A2C_ABC):
    model_keys = {
        'agent': [
            'action_model',
            'observation_model',
            'history_model',
            'policy_model',
        ],
        'critic': [
            'state_model',
            'action_model',
            'observation_model',
            'history_model',
            'vh_model',
        ],
    }

    def compute_v_values(
        self, models: nn.ModuleDict, episode: Episode
    ) -> torch.Tensor:

        history_features = self.compute_history_features(
            models.critic.action_model,
            models.critic.observation_model,
            models.critic.history_model,
            episode.actions,
            episode.observations,
        )
        vh_values = models.critic.vh_model(history_features).squeeze(-1)
        return vh_values
