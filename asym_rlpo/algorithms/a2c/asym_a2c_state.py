import torch
import torch.nn as nn

from asym_rlpo.data import Episode

from .base import A2C_ABC


class AsymA2C_State(A2C_ABC):
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
            'vs_model',
        ],
    }

    def compute_v_values(
        self, models: nn.ModuleDict, episode: Episode
    ) -> torch.Tensor:

        state_features = models.critic.state_model(episode.states)
        vs_values = models.critic.vs_model(state_features).squeeze(-1)
        return vs_values
