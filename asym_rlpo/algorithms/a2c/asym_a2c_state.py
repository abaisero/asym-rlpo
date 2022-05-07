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
            'latent_model',
            'action_model',
            'observation_model',
            'history_model',
            'vz_model',
        ],
    }

    def compute_v_values(
        self, models: nn.ModuleDict, episode: Episode
    ) -> torch.Tensor:

        latent_features = models.critic.latent_model(episode.states)
        vz_values = models.critic.vz_model(latent_features).squeeze(-1)
        return vz_values
