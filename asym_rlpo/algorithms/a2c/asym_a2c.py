import torch
import torch.nn as nn

from asym_rlpo.data import Episode

from .base import A2C_ABC


class AsymA2C(A2C_ABC):
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
            'vhz_model',
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
        latent_features = models.critic.latent_model(episode.latents)
        inputs = torch.cat([history_features, latent_features], dim=-1)
        vhz_values = models.critic.vhz_model(inputs).squeeze(-1)
        return vhz_values
