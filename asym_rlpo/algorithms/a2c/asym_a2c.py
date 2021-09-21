import torch
import torch.nn as nn

from asym_rlpo.data import Episode

from .base import PO_A2C_ABC


class AsymA2C(PO_A2C_ABC):
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
            'vhs_model',
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
        state_features = models.critic.state_model(episode.states)
        inputs = torch.cat([history_features, state_features], dim=-1)
        vhs_values = models.critic.vhs_model(inputs).squeeze(-1)
        return vhs_values
