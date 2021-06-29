import torch
import torch.nn as nn

from asym_rlpo.data import Episode

from .base import PO_A2C_ABC


class AsymA2C_State(PO_A2C_ABC):
    model_keys = [
        # actor
        'action_model',
        'observation_model',
        'history_model',
        'policy_model',
        # critic
        'critic_state_model',
        'vs_model',
    ]

    def compute_v_values(
        self, models: nn.ModuleDict, episode: Episode
    ) -> torch.Tensor:

        state_features = models.critic_state_model(episode.states)
        vs_values = models.vs_model(state_features).squeeze(-1)
        return vs_values
