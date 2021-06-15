import torch
import torch.nn as nn

from asym_rlpo.data import Episode

from .base import A2C_Base


class AsymA2C_State(A2C_Base):
    model_keys = [
        'state_model',
        'action_model',
        'observation_model',
        'history_model',
        'policy_model',
        'vs_model',
    ]

    @staticmethod
    def compute_v_values(
        models: nn.ModuleDict, episode: Episode
    ) -> torch.Tensor:

        state_features = models.state_model(episode.states)
        vs_values = models.vs_model(state_features).squeeze(-1)
        return vs_values
