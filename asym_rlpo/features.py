import torch
import torch.nn as nn

from asym_rlpo.data import Torch_O


def compute_history_features(
    models: nn.ModuleDict, actions: torch.Tensor, observations: Torch_O
) -> torch.Tensor:

    action_features = models.action_model(actions)
    action_features = action_features.roll(1, 0)
    action_features[0, :] = 0.0
    observation_features = models.observation_model(observations)

    inputs = torch.cat([action_features, observation_features], dim=-1)
    history_features, _ = models.history_model(inputs.unsqueeze(0))
    history_features = history_features.squeeze(0)

    return history_features
