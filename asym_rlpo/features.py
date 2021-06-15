import torch
import torch.nn as nn

from asym_rlpo.data import Torch_O
from asym_rlpo.representations.base import Representation


def compute_history_features(
    action_model: Representation,
    observation_model: Representation,
    history_model: Representation,
    actions: torch.Tensor,
    observations: Torch_O,
) -> torch.Tensor:

    action_features = action_model(actions)
    action_features = action_features.roll(1, 0)
    action_features[0, :] = 0.0
    observation_features = observation_model(observations)

    inputs = torch.cat([action_features, observation_features], dim=-1)
    history_features, _ = history_model(inputs.unsqueeze(0))
    history_features = history_features.squeeze(0)

    return history_features
