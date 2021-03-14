from typing import Dict

import numpy as np
import torch

from .base import Representation

# gridverse types
GV_State = Dict[str, np.ndarray]
GV_Observation = Dict[str, np.ndarray]


class GV_ObservationRepresentation(Representation, torch.Module):
    # TODO implement GV primary variant, which receives the gv observation type and whatnot
    def __init__(self, observation_space: gym.Space):
        super().__init__()
        # the GV observation is a dictionary with fields:
        # * grid
        # * agent_ids
        # * agent
        # * item

    def __call__(self, observations: GV_Observation):
        # TODO this method should be able to receive multiple states..
        # how should that be structured?  directly as compacted tensors?
        raise NotImplementedError


class GV_StateRepresentation(Representation, torch.Module):
    # TODO implement GV primary variant, which receives the gv observation type and whatnot
    def __init__(self, observation_space: gym.Space):
        super().__init__()
        # the GV observation is a dictionary with fields:
        # * grid
        # * agent_ids
        # * item

    def __call__(self, states: GV_State):
        # TODO this method should be able to receive multiple states..
        # how should that be structured?  directly as compacted tensors?
        raise NotImplementedError
