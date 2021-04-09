from typing import Dict

import gym
import numpy as np
import torch
import torch.nn as nn

from asym_rlpo.utils.debugging import checkraise

from .base import Representation

# gridverse types
GV_State = Dict[str, np.ndarray]
GV_Observation = Dict[str, np.ndarray]


class GV_ObservationRepresentation(Representation, nn.Module):
    # TODO implement GV primary variant, which receives the gv observation type and whatnot
    def __init__(self, observation_space: gym.Space):
        super().__init__()
        self.observation_space = observation_space
        # the GV observation is a dictionary with fields:
        # * grid
        # * agent_ids
        # * agent
        # * item

        # TODO temporarily commented out
        # checkraise(
        #     isinstance(observation_space, gym.spaces.Dict),
        #     TypeError,
        #     f'space type ({type(observation_space)}) is not gym.spaces.Dict',
        # )
        # checkraise(
        #     'grid' in observation_space.spaces.keys(),
        #     KeyError,
        #     'space does not contain `grid` key',
        # )
        # checkraise(
        #     'agent_ids' in observation_space.spaces.keys(),
        #     KeyError,
        #     'space does not contain `agent_ids` key',
        # )
        # checkraise(
        #     'agent' in observation_space.spaces.keys(),
        #     KeyError,
        #     'space does not contain `agent` key',
        # )
        # checkraise(
        #     'item' in observation_space.spaces.keys(),
        #     KeyError,
        #     'space does not contain `item` key',
        # )

        # TODO initialize models

        # TODO initialize models
        # TODO temporarily assume all values are features
        self.__in_dim = sum(v.size for v in observation_space.sample().values())
        self.__out_dim = 128

        self.model = nn.Linear(self.__in_dim, self.__out_dim)
        self(observation_space.sample())

    @property
    def dim(self):
        return self.__out_dim

    def forward(self, observations: GV_Observation):
        # TODO this method should be able to receive multiple states..
        # how should that be structured?  directly as compacted tensors?
        inputs = torch.cat(
            [
                torch.from_numpy(v).flatten().float()
                for v in observations.values()
            ]
        )
        return self.model(inputs)


class GV_StateRepresentation(Representation, nn.Module):
    # TODO implement GV primary variant, which receives the gv observation type and whatnot
    def __init__(self, state_space: gym.Space):
        super().__init__()
        self.state_space = state_space
        # the GV observation is a dictionary with fields:
        # * grid
        # * agent_ids
        # * item

        # TODO temporarily commented out
        # checkraise(
        #     isinstance(observation_space, gym.spaces.Dict),
        #     TypeError,
        #     f'space type ({type(observation_space)}) is not gym.spaces.Dict',
        # )
        # checkraise(
        #     'grid' in observation_space.spaces.keys(),
        #     KeyError,
        #     'space does not contain `grid` key',
        # )
        # checkraise(
        #     'item' in observation_space.spaces.keys(),
        #     KeyError,
        #     'space does not contain `item` key',
        # )

        # TODO initialize models
        # TODO temporarily assume all values are features
        self.__in_dim = sum(v.size for v in state_space.sample().values())
        self.__out_dim = 128

        self.model = nn.Linear(self.__in_dim, self.__out_dim)
        self(state_space.sample())

    @property
    def dim(self):
        return self.__out_dim

    def forward(self, states: GV_State):
        # TODO this method should be able to receive multiple states..
        # how should that be structured?  directly as compacted tensors?
        inputs = torch.cat(
            [torch.from_numpy(v).flatten().float() for v in states.values()]
        )
        return self.model(inputs)
