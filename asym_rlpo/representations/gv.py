from operator import itemgetter
from typing import Dict

import gym
import numpy as np
import torch
import torch.nn as nn

from asym_rlpo.representations.embedding import EmbeddingRepresentation
from asym_rlpo.utils.convert import numpy2torch
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
    def __init__(self, state_space: gym.Space):
        super().__init__()
        self.state_space = state_space
        # the GV state is a dictionary with fields:
        # * grid
        # * agent_ids
        # * item
        # * agent

        checkraise(
            isinstance(state_space, gym.spaces.Dict),
            TypeError,
            f'space type ({type(state_space)}) is not gym.spaces.Dict',
        )
        for k in gv_keys():
            checkraise(
                k in state_space.spaces.keys(),
                KeyError,
                f'space does not contain `{k}` key',
            )

        num_embeddings = 0
        for k in ['grid', 'item']:
            s = state_space[k]
            num_embeddings = max(num_embeddings, s.high.max() - s.low.min() + 1)
        embedding_size = 8
        self.embedding = EmbeddingRepresentation(num_embeddings, embedding_size)

        in_channels = 3 * embedding_size + 1
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
        )

        test_state = batchify(numpy2torch(state_space.sample()))
        y = self(test_state)
        self.__out_dim = y.shape[1]

    @property
    def dim(self):
        return self.__out_dim

    def forward(self, states: GV_State):
        unpack = itemgetter(*gv_keys())
        agent, agent_id_grid, grid, item = unpack(states)

        # Categorical components require embeddings
        grid = self.embedding(grid).flatten(start_dim=-2)
        item = self.embedding(item).flatten(start_dim=-2)

        # Handle the grid components (these are jointly processed by a CNN)
        agent_id_grid.unsqueeze_(-1)
        cnn_input = torch.cat([grid, agent_id_grid], dim=-1)
        cnn_input = torch.transpose(cnn_input, 1, 3)
        cnn_output = self.cnn(cnn_input)
        cnn_output = cnn_output.flatten(start_dim=1)

        # Concatenate CNN output with the vector inputs
        return torch.cat([agent, item, cnn_output], dim=-1)


def gv_keys():
    """An iterator over the keys we care about for gridverse."""
    return ('agent', 'agent_id_grid', 'grid', 'item')


def batchify(gv_dict):
    """Adds a batch axis to every subcomponent of a gridverse state or observation."""
    return {k: v.unsqueeze(0) for k, v in gv_dict.items()}
