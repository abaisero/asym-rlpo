from typing import Dict, Union

import gym
import torch
import torch.nn as nn

import asym_rlpo.generalized_torch as gtorch
from asym_rlpo.representations.embedding import EmbeddingRepresentation
from asym_rlpo.utils.convert import numpy2torch
from asym_rlpo.utils.debugging import checkraise

from .base import Representation

# gridverse types
GV_State = Dict[str, torch.Tensor]
GV_Observation = Dict[str, torch.Tensor]


class GV_ObservationRepresentation(Representation, nn.Module):
    def __init__(self, observation_space: gym.Space):
        super().__init__()
        self.observation_space = observation_space
        # the GV observation is a dictionary with fields;  we use the fields
        # `grid` and `item`, and ignore the rest.

        checkraise(
            isinstance(observation_space, gym.spaces.Dict),
            TypeError,
            f'space type ({type(observation_space)}) is not gym.spaces.Dict',
        )
        checkraise(
            'grid' in observation_space.spaces,
            KeyError,
            'space does not contain `grid` key',
        )
        checkraise(
            'item' in observation_space.spaces,
            KeyError,
            'space does not contain `item` key',
        )

        num_embeddings = max(
            observation_space['grid'].high.max() + 1,
            observation_space['item'].high.max() + 1,
        )
        embedding_size = 1
        self.embedding = EmbeddingRepresentation(num_embeddings, embedding_size)

        in_channels = 3 * embedding_size
        self.cnn = gv_cnn(in_channels)

        # get empirical out_dim
        test_observation = batchify(numpy2torch(observation_space.sample()))
        self.__dim = self.forward(test_observation).shape[1]

    @property
    def dim(self):
        return self.__dim

    def forward(self, inputs: GV_Observation):
        grid = inputs['grid']
        item = inputs['item']

        # Categorical components require embeddings
        grid = self.embedding(grid).flatten(start_dim=-2)
        item = self.embedding(item).flatten(start_dim=-2)

        # Handle the grid component (processed by a CNN)
        cnn_input = torch.transpose(grid, 1, 3)
        cnn_output = self.cnn(cnn_input)
        cnn_output = cnn_output.flatten(start_dim=1)

        # Concatenate CNN output with the vector input
        return torch.cat([item, cnn_output], dim=-1)


class FullyConnected_GV_ObservationRepresentation(Representation, nn.Module):
    def __init__(self, observation_space: gym.Space):
        super().__init__()
        self.observation_space = observation_space
        # the GV observation is a dictionary with fields;  we use the fields
        # `grid` and `item`, and ignore the rest.

        checkraise(
            isinstance(observation_space, gym.spaces.Dict),
            TypeError,
            f'space type ({type(observation_space)}) is not gym.spaces.Dict',
        )
        checkraise(
            'grid' in observation_space.spaces,
            KeyError,
            'space does not contain `grid` key',
        )
        checkraise(
            'item' in observation_space.spaces,
            KeyError,
            'space does not contain `item` key',
        )

        num_embeddings = max(
            observation_space['grid'].high.max() + 1,
            observation_space['item'].high.max() + 1,
        )
        embedding_size = 8
        self.embedding = EmbeddingRepresentation(num_embeddings, embedding_size)

        self.__dim = embedding_size * (
            observation_space['grid'].sample().size
            + observation_space['item'].sample().size
        )

    @property
    def dim(self):
        return self.__dim

    def forward(self, inputs: GV_Observation):
        grid = inputs['grid']
        item = inputs['item']

        # Categorical components require embeddings
        grid = self.embedding(grid).flatten(start_dim=1)
        item = self.embedding(item).flatten(start_dim=1)

        # just concatenate embeddings
        return torch.cat([grid, item], dim=-1)


class GV_StateRepresentation(Representation, nn.Module):
    def __init__(self, state_space: gym.Space):
        super().__init__()
        self.state_space = state_space
        # the GV state is a dictionary with fields;  we use the fields `grid`,
        # `agent_id_grid`, `item`, and `agent`, and ignore the rest.

        checkraise(
            isinstance(state_space, gym.spaces.Dict),
            TypeError,
            f'space type ({type(state_space)}) is not gym.spaces.Dict',
        )
        checkraise(
            'grid' in state_space.spaces,
            KeyError,
            'space does not contain `grid` key',
        )
        checkraise(
            'agent_id_grid' in state_space.spaces,
            KeyError,
            'space does not contain `agent_id_grid` key',
        )
        checkraise(
            'item' in state_space.spaces,
            KeyError,
            'space does not contain `item` key',
        )
        checkraise(
            'agent' in state_space.spaces,
            KeyError,
            'space does not contain `agent` key',
        )

        # `grid` and `item` are the only categorical fields
        num_embeddings = max(
            state_space['grid'].high.max() + 1,
            state_space['item'].high.max() + 1,
        )
        embedding_size = 1
        self.embedding = EmbeddingRepresentation(num_embeddings, embedding_size)

        in_channels = 3 * embedding_size + 1  # adding one for agent_id_grid
        self.cnn = gv_cnn(in_channels)

        # get empirical out_dim
        test_state = batchify(numpy2torch(state_space.sample()))
        self.__dim = self.forward(test_state).shape[1]

    @property
    def dim(self):
        return self.__dim

    def forward(self, inputs: GV_State):
        agent = inputs['agent']
        agent_id_grid = inputs['agent_id_grid']
        grid = inputs['grid']
        item = inputs['item']

        # Categorical components require embeddings
        grid = self.embedding(grid).flatten(start_dim=-2)
        item = self.embedding(item).flatten(start_dim=-2)

        # Handle the grid components (these are jointly processed by a CNN)
        agent_id_grid = agent_id_grid.unsqueeze(-1)
        cnn_input = torch.cat([grid, agent_id_grid], dim=-1)
        cnn_input = torch.transpose(cnn_input, 1, 3)
        cnn_output = self.cnn(cnn_input)
        cnn_output = cnn_output.flatten(start_dim=1)

        # Concatenate CNN output with the vector inputs
        return torch.cat([agent, item, cnn_output], dim=-1)


def gv_cnn(in_channels):
    """Gridverse convolutional network shared by the observation/state representations."""
    return nn.Sequential(
        nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, padding=1),
        nn.ReLU(),
    )


def batchify(gv_type: Union[GV_Observation, GV_State]):
    """Adds a batch axis to every subcomponent of a gridverse state or observation."""
    return gtorch.unsqueeze(gv_type, 0)
