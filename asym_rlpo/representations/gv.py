import math
from functools import cached_property
from typing import Dict, Union

import gym
import torch
import torch.nn as nn

import asym_rlpo.generalized_torch as gtorch
from asym_rlpo.representations.cat import CatRepresentation
from asym_rlpo.representations.embedding import EmbeddingRepresentation
from asym_rlpo.utils.convert import numpy2torch
from asym_rlpo.utils.debugging import checkraise

from .base import Representation

# gridverse types
GV_State = Dict[str, torch.Tensor]
GV_Observation = Dict[str, torch.Tensor]


def _check_gv_observation_space_keys(space: gym.Space) -> bool:
    checkraise(
        isinstance(space, gym.spaces.Dict),
        TypeError,
        'incorrect observation space type',
    )

    for key in ['grid', 'item']:
        checkraise(
            key in space.spaces,
            KeyError,
            f'space does not contain `{key}` key',
        )


def _check_gv_state_space_keys(space: gym.Space) -> bool:
    checkraise(
        isinstance(space, gym.spaces.Dict),
        TypeError,
        'incorrect state space type',
    )

    for key in ['grid', 'agent_id_grid', 'agent', 'item']:
        checkraise(
            key in space.spaces,
            KeyError,
            f'space does not contain `{key}` key',
        )


class GV_ObservationRepresentation(Representation):
    # the GV observation is a dictionary with fields;  we use the fields
    # `grid` and `item`, and ignore the rest.

    def __init__(
        self,
        observation_space: gym.Space,
        *,
        embedding_size: int = 8,
        model_type: str,
    ):
        super().__init__()

        checkraise(
            model_type in ['cnn', 'fc'],
            ValueError,
            f'invalid `model_type` ({model_type})',
        )
        _check_gv_observation_space_keys(observation_space)
        self.observation_space = observation_space

        num_embeddings = max(
            observation_space['grid'].high.max() + 1,
            observation_space['item'].high.max() + 1,
        )
        self.embedding = EmbeddingRepresentation(num_embeddings, embedding_size)
        self.cat_representation = CatRepresentation(
            [
                GV_Grid_CNN_Representation(observation_space, self.embedding)
                if model_type == 'cnn'
                else GV_Grid_FC_Representation(
                    observation_space, self.embedding
                ),
                GV_Item_Representation(observation_space, self.embedding),
            ]
        )

    @property
    def dim(self):
        return self.cat_representation.dim

    def forward(self, inputs: GV_Observation):
        return self.cat_representation(inputs)


class GV_StateRepresentation(Representation):
    # the GV state is a dictionary with fields;  we use the fields `grid`,
    # `agent_id_grid`, `item`, and `agent`, and ignore the rest.

    def __init__(
        self,
        state_space: gym.Space,
        *,
        embedding_size: int = 1,
        model_type: str,
    ):
        super().__init__()

        checkraise(
            model_type in ['cnn', 'fc'],
            ValueError,
            f'invalid `model_type` ({model_type})',
        )
        _check_gv_state_space_keys(state_space)
        self.state_space = state_space

        # `grid` and `item` are the only categorical fields
        num_embeddings = max(
            state_space['grid'].high.max() + 1,
            state_space['item'].high.max() + 1,
        )
        embedding = EmbeddingRepresentation(num_embeddings, embedding_size)
        self.cat_representation = CatRepresentation(
            [
                GV_AgentGrid_CNN_Representation(state_space, embedding)
                if model_type == 'cnn'
                else GV_AgentGrid_FC_Representation(state_space, embedding),
                GV_Agent_Representation(state_space),
                GV_Item_Representation(state_space, embedding),
            ]
        )

    @property
    def dim(self):
        return self.cat_representation.dim

    def forward(self, inputs: GV_State):
        return self.cat_representation(inputs)


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


class GV_Agent_Representation(Representation):
    def __init__(self, space: gym.spaces.Dict):
        super().__init__()
        checkraise(
            'agent' in space.spaces,
            KeyError,
            'space does not contain `agent` key',
        )

        self.space = space

    @property
    def dim(self):
        (agent_dim,) = self.space['agent'].shape
        return agent_dim

    def forward(self, inputs: GV_Observation):
        agent = inputs['agent']
        return agent


class GV_Item_Representation(Representation):
    def __init__(
        self,
        space: gym.spaces.Dict,
        embedding: EmbeddingRepresentation,
    ):
        super().__init__()
        checkraise(
            'item' in space.spaces,
            KeyError,
            'space does not contain `item` key',
        )

        self.space = space
        self.embedding = embedding

    @property
    def dim(self):
        (item_dim,) = self.space['item'].shape
        return item_dim * self.embedding.dim

    def forward(self, inputs: GV_Observation):
        item = inputs['item']
        return self.embedding(item).flatten(start_dim=-2)


class GV_Grid_CNN_Representation(Representation):
    def __init__(
        self,
        space: gym.spaces.Dict,
        embedding: EmbeddingRepresentation,
    ):
        super().__init__()
        checkraise(
            'grid' in space.spaces,
            KeyError,
            'space does not contain `grid` key',
        )

        self.space = space
        self.embedding = embedding

        in_channels = 3 * embedding.dim
        self.cnn = gv_cnn(in_channels)

    @cached_property
    def dim(self):
        observation = self.space.sample()
        observation = batchify(numpy2torch(observation))
        return self.forward(observation).shape[1]

    def forward(self, inputs: GV_Observation):
        grid = inputs['grid']
        grid = self.embedding(grid).flatten(start_dim=-2)

        cnn_input = torch.transpose(grid, 1, 3)
        cnn_output = self.cnn(cnn_input)
        cnn_output = cnn_output.flatten(start_dim=1)

        return cnn_output


class GV_AgentGrid_CNN_Representation(Representation):
    def __init__(
        self,
        space: gym.spaces.Dict,
        embedding: EmbeddingRepresentation,
    ):
        super().__init__()
        checkraise(
            'grid' in space.spaces,
            KeyError,
            'space does not contain `grid` key',
        )
        checkraise(
            'agent_id_grid' in space.spaces,
            KeyError,
            'space does not contain `agent_id_grid` key',
        )

        self.space = space
        self.embedding = embedding

        in_channels = 3 * embedding.dim + 1  # adding one for agent_id_grid
        self.cnn = gv_cnn(in_channels)

    @cached_property
    def dim(self):
        state = self.space.sample()
        state = batchify(numpy2torch(state))
        return self.forward(state).shape[1]

    def forward(self, inputs: GV_Observation):
        grid = inputs['grid']
        agent_id_grid = inputs['agent_id_grid']

        grid = self.embedding(grid).flatten(start_dim=-2)
        agent_id_grid = agent_id_grid.unsqueeze(-1)
        cnn_input = torch.cat([grid, agent_id_grid], dim=-1)
        cnn_input = torch.transpose(cnn_input, 1, 3)
        cnn_output = self.cnn(cnn_input)
        cnn_output = cnn_output.flatten(start_dim=1)

        return cnn_output


class GV_Grid_FC_Representation(Representation):
    def __init__(
        self,
        space: gym.spaces.Dict,
        embedding: EmbeddingRepresentation,
    ):
        super().__init__()
        checkraise(
            'grid' in space.spaces,
            KeyError,
            'space does not contain `grid` key',
        )

        self.space = space
        self.embedding = embedding

    @property
    def dim(self):
        grid_dim = math.prod(self.space['grid'].shape)
        return grid_dim * self.embedding.dim

    def forward(self, inputs: GV_Observation):
        grid = inputs['grid']
        return self.embedding(grid).flatten(start_dim=-4)


class GV_AgentGrid_FC_Representation(Representation):
    def __init__(
        self,
        space: gym.spaces.Dict,
        embedding: EmbeddingRepresentation,
    ):
        super().__init__()
        checkraise(
            'grid' in space.spaces,
            KeyError,
            'space does not contain `grid` key',
        )
        checkraise(
            'agent_id_grid' in space.spaces,
            KeyError,
            'space does not contain `agent_id_grid` key',
        )

        self.space = space
        self.embedding = embedding

    @property
    def dim(self):
        grid_dim = math.prod(self.space['grid'].shape)
        agent_dim = math.prod(self.space['agent_id_grid'].shape)
        return grid_dim * self.embedding.dim + agent_dim

    def forward(self, inputs: GV_Observation):
        grid = inputs['grid']
        agent_id_grid = inputs['agent_id_grid']

        grid = self.embedding(grid).flatten(start_dim=-4)
        agent_id_grid = agent_id_grid.flatten(start_dim=-2)
        return torch.cat([grid, agent_id_grid], dim=-1)
