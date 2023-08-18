import math
from collections.abc import Iterable
from functools import cached_property

import gym
import gym.spaces
import torch
import torch.nn as nn
import yaml

import asym_rlpo.generalized_torch as gtorch
from asym_rlpo.models.cat import CatModel
from asym_rlpo.models.embedding import EmbeddingModel
from asym_rlpo.models.model import Model
from asym_rlpo.modules.mlp import make_mlp
from asym_rlpo.utils.cnn import make_cnn_from_filename
from asym_rlpo.utils.config import get_config
from asym_rlpo.utils.convert import numpy2torch

# gridverse types
GV_State = dict[str, torch.Tensor]
GV_Observation = dict[str, torch.Tensor]


def _check_gv_observation_space_keys(space: gym.Space) -> bool:
    if not isinstance(space, gym.spaces.Dict):
        raise TypeError('incorrect observation space type')

    for key in ['grid', 'item']:
        if key not in space.spaces:
            raise KeyError(f'space does not contain {key=}')


def _check_gv_state_space_keys(space: gym.Space) -> bool:
    if not isinstance(space, gym.spaces.Dict):
        raise TypeError('incorrect state space type')

    for key in ['grid', 'agent_id_grid', 'agent', 'item']:
        if key not in space.spaces:
            raise KeyError(f'space does not contain {key=}')


def make_cnn(channels: int) -> nn.Sequential:
    config = get_config()

    return make_cnn_from_filename(config.gv_cnn, channels)


class GV_Model(Model):
    def __init__(
        self,
        space: gym.spaces.Dict,
        names: Iterable[str],
        *,
        embedding_size: int,
        layers: list[int],
    ):
        super().__init__()
        self.space = space

        num_embeddings = max(
            space['grid'].high.max() + 1,
            space['item'].high.max() + 1,
        )
        self.embedding_model = EmbeddingModel(num_embeddings, embedding_size)
        gv_models = [self._make_gv_submodel(name) for name in names]
        self.cat_model = CatModel(gv_models)
        self.fc_model: nn.Module

        if len(layers) > 0:
            sizes = [self.cat_model.dim] + layers
            nonlinearities = ['relu'] * len(layers)
            self.fc_model = make_mlp(sizes, nonlinearities)
            self._dim = sizes[-1]

        else:
            self.fc_model = nn.Identity()
            self._dim = self.cat_model.dim

    @property
    def dim(self):
        return self._dim

    def forward(self, inputs: GV_State):
        return self.fc_model(self.cat_model(inputs))

    def _make_gv_submodel(self, name: str):
        config = get_config()

        assert self.space.spaces['grid'].shape[-1] == 3

        channels = [0, 1, 2]

        if config.gv_ignore_color_channel:
            channels.remove(1)

        if config.gv_ignore_state_channel:
            channels.remove(2)

        if name == 'agent':
            if 'agent' not in self.space.spaces:
                raise KeyError('space does not contain `agent` key')

            return GV_Agent_Model(self.space)

        if name == 'item':
            if 'item' not in self.space.spaces:
                raise KeyError('space does not contain `item` key')

            return GV_Item_Model(self.space, self.embedding_model, channels)

        if name == 'grid-cnn':
            if 'grid' not in self.space.spaces:
                raise KeyError('space does not contain `grid` key')

            return GV_Grid_CNN_Model(
                self.space,
                self.embedding_model,
                channels,
            )

        if name == 'grid-fc':
            if 'grid' not in self.space.spaces:
                raise KeyError('space does not contain `grid` key')

            return GV_Grid_FC_Model(
                self.space,
                self.embedding_model,
                channels,
            )

        if name == 'agent-grid-cnn':
            if 'grid' not in self.space.spaces:
                raise KeyError('space does not contain `grid` key')

            if 'agent_id_grid' not in self.space.spaces:
                raise KeyError('space does not contain `agent_id_grid` key')

            return GV_AgentGrid_CNN_Model(
                self.space,
                self.embedding_model,
                channels,
            )

        if name == 'agent-grid-fc':
            if 'grid' not in self.space.spaces:
                raise KeyError('space does not contain `grid` key')

            if 'agent_id_grid' not in self.space.spaces:
                raise KeyError('space does not contain `agent_id_grid` key')

            return GV_AgentGrid_FC_Model(
                self.space,
                self.embedding_model,
                channels,
            )

        raise ValueError(f'invalid gv model name {name}')


def batchify(gv_type: GV_Observation | GV_State):
    """Adds a batch axis to every subcomponent of a gridverse state or observation."""
    return gtorch.unsqueeze(gv_type, 0)


class GV_Agent_Model(Model):
    def __init__(self, space: gym.spaces.Dict):
        super().__init__()
        if 'agent' not in space.spaces:
            raise KeyError('space does not contain `agent` key')

        self.space = space

    @property
    def dim(self):
        (agent_dim,) = self.space['agent'].shape
        return agent_dim

    def forward(self, inputs: GV_Observation):
        agent = inputs['agent']
        return agent


class GV_Item_Model(Model):
    def __init__(
        self,
        space: gym.spaces.Dict,
        embedding_model: EmbeddingModel,
        channels: list[int],
    ):
        super().__init__()
        if 'item' not in space.spaces:
            raise KeyError('space does not contain `item` key')

        self.space = space
        self.embedding_model = embedding_model
        self.channels = channels

    @property
    def dim(self):
        (item_dim,) = self.space['item'].shape
        return item_dim * self.embedding_model.dim

    def forward(self, inputs: GV_Observation):
        item = inputs['item']
        item[..., self.channels]
        return self.embedding_model(item).flatten(start_dim=-2)


class GV_Grid_CNN_Model(Model):
    def __init__(
        self,
        space: gym.spaces.Dict,
        embedding_model: EmbeddingModel,
        channels: list[int],
    ):
        super().__init__()
        if 'grid' not in space.spaces:
            raise KeyError('space does not contain `grid` key')

        self.space = space
        self.embedding_model = embedding_model
        self.channels = channels

        in_channels = len(channels) * embedding_model.dim
        self.cnn_model = make_cnn(in_channels)

    @cached_property
    def dim(self):
        observation = self.space.sample()
        observation = batchify(numpy2torch(observation))
        return self.forward(observation).shape[1]

    def forward(self, inputs: GV_Observation):
        grid = inputs['grid']
        grid = grid[..., self.channels]
        grid = self.embedding_model(grid).flatten(start_dim=-2)

        cnn_input = torch.transpose(grid, 1, 3)
        cnn_output = self.cnn_model(cnn_input)
        cnn_output = cnn_output.flatten(start_dim=1)

        return cnn_output


class GV_AgentGrid_CNN_Model(Model):
    def __init__(
        self,
        space: gym.spaces.Dict,
        embedding_model: EmbeddingModel,
        channels: list[int],
    ):
        super().__init__()
        if 'grid' not in space.spaces:
            raise KeyError('space does not contain `grid` key')

        if 'agent_id_grid' not in space.spaces:
            raise KeyError('space does not contain `agent_id_grid` key')

        self.space = space
        self.embedding_model = embedding_model
        self.channels = channels

        # adding one for agent_id_grid
        in_channels = len(channels) * embedding_model.dim + 1
        self.cnn_model = make_cnn(in_channels)

    @cached_property
    def dim(self):
        state = self.space.sample()
        state = batchify(numpy2torch(state))
        return self.forward(state).shape[1]

    def forward(self, inputs: GV_Observation):
        grid = inputs['grid']
        grid = grid[..., self.channels]
        agent_id_grid = inputs['agent_id_grid']

        grid = self.embedding_model(grid).flatten(start_dim=-2)
        agent_id_grid = agent_id_grid.unsqueeze(-1)
        cnn_input = torch.cat([grid, agent_id_grid], dim=-1)
        cnn_input = torch.transpose(cnn_input, 1, 3)
        cnn_output = self.cnn_model(cnn_input)
        cnn_output = cnn_output.flatten(start_dim=1)

        return cnn_output


class GV_Grid_FC_Model(Model):
    def __init__(
        self,
        space: gym.spaces.Dict,
        embedding_model: EmbeddingModel,
        channels: list[int],
    ):
        super().__init__()
        if 'grid' not in space.spaces:
            raise KeyError('space does not contain `grid` key')

        self.space = space
        self.embedding_model = embedding_model
        self.channels = channels

        in_channels = len(channels) * embedding_model.dim
        self.cnn_model = make_cnn(in_channels)

    @property
    def dim(self):
        grid_dim = math.prod(self.space['grid'].shape)
        return grid_dim * self.embedding_model.dim

    def forward(self, inputs: GV_Observation):
        grid = inputs['grid']
        grid = grid[..., self.channels]
        return self.embedding_model(grid).flatten(start_dim=-4)


class GV_AgentGrid_FC_Model(Model):
    def __init__(
        self,
        space: gym.spaces.Dict,
        embedding_model: EmbeddingModel,
        channels: list[int],
    ):
        super().__init__()
        if 'grid' not in space.spaces:
            raise KeyError('space does not contain `grid` key')

        if 'agent_id_grid' not in space.spaces:
            raise KeyError('space does not contain `agent_id_grid` key')

        self.space = space
        self.embedding_model = embedding_model
        self.channels = channels

    @property
    def dim(self):
        grid_dim = math.prod(self.space['grid'].shape)
        agent_dim = math.prod(self.space['agent_id_grid'].shape)
        return grid_dim * self.embedding_model.dim + agent_dim

    def forward(self, inputs: GV_Observation):
        grid = inputs['grid']
        grid = grid[..., self.channels]
        agent_id_grid = inputs['agent_id_grid']

        grid = self.embedding_model(grid).flatten(start_dim=-4)
        agent_id_grid = agent_id_grid.flatten(start_dim=-2)
        return torch.cat([grid, agent_id_grid], dim=-1)


class GV_Memory_Model(Model):
    def __init__(self, space: gym.spaces.Box, *, embedding_size: int):
        if not isinstance(space, gym.spaces.Box):
            raise TypeError(
                f'Invalid space type; should be gym.spaces.Box, is {type(space)}'
            )

        if space.shape is None or len(space.shape) != 1:
            raise ValueError(
                f'Invalid space shape;  should have single dimension, has {space.shape}'
            )

        super().__init__()
        num_embeddings = space.high.item() + 1
        self.embedding_model = EmbeddingModel(num_embeddings, embedding_size)

    @property
    def dim(self):
        return self.embedding_model.dim

    def forward(self, inputs: torch.Tensor):
        return self.embedding_model(inputs)
