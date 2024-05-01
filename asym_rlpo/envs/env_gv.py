from __future__ import annotations

import gym
import gym.spaces
from gym_gridverse.debugging import reset_gv_debug
from gym_gridverse.envs.yaml.factory import factory_env_from_yaml
from gym_gridverse.grid_object import Beacon
from gym_gridverse.gym import outer_space_to_gym_space
from gym_gridverse.outer_env import OuterEnv
from gym_gridverse.representations.observation_representations import (
    make_observation_representation,
)
from gym_gridverse.representations.state_representations import (
    make_state_representation,
)

from asym_rlpo.envs.env import (
    Action,
    Latent,
    LatentEnvironmentModule,
    Observation,
    State,
    StatefulEnvironment,
    StateLatentEnvironmentModule,
)
from asym_rlpo.envs.types import EnvironmentType


def make_gv_env(
    path: str,
    *,
    latent_type: str,
    gv_representation: str,
) -> tuple[StatefulEnvironment, LatentEnvironmentModule]:
    reset_gv_debug(False)

    print('Loading using YAML')
    inner_env = factory_env_from_yaml(path)
    state_representation = make_state_representation(
        gv_representation,
        inner_env.state_space,
    )
    observation_representation = make_observation_representation(
        gv_representation,
        inner_env.observation_space,
    )
    outer_env = OuterEnv(
        inner_env,
        state_representation=state_representation,
        observation_representation=observation_representation,
    )

    stateful_env = GVStatefulEnvironment(outer_env)
    latent_env_module = make_gv_latent_env_module(stateful_env, latent_type)
    return stateful_env, latent_env_module


def make_gv_latent_env_module(
    env: GVStatefulEnvironment,
    latent_type: str,
) -> LatentEnvironmentModule:
    if latent_type == 'state':
        return StateLatentEnvironmentModule(env)

    if latent_type == 'gv-beacon':
        return BeaconLatentEnvironmentModule(env)

    raise ValueError(f'invalid latent type {latent_type}')


class GVStatefulEnvironment(StatefulEnvironment):
    state_space: gym.spaces.Dict
    action_space: gym.spaces.Discrete
    observation_space: gym.spaces.Dict

    def __init__(self, env: OuterEnv):
        self._gv_outer_env = env
        self.type = EnvironmentType.GV

        assert env.state_representation is not None
        self.state_space = outer_space_to_gym_space(env.state_representation.space)
        self.action_space = gym.spaces.Discrete(env.action_space.num_actions)
        assert env.observation_representation is not None
        self.observation_space = outer_space_to_gym_space(
            env.observation_representation.space
        )

    def seed(self, seed: int | None = None) -> None:
        self._gv_outer_env.inner_env.set_seed(seed)
        self.state_space.seed(seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)

    def reset(self) -> tuple[State, Observation]:
        self._gv_outer_env.reset()
        state = self._gv_outer_env.state
        observation = self._gv_outer_env.observation
        return state, observation

    def step(self, action: Action) -> tuple[State, Observation, float, bool]:
        gv_action = self._gv_outer_env.action_space.int_to_action(action)
        reward, done = self._gv_outer_env.step(gv_action)
        state = self._gv_outer_env.state
        observation = self._gv_outer_env.observation
        return state, observation, reward, done

    def render(self) -> None:
        raise NotImplementedError


class BeaconLatentEnvironmentModule(LatentEnvironmentModule):
    def __init__(self, env: GVStatefulEnvironment):
        self._env = env
        self.latent_type = 'gv-beacon'
        # TODO get better beacon info
        self.latent_space = gym.spaces.Discrete(env.state_space['item'].high[2] + 1)
        env._gv_outer_env.inner_env.col

    # NOTE:  this assumes the latent mapping is synchronous
    def __call__(self, state: State) -> Latent:
        # This assumes that there is only one possible beacon color, and that
        # the beacon color is static throughout an episode
        inner_state_grid = self._env._gv_outer_env.inner_env.state.grid
        beacon_position = next(
            position
            for position in inner_state_grid.area.positions()
            if isinstance(inner_state_grid[position], Beacon)
        )
        y, x = beacon_position.yx
        beacon_color = state['grid'][y, x, 2]
        return beacon_color
