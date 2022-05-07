from __future__ import annotations

from typing import Optional, Tuple

import gym
import gym.spaces
from gym_gridverse.debugging import reset_gv_debug
from gym_gridverse.envs.inner_env import InnerEnv
from gym_gridverse.envs.yaml.factory import factory_env_from_yaml
from gym_gridverse.gym import outer_space_to_gym_space
from gym_gridverse.representations.observation_representations import (
    make_observation_representation,
)
from gym_gridverse.representations.representation import (
    ObservationRepresentation,
    StateRepresentation,
)
from gym_gridverse.representations.state_representations import (
    make_state_representation,
)

from asym_rlpo.utils.config import get_config

from .env import Action, Environment, EnvironmentType, Observation, State


def make_gv_env(path: str) -> Environment:
    reset_gv_debug(False)

    config = get_config()

    print('Loading using YAML')
    inner_env = factory_env_from_yaml(path)
    state_representation = make_state_representation(
        config.gv_state_representation,
        inner_env.state_space,
    )
    observation_representation = make_observation_representation(
        config.gv_observation_representation,
        inner_env.observation_space,
    )
    return GVEnvironment(
        inner_env,
        state_representation,
        observation_representation,
    )


class GVEnvironment:
    def __init__(
        self,
        env: InnerEnv,
        state_representation: StateRepresentation,
        observation_representation: ObservationRepresentation,
    ):
        self._gv_inner_env = env
        self._gv_state_representation = state_representation
        self._gv_observation_representation = observation_representation
        self.type = EnvironmentType.GV

        self.state_space = outer_space_to_gym_space(state_representation.space)
        self.action_space = gym.spaces.Discrete(env.action_space.num_actions)
        self.observation_space = outer_space_to_gym_space(
            observation_representation.space
        )

    def seed(self, seed: Optional[int] = None) -> None:
        self._gv_inner_env.set_seed(seed)
        self.state_space.seed(seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)

    def reset(self) -> Tuple[State, Observation]:
        self._gv_inner_env.reset()
        state = self._gv_state_representation.convert(self._gv_inner_env.state)
        observation = self._gv_observation_representation.convert(
            self._gv_inner_env.observation
        )
        return state, observation

    def step(self, action: Action) -> Tuple[State, Observation, float, bool]:
        gv_action = self._gv_inner_env.action_space.int_to_action(action)
        reward, done = self._gv_inner_env.step(gv_action)
        state = self._gv_state_representation.convert(self._gv_inner_env.state)
        observation = self._gv_observation_representation.convert(
            self._gv_inner_env.observation
        )
        return state, observation, reward, done

    def render(self) -> None:
        # TODO implement, maybe?  maybe not
        pass
