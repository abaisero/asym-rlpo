from __future__ import annotations

from collections import Counter

import gym
import gym.spaces
from gym_gridverse.debugging import reset_gv_debug
from gym_gridverse.envs.yaml.factory import factory_env_from_yaml
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
    Environment,
    EnvironmentType,
    Latent,
    LatentType,
    Observation,
)


def make_gv_env(
    path: str,
    latent_type: LatentType,
    *,
    gv_representation: str,
) -> Environment:
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

    env = GVEnvironment(outer_env)

    if latent_type is LatentType.GV_MEMORY:
        env = GVEnvironment_MEMORY(env)

    return env


class GVEnvironment(Environment):
    def __init__(self, env: OuterEnv):
        self._gv_outer_env = env
        self.type = EnvironmentType.GV
        self.latent_type = LatentType.STATE

        self.action_space = gym.spaces.Discrete(env.action_space.num_actions)
        assert env.state_representation is not None
        self.latent_space = outer_space_to_gym_space(env.state_representation.space)
        assert env.observation_representation is not None
        self.observation_space = outer_space_to_gym_space(
            env.observation_representation.space
        )

    def seed(self, seed: int | None = None) -> None:
        self._gv_outer_env.inner_env.set_seed(seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
        self.latent_space.seed(seed)

    def reset(self) -> tuple[Observation, Latent]:
        self._gv_outer_env.reset()
        latent = self._gv_outer_env.state
        observation = self._gv_outer_env.observation
        return observation, latent

    def step(self, action: Action) -> tuple[Observation, Latent, float, bool]:
        gv_action = self._gv_outer_env.action_space.int_to_action(action)
        reward, done = self._gv_outer_env.step(gv_action)
        latent = self._gv_outer_env.state
        observation = self._gv_outer_env.observation
        return observation, latent, reward, done

    def render(self) -> None:
        raise NotImplementedError


class GVEnvironment_MEMORY(Environment):
    def __init__(self, env: GVEnvironment):
        super().__init__()
        self._env = env
        self.type = env.type
        self.latent_type = LatentType.GV_MEMORY

        self.action_space = env.action_space
        self.observation_space = env.observation_space

        assert isinstance(env.latent_space, gym.spaces.Dict)
        self.latent_space = gym.spaces.Box(
            env.latent_space['item'].low[2],
            env.latent_space['item'].high[2],
            shape=(),
            dtype=env.latent_space['item'].dtype,
        )

    def seed(self, seed: int | None = None) -> None:
        self._env.seed(seed)

    def reset(self) -> tuple[Observation, Latent]:
        observation, latent = self._env.reset()
        counts = Counter(latent['grid'].flatten())
        latent = next(k for k, v in counts.items() if v == 2)
        return observation, latent

    def step(self, action: Action) -> tuple[Observation, Latent, float, bool]:
        observation, latent, reward, done = self._env.step(action)
        counts = Counter(latent['grid'].flatten())
        latent = next(k for k, v in counts.items() if v == 2)
        return observation, latent, reward, done

    def render(self) -> None:
        self._env.render()
