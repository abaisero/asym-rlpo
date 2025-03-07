from __future__ import annotations

import re
from typing import Protocol

import gym
import gym.spaces

from asym_rlpo.envs.env_gym_stateful import StatefulGymEnv
from asym_rlpo.envs.env import (
    Action,
    Environment,
    EnvironmentType,
    Latent,
    LatentType,
    Observation,
)


def id_is_heavenhell(id: str) -> bool:
    return bool(re.fullmatch(r'POMDP-heavenhell_\d+-(episodic|continuing)-v\d+', id))


class LatentFunction(Protocol):
    def __call__(self, state: int, *, num_positions: int) -> int: ...


def latent_state(state: int, *, num_positions: int) -> int:
    return state


def latent_hh_heaven(state: int, *, num_positions: int) -> int:
    latent_exit, _ = divmod(state, num_positions)
    return latent_exit


def latent_hh_position(state: int, *, num_positions: int) -> int:
    _, latent_position = divmod(state, num_positions)
    return latent_position


def make_latent_function(latent_type: LatentType) -> LatentFunction:
    if latent_type is LatentType.STATE:
        return latent_state

    if latent_type is LatentType.HH_HEAVEN:
        return latent_hh_heaven

    if latent_type is LatentType.HH_POSITION:
        return latent_hh_position

    raise ValueError(f'invalid latent type {latent_type}')


def make_latent_space(
    latent_type: LatentType, *, num_states: int
) -> gym.spaces.Discrete:
    if latent_type is LatentType.STATE:
        return gym.spaces.Discrete(num_states)

    if latent_type is LatentType.HH_HEAVEN:
        return gym.spaces.Discrete(2)

    if latent_type is LatentType.HH_POSITION:
        num_positions = num_states // 2
        return gym.spaces.Discrete(num_positions)

    raise ValueError(f'invalid latent type {latent_type}')


class GymEnvironment_HH(Environment):
    """Converts HH environment to the Environment protocol"""

    def __init__(self, env: StatefulGymEnv, latent_type: LatentType):
        self._env = env
        self.type = EnvironmentType.FLAT
        self.latent_type = latent_type
        self.action_space: gym.spaces.Discrete = env.action_space
        self.observation_space: gym.spaces.Space = env.observation_space

        assert isinstance(env.state_space, gym.spaces.Discrete)
        self.num_states = env.state_space.n
        self.num_positions = self.num_states // 2
        self.latent_space = make_latent_space(latent_type, num_states=self.num_states)
        self.latent = make_latent_function(latent_type)

    def seed(self, seed: int | None = None) -> None:
        self._env.seed(seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
        self.latent_space.seed(seed)

    def reset(self) -> tuple[Observation, Latent]:
        observation = self._env.reset()
        latent = self.latent(self._env.state, num_positions=self.num_positions)
        return observation, latent

    def step(self, action: Action) -> tuple[Observation, Latent, float, bool]:
        observation, reward, done, _ = self._env.step(action)
        latent = self.latent(self._env.state, num_positions=self.num_positions)
        return observation, latent, reward, done

    def render(self) -> None:
        self._env.render()
