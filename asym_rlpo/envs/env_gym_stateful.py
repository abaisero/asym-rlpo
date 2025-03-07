from __future__ import annotations

from typing import Protocol

import gym
import gym.spaces

from asym_rlpo.envs.env import Observation, State


class StatefulGymEnv(Protocol):
    """Protocol based on gym.Env which also contains state_space and state"""

    state_space: gym.spaces.Space
    action_space: gym.spaces.Discrete
    observation_space: gym.spaces.Space

    state: State

    def seed(self, seed=None): ...

    def reset(self) -> Observation: ...

    def step(self, action) -> tuple[Observation, float, bool, dict]: ...

    def render(self, mode='human'): ...
