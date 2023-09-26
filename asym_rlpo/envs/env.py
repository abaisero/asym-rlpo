import enum
from typing import Any, Protocol, TypeAlias

import gym
import gym.spaces

State: TypeAlias = Any
Action: TypeAlias = int
Observation: TypeAlias = Any
Latent: TypeAlias = Any


class EnvironmentType(enum.Enum):
    OPENAI = enum.auto()
    FLAT = enum.auto()
    GV = enum.auto()
    EXTRA_DECTIGER = enum.auto()
    EXTRA_CLEANER = enum.auto()
    EXTRA_CARFLAG = enum.auto()
    OTHER = enum.auto()


class Environment(Protocol):
    """primary environment protocol"""

    type: EnvironmentType
    latent_type: str

    action_space: gym.spaces.Discrete
    observation_space: gym.spaces.Space
    latent_space: gym.spaces.Space

    def seed(self, seed: int | None = None) -> None:
        ...

    def reset(self) -> tuple[Observation, Latent]:
        ...

    def step(self, action: Action) -> tuple[Observation, Latent, float, bool]:
        ...

    # Kinda want to remove this..?
    def render(self) -> None:
        ...
