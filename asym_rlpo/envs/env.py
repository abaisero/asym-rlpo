import enum
from typing import Any, Optional, Protocol, Tuple

import gym
import gym.spaces

State = Any
Action = int
Observation = Any
Latent = Any

# TODO eventually add
# * how to type spaces and their elements more specifically?


class EnvironmentType(enum.Enum):
    OPENAI = enum.auto()
    FLAT = enum.auto()
    GV = enum.auto()
    EXTRA_DECTIGER = enum.auto()
    EXTRA_CLEANER = enum.auto()
    EXTRA_CARFLAG = enum.auto()
    OTHER = enum.auto()


class LatentType(enum.Enum):
    STATE = enum.auto()
    GV_MEMORY = enum.auto()


class Environment(Protocol):
    """primary environment protocol"""

    type: EnvironmentType
    latent_type: LatentType

    action_space: gym.spaces.Discrete
    observation_space: gym.spaces.Space
    latent_space: gym.spaces.Space

    def seed(self, seed: Optional[int] = None) -> None:
        ...

    def reset(self) -> Tuple[Observation, Latent]:
        ...

    def step(self, action: Action) -> Tuple[Observation, Latent, float, bool]:
        ...

    # Kinda want to remove this..?
    def render(self) -> None:
        ...
