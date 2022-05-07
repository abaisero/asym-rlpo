import enum
from typing import Any, Optional, Protocol, Tuple

import gym
import gym.spaces

State = Any
Action = int
Observation = Any

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


class Environment(Protocol):
    """primary environment protocol"""

    type: EnvironmentType
    state_space: gym.spaces.Space
    action_space: gym.spaces.Discrete
    observation_space: gym.spaces.Space

    def seed(self, seed: Optional[int] = None) -> None:
        ...

    def reset(self) -> Tuple[State, Observation]:
        ...

    def step(self, action: Action) -> Tuple[State, Observation, float, bool]:
        ...

    # Kinda want to remove this..?
    def render(self) -> None:
        ...
