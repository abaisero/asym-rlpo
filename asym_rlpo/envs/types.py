import enum
from typing import Any, TypeAlias

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
