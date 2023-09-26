import enum
from collections.abc import Callable
from typing import TypeAlias

import torch.nn as nn

from asym_rlpo.envs import Environment
from asym_rlpo.models.model import Model

ModelMaker: TypeAlias = Callable[[Environment], Model]

PolicyModule: TypeAlias = nn.Module
VModule: TypeAlias = nn.Module
QModule: TypeAlias = nn.Module
UModule: TypeAlias = nn.Module


class CriticType(enum.Enum):
    H = enum.auto()
    HZ = enum.auto()
    Z = enum.auto()


class QModelType(enum.Enum):
    H = enum.auto()
    HZ = enum.auto()
    Z = enum.auto()
