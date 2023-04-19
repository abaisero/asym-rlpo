import enum
from typing import TypeAlias

import torch.nn as nn

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
