from collections.abc import Callable, Iterable
from typing import TypeAlias

import torch
import torch.nn as nn

Action: TypeAlias = int
Memory: TypeAlias = int

OptimizerFactory: TypeAlias = Callable[
    [Iterable[nn.Parameter]],
    torch.optim.Optimizer,
]
OptimizerDict: TypeAlias = dict[str, torch.optim.Optimizer]
LossDict: TypeAlias = dict[str, torch.Tensor]
ParametersDict: TypeAlias = dict
GradientNormDict: TypeAlias = dict[str, torch.Tensor]

CategoricalFeatures: TypeAlias = torch.Tensor
Features: TypeAlias = torch.Tensor
InteractionFeatures: TypeAlias = Features
HistoryFeatures: TypeAlias = Features

Loss: TypeAlias = torch.Tensor

ActionLogits: TypeAlias = torch.Tensor
Values: TypeAlias = torch.Tensor
ActionValues: TypeAlias = torch.Tensor

PolicyFunction: TypeAlias = Callable[[HistoryFeatures], ActionLogits]
ActionValueFunction: TypeAlias = Callable[[HistoryFeatures], ActionValues]
