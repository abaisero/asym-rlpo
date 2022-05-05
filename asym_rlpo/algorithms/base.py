import abc
import copy
from typing import ClassVar, Dict, List, Union

import torch
import torch.nn as nn

from asym_rlpo.features import HistoryFeaturesComputer, HistoryIntegratorMaker

ModelKeysList = List[str]
ModelKeysDict = Dict[str, ModelKeysList]
ModelKeysDict = Dict[str, Union[ModelKeysDict, ModelKeysList]]


class Algorithm_ABC(metaclass=abc.ABCMeta):
    model_keys: ClassVar[ModelKeysDict]

    def __init__(
        self,
        models: nn.ModuleDict,
        *,
        make_history_integrator: HistoryIntegratorMaker,
        compute_history_features: HistoryFeaturesComputer,
    ):
        super().__init__()
        self.models = models
        self.target_models = copy.deepcopy(models)

        self.make_history_integrator = make_history_integrator
        self.compute_history_features = compute_history_features

    def to(self, device: torch.device):
        self.models.to(device)
        self.target_models.to(device)
