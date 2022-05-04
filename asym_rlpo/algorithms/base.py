import abc
import copy
from typing import Dict, List, Union

import torch
import torch.nn as nn

from asym_rlpo.data import Torch_O
from asym_rlpo.envs import Environment
from asym_rlpo.features import compute_history_features
from asym_rlpo.models import make_models

ModelKeysList = List[str]
ModelKeysDict = Dict[str, ModelKeysList]
ModelKeysDict = Dict[str, Union[ModelKeysDict, ModelKeysList]]


class Algorithm_ABC(metaclass=abc.ABCMeta):
    def __init__(self, env: Environment):
        super().__init__()

        self.models = make_models(env, keys=self.model_keys)
        self.target_models = copy.deepcopy(self.models)
        self.device = next(self.models.parameters()).device

    @property
    @abc.abstractmethod
    def model_keys(self) -> ModelKeysDict:
        assert False

    def to(self, device: torch.device):
        self.models.to(device)
        self.target_models.to(device)
        self.device = device


class FO_Algorithm_ABC(Algorithm_ABC):
    pass


class PO_Algorithm_ABC(Algorithm_ABC):
    def __init__(
        self,
        env: Environment,
        *,
        truncated_histories: bool,
        truncated_histories_n: int
    ):
        super().__init__(env)
        self.truncated_histories = truncated_histories
        self.truncated_histories_n = truncated_histories_n

    def compute_history_features(
        self,
        action_model: nn.Module,
        observation_model: nn.Module,
        history_model: nn.Module,
        actions: torch.Tensor,
        observations: Torch_O,
    ) -> torch.Tensor:

        return compute_history_features(
            action_model,
            observation_model,
            history_model,
            actions,
            observations,
            truncated=self.truncated_histories,
            n=self.truncated_histories_n,
        )
