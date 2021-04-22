import abc
from typing import FrozenSet, TypedDict

import gym
import torch
import torch.nn as nn

from asym_rlpo.data import Episode
from asym_rlpo.models import make_models
from asym_rlpo.policies.base import PartiallyObservablePolicy


class LossesDict(TypedDict):
    actor: torch.Tensor
    critic: torch.Tensor
    negentropy: torch.Tensor


class A2C(metaclass=abc.ABCMeta):
    def __init__(self, env: gym.Env):
        self.models = make_models(env, keys=self.model_keys)

    @property
    @abc.abstractmethod
    def model_keys(self) -> FrozenSet[str]:
        assert False

    def policy(self) -> PartiallyObservablePolicy:
        return ActorPolicy(self.models)

    @abc.abstractmethod
    def losses(self, episode: Episode, *, discount: float) -> LossesDict:
        raise NotImplementedError


class ActorPolicy(PartiallyObservablePolicy):
    def __init__(self, models: nn.ModuleDict):
        super().__init__()
        self.models = models

        self.history_features = None
        self.hidden = None

    def reset(self, observation):
        action_features = torch.zeros(self.models.action_model.dim)
        observation_features = self.models.observation_model(observation)
        self._update(action_features, observation_features)

    def step(self, action, observation):
        action_features = self.models.action_model(action)
        observation_features = self.models.observation_model(observation)
        self._update(action_features, observation_features)

    def _update(self, action_features, observation_features):
        input_features = (
            torch.cat([action_features, observation_features])
            .unsqueeze(0)
            .unsqueeze(0)
        )
        self.history_features, self.hidden = self.models.history_model(
            input_features, hidden=self.hidden
        )
        self.history_features = self.history_features.squeeze(0).squeeze(0)

    def po_sample_action(self):
        action_logits = self.models.policy_model(self.history_features)
        action_dist = torch.distributions.Categorical(logits=action_logits)
        return action_dist.sample().item()
