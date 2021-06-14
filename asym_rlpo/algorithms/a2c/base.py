import abc
import random
from typing import List, Optional, TypedDict

import gym
import torch
import torch.nn as nn

import asym_rlpo.generalized_torch as gtorch
from asym_rlpo.data import Episode
from asym_rlpo.models import make_models
from asym_rlpo.policies.base import PartiallyObservablePolicy
from asym_rlpo.q_estimators import Q_Estimator


class LossesDict(TypedDict):
    actor: torch.Tensor
    critic: torch.Tensor
    negentropy: torch.Tensor


class A2C_Base(metaclass=abc.ABCMeta):
    def __init__(self, env: gym.Env):
        self.models = make_models(env, keys=self.model_keys)
        self.target_models = make_models(env, keys=self.model_keys)
        self.device = next(self.models.parameters()).device

    def to(self, device: torch.device):
        self.models.to(device)
        self.target_models.to(device)
        self.device = device

    @property
    @abc.abstractmethod
    def model_keys(self) -> List[str]:
        assert False

    def behavior_policy(self) -> PartiallyObservablePolicy:
        return BehaviorPolicy(self.models, device=self.device)

    def evaluation_policy(self) -> PartiallyObservablePolicy:
        return EvaluationPolicy(self.models, device=self.device)

    @abc.abstractmethod
    def losses(
        self,
        episode: Episode,
        *,
        discount: float,
        q_estimator: Optional[Q_Estimator] = None
    ) -> LossesDict:
        raise NotImplementedError


class BehaviorPolicy(PartiallyObservablePolicy):
    def __init__(self, models: nn.ModuleDict, *, device: torch.device):
        super().__init__()
        self.models = models
        self.device = device

        self.history_features = None
        self.hidden = None

    def reset(self, observation):
        action_features = torch.zeros(
            1, self.models.action_model.dim, device=self.device
        )
        observation_features = self.models.observation_model(
            gtorch.to(gtorch.unsqueeze(observation, 0), self.device)
        )
        self._update(action_features, observation_features)

    def step(self, action, observation):
        action_features = self.models.action_model(
            action.unsqueeze(0).to(self.device)
        )
        observation_features = self.models.observation_model(
            gtorch.to(gtorch.unsqueeze(observation, 0), self.device)
        )
        self._update(action_features, observation_features)

    def _update(self, action_features, observation_features):
        input_features = torch.cat(
            [action_features, observation_features], dim=-1
        ).unsqueeze(1)
        self.history_features, self.hidden = self.models.history_model(
            input_features, hidden=self.hidden
        )
        self.history_features = self.history_features.squeeze(0).squeeze(0)

    def action_logits(self):
        return self.models.policy_model(self.history_features)

    def po_sample_action(self):
        action_dist = torch.distributions.Categorical(
            logits=self.action_logits()
        )
        return action_dist.sample().item()


class EvaluationPolicy(PartiallyObservablePolicy):
    def __init__(self, models: nn.ModuleDict, *, device: torch.device):
        super().__init__()
        self.behavior_policy = BehaviorPolicy(models, device=device)
        self.models = models
        self.device = device
        self.epsilon: float

    def reset(self, observation):
        self.behavior_policy.reset(observation)

    def step(self, action, observation):
        self.behavior_policy.step(action, observation)

    def po_sample_action(self):
        action_logits = self.behavior_policy.action_logits()
        return (
            torch.distributions.Categorical(logits=action_logits).sample()
            if random.random() < self.epsilon
            else action_logits.argmax()
        ).item()
