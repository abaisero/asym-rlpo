from __future__ import annotations

import abc
import random
from typing import Sequence, Union

import gym
import torch
import torch.nn as nn

import asym_rlpo.generalized_torch as gtorch
from asym_rlpo.data import Batch, Episode
from asym_rlpo.features import make_history_integrator
from asym_rlpo.policies.base import (
    FullyObservablePolicy,
    PartiallyObservablePolicy,
    Policy,
)
from asym_rlpo.utils.collate import collate_torch

from ..base import FO_Algorithm_ABC, PO_Algorithm_ABC


class DQN_ABC(metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def episodic_training(self) -> bool:
        assert False

    @property
    @abc.abstractmethod
    def batched_training(self) -> bool:
        assert False

    @abc.abstractmethod
    def target_policy(self) -> Policy:
        assert False

    @abc.abstractmethod
    def behavior_policy(self, action_space: gym.spaces.Discrete) -> Policy:
        assert False


class PO_DQN_ABC(PO_Algorithm_ABC, DQN_ABC):
    def target_policy(self) -> PO_TargetPolicy:
        return PO_TargetPolicy(
            self.models,
            truncated_histories=self.truncated_histories,
            truncated_histories_n=self.truncated_histories_n,
        )

    def behavior_policy(
        self, action_space: gym.spaces.Discrete
    ) -> PO_BehaviorPolicy:
        return PO_BehaviorPolicy(
            self.models,
            action_space,
            truncated_histories=self.truncated_histories,
            truncated_histories_n=self.truncated_histories_n,
        )


class FO_DQN_ABC(FO_Algorithm_ABC, DQN_ABC):
    def target_policy(self) -> FO_TargetPolicy:
        return FO_TargetPolicy(self.models)

    def behavior_policy(
        self, action_space: gym.spaces.Discrete
    ) -> FO_BehaviorPolicy:
        return FO_BehaviorPolicy(self.models, action_space)


class EpisodicDQN_ABC(DQN_ABC):
    episodic_training: bool = True
    batched_training: bool = False

    @abc.abstractmethod
    def episodic_loss(
        self,
        episodes: Sequence[Episode],
        *,
        discount: float,
    ) -> torch.Tensor:
        assert False


class PO_EpisodicDQN_ABC(PO_DQN_ABC, EpisodicDQN_ABC):
    pass


class FO_EpisodicDQN_ABC(FO_DQN_ABC, EpisodicDQN_ABC):
    pass


class FO_BatchedDQN_ABC(FO_DQN_ABC):
    episodic_training: bool = False
    batched_training: bool = True

    @abc.abstractmethod
    def batched_loss(
        self,
        batch: Batch,
        *,
        discount: float,
    ) -> torch.Tensor:
        assert False


EpisodicDQN_ABC = Union[PO_EpisodicDQN_ABC, FO_EpisodicDQN_ABC]
BatchedDQN_ABC = FO_BatchedDQN_ABC


# PO policies


class PO_TargetPolicy(PartiallyObservablePolicy):
    def __init__(
        self,
        models: nn.ModuleDict,
        *,
        truncated_histories: bool,
        truncated_histories_n: int,
    ):
        super().__init__()
        self.models = models

        self.history_integrator = make_history_integrator(
            models.agent.action_model,
            models.agent.observation_model,
            models.agent.history_model,
            truncated_histories=truncated_histories,
            truncated_histories_n=truncated_histories_n,
        )

    def reset(self, observation):
        self.history_integrator.reset(observation)

    def step(self, action, observation):
        self.history_integrator.step(action, observation)

    def po_sample_action(self):
        qh_values = self.models.agent.qh_model(self.history_integrator.features)
        return qh_values.argmax().item()


class PO_BehaviorPolicy(PartiallyObservablePolicy):
    def __init__(
        self,
        models: nn.ModuleDict,
        action_space: gym.Space,
        *,
        truncated_histories: bool,
        truncated_histories_n: int,
    ):
        super().__init__()
        self.target_policy = PO_TargetPolicy(
            models,
            truncated_histories=truncated_histories,
            truncated_histories_n=truncated_histories_n,
        )
        self.action_space = action_space
        self.epsilon: float

    def reset(self, observation):
        self.target_policy.reset(observation)

    def step(self, action, observation):
        self.target_policy.step(action, observation)

    def po_sample_action(self):
        return (
            self.action_space.sample()
            if random.random() < self.epsilon
            else self.target_policy.po_sample_action()
        )


# FO policies


class FO_TargetPolicy(FullyObservablePolicy):
    def __init__(self, models: nn.ModuleDict):
        super().__init__()
        self.models = models

    def fo_sample_action(self, state):
        device = next(self.models.agent.qs_model.parameters()).device
        state_batch = gtorch.to(collate_torch([state]), device)
        qs_values = self.models.agent.qs_model(
            self.models.agent.state_model(state_batch)
        )
        return qs_values.squeeze(0).argmax().item()


class FO_BehaviorPolicy(FullyObservablePolicy):
    def __init__(
        self,
        models: nn.ModuleDict,
        action_space: gym.Space,
    ):
        super().__init__()
        self.target_policy = FO_TargetPolicy(models)
        self.action_space = action_space
        self.epsilon: float

    def fo_sample_action(self, state):
        return (
            self.action_space.sample()
            if random.random() < self.epsilon
            else self.target_policy.fo_sample_action(state)
        )
