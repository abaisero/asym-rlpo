from __future__ import annotations

import abc
import random
from typing import Sequence, Union

import gym
import gym.spaces
import torch
import torch.nn as nn

import asym_rlpo.generalized_torch as gtorch
from asym_rlpo.data import Batch, Episode
from asym_rlpo.features import HistoryIntegrator, make_history_integrator
from asym_rlpo.policies import FullyObservablePolicy, HistoryPolicy, Policy
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
    def target_policy(self) -> QhPolicy:
        history_integrator = make_history_integrator(
            self.models.agent.action_model,
            self.models.agent.observation_model,
            self.models.agent.history_model,
            truncated_histories=self.truncated_histories,
            truncated_histories_n=self.truncated_histories_n,
        )
        return QhPolicy(history_integrator, self.models.agent.qh_model)

    def behavior_policy(
        self, action_space: gym.spaces.Discrete
    ) -> EpsilonGreedyQhPolicy:
        history_integrator = make_history_integrator(
            self.models.agent.action_model,
            self.models.agent.observation_model,
            self.models.agent.history_model,
            truncated_histories=self.truncated_histories,
            truncated_histories_n=self.truncated_histories_n,
        )
        return EpsilonGreedyQhPolicy(
            history_integrator,
            self.models.agent.qh_model,
            action_space,
        )


class FO_DQN_ABC(FO_Algorithm_ABC, DQN_ABC):
    def target_policy(self) -> QsPolicy:
        return QsPolicy(
            self.models.agent.state_model,
            self.models.agent.qs_model,
        )

    def behavior_policy(
        self, action_space: gym.spaces.Discrete
    ) -> EpsilonGreedyQsPolicy:
        return EpsilonGreedyQsPolicy(
            self.models.agent.state_model,
            self.models.agent.qs_model,
            action_space,
        )


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


class QhPolicy(HistoryPolicy):
    def __init__(
        self,
        history_integrator: HistoryIntegrator,
        qh_model: nn.Module,
    ):
        super().__init__(history_integrator)
        self.qh_model = qh_model

    def po_sample_action(self):
        qh_values = self.qh_model(self.history_integrator.features)
        return qh_values.argmax().item()


class EpsilonGreedyQhPolicy(HistoryPolicy):
    def __init__(
        self,
        history_integrator: HistoryIntegrator,
        qh_model: nn.Module,
        action_space: gym.Space,
    ):
        super().__init__(history_integrator)
        self.qh_model = qh_model
        self.action_space = action_space

    def po_sample_action(self):
        if random.random() < self.epsilon:
            return self.action_space.sample()

        qh_values = self.qh_model(self.history_integrator.features)
        return qh_values.argmax().item()


# FO policies


class QsPolicy(FullyObservablePolicy):
    def __init__(self, state_model: nn.ModuleDict, qs_model: nn.ModuleDict):
        super().__init__()
        self.state_model = state_model
        self.qs_model = qs_model

    def fo_sample_action(self, state):
        device = next(self.qs_model.parameters()).device
        state_batch = gtorch.to(collate_torch([state]), device)
        qs_values = self.qs_model(self.state_model(state_batch))
        return qs_values.squeeze(0).argmax().item()


class EpsilonGreedyQsPolicy(FullyObservablePolicy):
    def __init__(
        self,
        state_model: nn.Module,
        qs_model: nn.Module,
        action_space: gym.Space,
    ):
        super().__init__()
        self.state_model = state_model
        self.qs_model = qs_model
        self.action_space = action_space

    def fo_sample_action(self, state):
        if random.random() < self.epsilon:
            return self.action_space.sample()

        device = next(self.qs_model.parameters()).device
        state_batch = gtorch.to(collate_torch([state]), device)
        qs_values = self.qs_model(self.state_model(state_batch))
        return qs_values.squeeze(0).argmax().item()
