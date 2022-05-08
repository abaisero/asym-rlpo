from __future__ import annotations

import abc
import random
from typing import Callable, Sequence

import gym
import gym.spaces
import torch

from asym_rlpo.data import Episode
from asym_rlpo.features import HistoryIntegrator
from asym_rlpo.policies import HistoryPolicy

from ..base import Algorithm_ABC


class DQN_ABC(Algorithm_ABC):
    def target_policy(self) -> QhPolicy:
        history_integrator = self.make_history_integrator(
            self.models.agent.action_model,
            self.models.agent.observation_model,
            self.models.agent.history_model,
        )
        return QhPolicy(history_integrator, self.models.agent.qh_model)

    def behavior_policy(
        self, action_space: gym.spaces.Discrete
    ) -> EpsilonGreedyQhPolicy:
        history_integrator = self.make_history_integrator(
            self.models.agent.action_model,
            self.models.agent.observation_model,
            self.models.agent.history_model,
        )
        return EpsilonGreedyQhPolicy(
            history_integrator,
            self.models.agent.qh_model,
            action_space,
        )

    @abc.abstractmethod
    def episodic_loss(
        self,
        episodes: Sequence[Episode],
        *,
        discount: float,
    ) -> torch.Tensor:
        assert False


# policies


# function which maps history features to action-values
QhFunction = Callable[[torch.Tensor], torch.Tensor]


class QhPolicy(HistoryPolicy):
    def __init__(
        self,
        history_integrator: HistoryIntegrator,
        qh_function: QhFunction,
    ):
        super().__init__(history_integrator)
        self.qh_function = qh_function

    def sample_action(self):
        qh_values = self.qh_function(self.history_integrator.features)
        return qh_values.argmax().item()


class EpsilonGreedyQhPolicy(HistoryPolicy):
    def __init__(
        self,
        history_integrator: HistoryIntegrator,
        qh_function: QhFunction,
        action_space: gym.Space,
    ):
        super().__init__(history_integrator)
        self.qh_function = qh_function
        self.action_space = action_space

    def sample_action(self):
        if random.random() < self.epsilon:
            return self.action_space.sample()

        qh_values = self.qh_function(self.history_integrator.features)
        return qh_values.argmax().item()
