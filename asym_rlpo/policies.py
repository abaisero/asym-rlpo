import abc
import random

import gym
import gym.spaces
import torch

from asym_rlpo.models.history import HistoryIntegrator
from asym_rlpo.types import Action, ActionValueFunction, PolicyFunction


class Policy(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def reset(self, observation):
        assert False

    @abc.abstractmethod
    def step(self, action, observation):
        assert False

    @abc.abstractmethod
    def sample_action(self) -> tuple[Action, dict]:
        assert False


class RandomPolicy(Policy):
    def __init__(self, action_space: gym.spaces.Discrete):
        super().__init__()
        self.action_space = action_space

    def reset(self, observation):
        pass

    def step(self, action, observation):
        pass

    def sample_action(self) -> tuple[Action, dict]:
        action = self.action_space.sample()
        info = {}
        return action, info


class HistoryPolicy(Policy):
    def __init__(self, history_integrator: HistoryIntegrator):
        super().__init__()
        self.history_integrator = history_integrator

    def reset(self, observation):
        self.history_integrator.reset(observation)

    def step(self, action, observation):
        self.history_integrator.step(action, observation)


class StochasticHistoryPolicy(HistoryPolicy):
    def __init__(
        self,
        history_integrator: HistoryIntegrator,
        policy_function: PolicyFunction,
    ):
        super().__init__(history_integrator)
        self.policy_function = policy_function

    def sample_action(self) -> tuple[Action, dict]:
        (
            history_features,
            info,
        ) = self.history_integrator.sample_features()
        action_logits = self.policy_function(history_features)
        action_dist = torch.distributions.Categorical(logits=action_logits)
        action = int(action_dist.sample().item())
        info = {**info}
        return action, info


class EpsilonGreedyStochasticHistoryPolicy(HistoryPolicy):
    def __init__(
        self,
        history_integrator: HistoryIntegrator,
        policy_function: PolicyFunction,
    ):
        super().__init__(history_integrator)
        self.policy_function = policy_function
        self.epsilon: float = 1.0

    def sample_action(self) -> tuple[Action, dict]:
        (
            history_features,
            info,
        ) = self.history_integrator.sample_features()
        action_logits = self.policy_function(history_features)

        if random.random() < self.epsilon:
            action_dist = torch.distributions.Categorical(logits=action_logits)
            action = int(action_dist.sample().item())
        else:
            action = int(action_logits.argmax().item())

        info = {**info}
        return action, info


class GreedyHistoryPolicy(HistoryPolicy):
    def __init__(
        self,
        history_integrator: HistoryIntegrator,
        value_function: ActionValueFunction,
    ):
        super().__init__(history_integrator)
        self.value_function = value_function

    def sample_action(self) -> tuple[Action, dict]:
        (
            history_features,
            info,
        ) = self.history_integrator.sample_features()
        values = self.value_function(history_features)
        action = int(values.argmax().item())
        info = {**info}
        return action, info


class EpsilonGreedyHistoryPolicy(HistoryPolicy):
    def __init__(
        self,
        history_integrator: HistoryIntegrator,
        value_function: ActionValueFunction,
        action_space: gym.spaces.Discrete,
    ):
        super().__init__(history_integrator)
        self.value_function = value_function
        self.action_space = action_space
        self.epsilon: float = 1.0

    def sample_action(self) -> tuple[Action, dict]:
        (
            history_features,
            info,
        ) = self.history_integrator.sample_features()

        if random.random() < self.epsilon:
            action = self.action_space.sample()
        else:
            values = self.value_function(history_features)
            action = int(values.argmax().item())

        info = {**info}
        return action, info
