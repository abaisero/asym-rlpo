import abc
from typing import Sequence

import gym
import torch
import torch.nn as nn

from asym_rlpo.data import Batch, Episode
from asym_rlpo.policies.base import Policy


class DQN(metaclass=abc.ABCMeta):
    def __init__(self, env: gym.Env):
        self.models = self.make_models(env)
        self.target_models = self.make_models(env)

    @property
    @abc.abstractmethod
    def episodic_training(self) -> bool:
        assert False

    @property
    @abc.abstractmethod
    def batched_training(self) -> bool:
        assert False

    @abc.abstractmethod
    def make_models(self, env: gym.Env) -> nn.ModuleDict:
        assert False

    @abc.abstractmethod
    def target_policy(self) -> Policy:
        assert False

    @abc.abstractmethod
    def behavior_policy(self, action_space: gym.spaces.Discrete) -> Policy:
        assert False


class EpisodicDQN(DQN):
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


class BatchedDQN(DQN):
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
