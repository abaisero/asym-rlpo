import abc
from typing import List, Sequence

import gym
import torch

from asym_rlpo.data import Batch, Episode
from asym_rlpo.models import make_models
from asym_rlpo.policies.base import Policy


class DQN(metaclass=abc.ABCMeta):
    def __init__(self, env: gym.Env):
        super().__init__()
        self.models = make_models(env, keys=self.model_keys)
        self.target_models = make_models(env, keys=self.model_keys)

    @property
    @abc.abstractmethod
    def model_keys(self) -> List[str]:
        assert False

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
