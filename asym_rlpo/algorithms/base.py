import abc

import gym
import torch.nn as nn
from asym_rlpo.data import EpisodeBuffer
from asym_rlpo.policies.base import Policy
from asym_rlpo.utils.debugging import checkraise


class DQN(metaclass=abc.ABCMeta):
    def __init__(self, env: gym.Env):
        self.models = self.make_models(env)
        self.target_models = self.make_models(env)

    @abc.abstractmethod
    def make_models(self, env: gym.Env) -> nn.ModuleDict:
        assert False

    @abc.abstractmethod
    def target_policy(self) -> Policy:
        assert False

    @abc.abstractmethod
    def behavior_policy(self, action_space: gym.spaces.Discrete) -> Policy:
        assert False

    @abc.abstractmethod
    def loss(
        self,
        episode_buffer: EpisodeBuffer,
        *,
        discount: float,
        num_episodes: int,
        batch_size: int,
    ):
        assert False


class EpisodicDQN(DQN):
    def loss(
        self,
        episode_buffer: EpisodeBuffer,
        *,
        discount: float,
        num_episodes: int,
        batch_size: int,  # pylint: disable=unused-argument
    ):
        checkraise(
            num_episodes is not None,
            ValueError,
            f'`num_episodes` ({num_episodes}) should not be None',
        )
        return self.episodic_loss(
            episode_buffer, discount=discount, num_episodes=num_episodes
        )

    @abc.abstractmethod
    def episodic_loss(
        self,
        episode_buffer: EpisodeBuffer,
        *,
        discount: float,
        num_episodes: int,
    ):
        assert False


class BatchedDQN(DQN):
    def loss(
        self,
        episode_buffer: EpisodeBuffer,
        *,
        discount: float,
        num_episodes: int,  # pylint: disable=unused-argument
        batch_size: int,
    ):
        checkraise(
            batch_size is not None,
            ValueError,
            f'`batch_size` ({batch_size}) should not be None',
        )
        return self.batched_loss(
            episode_buffer, discount=discount, batch_size=batch_size
        )

    @abc.abstractmethod
    def batched_loss(
        self,
        episode_buffer: EpisodeBuffer,
        *,
        discount: float,
        batch_size: int,
    ):
        assert False
