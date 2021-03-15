import random
from typing import Sequence

import gym
import torch
import torch.nn as nn
from asym_rlpo.data import Episode, EpisodeBuffer
from asym_rlpo.policies import Policy
from asym_rlpo.representations.embedding import EmbeddingRepresentation
from asym_rlpo.representations.gv import GV_ObservationRepresentation
from asym_rlpo.representations.history import RNNHistoryRepresentation

from .base import Algorithm


class DQN(Algorithm):
    def __init__(self, env: gym.Env):
        super().__init__()
        self.env = env
        self.episode_buffer = EpisodeBuffer(maxlen=1_000)

        self.action_model = EmbeddingRepresentation(env.action_space.n, 64)
        self.observation_model = GV_ObservationRepresentation(
            env.observation_space
        )
        self.history_model = RNNHistoryRepresentation(
            self.action_model,
            self.observation_model,
            hidden_size=128,
        )
        self.q_model = nn.Linear(self.history_model.dim, env.action_space.n)

        self.models = nn.ModuleDict(
            {
                'action_model': self.action_model,
                'observation_model': self.observation_model,
                'history_model': self.history_model,
                'q_model': self.q_model,
            }
        )

    def process(self, episodes: Sequence[Episode]):
        self.episode_buffer.append_episodes(episodes)

        print(
            f'episode_buffer stats - '
            f'#interactions: {self.episode_buffer.num_interactions()}'
            f'\t#episodes: {self.episode_buffer.num_episodes()}'
        )

        self.episode_buffer.sample_episode()

    def behavior_policy(self) -> Policy:
        return DQN_BehaviorPolicy(self.models, self.env.action_space)

    def target_policy(self) -> Policy:
        return DQN_TargetPolicy(self.models, self.env.action_space)


class DQN_BehaviorPolicy(Policy):
    # TODO implement and instantiate the epsilon-greedy policy

    def __init__(self, models: nn.ModuleDict, action_space: gym.Space):
        super().__init__()
        self.target_policy = DQN_TargetPolicy(models, action_space)
        self.action_space = action_space

    def reset(self, observation):
        self.target_policy.reset(observation)

    def step(self, action, observation):
        self.target_policy.step(action, observation)

    def sample_action(self):
        # TODO how to inject non-constant epsilon
        return (
            self.target_policy.sample_action()
            if random.random() > 0.1
            else self.action_space.sample()
        )


class DQN_TargetPolicy(Policy):
    # TODO implement and instantiate the argmax policy

    def __init__(self, models: nn.ModuleDict, action_space: gym.Space):
        super().__init__()
        self.models = models
        self.action_space = action_space

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

    def sample_action(self):
        q_values = self.models.q_model(self.history_features)
        return q_values.argmax().item()
