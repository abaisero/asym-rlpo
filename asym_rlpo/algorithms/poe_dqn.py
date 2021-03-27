from __future__ import annotations

import random
import re
from typing import Sequence

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from asym_rlpo.data import Episode
from asym_rlpo.modules import make_module
from asym_rlpo.policies.base import PartiallyObservablePolicy
from asym_rlpo.representations.embedding import EmbeddingRepresentation
from asym_rlpo.representations.gv import GV_ObservationRepresentation
from asym_rlpo.representations.history import RNNHistoryRepresentation
from asym_rlpo.representations.identity import IdentityRepresentation
from asym_rlpo.representations.mlp import MLPRepresentation
from asym_rlpo.representations.onehot import OneHotRepresentation

from .base import EpisodicDQN


class POE_DQN(EpisodicDQN):
    def make_models(self, env: gym.Env) -> nn.ModuleDict:
        if re.fullmatch(r'CartPole-v\d+', env.spec.id):
            return make_models_cartpole(env)

        # if ###:
        #     return make_models_gv(env)

        raise NotImplementedError

    def target_policy(self) -> TargetPolicy:
        return TargetPolicy(self.models)

    def behavior_policy(
        self, action_space: gym.spaces.Discrete
    ) -> BehaviorPolicy:
        return BehaviorPolicy(self.models, action_space)

    def episodic_loss(
        self, episodes: Sequence[Episode], *, discount: float
    ) -> torch.Tensor:
        def compute_q_values(models, actions, observations):
            action_features = models.action_model(actions)
            action_features = action_features.roll(1, 0)
            action_features[0, :] = 0.0
            observation_features = models.observation_model(observations)
            inputs = torch.cat([action_features, observation_features], dim=-1)
            history_features, _ = models.history_model(inputs.unsqueeze(0))
            history_features = history_features.squeeze(0)
            q_values = models.q_model(history_features)
            return q_values

        losses = []
        for episode in episodes:

            q_values = compute_q_values(
                self.models, episode.actions, episode.observations
            )
            with torch.no_grad():
                target_q_values = compute_q_values(
                    self.target_models, episode.actions, episode.observations
                )

            q_values = q_values.gather(
                1, episode.actions.unsqueeze(-1)
            ).squeeze(-1)
            q_values_bootstrap = torch.tensor(0.0).where(
                episode.dones, target_q_values.max(-1).values.roll(-1, 0)
            )
            loss = F.mse_loss(
                q_values,
                episode.rewards + discount * q_values_bootstrap,
            )
            losses.append(loss)

        return sum(losses, start=torch.tensor(0.0)) / len(losses)
        # return sum(losses, start=torch.tensor(0.0)) / sum(
        #     len(episode) for episode in episodes
        # )


class TargetPolicy(PartiallyObservablePolicy):
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
        q_values = self.models.q_model(self.history_features)
        return q_values.argmax().item()


class BehaviorPolicy(PartiallyObservablePolicy):
    def __init__(self, models: nn.ModuleDict, action_space: gym.Space):
        super().__init__()
        self.target_policy = TargetPolicy(models)
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


def make_models_cartpole(env: gym.Env) -> nn.ModuleDict:
    # action_model = EmbeddingRepresentation(env.action_space.n, 128)
    # observation_model = MLPRepresentation(env.observation_space, 128)

    action_model = OneHotRepresentation(env.action_space)
    observation_model = IdentityRepresentation(env.observation_space)

    history_model = RNNHistoryRepresentation(
        action_model,
        observation_model,
        hidden_size=128,
        nonlinearity='tanh',
    )
    q_model = nn.Sequential(
        make_module('linear', 'leaky_relu', history_model.dim, 512),
        nn.LeakyReLU(),
        make_module('linear', 'leaky_relu', 512, 256),
        nn.LeakyReLU(),
        make_module('linear', 'linear', 256, env.action_space.n),
    )
    return nn.ModuleDict(
        {
            'action_model': action_model,
            'observation_model': observation_model,
            'history_model': history_model,
            'q_model': q_model,
        }
    )


def make_models_gv(env: gym.Env) -> nn.ModuleDict:
    raise NotImplementedError
    # action_model = EmbeddingRepresentation(env.action_space.n, 64)
    # observation_model = GV_ObservationRepresentation(env.observation_space)
    # history_model = RNNHistoryRepresentation(
    #     action_model,
    #     observation_model,
    #     hidden_size=128,
    # )
    # q_model = nn.Sequential(
    #     nn.Linear(history_model.dim, 128),
    #     nn.ReLU(),
    #     nn.Linear(128, 128),
    #     nn.ReLU(),
    #     nn.Linear(128, env.action_space.n),
    # )
    # models = nn.ModuleDict(
    #     {
    #         'action_model': action_model,
    #         'observation_model': observation_model,
    #         'history_model': history_model,
    #         'q_model': q_model,
    #     }
    # )
