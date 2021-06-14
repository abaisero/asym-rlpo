from __future__ import annotations

import random
from typing import Sequence

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F

import asym_rlpo.generalized_torch as gtorch
from asym_rlpo.data import Episode, Torch_O
from asym_rlpo.features import compute_history_features
from asym_rlpo.policies.base import PartiallyObservablePolicy

from .base import EpisodicDQN


class DQN(EpisodicDQN):
    model_keys = [
        'action_model',
        'observation_model',
        'history_model',
        'qh_model',
    ]

    def target_policy(self) -> TargetPolicy:
        return TargetPolicy(self.models, device=self.device)

    def behavior_policy(
        self, action_space: gym.spaces.Discrete
    ) -> BehaviorPolicy:
        return BehaviorPolicy(self.models, action_space, device=self.device)

    @staticmethod
    def compute_q_values(
        models: nn.ModuleDict,
        actions: torch.Tensor,
        observations: Torch_O,
    ) -> torch.Tensor:
        history_features = compute_history_features(
            models, actions, observations
        )
        qh_values = models.qh_model(history_features)
        return qh_values

    def episodic_loss(
        self, episodes: Sequence[Episode], *, discount: float
    ) -> torch.Tensor:

        losses = []
        for episode in episodes:

            q_values = self.compute_q_values(
                self.models, episode.actions, episode.observations
            )
            with torch.no_grad():
                target_q_values = self.compute_q_values(
                    self.target_models, episode.actions, episode.observations
                )

            q_values = q_values.gather(
                1, episode.actions.unsqueeze(-1)
            ).squeeze(-1)
            q_values_bootstrap = target_q_values.max(-1).values.roll(-1, 0)
            q_values_bootstrap[-1] = 0.0
            loss = F.mse_loss(
                q_values,
                episode.rewards + discount * q_values_bootstrap,
            )
            losses.append(loss)

        return sum(losses) / len(losses)  # type: ignore


class TargetPolicy(PartiallyObservablePolicy):
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

    def po_sample_action(self):
        qh_values = self.models.qh_model(self.history_features)
        return qh_values.argmax().item()


class BehaviorPolicy(PartiallyObservablePolicy):
    def __init__(
        self,
        models: nn.ModuleDict,
        action_space: gym.Space,
        *,
        device: torch.device
    ):
        super().__init__()
        self.target_policy = TargetPolicy(models, device=device)
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
