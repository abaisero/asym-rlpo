from __future__ import annotations

import random
from typing import Sequence

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F

import asym_rlpo.generalized_torch as gtorch
from asym_rlpo.data import Episode
from asym_rlpo.policies.base import FullyObservablePolicy
from asym_rlpo.utils.collate import collate_torch

from .base import EpisodicDQN


class FOE_DQN(EpisodicDQN):
    model_keys = ['state_model', 'q_model']

    def target_policy(self) -> TargetPolicy:
        return TargetPolicy(self.models, device=self.device)

    def behavior_policy(
        self, action_space: gym.spaces.Discrete
    ) -> BehaviorPolicy:
        return BehaviorPolicy(self.models, action_space, device=self.device)

    def episodic_loss(
        self, episodes: Sequence[Episode], *, discount: float
    ) -> torch.Tensor:

        losses = []
        for episode in episodes:

            q_values = self.models.q_model(
                self.models.state_model(episode.states)
            )
            with torch.no_grad():
                target_q_values = self.target_models.q_model(
                    self.models.state_model(episode.states)
                )

            q_values = q_values.gather(
                1, episode.actions.unsqueeze(-1)
            ).squeeze(-1)
            q_values_bootstrap = torch.tensor(0.0, device=self.device).where(
                episode.dones, target_q_values.max(-1).values.roll(-1, 0)
            )

            loss = F.mse_loss(
                q_values,
                episode.rewards + discount * q_values_bootstrap,
            )
            losses.append(loss)

        return sum(losses, start=torch.tensor(0.0, device=self.device)) / len(
            losses
        )
        # return sum(losses, start=torch.tensor(0.0)) / sum(
        #     len(episode) for episode in episodes
        # )


class TargetPolicy(FullyObservablePolicy):
    def __init__(self, models: nn.ModuleDict, *, device: torch.device):
        super().__init__()
        self.models = models
        self.device = device

    def fo_sample_action(self, state):
        state_batch = gtorch.to(collate_torch([state]), self.device)
        q_values = self.models.q_model(self.models.state_model(state_batch))
        return q_values.squeeze(0).argmax().item()


class BehaviorPolicy(FullyObservablePolicy):
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

    def fo_sample_action(self, state):
        return (
            self.action_space.sample()
            if random.random() < self.epsilon
            else self.target_policy.fo_sample_action(state)
        )
