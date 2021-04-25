from __future__ import annotations

import random

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F

import asym_rlpo.generalized_torch as gtorch
from asym_rlpo.data import Batch
from asym_rlpo.policies.base import FullyObservablePolicy
from asym_rlpo.utils.collate import collate_torch

from .base import BatchedDQN


class FOB_DQN(BatchedDQN):
    model_keys = ['state_model', 'qs_model']

    def target_policy(self) -> TargetPolicy:
        return TargetPolicy(self.models, device=self.device)

    def behavior_policy(
        self, action_space: gym.spaces.Discrete
    ) -> BehaviorPolicy:
        return BehaviorPolicy(self.models, action_space, device=self.device)

    def batched_loss(self, batch: Batch, *, discount: float) -> torch.Tensor:

        qs_values = self.models.qs_model(self.models.state_model(batch.states))
        with torch.no_grad():
            target_qs_values = self.target_models.qs_model(
                self.models.state_model(batch.next_states)
            )

        qs_values = qs_values.gather(1, batch.actions.unsqueeze(-1)).squeeze(-1)
        qs_values_bootstrap = torch.tensor(0.0, device=self.device).where(
            batch.dones, target_qs_values.max(-1).values
        )

        loss = F.mse_loss(
            qs_values,
            batch.rewards + discount * qs_values_bootstrap,
        )
        return loss


class TargetPolicy(FullyObservablePolicy):
    def __init__(self, models: nn.ModuleDict, *, device: torch.device):
        super().__init__()
        self.models = models
        self.device = device

    def fo_sample_action(self, state):
        state_batch = gtorch.to(collate_torch([state]), self.device)
        qs_values = self.models.qs_model(self.models.state_model(state_batch))
        return qs_values.squeeze(0).argmax().item()


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
