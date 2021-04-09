from __future__ import annotations

import random

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F

from asym_rlpo.data import Batch
from asym_rlpo.models import make_models
from asym_rlpo.policies.base import FullyObservablePolicy

from .base import BatchedDQN


class FOB_DQN(BatchedDQN):
    """FOB_DQN.

    Fully observable batched DQN
    """

    def make_models(self, env: gym.Env) -> nn.ModuleDict:
        return make_models(env)

    def target_policy(self) -> TargetPolicy:
        return TargetPolicy(self.models)

    def behavior_policy(
        self, action_space: gym.spaces.Discrete
    ) -> BehaviorPolicy:
        return BehaviorPolicy(self.models, action_space)

    def batched_loss(self, batch: Batch, *, discount: float) -> torch.Tensor:

        q_values = self.models.q_model(batch.states)
        with torch.no_grad():
            target_q_values = self.target_models.q_model(batch.next_states)

        q_values = q_values.gather(1, batch.actions.unsqueeze(-1)).squeeze(-1)
        q_values_bootstrap = torch.tensor(0.0).where(
            batch.dones, target_q_values.max(-1).values
        )

        loss = F.mse_loss(
            q_values,
            batch.rewards + discount * q_values_bootstrap,
        )
        return loss


class TargetPolicy(FullyObservablePolicy):
    def __init__(self, models: nn.ModuleDict):
        super().__init__()
        self.models = models

    def fo_sample_action(self, state):
        q_values = self.models.q_model(state.unsqueeze(0)).squeeze(0)
        return q_values.argmax().item()


class BehaviorPolicy(FullyObservablePolicy):
    def __init__(self, models: nn.ModuleDict, action_space: gym.Space):
        super().__init__()
        self.target_policy = TargetPolicy(models)
        self.action_space = action_space
        self.epsilon: float

    def fo_sample_action(self, state):
        return (
            self.action_space.sample()
            if random.random() < self.epsilon
            else self.target_policy.fo_sample_action(state)
        )
