from __future__ import annotations

import random
import re

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from asym_rlpo.data import EpisodeBuffer
from asym_rlpo.modules import make_module
from asym_rlpo.policies.base import FullyObservablePolicy

from .base import EpisodicDQN


class FOE_DQN(EpisodicDQN):
    """FOE_DQN.

    Fully observable episodic DQN
    """

    def make_models(self, env: gym.Env) -> nn.ModuleDict:
        if re.fullmatch(r'CartPole-v\d+', env.spec.id):
            (input_dim,) = env.state_space.shape
            q_model = nn.Sequential(
                make_module('linear', 'leaky_relu', input_dim, 512),
                nn.LeakyReLU(),
                make_module('linear', 'leaky_relu', 512, 256),
                nn.LeakyReLU(),
                make_module('linear', 'linear', 256, env.action_space.n),
            )
            models = nn.ModuleDict(
                {
                    'q_model': q_model,
                }
            )

        else:
            raise NotImplementedError
            # observation_model = GV_ObservationRepresentation(env.observation_space)
            # q_model = nn.Sequential(
            #     nn.Linear(history_model.dim, 128),
            #     nn.ReLU(),
            #     nn.Linear(128, 128),
            #     nn.ReLU(),
            #     nn.Linear(128, env.action_space.n),
            # )
            # models = nn.ModuleDict(
            #     {
            #         'state_model': state_model,
            #         'q_model': q_model,
            #     }
            # )

        return models

    def target_policy(self) -> TargetPolicy:
        return TargetPolicy(self.models)

    def behavior_policy(
        self, action_space: gym.spaces.Discrete
    ) -> BehaviorPolicy:
        return BehaviorPolicy(self.models, action_space)

    def episodic_loss(
        self,
        episode_buffer: EpisodeBuffer,
        *,
        discount: float,
        num_episodes: int,
    ):

        episodes = episode_buffer.sample_episodes(
            num_samples=num_episodes,
            replacement=True,
        )

        losses = []
        for episode in episodes:
            episode = episode.torch()

            q_values = self.models.q_model(episode.states)
            with torch.no_grad():
                target_q_values = self.target_models.q_model(episode.states)

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
