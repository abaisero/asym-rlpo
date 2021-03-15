#!/usr/bin/env python

import random
from typing import Sequence

import gym
import torch
import torch.nn as nn

from asym_rlpo.data import Episode, EpisodeBuffer
from asym_rlpo.env import make_env
from asym_rlpo.evaluation import evaluate
from asym_rlpo.policies import Policy, RandomPolicy
from asym_rlpo.representations.embedding import EmbeddingRepresentation
from asym_rlpo.representations.gv import GV_ObservationRepresentation
from asym_rlpo.representations.history import RNNHistoryRepresentation
from asym_rlpo.sampling import sample_episodes
from asym_rlpo.utils.scheduling import make_schedule
from asym_rlpo.utils.stats import standard_error


class DQN_BehaviorPolicy(Policy):
    def __init__(self, models: nn.ModuleDict, action_space: gym.Space):
        super().__init__()
        self.target_policy = DQN_TargetPolicy(models, action_space)
        self.action_space = action_space
        self.epsilon = 1.0

    def reset(self, observation):
        self.target_policy.reset(observation)

    def step(self, action, observation):
        self.target_policy.step(action, observation)

    def sample_action(self):
        return (
            self.target_policy.sample_action()
            if random.random() > self.epsilon
            else self.action_space.sample()
        )


class DQN_TargetPolicy(Policy):
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


def make_models(env: gym.Env) -> nn.ModuleDict:
    # TODO eventually change models depending on environment

    action_model = EmbeddingRepresentation(env.action_space.n, 64)
    observation_model = GV_ObservationRepresentation(env.observation_space)
    history_model = RNNHistoryRepresentation(
        action_model,
        observation_model,
        hidden_size=128,
    )
    q_model = nn.Linear(history_model.dim, env.action_space.n)

    return nn.ModuleDict(
        {
            'action_model': action_model,
            'observation_model': observation_model,
            'history_model': history_model,
            'q_model': q_model,
        }
    )


def main():  # pylint: disable=too-many-locals
    # hyper-parameters
    num_epochs = 1_000
    num_episodes_training = 2
    num_episodes_evaluation = 20
    num_steps = 100
    evaluation_period = 10

    num_episodes_buffer_prepopulate = 100
    num_episodes_buffer = 1_000
    num_episodes_buffer_sample = 5

    epsilon_schedule_name = 'linear'
    epsilon_value_from = 0.9
    epsilon_value_to = 0.1
    epsilon_nsteps = num_epochs

    lr = 0.003

    # insiantiate environment
    print('creating environment')
    env = make_env('gv_yaml/gv_nine_rooms.13x13.yaml')
    discount = 0.99

    # instantiate models and policies
    print('creating models')
    models = make_models(env)
    behavior_policy = DQN_BehaviorPolicy(models, env.action_space)
    target_policy = DQN_TargetPolicy(models, env.action_space)
    random_policy = RandomPolicy(env.action_space)

    # instantiate optimizer
    optimizer = torch.optim.Adam(models.parameters(), lr=lr)

    # instantiate and prepopulate buffer
    print('creating episode_buffer')
    episode_buffer = EpisodeBuffer(maxlen=num_episodes_buffer)
    print('prepopulating episode_buffer')
    episodes = sample_episodes(
        env,
        random_policy,
        num_episodes=num_episodes_buffer_prepopulate,
        num_steps=num_steps,
    )
    episode_buffer.append_episodes(episodes)

    epsilon_schedule = make_schedule(
        epsilon_schedule_name,
        value_from=epsilon_value_from,
        value_to=epsilon_value_to,
        nsteps=epsilon_nsteps,
    )

    # main learning loop
    for epoch in range(num_epochs):

        # evaluate target policy
        if epoch % evaluation_period == 0:
            returns = evaluate(
                env,
                target_policy,
                discount=discount,
                num_episodes=num_episodes_evaluation,
                num_steps=num_steps,
            )
            mean, sem = returns.mean(), standard_error(returns)
            print(f'EVALUATE epoch {epoch} return {mean:.3f} ({sem:.3f})')

        # populate episode buffer
        behavior_policy.epsilon = epsilon_schedule(epoch)
        episodes = sample_episodes(
            env,
            behavior_policy,
            num_episodes=num_episodes_training,
            num_steps=num_steps,
        )
        episode_buffer.append_episodes(episodes)
        print(
            f'episode_buffer stats - '
            f'#interactions: {episode_buffer.num_interactions()}'
            f'\t#episodes: {episode_buffer.num_episodes()}'
        )

        # train based on episode buffer
        episodes = episode_buffer.sample_episodes(
            num_samples=num_episodes_buffer_sample,
            replacement=True,
        )

        optimizer.zero_grad()
        loss = dqn_loss(models, episodes)
        loss.backward()
        optimizer.step()


def dqn_loss(
    models: nn.ModuleDict,
    episodes: Sequence[Episode],
) -> torch.Tensor:

    # TODO for now just assume a single episode
    episode = episodes[0]

    loss = torch.tensor([0.0], requires_grad=True)

    return loss


if __name__ == '__main__':
    main()
