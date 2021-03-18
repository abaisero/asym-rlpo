#!/usr/bin/env python
import random
import re
from typing import Sequence

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F

from asym_rlpo.data import Batch, Episode, EpisodeBuffer
from asym_rlpo.env import make_env
from asym_rlpo.evaluation import evaluate
from asym_rlpo.modules import make_module
from asym_rlpo.policies.base import FullyObservablePolicy
from asym_rlpo.policies.random import RandomPolicy
from asym_rlpo.sampling import sample_episodes
from asym_rlpo.utils.scheduling import make_schedule
from asym_rlpo.utils.stats import standard_error


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


def make_models(env: gym.Env) -> nn.ModuleDict:
    # TODO eventually change models depending on environment
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


def main():  # pylint: disable=too-many-locals
    # hyper-parameters
    num_epochs = 1_000_000
    num_episodes_training = 1
    num_episodes_evaluation = 10
    num_steps = 1000
    evaluation_period = 10
    evaluation_render = True

    num_episodes_buffer_prepopulate = 1_000
    num_episodes_buffer_size = 10_000
    num_episodes_buffer_samples = 4  # NOTE higher is good
    target_update_period = 10  # unclear;  lower maybe?
    num_optimizer_steps = 4  # TODO make dynamic

    episodic_training = True  #  selected between episodic and batch training

    episodes_buffer_batch_size = 64

    # TODO still can't get consistently good performance;   we might need to do
    # hyper-param optimization right away..

    epsilon_schedule_name = 'linear'
    epsilon_value_from = 1.0
    epsilon_value_to = 0.05
    epsilon_nsteps = num_epochs

    optim_lr = 0.001
    # optim_lr = 0.0001

    optim_eps = 1e-08
    # optim_eps = 1e-04

    # insiantiate environment
    print('creating environment')
    env = make_env('PO-CartPole-v1')
    # env = make_env('gv_yaml/gv_nine_rooms.13x13.yaml')
    discount = 1.0

    # instantiate models and policies
    print('creating models')
    models = make_models(env)
    target_models = make_models(env)
    print('creating policies')
    random_policy = RandomPolicy(env.action_space)
    behavior_policy = BehaviorPolicy(models, env.action_space)
    target_policy = TargetPolicy(models)

    # instantiate optimizer
    optimizer = torch.optim.Adam(
        models.parameters(), lr=optim_lr, eps=optim_eps
    )

    # instantiate and prepopulate buffer
    print('creating episode_buffer')
    episode_buffer = EpisodeBuffer(maxlen=num_episodes_buffer_size)
    print('prepopulating episode_buffer')
    episodes = sample_episodes(
        env,
        random_policy,
        num_episodes=num_episodes_buffer_prepopulate,
        num_steps=num_steps,
    )
    # TODO consider storing pytorch format directly.. at least we do conversion
    # only once!
    episode_buffer.append_episodes(episodes)

    epsilon_schedule = make_schedule(
        epsilon_schedule_name,
        value_from=epsilon_value_from,
        value_to=epsilon_value_to,
        nsteps=epsilon_nsteps,
    )

    # main learning loop
    for epoch in range(num_epochs):
        models.eval()

        # evaluate target policy
        if epoch % evaluation_period == 0:
            if evaluation_render:
                # rendering
                sample_episodes(
                    env,
                    target_policy,
                    num_episodes=1,
                    num_steps=num_steps,
                    render=True,
                )

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
        # print(f'epsilon: {behavior_policy.epsilon}')
        episodes = sample_episodes(
            env,
            behavior_policy,
            num_episodes=num_episodes_training,
            num_steps=num_steps,
        )
        episode_buffer.append_episodes(episodes)
        # print(
        #     f'episode_buffer stats - '
        #     f'#interactions: {episode_buffer.num_interactions()}'
        #     f'\t#episodes: {episode_buffer.num_episodes()}'
        # )

        # train based on episode buffer
        if epoch % target_update_period == 0:
            target_models.load_state_dict(models.state_dict())

        models.train()
        target_models.train()

        for _ in range(num_optimizer_steps):
            optimizer.zero_grad()

            if episodic_training:
                # sample and train on entire episodes
                episodes = episode_buffer.sample_episodes(
                    num_samples=num_episodes_buffer_samples,
                    replacement=True,
                )
                loss = dqn_loss_episodic(
                    models, target_models, episodes, discount
                )

            else:
                # sample and train on a batch of independent transitions
                batch = episode_buffer.sample_batch(
                    batch_size=episodes_buffer_batch_size,
                )
                loss = dqn_loss_batch(models, target_models, batch, discount)

            loss.backward()
            nn.utils.clip_grad_norm_(models.parameters(), max_norm=10.0)

            # # print(f'LOSS: {loss.item()}')
            # total_norm = 0.0
            # for p in models.parameters():
            #     param_norm = p.grad.data.norm(2)
            #     total_norm += param_norm.item() ** 2
            # total_norm = total_norm ** (1.0 / 2)
            # print(f'P-NORM: {total_norm}')
            # inf_norm = max(p.grad.data.abs().max() for p in models.parameters())
            # print(f'inf-NORM: {inf_norm}')

            # TODO clip gradients
            optimizer.step()


def dqn_loss_batch(
    models: nn.ModuleDict,
    target_models: nn.ModuleDict,
    batch: Batch,
    discount: float,
) -> torch.Tensor:

    batch = batch.torch()

    q_values = models.q_model(batch.states)
    with torch.no_grad():
        target_q_values = target_models.q_model(batch.next_states)

    # print(q_values.argmax(-1))
    # print(
    #     torch.stack(
    #         [
    #             q_values.mean(-1),
    #             # q_values.max(-1).values - q_values.min(-1).values,
    #             q_values[:, 0] - q_values.mean(-1),
    #             q_values[:, 1] - q_values.mean(-1),
    #         ],
    #         -1,
    #     )
    # )

    q_values = q_values.gather(1, batch.actions.unsqueeze(-1)).squeeze(-1)
    q_values_bootstrap = torch.tensor(0.0).where(
        batch.dones, target_q_values.max(-1).values
    )

    # print(batch.rewards + discount * q_values_bootstrap)

    # print(q_values)
    loss = F.mse_loss(
        q_values,
        batch.rewards + discount * q_values_bootstrap,
    )
    return loss


def dqn_loss_episodic(
    models: nn.ModuleDict,
    target_models: nn.ModuleDict,
    episodes: Sequence[Episode],
    discount: float,
) -> torch.Tensor:

    losses = []
    for episode in episodes:
        episode = episode.torch()

        q_values = models.q_model(episode.states)
        with torch.no_grad():
            target_q_values = target_models.q_model(episode.states)

        # print(q_values.argmax(-1))
        # print(
        #     torch.stack(
        #         [
        #             q_values.mean(-1),
        #             # q_values.max(-1).values - q_values.min(-1).values,
        #             q_values[:, 0] - q_values.mean(-1),
        #             q_values[:, 1] - q_values.mean(-1),
        #         ],
        #         -1,
        #     )
        # )
        q_values = q_values.gather(1, episode.actions.unsqueeze(-1)).squeeze(-1)
        q_values_bootstrap = torch.tensor(0.0).where(
            episode.dones, target_q_values.max(-1).values.roll(-1, 0)
        )

        # print(episode.rewards + discount * q_values_bootstrap)

        # print(q_values)
        loss = F.mse_loss(
            q_values,
            episode.rewards + discount * q_values_bootstrap,
        )
        losses.append(loss)

    return sum(losses, start=torch.tensor(0.0)) / len(losses)
    # return sum(losses, start=torch.tensor(0.0)) / sum(
    #     len(episode) for episode in episodes
    # )


if __name__ == '__main__':
    main()
