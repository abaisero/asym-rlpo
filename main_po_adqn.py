#!/usr/bin/env python
import random
import re
from typing import Sequence

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F

import wandb
from asym_rlpo.data import Episode, EpisodeBuffer
from asym_rlpo.env import make_env
from asym_rlpo.evaluation import evaluate
from asym_rlpo.modules import make_module
from asym_rlpo.policies.base import PartiallyObservablePolicy
from asym_rlpo.policies.random import RandomPolicy
from asym_rlpo.representations.embedding import EmbeddingRepresentation
from asym_rlpo.representations.gv import GV_ObservationRepresentation
from asym_rlpo.representations.history import RNNHistoryRepresentation
from asym_rlpo.representations.identity import IdentityRepresentation
from asym_rlpo.representations.mlp import MLPRepresentation
from asym_rlpo.representations.onehot import OneHotRepresentation
from asym_rlpo.sampling import sample_episodes
from asym_rlpo.utils.scheduling import make_schedule
from asym_rlpo.utils.stats import standard_error


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
        q_values = self.models.qh_model(self.history_features)
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


def make_models(env: gym.Env) -> nn.ModuleDict:
    # TODO eventually change models depending on environment
    if re.fullmatch(r'CartPole-v\d+', env.spec.id):
        # action_model = EmbeddingRepresentation(env.action_space.n, 128)
        # observation_model = MLPRepresentation(env.observation_space, 128)

        action_model = OneHotRepresentation(env.action_space)
        state_model = IdentityRepresentation(env.state_space)
        observation_model = IdentityRepresentation(env.observation_space)

        history_model = RNNHistoryRepresentation(
            action_model,
            observation_model,
            hidden_size=128,
            nonlinearity='tanh',
        )
        qh_model = nn.Sequential(
            make_module('linear', 'leaky_relu', history_model.dim, 512),
            nn.LeakyReLU(),
            make_module('linear', 'leaky_relu', 512, 256),
            nn.LeakyReLU(),
            make_module('linear', 'linear', 256, env.action_space.n),
        )
        qhs_model = nn.Sequential(
            make_module(
                'linear', 'leaky_relu', history_model.dim + state_model.dim, 512
            ),
            nn.LeakyReLU(),
            make_module('linear', 'leaky_relu', 512, 256),
            nn.LeakyReLU(),
            make_module('linear', 'linear', 256, env.action_space.n),
        )
        models = nn.ModuleDict(
            {
                'action_model': action_model,
                'observation_model': observation_model,
                'state_model': state_model,
                'history_model': history_model,
                'qh_model': qh_model,
                'qhs_model': qhs_model,
            }
        )

    else:
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

    return models


def main():  # pylint: disable=too-many-locals
    run = wandb.init(project='asym-rlpo', entity='abaisero', group='adqn')

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
    env = make_env('PO-pos-CartPole-v1')
    # env = make_env('PO-vel-CartPole-v1')
    # env = make_env('PO-full-CartPole-v1')
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

    wandb.watch(models)

    # main learning loop
    timestep = 0
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
            for ret in returns:
                wandb.log(
                    {
                        'epoch': epoch,
                        'timestep': timestep,
                        'return': ret,
                    }
                )

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

        for optimization_step in range(num_optimizer_steps):
            training_episodes = episode_buffer.sample_episodes(
                num_samples=num_episodes_buffer_samples,
                replacement=True,
            )

            optimizer.zero_grad()
            loss = dqn_loss(models, target_models, training_episodes, discount)
            loss.backward()

            wandb.log(
                {
                    'epoch': epoch,
                    'timestep': timestep,
                    'optimization_step': optimization_step,
                    'loss': loss,
                }
            )

            # TODO right now the losses and the gradients are both increasing,
            # slowly but surely, over time

            # print(f'LOSS: {loss.item()}')
            # total_norm = 0.0
            # for p in models.parameters():
            #     param_norm = p.grad.data.norm(2)
            #     total_norm += param_norm.item() ** 2
            # total_norm = total_norm ** (1.0 / 2)
            # print(f'P-NORM: {total_norm}')

            # TODO clip gradients
            optimizer.step()

        timestep += sum(len(episode) for episode in episodes)

    run.finish()


def dqn_loss(
    models: nn.ModuleDict,
    target_models: nn.ModuleDict,
    episodes: Sequence[Episode],
    discount: float,
) -> torch.Tensor:
    def compute_q_values(models, actions, observations, states):
        action_features = models.action_model(actions)
        action_features = action_features.roll(1, 0)
        action_features[0, :] = 0.0
        observation_features = models.observation_model(observations)

        inputs = torch.cat([action_features, observation_features], dim=-1)
        history_features, _ = models.history_model(inputs.unsqueeze(0))
        history_features = history_features.squeeze(0)
        qh_values = models.qh_model(history_features)

        state_features = models.state_model(states)
        inputs = torch.cat([history_features, state_features], dim=-1)
        qhs_values = models.qhs_model(inputs)

        return qh_values, qhs_values

    def qhs_loss(
        episode,
        qh_values,
        qhs_values,
        target_qh_values,
        target_qhs_values,
    ) -> torch.Tensor:

        qhs_values = qhs_values.gather(
            1, episode.actions.unsqueeze(-1)
        ).squeeze(-1)
        qhs_values_bootstrap = torch.tensor(0.0).where(
            episode.dones,
            target_qhs_values.gather(
                1, target_qh_values.argmax(-1).unsqueeze(-1)
            )
            .squeeze(-1)
            .roll(-1, 0),
        )

        loss = F.mse_loss(
            qhs_values,
            episode.rewards + discount * qhs_values_bootstrap,
        )
        return loss

    def qh_loss(
        episode,
        qh_values,
        qhs_values,
        target_qh_values,
        target_qhs_values,
    ) -> torch.Tensor:
        # loss = F.mse_loss(qh_values, target_qhs_values)
        # return loss

        # qh_values = qh_values.gather(1, episode.actions.unsqueeze(-1)).squeeze(
        #     -1
        # )
        # target_qhs_values = target_qhs_values.gather(
        #     1, episode.actions.unsqueeze(-1)
        # ).squeeze(-1)
        # loss = F.mse_loss(qh_values, target_qhs_values)
        # return loss

        qh_values = qh_values.gather(1, episode.actions.unsqueeze(-1)).squeeze(
            -1
        )
        qhs_values_bootstrap = torch.tensor(0.0).where(
            episode.dones,
            target_qhs_values.gather(
                1, target_qh_values.argmax(-1).unsqueeze(-1)
            )
            .squeeze(-1)
            .roll(-1, 0),
        )

        loss = F.mse_loss(
            qh_values,
            episode.rewards + discount * qhs_values_bootstrap,
        )
        return loss

    losses = []
    for episode in episodes:
        episode = episode.torch()

        qh_values, qhs_values = compute_q_values(
            models, episode.actions, episode.observations, episode.states
        )
        with torch.no_grad():
            target_qh_values, target_qhs_values = compute_q_values(
                target_models,
                episode.actions,
                episode.observations,
                episode.states,
            )

        loss = (
            qhs_loss(
                episode,
                qh_values,
                qhs_values,
                target_qh_values,
                target_qhs_values,
            )
            + qh_loss(
                episode,
                qh_values,
                qhs_values,
                target_qh_values,
                target_qhs_values,
            )
        ) / 2

        losses.append(loss)

    return sum(losses, start=torch.tensor(0.0)) / len(losses)
    # return sum(losses, start=torch.tensor(0.0)) / sum(
    #     len(episode) for episode in episodes
    # )


if __name__ == '__main__':
    main()
