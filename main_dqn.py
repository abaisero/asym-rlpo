#!/usr/bin/env python
import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import wandb
from gym_gridverse.rng import reset_gv_rng

from asym_rlpo.algorithms import make_dqn_algorithm
from asym_rlpo.data import EpisodeBuffer
from asym_rlpo.env import make_env
from asym_rlpo.evaluation import evaluate_returns
from asym_rlpo.policies.random import RandomPolicy
from asym_rlpo.sampling import sample_episodes
from asym_rlpo.utils.device import get_device
from asym_rlpo.utils.running_average import (
    InfiniteRunningAverage,
    WindowRunningAverage,
)
from asym_rlpo.utils.scheduling import make_schedule
from asym_rlpo.utils.timer import Dispenser, Timer


def parse_args():
    parser = argparse.ArgumentParser()

    # wandb arguments
    parser.add_argument('--wandb-project', default='asym-rlpo')
    parser.add_argument('--wandb-entity', default='abaisero')
    parser.add_argument('--wandb-group', default=None)
    parser.add_argument('--wandb-tag', action='append', dest='wandb_tags')
    parser.add_argument('--wandb-offline', action='store_true')

    # wandb related
    parser.add_argument('--num-wandb-logs', type=int, default=200)

    # algorithm and environment
    parser.add_argument('env')
    parser.add_argument(
        'algo',
        choices=[
            'fob-dqn',
            'foe-dqn',
            'dqn',
            'adqn',
            'adqn-bootstrap',
            'adqn-state',
            'adqn-state-bootstrap',
        ],
    )

    parser.add_argument('--env-label', default=None)
    parser.add_argument('--algo-label', default=None)

    # truncated histories
    parser.add_argument('--truncated-histories', action='store_true')
    parser.add_argument('--truncated-histories-n', type=int, default=3)

    # reproducibility
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--deterministic', action='store_true')

    # general
    parser.add_argument(
        '--max-simulation-timesteps', type=int, default=2_000_000
    )
    parser.add_argument('--max-episode-timesteps', type=int, default=1_000)
    parser.add_argument('--simulation-num-episodes', type=int, default=1)

    # evaluation
    parser.add_argument('--evaluation-period', type=int, default=10)
    parser.add_argument('--evaluation-num-episodes', type=int, default=1)

    # discounts
    parser.add_argument('--evaluation-discount', type=float, default=1.0)
    parser.add_argument('--training-discount', type=float, default=0.99)

    # episode buffer
    parser.add_argument(
        '--episode-buffer-max-timesteps', type=int, default=1_000_000
    )
    parser.add_argument(
        '--episode-buffer-prepopulate-timesteps', type=int, default=50_000
    )

    # target
    parser.add_argument('--target-update-period', type=int, default=10_000)

    # training parameters
    parser.add_argument(
        '--training-timesteps-per-simulation-timestep', type=int, default=8
    )
    parser.add_argument('--training-num-episodes', type=int, default=1)
    parser.add_argument('--training-batch-size', type=int, default=32)

    # epsilon schedule
    parser.add_argument('--epsilon-schedule', default='linear')
    parser.add_argument('--epsilon-value-from', type=float, default=1.0)
    parser.add_argument('--epsilon-value-to', type=float, default=0.1)
    parser.add_argument('--epsilon-nsteps', type=int, default=1_000_000)

    # optimization
    parser.add_argument('--optim-lr', type=float, default=1e-4)
    parser.add_argument('--optim-eps', type=float, default=1e-4)
    parser.add_argument('--optim-max-norm', type=float, default=float('inf'))

    # device
    parser.add_argument('--device', default='auto')

    parser.add_argument('--render', action='store_true')

    args = parser.parse_args()
    args.env_label = args.env if args.env_label is None else args.env_label
    args.algo_label = args.algo if args.algo_label is None else args.algo_label
    return args


def run():  # pylint: disable=too-many-locals,too-many-statements
    config = wandb.config
    # pylint: disable=no-member

    print(f'run {config.env_label} {config.algo_label}')

    device = get_device(config.device)

    # counts and stats useful as x-axis
    xstats = {
        'epoch': 0,
        'simulation_episodes': 0,
        'simulation_timesteps': 0,
        'optimizer_steps': 0,
        'training_episodes': 0,
        'training_timesteps': 0,
    }

    # counts and stats useful as y-axis
    avg_target_returns = InfiniteRunningAverage()
    avg_behavior_returns = InfiniteRunningAverage()
    avg100_behavior_returns = WindowRunningAverage(100)

    # insiantiate environment
    print('creating environment')
    env = make_env(
        config.env,
        max_episode_timesteps=config.max_episode_timesteps,
    )

    # reproducibility
    if config.seed is not None:
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        reset_gv_rng(config.seed)
        env.seed(config.seed)
        env.state_space.seed(config.seed)
        env.action_space.seed(config.seed)
        env.observation_space.seed(config.seed)

    if config.deterministic:
        torch.use_deterministic_algorithms(True)

    # instantiate models and policies
    print('creating models and policies')
    algo = make_dqn_algorithm(
        config.algo,
        env,
        truncated_histories=config.truncated_histories,
        truncated_histories_n=config.truncated_histories_n,
    )
    algo.to(device)

    random_policy = RandomPolicy(env.action_space)
    behavior_policy = algo.behavior_policy(env.action_space)
    target_policy = algo.target_policy()

    # instantiate optimizer
    optimizer = torch.optim.Adam(
        algo.models.parameters(), lr=config.optim_lr, eps=config.optim_eps
    )

    # instantiate timer
    timer = Timer()

    # instantiate and prepopulate buffer
    print('creating episode_buffer')
    episode_buffer = EpisodeBuffer(config.episode_buffer_max_timesteps)
    print('prepopulating episode_buffer')
    while (
        episode_buffer.num_interactions()
        < config.episode_buffer_prepopulate_timesteps
    ):
        (episode,) = sample_episodes(
            env,
            random_policy,
            num_episodes=1,
        )
        # storing torch data directly
        episode = episode.torch()
        episode_buffer.append_episode(episode)
    xstats['simulation_episodes'] = episode_buffer.num_episodes()
    xstats['simulation_timesteps'] = episode_buffer.num_interactions()

    epsilon_schedule = make_schedule(
        config.epsilon_schedule,
        value_from=config.epsilon_value_from,
        value_to=config.epsilon_value_to,
        nsteps=config.epsilon_nsteps,
    )

    # Tracks when we last updated the target network
    algo.target_models.load_state_dict(algo.models.state_dict())
    target_update_dispenser = Dispenser(config.target_update_period)

    # wandb log dispenser
    wandb_log_period = config.max_simulation_timesteps // config.nuim_wandb_logs
    wandb_log_dispenser = Dispenser(wandb_log_period)

    # main learning loop
    wandb.watch(algo.models)
    while xstats['simulation_timesteps'] < config.max_simulation_timesteps:
        algo.models.eval()

        # evaluate target policy
        if xstats['epoch'] % config.evaluation_period == 0:
            if config.render:
                sample_episodes(
                    env,
                    target_policy,
                    num_episodes=1,
                    render=True,
                )

            episodes = sample_episodes(
                env,
                target_policy,
                num_episodes=config.evaluation_num_episodes,
            )
            mean_length = sum(map(len, episodes)) / len(episodes)
            returns = evaluate_returns(
                episodes, discount=config.evaluation_discount
            )
            avg_target_returns.extend(returns.tolist())
            print(
                f'EVALUATE epoch {xstats["epoch"]}'
                f' simulation_timestep {xstats["simulation_timesteps"]}'
                f' return {returns.mean():.3f}'
            )
            wandb.log(
                {
                    **xstats,
                    'hours': timer.hours,
                    'diagnostics/target_mean_episode_length': mean_length,
                    'performance/target_mean_return': returns.mean(),
                    'performance/avg_target_mean_return': avg_target_returns.value(),
                }
            )

        # populate episode buffer
        behavior_policy.epsilon = epsilon_schedule(
            xstats['simulation_timesteps']
            - config.episode_buffer_prepopulate_timesteps
        )
        episodes = sample_episodes(
            env,
            behavior_policy,
            num_episodes=config.simulation_num_episodes,
        )

        mean_length = sum(map(len, episodes)) / len(episodes)
        returns = evaluate_returns(
            episodes, discount=config.evaluation_discount
        )
        avg_behavior_returns.extend(returns.tolist())
        avg100_behavior_returns.extend(returns.tolist())

        wandb_log = wandb_log_dispenser.dispense(xstats['simulation_timesteps'])

        if wandb_log:
            wandb.log(
                {
                    **xstats,
                    'hours': timer.hours,
                    'diagnostics/epsilon': behavior_policy.epsilon,
                    'diagnostics/behavior_mean_episode_length': mean_length,
                    'performance/behavior_mean_return': returns.mean(),
                    'performance/avg_behavior_mean_return': avg_behavior_returns.value(),
                    'performance/avg100_behavior_mean_return': avg100_behavior_returns.value(),
                }
            )

        # storing torch data directly
        episodes = [episode.torch() for episode in episodes]
        episode_buffer.append_episodes(episodes)
        xstats['simulation_episodes'] += len(episodes)
        xstats['simulation_timesteps'] += sum(
            len(episode) for episode in episodes
        )

        # target model update
        if target_update_dispenser.dispense(xstats['simulation_timesteps']):
            # Update the target network
            algo.target_models.load_state_dict(algo.models.state_dict())

        # train based on episode buffer
        algo.models.train()
        while (
            xstats['training_timesteps']
            < (
                xstats['simulation_timesteps']
                - config.episode_buffer_prepopulate_timesteps
            )
            * config.training_timesteps_per_simulation_timestep
        ):
            optimizer.zero_grad()

            if algo.episodic_training:
                episodes = episode_buffer.sample_episodes(
                    num_samples=config.training_num_episodes,
                    replacement=True,
                )
                episodes = [episode.to(device) for episode in episodes]
                loss = algo.episodic_loss(
                    episodes, discount=config.training_discount
                )

            else:
                batch = episode_buffer.sample_batch(
                    batch_size=config.training_batch_size
                )
                batch = batch.to(device)
                loss = algo.batched_loss(
                    batch, discount=config.training_discount
                )

            loss.backward()
            gradient_norm = nn.utils.clip_grad_norm_(
                algo.models.parameters(), max_norm=config.optim_max_norm
            )

            if wandb_log:
                wandb.log(
                    {
                        **xstats,
                        'hours': timer.hours,
                        'training/loss': loss,
                        'training/gradient_norm': gradient_norm,
                    }
                )

            optimizer.step()

            xstats['optimizer_steps'] += 1
            xstats['training_episodes'] = (
                xstats['training_episodes'] + len(episodes)
                if algo.episodic_training
                else None
            )
            xstats['training_timesteps'] += (
                sum(len(episode) for episode in episodes)
                if algo.episodic_training
                else len(batch)
            )

        xstats['epoch'] += 1


def main():
    args = parse_args()
    with wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=args.wandb_group,
        tags=args.wandb_tags,
        mode='offline' if args.wandb_offline else None,
        config=args,
    ) as wandb_run:  # pylint: disable=unused-variable
        run()


if __name__ == '__main__':
    main()
