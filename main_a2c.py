#!/usr/bin/env python
import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import wandb
from gym_gridverse.rng import reset_gv_rng

from asym_rlpo.algorithms import make_a2c_algorithm
from asym_rlpo.envs import make_env
from asym_rlpo.evaluation import evaluate_returns
from asym_rlpo.q_estimators import q_estimator_factory
from asym_rlpo.sampling import sample_episodes
from asym_rlpo.utils.aggregate import average
from asym_rlpo.utils.checkpointing import save_data
from asym_rlpo.utils.config import Config, get_config
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
    parser.add_argument('algo', choices=['a2c', 'asym-a2c', 'asym-a2c-state'])

    parser.add_argument('--env-label', default=None)
    parser.add_argument('--algo-label', default=None)

    # truncated histories
    parser.add_argument('--truncated-histories', action='store_true')
    parser.add_argument('--truncated-histories-n', type=int, default=-1)

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
    parser.add_argument('--evaluation', action='store_true')
    parser.add_argument('--evaluation-period', type=int, default=10)
    parser.add_argument('--evaluation-num-episodes', type=int, default=1)
    parser.add_argument('--evaluation-epsilon', type=float, default=1.0)

    # discounts
    parser.add_argument('--evaluation-discount', type=float, default=1.0)
    parser.add_argument('--training-discount', type=float, default=0.99)

    # target
    parser.add_argument('--target-update-period', type=int, default=10_000)

    # q-estimator
    parser.add_argument(
        '--q-estimator',
        choices=['mc', 'td0', 'td-n', 'td-lambda'],
        default='td0',
    )
    parser.add_argument('--q-estimator-n', type=int, default=None)
    parser.add_argument('--q-estimator-lambda', type=float, default=None)

    # negentropy schedule
    parser.add_argument('--negentropy-schedule', default='linear')
    # linear
    parser.add_argument('--negentropy-value-from', type=float, default=1.0)
    parser.add_argument('--negentropy-value-to', type=float, default=0.01)
    parser.add_argument('--negentropy-nsteps', type=int, default=2_000_000)
    # exponential
    parser.add_argument('--negentropy-halflife', type=int, default=500_000)

    # optimization
    parser.add_argument('--optim-lr-actor', type=float, default=1e-4)
    parser.add_argument('--optim-eps-actor', type=float, default=1e-4)
    parser.add_argument('--optim-lr-critic', type=float, default=1e-4)
    parser.add_argument('--optim-eps-critic', type=float, default=1e-4)
    parser.add_argument('--optim-max-norm', type=float, default=float('inf'))

    # device
    parser.add_argument('--device', default='auto')

    # misc
    parser.add_argument('--render', action='store_true')

    # temporary / development
    parser.add_argument('--hs-features-dim', type=int, default=0)
    parser.add_argument('--normalize-hs-features', action='store_true')

    # checkpointing
    parser.add_argument('--save-model', action='store_true')
    parser.add_argument('--model-filename', default=None)

    args = parser.parse_args()
    args.env_label = args.env if args.env_label is None else args.env_label
    args.algo_label = args.algo if args.algo_label is None else args.algo_label
    return args


def run():  # pylint: disable=too-many-locals,too-many-statements
    config = get_config()

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

    # initialize return type
    q_estimator = q_estimator_factory(
        config.q_estimator,
        n=config.q_estimator_n,
        lambda_=config.q_estimator_lambda,
    )

    # instantiate models and policies
    print('creating models and policies')
    algo = make_a2c_algorithm(
        config.algo,
        env,
        truncated_histories=config.truncated_histories,
        truncated_histories_n=config.truncated_histories_n,
    )
    algo.to(device)

    behavior_policy = algo.behavior_policy()
    evaluation_policy = algo.evaluation_policy()
    evaluation_policy.epsilon = 0.1

    # instantiate optimizer
    optimizer_actor = torch.optim.Adam(
        algo.models.parameters(),
        lr=config.optim_lr_actor,
        eps=config.optim_eps_actor,
    )
    optimizer_critic = torch.optim.Adam(
        algo.models.parameters(),
        lr=config.optim_lr_critic,
        eps=config.optim_eps_critic,
    )

    # instantiate timer
    timer = Timer()

    negentropy_schedule = make_schedule(
        config.negentropy_schedule,
        value_from=config.negentropy_value_from,
        value_to=config.negentropy_value_to,
        nsteps=config.negentropy_nsteps,
        halflife=config.negentropy_halflife,
    )
    weight_negentropy = negentropy_schedule(xstats['simulation_timesteps'])

    # Tracks when we last updated the target network
    algo.target_models.load_state_dict(algo.models.state_dict())
    target_update_dispenser = Dispenser(config.target_update_period)

    # wandb log dispenser
    wandb_log_period = config.max_simulation_timesteps // config.num_wandb_logs
    wandb_log_dispenser = Dispenser(wandb_log_period)

    # main learning loop
    wandb.watch(algo.models)
    while xstats['simulation_timesteps'] < config.max_simulation_timesteps:
        algo.models.eval()

        # evaluate policy
        if (
            config.evaluation
            and xstats['epoch'] % config.evaluation_period == 0
        ):
            if config.render:
                sample_episodes(
                    env,
                    evaluation_policy,
                    num_episodes=1,
                    render=True,
                )

            episodes = sample_episodes(
                env,
                evaluation_policy,
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
                    'diagnostics/behavior_mean_episode_length': mean_length,
                    'performance/behavior_mean_return': returns.mean(),
                    'performance/avg_behavior_mean_return': avg_behavior_returns.value(),
                    'performance/avg100_behavior_mean_return': avg100_behavior_returns.value(),
                }
            )

        # storing torch data directly
        episodes = [episode.torch().to(device) for episode in episodes]
        xstats['simulation_episodes'] += len(episodes)
        xstats['simulation_timesteps'] += sum(
            len(episode) for episode in episodes
        )
        weight_negentropy = negentropy_schedule(xstats['simulation_timesteps'])

        # target model update
        if target_update_dispenser.dispense(xstats['simulation_timesteps']):
            # Update the target network
            algo.target_models.load_state_dict(algo.models.state_dict())

        algo.models.train()

        # critic
        optimizer_critic.zero_grad()
        losses = [
            algo.critic_loss(
                episode,
                discount=config.training_discount,
                q_estimator=q_estimator,
            )
            for episode in episodes
        ]
        critic_loss = average(losses)
        critic_loss.backward()
        critic_gradient_norm = nn.utils.clip_grad_norm_(
            algo.models.parameters(), max_norm=config.optim_max_norm
        )
        optimizer_critic.step()

        # actor
        optimizer_actor.zero_grad()
        losses = [
            algo.actor_losses(
                episode,
                discount=config.training_discount,
                q_estimator=q_estimator,
            )
            for episode in episodes
        ]

        actor_losses, negentropy_losses = zip(*losses)
        actor_loss = average(actor_losses)
        negentropy_loss = average(negentropy_losses)

        loss = actor_loss + weight_negentropy * negentropy_loss
        loss.backward()
        actor_gradient_norm = nn.utils.clip_grad_norm_(
            algo.models.parameters(), max_norm=config.optim_max_norm
        )
        optimizer_actor.step()

        if wandb_log:
            wandb.log(
                {
                    **xstats,
                    'hours': timer.hours,
                    'training/losses/actor': actor_loss,
                    'training/losses/critic': critic_loss,
                    'training/losses/negentropy': negentropy_loss,
                    'training/weights/negentropy': weight_negentropy,
                    'training/gradient_norms/actor': actor_gradient_norm,
                    'training/gradient_norms/critic': critic_gradient_norm,
                }
            )

        xstats['epoch'] += 1
        xstats['optimizer_steps'] += 1
        xstats['training_episodes'] += len(episodes)
        xstats['training_timesteps'] += sum(
            len(episode) for episode in episodes
        )

    if config.save_model and config.model_filename is not None:
        data = {
            'metadata': {
                'config': config._as_dict(),
            },
            'data': {
                'models.state_dict': algo.models.state_dict(),
            },
        }
        save_data(config.model_filename, data)


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

        # setup config
        config = get_config()
        config._update(dict(wandb.config))

        run()


if __name__ == '__main__':
    main()
