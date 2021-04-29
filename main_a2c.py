#!/usr/bin/env python
import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import wandb
from gym_gridverse.rng import reset_gv_rng

from asym_rlpo.algorithms import make_algorithm
from asym_rlpo.env import make_env
from asym_rlpo.evaluation import evaluate_returns
from asym_rlpo.sampling import sample_episodes
from asym_rlpo.targets import target_function_factory
from asym_rlpo.utils.aggregate import average
from asym_rlpo.utils.device import get_device
from asym_rlpo.utils.scheduling import make_schedule
from asym_rlpo.utils.timer import Timer


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--wandb-project', default='asym-rlpo')
    parser.add_argument('--wandb-entity', default='abaisero')
    parser.add_argument('--wandb-group', default=None)

    # algorithm and environment
    parser.add_argument('env')
    parser.add_argument('algo', choices=['sym-a2c', 'asym-a2c'])

    # reproducibility
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--deterministic', action='store_true')

    # general
    parser.add_argument(
        '--max-simulation-timesteps', type=int, default=2_000_000
    )
    parser.add_argument('--max-steps-per-episode', type=int, default=1_000)
    parser.add_argument('--simulation-num-episodes', type=int, default=1)

    # evaluation
    parser.add_argument('--evaluation-period', type=int, default=10)
    parser.add_argument('--evaluation-num-episodes', type=int, default=1)

    # targets
    parser.add_argument(
        '--target',
        choices=['mc', 'td0', 'td-n', 'td-lambda'],
        default='td0',
    )
    parser.add_argument('--target-n', type=int, default=None)
    parser.add_argument('--target-lambda', type=float, default=None)

    # negentropy schedule
    parser.add_argument('--negentropy-schedule', default='linear')
    parser.add_argument('--negentropy-value-from', type=float, default=1.0)
    parser.add_argument('--negentropy-value-to', type=float, default=0.01)
    parser.add_argument(
        '--negentropy-value-to-factor', type=float, default=None
    )
    parser.add_argument('--negentropy-nsteps', type=int, default=500_000)

    # optimization
    parser.add_argument('--optim-lr', type=float, default=0.001)
    parser.add_argument('--optim-eps', type=float, default=1e-8)
    parser.add_argument('--optim-max-norm', type=float, default=float('inf'))

    # device
    parser.add_argument('--device', default='auto')

    parser.add_argument('--render', action='store_true')

    return parser.parse_args()


def main():  # pylint: disable=too-many-locals,too-many-statements
    config = wandb.config
    # pylint: disable=no-member

    device = get_device(config.device)

    # counts and stats useful as x-axis
    xstats = {
        'epoch': 0,
        'simulation_episodes': 0,
        'simulation_timesteps': 0,
        'evaluation_steps': 0,
        'optimizer_steps': 0,
        'training_episodes': 0,
        'training_timesteps': 0,
    }

    # counts and stats useful as y-axis
    ystats = {
        'performance/cum_target_mean_return': 0.0,
        'performance/cum_behavior_mean_return': 0.0,
    }

    # insiantiate environment
    print('creating environment')
    env = make_env(config.env)
    discount = 1.0

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
    target_f = target_function_factory(
        config.target,
        n=config.target_n,
        lambda_=config.target_lambda,
    )

    # instantiate models and policies
    print('creating models and policies')
    algo = make_algorithm(config.algo, env)
    algo.to(device)
    policy = algo.policy()

    # instantiate optimizer
    optimizer = torch.optim.Adam(
        algo.models.parameters(), lr=config.optim_lr, eps=config.optim_eps
    )

    # instantiate timer
    timer = Timer()

    weight_actor = 1.0
    weight_critic = 1.0
    negentropy_schedule = make_schedule(
        config.negentropy_schedule,
        value_from=config.negentropy_value_from,
        value_to=config.negentropy_value_to
        if config.negentropy_value_to_factor is None
        else config.negentropy_value_from * config.negentropy_value_to_factor,
        nsteps=config.negentropy_nsteps,
    )
    weight_negentropy = negentropy_schedule(xstats['simulation_timesteps'])

    # main learning loop
    wandb.watch(algo.models)
    while xstats['simulation_timesteps'] < config.max_simulation_timesteps:
        algo.models.eval()

        # evaluate policy
        if xstats['epoch'] % config.evaluation_period == 0:
            if config.render:
                sample_episodes(
                    env,
                    policy,
                    num_episodes=1,
                    num_steps=config.max_steps_per_episode,
                    render=True,
                )

            episodes = sample_episodes(
                env,
                policy,
                num_episodes=config.evaluation_num_episodes,
                num_steps=config.max_steps_per_episode,
            )
            mean_length = sum(map(len, episodes)) / len(episodes)
            mean_return = evaluate_returns(episodes, discount=discount).mean()
            print(
                f'EVALUATE epoch {xstats["epoch"]}'
                f' simulation_timestep {xstats["simulation_timesteps"]}'
                f' return {mean_return:.3f}'
            )
            ystats['performance/cum_target_mean_return'] += mean_return
            wandb.log(
                {
                    **xstats,
                    'hours': timer.hours,
                    'diagnostics/target_mean_episode_length': mean_length,
                    'performance/target_mean_return': mean_return,
                    'performance/avg_target_mean_return': ystats[
                        'performance/cum_target_mean_return'
                    ]
                    / (xstats['evaluation_steps'] + 1),
                }
            )
            xstats['evaluation_steps'] += 1

        episodes = sample_episodes(
            env,
            policy,
            num_episodes=config.simulation_num_episodes,
            num_steps=config.max_steps_per_episode,
        )

        mean_length = sum(map(len, episodes)) / len(episodes)
        mean_return = evaluate_returns(episodes, discount=discount).mean()
        ystats['performance/cum_behavior_mean_return'] += mean_return
        wandb.log(
            {
                **xstats,
                'hours': timer.hours,
                'diagnostics/behavior_mean_episode_length': mean_length,
                'performance/behavior_mean_return': mean_return,
                'performance/avg_behavior_mean_return': ystats[
                    'performance/cum_behavior_mean_return'
                ]
                / (xstats['epoch'] + 1),
            }
        )

        # storing torch data directly
        episodes = [episode.torch().to(device) for episode in episodes]
        xstats['simulation_episodes'] += len(episodes)
        xstats['simulation_timesteps'] += sum(
            len(episode) for episode in episodes
        )
        weight_negentropy = negentropy_schedule(xstats['simulation_timesteps'])

        algo.models.train()
        optimizer.zero_grad()
        losses = [
            algo.losses(episode, discount=discount, target_f=target_f)
            for episode in episodes
        ]
        loss_actor = average([l['actor'] for l in losses])
        loss_critic = average([l['critic'] for l in losses])
        loss_negentropy = average([l['negentropy'] for l in losses])

        loss = (
            weight_actor * loss_actor
            + weight_critic * loss_critic
            + weight_negentropy * loss_negentropy
        )

        loss.backward()
        gradient_norm = nn.utils.clip_grad_norm_(
            algo.models.parameters(), max_norm=config.optim_max_norm
        )

        wandb.log(
            {
                **xstats,
                'hours': timer.hours,
                'training/loss': loss,
                'training/weights/actor': weight_actor,
                'training/weights/critic': weight_critic,
                'training/weights/negentropy': weight_negentropy,
                'training/losses/actor': loss_actor,
                'training/losses/critic': loss_critic,
                'training/losses/negentropy': loss_negentropy,
                'training/gradient_norm': gradient_norm,
            }
        )

        optimizer.step()

        xstats['epoch'] += 1
        xstats['optimizer_steps'] += 1
        xstats['training_episodes'] += len(episodes)
        xstats['training_timesteps'] += sum(
            len(episode) for episode in episodes
        )


if __name__ == '__main__':
    args = parse_args()
    with wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=args.wandb_group,
        config=args,
    ) as run:
        main()
