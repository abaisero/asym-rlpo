#!/usr/bin/env python
import argparse

import torch
import torch.nn as nn
import wandb

from asym_rlpo.algorithms import make_algorithm
from asym_rlpo.data import EpisodeBuffer
from asym_rlpo.env import make_env
from asym_rlpo.evaluation import evaluate, evaluate_returns
from asym_rlpo.policies.random import RandomPolicy
from asym_rlpo.sampling import sample_episodes
from asym_rlpo.utils.scheduling import make_schedule
from asym_rlpo.utils.stats import standard_error


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--project', default='asym-rlpo')

    parser.add_argument(
        'algo',
        choices=['fob-dqn', 'foe-dqn', 'poe-dqn', 'poe-adqn'],
    )

    parser.add_argument(
        '--env',
        choices=[
            'PO-pos-CartPole-v0',
            'PO-vel-CartPole-v0',
            'PO-full-CartPole-v0',
            'PO-pos-CartPole-v1',
            'PO-vel-CartPole-v1',
            'PO-full-CartPole-v1',
        ],
        default='PO-pos-CartPole-v1',
    )

    # general
    parser.add_argument(
        '--max-simulation-timesteps', type=int, default=1_000_000
    )
    parser.add_argument('--max-steps-per-episode', type=int, default=1_000)

    # evaluation
    parser.add_argument('--evaluation-period', type=int, default=100)
    parser.add_argument('--evaluation-num-episodes', type=int, default=20)

    # episode buffer
    parser.add_argument('--episode-buffer-size', type=int, default=10_000)
    parser.add_argument('--episode-buffer-prepopulate', type=int, default=1_000)
    parser.add_argument(
        '--episode-buffer-episodes-per-epoch', type=int, default=1
    )

    # target
    parser.add_argument('--target-update-period', type=int, default=10)

    # training parameters
    # TODO make dynamic?
    parser.add_argument('--training-steps-per-epoch', type=int, default=4)
    parser.add_argument('--training-num-episodes', type=int, default=4)
    parser.add_argument('--training-batch-size', type=int, default=64)

    # epsilon schedule
    parser.add_argument('--epsilon-schedule', default='linear')
    parser.add_argument('--epsilon-value-from', type=float, default=1.0)
    parser.add_argument('--epsilon-value-to', type=float, default=0.05)
    parser.add_argument('--epsilon-nsteps', type=int, default=10_000)

    # optimization
    parser.add_argument('--optim-lr', type=float, default=0.001)
    parser.add_argument('--optim-eps', type=float, default=1e-8)
    parser.add_argument('--optim-max-norm', type=float, default=10.0)

    parser.add_argument('--render', action='store_true')

    return parser.parse_args()


def main():  # pylint: disable=too-many-locals,too-many-statements
    config = wandb.config
    # pylint: disable=no-member

    # counts and stats useful as x-axis
    xstats = {
        'epoch': 0,
        'simulation_episodes': 0,
        'simulation_timesteps': 0,
        'training_steps': 0,
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
    # env = make_env('PO-pos-CartPole-v1')
    # env = make_env('PO-vel-CartPole-v1')
    # env = make_env('PO-full-CartPole-v1')
    # env = make_env('gv_yaml/gv_nine_rooms.13x13.yaml')
    discount = 1.0

    # instantiate models and policies
    print('creating models and policies')
    algo = make_algorithm(config.algo, env)

    random_policy = RandomPolicy(env.action_space)
    behavior_policy = algo.behavior_policy(env.action_space)
    target_policy = algo.target_policy()

    # instantiate optimizer
    optimizer = torch.optim.Adam(
        algo.models.parameters(), lr=config.optim_lr, eps=config.optim_eps
    )

    # instantiate and prepopulate buffer
    print('creating episode_buffer')
    episode_buffer = EpisodeBuffer(maxlen=config.episode_buffer_size)
    print('prepopulating episode_buffer')
    episodes = sample_episodes(
        env,
        random_policy,
        num_episodes=config.episode_buffer_prepopulate,
        num_steps=config.max_steps_per_episode,
    )
    # TODO consider storing pytorch format directly.. at least we do conversion
    # only once!
    episode_buffer.append_episodes(episodes)
    xstats['simulation_episodes'] = len(episodes)
    xstats['simulation_timesteps'] = sum(len(episode) for episode in episodes)

    epsilon_schedule = make_schedule(
        config.epsilon_schedule,
        value_from=config.epsilon_value_from,
        value_to=config.epsilon_value_to,
        nsteps=config.epsilon_nsteps,
    )

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
                    num_steps=config.max_steps_per_episode,
                    render=True,
                )

            returns = evaluate(
                env,
                target_policy,
                discount=discount,
                num_episodes=config.evaluation_num_episodes,
                num_steps=config.max_steps_per_episode,
            )
            mean_return = returns.mean().item()
            sem_return = standard_error(returns)
            print(
                f'EVALUATE epoch {xstats["epoch"]}'
                f' return {mean_return:.3f} ({sem_return:.3f})'
            )
            ystats['performance/cum_target_mean_return'] += mean_return
            wandb.log(
                {
                    **xstats,
                    'performance/target_mean_return': mean_return,
                    'performance/cum_target_mean_return': ystats[
                        'performance/cum_target_mean_return'
                    ],
                }
            )

        # populate episode buffer
        behavior_policy.epsilon = epsilon_schedule(xstats['epoch'])
        episodes = sample_episodes(
            env,
            behavior_policy,
            num_episodes=config.episode_buffer_episodes_per_epoch,
            num_steps=config.max_steps_per_episode,
        )

        returns = evaluate_returns(episodes, discount=discount)
        mean_return = returns.mean().item()
        ystats['performance/cum_behavior_mean_return'] += mean_return
        wandb.log(
            {
                **xstats,
                'diagnostics/epsilon': behavior_policy.epsilon,
                'performance/behavior_mean_return': mean_return,
                'performance/cum_behavior_mean_return': ystats[
                    'performance/cum_behavior_mean_return'
                ],
            }
        )

        episode_buffer.append_episodes(episodes)
        xstats['simulation_episodes'] += len(episodes)
        xstats['simulation_timesteps'] += sum(
            len(episode) for episode in episodes
        )

        # train based on episode buffer
        if xstats['epoch'] % config.target_update_period == 0:
            algo.target_models.load_state_dict(algo.models.state_dict())

        algo.models.train()
        for _ in range(config.training_steps_per_epoch):
            optimizer.zero_grad()

            if algo.episodic_training:
                episodes = episode_buffer.sample_episodes(
                    num_samples=config.training_num_episodes,
                    replacement=True,
                )
                episodes = [episode.torch() for episode in episodes]

                loss = algo.episodic_loss(episodes, discount=discount)

            else:
                batch = episode_buffer.sample_batch(
                    batch_size=config.training_batch_size
                )
                batch = batch.torch()

                loss = algo.batched_loss(batch, discount=discount)

            loss.backward()
            gradient_norm = nn.utils.clip_grad_norm_(
                algo.models.parameters(), max_norm=config.optim_max_norm
            )

            wandb.log(
                {
                    **xstats,
                    'training/loss': loss,
                    'training/gradient_norm': gradient_norm,
                }
            )

            optimizer.step()

            xstats['training_steps'] += 1
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


if __name__ == '__main__':
    args = parse_args()
    with wandb.init(
        project=args.project,
        entity='abaisero',
        name=args.algo,
        config=args,
    ) as run:
        main()
