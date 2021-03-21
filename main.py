#!/usr/bin/env python
import argparse

import torch
import torch.nn as nn

import wandb
from asym_rlpo.algorithms import make_algorithm
from asym_rlpo.data import EpisodeBuffer
from asym_rlpo.env import make_env
from asym_rlpo.evaluation import evaluate
from asym_rlpo.policies.random import RandomPolicy
from asym_rlpo.sampling import sample_episodes
from asym_rlpo.utils.scheduling import make_schedule
from asym_rlpo.utils.stats import standard_error


def parse_args():
    parser = argparse.ArgumentParser()

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

    parser.add_argument('--render', action='store_true')

    return parser.parse_args()


def main(args):  # pylint: disable=too-many-locals,too-many-statements
    run = wandb.init(
        project='asym-rlpo-new-metric',
        entity='abaisero',
        name=f'{args.algo} - {args.env}',
    )
    wandb.config.update(args)  # pylint: disable=no-member

    # hyper-parameters
    num_epochs = 10_000
    max_steps_per_episode = 1_000  # NOTE over the practical limit

    episode_buffer_size = 10_000
    episode_buffer_prepopulate = 1_000
    episode_buffer_episodes_per_epoch = 1

    target_update_period = 10

    evaluation_period = 10
    evaluation_num_episodes = 10

    training_steps_per_epoch = 4  # TODO make dynamic?
    training_num_episodes = 4
    training_batch_size = 64

    epsilon_schedule_name = 'linear'
    epsilon_value_from = 1.0
    epsilon_value_to = 0.05
    epsilon_nsteps = 10_000

    optim_lr = 0.001
    # optim_lr = 0.0001
    optim_eps = 1e-08
    # optim_eps = 1e-04

    # insiantiate environment
    print('creating environment')
    env = make_env(args.env)
    # env = make_env('PO-pos-CartPole-v1')
    # env = make_env('PO-vel-CartPole-v1')
    # env = make_env('PO-full-CartPole-v1')
    # env = make_env('gv_yaml/gv_nine_rooms.13x13.yaml')
    discount = 1.0

    # instantiate models and policies
    print('creating models and policies')
    algo = make_algorithm(args.algo, env)

    random_policy = RandomPolicy(env.action_space)
    behavior_policy = algo.behavior_policy(env.action_space)
    target_policy = algo.target_policy()

    # instantiate optimizer
    optimizer = torch.optim.Adam(
        algo.models.parameters(), lr=optim_lr, eps=optim_eps
    )

    # instantiate and prepopulate buffer
    print('creating episode_buffer')
    episode_buffer = EpisodeBuffer(maxlen=episode_buffer_size)
    print('prepopulating episode_buffer')
    episodes = sample_episodes(
        env,
        random_policy,
        num_episodes=episode_buffer_prepopulate,
        num_steps=max_steps_per_episode,
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

    wandb.watch(algo.models)

    # main learning loop
    simulation_timesteps = 0
    training_timesteps = 0
    for epoch in range(num_epochs):
        algo.models.eval()

        # evaluate target policy
        if epoch % evaluation_period == 0:
            if args.render:
                sample_episodes(
                    env,
                    target_policy,
                    num_episodes=1,
                    num_steps=max_steps_per_episode,
                    render=True,
                )

            returns = evaluate(
                env,
                target_policy,
                discount=discount,
                num_episodes=evaluation_num_episodes,
                num_steps=max_steps_per_episode,
            )
            mean, sem = returns.mean(), standard_error(returns)
            print(f'EVALUATE epoch {epoch} return {mean:.3f} ({sem:.3f})')
            for evaluation_step, return_ in enumerate(returns):
                wandb.log(
                    {
                        'epoch': epoch,
                        'simulation_timesteps': simulation_timesteps,
                        'evaluation_step': evaluation_step,
                        'return': return_,
                    }
                )

        # populate episode buffer
        behavior_policy.epsilon = epsilon_schedule(epoch)
        episodes = sample_episodes(
            env,
            behavior_policy,
            num_episodes=episode_buffer_episodes_per_epoch,
            num_steps=max_steps_per_episode,
        )
        episode_buffer.append_episodes(episodes)
        simulation_timesteps += sum(len(episode) for episode in episodes)

        # train based on episode buffer
        if epoch % target_update_period == 0:
            algo.target_models.load_state_dict(algo.models.state_dict())

        algo.models.train()
        for _ in range(training_steps_per_epoch):
            optimizer.zero_grad()

            if algo.episodic_training:
                episodes = episode_buffer.sample_episodes(
                    num_samples=training_num_episodes,
                    replacement=True,
                )
                episodes = [episode.torch() for episode in episodes]

                loss = algo.episodic_loss(episodes, discount=discount)

            else:
                batch = episode_buffer.sample_batch(
                    batch_size=training_batch_size
                )
                batch = batch.torch()

                loss = algo.batched_loss(batch, discount=discount)

            loss.backward()

            wandb.log(
                {
                    'epoch': epoch,
                    'simulation_timesteps': simulation_timesteps,
                    'training_timesteps': training_timesteps,
                    'loss': loss,
                }
            )

            training_timesteps += (
                sum(len(episode) for episode in episodes)
                if algo.episodic_training
                else len(batch)
            )

            nn.utils.clip_grad_norm_(algo.models.parameters(), max_norm=10.0)
            optimizer.step()

    run.finish()


if __name__ == '__main__':
    main(parse_args())
