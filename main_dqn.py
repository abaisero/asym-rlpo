#!/usr/bin/env python
import argparse
import logging
import logging.config
import os
import random
import signal
from dataclasses import asdict, dataclass
from typing import Dict, NamedTuple

import gym
import numpy as np
import torch
import torch.nn as nn
import wandb
from gym_gridverse.rng import reset_gv_rng

from asym_rlpo.algorithms import make_dqn_algorithm
from asym_rlpo.algorithms.dqn.base import PO_DQN_ABC
from asym_rlpo.data import EpisodeBuffer
from asym_rlpo.envs import make_env
from asym_rlpo.evaluation import evaluate_returns
from asym_rlpo.policies.base import Policy
from asym_rlpo.policies.random import RandomPolicy
from asym_rlpo.sampling import sample_episodes
from asym_rlpo.utils.checkpointing import Serializable, load_data, save_data
from asym_rlpo.utils.config import get_config
from asym_rlpo.utils.device import get_device
from asym_rlpo.utils.running_average import (
    InfiniteRunningAverage,
    RunningAverage,
    WindowRunningAverage,
)
from asym_rlpo.utils.scheduling import make_schedule
from asym_rlpo.utils.timer import Dispenser, Timer
from asym_rlpo.utils.wandb_logger import WandbLogger

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()

    # wandb arguments
    parser.add_argument('--wandb-entity', default='abaisero')
    parser.add_argument('--wandb-project', default=None)
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
            'adqn-short',
            'adqn-state',
            'adqn-state-bootstrap',
        ],
    )

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

    # temporary / development
    parser.add_argument('--hs-features-dim', type=int, default=0)
    parser.add_argument('--normalize-hs-features', action='store_true')

    # gv models
    parser.add_argument(
        '--gv-observation-model-type',
        choices=['cnn', 'fc'],
        default='fc',
    )
    parser.add_argument(
        '--gv-state-model-type',
        choices=['cnn', 'fc'],
        default='fc',
    )

    # checkpoint
    parser.add_argument('--checkpoint', default=None)

    parser.add_argument('--save-model', action='store_true')
    parser.add_argument('--model-filename', default=None)

    parser.add_argument('--save-modelseq', action='store_true')
    parser.add_argument('--modelseq-filename', default=None)

    args = parser.parse_args()
    args.env_label = args.env if args.env_label is None else args.env_label
    args.algo_label = args.algo if args.algo_label is None else args.algo_label
    args.wandb_mode = 'offline' if args.wandb_offline else None
    return args


@dataclass
class XStats(Serializable):
    epoch: int = 0
    simulation_episodes: int = 0
    simulation_timesteps: int = 0
    optimizer_steps: int = 0
    training_episodes: int = 0
    training_timesteps: int = 0

    def asdict(self):
        return asdict(self)

    def state_dict(self):
        return self.asdict()

    def load_state_dict(self, data):
        self.epoch = data['epoch']
        self.simulation_episodes = data['simulation_episodes']
        self.simulation_timesteps = data['simulation_timesteps']
        self.optimizer_steps = data['optimizer_steps']
        self.training_episodes = data['training_episodes']
        self.training_timesteps = data['training_timesteps']


# NOTE:  namedtuple does not allow multiple inheritance.. luckily Serializable
# is only an interface...
# class RunState(NamedTuple, Serializable):
class RunState(NamedTuple):
    env: gym.Env
    algo: PO_DQN_ABC
    optimizer: torch.optim.Optimizer
    wandb_logger: WandbLogger
    xstats: XStats
    timer: Timer
    running_averages: Dict[str, RunningAverage]
    dispensers: Dict[str, Dispenser]

    def state_dict(self):
        return {
            'models': self.algo.models.state_dict(),
            'target_models': self.algo.target_models.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'wandb_logger': self.wandb_logger.state_dict(),
            'xstats': self.xstats.state_dict(),
            'timer': self.timer.state_dict(),
            'running_averages': {
                k: v.state_dict() for k, v in self.running_averages.items()
            },
            'dispensers': {
                k: v.state_dict() for k, v in self.dispensers.items()
            },
        }

    def load_state_dict(self, data):
        self.algo.models.load_state_dict(data['models'])
        self.algo.target_models.load_state_dict(data['target_models'])
        self.optimizer.load_state_dict(data['optimizer'])
        self.wandb_logger.load_state_dict(data['wandb_logger'])
        self.xstats.load_state_dict(data['xstats'])
        self.timer.load_state_dict(data['timer'])

        data_keys = data['running_averages'].keys()
        self_keys = self.running_averages.keys()
        if set(data_keys) != set(self_keys):
            raise RuntimeError()
        for k, running_average in self.running_averages.items():
            running_average.load_state_dict(data['running_averages'][k])

        data_keys = data['dispensers'].keys()
        self_keys = self.dispensers.keys()
        if set(data_keys) != set(self_keys):
            raise RuntimeError()
        for k, dispenser in self.dispensers.items():
            dispenser.load_state_dict(data['dispensers'][k])


def setup() -> RunState:
    config = get_config()

    env = make_env(
        config.env,
        max_episode_timesteps=config.max_episode_timesteps,
    )

    algo = make_dqn_algorithm(
        config.algo,
        env,
        truncated_histories=config.truncated_histories,
        truncated_histories_n=config.truncated_histories_n,
    )

    optimizer = torch.optim.Adam(
        algo.models.parameters(),
        lr=config.optim_lr,
        eps=config.optim_eps,
    )

    wandb_logger = WandbLogger()

    xstats = XStats()
    timer = Timer()

    running_averages = {
        'avg_target_returns': InfiniteRunningAverage(),
        'avg_behavior_returns': InfiniteRunningAverage(),
        'avg100_behavior_returns': WindowRunningAverage(100),
    }

    wandb_log_period = config.max_simulation_timesteps // config.num_wandb_logs
    dispensers = {
        'target_update_dispenser': Dispenser(config.target_update_period),
        'wandb_log_dispenser': Dispenser(wandb_log_period),
    }

    return RunState(
        env,
        algo,
        optimizer,
        wandb_logger,
        xstats,
        timer,
        running_averages,
        dispensers,
    )


def run(runstate: RunState) -> bool:
    config = get_config()
    logger.info('run %s %s', config.env_label, config.algo_label)

    (
        env,
        algo,
        optimizer,
        wandb_logger,
        xstats,
        timer,
        running_averages,
        dispensers,
    ) = runstate

    avg_target_returns = running_averages['avg_target_returns']
    avg_behavior_returns = running_averages['avg_behavior_returns']
    avg100_behavior_returns = running_averages['avg100_behavior_returns']
    target_update_dispenser = dispensers['target_update_dispenser']
    wandb_log_dispenser = dispensers['wandb_log_dispenser']

    device = get_device(config.device)
    algo.to(device)

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

    epsilon_schedule = make_schedule(
        config.epsilon_schedule,
        value_from=config.epsilon_value_from,
        value_to=config.epsilon_value_to,
        nsteps=config.epsilon_nsteps,
    )

    behavior_policy = algo.behavior_policy(env.action_space)
    target_policy = algo.target_policy()

    prepopulate_policy: Policy
    if xstats.simulation_timesteps == 0:
        # prepopulate with random policy
        prepopulate_policy = RandomPolicy(env.action_space)
        prepopulate_timesteps = config.episode_buffer_prepopulate_timesteps
    else:
        # prepopulate with current behavior policy
        prepopulate_policy = algo.behavior_policy(env.action_space)
        prepopulate_policy.epsilon = epsilon_schedule(
            xstats.simulation_timesteps
            - config.episode_buffer_prepopulate_timesteps
        )
        prepopulate_timesteps = xstats.simulation_timesteps

    # instantiate and prepopulate buffer
    logger.info('prepopulating episode buffer...')
    episode_buffer = EpisodeBuffer(config.episode_buffer_max_timesteps)
    while episode_buffer.num_interactions() < prepopulate_timesteps:
        (episode,) = sample_episodes(env, prepopulate_policy, num_episodes=1)
        episode_buffer.append_episode(episode.torch())
    logger.info('prepopulating DONE')

    if xstats.simulation_timesteps == 0:
        xstats.simulation_episodes = episode_buffer.num_episodes()
        xstats.simulation_timesteps = episode_buffer.num_interactions()

    # setup interrupt flag via signal
    interrupt = False

    def set_interrupt_flag():
        nonlocal interrupt
        interrupt = True
        logger.debug('signal received, setting interrupt=True')

    signal.signal(signal.SIGUSR1, lambda signal, frame: set_interrupt_flag())

    # main learning loop
    wandb.watch(algo.models)
    while xstats.simulation_timesteps < config.max_simulation_timesteps:
        if interrupt:
            break

        algo.models.eval()

        # evaluate target policy
        if config.evaluation and xstats.epoch % config.evaluation_period == 0:
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
            logger.info(
                'EVALUATE epoch %d simulation_step %d return %.3f',
                xstats.epoch,
                xstats.simulation_timesteps,
                returns.mean(),
            )
            wandb_logger.log(
                {
                    **xstats.asdict(),
                    'hours': timer.hours,
                    'diagnostics/target_mean_episode_length': mean_length,
                    'performance/target_mean_return': returns.mean(),
                    'performance/avg_target_mean_return': avg_target_returns.value(),
                }
            )

        # populate episode buffer
        behavior_policy.epsilon = epsilon_schedule(
            xstats.simulation_timesteps
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

        wandb_log = wandb_log_dispenser.dispense(xstats.simulation_timesteps)

        if wandb_log:
            logger.info(
                'behavior log - simulation_step %d return %.3f',
                xstats.simulation_timesteps,
                returns.mean(),
            )
            wandb_logger.log(
                {
                    **xstats.asdict(),
                    'hours': timer.hours,
                    'diagnostics/epsilon': behavior_policy.epsilon,
                    'diagnostics/behavior_mean_episode_length': mean_length,
                    'performance/behavior_mean_return': returns.mean(),
                    'performance/avg_behavior_mean_return': avg_behavior_returns.value(),
                    'performance/avg100_behavior_mean_return': avg100_behavior_returns.value(),
                }
            )

        # storing torch data directly
        episodes = [episode.torch().to(device) for episode in episodes]
        episode_buffer.append_episodes(episodes)
        xstats.simulation_episodes += len(episodes)
        xstats.simulation_timesteps += sum(len(episode) for episode in episodes)

        # target model update
        if target_update_dispenser.dispense(xstats.simulation_timesteps):
            # Update the target network
            algo.target_models.load_state_dict(algo.models.state_dict())

        # train based on episode buffer
        algo.models.train()
        while (
            xstats.training_timesteps
            < (
                xstats.simulation_timesteps
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
            optimizer.step()

            if wandb_log:
                logger.info(
                    'training log - simulation_step %d loss %.3f',
                    xstats.simulation_timesteps,
                    loss,
                )
                wandb_logger.log(
                    {
                        **xstats.asdict(),
                        'hours': timer.hours,
                        'training/loss': loss,
                        'training/gradient_norm': gradient_norm,
                    }
                )

            if config.save_modelseq and config.modelseq_filename is not None:
                data = {
                    'metadata': {'config': config._as_dict()},
                    'data': {
                        'timestep': xstats.simulation_timesteps,
                        'model.state_dict': algo.models.state_dict(),
                    },
                }
                filename = config.modelseq_filename.format(
                    xstats.simulation_timesteps
                )
                save_data(filename, data)

            xstats.optimizer_steps += 1
            if algo.episodic_training:
                xstats.training_episodes += len(episodes)
                xstats.training_timesteps += sum(
                    len(episode) for episode in episodes
                )
            else:
                xstats.training_timesteps += len(batch)

        xstats.epoch += 1

    done = not interrupt

    if done and config.save_model and config.model_filename is not None:
        data = {
            'metadata': {'config': config._as_dict()},
            'data': {'models.state_dict': algo.models.state_dict()},
        }
        save_data(config.model_filename, data)

    return done


def main():
    args = parse_args()
    wandb_kwargs = {
        'project': args.wandb_project,
        'entity': args.wandb_entity,
        'group': args.wandb_group,
        'tags': args.wandb_tags,
        'mode': args.wandb_mode,
        'config': args,
    }

    try:
        checkpoint = load_data(args.checkpoint)
    except (TypeError, FileNotFoundError):
        checkpoint = None
    else:
        wandb_kwargs.update(
            {
                'resume': 'must',
                'id': checkpoint['metadata']['wandb_id'],
            }
        )

    with wandb.init(**wandb_kwargs):
        config = get_config()
        config._update(dict(wandb.config))

        logger.info('setup of runstate...')
        runstate = setup()
        logger.info('setup DONE')

        if checkpoint is not None:
            if checkpoint['metadata']['config'] != config._as_dict():
                raise RuntimeError(
                    'checkpoint config inconsistent with program config'
                )

            logger.debug('updating runstate from checkpoint')
            runstate.load_state_dict(checkpoint['data'])

        logger.info('run...')
        done = run(runstate)
        logger.info('run DONE')

        wandb_run_id = wandb.run.id

    if config.checkpoint is not None:
        if not done:
            logger.info('checkpointing...')
            checkpoint = {
                'metadata': {
                    'config': config._as_dict(),
                    'wandb_id': wandb_run_id,
                },
                'data': runstate.state_dict(),
            }
            save_data(config.checkpoint, checkpoint)
            logger.info('checkpointing DONE')

        else:
            try:
                os.remove(config.checkpoint)
            except FileNotFoundError:
                pass


if __name__ == '__main__':
    logging.config.dictConfig(
        {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'standard': {
                    'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
                },
            },
            'handlers': {
                'default_handler': {
                    'class': 'logging.StreamHandler',
                    'level': 'DEBUG',
                    'formatter': 'standard',
                    'stream': 'ext://sys.stdout',
                },
            },
            'loggers': {
                '': {
                    'handlers': ['default_handler'],
                    'level': 'DEBUG',
                    'propagate': False,
                }
            },
        }
    )

    main()
