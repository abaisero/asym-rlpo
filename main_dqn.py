#!/usr/bin/env python
import argparse
import functools
import logging
import logging.config
import random
import signal
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn
import wandb
import wandb.sdk
from gym_gridverse.rng import reset_gv_rng

from asym_rlpo.algorithms import ValueBasedAlgorithm, make_dqn_algorithm
from asym_rlpo.data import (
    Episode,
    EpisodeBuffer,
    EpisodeBufferSampler,
    EpisodesFactory,
    populate_episode_buffer,
)
from asym_rlpo.data_logging.logger import DataLogger
from asym_rlpo.data_logging.wandb_logger import WandbLogger
from asym_rlpo.envs import Environment, make_env
from asym_rlpo.evaluation import evaluate_episodes
from asym_rlpo.models import make_model_factory
from asym_rlpo.policies import Policy, RandomPolicy
from asym_rlpo.runs.xstats import (
    XStats,
    update_xstats_epoch,
    update_xstats_optimizer,
    update_xstats_simulation,
    update_xstats_training,
)
from asym_rlpo.sampling import sample_episode, sample_episodes
from asym_rlpo.types import GradientNormDict, LossDict
from asym_rlpo.utils.aggregate import average_losses
from asym_rlpo.utils.argparse import (
    int_non_neg,
    int_pos,
    int_pow_2,
    history_model_type,
)
from asym_rlpo.utils.checkpointing import load_data, save_data
from asym_rlpo.utils.config import get_config
from asym_rlpo.utils.device import get_device
from asym_rlpo.utils.dispenser import Dispenser, TimeDispenser
from asym_rlpo.utils.running_average import (
    InfiniteRunningAverage,
    WindowRunningAverage,
)
from asym_rlpo.utils.scheduling import Schedule, make_schedule
from asym_rlpo.utils.target_update_functions import (
    TargetUpdater,
    make_target_updater,
)
from asym_rlpo.utils.timer import Timer, timestamp_is_past

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()

    # wandb arguments
    parser.add_argument('--wandb-entity', default='abaisero')
    parser.add_argument('--wandb-project', default=None)
    parser.add_argument('--wandb-group', default=None)
    parser.add_argument('--wandb-tag', action='append', dest='wandb_tags')
    parser.add_argument('--wandb-offline', action='store_true')

    # custom meta groups
    parser.add_argument(
        '--wandb-metagroup',
        nargs=2,
        action='append',
        dest='wandb_metagroups',
        default=[],
    )

    # data-logging
    parser.add_argument('--num-data-logs', type=int_pos, default=200)

    # algorithm and environment
    parser.add_argument('env')
    parser.add_argument(
        'algo',
        choices=[
            'dqn',
            'adqn',
            'adqn-vr',
            'adqn-state',
            'adqn-state-vr',
            'adqn-short',
            'adqn-short-vr',
        ],
    )

    parser.add_argument('--env-label', default=None)
    parser.add_argument('--algo-label', default=None)

    # truncated histories
    parser.add_argument(
        '--history-model',
        type=history_model_type,
        default='gru',
    )
    parser.add_argument(
        '--history-model-memory-size', type=int_non_neg, default=0
    )

    # reproducibility
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--deterministic', action='store_true')

    # general
    parser.add_argument(
        '--max-simulation-timesteps', type=int_pos, default=2_000_000
    )
    parser.add_argument('--max-episode-timesteps', type=int_pos, default=1_000)
    parser.add_argument('--simulation-num-episodes', type=int_pos, default=1)

    # evaluation
    parser.add_argument('--evaluation', action='store_true')
    parser.add_argument('--evaluation-period', type=int_pos, default=10)
    parser.add_argument('--evaluation-num-episodes', type=int_pos, default=1)

    # discounts
    parser.add_argument('--evaluation-discount', type=float, default=1.0)
    parser.add_argument('--training-discount', type=float, default=0.99)

    # episode buffer
    parser.add_argument(
        '--episode-buffer-max-timesteps', type=int_pos, default=1_000_000
    )
    parser.add_argument(
        '--episode-buffer-prepopulate-timesteps', type=int_pos, default=50_000
    )
    # target
    parser.add_argument(
        '--target-update-function', choices=['full', 'polyak'], default='full'
    )
    parser.add_argument(
        '--target-update-full-period', type=int_pos, default=10_000
    )
    parser.add_argument('--target-update-polyak-tau', type=float, default=0.001)

    # training parameters
    parser.add_argument(
        '--training-timesteps-per-simulation-timestep', type=int_pos, default=8
    )
    parser.add_argument('--training-num-episodes', type=int_pos, default=1)

    # epsilon schedule
    parser.add_argument('--epsilon-schedule', default='linear')
    parser.add_argument('--epsilon-value-from', type=float, default=1.0)
    parser.add_argument('--epsilon-value-to', type=float, default=0.1)
    parser.add_argument('--epsilon-nsteps', type=int_pos, default=1_000_000)

    # optimization
    parser.add_argument('--optim-lr', type=float, default=1e-4)
    parser.add_argument('--optim-eps', type=float, default=1e-4)
    parser.add_argument('--optim-max-norm', type=float, default=float('inf'))

    # device
    parser.add_argument('--device', default='auto')

    # temporary / development
    parser.add_argument('--hs-features-dim', type=int_non_neg, default=0)
    parser.add_argument('--normalize-hs-features', action='store_true')

    # latent observation
    parser.add_argument(
        '--latent-type', default='state', choices=['state', 'heaven', 'beacon-color']
    )

    # representation options
    parser.add_argument('--attention-num-heads', type=int_pow_2, default=2)

    # gv models
    parser.add_argument('--gv-representation', default='compact')
    parser.add_argument('--gv-ignore-color-channel', action='store_true')
    parser.add_argument('--gv-ignore-state-channel', action='store_true')
    parser.add_argument('--gv-cnn', default=None)

    parser.add_argument(
        '--gv-observation-submodels',
        nargs='+',
        choices=[
            'agent',
            'item',
            'grid-cnn',
            'grid-fc',
            'agent-grid-cnn',
            'agent-grid-fc',
        ],
    )
    parser.add_argument(
        '--gv-observation-representation-layers',
        type=int_non_neg,
        default=0,
    )

    parser.add_argument(
        '--gv-state-submodels',
        nargs='+',
        choices=[
            'agent',
            'item',
            'grid-cnn',
            'grid-fc',
            'agent-grid-cnn',
            'agent-grid-fc',
        ],
    )
    parser.add_argument(
        '--gv-state-representation-layers',
        type=int_non_neg,
        default=0,
    )

    # checkpoint
    parser.add_argument('--run-path', default=None)

    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--checkpoint-period', type=int_pos, default=10 * 60)
    parser.add_argument('--check-checkpoint-consistency', action='store_true')

    parser.add_argument('--timeout-timestamp', type=float, default=float('inf'))

    parser.add_argument('--save-model', action='store_true')
    parser.add_argument('--save-modelseq', action='store_true')

    args = parser.parse_args()

    args.env_label = args.env if args.env_label is None else args.env_label
    args.algo_label = args.algo if args.algo_label is None else args.algo_label
    args.wandb_mode = 'offline' if args.wandb_offline else None

    if args.run_path is None:
        args.checkpoint_path = None
        args.model_path = None
        args.modelseq_path_template = None
    else:
        args.checkpoint_path = f'{args.run_path}/checkpoint.pkl'
        args.model_path = f'{args.run_path}/model.pkl'
        args.modelseq_path_template = (
            f'{args.run_path}/modelseq/modelseq.{{}}.pkl'
        )

    for name, value in args.wandb_metagroups:
        setattr(args, f'wandb_metagroup_{name}', value)

    return args


@dataclass
class Controlflow:
    log_data: bool = False
    update_target_parameters: bool = False
    evaluate: bool = False
    train: bool = False
    save_modelseq: bool = False


@dataclass
class Runflags:
    done: bool = False
    timeout: bool = False
    interrupt: bool = False

    def stop_run(self) -> bool:
        return self.done or self.timeout or self.interrupt


class RunstatePolicies(NamedTuple):
    behavior: Policy
    target: Policy


class RunstateAverages(NamedTuple):
    target: InfiniteRunningAverage
    behavior: InfiniteRunningAverage
    behavior100: WindowRunningAverage


class RunstateDispensers(NamedTuple):
    target_update: Dispenser
    datalog: Dispenser
    checkpoint: TimeDispenser


class RunstateEpisodesFactories(NamedTuple):
    behavior_factory: EpisodesFactory
    evaluation_factory: EpisodesFactory
    episode_buffer_factory: EpisodesFactory


class Runstate(NamedTuple):
    # original runstate
    env: Environment
    algo: ValueBasedAlgorithm
    policies: RunstatePolicies
    episode_buffer: EpisodeBuffer
    datalogger: DataLogger
    timer: Timer
    xstats: XStats
    averages: RunstateAverages
    dispensers: RunstateDispensers
    target_updater: TargetUpdater
    # original loopstate
    episodes_factories: RunstateEpisodesFactories
    device: torch.device
    epsilon_schedule: Schedule


class CheckpointMetadata(NamedTuple):
    config: dict
    wandb_run_id: str


class CheckpointData(NamedTuple):
    algo_state_dict: dict
    episode_buffer: EpisodeBuffer
    datalogger: DataLogger
    timer: Timer
    xstats: XStats
    averages: RunstateAverages
    dispensers: RunstateDispensers


class Checkpoint(NamedTuple):
    metadata: CheckpointMetadata
    data: CheckpointData


def make_runstate(checkpoint: Checkpoint | None) -> Runstate:
    config = get_config()

    env = make_env(
        config.env,
        latent_type=config.latent_type,
        max_episode_timesteps=config.max_episode_timesteps,
        gv_representation=config.gv_representation,
    )

    def optimizer_factory(parameters) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            parameters,
            lr=config.optim_lr,
            eps=config.optim_eps,
        )

    model_factory = make_model_factory(env)
    model_factory.history_model = config.history_model
    model_factory.attention_num_heads = config._get('attention_num_heads')
    model_factory.history_model_memory_size = config.history_model_memory_size

    algo = make_dqn_algorithm(
        config.algo,
        model_factory,
        optimizer_factory=optimizer_factory,
        max_gradient_norm=config.optim_max_norm,
    )

    device = get_device(config.device)
    algo.models.to(device)

    datalogger = WandbLogger()

    timer = Timer()
    xstats = XStats()

    averages = RunstateAverages(
        target=InfiniteRunningAverage(),
        behavior=InfiniteRunningAverage(),
        behavior100=WindowRunningAverage(100),
    )

    def make_target_update_dispenser():
        if config.target_update_function == 'full':
            return Dispenser(0, config.target_update_full_period)

        if config.target_update_function == 'polyak':
            return Dispenser(0, 0)

        assert False

    datalog_period = config.max_simulation_timesteps // config.num_data_logs
    dispensers = RunstateDispensers(
        target_update=make_target_update_dispenser(),
        datalog=Dispenser(0, datalog_period),
        checkpoint=TimeDispenser(config.checkpoint_period),
    )
    dispensers.checkpoint.dispense()  # consume first checkpoint dispense

    policies = RunstatePolicies(
        algo.qha_model.epsilon_greedy_policy(),
        algo.qha_model.policy(),
    )

    episode_buffer = (
        EpisodeBuffer(config.episode_buffer_max_timesteps)
        if checkpoint is None
        else checkpoint.data.episode_buffer
    )
    episode_buffer_sampler = EpisodeBufferSampler(episode_buffer)

    episodes_factories = RunstateEpisodesFactories(
        behavior_factory=functools.partial(
            sample_episodes,
            env,
            policies.behavior,
            num_episodes=config.simulation_num_episodes,
        ),
        evaluation_factory=functools.partial(
            sample_episodes,
            env,
            policies.target,
            num_episodes=config.simulation_num_episodes,
        ),
        episode_buffer_factory=functools.partial(
            episode_buffer_sampler.sample_episodes,
            config.training_num_episodes,
            replacement=True,
        ),
    )

    epsilon_schedule = make_schedule(
        config.epsilon_schedule,
        value_from=config.epsilon_value_from,
        value_to=config.epsilon_value_to,
        nsteps=config.epsilon_nsteps,
    )

    target_updater = make_target_updater(
        config.target_update_function,
        tau=config.target_update_polyak_tau,
    )

    if checkpoint is not None:
        algo.load_state_dict(checkpoint.data.algo_state_dict)

        datalogger = checkpoint.data.datalogger
        timer = checkpoint.data.timer
        xstats = checkpoint.data.xstats
        averages = checkpoint.data.averages
        dispensers = checkpoint.data.dispensers

    return Runstate(
        # original runstate
        env,
        algo,
        policies,
        episode_buffer,
        datalogger,
        timer,
        xstats,
        averages,
        dispensers,
        target_updater,
        # original loopstate
        episodes_factories,
        device,
        epsilon_schedule,
    )


def make_checkpoint(runstate: Runstate) -> Checkpoint:
    config = get_config()

    return Checkpoint(
        CheckpointMetadata(
            config._as_dict(),
            config.wandb_run_id,
        ),
        CheckpointData(
            runstate.algo.state_dict(),
            runstate.episode_buffer,
            runstate.datalogger,
            runstate.timer,
            runstate.xstats,
            runstate.averages,
            runstate.dispensers,
        ),
    )


def save_checkpoint(runstate: Runstate):
    config = get_config()

    if config.checkpoint_path is None:
        logger.info('no check point path available')
        return

    checkpoint = make_checkpoint(runstate)
    save_data(config.checkpoint_path, checkpoint)


def save_model(models):
    config = get_config()

    data = {
        'metadata': {'config': config._as_dict()},
        'data': {'models.state_dict': models.state_dict()},
    }
    save_data(config.model_filename, data)


def run(runstate: Runstate) -> Runflags:
    config = get_config()
    logger.info('run %s %s', config.env_label, config.algo_label)

    # TODO somehow integrate reproducibility stuff into the checkpoint
    if config.seed is not None:
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        reset_gv_rng(config.seed)
        runstate.env.seed(config.seed)

    if config.deterministic:
        torch.use_deterministic_algorithms(True)

    controlflow = Controlflow()
    runflags = Runflags()

    setup_interruption_handling(runflags)

    wandb.watch(runstate.algo.models)
    if runstate.episode_buffer.num_episodes() == 0:
        prepopulate_episode_buffer(runstate)

    while True:
        update_runflags(runstate, runflags)

        if runflags.stop_run():
            break

        run_epoch(runstate, controlflow)

        if runstate.dispensers.checkpoint.dispense():
            save_checkpoint(runstate)

    save_checkpoint(runstate)

    if runflags.done and config.save_model:
        save_model(runstate.algo.models)

    return runflags


def setup_interruption_handling(runflags: Runflags):
    def handle_interrupt(signum, _):
        logger.info(f'handling signal {signal.Signals(signum)!r}')
        runflags.interrupt = True

    signal.signal(signal.SIGINT, handle_interrupt)
    signal.signal(signal.SIGTERM, handle_interrupt)


def prepopulate_episode_buffer(runstate: Runstate):
    config = get_config()

    random_episode_factory = functools.partial(
        sample_episode,
        runstate.env,
        RandomPolicy(runstate.env.action_space),
    )

    populate_episode_buffer(
        runstate.episode_buffer,
        random_episode_factory,
        timesteps=config.episode_buffer_prepopulate_timesteps,
    )

    update_xstats_simulation(
        runstate.xstats,
        runstate.episode_buffer.episodes,
    )


def update_runflags(runstate: Runstate, runflags: Runflags):
    config = get_config()

    runflags.done = (
        runstate.xstats.simulation_timesteps >= config.max_simulation_timesteps
    )
    runflags.timeout = timestamp_is_past(config.timeout_timestamp)


def update_epoch_controlflow(runstate: Runstate, controlflow: Controlflow):
    config = get_config()

    log_data = runstate.dispensers.datalog.dispense(
        runstate.xstats.simulation_timesteps
    )
    update_target = runstate.dispensers.target_update.dispense(
        runstate.xstats.simulation_timesteps
    )
    evaluate = runstate.xstats.epoch % config.evaluation_period == 0

    controlflow.log_data = log_data
    controlflow.update_target_parameters = update_target
    controlflow.evaluate = evaluate and config.evaluation
    controlflow.save_modelseq = log_data and config.save_modelseq


def update_training_controlflow(runstate: Runstate, controlflow: Controlflow):
    config = get_config()

    controlflow.train = (
        runstate.xstats.training_timesteps
        < (
            runstate.xstats.simulation_timesteps
            - config.episode_buffer_prepopulate_timesteps
        )
        * config.training_timesteps_per_simulation_timestep
    )


def log_xstats(runstate: Runstate):
    runstate.datalogger.log(
        {
            'runstate.xstats.epoch': runstate.xstats.epoch,
            'runstate.xstats.simulation_episodes': runstate.xstats.simulation_episodes,
            'runstate.xstats.simulation_timesteps': runstate.xstats.simulation_timesteps,
            'runstate.xstats.training_episodes': runstate.xstats.training_episodes,
            'runstate.xstats.training_timesteps': runstate.xstats.training_timesteps,
            'runstate.xstats.optimizer_steps': runstate.xstats.optimizer_steps,
            'hours': runstate.timer.hours,
        },
        commit=False,
    )


def run_epoch(runstate: Runstate, controlflow: Controlflow):
    update_epoch_controlflow(runstate, controlflow)

    if controlflow.log_data:
        log_xstats(runstate)

    if controlflow.evaluate:
        run_evaluation(runstate)

    episodes = run_simulation(runstate, controlflow)
    episodes = [episode.torch() for episode in episodes]
    runstate.episode_buffer.append_episodes(episodes)

    # TODO probably makes more sense to move this into run_training_step for polyak
    if controlflow.update_target_parameters:
        runstate.target_updater(runstate.algo.target_pairs())

    run_training(runstate, controlflow)

    if controlflow.save_modelseq:
        save_modelseq(
            runstate.xstats.simulation_timesteps,
            runstate.algo.models,
        )

    update_xstats_epoch(runstate.xstats)

    if controlflow.log_data:
        runstate.datalogger.commit()


def run_evaluation(runstate: Runstate):
    runstate.algo.models.eval()

    with torch.inference_mode():
        episodes = runstate.episodes_factories.evaluation_factory()

    runstate.algo.models.train()

    log_evaluation(runstate, episodes)


def log_evaluation(runstate: Runstate, episodes: Sequence[Episode]):
    config = get_config()

    evalstats = evaluate_episodes(episodes, discount=config.evaluation_discount)
    runstate.averages.target.extend(evalstats.returns.tolist())

    logger.info(
        '%s - EVALUATE - epoch %d simulation_timestep %d return %.3f',
        runstate.timer,
        runstate.xstats.epoch,
        runstate.xstats.simulation_timesteps,
        evalstats.returns.mean(),
    )

    runstate.datalogger.log(
        {
            'diagnostics/target_mean_episode_length': evalstats.lengths.mean(),
            'performance/target_mean_return': evalstats.returns.mean(),
            'performance/avg_target_mean_return': runstate.averages.target.value(),
        },
        commit=False,
    )


def run_simulation(
    runstate: Runstate,
    controlflow: Controlflow,
) -> list[Episode]:
    runstate.policies.behavior.epsilon = runstate.epsilon_schedule(
        runstate.xstats.simulation_timesteps
    )
    episodes = runstate.episodes_factories.behavior_factory()

    if controlflow.log_data:
        log_simulation(runstate, episodes)

    update_xstats_simulation(runstate.xstats, episodes)

    return episodes


def log_simulation(runstate: Runstate, episodes: Sequence[Episode]):
    config = get_config()

    evalstats = evaluate_episodes(episodes, discount=config.evaluation_discount)
    runstate.averages.behavior.extend(evalstats.returns.tolist())
    runstate.averages.behavior100.extend(evalstats.returns.tolist())

    logger.info(
        '%s - BEHAVIOR - epoch %d simulation_timestep %d '
        'epsilon %.3f '
        'return %.3f '
        'avg100 %.3f',
        runstate.timer,
        runstate.xstats.epoch,
        runstate.xstats.simulation_timesteps,
        runstate.policies.behavior.epsilon,
        evalstats.returns.mean(),
        runstate.averages.behavior100.value(),
    )

    runstate.datalogger.log(
        {
            'diagnostics/behavior_epsilon': runstate.policies.behavior.epsilon,
            'diagnostics/behavior_mean_episode_length': evalstats.lengths.mean(),
            'performance/behavior_mean_return': evalstats.returns.mean(),
            'performance/avg_behavior_mean_return': runstate.averages.behavior.value(),
            'performance/avg100_behavior_mean_return': runstate.averages.behavior100.value(),
        },
        commit=False,
    )


class TrainingData(NamedTuple):
    losses: LossDict
    gradient_norms: GradientNormDict


def run_training(runstate: Runstate, controlflow: Controlflow):
    training_datas: list[TrainingData] = []

    while True:
        update_training_controlflow(runstate, controlflow)

        if not controlflow.train:
            break

        training_data = run_training_step(runstate)
        training_datas.append(training_data)

        if controlflow.log_data:
            log_training_step(runstate, training_data)

    if controlflow.log_data and training_datas:
        log_training(runstate, training_datas)


def run_training_step(runstate: Runstate) -> TrainingData:
    config = get_config()

    episodes = runstate.episodes_factories.episode_buffer_factory()
    episodes = [episode.to(runstate.device) for episode in episodes]

    losses = average_losses(
        [
            runstate.algo.compute_losses(
                episode,
                discount=config.training_discount,
            )
            for episode in episodes
        ]
    )
    gradient_norms = runstate.algo.trainer.gradient_step(losses)

    update_xstats_training(runstate.xstats, episodes)
    update_xstats_optimizer(runstate.xstats)

    return TrainingData(losses, gradient_norms)


def log_training_step(runstate: Runstate, training_data: TrainingData):
    losses_string = ' '.join(
        f'{k}-loss {loss:.3f}' for k, loss in training_data.losses.items()
    )
    logger.info(
        '%s - TRAINING - epoch %d simulation_timestep %d %s',
        runstate.timer,
        runstate.xstats.epoch,
        runstate.xstats.simulation_timesteps,
        losses_string,
    )


def log_training(runstate: Runstate, training_datas: Sequence[TrainingData]):
    keys = training_datas[0].losses.keys()
    losses_logdata = {
        f'training/losses/{key}': [
            training_data.losses[key] for training_data in training_datas
        ]
        for key in keys
    }
    gradient_norms_logdata = {
        f'training/gradient_norms/{key}': [
            training_data.gradient_norms[key]
            for training_data in training_datas
        ]
        for key in keys
    }
    runstate.datalogger.log(
        {**losses_logdata, **gradient_norms_logdata},
        commit=False,
    )


def save_modelseq(timestep: int, models: nn.Module):
    config = get_config()
    data = {
        'metadata': {'config': config._as_dict()},
        'data': {
            'timestep': timestep,
            'model.state_dict': models.state_dict(),
        },
    }
    filename = config.modelseq_path_template.format(timestep)
    save_data(filename, data)


def define_metrics():
    wandb.define_metric('epoch')
    wandb.define_metric('simulation_episodes')
    wandb.define_metric('simulation_timesteps')
    wandb.define_metric('training_episodes')
    wandb.define_metric('training_timesteps')
    wandb.define_metric('optimizer_steps')

    wandb.define_metric('hours')

    wandb.define_metric(
        'diagnostics/target_mean_episode_length',
        step_metric='simulation_timesteps',
    )
    wandb.define_metric(
        'performance/target_mean_return',
        step_metric='simulation_timesteps',
    )
    wandb.define_metric(
        'performance/avg_target_mean_return',
        step_metric='simulation_timesteps',
    )

    wandb.define_metric(
        'diagnostics/behavior_epsilon',
        step_metric='simulation_timesteps',
    )
    wandb.define_metric(
        'diagnostics/behavior_mean_episode_length',
        step_metric='simulation_timesteps',
    )

    wandb.define_metric(
        'performance/behavior_mean_return',
        step_metric='simulation_timesteps',
    )
    wandb.define_metric(
        'performance/avg_behavior_mean_return',
        step_metric='simulation_timesteps',
    )
    wandb.define_metric(
        'performance/avg100_behavior_mean_return',
        step_metric='simulation_timesteps',
    )

    wandb.define_metric(
        'training/losses',
        step_metric='simulation_timesteps',
    )
    wandb.define_metric(
        'training/gradient_norms',
        step_metric='simulation_timesteps',
    )


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

    checkpoint: Checkpoint | None
    try:
        checkpoint = load_data(args.checkpoint_path)
    except (TypeError, FileNotFoundError):
        checkpoint = None
    else:
        assert checkpoint is not None
        wandb_kwargs.update(
            {
                'resume': 'must',
                'id': checkpoint.metadata.wandb_run_id,
            }
        )

    wandb.init(**wandb_kwargs)
    define_metrics()

    config = get_config()
    config._update(dict(wandb.config))
    assert wandb.run is not None
    config._update({'wandb_run_id': wandb.run.id})

    if (
        checkpoint is not None
        and config.check_checkpoint_consistency
        and checkpoint.metadata.config != config._as_dict()
    ):
        raise RuntimeError('checkpoint config inconsistent with program config')

    logger.info('making runstate')
    runstate = make_runstate(checkpoint)

    logger.info('starting run')
    runflags = run(runstate)
    logger.info(f'stopping run with flags {runflags}')

    retvalue = int(not runflags.done)
    logger.info(f'returning {retvalue}')

    return retvalue


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

    sys.exit(main())
