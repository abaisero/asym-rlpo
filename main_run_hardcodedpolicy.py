#!/usr/bin/env python
import argparse
import logging
import logging.config
import random
from dataclasses import asdict, dataclass
from typing import Dict, NamedTuple

import gym
import gym.spaces
import numpy as np
import torch
import torch.nn as nn
import wandb
from gym_gridverse.rng import reset_gv_rng

from asym_rlpo.data_logging.wandb_logger import (
    WandbLogger,
    WandbLoggerSerializer,
)
from asym_rlpo.envs import Environment, LatentType, make_env
from asym_rlpo.evaluation import evaluate, evaluate_returns
from asym_rlpo.policies import Policy
from asym_rlpo.q_estimators import q_estimator_factory
from asym_rlpo.sampling import sample_episodes
from asym_rlpo.utils.aggregate import average
from asym_rlpo.utils.checkpointing import Serializer, load_data, save_data
from asym_rlpo.utils.config import get_config
from asym_rlpo.utils.device import get_device
from asym_rlpo.utils.dispenser import (
    StepDispenser,
    StepDispenserSerializer,
    TimePeriodDispenser,
    TimestampDispenser,
)
from asym_rlpo.utils.running_average import (
    InfiniteRunningAverage,
    RunningAverage,
    RunningAverageSerializer,
    WindowRunningAverage,
)
from asym_rlpo.utils.scheduling import make_schedule
from asym_rlpo.utils.timer import Timer, TimerSerializer

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()

    # wandb arguments
    parser.add_argument("--wandb-entity", default="abaisero")
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-tag", action="append", dest="wandb_tags")
    parser.add_argument("--wandb-offline", action="store_true")

    # wandb related
    parser.add_argument("--num-wandb-logs", type=int, default=200)

    # algorithm and environment
    parser.add_argument("env")

    parser.add_argument("--env-label", default=None)
    parser.add_argument("--algo-label", default=None)

    # truncated histories
    parser.add_argument(
        "--history-model",
        choices=["rnn", "gru", "attention"],
        default="gru",
    )
    parser.add_argument("--truncated-histories-n", type=int, default=None)

    # reproducibility
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--deterministic", action="store_true")

    # general
    parser.add_argument("--max-simulation-timesteps", type=int, default=2_000_000)
    parser.add_argument("--max-episode-timesteps", type=int, default=1_000)
    parser.add_argument("--simulation-num-episodes", type=int, default=1)

    # evaluation
    parser.add_argument("--evaluation", action="store_true")
    parser.add_argument("--evaluation-period", type=int, default=10)
    parser.add_argument("--evaluation-num-episodes", type=int, default=1)
    parser.add_argument("--evaluation-epsilon", type=float, default=1.0)

    # discounts
    parser.add_argument("--evaluation-discount", type=float, default=1.0)
    parser.add_argument("--training-discount", type=float, default=0.99)

    # target
    parser.add_argument("--target-update-period", type=int, default=10_000)

    # q-estimator
    parser.add_argument(
        "--q-estimator",
        choices=["mc", "td0", "td-n", "td-lambda"],
        default="td0",
    )
    parser.add_argument("--q-estimator-n", type=int, default=None)
    parser.add_argument("--q-estimator-lambda", type=float, default=None)

    # negentropy schedule
    parser.add_argument("--negentropy-schedule", default="linear")
    # linear
    parser.add_argument("--negentropy-value-from", type=float, default=1.0)
    parser.add_argument("--negentropy-value-to", type=float, default=0.01)
    parser.add_argument("--negentropy-nsteps", type=int, default=2_000_000)
    # exponential
    parser.add_argument("--negentropy-halflife", type=int, default=500_000)

    # optimization
    parser.add_argument("--optim-lr-actor", type=float, default=1e-4)
    parser.add_argument("--optim-eps-actor", type=float, default=1e-4)
    parser.add_argument("--optim-lr-critic", type=float, default=1e-4)
    parser.add_argument("--optim-eps-critic", type=float, default=1e-4)
    parser.add_argument("--optim-max-norm", type=float, default=float("inf"))

    # device
    parser.add_argument("--device", default="auto")

    # temporary / development
    parser.add_argument("--hs-features-dim", type=int, default=0)
    parser.add_argument("--normalize-hs-features", action="store_true")

    # latent observation
    parser.add_argument("--latent-type", default="state")

    # representation options
    parser.add_argument(
        "--attention-num-heads",
        choices=[2**k for k in range(10)],
        type=int,
        default=2,
    )

    # gv models
    parser.add_argument("--gv-observation-representation", default="compact")
    parser.add_argument("--gv-state-representation", default="compact")

    parser.add_argument(
        "--gv-observation-grid-model-type",
        choices=["cnn", "fc"],
        default="fc",
    )
    parser.add_argument(
        "--gv-observation-representation-layers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--gv-state-grid-model-type",
        choices=["cnn", "fc"],
        default="fc",
    )
    parser.add_argument(
        "--gv-state-representation-layers",
        type=int,
        default=0,
    )

    # checkpoint
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoint-period", type=int, default=10 * 60)

    parser.add_argument("--timeout-timestamp", type=float, default=float("inf"))

    parser.add_argument("--save-model", action="store_true")
    parser.add_argument("--model-filename", default=None)

    parser.add_argument("--save-modelseq", action="store_true")
    parser.add_argument("--modelseq-filename", default=None)

    args = parser.parse_args()
    args.env_label = args.env if args.env_label is None else args.env_label
    # args.algo_label = args.algo if args.algo_label is None else args.algo_label
    args.wandb_mode = "offline" if args.wandb_offline else None
    return args


@dataclass
class XStats:
    epoch: int = 0
    simulation_episodes: int = 0
    simulation_timesteps: int = 0
    optimizer_steps: int = 0
    training_episodes: int = 0
    training_timesteps: int = 0

    def asdict(self):
        return asdict(self)


class XStatsSerializer(Serializer[XStats]):
    def serialize(self, xstats: XStats) -> Dict:
        return xstats.asdict()

    def deserialize(self, xstats: XStats, data: Dict):
        xstats.epoch = data["epoch"]
        xstats.simulation_episodes = data["simulation_episodes"]
        xstats.simulation_timesteps = data["simulation_timesteps"]
        xstats.optimizer_steps = data["optimizer_steps"]
        xstats.training_episodes = data["training_episodes"]
        xstats.training_timesteps = data["training_timesteps"]


class RunState(NamedTuple):
    env: Environment
    policy: Policy
    wandb_logger: WandbLogger
    xstats: XStats
    timer: Timer
    running_averages: Dict[str, RunningAverage]


class RunStateSerializer(Serializer[RunState]):
    def __init__(self):
        self.wandb_logger_serializer = WandbLoggerSerializer()
        self.xstats_serializer = XStatsSerializer()
        self.timer_serializer = TimerSerializer()
        self.running_average_serializer = RunningAverageSerializer()

    def serialize(self, runstate: RunState) -> Dict:
        return {
            "wandb_logger": self.wandb_logger_serializer.serialize(
                runstate.wandb_logger
            ),
            "xstats": self.xstats_serializer.serialize(runstate.xstats),
            "timer": self.timer_serializer.serialize(runstate.timer),
            "running_averages": {
                k: self.running_average_serializer.serialize(v)
                for k, v in runstate.running_averages.items()
            },
        }

    def deserialize(self, runstate: RunState, data: Dict):
        self.wandb_logger_serializer.deserialize(
            runstate.wandb_logger,
            data["wandb_logger"],
        )
        self.xstats_serializer.deserialize(runstate.xstats, data["xstats"])
        self.timer_serializer.deserialize(runstate.timer, data["timer"])

        data_keys = data["running_averages"].keys()
        obj_keys = runstate.running_averages.keys()
        if set(data_keys) != set(obj_keys):
            raise RuntimeError()
        for k, running_average in runstate.running_averages.items():
            self.running_average_serializer.deserialize(
                running_average,
                data["running_averages"][k],
            )


class HardcodedPolicy(Policy):
    def __init__(self, action_space: gym.spaces.Space):
        super().__init__()

        self.action_space = action_space

    def reset(self, observation):
        # @ Ankit: TODO implement reset
        pass

    def step(self, action, observation):
        # @ Ankit: TODO implement step function
        pass

    def sample_action(self):
        # @ Ankit: TODO implement sample function
        return self.action_space.sample()


def make_policy(env_name: str, env: Environment) -> Policy:
    # TODO: based on env_name and env, create a specific hardcoded policy
    return HardcodedPolicy(env.action_space)


def setup() -> RunState:
    config = get_config()

    table = str.maketrans({"-": "_"})
    latent_type = LatentType[config.latent_type.upper().translate(table)]
    env = make_env(
        config.env,
        latent_type=latent_type,
        max_episode_timesteps=config.max_episode_timesteps,
    )

    policy = make_policy(config.env, env)

    wandb_logger = WandbLogger()

    xstats = XStats()
    timer = Timer()

    running_averages = {
        "avg_returns": InfiniteRunningAverage(),
        "avg100_returns": WindowRunningAverage(100),
    }

    return RunState(
        env,
        policy,
        wandb_logger,
        xstats,
        timer,
        running_averages,
    )


def save_checkpoint(runstate: RunState):
    """saves a checkpoint with the current runstate

    NOTE:  must be called within an active wandb.init context manager
    """
    config = get_config()

    if config.checkpoint is not None:
        assert wandb.run is not None

        logger.info("checkpointing...")
        runstate_serializer = RunStateSerializer()
        checkpoint = {
            "metadata": {
                "config": config._as_dict(),
                "wandb_id": wandb.run.id,
            },
            "data": runstate_serializer.serialize(runstate),
        }
        save_data(config.checkpoint, checkpoint)
        logger.info("checkpointing DONE")


def run(runstate: RunState) -> bool:
    config = get_config()
    logger.info("run %s %s", config.env_label)
    # logger.info("run %s %s", config.env_label, config.algo_label)

    (
        env,
        policy,
        wandb_logger,
        xstats,
        timer,
        running_averages,
    ) = runstate

    avg_returns = running_averages["avg_returns"]

    # reproducibility
    if config.seed is not None:
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        reset_gv_rng(config.seed)
        env.seed(config.seed)

    if config.deterministic:
        torch.use_deterministic_algorithms(True)

    # main learning loop
    for _ in range(1000):
        with torch.inference_mode():
            evalstats = evaluate(
                env,
                policy,
                discount=config.evaluation_discount,
                num_episodes=config.evaluation_num_episodes,
            )

            avg_returns.extend(evalstats.returns.tolist())
            logger.info(
                "EVALUATE epoch %d simulation_timestep %d return % .3f",
                xstats.epoch,
                xstats.simulation_timesteps,
                evalstats.returns.mean(),
            )
            wandb_logger.log(
                {
                    **xstats.asdict(),
                    "hours": timer.hours,
                    "diagnostics/target_mean_episode_length": evalstats.lengths.mean(),
                    "performance/target_mean_return": evalstats.returns.mean(),
                    "performance/avg_mean_return": avg_returns.value(),
                }
            )

    done = True

    return done


def main():
    args = parse_args()
    wandb_kwargs = {
        "project": args.wandb_project,
        "entity": args.wandb_entity,
        "group": args.wandb_group,
        "tags": args.wandb_tags,
        "mode": args.wandb_mode,
        "config": args,
    }

    try:
        checkpoint = load_data(args.checkpoint)
    except (TypeError, FileNotFoundError):
        checkpoint = None
    else:
        wandb_kwargs.update(
            {
                "resume": "must",
                "id": checkpoint["metadata"]["wandb_id"],
            }
        )

    with wandb.init(**wandb_kwargs):
        config = get_config()
        config._update(dict(wandb.config))

        logger.info("setup of runstate...")
        runstate = setup()
        logger.info("setup DONE")

        if checkpoint is not None:
            if checkpoint["metadata"]["config"] != config._as_dict():
                raise RuntimeError("checkpoint config inconsistent with program config")

            logger.debug("updating runstate from checkpoint")
            runstate_serializer = RunStateSerializer()
            runstate_serializer.deserialize(runstate, checkpoint["data"])

        logger.info("run...")
        done = run(runstate)
        logger.info("run DONE")

    done = True
    return int(not done)


if __name__ == "__main__":
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
                },
            },
            "handlers": {
                "default_handler": {
                    "class": "logging.StreamHandler",
                    "level": "DEBUG",
                    "formatter": "standard",
                    "stream": "ext://sys.stdout",
                },
            },
            "loggers": {
                "": {
                    "handlers": ["default_handler"],
                    "level": "DEBUG",
                    "propagate": False,
                }
            },
        }
    )

    raise SystemExit(main())
