#!/usr/bin/env python
import argparse
import logging
import logging.config
import random
from typing import NamedTuple

import numpy as np
import torch
import yaml
from gym_gridverse.rng import reset_gv_rng

from asym_rlpo.envs import Environment, LatentType, make_env
from asym_rlpo.evaluation import evaluate
from asym_rlpo.policies import Policy
from asym_rlpo.policies.hardcoded import make_hardcoded_policy
from asym_rlpo.utils.config import get_config

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()

    # algorithm and environment
    parser.add_argument("env")

    parser.add_argument("--env-label", default=None)

    # reproducibility
    parser.add_argument("--seed", type=int, default=None)

    # evaluation
    parser.add_argument("--max-episode-timesteps", type=int, default=1_000)
    parser.add_argument("--evaluation-num-epochs", type=int, default=10)
    parser.add_argument("--evaluation-num-episodes", type=int, default=100)
    parser.add_argument("--evaluation-discount", type=float, default=1.0)

    # latent observation
    parser.add_argument("--latent-type", default="state")

    # gv models
    parser.add_argument("--gv-observation-representation", default="compact")
    parser.add_argument("--gv-state-representation", default="compact")

    parser.add_argument("--logconfig", default="logconfig.yaml")

    args = parser.parse_args()
    args.env_label = args.env if args.env_label is None else args.env_label
    return args


class RunState(NamedTuple):
    env: Environment
    policy: Policy


def setup() -> RunState:
    config = get_config()

    table = str.maketrans({"-": "_"})
    latent_type = LatentType[config.latent_type.upper().translate(table)]
    env = make_env(
        config.env,
        latent_type=latent_type,
        max_episode_timesteps=config.max_episode_timesteps,
    )

    policy = make_hardcoded_policy(config.env, env)

    return RunState(env, policy)


def run(runstate: RunState):
    config = get_config()
    logger.info("run %s", config.env_label)

    env, policy = runstate

    # reproducibility
    if config.seed is not None:
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        reset_gv_rng(config.seed)
        env.seed(config.seed)

    # main evaluation loop
    for epoch in range(config.evaluation_num_epochs):
        evalstats = evaluate(
            env,
            policy,
            discount=config.evaluation_discount,
            num_episodes=config.evaluation_num_episodes,
        )

        logger.info(
            "EVALUATE epoch %d episodes %d return % .3f ± % .3f",
            epoch,
            config.evaluation_num_episodes,
            evalstats.returns.mean(),
            evalstats.returns.std(),
        )


def main():
    args = parse_args()
    config = get_config()
    config._update(vars(args))

    with open(config.logconfig, "r") as f:
        logconfig = yaml.safe_load(f.read())
    logging.config.dictConfig(logconfig)

    logger.info("setup of runstate...")
    runstate = setup()
    logger.info("setup DONE")

    logger.info("run...")
    run(runstate)
    logger.info("run DONE")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
