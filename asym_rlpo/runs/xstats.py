from collections.abc import Sequence
from dataclasses import dataclass

from asym_rlpo.data import Episode


@dataclass
class XStats:
    epoch: int = 0
    simulation_episodes: int = 0
    simulation_timesteps: int = 0
    training_episodes: int = 0
    training_timesteps: int = 0
    optimizer_steps: int = 0


def update_xstats_epoch(xstats: XStats):
    xstats.epoch += 1


def update_xstats_simulation(xstats: XStats, episodes: Sequence[Episode]):
    xstats.simulation_episodes += len(episodes)
    xstats.simulation_timesteps += sum(len(episode) for episode in episodes)


def update_xstats_training(xstats: XStats, episodes: Sequence[Episode]):
    xstats.training_episodes += len(episodes)
    xstats.training_timesteps += sum(len(episode) for episode in episodes)


def update_xstats_optimizer(xstats: XStats):
    xstats.optimizer_steps += 1
