from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from asym_rlpo.data import Episode
from asym_rlpo.utils.returns import returns


@dataclass
class EvalStats:
    lengths: np.ndarray
    returns: np.ndarray


def evaluate_episodes(
    episodes: Sequence[Episode],
    *,
    discount: float,
) -> EvalStats:
    """Return evaluation stats"""
    lengths = np.array([len(episode) for episode in episodes])
    returns = evaluate_returns(episodes, discount=discount)
    return EvalStats(lengths, returns)


def evaluate_returns(
    episodes: Sequence[Episode],
    *,
    discount: float,
) -> np.ndarray:
    rewards = [episode.rewards for episode in episodes]
    tot_num_steps = max(len(rs) for rs in rewards)
    rewards_array = np.vstack(
        [
            np.pad(
                r_array, (0, tot_num_steps - r_array.size)
            )  # pad zeros to the right
            for r_array in rewards
        ]
    )
    return returns(rewards_array, discount)
