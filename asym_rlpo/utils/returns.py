from collections import defaultdict
from typing import Dict

import numpy as np

from asym_rlpo.utils.debugging import checkraise


def discounts_uncached(num_steps: int, discount: float) -> np.ndarray:
    """Return the discounts array $[1., \\gamma, \\gamma^2, \\ldots]$.

    :param num_steps:  size of the output array
    :param discount:  discount factor
    :rtype: (N,) np.ndarray of discounts
    """
    checkraise(num_steps > 0, ValueError, 'invalid `num_steps` {}', num_steps)
    checkraise(
        0.0 <= discount <= 1.0, ValueError, 'invalid `discount` {}', discount
    )

    return discount ** np.arange(num_steps, dtype=float)


discounts_cache: Dict[float, np.ndarray] = defaultdict(lambda: np.array([1.0]))


def discounts(num_steps: int, discount: float) -> np.ndarray:
    """Return the discounts array $[1., \\gamma, \\gamma^2, \\ldots]$.

    :param num_steps:  size of the output array
    :param discount:  discount factor
    :rtype: (N,) np.ndarray of discounts
    """
    checkraise(num_steps > 0, ValueError, 'invalid `num_steps` {}', num_steps)
    checkraise(
        0.0 <= discount <= 1.0, ValueError, 'invalid `discount` {}', discount
    )

    cached_discounts = discounts_cache[discount]

    if cached_discounts.size >= num_steps:
        discounts_ = cached_discounts[:num_steps]

    if cached_discounts.size < num_steps:
        discounts_ = discounts_uncached(num_steps, discount)
        discounts_cache[discount] = discounts_
        return discounts_

    return discounts_


def returns(rewards: np.ndarray, discount: float) -> np.ndarray:
    """Return the empirical episodic returns from rewards.

    :param rewards:  (B, T) np.ndarray of rewards
    :param discount:  discount factor
    :rtype: (B,) np.ndarray of empirical returns
    """
    checkraise(
        rewards.ndim > 1, ValueError, 'invalid rewards.ndim {}', rewards.ndim
    )

    num_steps = rewards.shape[-1]
    return np.einsum('j,...j->...', discounts(num_steps, discount), rewards)
