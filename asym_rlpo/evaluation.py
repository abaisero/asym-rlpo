import gym
import numpy as np

from asym_rlpo.sampling import sample_rewards
from asym_rlpo.utils.returns import returns


def evaluate(
    env: gym.Env, *, discount: float, num_episodes: int, num_steps: int
) -> np.ndarray:
    """Return a few empirical returns

    Args:
        env (gym.Env): env
        discount (float): discount
        num_episodes (int): number of independent sample episode
        num_steps (int): max number of time-steps

    Returns:
        np.ndarray:
    """
    rewards = [
        sample_rewards(env, num_steps=num_steps) for _ in range(num_episodes)
    ]
    max_num_steps = max(len(rs) for rs in rewards)
    rewards = [np.pad(rs, (0, max_num_steps - len(rs))) for rs in rewards]
    rewards = np.vstack(rewards)
    return returns(rewards, discount)
