import gym
import numpy as np

from asym_rlpo.policies import Policy
from asym_rlpo.sampling import sample_episode
from asym_rlpo.utils.returns import returns


def evaluate(
    env: gym.Env,
    policy: Policy,
    *,
    discount: float,
    num_episodes: int,
    num_steps: int
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
        sample_episode(env, policy, num_steps=num_steps).rewards
        for _ in range(num_episodes)
    ]
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
