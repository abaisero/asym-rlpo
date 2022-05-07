from typing import List

import gym
import gym.spaces
import numpy as np

from asym_rlpo.utils.debugging import checkraise


class IndexWrapper(gym.ObservationWrapper):
    """IndexWrapper.

    Takes a gym.Env with a flat Box observation space, and filters such that
    only the dimensions indicated by `indices` are observable.
    """

    def __init__(self, env: gym.Env, indices: List[int]):
        checkraise(
            isinstance(env.observation_space, gym.spaces.Box)
            and len(env.observation_space.shape) == 1,
            ValueError,
            'env.observation_space must be flat Box',
        )

        checkraise(
            len(set(indices)) == len(indices),
            ValueError,
            'indices must be unique',
        )

        assert isinstance(env.observation_space, gym.spaces.Box)
        checkraise(
            len(indices) <= env.observation_space.shape[0],
            ValueError,
            'number of indices must not exceed state dimensions',
        )

        checkraise(
            min(indices) >= 0,
            ValueError,
            'indices must be non-negative',
        )

        checkraise(
            max(indices) < env.observation_space.shape[0],
            ValueError,
            'indices must be lower than state dimensions',
        )

        super().__init__(env)

        self.indices = indices
        self.state_space = env.observation_space
        self.observation_space = gym.spaces.Box(
            env.observation_space.low[self.indices],
            env.observation_space.high[self.indices],
        )

        self.state: np.ndarray

    def observation(self, observation):
        self.state = observation
        return observation[self.indices]


class FlatPaddingWrapper(gym.ObservationWrapper):
    """FlatPaddingWrapper.

    Takes a gym_pomdps.POMDP and extends the observation space by introducing a
    novel integer observation to be received upon reset.
    """

    def __init__(self, env: gym.Env):
        checkraise(
            isinstance(env.observation_space, gym.spaces.Discrete),
            ValueError,
            'env.observation_space must be Discrete',
        )

        super().__init__(env)

        assert isinstance(env.observation_space, gym.spaces.Discrete)
        self.observation_space = gym.spaces.Discrete(
            env.observation_space.n + 1
        )

    def observation(self, observation):
        return (
            observation
            if observation is not None
            else self.observation_space.n - 1
        )
