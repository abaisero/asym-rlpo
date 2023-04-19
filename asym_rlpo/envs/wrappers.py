from collections.abc import Sequence

import gym
import gym.spaces
import numpy as np


class IndexWrapper(gym.ObservationWrapper):
    """IndexWrapper.

    Takes a gym.Env with a flat Box observation space, and filters such that
    only the dimensions indicated by `indices` are observable.
    """

    def __init__(self, env: gym.Env, indices: Sequence[int]):
        if (
            not isinstance(env.observation_space, gym.spaces.Box)
            or len(env.observation_space.shape) != 1
        ):
            raise ValueError('env.observation_space must be flat Box')

        if len(set(indices)) != len(indices):
            raise ValueError('indices must be unique')

        assert isinstance(env.observation_space, gym.spaces.Box)
        if len(indices) > env.observation_space.shape[0]:
            raise ValueError(
                'number of indices must not exceed state dimensions'
            )

        if min(indices) < 0:
            raise ValueError('indices must be non-negative')

        if max(indices) >= env.observation_space.shape[0]:
            raise ValueError('indices must be lower than state dimensions')

        super().__init__(env)

        self._indices = indices
        self.state_space = env.observation_space
        self.observation_space = gym.spaces.Box(
            env.observation_space.low[indices],
            env.observation_space.high[indices],
        )

        self.state: np.ndarray

    def observation(self, observation):
        self.state = observation
        return observation[self._indices]


class SingleAgentWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        assert isinstance(self.env.action_space, gym.spaces.Tuple)
        assert all(
            isinstance(action_space, gym.spaces.Discrete)
            for action_space in self.env.action_space.spaces
        )
        assert isinstance(self.env.observation_space, gym.spaces.Tuple)
        assert all(
            isinstance(observation_space, gym.spaces.Box)
            for observation_space in self.env.observation_space.spaces
        )

        self.num_actions = [
            action_space.n for action_space in self.env.action_space.spaces
        ]
        self.action_space = gym.spaces.Discrete(np.prod(self.num_actions))
        low = np.concatenate(
            [
                observation_space.low
                for observation_space in self.env.observation_space
            ]
        )
        high = np.concatenate(
            [
                observation_space.high
                for observation_space in self.env.observation_space
            ]
        )
        self.observation_space = gym.spaces.Box(low, high)

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        observation = observation.flatten()
        return observation

    def step(self, action):
        observation, *ret = self.env.step(self.action(action))
        observation = observation.flatten()
        return (observation, *ret)

    def action(self, action):
        return np.unravel_index(action, self.num_actions)
