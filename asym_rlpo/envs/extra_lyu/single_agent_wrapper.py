import gym
import numpy as np


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
