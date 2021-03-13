import gym

from .base import Policy


class RandomPolicy(Policy):
    def __init__(self, action_space: gym.spaces.Space):
        super().__init__()
        self.action_space = action_space

    def reset(self, observation):
        pass

    def step(self, action, observation):
        pass

    def sample_action(self):
        return self.action_space.sample()
