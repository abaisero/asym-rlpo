import gym

from .base import Algorithm
from .dqn import DQN


def make_algorithm(name: str, env: gym.Env) -> Algorithm:
    if name in {'DQN', 'dqn'}:
        return DQN(env)

    raise ValueError(f'invalid algorithm name {name}')
