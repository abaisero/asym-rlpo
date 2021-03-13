import gym

from .adqn import ADQN
from .base import Algorithm
from .dqn import DQN


def make_algorithm(name: str, env: gym.Env) -> Algorithm:
    if name in {'DQN', 'dqn'}:
        return DQN(env)

    if name in {'ADQN', 'adqn'}:
        return ADQN(env)

    raise ValueError(f'invalid algorithm name {name}')
