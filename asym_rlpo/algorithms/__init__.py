from .adqn import ADQN
from .base import Algorithm
from .dqn import DQN


def make_algorithm(name: str) -> Algorithm:
    if name in {'DQN', 'dqn'}:
        return DQN()

    if name in {'ADQN', 'adqn'}:
        return ADQN()

    raise ValueError(f'invalid algorithm name {name}')
