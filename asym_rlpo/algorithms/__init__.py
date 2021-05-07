from typing import Union

import gym

from .a2c.asym_a2c import AsymA2C
from .a2c.base import A2C
from .a2c.sym_a2c import SymA2C
from .dqn.adqn import ADQN, ADQN_Bootstrap
from .dqn.base import DQN
from .dqn.dqn import DQN
from .dqn.fob_dqn import FOB_DQN
from .dqn.foe_dqn import FOE_DQN


def make_algorithm(name, env: gym.Env) -> Union[DQN, A2C]:
    if name == 'fob-dqn':
        return FOB_DQN(env)

    if name == 'foe-dqn':
        return FOE_DQN(env)

    if name == 'dqn':
        return DQN(env)

    if name == 'adqn':
        return ADQN(env)

    if name == 'adqn-bootstrap':
        return ADQN_Bootstrap(env)

    if name == 'sym-a2c':
        return SymA2C(env)

    if name == 'asym-a2c':
        return AsymA2C(env)

    raise ValueError(f'invalid algorithm name {name}')
