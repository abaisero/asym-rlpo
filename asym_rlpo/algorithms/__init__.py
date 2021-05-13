from typing import Union

import gym

from .a2c.a2c import A2C
from .a2c.asym_a2c import AsymA2C
from .a2c.asym_a2c_state import AsymA2C_State
from .a2c.base import A2C_Base
from .dqn.adqn import ADQN, ADQN_Bootstrap
from .dqn.adqn_state import ADQN_State, ADQN_State_Bootstrap
from .dqn.base import DQN_Base
from .dqn.dqn import DQN
from .dqn.fob_dqn import FOB_DQN
from .dqn.foe_dqn import FOE_DQN


def make_a2c_algorithm(name, env: gym.Env) -> A2C_Base:
    if name == 'a2c':
        return A2C(env)

    if name == 'asym-a2c':
        return AsymA2C(env)

    if name == 'asym-a2c-state':
        return AsymA2C_State(env)

    raise ValueError(f'invalid algorithm name {name}')


def make_dqn_algorithm(name, env: gym.Env) -> DQN_Base:
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

    if name == 'adqn-state':
        return ADQN_State(env)

    if name == 'adqn-state-bootstrap':
        return ADQN_State_Bootstrap(env)

    raise ValueError(f'invalid algorithm name {name}')
