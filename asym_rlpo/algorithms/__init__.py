import gym

from .a2c.a2c import A2C
from .a2c.asym_a2c import AsymA2C
from .a2c.asym_a2c_state import AsymA2C_State
from .a2c.base import PO_A2C_ABC
from .dqn.adqn import ADQN, ADQN_Bootstrap
from .dqn.adqn_short import ADQN_Short
from .dqn.adqn_state import ADQN_State, ADQN_State_Bootstrap
from .dqn.base import DQN_ABC
from .dqn.dqn import DQN
from .dqn.fob_dqn import FOB_DQN
from .dqn.foe_dqn import FOE_DQN


def make_a2c_algorithm(
    name, env: gym.Env, *, truncated_histories: bool, truncated_histories_n: int
) -> PO_A2C_ABC:
    if name == 'a2c':
        return A2C(
            env,
            truncated_histories=truncated_histories,
            truncated_histories_n=truncated_histories_n,
        )

    if name == 'asym-a2c':
        return AsymA2C(
            env,
            truncated_histories=truncated_histories,
            truncated_histories_n=truncated_histories_n,
        )

    if name == 'asym-a2c-state':
        return AsymA2C_State(
            env,
            truncated_histories=truncated_histories,
            truncated_histories_n=truncated_histories_n,
        )

    raise ValueError(f'invalid algorithm name {name}')


def make_dqn_algorithm(
    name, env: gym.Env, *, truncated_histories: bool, truncated_histories_n: int
) -> DQN_ABC:
    if name == 'fob-dqn':
        return FOB_DQN(env)

    if name == 'foe-dqn':
        return FOE_DQN(env)

    if name == 'dqn':
        return DQN(
            env,
            truncated_histories=truncated_histories,
            truncated_histories_n=truncated_histories_n,
        )

    if name == 'adqn':
        return ADQN(
            env,
            truncated_histories=truncated_histories,
            truncated_histories_n=truncated_histories_n,
        )

    if name == 'adqn-bootstrap':
        return ADQN_Bootstrap(
            env,
            truncated_histories=truncated_histories,
            truncated_histories_n=truncated_histories_n,
        )

    if name == 'adqn-state':
        return ADQN_State(
            env,
            truncated_histories=truncated_histories,
            truncated_histories_n=truncated_histories_n,
        )

    if name == 'adqn-state-bootstrap':
        return ADQN_State_Bootstrap(
            env,
            truncated_histories=truncated_histories,
            truncated_histories_n=truncated_histories_n,
        )

    if name == 'adqn-short':
        return ADQN_Short(
            env,
            truncated_histories=truncated_histories,
            truncated_histories_n=truncated_histories_n,
        )

    raise ValueError(f'invalid algorithm name {name}')
