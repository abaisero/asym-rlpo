import re
from typing import Optional

import gym
import gym_pomdps
from gym_gridverse.debugging import reset_gv_debug
from gym_gridverse.envs.yaml.factory import factory_env_from_yaml
from gym_gridverse.gym import GymEnvironment as GVGymEnvironment
from gym_gridverse.outer_env import OuterEnv
from gym_gridverse.representations.observation_representations import (
    make_observation_representation,
)
from gym_gridverse.representations.state_representations import (
    make_state_representation,
)

from asym_rlpo.utils.config import get_config
from asym_rlpo.utils.debugging import checkraise
from asym_rlpo.wrapper import FlatPaddingWrapper, IndexWrapper

from . import extra_hai, extra_lyu
from .env import (
    Environment,
    EnvironmentType,
    GymEnvironment,
    TimeLimitEnvironment,
)


def make_env(
    id_or_path: str,
    *,
    max_episode_timesteps: Optional[int] = None,
) -> Environment:

    try:
        env = make_po_env(id_or_path)

    except ValueError:

        try:
            print('Loading using gym.make')
            gym_env = gym.make(id_or_path)

        except gym.error.Error:
            print(
                f'Environment with id {id_or_path} not found.'
                'Trying as a GV YAML environment'
            )
            env = make_gv_env(id_or_path)

        else:

            if isinstance(gym_env.unwrapped, gym_pomdps.POMDP):
                gym_env = FlatPaddingWrapper(gym_env)
                env = GymEnvironment(gym_env, EnvironmentType.FLAT)

            elif re.fullmatch(r'extra-dectiger-v\d+', gym_env.spec.id):
                env = GymEnvironment(gym_env, EnvironmentType.EXTRA_DECTIGER)

            elif re.fullmatch(r'extra-cleaner-v\d+', gym_env.spec.id):
                env = GymEnvironment(gym_env, EnvironmentType.EXTRA_CLEANER)

            elif re.fullmatch(r'extra-car-flag-v\d+', gym_env.spec.id):
                env = GymEnvironment(gym_env, EnvironmentType.EXTRA_CARFLAG)

            else:
                env = GymEnvironment(gym_env, EnvironmentType.OTHER)

    if max_episode_timesteps is not None:
        env = TimeLimitEnvironment(env, max_episode_timesteps)

    return env


def make_po_env(name: str) -> Environment:
    """convert a fully observable openai environment into a partially observable openai environment"""

    pattern = r'^PO-([\w:.-]+)-([\w:.-]+)-v(\d+)$'
    m = re.match(pattern, name)
    # m[0] is the full name
    # m[1] is the first capture, i.e., the type of partial observability
    # m[2] is the second capture, i.e., the name w/o the version
    # m[3] is the third capture, i.e., the version

    checkraise(
        m is not None,
        ValueError,
        f'env name {name} does not satisfy regex',
    )

    assert m is not None  # silly forcing of type checking
    po_type = m[1]
    env_name = m[2]
    version = m[3]
    non_po_name = f'{env_name}-v{version}'

    if env_name == 'CartPole':
        indices_dict = {
            'pos': [0, 2],  # ignore velocities
            'vel': [1, 3],  # ignore positions
            'full': [0, 1, 2, 3],  # ignore nothing
        }

        checkraise(
            po_type in indices_dict.keys(),
            ValueError,
            f'invalid partial observability {po_type}',
        )

        env = gym.make(non_po_name)
        indices = indices_dict[po_type]
        env = IndexWrapper(env, indices)

    elif env_name == 'LunarLander':
        indices_dict = {
            'pos': [0, 1, 4, 6, 7],  # ignore velocities
            'vel': [2, 3, 5, 6, 7],  # ignore positions
            'full': [0, 1, 2, 3, 4, 5, 6, 7],  # ignore nothing
        }

        checkraise(
            po_type in indices_dict.keys(),
            ValueError,
            f'invalid partial observability {po_type}',
        )

        env = gym.make(non_po_name)
        indices = indices_dict[po_type]
        env = IndexWrapper(env, indices)

    elif env_name == 'Acrobot':
        indices_dict = {
            'pos': [0, 1, 2, 3],  # ignore velocities
            'vel': [4, 5],  # ignore positions
            'full': [0, 1, 2, 3, 4, 5],  # ignore nothing
        }

        checkraise(
            po_type in indices_dict.keys(),
            ValueError,
            f'invalid partial observability {po_type}',
        )

        env = gym.make(non_po_name)
        indices = indices_dict[po_type]
        env = IndexWrapper(env, indices)

    else:
        raise ValueError('invalid env name {env_name}')

    return GymEnvironment(env, EnvironmentType.OPENAI)


def make_gv_env(path: str) -> Environment:
    reset_gv_debug(False)

    config = get_config()

    print('Loading using YAML')
    inner_env = factory_env_from_yaml(path)
    observation_representation = make_observation_representation(
        config.gv_observation_representation,
        inner_env.observation_space,
    )
    state_representation = make_state_representation(
        config.gv_state_representation,
        inner_env.state_space,
    )
    outer_env = OuterEnv(
        inner_env,
        observation_representation=observation_representation,
        state_representation=state_representation,
    )
    gym_env = GVGymEnvironment(outer_env)
    return GymEnvironment(gym_env, EnvironmentType.GV)
