import re
from typing import Optional

import gym
import gym_pomdps
from gym.wrappers import TimeLimit
from gym_gridverse.debugging import reset_gv_debug
from gym_gridverse.envs.yaml.factory import factory_env_from_yaml
from gym_gridverse.gym import GymEnvironment
from gym_gridverse.outer_env import OuterEnv
from gym_gridverse.representations.observation_representations import (
    DefaultObservationRepresentation,
)
from gym_gridverse.representations.state_representations import (
    DefaultStateRepresentation,
)

from asym_rlpo.utils.debugging import checkraise
from asym_rlpo.wrapper import FlatPaddingWrapper, IndexWrapper

from . import extra_hai, extra_lyu


def make_env(
    id_or_path: str, *, max_episode_timesteps: Optional[int] = None
) -> gym.Env:
    try:
        env = make_po_env(id_or_path)

    except ValueError:

        try:
            print('Loading using gym.make')
            env = gym.make(id_or_path)

        except gym.error.Error:
            print(
                f'Environment with id {id_or_path} not found.'
                'Trying as a GV YAML environment'
            )
            env = make_gv_env(id_or_path)

    checkraise(
        hasattr(env, 'state_space'),
        ValueError,
        f'env {id_or_path} does not have state_space',
    )

    if isinstance(env.unwrapped, gym_pomdps.POMDP):
        env = FlatPaddingWrapper(env)

    if max_episode_timesteps is not None:
        env = TimeLimit(env, max_episode_timesteps)

    return env


po_env_id_re = re.compile(r'^PO-([\w:.-]+)-([\w:.-]+)-v(\d+)$')


def make_po_env(name: str) -> gym.Env:
    m = po_env_id_re.match(name)
    # m[0] is the full name
    # m[1] is the first capture, i.e., the type of partial observability
    # m[2] is the second capture, i.e., the name w/o the version
    # m[3] is the third capture, i.e., the version

    checkraise(
        m is not None, ValueError, f'env name {name} does not satisfy regex'
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
        return IndexWrapper(env, indices)

    if env_name == 'LunarLander':
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
        return IndexWrapper(env, indices)

    if env_name == 'Acrobot':
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
        return IndexWrapper(env, indices)

    raise ValueError('invalid env name {env_name}')


def make_gv_env(path: str) -> GymEnvironment:
    reset_gv_debug(False)

    print('Loading using YAML')
    inner_env = factory_env_from_yaml(path)
    observation_representation = DefaultObservationRepresentation(
        inner_env.observation_space
    )
    state_representation = DefaultStateRepresentation(inner_env.state_space)
    outer_env = OuterEnv(
        inner_env,
        observation_representation=observation_representation,
        state_representation=state_representation,
    )
    return GymEnvironment.from_environment(outer_env)
