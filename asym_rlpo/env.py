import re

import gym
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
from asym_rlpo.wrapper import IndexWrapper


def make_env(id_or_path: str) -> gym.Env:
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
    return env


def make_po_env(name: str) -> gym.Env:
    m = re.match(r'PO-(?P<po_type>\w+)-(?P<fo_name>(?P<name>\w+)-v\d+)', name)
    # m[0] is the full name
    # m[1] is the first capture, i.e., the type of partial observability
    # m[2] is the second capture, i.e., the name w/ the version
    # m[3] is the third capture, i.e., the name w/o the version

    if m and m['name'] == 'CartPole':
        indices_dict = {
            'pos': [0, 2],  # ignore velocities
            'vel': [1, 3],  # ignore positions
            'full': [0, 1, 2, 3],  # ignore nothing
        }

        checkraise(
            m['po_type'] in indices_dict.keys(),
            ValueError,
            f'invalid partial observability {m["po_type"]}',
        )

        env = gym.make(m['fo_name'])
        indices = indices_dict[m['po_type']]
        return IndexWrapper(env, indices)

    raise ValueError('invalid env name {name}')


def make_gv_env(path: str) -> GymEnvironment:
    print('Loading using YAML')
    inner_env = factory_env_from_yaml(path)
    obs_rep = DefaultObservationRepresentation(inner_env.observation_space)
    state_rep = DefaultStateRepresentation(inner_env.state_space)
    outer_env = OuterEnv(
        inner_env, observation_rep=obs_rep, state_rep=state_rep
    )
    return GymEnvironment.from_environment(outer_env)
