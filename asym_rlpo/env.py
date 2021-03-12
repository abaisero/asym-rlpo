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


# TODO for now this only generates GV environments
def make_env(id_or_path: str) -> GymEnvironment:
    try:
        print('Loading using gym.make')
        env = gym.make(id_or_path)

    except gym.error.Error:
        print(f'Environment with id {id_or_path} not found.')
        print('Loading using YAML')
        inner_env = factory_env_from_yaml(id_or_path)
        obs_rep = DefaultObservationRepresentation(inner_env.observation_space)
        state_rep = DefaultStateRepresentation(inner_env.state_space)
        outer_env = OuterEnv(
            inner_env, observation_rep=obs_rep, state_rep=state_rep
        )
        env = GymEnvironment.from_environment(outer_env)

    else:
        checkraise(
            isinstance(env, GymEnvironment),
            ValueError,
            'gym id {} is not associated with a GridVerse environment',
            id_or_path,
        )

    return env
