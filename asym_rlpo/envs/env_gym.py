from __future__ import annotations

import re
from typing import Protocol

import gym
import gym.spaces
import gym_pomdps

from asym_rlpo.envs.env import (
    Action,
    EnvironmentType,
    Latent,
    LatentEnvironmentModule,
    Observation,
    State,
    StatefulEnvironment,
    StateLatentEnvironmentModule,
)
from asym_rlpo.envs.wrappers import IndexWrapper


def make_gym_env(
    id: str,
    *,
    latent_type: str,
) -> tuple[StatefulEnvironment, LatentEnvironmentModule]:
    """makes a stateful gym environment or converts a fully observable openai environment into a partially observable openai environment"""

    try:
        gym_env = make_po_gym_env(id)

    except ValueError:
        pass
    else:
        stateful_env = GymStatefulEnvironment(gym_env, EnvironmentType.OPENAI)
        latent_env_module = make_gym_latent_env_module(stateful_env, latent_type)
        return stateful_env, latent_env_module

    print('Loading using gym.make')
    try:
        gym_env = gym.make(id)

    except gym.error.Error as e:
        raise ValueError from e

    else:
        if isinstance(gym_env.unwrapped, gym_pomdps.POMDP):
            stateful_env = GymStatefulEnvironment(gym_env, EnvironmentType.FLAT)
            latent_env_module = make_gym_latent_env_module(
                stateful_env, latent_type
            )
            return stateful_env, latent_env_module

        if re.fullmatch(r'extra-dectiger-v\d+', gym_env.spec.id):
            stateful_env = GymStatefulEnvironment(
                gym_env, EnvironmentType.EXTRA_DECTIGER
            )
            latent_env_module = make_gym_latent_env_module(
                stateful_env, latent_type
            )
            return stateful_env, latent_env_module

        if re.fullmatch(r'extra-cleaner-v\d+', gym_env.spec.id):
            stateful_env = GymStatefulEnvironment(
                gym_env, EnvironmentType.EXTRA_CLEANER
            )
            latent_env_module = make_gym_latent_env_module(
                stateful_env, latent_type
            )
            return stateful_env, latent_env_module

        if re.fullmatch(r'extra-car-flag-v\d+', gym_env.spec.id):
            stateful_env = GymStatefulEnvironment(
                gym_env, EnvironmentType.EXTRA_CARFLAG
            )
            latent_env_module = make_gym_latent_env_module(
                stateful_env, latent_type
            )
            return stateful_env, latent_env_module

        stateful_env = GymStatefulEnvironment(gym_env, EnvironmentType.OTHER)
        latent_env_module = make_gym_latent_env_module(stateful_env, latent_type)
        return stateful_env, latent_env_module



def make_gym_latent_env_module(
    env: GymStatefulEnvironment,
    latent_type: str,
) -> LatentEnvironmentModule:
    if latent_type == 'state':
        return StateLatentEnvironmentModule(env)

    if latent_type == 'hh-heaven':
        return HeavenLatentEnvironmentModule(env)

    raise ValueError(f'invalid latent type {latent_type}')


def make_po_gym_env(name: str) -> StatefulGymEnv:
    """convert a fully observable openai environment into a partially observable openai environment"""

    pattern = r'^PO-([\w:.-]+)-([\w:.-]+)-v(\d+)$'
    m = re.match(pattern, name)
    # m[0] is the full name
    # m[1] is the first capture, i.e., the type of partial observability
    # m[2] is the second capture, i.e., the name w/o the version
    # m[3] is the third capture, i.e., the version

    if m is None:
        raise ValueError(f'env name {name} does not satisfy regex')

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
    elif env_name == 'LunarLander':
        indices_dict = {
            'pos': [0, 1, 4, 6, 7],  # ignore velocities
            'vel': [2, 3, 5, 6, 7],  # ignore positions
            'full': [0, 1, 2, 3, 4, 5, 6, 7],  # ignore nothing
        }
    elif env_name == 'Acrobot':
        indices_dict = {
            'pos': [0, 1, 2, 3],  # ignore velocities
            'vel': [4, 5],  # ignore positions
            'full': [0, 1, 2, 3, 4, 5],  # ignore nothing
        }
    else:
        raise ValueError('invalid env name {env_name}')

    gym_env = gym.make(non_po_name)

    try:
        indices = indices_dict[po_type]
    except KeyError as e:
        raise ValueError(f'invalid partial observability {po_type}') from e

    return IndexWrapper(gym_env, indices)


class StatefulGymEnv(Protocol):
    """Protocol based on gym.Env which also contains state_space and state"""

    state_space: gym.spaces.Space
    action_space: gym.spaces.Discrete
    observation_space: gym.spaces.Space

    state: State

    def seed(self, seed=None):
        ...

    def reset(self) -> Observation:
        ...

    def step(self, action) -> tuple[Observation, float, bool, dict]:
        ...

    def render(self, mode='human'):
        ...


class GymStatefulEnvironment(StatefulEnvironment):
    """Converts gym.Env to the Environment protocol"""

    state_space: gym.spaces.Space
    action_space: gym.spaces.Discrete
    observation_space: gym.spaces.Space

    def __init__(self, env: StatefulGymEnv, type: EnvironmentType):
        self._env = env
        self.type = type
        self.state_space = env.state_space
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    def seed(self, seed: int | None = None) -> None:
        self._env.seed(seed)
        self.state_space.seed(seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)

    def reset(self) -> tuple[State, Observation]:
        observation = self._env.reset()
        state = self._env.state
        return state, observation

    def step(self, action: Action) -> tuple[Observation, Latent, float, bool]:
        observation, reward, done, _ = self._env.step(action)
        state = self._env.state
        return state, observation, reward, done

    def render(self) -> None:
        self._env.render()


class HeavenLatentEnvironmentModule(LatentEnvironmentModule):
    def __init__(self, env: GymStatefulEnvironment):
        self._env = env
        self.latent_type = 'hh-heaven'
        self.latent_space = gym.spaces.Discrete(2)

        assert isinstance(env.state_space, gym.spaces.Discrete)
        self.heaven_right_threshold = env.state_space.n // 2

    def __call__(self, state: State) -> Latent:
        if state == -1:
            return -1

        heaven_right = state >= self.heaven_right_threshold
        latent = int(heaven_right)
        print(f'state {state} latent {latent}')
        return latent
