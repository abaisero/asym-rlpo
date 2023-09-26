from __future__ import annotations

import re
from typing import Protocol

import gym
import gym.spaces
import gym_pomdps

from asym_rlpo.envs.env import (
    Action,
    Environment,
    EnvironmentType,
    Latent,
    Observation,
    State,
)
from asym_rlpo.envs.wrappers import IndexWrapper


def make_gym_env(id: str, *, latent_type: str) -> Environment:
    """makes a stateful gym environment or converts a fully observable openai environment into a partially observable openai environment"""

    heavenhell_latent = (
        re.fullmatch(r'POMDP-heavenhell_\d+-episodic-v0', id)
        and latent_type == 'heaven'
    )

    if latent_type != 'state' and not heavenhell_latent:
        raise ValueError(f'Invalid latent type {latent_type} for gym env id {id}')

    try:
        return make_po_gym_env(id)

    except ValueError:
        print('Loading using gym.make')
        try:
            gym_env = gym.make(id)

        except gym.error.Error as e:
            raise ValueError from e

        else:
            if isinstance(gym_env.unwrapped, gym_pomdps.POMDP):
                if heavenhell_latent:
                    return HeavenHellGymEnvironment(gym_env, EnvironmentType.FLAT)

                return GymEnvironment(gym_env, EnvironmentType.FLAT)

            if re.fullmatch(r'extra-dectiger-v\d+', gym_env.spec.id):
                return GymEnvironment(gym_env, EnvironmentType.EXTRA_DECTIGER)

            if re.fullmatch(r'extra-cleaner-v\d+', gym_env.spec.id):
                return GymEnvironment(gym_env, EnvironmentType.EXTRA_CLEANER)

            if re.fullmatch(r'extra-car-flag-v\d+', gym_env.spec.id):
                return GymEnvironment(gym_env, EnvironmentType.EXTRA_CARFLAG)

            return GymEnvironment(gym_env, EnvironmentType.OTHER)


def make_po_gym_env(name: str) -> Environment:
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

    env: StatefulGymEnv

    if env_name == 'CartPole':
        indices_dict = {
            'pos': [0, 2],  # ignore velocities
            'vel': [1, 3],  # ignore positions
            'full': [0, 1, 2, 3],  # ignore nothing
        }

        if po_type not in indices_dict.keys():
            raise ValueError(f'invalid partial observability {po_type}')

        gym_env = gym.make(non_po_name)
        indices = indices_dict[po_type]
        env = IndexWrapper(gym_env, indices)

    elif env_name == 'LunarLander':
        indices_dict = {
            'pos': [0, 1, 4, 6, 7],  # ignore velocities
            'vel': [2, 3, 5, 6, 7],  # ignore positions
            'full': [0, 1, 2, 3, 4, 5, 6, 7],  # ignore nothing
        }

        if po_type not in indices_dict.keys():
            raise ValueError(f'invalid partial observability {po_type}')

        gym_env = gym.make(non_po_name)
        indices = indices_dict[po_type]
        env = IndexWrapper(gym_env, indices)

    elif env_name == 'Acrobot':
        indices_dict = {
            'pos': [0, 1, 2, 3],  # ignore velocities
            'vel': [4, 5],  # ignore positions
            'full': [0, 1, 2, 3, 4, 5],  # ignore nothing
        }

        if po_type not in indices_dict.keys():
            raise ValueError(f'invalid partial observability {po_type}')

        gym_env = gym.make(non_po_name)
        indices = indices_dict[po_type]

        env = IndexWrapper(gym_env, indices)

    else:
        raise ValueError('invalid env name {env_name}')

    return GymEnvironment(env, EnvironmentType.OPENAI)


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


class GymEnvironment(Environment):
    """Converts gym.Env to the Environment protocol"""

    def __init__(self, env: StatefulGymEnv, type: EnvironmentType):
        self._env = env
        self.type = type
        self.latent_type = 'state'
        self.action_space: gym.spaces.Discrete = env.action_space
        self.observation_space: gym.spaces.Space = env.observation_space
        self.state_space: gym.spaces.Space = env.state_space
        self.latent_space: gym.spaces.Space = self._make_latent_space()

    def _make_latent_space(self) -> gym.spaces.Space:
        return self.state_space

    def _map_latent(self, state: State) -> Latent:
        return state

    def seed(self, seed: int | None = None) -> None:
        self._env.seed(seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
        self.latent_space.seed(seed)

    def reset(self) -> tuple[Observation, Latent]:
        observation = self._env.reset()
        latent = self._map_latent(self._env.state)
        return observation, latent

    def step(self, action: Action) -> tuple[Observation, Latent, float, bool]:
        observation, reward, done, _ = self._env.step(action)
        latent = self._map_latent(self._env.state)
        return observation, latent, reward, done

    def render(self) -> None:
        self._env.render()


class HeavenHellGymEnvironment(GymEnvironment):
    def __init__(self, env: StatefulGymEnv, type: EnvironmentType):
        super().__init__(env, type)
        self.latent_type = 'heaven'

    def _make_latent_space(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(2)

    def _map_latent(self, state: State) -> Latent:
        if state == -1:
            return -1

        assert isinstance(self.state_space, gym.spaces.Discrete)
        heaven_right = state >= self.state_space.n // 2
        latent = int(heaven_right)
        print(f'state {state} latent {latent}')
        return latent
