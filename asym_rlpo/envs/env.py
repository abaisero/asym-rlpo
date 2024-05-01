from typing import Protocol

import gym
import gym.spaces

from asym_rlpo.envs.types import Action, EnvironmentType, Latent, Observation, State


class StatefulEnvironment(Protocol):
    type: EnvironmentType

    state_space: gym.spaces.Space
    action_space: gym.spaces.Discrete
    observation_space: gym.spaces.Space

    def seed(self, seed: int | None = None) -> None:
        ...

    def reset(self) -> tuple[State, Observation]:
        ...

    def step(self, action: Action) -> tuple[State, Observation, float, bool]:
        ...

    def render(self) -> None:
        ...


class LatentEnvironmentModule(Protocol):
    latent_type: str
    latent_space: gym.spaces.Space

    def __call__(self, state: State) -> Latent:
        ...


class StateLatentEnvironmentModule(LatentEnvironmentModule):
    def __init__(self, env: StatefulEnvironment):
        self.latent_type = 'state'
        self.latent_space = env.state_space

    def __call__(self, state: State) -> Latent:
        return state


class Environment:
    def __init__(
        self,
        env: StatefulEnvironment,
        latent_env_module: LatentEnvironmentModule,
    ):
        super().__init__()
        self._env = env
        self._latent_env_module = latent_env_module

    @property
    def type(self) -> EnvironmentType:
        return self._env.type

    @property
    def latent_type(self) -> str:
        return self._latent_env_module.latent_type

    @property
    def state_space(self) -> gym.spaces.Space:
        return self._env.state_space

    @property
    def action_space(self) -> gym.spaces.Discrete:
        return self._env.action_space

    @property
    def observation_space(self) -> gym.spaces.Space:
        return self._env.observation_space

    @property
    def latent_space(self) -> gym.spaces.Space:
        return self._latent_env_module.latent_space

    def seed(self, seed: int | None = None) -> None:
        self._env.seed(seed)

    def reset(self) -> tuple[Observation, Latent]:
        state, observation = self._env.reset()
        latent = self._latent_env_module(state)
        return observation, latent

    def step(self, action: Action) -> tuple[Observation, Latent, float, bool]:
        state, observation, reward, done = self._env.step(action)
        latent = self._latent_env_module(state)
        return observation, latent, reward, done

    def render(self) -> None:
        self._env.render()


class TimedEnvironment(Environment):
    def __init__(
        self,
        env: StatefulEnvironment,
        latent_env_module: LatentEnvironmentModule,
        *,
        max_timestep: int,
    ):
        super().__init__(env, latent_env_module)
        self._timestep: int
        self._max_timestep = max_timestep

    def reset(self) -> tuple[Observation, Latent]:
        self._timestep = 0
        state, observation = self._env.reset()
        latent = self._latent_env_module(state)
        return observation, latent

    def step(self, action: Action) -> tuple[Observation, Latent, float, bool]:
        self._timestep += 1
        state, observation, reward, done = self._env.step(action)
        latent = self._latent_env_module(state)
        done = done or self._timestep >= self._max_timestep
        return observation, latent, reward, done
