from typing import Optional, Tuple

from .env import Action, Environment, EnvironmentType, Latent, Observation
from .env_gv import make_gv_env
from .env_gym import make_gym_env


def make_env(
    id_or_path: str,
    *,
    max_episode_timesteps: Optional[int] = None,
) -> Environment:

    try:
        env = make_gym_env(id_or_path)

    except ValueError:
        print(
            f'Environment with id {id_or_path} not found.'
            ' Trying as a GV YAML environment.'
        )
        env = make_gv_env(id_or_path)

    if max_episode_timesteps is not None:
        env = TimeLimitEnvironment(env, max_episode_timesteps)

    return env


class TimeLimitEnvironment:
    """terminates episodes after a given number of timesteps"""

    def __init__(self, env: Environment, max_timestep: int):
        self._env = env
        self.type = self._env.type
        self.latent_space = self._env.latent_space
        self.action_space = self._env.action_space
        self.observation_space = self._env.observation_space

        self._timestep: int
        self._max_timestep = max_timestep

    def seed(self, seed: Optional[int] = None) -> None:
        self._env.seed(seed)

    def reset(self) -> Tuple[Observation, Latent]:
        self._timestep = 0
        return self._env.reset()

    def step(self, action: Action) -> Tuple[Observation, Latent, float, bool]:
        self._timestep += 1
        observation, latent, reward, done = self._env.step(action)
        done = done or self._timestep >= self._max_timestep
        return observation, latent, reward, done

    def render(self) -> None:
        self._env.render()
