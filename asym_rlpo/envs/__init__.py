from asym_rlpo.envs.carflag import register as register_carflag
from asym_rlpo.envs.cleaner import register as register_cleaner
from asym_rlpo.envs.dectiger import register as register_dectiger
from asym_rlpo.envs.env import (
    Action,
    Environment,
    Latent,
    LatentType,
    Observation,
)
from asym_rlpo.envs.env_gv import make_gv_env, env_is_gv
from asym_rlpo.envs.env_gym import make_gym_env
from asym_rlpo.envs.env_gym_hh import id_is_heavenhell

register_carflag()
register_cleaner()
register_dectiger()


def valid_env_id_and_latent_type(id_or_path: str, *, latent_type: LatentType) -> bool:
    if latent_type is LatentType.STATE:
        return True

    if id_is_heavenhell(id_or_path) and latent_type in [
        LatentType.HH_HEAVEN,
        LatentType.HH_POSITION,
    ]:
        return True

    if env_is_gv(id_or_path) and latent_type is LatentType.GV_MEMORY:
        return True

    return False


def make_env(
    id_or_path: str,
    *,
    latent_type: LatentType,
    max_episode_timesteps: int | None = None,
    gv_representation: str = 'compact',
) -> Environment:
    if not valid_env_id_and_latent_type(id_or_path, latent_type=latent_type):
        raise ValueError(f'Invalid {id_or_path=} {latent_type=} for gym envs')

    try:
        env = make_gym_env(id_or_path, latent_type=latent_type)

    except ValueError:
        print(
            f'Environment with id {id_or_path} not found.'
            ' Trying as a GV YAML environment.'
        )
        env = make_gv_env(
            id_or_path,
            latent_type=latent_type,
            gv_representation=gv_representation,
        )

    if max_episode_timesteps is not None:
        env = TimeLimitEnvironment(env, max_episode_timesteps)

    return env


class TimeLimitEnvironment:
    """terminates episodes after a given number of timesteps"""

    def __init__(self, env: Environment, max_timestep: int):
        self._env = env
        self.type = self._env.type
        self.latent_type = self._env.latent_type
        self.latent_space = self._env.latent_space
        self.action_space = self._env.action_space
        self.observation_space = self._env.observation_space

        self._timestep: int
        self._max_timestep = max_timestep

    def seed(self, seed: int | None = None) -> None:
        self._env.seed(seed)

    def reset(self) -> tuple[Observation, Latent]:
        self._timestep = 0
        return self._env.reset()

    def step(self, action: Action) -> tuple[Observation, Latent, float, bool]:
        self._timestep += 1
        observation, latent, reward, done = self._env.step(action)
        done = done or self._timestep >= self._max_timestep
        return observation, latent, reward, done

    def render(self) -> None:
        self._env.render()
