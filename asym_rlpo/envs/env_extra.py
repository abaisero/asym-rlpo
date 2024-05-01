from __future__ import annotations

from asym_rlpo.envs.carflag import make_extra_carflag
from asym_rlpo.envs.cleaner import make_extra_cleaner
from asym_rlpo.envs.dectiger import make_extra_dectiger
from asym_rlpo.envs.env import (
    LatentEnvironmentModule,
    StatefulEnvironment,
    StateLatentEnvironmentModule,
)


def make_extra_env(
    name: str,
    *,
    latent_type: str,
) -> tuple[StatefulEnvironment, LatentEnvironmentModule]:
    if name == 'extra-dectiger-v0' and latent_type == 'state':
        env = make_extra_dectiger()
        latent_env_module = StateLatentEnvironmentModule(env)
        return env, latent_env_module

    if name == 'extra-cleaner-v0' and latent_type == 'state':
        env = make_extra_cleaner()
        latent_env_module = StateLatentEnvironmentModule(env)
        return env, latent_env_module

    if name == 'extra-car-flag-v0' and latent_type == 'state':
        env = make_extra_carflag()
        latent_env_module = StateLatentEnvironmentModule(env)
        return env, latent_env_module

    raise ValueError(f'Invalid extra env name {name} and/or latent type {latent_type}')
