from asym_rlpo.envs.env import Environment, TimedEnvironment
from asym_rlpo.envs.env_extra import make_extra_env
from asym_rlpo.envs.env_gv import make_gv_env
from asym_rlpo.envs.env_gym import make_gym_env


def make_env(
    id_or_path: str,
    *,
    latent_type: str,
    max_episode_timesteps: int | None = None,
    gv_representation: str = 'compact',
) -> Environment:
    stateful_env, latent_env_module = make_stateful_env(
        id_or_path,
        latent_type=latent_type,
        gv_representation=gv_representation,
    )

    return (
        Environment(stateful_env, latent_env_module)
        if max_episode_timesteps is None
        else TimedEnvironment(
            stateful_env,
            latent_env_module,
            max_timestep=max_episode_timesteps,
        )
    )


def make_stateful_env(
    id_or_path: str,
    *,
    latent_type: str,
    gv_representation: str = 'compact',
):
    print(f'Making stateful env {id_or_path}.')

    try:
        print('Trying as an extra environment.')
        return make_extra_env(id_or_path, latent_type=latent_type)
    except ValueError:
        print('Extra env loading failed.')

    try:
        print('Trying as a gym environment.')
        return make_gym_env(id_or_path, latent_type=latent_type)

    except ValueError:
        print('Gym env loading failed.')

    print('Trying as a GV YAML environment.')
    return make_gv_env(
        id_or_path,
        latent_type=latent_type,
        gv_representation=gv_representation,
    )
