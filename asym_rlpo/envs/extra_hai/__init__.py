from gym.envs.registration import register

from .car_flag import CarEnv, CarEnvWrapper

register(
    id='extra-car-flag-v0',
    entry_point=lambda: CarEnvWrapper(CarEnv(), num_actions=7),
)
