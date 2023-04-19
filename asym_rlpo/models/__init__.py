from asym_rlpo.envs import Environment
from asym_rlpo.envs.env import EnvironmentType
from asym_rlpo.models.envs.carflag import CarFlagModelFactory
from asym_rlpo.models.envs.cleaner import CleanerModelFactory
from asym_rlpo.models.envs.dectiger import DecTigerModelFactory
from asym_rlpo.models.envs.flat import FlatModelFactory
from asym_rlpo.models.envs.gv import GVModelFactory
from asym_rlpo.models.envs.openai import OpenAIModelFactory
from asym_rlpo.models.factory import ModelFactory


def make_model_factory(env: Environment) -> ModelFactory:
    if env.type is EnvironmentType.GV:
        return GVModelFactory(env)

    elif env.type is EnvironmentType.OPENAI:
        return OpenAIModelFactory(env)

    elif env.type is EnvironmentType.EXTRA_DECTIGER:
        return DecTigerModelFactory(env)

    elif env.type is EnvironmentType.EXTRA_CLEANER:
        return CleanerModelFactory(env)

    elif env.type is EnvironmentType.EXTRA_CARFLAG:
        return CarFlagModelFactory(env)

    elif env.type is EnvironmentType.FLAT:
        return FlatModelFactory(env)

    else:
        raise NotImplementedError
