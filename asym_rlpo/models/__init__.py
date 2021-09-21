import re
from typing import Iterable, Optional

import gym
import gym_gridverse as gv
import gym_pomdps
import torch.nn as nn

from asym_rlpo.models.models_flat import make_models as make_models_flat
from asym_rlpo.models.models_gv import make_models as make_models_gv
from asym_rlpo.models.models_openai import make_models as make_models_openai
from asym_rlpo.utils.debugging import checkraise


def make_models(
    env: gym.Env, *, keys: Optional[Iterable[str]] = None
) -> nn.ModuleDict:

    if isinstance(env.unwrapped, gv.gym.GymEnvironment):
        models = make_models_gv(env)

    elif (
        re.fullmatch(r'CartPole-v\d+', env.spec.id)
        or re.fullmatch(r'Acrobot-v\d+', env.spec.id)
        or re.fullmatch(r'LunarLander-v\d+', env.spec.id)
    ):
        models = make_models_openai(env)

    elif isinstance(env.unwrapped, gym_pomdps.POMDP):
        models = make_models_flat(env)

    else:
        raise NotImplementedError

    return models if keys is None else filter_models(models, keys)


def filter_models(models: nn.ModuleDict, keys) -> nn.ModuleDict:
    if isinstance(keys, list):
        missing_keys = set(keys) - models.keys()
        checkraise(
            len(missing_keys) == 0,
            ValueError,
            'models dictionary does not contains keys {}',
            missing_keys,
        )
        return nn.ModuleDict({k: models[k] for k in keys})

    if isinstance(keys, dict):
        missing_keys = set(keys.keys()) - models.keys()
        checkraise(
            len(missing_keys) == 0,
            ValueError,
            'models dictionary does not contains keys {}',
            missing_keys,
        )
        return nn.ModuleDict(
            {k: filter_models(models[k], v) for k, v in keys.items()}
        )

    raise NotImplementedError
