from typing import Iterable, Optional

import torch.nn as nn

from asym_rlpo.envs import Environment, EnvironmentType
from asym_rlpo.models.models_extra_carflag import (
    make_models as make_models_extra_carflag,
)
from asym_rlpo.models.models_extra_cleaner import (
    make_models as make_models_extra_cleaner,
)
from asym_rlpo.models.models_extra_dectiger import (
    make_models as make_models_extra_dectiger,
)
from asym_rlpo.models.models_flat import make_models as make_models_flat
from asym_rlpo.models.models_gv import make_models as make_models_gv
from asym_rlpo.models.models_openai import make_models as make_models_openai
from asym_rlpo.utils.debugging import checkraise


def make_models(
    env: Environment,
    *,
    keys: Optional[Iterable[str]] = None,
) -> nn.ModuleDict:

    if env.type is EnvironmentType.GV:
        models = make_models_gv(env)

    elif env.type is EnvironmentType.OPENAI:
        models = make_models_openai(env)

    elif env.type is EnvironmentType.EXTRA_DECTIGER:
        models = make_models_extra_dectiger(env)

    elif env.type is EnvironmentType.EXTRA_CLEANER:
        models = make_models_extra_cleaner(env)

    elif env.type is EnvironmentType.EXTRA_CARFLAG:
        models = make_models_extra_carflag(env)

    elif env.type is EnvironmentType.FLAT:
        models = make_models_flat(env)

    else:
        raise NotImplementedError

    return models if keys is None else filter_models(models, keys)


def filter_models(models: nn.ModuleDict, keys: Iterable[str]) -> nn.ModuleDict:
    if isinstance(keys, list):
        missing_keys = set(keys) - set(models.keys())
        checkraise(
            len(missing_keys) == 0,
            ValueError,
            'models dictionary does not contains keys {}',
            missing_keys,
        )
        return nn.ModuleDict({k: models[k] for k in keys})

    if isinstance(keys, dict):
        missing_keys = set(keys.keys()) - set(models.keys())
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
