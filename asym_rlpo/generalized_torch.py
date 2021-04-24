from typing import Dict, TypeVar

import torch

GTensor = TypeVar('GTensor', torch.Tensor, Dict[str, torch.Tensor])


def f_apply(f, x: GTensor, *args, **kwargs) -> GTensor:
    return (
        {k: f(v, *args, **kwargs) for k, v in x.items()}
        if isinstance(x, dict)
        else f(x, *args, **kwargs)
    )


def tensor_apply(x: GTensor, method_name: str, *args, **kwargs) -> GTensor:
    return (
        {k: getattr(v, method_name)(*args, **kwargs) for k, v in x.items()}
        if isinstance(x, dict)
        else getattr(x, method_name)(*args, **kwargs)
    )


def zeros_like(x: GTensor, *args, **kwargs) -> GTensor:
    return f_apply(torch.zeros_like, x, *args, **kwargs)


def squeeze(x: GTensor, *args, **kwargs) -> GTensor:
    return tensor_apply(x, 'squeeze', *args, **kwargs)


def unsqueeze(x: GTensor, *args, **kwargs) -> GTensor:
    return tensor_apply(x, 'unsqueeze', *args, **kwargs)


def to(x: GTensor, *args, **kwargs) -> GTensor:
    return tensor_apply(x, 'to', *args, **kwargs)
