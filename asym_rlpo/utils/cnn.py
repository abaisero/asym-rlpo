import torch.nn as nn
import yaml


def make_cnn_from_filename(filename: str, channels: int) -> nn.Sequential:
    with open(filename) as f:
        data = yaml.safe_load(f)
        return make_cnn_from_data(data, channels)


def make_cnn_from_data(data: dict, channels: int) -> nn.Sequential:
    modules: list[nn.Module] = []
    for d in data:
        module, channels = _make_cnn_module(d, channels)
        modules.append(module)

    return nn.Sequential(*modules)


def _make_cnn_module(data: dict, in_channels: int) -> tuple[nn.Module, int]:
    name = data['name']

    if name == 'conv2d':
        out_channels = data['channels']
        return (
            nn.Conv2d(in_channels, out_channels, **data['kwargs']),
            out_channels,
        )

    if name == 'relu':
        return nn.ReLU(), in_channels

    if name == 'maxpool2d':
        return nn.MaxPool2d(**data['kwargs']), in_channels

    raise ValueError(f'invalid module name {name}')
