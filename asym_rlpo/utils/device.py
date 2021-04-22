import torch


def get_device(device: str) -> torch.device:
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    return torch.device(device)
