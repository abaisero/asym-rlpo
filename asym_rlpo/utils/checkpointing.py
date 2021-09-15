# import torch
import os
import pickle
from typing import Any


def save_data(filename: str, data: Any):
    os.mkdir(os.path.dirname(filename))
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def load_data(filename: str) -> Any:
    with open(filename, 'rb') as f:
        return pickle.load(f)
