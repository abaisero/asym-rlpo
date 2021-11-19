import abc
import os
import pickle
from typing import Any


def save_data(filename: str, data: Any):
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    except FileNotFoundError:
        pass

    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def load_data(filename: str) -> Any:
    with open(filename, 'rb') as f:
        return pickle.load(f)


class Serializable(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def state_dict(self):
        assert False

    @abc.abstractmethod
    def load_state_dict(self, data):
        assert False
