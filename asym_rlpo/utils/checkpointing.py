import abc
import os
import pickle
from typing import Any, Dict, Generic, TypeVar


def save_data(filename: str, data: Any):
    dirname = os.path.dirname(filename)
    os.makedirs(dirname, exist_ok=True)

    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def load_data(filename: str) -> Any:
    with open(filename, 'rb') as f:
        return pickle.load(f)


T = TypeVar('T')


class Serializer(Generic[T], metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def serialize(self, obj: T) -> Dict:
        assert False

    @abc.abstractmethod
    def deserialize(self, obj: T, data: Dict):
        assert False
