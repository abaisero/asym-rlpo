import abc
from typing import Dict

class DataLogger(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def log(self, data: Dict) -> None:
        assert False
