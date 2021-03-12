import abc
from typing import Sequence

from asym_rlpo.data import Episode


class Algorithm(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def process(self, episodes: Sequence[Episode]):
        assert False
