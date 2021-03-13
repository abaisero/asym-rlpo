import abc
from typing import Sequence

from asym_rlpo.data import Episode
from asym_rlpo.policies import Policy


class Algorithm(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def process(self, episodes: Sequence[Episode]):
        assert False

    @abc.abstractmethod
    def behavior_policy(self) -> Policy:
        assert False

    @abc.abstractmethod
    def target_policy(self) -> Policy:
        assert False
