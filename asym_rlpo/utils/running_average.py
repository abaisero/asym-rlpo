import abc
from collections import deque
from typing import Deque, Sequence

from asym_rlpo.utils.checkpointing import Serializable


class RunningAverage(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def append(self, value: float):
        assert False

    @abc.abstractmethod
    def extend(self, values: Sequence[float]):
        assert False

    @abc.abstractmethod
    def value(self) -> float:
        assert False


class InfiniteRunningAverage(RunningAverage, Serializable):
    def __init__(self):
        self.cum_value = 0.0
        self.num_values = 0

    def append(self, value: float):
        self.cum_value += value
        self.num_values += 1

    def extend(self, values: Sequence[float]):
        self.cum_value += sum(values)
        self.num_values += len(values)

    def value(self) -> float:
        return self.cum_value / self.num_values

    def state_dict(self):
        return {
            'cum_value': self.cum_value,
            'num_values': self.num_values,
        }

    def load_state_dict(self, data):
        self.cum_value = data['cum_value']
        self.num_values = data['num_values']


class WindowRunningAverage(RunningAverage, Serializable):
    def __init__(self, size: int):
        self.values: Deque[float] = deque(maxlen=size)

    def append(self, value: float):
        self.values.append(value)

    def extend(self, values: Sequence[float]):
        self.values.extend(values)

    def value(self) -> float:
        return sum(self.values) / len(self.values)

    def state_dict(self):
        return {'values': self.values}

    def load_state_dict(self, data):
        self.values = data['values']
