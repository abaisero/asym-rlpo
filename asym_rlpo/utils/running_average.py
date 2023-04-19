import abc
from collections import deque
from collections.abc import Sequence


class RunningAverage(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def append(self, value: float):
        assert False

    def extend(self, values: Sequence[float]):
        for value in values:
            self.append(value)

    @abc.abstractmethod
    def value(self) -> float:
        assert False


class InfiniteRunningAverage(RunningAverage):
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


class WindowRunningAverage(RunningAverage):
    def __init__(self, size: int):
        self.values = deque(maxlen=size)

    def append(self, value: float):
        self.values.append(value)

    def extend(self, values: Sequence[float]):
        self.values.extend(values)

    def value(self) -> float:
        return sum(self.values) / len(self.values)
