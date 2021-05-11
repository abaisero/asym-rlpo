from collections import deque
from typing import Deque, Sequence


class InfiniteRunningAverage:
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


class WindowRunningAverage:
    def __init__(self, size: int):
        self.values: Deque[float] = deque(maxlen=size)

    def append(self, value: float):
        self.values.append(value)

    def extend(self, values: Sequence[float]):
        self.values.extend(values)

    def value(self) -> float:
        return sum(self.values) / len(self.values)
