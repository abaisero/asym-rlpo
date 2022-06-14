import abc
from collections import deque
from typing import Deque, Dict, Sequence

from asym_rlpo.utils.checkpointing import Serializer


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


class InfiniteRunningAverageSerializer(Serializer[RunningAverage]):
    def serialize(self, obj: InfiniteRunningAverage) -> Dict:
        return {
            'cum_value': obj.cum_value,
            'num_values': obj.num_values,
        }

    def deserialize(self, obj: InfiniteRunningAverage, data: Dict):
        obj.cum_value = data['cum_value']
        obj.num_values = data['num_values']


class WindowRunningAverage(RunningAverage):
    def __init__(self, size: int):
        self.values: Deque[float] = deque(maxlen=size)

    def append(self, value: float):
        self.values.append(value)

    def extend(self, values: Sequence[float]):
        self.values.extend(values)

    def value(self) -> float:
        return sum(self.values) / len(self.values)


class WindowRunningAverageSerializer(Serializer[WindowRunningAverage]):
    def serialize(self, obj: WindowRunningAverage) -> Dict:
        return {'values': obj.values}

    def deserialize(self, obj: WindowRunningAverage, data: Dict):
        obj.values = data['values']


class RunningAverageSerializer(Serializer[RunningAverage]):
    def __init__(self):
        self.infinite_running_average_serializer = (
            InfiniteRunningAverageSerializer()
        )
        self.window_running_average_serializer = (
            WindowRunningAverageSerializer()
        )

    def serialize(self, obj: RunningAverage) -> Dict:
        if isinstance(obj, InfiniteRunningAverage):
            return self.infinite_running_average_serializer.serialize(obj)

        if isinstance(obj, WindowRunningAverage):
            return self.window_running_average_serializer.serialize(obj)

        raise TypeError(f'invalid type {type(obj)}')

    def deserialize(self, obj: RunningAverage, data: Dict):
        if isinstance(obj, InfiniteRunningAverage):
            return self.infinite_running_average_serializer.deserialize(
                obj, data
            )

        if isinstance(obj, WindowRunningAverage):
            return self.window_running_average_serializer.deserialize(obj, data)

        raise TypeError(f'invalid type {type(obj)}')
