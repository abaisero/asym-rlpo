from __future__ import annotations

from time import time
from typing import Dict

from dateutil.parser import parse as datetime_parser

from asym_rlpo.utils.checkpointing import Serializer


class StepDispenser:
    """Dispenses `True` no more than once every `n` steps."""

    def __init__(self, n: int):
        self.n = n
        self.next_i = 0

    def dispense(self, i) -> bool:
        if i >= self.next_i:
            self.next_i = i + self.n
            return True

        return False


class StepDispenserSerializer(Serializer[StepDispenser]):
    def serialize(self, obj: StepDispenser) -> Dict:
        return {
            'n': obj.n,
            'next_i': obj.next_i,
        }

    def deserialize(self, obj: StepDispenser, data: Dict):
        obj.n = data['n']
        obj.next_i = data['next_i']


class TimePeriodDispenser:
    """Dispenses `True` no more than every period (in seconds)."""

    def __init__(self, period: float):
        self.period = period
        self.next_t = time()

    def dispense(self) -> bool:
        t = time()

        if t < self.next_t:
            return False

        self.next_t = t + self.period
        return True


class TimestampDispenser:
    """Dispenses `True` after a given timestamp."""

    def __init__(self, timestamp: float):
        self.timestamp = timestamp

    @staticmethod
    def from_datetime(datetime: str) -> TimestampDispenser:
        timestamp = datetime_parser(datetime).timestamp()
        return TimestampDispenser(timestamp)

    def dispense(self) -> bool:
        return time() >= self.timestamp
