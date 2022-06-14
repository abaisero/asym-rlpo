import time
from typing import Dict

from asym_rlpo.utils.checkpointing import Serializer


class DiscreteDispenser:
    """Dispenses `True` no more than once every `n` steps."""

    def __init__(self, n: int):
        self.n = n
        self.next_i = 0

    def dispense(self, i) -> bool:
        if i >= self.next_i:
            self.next_i = i + self.n
            return True

        return False


class DiscreteDispenserSerializer(Serializer[DiscreteDispenser]):
    def serialize(self, obj: DiscreteDispenser) -> Dict:
        return {
            'n': obj.n,
            'next_i': obj.next_i,
        }

    def deserialize(self, obj: DiscreteDispenser, data: Dict):
        obj.n = data['n']
        obj.next_i = data['next_i']


class TimeDispenser:
    """Dispenses `True` no more than once every period (in seconds)."""

    def __init__(self, period: float):
        self.period = period
        self.next_t = time.time()

    def dispense(self) -> bool:
        t = time.time()
        if t >= self.next_t:
            self.next_t = t + self.period
            return True

        return False
