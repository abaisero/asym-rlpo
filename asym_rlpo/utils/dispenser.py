import time

from asym_rlpo.utils.checkpointing import Serializable


class DiscreteDispenser(Serializable):
    """Dispenses `True` no more than once every `n` steps."""

    def __init__(self, n: int):
        self.n = n
        self.next_i = 0

    def dispense(self, i) -> bool:
        if i >= self.next_i:
            self.next_i = i + self.n
            return True

        return False

    def state_dict(self):
        return {
            'n': self.n,
            'next_i': self.next_i,
        }

    def load_state_dict(self, data):
        self.n = data['n']
        self.next_i = data['next_i']


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
