import time

from asym_rlpo.utils.checkpointing import Serializable


class Timer(Serializable):
    def __init__(self):
        self.reset()

    def reset(self):
        self.start = time.time()

    @property
    def seconds(self) -> float:
        return time.time() - self.start

    @property
    def hours(self) -> float:
        return self.seconds / 3600

    def state_dict(self):
        return {'seconds': self.seconds}

    def load_state_dict(self, data):
        self.start = time.time() - data['seconds']


class Dispenser(Serializable):
    """Returns True no more than once every `n` steps."""

    def __init__(self, n: int):
        self.n = n
        self.next_i = 0

    def dispense(self, i):
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
