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
