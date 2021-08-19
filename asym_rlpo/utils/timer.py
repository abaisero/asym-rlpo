import time


class Timer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.start = time.time()

    @property
    def hours(self) -> float:
        return (time.time() - self.start) / 3600


class Dispenser:
    """Returns True no more than once every `n` steps."""

    def __init__(self, n: int):
        self.n = n
        self.next_i = 0

    def dispense(self, i):
        if i >= self.next_i:
            self.next_i = i + self.n
            return True

        return False
