import time


class Timer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.start = time.time()

    @property
    def hours(self) -> float:
        return (time.time() - self.start) / 3600
