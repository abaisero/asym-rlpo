from datetime import timedelta
from time import time


class Timer:
    def __init__(self):
        self.__start_timestamp = time()

    @property
    def seconds(self) -> float:
        return time() - self.__start_timestamp

    @property
    def hours(self) -> float:
        return self.seconds / 3600

    def __str__(self) -> str:
        return str(timedelta(seconds=int(self.seconds)))

    def __getstate__(self):
        return self.seconds

    def __setstate__(self, data):
        self.__start_timestamp = time() - data


def timestamp_is_future(timestamp: float) -> bool:
    return time() < timestamp


def timestamp_is_past(timestamp: float) -> bool:
    return timestamp < time()
