import abc


class DataLogger(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def log(self, data: dict, *, commit: bool = True) -> None:
        assert False

    @abc.abstractmethod
    def commit(self) -> None:
        assert False
