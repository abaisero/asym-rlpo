import abc


class Representation(metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def dim(self):
        assert False
