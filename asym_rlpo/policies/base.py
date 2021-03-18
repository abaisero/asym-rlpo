import abc


class Policy(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def reset(self, observation):
        assert False

    @abc.abstractmethod
    def step(self, action, observation):
        assert False

    @abc.abstractmethod
    def sample_action(self, state):
        assert False


class FullyObservablePolicy(Policy):
    def reset(self, observation):
        pass

    def step(self, action, observation):
        pass

    def sample_action(self, state):
        return self.fo_sample_action(state)

    @abc.abstractmethod
    def fo_sample_action(self, state):
        assert False


class PartiallyObservablePolicy(Policy):
    def sample_action(self, state):
        return self.po_sample_action()

    @abc.abstractmethod
    def po_sample_action(self):
        assert False
