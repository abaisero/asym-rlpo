import pytest

from asym_rlpo.utils.config import get_config


def pytest_configure(config):
    config.addinivalue_line("markers", "system: marks slow system tests")
    config.addinivalue_line("markers", "a2c: marks slow a2c system tests")
    config.addinivalue_line("markers", "dqn: marks slow dqn system tests")


# setup config singleton when running tests
@pytest.fixture(scope='session', autouse=True)
def execute_before_session():
    pass
    # this is where I can set up config data
    # config = get_config()
    # config._update(
    #     {
    #         ...
    #     }
    # )


# setup config singleton when running tests
@pytest.fixture(autouse=True)
def execute_before_tests():
    config = get_config()
    config._update({'history_model': 'rnn'})
