import pytest

from asym_rlpo.utils.config import get_config


# setup config singleton when running tests
@pytest.fixture(scope='session', autouse=True)
def execute_before_tests():
    config = get_config()
    config._update(
        {
            'hs_features_dim': 0,
            'normalize_hs_features': False,
        }
    )
