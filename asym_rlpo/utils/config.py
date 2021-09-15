from typing import Dict, Union

BasicType = Union[str, float, int, bool, None]
ConfigDict = Dict[str, BasicType]


class Config:
    def __init__(self):
        self._cfg = {}

    def _update(self, cd: ConfigDict):
        self._cfg.update(cd)

    def _as_dict(self) -> ConfigDict:
        return self._cfg.copy()

    def __getattr__(self, name):
        return self._cfg[name]

    def __getitem__(self, key):
        return self._cfg[key]


_config = None


def get_config() -> Config:
    global _config  # pylint: disable=global-statement

    if _config is None:
        _config = Config()

    return _config
