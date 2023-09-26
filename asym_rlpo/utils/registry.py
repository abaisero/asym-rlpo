from typing import Generic, TypeVar

T = TypeVar('T')

class Registry(Generic[T]):
    def __init__(self):
        self._registry: dict[str, T] = {}

    def __getitem__(self, name: str) -> T:
        return self._registry[name]

    def __setitem__(self, name: str, item: T):
        self._registry[name] = item

    def decorated(self, name: str):
        def inner(item: T):
            self[name] = item
            return item
        return inner
