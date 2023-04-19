from collections.abc import Callable
from typing import Generic, TypeVar

T = TypeVar('T')

Observer = Callable[[T], None]


class Subject(Generic[T]):
    """Implementation of the observer pattern"""

    def __init__(self):
        self.__observers: list[Observer[T]] = []

    def attach(self, observer: Observer[T]) -> Observer[T]:
        self.__observers.append(observer)
        return observer

    def notify(self, obj: T):
        for observer in self.__observers:
            observer(obj)
