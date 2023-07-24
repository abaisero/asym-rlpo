from __future__ import annotations

from time import time


class Dispenser:
    """Dispenses `True` no more than once every `period`."""

    def __init__(self, value: float, period: float):
        self.__next_value = value
        self.__period = period

    def dispense(self, value, *, consume=True) -> bool:
        dispense = self.__next_value <= value

        if dispense and consume:
            self.__next_value = value + self.__period

        return dispense


class TimeDispenser:
    """Dispenses `True` no more than once every `period` seconds."""

    def __init__(self, period: float):
        self.__dispenser = Dispenser(time(), period)

    def dispense(self, *, consume=True) -> bool:
        return self.__dispenser.dispense(time(), consume=consume)
