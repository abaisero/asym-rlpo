import time

from asym_rlpo.utils.dispenser import Dispenser, TimeDispenser


def test_dispenser():
    dispenser = Dispenser(0, 3)

    assert dispenser.dispense(0)
    assert not dispenser.dispense(0)
    assert not dispenser.dispense(1)
    assert not dispenser.dispense(2)
    assert dispenser.dispense(3)
    assert not dispenser.dispense(3)
    assert not dispenser.dispense(4)
    assert not dispenser.dispense(5)
    assert dispenser.dispense(6)
    assert not dispenser.dispense(6)
    assert not dispenser.dispense(7)
    assert not dispenser.dispense(8)
    assert dispenser.dispense(9)
    assert not dispenser.dispense(9)

    assert dispenser.dispense(12)
    assert not dispenser.dispense(12)
    assert dispenser.dispense(15)
    assert not dispenser.dispense(15)
    assert dispenser.dispense(18)
    assert not dispenser.dispense(18)

    assert dispenser.dispense(100)
    assert not dispenser.dispense(100)
    assert not dispenser.dispense(101)
    assert not dispenser.dispense(102)
    assert dispenser.dispense(103)
    assert not dispenser.dispense(103)
    assert not dispenser.dispense(104)
    assert not dispenser.dispense(105)
    assert dispenser.dispense(106)
    assert not dispenser.dispense(106)
    assert not dispenser.dispense(107)
    assert not dispenser.dispense(108)
    assert dispenser.dispense(109)
    assert not dispenser.dispense(109)


def test_time_dispenser():
    dispenser = TimeDispenser(0.1)

    assert dispenser.dispense()
    for _ in range(100):
        assert not dispenser.dispense()

    time.sleep(0.2)
    assert dispenser.dispense()
    for _ in range(100):
        assert not dispenser.dispense()

    time.sleep(0.2)
    assert dispenser.dispense()
    for _ in range(100):
        assert not dispenser.dispense()

    time.sleep(0.2)
    assert dispenser.dispense()
    for _ in range(100):
        assert not dispenser.dispense()

    time.sleep(0.2)
    assert dispenser.dispense()
    for _ in range(100):
        assert not dispenser.dispense()
