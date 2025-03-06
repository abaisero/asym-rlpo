import enum

from asym_rlpo.policies import Policy


class Locations(enum.Enum):
    CENTER_CORRIDOR = enum.auto()
    TOP_FORK = enum.auto()
    LEFT_CORRIDOR = enum.auto()
    RIGHT_CORRIDOR = enum.auto()
    BOTTOM_FORK = enum.auto()
    BOTTOM_CORRIDOR = enum.auto()


class SubGoals(enum.Enum):
    REACH_ORACLE = enum.auto()
    REACH_LEFT_EXIT = enum.auto()
    REACH_RIGHT_EXIT = enum.auto()


class Actions(enum.Enum):
    UP = enum.auto()
    DOWN = enum.auto()
    RIGHT = enum.auto()
    LEFT = enum.auto()


class HeavenHell_HardcodedPolicy(Policy):
    def __init__(self, size: int):
        super().__init__()

        self.size = size

    def reset(self, observation):
        self.location = Locations.CENTER_CORRIDOR
        self.subgoal = SubGoals.REACH_ORACLE

    def step(self, action, observation):
        self.location = get_location(observation, size=self.size)
        self.subgoal = get_subgoal(self.subgoal, observation, size=self.size)

    def sample_action(self):
        action = _istate_to_action[(self.subgoal, self.location)]
        return _action_to_int[action]


def get_location(observation: int, *, size: int) -> Locations:
    if 0 <= observation <= size - 1:
        return Locations.CENTER_CORRIDOR

    if observation == size:
        return Locations.TOP_FORK

    if size + 1 <= observation <= 2 * size:
        return Locations.LEFT_CORRIDOR

    if 2 * size + 1 <= observation <= 3 * size:
        return Locations.RIGHT_CORRIDOR

    if observation == 3 * size + 1:
        return Locations.BOTTOM_FORK

    if 3 * size + 2 <= observation <= 4 * size + 2:
        return Locations.BOTTOM_CORRIDOR

    raise ValueError(f"Invalid {observation=}")


def get_subgoal(subgoal: SubGoals, observation: int, *, size: int) -> SubGoals:
    if subgoal is SubGoals.REACH_ORACLE:
        if observation == 4 * size + 1:
            return SubGoals.REACH_LEFT_EXIT

        if observation == 4 * size + 2:
            return SubGoals.REACH_RIGHT_EXIT

    return SubGoals.REACH_ORACLE


_istate_to_action: dict[tuple[SubGoals, Locations], Actions] = {
    # REACH_ORACLE
    (SubGoals.REACH_ORACLE, Locations.CENTER_CORRIDOR): Actions.DOWN,
    (SubGoals.REACH_ORACLE, Locations.TOP_FORK): Actions.DOWN,
    (SubGoals.REACH_ORACLE, Locations.LEFT_CORRIDOR): Actions.RIGHT,
    (SubGoals.REACH_ORACLE, Locations.RIGHT_CORRIDOR): Actions.LEFT,
    (SubGoals.REACH_ORACLE, Locations.BOTTOM_FORK): Actions.RIGHT,
    (SubGoals.REACH_ORACLE, Locations.BOTTOM_CORRIDOR): Actions.RIGHT,
    # REACH_LEFT_EXIT
    (SubGoals.REACH_LEFT_EXIT, Locations.CENTER_CORRIDOR): Actions.UP,
    (SubGoals.REACH_LEFT_EXIT, Locations.TOP_FORK): Actions.LEFT,
    (SubGoals.REACH_LEFT_EXIT, Locations.LEFT_CORRIDOR): Actions.LEFT,
    (SubGoals.REACH_LEFT_EXIT, Locations.RIGHT_CORRIDOR): Actions.LEFT,
    (SubGoals.REACH_LEFT_EXIT, Locations.BOTTOM_FORK): Actions.UP,
    (SubGoals.REACH_LEFT_EXIT, Locations.BOTTOM_CORRIDOR): Actions.LEFT,
    # REACH_RIGHT_EXIT
    (SubGoals.REACH_RIGHT_EXIT, Locations.CENTER_CORRIDOR): Actions.UP,
    (SubGoals.REACH_RIGHT_EXIT, Locations.TOP_FORK): Actions.RIGHT,
    (SubGoals.REACH_RIGHT_EXIT, Locations.LEFT_CORRIDOR): Actions.RIGHT,
    (SubGoals.REACH_RIGHT_EXIT, Locations.RIGHT_CORRIDOR): Actions.RIGHT,
    (SubGoals.REACH_RIGHT_EXIT, Locations.BOTTOM_FORK): Actions.UP,
    (SubGoals.REACH_RIGHT_EXIT, Locations.BOTTOM_CORRIDOR): Actions.LEFT,
}

_action_to_int = {
    Actions.DOWN: 0,
    Actions.UP: 1,
    Actions.RIGHT: 2,
    Actions.LEFT: 3,
}
