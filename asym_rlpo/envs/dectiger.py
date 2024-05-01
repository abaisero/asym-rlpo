from __future__ import annotations

import gym
import gym.spaces
import numpy as np
from gym import Env, spaces

from asym_rlpo.envs.env import StatefulEnvironment
from asym_rlpo.envs.env_gym import StatefulGymEnv
from asym_rlpo.envs.types import Action, EnvironmentType, Observation, State
from asym_rlpo.envs.wrappers import SingleAgentWrapper

# DecTiger is from Hai;  Since it was not properly organized in a repo, I had
# to just copy/paste it here.  DecTiger_Fix and SingleAgentWrapper are both my
# own code that fix some underlying issues.


"""
Dec-tiger domain from http://masplan.org/problem_domains

2 agents

Uniform random initial state: { tiger-left, tiger-right }

Actions for each agent: { listen, open-left, open-right }

Observations: { hear-left, hear-right }

Observation model:
    both hear correct door: 0.7225
    one hear correct door: 0.1275
    both hear wrong door: 0.0225

Reward model:
    both listen: -2
    both open wrong door: -50
    both open right door: +20
    agent open different door: -100
    one agent open wrong door + other agent listen: -101
    one agent open right door + other agent listen: +9

"""


def make_extra_dectiger() -> DecTigerStatefulEnvironment:
    env = DecTiger()
    env = DecTiger_Fix(env)
    env = SingleAgentWrapper(env)
    return DecTigerStatefulEnvironment(env)


ACTIONS = ('open-left', 'open-right', 'listen')
STATES = ('tiger-left', 'tiger-right')
OBSERVAIONS = ('hear-left', 'hear-right')
JOINT_OBSERVATIONS_RIGHT = ((1, 1), (1, 0), (0, 1), (0, 0))
JOINT_OBSERVATIONS_LEFT = ((0, 0), (1, 0), (0, 1), (1, 1))


class DecTiger(Env):
    def __init__(self):
        super().__init__()
        self.n_agent = 2
        self.action_space = spaces.Discrete(len(ACTIONS))
        self.observation_space = spaces.Discrete(2)
        self.state_space = spaces.Discrete(1)

    def reset(self):
        self.tiger = np.random.randint(0, 2)
        return self.obs_out(None)

    @staticmethod
    def obs_out(obs):
        # onehot encoded observations, supports empty observation
        res = np.zeros((2, 2))
        if obs is not None:
            for i, o in enumerate(obs):
                res[(i, o)] = 1
        return res

    def get_state(self):
        # state space only got two states: tiger-left (0) and riget-right (1)
        return [self.tiger]

    def step(self, a):
        done = True
        o = None
        a1, a2 = a
        if a1 == a2 == 2:  # both listen
            obs = (
                JOINT_OBSERVATIONS_LEFT if self.tiger == 0 else JOINT_OBSERVATIONS_RIGHT
            )
            i = np.random.choice(4, p=(0.7225, 0.1275, 0.1275, 0.0225))
            o = obs[i]
            r = -2
            done = False
        elif a1 == a2 != self.tiger:
            r = 20  # both opened treasure door
        elif a1 == a2 == self.tiger:
            r = -50  # both opened tiger door
        elif a1 != a2:
            if 2 in a:  # one agent listens
                if self.tiger in a:
                    r = -101  # one agent opened tiger door
                else:
                    r = 9  # one agent opened treasure door
            else:
                r = -100  # different doors was open
        return self.obs_out(o), float(r), done


class DecTiger_Fix(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)

        self.num_agents = self.env.n_agent
        self.state_space = spaces.Discrete(2)
        self.action_space = gym.spaces.Tuple(
            [
                gym.spaces.Discrete(self.env.action_space.n)
                for _ in range(self.num_agents)
            ]
        )
        self.observation_space = gym.spaces.Tuple(
            [
                gym.spaces.Box(
                    np.zeros(2, np.float32),
                    np.ones(2, np.float32),
                )
                for _ in range(self.num_agents)
            ]
        )

    def step(self, action):
        ret = self.env.step(action)
        info = {}
        return (*ret, info)

    @property
    def state(self):
        return self.tiger


class DecTigerStatefulEnvironment(StatefulEnvironment):

    def __init__(self, env: StatefulGymEnv):
        super().__init__()
        self._env = env
        self.type = EnvironmentType.EXTRA_DECTIGER

    @property
    def state_space(self) -> gym.spaces.Space:
        return self._env.state_space

    @property
    def action_space(self) -> gym.spaces.Discrete:
        return self._env.action_space

    @property
    def observation_space(self) -> gym.spaces.Space:
        return self._env.observation_space

    def seed(self, seed: int | None = None) -> None:
        self._env.seed(seed)

    def reset(self) -> tuple[State, Observation]:
        observation = self._env.reset()
        state = self._env.state
        return state, observation

    def step(self, action: Action) -> tuple[State, Observation, float, bool]:
        observation, reward, done, _ = self._env.step(action)
        state = self._env.state
        return state, observation, reward, done

    def render(self) -> None:
        self._env.render()
