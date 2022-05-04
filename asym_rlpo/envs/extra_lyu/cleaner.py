import itertools as itt
import time

# import cv2
import gym
import numpy as np
from gym import spaces

from .maze import Maze
from .single_agent_wrapper import SingleAgentWrapper


class EnvCleaner(object):
    def __init__(self, N_agent=2, map_size=13, seed=5):
        self.map_size = map_size
        self.seed = seed
        self.occupancy = self.generate_maze(seed)
        self.N_agent = N_agent
        self.agt_pos_list = []
        self.obs_size = 3
        n_channel = 3
        self.observation_space = spaces.Box(
            low=-255,
            high=255,
            dtype=np.float32,
            shape=(self.obs_size * self.obs_size * n_channel,),
        )
        self.action_space = spaces.Discrete(4)
        self.state_space = spaces.Box(
            low=-255, high=255, shape=(169,), dtype=np.float32
        )
        for i in range(self.N_agent):
            self.agt_pos_list.append([1, 1])

    def generate_maze(self, seed):
        symbols = {
            # default symbols
            "start": "S",
            "end": "X",
            "wall_v": "|",
            "wall_h": "-",
            "wall_c": "+",
            "head": "#",
            "tail": "o",
            "empty": " ",
        }
        maze_obj = Maze(
            int((self.map_size - 1) / 2),
            int((self.map_size - 1) / 2),
            seed,
            symbols,
            1,
        )
        grid_map = maze_obj.to_np()
        for i in range(self.map_size):
            for j in range(self.map_size):
                if grid_map[i][j] == 0:
                    grid_map[i][j] = 2
        return grid_map

    def step(self, action_list):
        reward = 0.0
        self.i_step += 1
        for i in range(len(action_list)):
            if action_list[i] == 0:  # up
                # if can move
                if (
                    self.occupancy[self.agt_pos_list[i][0] - 1][
                        self.agt_pos_list[i][1]
                    ]
                    != 1
                ):
                    self.agt_pos_list[i][0] = self.agt_pos_list[i][0] - 1
            if action_list[i] == 1:  # down
                # if can move
                if (
                    self.occupancy[self.agt_pos_list[i][0] + 1][
                        self.agt_pos_list[i][1]
                    ]
                    != 1
                ):
                    self.agt_pos_list[i][0] = self.agt_pos_list[i][0] + 1
            if action_list[i] == 2:  # left
                # if can move
                if (
                    self.occupancy[self.agt_pos_list[i][0]][
                        self.agt_pos_list[i][1] - 1
                    ]
                    != 1
                ):
                    self.agt_pos_list[i][1] = self.agt_pos_list[i][1] - 1
            if action_list[i] == 3:  # right
                # if can move
                if (
                    self.occupancy[self.agt_pos_list[i][0]][
                        self.agt_pos_list[i][1] + 1
                    ]
                    != 1
                ):
                    self.agt_pos_list[i][1] = self.agt_pos_list[i][1] + 1
            # if the spot is dirty
            if (
                self.occupancy[self.agt_pos_list[i][0]][self.agt_pos_list[i][1]]
                == 2
            ):
                self.occupancy[self.agt_pos_list[i][0]][
                    self.agt_pos_list[i][1]
                ] = 0
                reward = reward + 1
        return self.get_obs(), reward, self.i_step >= 200

    def get_obs(self):
        return [
            self.get_local_obs(self.agt_pos_list[0], self.agt_pos_list[1]),
            self.get_local_obs(self.agt_pos_list[1], self.agt_pos_list[0]),
        ]

    def get_local_obs(self, agt_pos, teammate_pos, flat=True):
        obs = np.zeros((3, 3, 3))
        for i in range(3):
            for j in range(3):
                if self.occupancy[agt_pos[0] - 1 + i][agt_pos[1] - 1 + j] == 0:
                    obs[i, j, 0] = 1.0
                    obs[i, j, 1] = 1.0
                    obs[i, j, 2] = 1.0
                if self.occupancy[agt_pos[0] - 1 + i][agt_pos[1] - 1 + j] == 2:
                    obs[i, j, 0] = 0.0
                    obs[i, j, 1] = 1.0
                    obs[i, j, 2] = 0.0
                d_x = teammate_pos[0] - agt_pos[0]
                d_y = teammate_pos[1] - agt_pos[1]
                if d_x >= -1 and d_x <= 1 and d_y >= -1 and d_y <= 1:
                    obs[1 + d_x, 1 + d_y, 0] = 0.0
                    obs[1 + d_x, 1 + d_y, 1] = 0.0
                    obs[1 + d_x, 1 + d_y, 2] = 1.0
        obs[1, 1, 0] = 1.0
        obs[1, 1, 1] = 0.0
        obs[1, 1, 2] = 0.0
        if flat:
            obs = obs.reshape(-1)
        return obs

    def get_global_obs(self):
        obs = np.zeros((self.map_size, self.map_size, 3))
        for i in range(self.map_size):
            for j in range(self.map_size):
                if self.occupancy[i, j] == 0:
                    obs[i, j, 0] = 1.0
                    obs[i, j, 1] = 1.0
                    obs[i, j, 2] = 1.0
                if self.occupancy[i, j] == 2:
                    obs[i, j, 0] = 0.0
                    obs[i, j, 1] = 1.0
                    obs[i, j, 2] = 0.0
        for i in range(self.N_agent):
            obs[self.agt_pos_list[i][0], self.agt_pos_list[i][1], 0] = 1.0
            obs[self.agt_pos_list[i][0], self.agt_pos_list[i][1], 1] = 0.0
            obs[self.agt_pos_list[i][0], self.agt_pos_list[i][1], 2] = 0.0
        return obs

    def reset(self):
        self.i_step = 0
        self.occupancy = self.generate_maze(self.seed)
        self.agt_pos_list = []
        for i in range(self.N_agent):
            self.agt_pos_list.append([1, 1])
        return self.get_obs()

    def get_state(self):
        obs = self.occupancy.copy()
        obs[self.agt_pos_list[0][0], self.agt_pos_list[0][1]] = 3
        obs[self.agt_pos_list[1][0], self.agt_pos_list[1][1]] = 4
        obs = obs / 4
        return obs.reshape(-1)

    def render(self):
        obs = self.get_global_obs()
        enlarge = 5
        new_obs = np.ones((self.map_size * enlarge, self.map_size * enlarge, 3))
        for i in range(self.map_size):
            for j in range(self.map_size):
                if (
                    obs[i][j][0] == 0.0
                    and obs[i][j][1] == 0.0
                    and obs[i][j][2] == 0.0
                ):
                    cv2.rectangle(
                        new_obs,
                        (i * enlarge, j * enlarge),
                        (i * enlarge + enlarge, j * enlarge + enlarge),
                        (0, 0, 0),
                        -1,
                    )
                if (
                    obs[i][j][0] == 1.0
                    and obs[i][j][1] == 0.0
                    and obs[i][j][2] == 0.0
                ):
                    cv2.rectangle(
                        new_obs,
                        (i * enlarge, j * enlarge),
                        (i * enlarge + enlarge, j * enlarge + enlarge),
                        (0, 0, 255),
                        -1,
                    )
                if (
                    obs[i][j][0] == 0.0
                    and obs[i][j][1] == 1.0
                    and obs[i][j][2] == 0.0
                ):
                    cv2.rectangle(
                        new_obs,
                        (i * enlarge, j * enlarge),
                        (i * enlarge + enlarge, j * enlarge + enlarge),
                        (0, 255, 0),
                        -1,
                    )
        cv2.imshow("image", new_obs)
        cv2.waitKey(10)


class EnvCleaner_Fix(gym.Env):
    def __init__(self, env: gym.Env):
        super().__init__()

        self.env = env
        self.num_agents = self.env.N_agent
        self.state_space = gym.spaces.Box(
            np.zeros((self.env.map_size**2 * 5), np.float32),
            np.ones((self.env.map_size**2 * 5), np.float32),
        )
        self.action_space = gym.spaces.Tuple(
            [
                gym.spaces.Discrete(self.env.action_space.n)
                for _ in range(self.num_agents)
            ]
        )
        self.observation_space = gym.spaces.Tuple(
            [
                gym.spaces.Box(
                    np.zeros(3 * 3 * 3, np.float32),
                    np.ones(3 * 3 * 3, np.float32),
                )
                for _ in range(self.num_agents)
            ]
        )

    @property
    def state(self):
        state = np.zeros((self.env.map_size, self.env.map_size, 5), np.float32)
        state[:, :, 0] = self.env.occupancy == 0
        state[:, :, 1] = self.env.occupancy == 1
        state[:, :, 2] = self.env.occupancy == 2
        state[(*self.env.agt_pos_list[0], 3)] = 1
        state[(*self.env.agt_pos_list[1], 4)] = 1
        return state.flatten()

    def reset(self, **kwargs):
        observations = self.env.reset(**kwargs)
        observations = np.stack(observations)
        return observations

    def step(self, actions):
        observations, *ret = self.env.step(actions)
        observations = np.stack(observations)
        info = {}
        return (observations, *ret, info)


def main():
    env = EnvCleaner()
    env = EnvCleaner_Fix(env)
    env = SingleAgentWrapper(env)

    while True:
        observations = env.reset()
        print(f'state: {env.state}')
        assert env.state_space.contains(env.state)
        print(f'observations: {observations}')
        assert env.observation_space.contains(observations)

        for t in itt.count():
            time.sleep(1)
            print(f't: {t}')

            actions = env.action_space.sample()
            print(f'actions: {actions}')
            assert env.action_space.contains(actions)

            observations, rewards, done = env.step(actions)
            print(f'state: {env.state}')
            assert env.state_space.contains(env.state)
            print(f'observations: {observations}')
            assert env.observation_space.contains(observations)
            print(f'rewards: {rewards}')
            print(f'done: {done}')

            if done:
                time.sleep(1)
                break


if __name__ == '__main__':
    main()
