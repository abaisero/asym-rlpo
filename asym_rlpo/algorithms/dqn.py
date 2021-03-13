from typing import Sequence

import gym
from asym_rlpo.data import Episode, EpisodeBuffer, EpisodeBuffer2
from asym_rlpo.policies import Policy, RandomPolicy

from .base import Algorithm


class DQN(Algorithm):
    def __init__(self, env: gym.Env):
        super().__init__()
        self.env = env
        # self.episode_buffer = EpisodeBuffer(maxlen=1_000_000)
        self.episode_buffer = EpisodeBuffer2(maxlen=1_000)

    def process(self, episodes: Sequence[Episode]):
        self.episode_buffer.append_episodes(episodes)

        print(
            f'episode_buffer stats - '
            f'#interactions: {self.episode_buffer.num_interactions()}'
            f'\t#episodes: {self.episode_buffer.num_episodes()}'
        )

        self.episode_buffer.sample_episode()

    def behavior_policy(self) -> Policy:
        return DQN_BehaviorPolicy(self.env.action_space)

    def target_policy(self) -> Policy:
        return DQN_TargetPolicy(self.env.action_space)


class DQN_BehaviorPolicy(Policy):
    # TODO implement and instantiate the epsilon-greedy policy

    def __init__(self, action_space: gym.Space):
        super().__init__()
        self.__tmp_policy = RandomPolicy(action_space)

    def reset(self, observation):
        # TODO implement real version
        self.__tmp_policy.reset(observation)

    def step(self, action, observation):
        # TODO implement real version
        self.__tmp_policy.step(action, observation)

    def sample_action(self):
        # TODO implement real version
        return self.__tmp_policy.sample_action()


class DQN_TargetPolicy(Policy):
    # TODO implement and instantiate the argmax policy

    def __init__(self, action_space: gym.Space):
        super().__init__()
        self.__tmp_policy = RandomPolicy(action_space)

    def reset(self, observation):
        # TODO implement real version
        self.__tmp_policy.reset(observation)

    def step(self, action, observation):
        # TODO implement real version
        self.__tmp_policy.step(action, observation)

    def sample_action(self):
        # TODO implement real version
        return self.__tmp_policy.sample_action()
