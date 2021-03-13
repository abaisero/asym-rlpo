from typing import Dict, List

import gym
import numpy as np

from asym_rlpo.policies import Policy
from asym_rlpo.utils.returns import returns

from .data import Episode, Interaction

# gridverse types
GV_State = Dict[str, np.ndarray]
GV_Action = int
GV_Observation = Dict[str, np.ndarray]
GV_Interaction = Interaction[GV_State, GV_Action, GV_Observation]
GV_Episode = Episode[GV_State, GV_Action, GV_Observation]


# TODO do we even need a full-episoe sampling method???  I use it for on-policy
# actor-critic, but do we need it for DQN?..


# TODO for now this is only a gridverse method (just because of the typing
# though)
def sample_episode(
    env: gym.Env, policy: Policy, *, num_steps: int
) -> GV_Episode:
    env.reset()

    interactions: List[GV_Interaction] = []

    start, done = True, False
    observation = env.reset()
    state = env.state
    policy.reset(observation)

    for _ in range(num_steps):
        action = policy.sample_action()
        observation, reward, done, _ = env.step(action)
        state = env.state
        policy.step(action, observation)

        interactions.append(
            Interaction(
                state=state,
                observation=observation,
                action=action,
                reward=reward,
                start=start,
                done=done,
            )
        )

        start = False

        if done:
            break

    return Episode(interactions)


def sample_episodes(
    env: gym.Env, policy: Policy, *, num_episodes: int, num_steps: int
) -> List[GV_Episode]:
    return [
        sample_episode(env, policy, num_steps=num_steps)
        for _ in range(num_episodes)
    ]


def sample_rewards(
    env: gym.Env, policy: Policy, *, num_steps: int
) -> List[float]:
    # TODO this is means for evaluation, so we need to inject the behavior
    # policy into the sample_episode methods
    episode = sample_episode(env, policy, num_steps=num_steps)
    return [interaction.reward for interaction in episode.interactions]


def evaluate(
    env: gym.Env,
    policy: Policy,
    *,
    discount: float,
    num_episodes: int,
    num_steps: int
) -> np.ndarray:
    # """Return a few empirical returns

    # Args:
    #     env (gym.Env): env
    #     discount (float): discount
    #     num_episodes (int): number of independent sample episode
    #     num_steps (int): max number of time-steps

    # Returns:
    #     np.ndarray:
    # """
    rewards = [
        sample_rewards(env, policy, num_steps=num_steps)
        for _ in range(num_episodes)
    ]
    max_num_steps = max(len(rs) for rs in rewards)
    rewards = [np.pad(rs, (0, max_num_steps - len(rs))) for rs in rewards]
    rewards = np.vstack(rewards)
    return returns(rewards, discount)
