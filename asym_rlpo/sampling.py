from typing import Dict, List

import gym
import numpy as np

from asym_rlpo.policies import Policy

from .data import Episode, Interaction, RawEpisode

# gridverse types
GV_State = Dict[str, np.ndarray]
GV_Observation = Dict[str, np.ndarray]
GV_Interaction = Interaction[GV_State, GV_Observation]
GV_Episode = Episode[GV_State, GV_Observation]


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

    return Episode(RawEpisode(interactions))


def sample_episodes(
    env: gym.Env, policy: Policy, *, num_episodes: int, num_steps: int
) -> List[GV_Episode]:
    return [
        sample_episode(env, policy, num_steps=num_steps)
        for _ in range(num_episodes)
    ]


# def sample_rewards(
#     env: gym.Env, policy: Policy, *, num_steps: int
# ) -> np.ndarray:
#     # TODO this is means for evaluation, so we need to inject the behavior
#     # policy into the sample_episode methods
#     episode = sample_episode(env, policy, num_steps=num_steps)
#     return [interaction.reward for interaction in episode.interactions]
